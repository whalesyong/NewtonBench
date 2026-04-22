"""Generate a counterfactual preference dataset from completed NewtonBench trials.

Given a directory of saved `trial*.json` files, this script samples N trajectories,
branches each at a random `<run_experiment>` turn, re-samples the next action at a
higher temperature, rolls forward to a final law submission, and scores both the
original and counterfactual laws against the same seeded test set to produce
preference pairs.

Example:
    python scripts/generate_counterfactuals.py \
        --trial-dir evaluation_results/dsr1/exp_1/m0_gravity/vanilla_agent \
        --num-traj 200 \
        --out counterfactual_pairs.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import traceback
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make sure we can import project modules when invoked from project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import warnings

warnings.filterwarnings("ignore")

from utils.vanilla_agent import _run_from_messages  # noqa: E402


# ---------------------------------------------------------------------------
# Trajectory discovery
# ---------------------------------------------------------------------------
def discover_trials(trial_dir: Path) -> List[Path]:
    """Find all completed (non-failure) trial JSON files under `trial_dir`."""
    candidates = sorted(trial_dir.rglob("trial*.json"))
    out: List[Path] = []
    for p in candidates:
        if p.name.endswith("_fail.json"):
            continue
        out.append(p)
    return out


def load_trial(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not load {path}: {exc}")
        return None


def find_branch_points(chat_history: List[Dict[str, str]]) -> List[Tuple[int, int]]:
    """Return list of (loop_turn_k, message_index) for assistant turns containing
    <run_experiment>. Excludes the final assistant turn if it submitted <final_law>.
    """
    out: List[Tuple[int, int]] = []
    for idx, msg in enumerate(chat_history):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        if "<run_experiment>" not in content:
            continue
        # Loop-turn index: assistant at chat_history[2 + 2k] (0=system, 1=user).
        if idx < 2 or (idx - 2) % 2 != 0:
            # Non-standard history shape; skip.
            continue
        k = (idx - 2) // 2
        out.append((k, idx))
    return out


# ---------------------------------------------------------------------------
# Scoring & preference
# ---------------------------------------------------------------------------
def _stable_seed(trial_id: Any) -> int:
    """Derive a deterministic int seed from a trial id (which may be int or str)."""
    try:
        return int(trial_id) & 0xFFFFFFFF
    except (TypeError, ValueError):
        pass
    h = hashlib.md5(str(trial_id).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _preference(a_eval: Dict[str, Any], b_eval: Dict[str, Any]) -> str:
    """Return 'a', 'b', or 'tie' where a is the preferred one.

    Primary: higher exact_accuracy. Tiebreak: lower RMSLE (NaN is worst).
    """
    a_acc = float(a_eval.get("exact_accuracy") or 0.0)
    b_acc = float(b_eval.get("exact_accuracy") or 0.0)
    if a_acc != b_acc:
        return "a" if a_acc > b_acc else "b"

    def _rmsle(e: Dict[str, Any]) -> float:
        v = e.get("rmsle")
        try:
            v = float(v)
        except (TypeError, ValueError):
            return float("inf")
        if v != v:  # NaN
            return float("inf")
        return v

    a_r = _rmsle(a_eval)
    b_r = _rmsle(b_eval)
    if a_r == b_r:
        return "tie"
    return "a" if a_r < b_r else "b"


# ---------------------------------------------------------------------------
# Counterfactual rollout for a single trial
# ---------------------------------------------------------------------------
def generate_one(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Produce one counterfactual preference record.

    `task` is a dict (picklable) with the config needed for one rollout.
    Returns the preference record dict, or None on failure.
    """
    trial_path: str = task["trial_path"]
    cf_temperature: float = task["cf_temperature"]
    cli_max_turns: int = task["max_turns"]
    judge_model_name: str = task["judge_model_name"]
    rng_seed: int = task["rng_seed"]
    pair_index: int = task["pair_index"]

    rng = random.Random(rng_seed)

    trial = load_trial(Path(trial_path))
    if trial is None:
        return None

    chat_history = trial.get("chat_history") or []
    if not chat_history:
        return None

    # Prefer the trial's recorded max_turns (if the rollout saved it) over the CLI value.
    max_turns = int(trial.get("max_turns") or cli_max_turns)

    branch_points = find_branch_points(chat_history)
    if not branch_points:
        return None

    # Leave at least 1 remaining turn for the counterfactual to explore, and honor the
    # minimum-branch-turn policy (the branch point must be a truly intermediate state).
    min_branch_turn: int = task.get("min_branch_turn", 1)
    viable = [
        bp for bp in branch_points
        if min_branch_turn <= bp[0] <= max_turns - 1
    ]
    if not viable:
        return None

    branch_turn, branch_msg_idx = rng.choice(viable)

    module_name = trial.get("module_name")
    model_name = trial.get("model_name")
    difficulty = trial.get("equation_difficulty", "easy")
    system = trial.get("model_system", "vanilla_equation")
    law_version = trial.get("law_version")
    noise_level = trial.get("noise_level", 0.0)
    trial_id = trial.get("trial_id")

    if module_name is None or model_name is None:
        print(f"[WARN] {trial_path}: missing module_name or model_name; skipping.")
        return None

    try:
        module = importlib.import_module(f"modules.{module_name}")
    except Exception as exc:
        print(f"[WARN] {trial_path}: could not import module {module_name}: {exc}")
        return None

    # Truncate messages to the point just before the chosen assistant turn.
    branch_messages = deepcopy(chat_history[:branch_msg_idx])

    trial_info = {
        "trial_id": f"cf_{trial_id}_pair{pair_index}",
        "trial_dir": str(Path(trial_path).parent),
    }

    try:
        cf_result = _run_from_messages(
            module=module,
            model_name=model_name,
            messages=branch_messages,
            noise_level=noise_level,
            difficulty=difficulty,
            system=system,
            law_version=law_version,
            max_turns=max_turns,
            start_turn=branch_turn,
            trial_info=trial_info,
            temperature=cf_temperature,
        )
    except Exception as exc:
        print(f"[WARN] counterfactual rollout failed for {trial_path}: {exc}")
        traceback.print_exc()
        return None

    test_seed = _stable_seed(trial_id)

    original_law = trial.get("submitted_law") or ""
    cf_law = cf_result.get("submitted_law") or ""

    try:
        original_eval = module.evaluate_law(
            original_law,
            param_description=module.PARAM_DESCRIPTION,
            difficulty=difficulty,
            law_version=law_version,
            judge_model_name=judge_model_name,
            trial_info=trial_info,
            test_seed=test_seed,
        )
    except Exception as exc:
        print(f"[WARN] original eval failed for {trial_path}: {exc}")
        original_eval = {"rmsle": float("nan"), "exact_accuracy": 0.0, "error": str(exc)}

    try:
        cf_eval = module.evaluate_law(
            cf_law,
            param_description=module.PARAM_DESCRIPTION,
            difficulty=difficulty,
            law_version=law_version,
            judge_model_name=judge_model_name,
            trial_info=trial_info,
            test_seed=test_seed,
        )
    except Exception as exc:
        print(f"[WARN] cf eval failed for {trial_path}: {exc}")
        cf_eval = {"rmsle": float("nan"), "exact_accuracy": 0.0, "error": str(exc)}

    pref = _preference(original_eval, cf_eval)
    if pref == "a":
        preferred = "original"
    elif pref == "b":
        preferred = "counterfactual"
    else:
        preferred = "tie"

    return {
        "pair_index": pair_index,
        "trial_source": trial_path,
        "trial_id": trial_id,
        "module_name": module_name,
        "model_name": model_name,
        "equation_difficulty": difficulty,
        "model_system": system,
        "law_version": law_version,
        "noise_level": noise_level,
        "branch_turn": branch_turn,
        "branch_message_index": branch_msg_idx,
        "cf_temperature": cf_temperature,
        "max_turns": max_turns,
        "test_seed": test_seed,
        "original": {
            "submitted_law": original_law,
            "evaluation": original_eval,
            "chat_history": chat_history,
        },
        "counterfactual": {
            "submitted_law": cf_law,
            "evaluation": cf_eval,
            "chat_history": cf_result.get("chat_history"),
            "status": cf_result.get("status"),
            "rounds": cf_result.get("rounds"),
            "num_experiments": cf_result.get("num_experiments"),
            "total_tokens": cf_result.get("total_tokens"),
        },
        "preference": preferred,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True,
                        help="Directory (searched recursively) of trial*.json files.")
    parser.add_argument("--num-traj", type=int, required=True,
                        help="Total number of counterfactual rollouts to generate.")
    parser.add_argument("--out", type=Path, default=Path("counterfactual_pairs.jsonl"),
                        help="Output JSONL path.")
    parser.add_argument("--cf-temperature", type=float, default=1.0,
                        help="Temperature for counterfactual action sampling (default: 1.0).")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max LLM turns for the counterfactual rollout (default: 10).")
    parser.add_argument("--judge-model-name", type=str, default="nemotron-ultra",
                        help="LLM judge model for symbolic equivalence (default: nemotron-ultra).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Top-level RNG seed for trajectory sampling and branch-point choice.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default: 1).")
    parser.add_argument("--sample-with-replacement", action="store_true",
                        help="Sample trajectories with replacement (default: without, capped at num available).")
    parser.add_argument("--min-branch-turn", type=int, default=1,
                        help="Minimum loop-turn index eligible for branching (default: 1, so the first "
                             "assistant action is never re-sampled — counterfactuals require prior context).")
    args = parser.parse_args()

    if not args.trial_dir.exists():
        print(f"[ERROR] trial-dir {args.trial_dir} does not exist.", file=sys.stderr)
        return 2

    trials = discover_trials(args.trial_dir)
    if not trials:
        print(f"[ERROR] No trial*.json files found under {args.trial_dir}.", file=sys.stderr)
        return 2

    # Filter to trials that have at least one branchable turn, using each trial's own
    # recorded max_turns if present (falling back to the CLI value).
    branchable: List[Path] = []
    for p in trials:
        t = load_trial(p)
        if t is None:
            continue
        effective_max_turns = int(t.get("max_turns") or args.max_turns)
        bps = find_branch_points(t.get("chat_history") or [])
        if any(args.min_branch_turn <= k <= effective_max_turns - 1 for k, _ in bps):
            branchable.append(p)

    if not branchable:
        print(f"[ERROR] No trials have a valid branch point with "
              f"min_branch_turn={args.min_branch_turn}, max_turns={args.max_turns} "
              f"(or saved per-trial max_turns).",
              file=sys.stderr)
        return 2

    print(f"[INFO] Found {len(trials)} trials; {len(branchable)} branchable under max_turns={args.max_turns}.")

    rng = random.Random(args.seed)

    if args.sample_with_replacement:
        sampled = [rng.choice(branchable) for _ in range(args.num_traj)]
    else:
        n = min(args.num_traj, len(branchable))
        sampled = rng.sample(branchable, n)
        if n < args.num_traj:
            print(f"[WARN] Only {len(branchable)} unique branchable trials; producing {n} pairs "
                  f"(use --sample-with-replacement to reach {args.num_traj}).")

    # Build tasks with per-pair seeds so branch-point choice is reproducible.
    tasks: List[Dict[str, Any]] = []
    for i, p in enumerate(sampled):
        tasks.append({
            "trial_path": str(p),
            "cf_temperature": args.cf_temperature,
            "max_turns": args.max_turns,
            "min_branch_turn": args.min_branch_turn,
            "judge_model_name": args.judge_model_name,
            "rng_seed": rng.randint(0, 2**31 - 1),
            "pair_index": i,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out.open("w", encoding="utf-8") as fout:
        if args.workers > 1:
            with Pool(processes=args.workers) as pool:
                for rec in pool.imap_unordered(generate_one, tasks):
                    if rec is None:
                        continue
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    written += 1
                    print(f"[PAIR {rec['pair_index']}] preference={rec['preference']} "
                          f"(trial {rec['trial_id']}, branch_turn={rec['branch_turn']})")
        else:
            for task in tasks:
                rec = generate_one(task)
                if rec is None:
                    continue
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                written += 1
                print(f"[PAIR {rec['pair_index']}] preference={rec['preference']} "
                      f"(trial {rec['trial_id']}, branch_turn={rec['branch_turn']})")

    print(f"[DONE] Wrote {written} preference records to {args.out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
