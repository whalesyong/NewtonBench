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
import random
import sys
import traceback
from collections import Counter
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

from utils.vanilla_agent import _run_from_messages as _run_vanilla_from_messages  # noqa: E402
from utils.code_assisted_agent import (  # noqa: E402
    _run_from_messages as _run_code_assisted_from_messages,
    normalize_saved_chat_history_for_messages,
)


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


def _effective_action(content: str, agent_backend: str) -> str:
    """Infer which action the assistant actually took in a response."""
    positions = {
        "run_experiment": content.rfind("<run_experiment>"),
        "final_law": content.rfind("<final_law>"),
        "python": content.rfind("<python>"),
    }

    if agent_backend != "code_assisted_agent":
        positions["python"] = -1

    action, last_pos = max(positions.items(), key=lambda item: item[1])
    return action if last_pos >= 0 else "none"


def find_branch_points(chat_history: List[Dict[str, str]], agent_backend: str) -> List[Tuple[int, int]]:
    """Return (outer_turn_index, assistant_message_index) branchable run-experiment turns."""
    out: List[Tuple[int, int]] = []
    turn_index = 0
    for idx, msg in enumerate(chat_history):
        if msg.get("role") != "assistant":
            continue

        action = _effective_action(msg.get("content", "") or "", agent_backend)
        if agent_backend == "code_assisted_agent" and action == "python":
            continue

        if action == "run_experiment":
            out.append((turn_index, idx))

        turn_index += 1
    return out


def _format_backend_counts(counts: Counter) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{backend}={counts[backend]}" for backend in sorted(counts))


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


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
    verbose: bool = task.get("verbose", False)

    rng = random.Random(rng_seed)

    trial = load_trial(Path(trial_path))
    if trial is None:
        _log(verbose, f"[PAIR {pair_index}] Skipping unreadable trial: {trial_path}")
        return None

    chat_history = trial.get("chat_history") or []
    if not chat_history:
        _log(verbose, f"[PAIR {pair_index}] Skipping trial with empty chat history: {trial_path}")
        return None

    agent_backend = trial.get("agent_backend", "vanilla_agent")

    # Prefer the trial's recorded max_turns (if the rollout saved it) over the CLI value.
    max_turns = int(trial.get("max_turns") or cli_max_turns)

    branch_points = find_branch_points(chat_history, agent_backend)
    if not branch_points:
        _log(verbose, f"[PAIR {pair_index}] No branch points found for {trial_path}")
        return None

    # Leave at least 1 remaining turn for the counterfactual to explore, and honor the
    # minimum-branch-turn policy (the branch point must be a truly intermediate state).
    min_branch_turn: int = task.get("min_branch_turn", 1)
    viable = [
        bp for bp in branch_points
        if min_branch_turn <= bp[0] <= max_turns - 1
    ]
    if not viable:
        _log(verbose, f"[PAIR {pair_index}] No viable branch points after min/max turn filter for {trial_path}")
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

    _log(
        verbose,
        f"[PAIR {pair_index}] trial={trial_path} backend={agent_backend} module={module_name} "
        f"branch_turn={branch_turn} branch_msg_idx={branch_msg_idx} viable_branches={len(viable)}"
    )

    try:
        module = importlib.import_module(f"modules.{module_name}")
    except Exception as exc:
        print(f"[WARN] {trial_path}: could not import module {module_name}: {exc}")
        return None

    # Truncate history to the point just before the chosen assistant turn.
    branch_history = deepcopy(chat_history[:branch_msg_idx])

    trial_info = {
        "trial_id": f"cf_{trial_id}_pair{pair_index}",
        "trial_dir": str(Path(trial_path).parent),
    }

    try:
        if agent_backend == "code_assisted_agent":
            cf_result = _run_code_assisted_from_messages(
                module=module,
                model_name=model_name,
                messages=normalize_saved_chat_history_for_messages(deepcopy(branch_history)),
                chat_history=branch_history,
                noise_level=noise_level,
                difficulty=difficulty,
                system=system,
                law_version=law_version,
                max_turns=max_turns,
                start_turn=branch_turn,
                trial_info=trial_info,
                temperature=cf_temperature,
            )
        else:
            cf_result = _run_vanilla_from_messages(
                module=module,
                model_name=model_name,
                messages=branch_history,
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

    _log(
        verbose,
        f"[PAIR {pair_index}] original_acc={original_eval.get('exact_accuracy')} "
        f"cf_acc={cf_eval.get('exact_accuracy')} original_rmsle={original_eval.get('rmsle')} "
        f"cf_rmsle={cf_eval.get('rmsle')} preference={preferred}"
    )

    return {
        "pair_index": pair_index,
        "trial_source": trial_path,
        "trial_id": trial_id,
        "module_name": module_name,
        "model_name": model_name,
        "agent_backend": agent_backend,
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
                        help="Directory searched recursively for completed trial*.json files; it may include mixed modules or backends.")
    parser.add_argument("--num-traj", type=int, required=True,
                        help="Total number of counterfactual rollouts to generate.")
    parser.add_argument("--out", type=Path, default=Path("counterfactual_pairs.jsonl"),
                        help="Output JSONL path.")
    parser.add_argument("--cf-temperature", type=float, default=1.0,
                        help="Temperature for counterfactual action sampling (default: 1.0).")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max LLM turns for the counterfactual rollout (default: 10).")
    parser.add_argument("--judge-model-name", type=str, default="vllm-judge-local",
                        help="LLM judge model for symbolic equivalence (default: vllm-judge-local).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Top-level RNG seed for trajectory sampling and branch-point choice.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default: 1).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-pair sampling, branching, and evaluation details.")
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
    discovered_by_backend: Counter = Counter()
    branchable_by_backend: Counter = Counter()
    for p in trials:
        t = load_trial(p)
        if t is None:
            continue
        agent_backend = t.get("agent_backend", "vanilla_agent")
        discovered_by_backend[agent_backend] += 1
        effective_max_turns = int(t.get("max_turns") or args.max_turns)
        bps = find_branch_points(t.get("chat_history") or [], agent_backend)
        if any(args.min_branch_turn <= k <= effective_max_turns - 1 for k, _ in bps):
            branchable.append(p)
            branchable_by_backend[agent_backend] += 1

    if not branchable:
        print(f"[ERROR] No trials have a valid branch point with "
              f"min_branch_turn={args.min_branch_turn}, max_turns={args.max_turns} "
              f"(or saved per-trial max_turns).",
              file=sys.stderr)
        return 2

    print(
        f"[INFO] Found {len(trials)} completed trials under {args.trial_dir}; "
        f"{len(branchable)} are branchable under max_turns={args.max_turns}."
    )
    print(f"[INFO] Completed trials by backend: {_format_backend_counts(discovered_by_backend)}")
    print(f"[INFO] Branchable trials by backend: {_format_backend_counts(branchable_by_backend)}")

    rng = random.Random(args.seed)

    if args.sample_with_replacement:
        sampled = [rng.choice(branchable) for _ in range(args.num_traj)]
    else:
        n = min(args.num_traj, len(branchable))
        sampled = rng.sample(branchable, n)
        if n < args.num_traj:
            print(f"[WARN] Only {len(branchable)} unique branchable trials; producing {n} pairs "
                  f"(use --sample-with-replacement to reach {args.num_traj}).")

    if args.verbose:
        sampled_by_backend: Counter = Counter()
        for p in sampled:
            t = load_trial(p)
            if t is None:
                continue
            sampled_by_backend[t.get("agent_backend", "vanilla_agent")] += 1
        print(f"[INFO] Sampled {len(sampled)} trials for rollout.")
        print(f"[INFO] Sampled trials by backend: {_format_backend_counts(sampled_by_backend)}")
        for i, p in enumerate(sampled):
            print(f"[SAMPLED {i}] {p}")

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
            "verbose": args.verbose,
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
