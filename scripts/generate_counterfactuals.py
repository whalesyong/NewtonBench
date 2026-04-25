"""Generate a counterfactual preference dataset from completed NewtonBench trials.

Given a directory of saved `trial*.json` files, this script samples branch roots,
re-samples the next assistant action at that decision point, realizes the single
post-action state induced by that action, and then fans out multiple independent
suffix rollouts from that shared state. Each suffix rollout is evaluated against
the same seeded test set so downstream aggregation can reduce variance for the
chosen counterfactual first action.

Example:
    python scripts/generate_counterfactuals.py \
        --trial-dir evaluation_results/dsr1/exp_1/m0_gravity/vanilla_agent \
        --num-traj 200 \
        --num-suffix-rollouts 3 \
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
from typing import Any, Dict, List, Optional, Set, Tuple

# Make sure we can import project modules when invoked from project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import warnings

warnings.filterwarnings("ignore")

from utils.code_assisted_agent import (  # noqa: E402
    TURN_REMINDER,
    _call_llm_and_format_response,
    build_final_submission_prompt,
    extract_final_law,
    format_experiment_results,
    normalize_saved_chat_history_for_messages,
    run_experiment_from_response,
)
from utils.code_executor import CodeExecutor  # noqa: E402
from utils.vanilla_agent import (  # noqa: E402
    FINAL_LAW_PROMPT,
    INVALID_RESPONSE_PROMPT,
    _call_llm_and_process_response,
    _extract_final_law,
    parse_experiment_request,
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


def _format_backend_counts(counts: Counter) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{backend}={counts[backend]}" for backend in sorted(counts))


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _make_vanilla_state(
    messages: List[Dict[str, str]],
    current_turn: int,
    phase: str,
    total_tokens: int = 0,
    num_experiments_run: int = 0,
) -> Dict[str, Any]:
    return {
        "messages": deepcopy(messages),
        "current_turn": current_turn,
        "phase": phase,
        "total_tokens": total_tokens,
        "num_experiments_run": num_experiments_run,
    }


def _make_code_assisted_state(
    messages: List[Dict[str, str]],
    chat_history: List[Dict[str, str]],
    current_turn: int,
    phase: str,
    python_calls_this_turn: int = 0,
    total_tokens: int = 0,
    num_experiments: int = 0,
    python_tags_used_total: int = 0,
) -> Dict[str, Any]:
    return {
        "messages": deepcopy(messages),
        "chat_history": deepcopy(chat_history),
        "current_turn": current_turn,
        "phase": phase,
        "python_calls_this_turn": python_calls_this_turn,
        "total_tokens": total_tokens,
        "num_experiments": num_experiments,
        "python_tags_used_total": python_tags_used_total,
    }


def _append_or_extend_user_message(messages: List[Dict[str, str]], content: str) -> None:
    if messages and messages[-1].get("role") == "user":
        messages[-1]["content"] += "\n\n" + content
    else:
        messages.append({"role": "user", "content": content})


def _append_code_assisted_user_message(
    messages: List[Dict[str, str]],
    chat_history: List[Dict[str, str]],
    message_content: str,
    chat_history_content: Optional[str] = None,
) -> None:
    chat_history_content = message_content if chat_history_content is None else chat_history_content

    if messages and messages[-1].get("role") == "user":
        messages[-1]["content"] += "\n\n" + message_content
    else:
        messages.append({"role": "user", "content": message_content})

    if chat_history and chat_history[-1].get("role") == "user":
        chat_history[-1]["content"] += "\n\n" + chat_history_content
    else:
        chat_history.append({"role": "user", "content": chat_history_content})


def _prepare_code_assisted_same_turn_state(
    messages: List[Dict[str, str]],
    chat_history: List[Dict[str, str]],
    current_turn: int,
    python_calls_this_turn: int,
    total_tokens: int,
    num_experiments: int,
    python_tags_used_total: int,
) -> Dict[str, Any]:
    _append_code_assisted_user_message(messages, chat_history, TURN_REMINDER)
    return _make_code_assisted_state(
        messages=messages,
        chat_history=chat_history,
        current_turn=current_turn,
        phase="same_turn",
        python_calls_this_turn=python_calls_this_turn,
        total_tokens=total_tokens,
        num_experiments=num_experiments,
        python_tags_used_total=python_tags_used_total,
    )


def _prepare_code_assisted_next_turn_state(
    messages: List[Dict[str, str]],
    chat_history: List[Dict[str, str]],
    next_turn: int,
    module: Any,
    max_turns: int,
    total_tokens: int,
    num_experiments: int,
    python_tags_used_total: int,
) -> Dict[str, Any]:
    if next_turn >= max_turns:
        _append_code_assisted_user_message(
            messages,
            chat_history,
            build_final_submission_prompt(module),
        )
        next_phase = "forced_final"
    else:
        _append_code_assisted_user_message(messages, chat_history, TURN_REMINDER)
        next_phase = "turn_start"

    return _make_code_assisted_state(
        messages=messages,
        chat_history=chat_history,
        current_turn=next_turn,
        phase=next_phase,
        python_calls_this_turn=0,
        total_tokens=total_tokens,
        num_experiments=num_experiments,
        python_tags_used_total=python_tags_used_total,
    )


def _extract_vanilla_decision_points(
    chat_history: List[Dict[str, str]],
    max_turns: int,
    include_state: bool,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    current_turn = 0
    decision_index = 0

    for idx, msg in enumerate(chat_history):
        if msg.get("role") != "assistant":
            continue

        phase = "forced_final" if current_turn >= max_turns else "regular"
        action_type = _effective_action(msg.get("content", "") or "", "vanilla_agent")
        point: Dict[str, Any] = {
            "decision_index": decision_index,
            "assistant_message_index": idx,
            "outer_turn": current_turn,
            "phase": phase,
            "python_calls_this_turn": 0,
            "original_action_type": action_type,
        }
        if include_state:
            point["state"] = _make_vanilla_state(
                messages=chat_history[:idx],
                current_turn=current_turn,
                phase=phase,
            )
        points.append(point)
        decision_index += 1

        if phase == "forced_final" or action_type == "final_law":
            break
        current_turn += 1

    return points


def _extract_code_assisted_decision_points(
    chat_history: List[Dict[str, str]],
    max_turns: int,
    include_state: bool,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    current_turn = 0
    phase = "turn_start"
    python_calls_this_turn = 0
    decision_index = 0

    for idx, msg in enumerate(chat_history):
        if msg.get("role") != "assistant":
            continue

        effective_phase = "forced_final" if current_turn >= max_turns else phase
        action_type = _effective_action(msg.get("content", "") or "", "code_assisted_agent")
        point: Dict[str, Any] = {
            "decision_index": decision_index,
            "assistant_message_index": idx,
            "outer_turn": current_turn,
            "phase": effective_phase,
            "python_calls_this_turn": python_calls_this_turn,
            "original_action_type": action_type,
        }
        if include_state:
            prefix_chat_history = chat_history[:idx]
            point["state"] = _make_code_assisted_state(
                messages=normalize_saved_chat_history_for_messages(deepcopy(prefix_chat_history)),
                chat_history=prefix_chat_history,
                current_turn=current_turn,
                phase=effective_phase,
                python_calls_this_turn=python_calls_this_turn,
            )
        points.append(point)
        decision_index += 1

        if effective_phase == "forced_final" or action_type == "final_law":
            break

        if action_type == "python":
            next_user_content = ""
            if idx + 1 < len(chat_history) and chat_history[idx + 1].get("role") == "user":
                next_user_content = chat_history[idx + 1].get("content", "") or ""

            if next_user_content.startswith("[Code Execution Feedback]") and "Validation Failed" in next_user_content:
                phase = "same_turn"
                python_calls_this_turn = 0
            else:
                current_turn += 1
                phase = "turn_start"
                python_calls_this_turn = 0
        else:
            current_turn += 1
            phase = "turn_start"
            python_calls_this_turn = 0

    return points


def extract_decision_points(
    chat_history: List[Dict[str, str]],
    agent_backend: str,
    max_turns: int,
    include_state: bool = False,
) -> List[Dict[str, Any]]:
    if agent_backend == "code_assisted_agent":
        return _extract_code_assisted_decision_points(chat_history, max_turns, include_state)
    return _extract_vanilla_decision_points(chat_history, max_turns, include_state)


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


def _rmsle(e: Dict[str, Any]) -> float:
    """Extract RMSLE from eval dict, treating NaN/invalid as inf."""
    v = e.get("rmsle")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("inf")
    if v != v:  # NaN
        return float("inf")
    return v


def _preference(a_eval: Dict[str, Any], b_eval: Dict[str, Any]) -> str:
    """Return 'a', 'b', or 'tie' where a is the preferred one.

    Primary: lower RMSLE (NaN is worst).  Tiebreak: higher exact_accuracy.
    """
    a_r = _rmsle(a_eval)
    b_r = _rmsle(b_eval)

    if a_r == b_r:
        # Identical RMSLE — fall back to symbolic-equivalence tiebreak
        a_acc = float(a_eval.get("exact_accuracy") or 0.0)
        b_acc = float(b_eval.get("exact_accuracy") or 0.0)
        if a_acc != b_acc:
            return "a" if a_acc > b_acc else "b"
        return "tie"

    return "a" if a_r < b_r else "b"


def _evaluate_law_safe(
    module: Any,
    submitted_law: str,
    difficulty: str,
    law_version: Optional[str],
    judge_model_name: str,
    trial_info: Dict[str, Any],
    test_seed: int,
    label: str,
    verbose: bool,
) -> Dict[str, Any]:
    try:
        return module.evaluate_law(
            submitted_law,
            param_description=module.PARAM_DESCRIPTION,
            difficulty=difficulty,
            law_version=law_version,
            judge_model_name=judge_model_name,
            trial_info=trial_info,
            test_seed=test_seed,
        )
    except Exception as exc:
        _log(verbose, f"[WARN] {label} eval failed for {trial_info.get('trial_id')}: {exc}")
        return {"rmsle": float("nan"), "exact_accuracy": 0.0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Counterfactual environment stepping
# ---------------------------------------------------------------------------
def _step_vanilla_from_state(
    module: Any,
    model_name: str,
    state: Dict[str, Any],
    noise_level: float,
    difficulty: str,
    system: str,
    law_version: Optional[str],
    max_turns: int,
    trial_info: Dict[str, Any],
    temperature: float,
) -> Dict[str, Any]:
    messages = deepcopy(state["messages"])
    current_turn = int(state["current_turn"])
    phase = state["phase"]
    total_tokens = int(state.get("total_tokens") or 0)
    num_experiments_run = int(state.get("num_experiments_run") or 0)

    messages, tokens, response_text = _call_llm_and_process_response(
        messages,
        model_name,
        trial_info,
        temperature=temperature,
    )
    total_tokens += tokens
    assistant_content = messages[-1]["content"]
    action_type = _effective_action(response_text or "", "vanilla_agent")

    if phase == "forced_final":
        _, submitted_law = _extract_final_law(response_text or "", module.FUNCTION_SIGNATURE)
        return {
            "terminal": True,
            "action_type": action_type,
            "response_text": response_text,
            "assistant_content": assistant_content,
            "final_result": {
                "status": "max_turns_reached",
                "submitted_law": submitted_law,
                "rounds": max_turns,
                "max_turns": max_turns,
                "total_tokens": total_tokens,
                "num_experiments": num_experiments_run,
                "chat_history": messages,
            },
        }

    is_submitted, submitted_law = _extract_final_law(response_text or "", module.FUNCTION_SIGNATURE)
    if is_submitted:
        return {
            "terminal": True,
            "action_type": action_type,
            "response_text": response_text,
            "assistant_content": assistant_content,
            "final_result": {
                "status": "completed",
                "submitted_law": submitted_law,
                "rounds": current_turn + 1,
                "max_turns": max_turns,
                "total_tokens": total_tokens,
                "num_experiments": num_experiments_run,
                "chat_history": messages,
            },
        }

    experiments_to_run = parse_experiment_request(response_text if response_text is not None else "")
    if experiments_to_run:
        num_experiments_run += len(experiments_to_run)
        results = []
        for exp in experiments_to_run:
            result = module.run_experiment_for_module(
                **exp,
                noise_level=noise_level,
                difficulty=difficulty,
                system=system,
                law_version=law_version,
            )
            if system == "vanilla_equation":
                result = "{:.15e}".format(result)
            results.append(result)

        output_str = f"<experiment_output>\n{json.dumps(results)}\n</experiment_output>"
        messages.append({"role": "user", "content": output_str})
    else:
        messages.append({"role": "user", "content": INVALID_RESPONSE_PROMPT})

    next_turn = current_turn + 1
    next_phase = "regular"
    if next_turn >= max_turns:
        _append_or_extend_user_message(messages, FINAL_LAW_PROMPT)
        next_phase = "forced_final"

    return {
        "terminal": False,
        "action_type": action_type,
        "response_text": response_text,
        "assistant_content": assistant_content,
        "next_state": _make_vanilla_state(
            messages=messages,
            current_turn=next_turn,
            phase=next_phase,
            total_tokens=total_tokens,
            num_experiments_run=num_experiments_run,
        ),
    }


def _step_code_assisted_from_state(
    module: Any,
    model_name: str,
    state: Dict[str, Any],
    noise_level: float,
    difficulty: str,
    system: str,
    law_version: Optional[str],
    max_turns: int,
    trial_info: Dict[str, Any],
    temperature: float,
) -> Dict[str, Any]:
    messages = deepcopy(state["messages"])
    chat_history = deepcopy(state["chat_history"])
    current_turn = int(state["current_turn"])
    phase = state["phase"]
    python_calls_this_turn = int(state.get("python_calls_this_turn") or 0)
    total_tokens = int(state.get("total_tokens") or 0)
    num_experiments = int(state.get("num_experiments") or 0)
    python_tags_used_total = int(state.get("python_tags_used_total") or 0)

    response, combined_content, tokens = _call_llm_and_format_response(
        messages,
        model_name,
        trial_info,
        temperature,
    )
    total_tokens += tokens
    chat_history.append({"role": "assistant", "content": combined_content})
    action_type = _effective_action(response or "", "code_assisted_agent")

    if phase == "forced_final":
        return {
            "terminal": True,
            "action_type": action_type,
            "response_text": response,
            "assistant_content": combined_content,
            "final_result": {
                "status": "max_turns_reached",
                "submitted_law": extract_final_law(chat_history, module),
                "chat_history": chat_history,
                "rounds": max_turns,
                "max_turns": max_turns,
                "total_tokens": total_tokens,
                "python_tags_used_total": python_tags_used_total,
                "num_experiments": num_experiments,
                "exploration_mode": "code_assisted_agent",
            },
        }

    if response and "<final_law>" in response and "</final_law>" in response:
        return {
            "terminal": True,
            "action_type": action_type,
            "response_text": response,
            "assistant_content": combined_content,
            "final_result": {
                "status": "completed",
                "submitted_law": extract_final_law(chat_history, module),
                "chat_history": chat_history,
                "rounds": current_turn + 1,
                "max_turns": max_turns,
                "total_tokens": total_tokens,
                "python_tags_used_total": python_tags_used_total,
                "num_experiments": num_experiments,
                "exploration_mode": "code_assisted_agent",
            },
        }

    code_executor = CodeExecutor(
        module_name=module.__name__.split(".")[-1],
        difficulty=difficulty,
        system=system,
    )
    code_executor.turn_number = current_turn + 1
    code_executor.python_calls_this_turn = python_calls_this_turn

    python_pos = response.rfind("<python>") if response else -1
    experiment_pos = response.rfind("<run_experiment>") if response else -1
    code_result = code_executor.process_llm_response(response or "")

    if code_result["has_python_tag"] and python_pos > experiment_pos:
        feedback = code_executor.format_execution_feedback(code_result)
        messages.append({"role": "assistant", "content": combined_content})

        if code_result.get("limit_reached", False):
            _append_code_assisted_user_message(
                messages,
                chat_history,
                feedback,
                f"[Code Execution Feedback - Turn Limit]\n{feedback}",
            )
            next_state = _prepare_code_assisted_next_turn_state(
                messages=messages,
                chat_history=chat_history,
                next_turn=current_turn + 1,
                module=module,
                max_turns=max_turns,
                total_tokens=total_tokens,
                num_experiments=num_experiments,
                python_tags_used_total=python_tags_used_total,
            )
            return {
                "terminal": False,
                "action_type": action_type,
                "response_text": response,
                "assistant_content": combined_content,
                "next_state": next_state,
            }

        python_tags_used_total += 1
        _append_code_assisted_user_message(
            messages,
            chat_history,
            feedback,
            f"[Code Execution Feedback]\n{feedback}",
        )

        if not code_result.get("validation_success", True):
            next_state = _prepare_code_assisted_same_turn_state(
                messages=messages,
                chat_history=chat_history,
                current_turn=current_turn,
                python_calls_this_turn=code_executor.python_calls_this_turn,
                total_tokens=total_tokens,
                num_experiments=num_experiments,
                python_tags_used_total=python_tags_used_total,
            )
        else:
            next_state = _prepare_code_assisted_next_turn_state(
                messages=messages,
                chat_history=chat_history,
                next_turn=current_turn + 1,
                module=module,
                max_turns=max_turns,
                total_tokens=total_tokens,
                num_experiments=num_experiments,
                python_tags_used_total=python_tags_used_total,
            )

        return {
            "terminal": False,
            "action_type": action_type,
            "response_text": response,
            "assistant_content": combined_content,
            "next_state": next_state,
        }

    messages.append({"role": "assistant", "content": combined_content})

    if response and "<run_experiment>" in response and "</run_experiment>" in response:
        experiment_result = run_experiment_from_response(
            module,
            response,
            system,
            noise_level,
            difficulty,
            law_version,
        )
        if experiment_result:
            if isinstance(experiment_result, list):
                num_experiments += len(experiment_result)
            else:
                num_experiments += 1

            experiment_message = format_experiment_results(experiment_result)
            _append_code_assisted_user_message(
                messages,
                chat_history,
                experiment_message,
                f"[Experiment Results]\n{experiment_message}",
            )
    else:
        reminder_message = (
            "**Action Reminder:** Please use exactly 1 action per turn with correct format: "
            "<run_experiment> tag with the correct JSON format, <python> tag to execute Python code, "
            f"or <final_law> tag to submit the law. {code_executor.get_turn_usage_info()}"
        )
        _append_code_assisted_user_message(messages, chat_history, reminder_message)

    next_state = _prepare_code_assisted_next_turn_state(
        messages=messages,
        chat_history=chat_history,
        next_turn=current_turn + 1,
        module=module,
        max_turns=max_turns,
        total_tokens=total_tokens,
        num_experiments=num_experiments,
        python_tags_used_total=python_tags_used_total,
    )
    return {
        "terminal": False,
        "action_type": action_type,
        "response_text": response,
        "assistant_content": combined_content,
        "next_state": next_state,
    }


def _step_from_state(
    agent_backend: str,
    module: Any,
    model_name: str,
    state: Dict[str, Any],
    noise_level: float,
    difficulty: str,
    system: str,
    law_version: Optional[str],
    max_turns: int,
    trial_info: Dict[str, Any],
    temperature: float,
) -> Dict[str, Any]:
    if agent_backend == "code_assisted_agent":
        return _step_code_assisted_from_state(
            module=module,
            model_name=model_name,
            state=state,
            noise_level=noise_level,
            difficulty=difficulty,
            system=system,
            law_version=law_version,
            max_turns=max_turns,
            trial_info=trial_info,
            temperature=temperature,
        )

    return _step_vanilla_from_state(
        module=module,
        model_name=model_name,
        state=state,
        noise_level=noise_level,
        difficulty=difficulty,
        system=system,
        law_version=law_version,
        max_turns=max_turns,
        trial_info=trial_info,
        temperature=temperature,
    )


def _rollout_to_completion(
    agent_backend: str,
    module: Any,
    model_name: str,
    state: Dict[str, Any],
    noise_level: float,
    difficulty: str,
    system: str,
    law_version: Optional[str],
    max_turns: int,
    trial_info: Dict[str, Any],
    temperature: float,
) -> Dict[str, Any]:
    current_state = deepcopy(state)
    max_steps = max_turns * 4 + 10

    for _ in range(max_steps):
        step = _step_from_state(
            agent_backend=agent_backend,
            module=module,
            model_name=model_name,
            state=current_state,
            noise_level=noise_level,
            difficulty=difficulty,
            system=system,
            law_version=law_version,
            max_turns=max_turns,
            trial_info=trial_info,
            temperature=temperature,
        )
        if step["terminal"]:
            return step["final_result"]
        current_state = step["next_state"]

    raise RuntimeError(f"Continuation exceeded {max_steps} assistant steps for {trial_info.get('trial_id')}")


def _build_record(
    common_record: Dict[str, Any],
    rollout_index: int,
    fanout_size: int,
    num_suffix_rollouts_requested: int,
    cf_result: Dict[str, Any],
    cf_eval: Dict[str, Any],
    original_eval: Dict[str, Any],
    cf_group_id: str,
) -> Dict[str, Any]:
    pref = _preference(original_eval, cf_eval)
    if pref == "a":
        preferred = "original"
    elif pref == "b":
        preferred = "counterfactual"
    else:
        preferred = "tie"

    record = dict(common_record)
    record.update(
        {
            "cf_group_id": cf_group_id,
            "rollout_index": rollout_index,
            "fanout_size": fanout_size,
            "num_suffix_rollouts_requested": num_suffix_rollouts_requested,
            "counterfactual": {
                "submitted_law": cf_result.get("submitted_law") or "",
                "evaluation": cf_eval,
                "chat_history": cf_result.get("chat_history"),
                "status": cf_result.get("status"),
                "rounds": cf_result.get("rounds"),
                "num_experiments": cf_result.get("num_experiments"),
                "total_tokens": cf_result.get("total_tokens"),
                "python_tags_used_total": cf_result.get("python_tags_used_total"),
            },
            "preference": preferred,
        }
    )
    return record


def materialize_one_root(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    trial_path = Path(task["trial_path"])
    decision_index = int(task["decision_index"])
    pair_index = int(task["pair_index"])
    cf_temperature = float(task["cf_temperature"])
    cli_max_turns = int(task["max_turns"])
    judge_model_name = task["judge_model_name"]
    num_suffix_rollouts = int(task["num_suffix_rollouts"])
    verbose = bool(task.get("verbose", False))

    # purpose: temp sample a different action, a_t, get s_{t+1} <- P(s_t, a_t), then 
    # fan out independent rollouts from s_{t+1}

    trial = load_trial(trial_path)
    if trial is None:
        _log(verbose, f"[PAIR {pair_index}] Skipping unreadable trial: {trial_path}")
        return None

    chat_history = trial.get("chat_history") or []
    if not chat_history:
        _log(verbose, f"[PAIR {pair_index}] Skipping trial with empty chat history: {trial_path}")
        return None

    agent_backend = trial.get("agent_backend", "vanilla_agent")
    max_turns = int(trial.get("max_turns") or cli_max_turns)
    decision_points = extract_decision_points(chat_history, agent_backend, max_turns, include_state=True)
    point = next((p for p in decision_points if p["decision_index"] == decision_index), None)
    if point is None:
        _log(verbose, f"[PAIR {pair_index}] Decision index {decision_index} no longer exists for {trial_path}")
        return None

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

    _log(
        verbose,
        f"[PAIR {pair_index}] trial={trial_path} backend={agent_backend} module={module_name} "
        f"decision_index={decision_index} outer_turn={point['outer_turn']} phase={point['phase']} "
        f"assistant_msg_idx={point['assistant_message_index']} original_action={point['original_action_type']}"
    )

    test_seed = _stable_seed(trial_id)
    trial_info_base = {
        "trial_id": f"cf_{trial_id}_pair{pair_index}",
        "trial_dir": str(trial_path.parent),
    }

    original_law = trial.get("submitted_law") or ""
    original_eval = _evaluate_law_safe(
        module=module,
        submitted_law=original_law,
        difficulty=difficulty,
        law_version=law_version,
        judge_model_name=judge_model_name,
        trial_info=trial_info_base,
        test_seed=test_seed,
        label="original",
        verbose=verbose,
    )

    try:
        # take the counterfactual action
        first_step = _step_from_state(
            agent_backend=agent_backend,
            module=module,
            model_name=model_name,
            state=point["state"],
            noise_level=noise_level,
            difficulty=difficulty,
            system=system,
            law_version=law_version,
            max_turns=max_turns,
            trial_info={
                "trial_id": f"{trial_info_base['trial_id']}_first",
                "trial_dir": trial_info_base["trial_dir"],
            },
            temperature=cf_temperature,
        )
    except Exception as exc:
        print(f"[WARN] counterfactual first step failed for {trial_path}: {exc}")
        traceback.print_exc()
        return None
    
    cf_group_id = f"{trial_info_base['trial_id']}_d{decision_index}"
    common_record = {
        "pair_index": pair_index,
        "trial_source": str(trial_path),
        "trial_id": trial_id,
        "module_name": module_name,
        "model_name": model_name,
        "agent_backend": agent_backend,
        "equation_difficulty": difficulty,
        "model_system": system,
        "law_version": law_version,
        "noise_level": noise_level,
        "branch_turn": point["outer_turn"],
        "branch_message_index": point["assistant_message_index"],
        "decision_index": point["decision_index"],
        "decision_phase": point["phase"],
        "branch_python_calls_this_turn": point.get("python_calls_this_turn", 0),
        "original_action_type": point["original_action_type"],
        "counterfactual_first_action_type": first_step["action_type"],
        "counterfactual_first_response": first_step.get("response_text") or "",
        "counterfactual_first_assistant_message": first_step.get("assistant_content") or "",
        "cf_temperature": cf_temperature,
        "max_turns": max_turns,
        "test_seed": test_seed,
        "original": {
            "submitted_law": original_law,
            "evaluation": original_eval,
            "chat_history": chat_history,
        },
    }

    if first_step["terminal"]:
        cf_result = first_step["final_result"]
        cf_law = cf_result.get("submitted_law") or ""
        cf_eval = _evaluate_law_safe(
            module=module,
            submitted_law=cf_law,
            difficulty=difficulty,
            law_version=law_version,
            judge_model_name=judge_model_name,
            trial_info={
                "trial_id": f"{trial_info_base['trial_id']}_rollout0",
                "trial_dir": trial_info_base["trial_dir"],
            },
            test_seed=test_seed,
            label="counterfactual",
            verbose=verbose,
        )
        rec = _build_record(
            common_record=common_record,
            rollout_index=0,
            fanout_size=1,
            num_suffix_rollouts_requested=num_suffix_rollouts,
            cf_result=cf_result,
            cf_eval=cf_eval,
            original_eval=original_eval,
            cf_group_id=cf_group_id,
        )
        return {"records": [rec], "continuation_tasks": []}

    continuation_tasks: List[Dict[str, Any]] = []
    for rollout_index in range(num_suffix_rollouts):
        continuation_tasks.append(
            {
                "agent_backend": agent_backend,
                "module_name": module_name,
                "model_name": model_name,
                "difficulty": difficulty,
                "system": system,
                "law_version": law_version,
                "noise_level": noise_level,
                "max_turns": max_turns,
                "cf_temperature": cf_temperature,
                "judge_model_name": judge_model_name,
                "test_seed": test_seed,
                "next_state": first_step["next_state"],
                "common_record": common_record,
                "original_eval": original_eval,
                "cf_group_id": cf_group_id,
                "rollout_index": rollout_index,
                "num_suffix_rollouts_requested": num_suffix_rollouts,
                "trial_dir": trial_info_base["trial_dir"],
                "trial_info_id": f"{trial_info_base['trial_id']}_rollout{rollout_index}",
                "verbose": verbose,
            }
        )

    return {"records": [], "continuation_tasks": continuation_tasks}


def complete_one_continuation(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    agent_backend = task["agent_backend"]
    module_name = task["module_name"]
    model_name = task["model_name"]
    difficulty = task["difficulty"]
    system = task["system"]
    law_version = task["law_version"]
    noise_level = task["noise_level"]
    max_turns = int(task["max_turns"])
    cf_temperature = float(task["cf_temperature"])
    judge_model_name = task["judge_model_name"]
    test_seed = int(task["test_seed"])
    rollout_index = int(task["rollout_index"])
    common_record = deepcopy(task["common_record"])
    original_eval = task["original_eval"]
    cf_group_id = task["cf_group_id"]
    num_suffix_rollouts_requested = int(task["num_suffix_rollouts_requested"])
    verbose = bool(task.get("verbose", False))

    try:
        module = importlib.import_module(f"modules.{module_name}")
    except Exception as exc:
        print(f"[WARN] could not import module {module_name} for continuation: {exc}")
        return None

    try:
        cf_result = _rollout_to_completion(
            agent_backend=agent_backend,
            module=module,
            model_name=model_name,
            state=task["next_state"],
            noise_level=noise_level,
            difficulty=difficulty,
            system=system,
            law_version=law_version,
            max_turns=max_turns,
            trial_info={
                "trial_id": task["trial_info_id"],
                "trial_dir": task["trial_dir"],
            },
            temperature=cf_temperature,
        )
    except Exception as exc:
        print(f"[WARN] counterfactual continuation failed for {task['trial_info_id']}: {exc}")
        traceback.print_exc()
        return None

    cf_law = cf_result.get("submitted_law") or ""
    cf_eval = _evaluate_law_safe(
        module=module,
        submitted_law=cf_law,
        difficulty=difficulty,
        law_version=law_version,
        judge_model_name=judge_model_name,
        trial_info={
            "trial_id": task["trial_info_id"],
            "trial_dir": task["trial_dir"],
        },
        test_seed=test_seed,
        label="counterfactual",
        verbose=verbose,
    )

    return _build_record(
        common_record=common_record,
        rollout_index=rollout_index,
        fanout_size=0,
        num_suffix_rollouts_requested=num_suffix_rollouts_requested,
        cf_result=cf_result,
        cf_eval=cf_eval,
        original_eval=original_eval,
        cf_group_id=cf_group_id,
    )


def _load_resume_records(stream_path: Path, num_suffix_rollouts: int
                         ) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """Read a prior run's partial JSONL and return ``(kept_records, complete_pair_indices)``.

    A pair is "complete" if it has a terminal record (``fanout_size == 1``) or
    its full set of ``num_suffix_rollouts`` continuation records. Records for
    partially-completed pairs are dropped, since we cannot deterministically
    resume mid-rollout (counterfactual action sampling is non-deterministic).
    Those pairs will be re-run from scratch."""
    if not stream_path.exists():
        return [], set()

    by_pair: Dict[int, List[Dict[str, Any]]] = {}
    with stream_path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a truncated final line from a hard kill.
                continue
            by_pair.setdefault(rec["pair_index"], []).append(rec)

    kept: List[Dict[str, Any]] = []
    complete: Set[int] = set()
    for pid, recs in by_pair.items():
        is_terminal = any(r.get("fanout_size") == 1 for r in recs)
        if is_terminal or len(recs) >= num_suffix_rollouts:
            complete.add(pid)
            kept.extend(recs)
    return kept, complete


def _run_task_map(func, tasks: List[Dict[str, Any]], workers: int, desc: str = "tasks",
                  on_result=None) -> List[Any]:
    """Run ``func`` over ``tasks`` in parallel, optionally invoking ``on_result``
    in the main process for each completed result. ``on_result`` lets callers
    stream incremental output (e.g. write results to disk as they arrive) without
    waiting for the full pool to drain."""
    if not tasks:
        return []
    total = len(tasks)
    print(f"[{desc}] starting {total} tasks ...")
    results: List[Any] = []
    if workers > 1:
        with Pool(processes=workers) as pool:
            for result in pool.imap_unordered(func, tasks):
                results.append(result)
                if on_result is not None:
                    on_result(result)
    else:
        for task in tasks:
            result = func(task)
            results.append(result)
            if on_result is not None:
                on_result(result)
    print(f"[{desc}] done: {len(results)}/{total}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True,
                        help="Directory searched recursively for completed trial*.json files; it may include mixed modules or backends.")
    parser.add_argument("--num-traj", type=int, required=True,
                        help="Total number of branch roots to sample.")
    parser.add_argument("--num-suffix-rollouts", type=int, default=3,
                        help="Number of independent suffix rollouts from each realized post-action state (default: 3).")
    parser.add_argument("--out", type=Path, default=Path("counterfactual_pairs.jsonl"),
                        help="Output JSONL path.")
    parser.add_argument("--cf-temperature", type=float, default=1.0,
                        help="Temperature for counterfactual action sampling and all suffix rollouts (default: 1.0).")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max LLM turns for the counterfactual rollout (default: 10).")
    parser.add_argument("--judge-model-name", type=str, default="vllm-judge-local",
                        help="LLM judge model for symbolic equivalence (default: vllm-judge-local).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Top-level RNG seed for branch-root sampling.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default: 1).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-root branching, first-action, and evaluation details.")
    parser.add_argument("--sample-with-replacement", action="store_true",
                        help="Sample branch roots with replacement (default: without, capped at num available).")
    parser.add_argument("--min-branch-turn", type=int, default=1,
                        help="Minimum outer-turn index eligible for branching (default: 1, so the first assistant action is never re-sampled).")
    parser.add_argument("--resume", action="store_true",
                        help="If <out>.partial exists, skip pairs that already have a complete record set "
                             "(terminal record, or all num-suffix-rollouts continuations).")
    args = parser.parse_args()

    if args.num_traj <= 0:
        print("[ERROR] --num-traj must be positive.", file=sys.stderr)
        return 2
    if args.num_suffix_rollouts <= 0:
        print("[ERROR] --num-suffix-rollouts must be positive.", file=sys.stderr)
        return 2

    if not args.trial_dir.exists():
        print(f"[ERROR] trial-dir {args.trial_dir} does not exist.", file=sys.stderr)
        return 2

    trials = discover_trials(args.trial_dir)
    if not trials:
        print(f"[ERROR] No trial*.json files found under {args.trial_dir}.", file=sys.stderr)
        return 2

    root_candidates: List[Dict[str, Any]] = []
    discovered_by_backend: Counter = Counter()
    branchable_trials_by_backend: Counter = Counter()
    branch_roots_by_backend: Counter = Counter()

    for p in trials:
        trial = load_trial(p)
        if trial is None:
            continue

        agent_backend = trial.get("agent_backend", "vanilla_agent")
        discovered_by_backend[agent_backend] += 1
        effective_max_turns = int(trial.get("max_turns") or args.max_turns)
        points = extract_decision_points(trial.get("chat_history") or [], agent_backend, effective_max_turns)
        viable_points = [point for point in points if point["outer_turn"] >= args.min_branch_turn]
        if viable_points:
            branchable_trials_by_backend[agent_backend] += 1

        for point in viable_points:
            root_candidates.append(
                {
                    "trial_path": str(p),
                    "decision_index": point["decision_index"],
                    "assistant_message_index": point["assistant_message_index"],
                    "outer_turn": point["outer_turn"],
                    "phase": point["phase"],
                    "original_action_type": point["original_action_type"],
                    "agent_backend": agent_backend,
                }
            )
            branch_roots_by_backend[agent_backend] += 1

    if not root_candidates:
        print(f"[ERROR] No trials have a valid branch root with min_branch_turn={args.min_branch_turn}.", file=sys.stderr)
        return 2

    print(
        f"[INFO] Found {len(trials)} completed trials under {args.trial_dir}; "
        f"{sum(branchable_trials_by_backend.values())} have at least one eligible branch root; "
        f"{len(root_candidates)} total branch roots are eligible."
    )
    print(f"[INFO] Completed trials by backend: {_format_backend_counts(discovered_by_backend)}")
    print(f"[INFO] Branchable trials by backend: {_format_backend_counts(branchable_trials_by_backend)}")
    print(f"[INFO] Branch roots by backend: {_format_backend_counts(branch_roots_by_backend)}")

    rng = random.Random(args.seed)
    if args.sample_with_replacement:
        sampled_roots = [rng.choice(root_candidates) for _ in range(args.num_traj)]
    else:
        n = min(args.num_traj, len(root_candidates))
        sampled_roots = rng.sample(root_candidates, n)
        if n < args.num_traj:
            print(f"[WARN] Only {len(root_candidates)} unique branch roots; producing {n} roots "
                  f"(use --sample-with-replacement to reach {args.num_traj}).")

    if args.verbose:
        sampled_by_backend: Counter = Counter(root["agent_backend"] for root in sampled_roots)
        print(f"[INFO] Sampled {len(sampled_roots)} branch roots.")
        print(f"[INFO] Sampled branch roots by backend: {_format_backend_counts(sampled_by_backend)}")
        for i, root in enumerate(sampled_roots):
            print(
                f"[SAMPLED {i}] trial={root['trial_path']} decision_index={root['decision_index']} "
                f"outer_turn={root['outer_turn']} phase={root['phase']} action={root['original_action_type']}"
            )

    root_tasks: List[Dict[str, Any]] = []
    for i, root in enumerate(sampled_roots):
        root_tasks.append(
            {
                "trial_path": root["trial_path"],
                "decision_index": root["decision_index"],
                "pair_index": i,
                "cf_temperature": args.cf_temperature,
                "max_turns": args.max_turns,
                "judge_model_name": args.judge_model_name,
                "num_suffix_rollouts": args.num_suffix_rollouts,
                "verbose": args.verbose,
            }
        )

    # Stream records to a partial file as they complete so progress survives
    # crashes / SIGKILL. After both stages finish, we rewrite the canonical
    # output file with computed fanout_size and sorted order.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stream_path = args.out.with_suffix(args.out.suffix + ".partial")

    all_records: List[Dict[str, Any]] = []
    skip_pair_indices: Set[int] = set()

    if args.resume:
        all_records, skip_pair_indices = _load_resume_records(stream_path, args.num_suffix_rollouts)
        if all_records:
            print(f"[RESUME] Loaded {len(all_records)} records from {stream_path}; "
                  f"{len(skip_pair_indices)} pairs complete and will be skipped.")
        else:
            print(f"[RESUME] No usable partial file at {stream_path}; running from scratch.")

    if skip_pair_indices:
        before = len(root_tasks)
        root_tasks = [t for t in root_tasks if t["pair_index"] not in skip_pair_indices]
        print(f"[RESUME] {len(root_tasks)}/{before} root tasks remain after skipping complete pairs.")

    # Truncate the partial file to just the kept records (drops any
    # incomplete-pair leftovers); subsequent writes append.
    with stream_path.open("w", encoding="utf-8") as fout_init:
        for rec in all_records:
            fout_init.write(json.dumps(rec) + "\n")

    print(f"[STREAM] Writing intermediate results to {stream_path}")

    continuation_tasks: List[Dict[str, Any]] = []

    with stream_path.open("a", encoding="utf-8") as fout_partial:
        def stream_record(rec: Dict[str, Any]) -> None:
            all_records.append(rec)
            fout_partial.write(json.dumps(rec) + "\n")
            fout_partial.flush()

        def on_root_result(result):
            if not result:
                return
            recs = result.get("records", [])
            conts = result.get("continuation_tasks", [])
            for rec in recs:
                stream_record(rec)
            continuation_tasks.extend(conts)
            if recs:
                r = recs[0]
                rmsle = r["counterfactual"]["evaluation"].get("rmsle", float("nan"))
                print(f"[PAIR {r['pair_index']}] terminal → preference={r['preference']} rmsle={rmsle:.4f}")
            elif conts:
                print(f"[PAIR {conts[0]['common_record']['pair_index']}] spawned {len(conts)} continuation(s)")

        def on_continuation_result(rec):
            if rec is None:
                return
            stream_record(rec)
            rmsle = rec["counterfactual"]["evaluation"].get("rmsle", float("nan"))
            print(
                f"[PAIR {rec['pair_index']}][ROLLOUT {rec['rollout_index']}] preference={rec['preference']} "
                f"rmsle={rmsle:.4f} (trial {rec['trial_id']}, decision_index={rec['decision_index']}, fanout={rec.get('fanout_size', '?')})"
            )

        _run_task_map(materialize_one_root, root_tasks, args.workers,
                      desc="roots", on_result=on_root_result)
        _run_task_map(complete_one_continuation, continuation_tasks, args.workers,
                      desc="continuations", on_result=on_continuation_result)

    # Final pass: compute fanout_size from successful rollouts per group, sort,
    # and write the canonical output file.
    fanout_counts = Counter(rec["cf_group_id"] for rec in all_records)
    for rec in all_records:
        rec["fanout_size"] = fanout_counts[rec["cf_group_id"]]
    all_records.sort(key=lambda rec: (rec["pair_index"], rec.get("rollout_index", 0)))

    written = 0
    with args.out.open("w", encoding="utf-8") as fout:
        for rec in all_records:
            fout.write(json.dumps(rec) + "\n")
            written += 1

    stream_path.unlink(missing_ok=True)
    print(f"[DONE] Wrote {written} preference records to {args.out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
