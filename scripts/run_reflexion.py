#!/usr/bin/env python3
"""
Run reflexion experiments: actor explores a physical law across multiple episodes,
with a reflector model providing guidance between episodes.

Each episode runs the standard agent backend (vanilla or code-assisted).
After each episode, a reflector model analyzes the trajectory summary and
produces guidance that gets prepended to the actor's task prompt for the
next episode.

Usage:
    python scripts/run_reflexion.py \\
        --model_name vllm-local \\
        --reflector_model vllm-judge-local \\
        --agent_backend vanilla_agent \\
        -m complex_system \\
        -l v0 \\
        --max_episodes 5 \\
        --trials 12 \\
        --temperature 0.4
"""
import os
import sys
import json
import argparse
import time
import re
import types
import importlib
import traceback
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------------------------
# Global lists — edit these to control which modules/difficulties are run
# ---------------------------------------------------------------------------

MODULES = [
    "m0_gravity",
    "m1_coulomb_force",
    "m2_magnetic_force",
    "m3_fourier_law",
    "m4_snell_law",
    "m5_radioactive_decay",
    "m6_underdamped_harmonic",
    "m7_malus_law",
    "m8_sound_speed",
    "m9_hooke_law",
    "m10_be_distribution",
    "m11_heat_transfer",
]

DIFFICULTIES = ["easy", "medium", "hard"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

warnings.filterwarnings("ignore")

from utils.exp_dir import get_next_exp_id
from utils.vanilla_agent import conduct_exploration
from utils.call_llm_api import call_llm_api

try:
    from utils.code_assisted_agent import conduct_code_assisted_exploration
    _WITH_CODE_ASSISTANCE = True
except Exception:
    _WITH_CODE_ASSISTANCE = False

# ---------------------------------------------------------------------------
# Reflector prompt templates
# ---------------------------------------------------------------------------

REFLECTOR_SYSTEM_PROMPT = """You are a scientific research advisor helping a scientist discover an unknown physical law through experimentation.

The scientist conducts experiments by varying physical parameters and observing results. Each episode, the scientist explores the parameter space and (usually) submits a candidate mathematical law.

Your task: analyze the scientist's completed episode and provide guidance for their next attempt.

GUIDELINES:
- You will be shown CONFIDENTIAL evaluation metrics (RMSLE, symbolic equivalence). Use them to calibrate the quality of your advice, but NEVER mention these metrics or specific numeric scores in your reflection.
- Focus on what patterns the scientist explored, what they may have missed, and what they should try next.
- Be concise (2–4 paragraphs). Offer concrete, actionable advice — not generic encouragement.
- If previous reflections exist (shown below), build on them; avoid repeating the same advice.
- Mention specific parameter ranges or relationships the scientist should investigate.

Format your response EXACTLY as:
<reflection>
Your 2-4 paragraph reflection here...
</reflection>"""

REFLECTOR_USER_PROMPT = """## Task Summary
{task_description}

## Previous Reflections
{previous_reflections}

## Episode {episode_num} Trajectory Summary
{action_summary}

## Outcome
Status: {status}
Submitted Law:
```python
{submitted_law}
```

## Confidential Evaluation (DO NOT MENTION THESE)
- RMSLE: {rmsle}
- Symbolically Equivalent: {symbolic_equivalent}

Provide your reflection to help the scientist improve in the next attempt."""

# ---------------------------------------------------------------------------
# Module monkey-patching
# ---------------------------------------------------------------------------

def create_reflected_module(module, reflection_text):
    """Create a shallow copy of *module* whose get_task_prompt prepends
    *reflection_text* (if any) to the original prompt."""
    mod = types.ModuleType(module.__name__)
    mod.__dict__.update(module.__dict__)

    original_get_task_prompt = module.get_task_prompt

    def patched_get_task_prompt(system, is_code_assisted=False, noise_level=0.0):
        base = original_get_task_prompt(system, is_code_assisted, noise_level)
        if reflection_text:
            return f"<reflection>\n{reflection_text}\n</reflection>\n\n{base}"
        return base

    mod.get_task_prompt = patched_get_task_prompt
    return mod


# ---------------------------------------------------------------------------
# Action summary extraction
# ---------------------------------------------------------------------------

def extract_action_summary(chat_history, agent_backend):
    """Walk the chat_history and extract all actions (experiments, python code)
    and their environment feedback (experiment outputs, python outputs)."""
    actions = []
    for msg in chat_history:
        content = msg.get("content", "")
        role = msg.get("role", "")

        if role == "assistant":
            # <run_experiment> blocks
            for m in re.finditer(r"<run_experiment>(.*?)</run_experiment>", content, re.DOTALL):
                raw = m.group(1).strip()
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        for item in parsed:
                            actions.append({"type": "experiment", "params": item})
                    else:
                        actions.append({"type": "experiment", "params": parsed})
                except (json.JSONDecodeError, TypeError):
                    actions.append({"type": "experiment", "raw": raw[:500]})

            # <python> blocks (code_assisted agent only)
            if agent_backend == "code_assisted_agent":
                for m in re.finditer(r"<python>(.*?)</python>", content, re.DOTALL):
                    code = m.group(1).strip()
                    actions.append({"type": "python", "code": code[:300]})

        elif role == "user":
            # <experiment_output> blocks
            for m in re.finditer(r"<experiment_output>(.*?)</experiment_output>", content, re.DOTALL):
                result = m.group(1).strip()
                actions.append({"type": "experiment_output", "result": result[:500]})

            # <python_output> blocks
            for m in re.finditer(r"<python_output>(.*?)</python_output>", content, re.DOTALL):
                result = m.group(1).strip()
                actions.append({"type": "python_output", "result": result[:500]})

    return actions


def format_action_summary(actions):
    """Render the action list as a readable text block."""
    if not actions:
        return "(No actions recorded)"

    lines = []
    for a in actions:
        t = a["type"]
        if t == "experiment":
            p = a.get("params", a.get("raw", "?"))
            lines.append(f"  Experiment: {json.dumps(p) if isinstance(p, dict) else p}")
        elif t == "experiment_output":
            lines.append(f"    → Result: {a['result']}")
        elif t == "python":
            lines.append(f"  Python code:\n```python\n{a['code']}\n```")
        elif t == "python_output":
            lines.append(f"    → Output: {a['result']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reflector call
# ---------------------------------------------------------------------------

def format_previous_reflections(reflections):
    if not reflections:
        return "(None — this is the first episode)"
    parts = []
    for i, ref in enumerate(reflections):
        parts.append(f"### After Episode {i + 1}\n{ref}")
    return "\n\n".join(parts)


def format_reflections_for_actor(reflections):
    if not reflections:
        return ""
    parts = []
    for i, ref in enumerate(reflections):
        parts.append(f"## Guidance from Episode {i + 1}\n{ref}")
    return "\n\n".join(parts)


def build_task_description(module, system):
    prompt = module.get_task_prompt(system, noise_level=0.0)
    max_len = 2000
    if len(prompt) > max_len:
        prompt = prompt[:max_len] + "\n...(truncated)..."
    return prompt


def call_reflector(reflector_model, task_desc, action_summary, submitted_law,
                   status, rmsle, symbolic_equivalent, previous_reflections,
                   episode_num, temperature, trial_info):
    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": REFLECTOR_USER_PROMPT.format(
            task_description=task_desc,
            previous_reflections=format_previous_reflections(previous_reflections),
            episode_num=episode_num + 1,
            action_summary=action_summary,
            status=status,
            submitted_law=submitted_law,
            rmsle=f"{rmsle:.6f}" if isinstance(rmsle, (int, float)) else str(rmsle),
            symbolic_equivalent=symbolic_equivalent,
        )}
    ]

    response_text, reasoning_response, tokens = call_llm_api(
        messages, model_name=reflector_model, temperature=temperature,
        trial_info=trial_info,
    )

    if response_text is None:
        response_text = ""

    match = re.search(r"<reflection>(.*?)</reflection>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip(), tokens
    return response_text.strip(), tokens


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_single_episode(module, model_name, agent_backend, noise_level, difficulty,
                       system, law_version, max_turns, temperature, trial_info):
    if agent_backend == "code_assisted_agent" and _WITH_CODE_ASSISTANCE:
        result = conduct_code_assisted_exploration(
            module=module,
            model_name=model_name,
            noise_level=noise_level,
            difficulty=difficulty,
            system=system,
            law_version=law_version,
            trial_info=trial_info,
            temperature=temperature,
        )
    else:
        result = conduct_exploration(
            module=module,
            model_name=model_name,
            noise_level=noise_level,
            difficulty=difficulty,
            system=system,
            law_version=law_version,
            max_turns=max_turns,
            trial_info=trial_info,
            temperature=temperature,
        )
    return result


# ---------------------------------------------------------------------------
# Per-trial reflexion loop
# ---------------------------------------------------------------------------

def run_reflexion_trial(args):
    (trial_id, noise_level, model_name, module_name, difficulty, system,
     law_version, max_episodes, max_turns, agent_backend, temperature,
     reflector_model, judge_model, trial_dir) = args

    print(f"[Reflexion Trial {trial_id}] Starting ({max_episodes} episodes, "
          f"actor={model_name}, reflector={reflector_model}, module={module_name}, "
          f"system={system}, difficulty={difficulty}, law={law_version})")

    trial_info = {"trial_id": trial_id, "trial_dir": str(trial_dir)}
    module = importlib.import_module(f"modules.{module_name}")
    task_desc = build_task_description(module, system)

    reflections = []
    all_episode_results = []
    total_tokens = 0
    best_episode = -1
    best_rmsle = float("inf")

    os.makedirs(trial_dir, exist_ok=True)

    for episode in range(max_episodes):
        ep_label = f"[Reflexion Trial {trial_id} Ep {episode + 1}/{max_episodes}]"

        try:
            # 1. Build module with accumulated reflections
            actor_reflection_text = format_reflections_for_actor(reflections)
            reflected_module = create_reflected_module(module, actor_reflection_text)

            # 2. Run the actor
            ep_info = {**trial_info, "trial_dir": str(trial_dir / f"episode_{episode}")}
            os.makedirs(trial_dir / f"episode_{episode}", exist_ok=True)

            exploration_result = run_single_episode(
                module=reflected_module,
                model_name=model_name,
                agent_backend=agent_backend,
                noise_level=noise_level,
                difficulty=difficulty,
                system=system,
                law_version=law_version,
                max_turns=max_turns,
                temperature=temperature,
                trial_info=ep_info,
            )
            total_tokens += exploration_result.get("total_tokens", 0)

            # 3. Evaluate this episode's law
            evaluation = module.evaluate_law(
                exploration_result["submitted_law"],
                param_description=module.PARAM_DESCRIPTION,
                difficulty=difficulty,
                law_version=law_version,
                judge_model_name=judge_model,
                trial_info=ep_info,
            )

            rmsle = evaluation.get("rmsle")
            sym_eq = evaluation.get("symbolic_equivalent", False)
            acc = evaluation.get("exact_accuracy", 0.0)
            chat_history = exploration_result.get("chat_history", [])

            episode_result = {
                "trial_id": trial_id,
                "episode": episode,
                "module_name": module_name,
                "model_name": model_name,
                "reflector_model": reflector_model,
                "agent_backend": agent_backend,
                "noise_level": noise_level,
                "temperature": temperature,
                "difficulty": difficulty,
                "system": system,
                "law_version": law_version,
                "reflection_count": len(reflections),
                "submitted_law": exploration_result.get("submitted_law"),
                "status": exploration_result.get("status"),
                "rounds": exploration_result.get("rounds"),
                "num_experiments": exploration_result.get("num_experiments"),
                "total_tokens": exploration_result.get("total_tokens", 0),
                "evaluation": evaluation,
            }
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"{ep_label} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

            rmsle = None
            sym_eq = False
            acc = 0.0
            chat_history = []
            evaluation = {
                "rmsle": None,
                "exact_accuracy": 0.0,
                "symbolic_equivalent": False,
                "symbolic_msg": None,
                "error": f"{type(e).__name__}: {e}",
            }
            episode_result = {
                "trial_id": trial_id,
                "episode": episode,
                "module_name": module_name,
                "model_name": model_name,
                "reflector_model": reflector_model,
                "agent_backend": agent_backend,
                "noise_level": noise_level,
                "temperature": temperature,
                "difficulty": difficulty,
                "system": system,
                "law_version": law_version,
                "reflection_count": len(reflections),
                "status": "failed",
                "submitted_law": f"{module.FUNCTION_SIGNATURE} return float('nan')",
                "rounds": 0,
                "num_experiments": 0,
                "total_tokens": 0,
                "evaluation": evaluation,
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb_str,
            }

        all_episode_results.append(episode_result)

        # Save episode result
        ep_json_path = trial_dir / f"episode_{episode}_result.json"
        with open(ep_json_path, "w", encoding="utf-8") as f:
            json.dump(episode_result, f, indent=2)

        # Save episode chat history
        if chat_history:
            ep_chat_path = trial_dir / f"episode_{episode}_chat_history.log"
            with open(ep_chat_path, "w", encoding="utf-8") as f:
                for msg in chat_history:
                    f.write(f"[{msg.get('role', '?')}]: {msg.get('content', '')}\n\n")

        # Track best
        if rmsle is not None and rmsle < best_rmsle:
            best_rmsle = rmsle
            best_episode = episode

        status_str = episode_result.get("status", "failed")
        print(f"{ep_label} status={status_str}, RMSLE={rmsle if rmsle is not None else 'N/A'}, "
              f"sym_eq={sym_eq}, acc={acc}, rounds={episode_result.get('rounds', 0)}, "
              f"experiments={episode_result.get('num_experiments', 0)}")

        # 4. Early stopping check (only if episode succeeded)
        if sym_eq and rmsle is not None and rmsle < 1e-6:
            print(f"{ep_label} Perfect score achieved — stopping early.")
            break

        # 5. Extract action summary for the reflector
        if episode_result.get("status") != "failed" and chat_history:
            actions = extract_action_summary(chat_history, agent_backend)
            action_summary = format_action_summary(actions)
        else:
            action_summary = "(Episode failed — no trajectory available)"

        # 6. Call reflector (skip after last episode)
        if episode < max_episodes - 1:
            try:
                reflection_text, ref_tokens = call_reflector(
                    reflector_model=reflector_model,
                    task_desc=task_desc,
                    action_summary=action_summary,
                    submitted_law=episode_result.get("submitted_law", "No law submitted"),
                    status=episode_result.get("status", "unknown"),
                    rmsle=rmsle,
                    symbolic_equivalent=sym_eq,
                    previous_reflections=reflections,
                    episode_num=episode,
                    temperature=temperature,
                    trial_info=trial_info,
                )
                total_tokens += ref_tokens
                reflections.append(reflection_text)

                # Save reflection
                ref_path = trial_dir / f"reflection_{episode}.txt"
                with open(ref_path, "w", encoding="utf-8") as f:
                    f.write(reflection_text)
                print(f"{ep_label} Reflection saved ({len(reflection_text)} chars)")
            except Exception as e:
                print(f"{ep_label} Reflector call FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                # Continue without this reflection; carry existing ones forward

    # 7. Write aggregated trial results
    aggregated = {
        "trial_id": trial_id,
        "module_name": module_name,
        "model_name": model_name,
        "reflector_model": reflector_model,
        "agent_backend": agent_backend,
        "noise_level": noise_level,
        "temperature": temperature,
        "difficulty": difficulty,
        "system": system,
        "law_version": law_version,
        "max_episodes": max_episodes,
        "episodes_run": len(all_episode_results),
        "best_episode": best_episode,
        "best_rmsle": best_rmsle if best_rmsle != float("inf") else None,
        "total_tokens": total_tokens,
        "episodes": all_episode_results,
    }

    agg_path = trial_dir / "aggregated_trial_results.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    print(f"[Reflexion Trial {trial_id}] Done. {len(all_episode_results)} episodes, "
          f"best_episode={best_episode}, best_rmsle={best_rmsle if best_rmsle != float('inf') else 'N/A'}, "
          f"total_tokens={total_tokens}")
    return aggregated


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run reflexion experiments with actor+reflector loop "
                    "across all modules and difficulties in the global lists.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model_name", required=True,
                        help="Actor model alias (e.g., vllm-local, gpt41mini)")
    parser.add_argument("--reflector_model", default=None,
                        help="Reflector model alias (default: vllm-judge-local)")
    parser.add_argument("--agent_backend", default="vanilla_agent",
                        choices=["vanilla_agent", "code_assisted_agent"],
                        help="Agent backend per episode")
    parser.add_argument("-m", "--model_system", default="complex_system",
                        help="Experiment system (default: complex_system)")
    parser.add_argument("-l", "--law_version", default="v0",
                        help="Law version (v0, v1, v2)")
    parser.add_argument("-n", "--noise", type=float, default=0.0,
                        help="Noise level")
    parser.add_argument("-t", "--trials", type=int, default=12,
                        help="Number of independent trials per combination")
    parser.add_argument("--max_episodes", type=int, default=5,
                        help="Maximum episodes per trial")
    parser.add_argument("--max_turns", type=int, default=10,
                        help="Maximum LLM turns per episode")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature")
    parser.add_argument("--exp_id", type=int, default=None,
                        help="Experiment ID (auto-detected if omitted)")
    parser.add_argument("--output_base", default="evaluation_results",
                        help="Base output directory")
    args = parser.parse_args()

    # Validate
    if args.agent_backend == "code_assisted_agent" and not _WITH_CODE_ASSISTANCE:
        print("ERROR: code_assisted_agent not available", file=sys.stderr)
        sys.exit(1)

    judge_model = "vllm-judge-local"
    reflector_model = args.reflector_model or "vllm-judge-local"

    combinations = [(m, d) for m in MODULES for d in DIFFICULTIES]
    total_trials_global = len(combinations) * args.trials

    print("=" * 60)
    print("NewtonBench — Reflexion Experiment Runner")
    print(f"  Actor model:     {args.model_name}")
    print(f"  Reflector model: {reflector_model}")
    print(f"  Agent backend:   {args.agent_backend}")
    print(f"  System:          {args.model_system}")
    print(f"  Law version:     {args.law_version}")
    print(f"  Noise:           {args.noise}")
    print(f"  Temperature:     {args.temperature}")
    print(f"  Max episodes:    {args.max_episodes}")
    print(f"  Max turns/ep:    {args.max_turns}")
    print(f"  Trials/comb:     {args.trials}")
    print(f"  Modules:         {len(MODULES)}")
    print(f"  Difficulties:    {len(DIFFICULTIES)}")
    print(f"  Combinations:    {len(combinations)}")
    print(f"  Total trials:    {total_trials_global}")
    print("=" * 60)

    # Resolve output directory
    exp_id = args.exp_id or get_next_exp_id(args.model_name, base_dir=args.output_base)
    exp_dir = Path(args.output_base) / args.model_name / f"exp_{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    noise_str = str(args.noise).replace(".", "_")
    temp_str = str(args.temperature).replace(".", "_")

    # Build trial args for every combination
    pool_args = []
    trial_counter = 0
    for module_name, difficulty in combinations:
        base_dir = exp_dir / module_name / "reflexion" / difficulty / args.law_version
        version_num = 1
        while True:
            experiment_name = f"{args.model_system}_noise{noise_str}_temp{temp_str}_v{version_num}"
            combo_results_dir = base_dir / experiment_name
            if not combo_results_dir.exists():
                break
            version_num += 1
        combo_results_dir.mkdir(parents=True, exist_ok=True)

        for local_trial in range(args.trials):
            pool_args.append((
                trial_counter, args.noise, args.model_name, module_name, difficulty,
                args.model_system, args.law_version, args.max_episodes, args.max_turns,
                args.agent_backend, args.temperature, reflector_model, judge_model,
                combo_results_dir / f"trial_{local_trial}",
            ))
            trial_counter += 1

    # Parallel execution
    actual_cpu_count = cpu_count()
    batch_size = min(actual_cpu_count, len(pool_args))
    num_batches = (len(pool_args) + batch_size - 1) // batch_size

    print(f"\nCPU count: {actual_cpu_count}, batch size: {batch_size}, "
          f"batches: {num_batches}\n")

    all_results = []
    start_time = time.time()

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(pool_args))
        batch_args = pool_args[start_idx:end_idx]
        actual_batch = len(batch_args)

        print(f"--- Batch {batch_num + 1}/{num_batches} "
              f"(trials {start_idx + 1}-{end_idx}) [{actual_batch} processes] ---")

        with Pool(processes=actual_batch) as pool:
            batch_results = pool.map(run_reflexion_trial, batch_args)
            all_results.extend(batch_results)

    elapsed = time.time() - start_time

    # Summary
    total_episodes = sum(r["episodes_run"] for r in all_results)
    best_episodes = [r.get("best_episode", -1) for r in all_results]
    best_rmsles = [r.get("best_rmsle") for r in all_results if r.get("best_rmsle") is not None]

    # Per-combination summary
    combo_summaries = {}
    for r in all_results:
        key = (r["module_name"], r["difficulty"])
        combo_summaries.setdefault(key, []).append(r.get("best_rmsle"))

    print("\n" + "=" * 60)
    print(f"All trials complete.  Elapsed: {elapsed:.0f}s")
    print(f"Combinations: {len(combinations)}, Total trials: {len(all_results)}, "
          f"Total episodes: {total_episodes}")
    print(f"\nPer-combination best RMSLE (mean across trials):")
    for (mod, diff), rmsles in sorted(combo_summaries.items()):
        valid = [x for x in rmsles if x is not None]
        if valid:
            import numpy as np
            print(f"  {mod:25s} {diff:6s}  mean={np.mean(valid):.4f}  "
                  f"std={np.std(valid):.4f}  min={np.min(valid):.4f}")
        else:
            print(f"  {mod:25s} {diff:6s}  (no valid RMSLE)")
    print(f"\nResults: {exp_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
