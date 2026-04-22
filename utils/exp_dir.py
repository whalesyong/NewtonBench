import os
import re


def get_next_exp_id(model_name, base_dir="evaluation_results"):
    model_dir = os.path.join(base_dir, model_name)
    if not os.path.exists(model_dir):
        return 1
    existing = []
    for name in os.listdir(model_dir):
        match = re.match(r'^exp_(\d+)$', name)
        if match and os.path.isdir(os.path.join(model_dir, name)):
            existing.append(int(match.group(1)))
    return max(existing) + 1 if existing else 1


def make_exp_dir(model_name, base_dir="evaluation_results"):
    exp_id = get_next_exp_id(model_name, base_dir)
    exp_dir = os.path.join(base_dir, model_name, f"exp_{exp_id}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_id, exp_dir


def find_exp_dirs(model_name, base_dir="evaluation_results"):
    model_dir = os.path.join(base_dir, model_name)
    if not os.path.exists(model_dir):
        return []
    dirs = []
    for name in os.listdir(model_dir):
        match = re.match(r'^exp_(\d+)$', name)
        if match and os.path.isdir(os.path.join(model_dir, name)):
            dirs.append((int(match.group(1)), os.path.join(model_dir, name)))
    dirs.sort(key=lambda x: x[0])
    return dirs