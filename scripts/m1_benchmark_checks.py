"""Device-free summaries and correctness gates shared by M1 benchmarks."""

import math
import statistics
from bench_final_stack import PROMPTS


def comparison(a, b):
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    return {
        "identical": a == b,
        "first_mismatch": first,
        "equal_prefix": len(a) if first is None else first,
        "positions_equal": sum(x == y for x, y in zip(a, b)),
        "lengths": [len(a), len(b)],
    }


def summarize(rows):
    summary = {}
    for label in ("mlx_bf16", "ane_lut6_ar", "ane_lut6_sd"):
        group = [r for r in rows if r["impl"] == label]
        if group:
            summary[label] = {
                "mean_tok_s": statistics.mean(r["tok_per_s_decode"] for r in group),
                "trials": len(group),
                "per_prompt": {
                    name: statistics.mean(
                        r["tok_per_s_decode"] for r in group if r["name"] == name
                    )
                    for name, _ in PROMPTS
                    if any(r["name"] == name for r in group)
                },
            }
    pairs = []
    for sd in (r for r in rows if r["impl"] == "ane_lut6_sd"):
        group = {
            r["impl"]: r
            for r in rows
            if (r["name"], r["repeat"]) == (sd["name"], sd["repeat"])
        }
        if len(group) != 3:
            continue
        pairs.append(
            {
                "name": sd["name"],
                "repeat": sd["repeat"],
                "vs_mlx": sd["tok_per_s_decode"]
                / group["mlx_bf16"]["tok_per_s_decode"],
                "vs_ane_ar": sd["tok_per_s_decode"]
                / group["ane_lut6_ar"]["tok_per_s_decode"],
                "tokens_vs_mlx": comparison(sd["tokens"], group["mlx_bf16"]["tokens"]),
                "tokens_vs_ane_ar": comparison(
                    sd["tokens"], group["ane_lut6_ar"]["tokens"]
                ),
            }
        )
    summary["pairs"] = pairs
    if pairs:
        for label in ("vs_mlx", "vs_ane_ar"):
            summary[label] = {
                "mean_paired_speedup": statistics.mean(p[label] for p in pairs),
                "max_trial_speedup": max(p[label] for p in pairs),
                "per_prompt": {
                    name: statistics.mean(p[label] for p in pairs if p["name"] == name)
                    for name, _ in PROMPTS
                    if any(p["name"] == name for p in pairs)
                },
            }
        for label in ("tokens_vs_mlx", "tokens_vs_ane_ar"):
            summary[label] = {
                "identical_trials": sum(p[label]["identical"] for p in pairs),
                "trials": len(pairs),
                "positions_equal": sum(p[label]["positions_equal"] for p in pairs),
                "total_positions": sum(p[label]["lengths"][0] for p in pairs),
            }
    return summary


def require_same_target_identity(rows, repeats, prompt_names=None):
    """Require complete, unique trials and exact SD/ANE-control token identity."""
    prompt_names = tuple(
        prompt_names if prompt_names is not None else (n for n, _ in PROMPTS)
    )
    if repeats < 1 or not prompt_names or len(set(prompt_names)) != len(prompt_names):
        raise ValueError("Need positive repeats and distinct prompts")
    labels = ("mlx_bf16", "ane_lut6_ar", "ane_lut6_sd")
    expected = {
        (name, repeat, label)
        for name in prompt_names
        for repeat in range(1, repeats + 1)
        for label in labels
    }
    observed = {}
    for row in rows:
        key = (row["name"], row["repeat"], row["impl"])
        if key not in expected or key in observed:
            raise ValueError(f"Unexpected or duplicate benchmark trial: {key}")
        if (
            not row["tokens"]
            or not math.isfinite(row["tok_per_s_decode"])
            or row["tok_per_s_decode"] <= 0
        ):
            raise ValueError(f"Empty tokens or invalid benchmark rate: {key}")
        observed[key] = row
    missing = expected - observed.keys()
    if missing:
        raise ValueError(f"Missing benchmark trials: {sorted(missing)}")
    checks = []
    for name in prompt_names:
        for repeat in range(1, repeats + 1):
            sd = observed[name, repeat, "ane_lut6_sd"]
            ar = observed[name, repeat, "ane_lut6_ar"]
            check = comparison(sd["tokens"], ar["tokens"])
            if not check["identical"]:
                raise ValueError(
                    f"SD differs from the same ANE target: {name}, repeat {repeat}, "
                    f"first mismatch {check['first_mismatch']}, lengths {check['lengths']}"
                )
            checks.append({"name": name, "repeat": repeat, **check})
    return {
        "passed": True,
        "pairs": len(checks),
        "matched_tokens": sum(c["lengths"][0] for c in checks),
        "checks": checks,
    }


def finish_benchmark(result):
    # A failed gate raises into the driver's failure handler before success is saved.
    result["same_target_check"] = require_same_target_identity(
        result["rows"], result["repeats"]
    )
    result["status"] = "complete"
