"""Statistical analysis functions for the audio enhancement benchmark.

Provides Friedman tests, post-hoc Wilcoxon signed-rank tests with
Bonferroni correction, bootstrap confidence intervals, per-stratum and
per-format breakdowns, and cross-metric agreement analysis (Spearman).
"""

from __future__ import annotations

from itertools import combinations

ANALYSIS_METRICS = [
    ("dnsmos_ovrl", "DNSMOS OVRL"),
    ("dnsmos_sig", "DNSMOS SIG"),
    ("dnsmos_bak", "DNSMOS BAK"),
    ("utmos_score", "UTMOS"),
    ("nisqa_mos", "NISQA MOS"),
]

# All sub-metrics for the quality profile table (no full statistical analysis)
_ALL_METRICS = [
    ("dnsmos_ovrl", "OVRL"),
    ("dnsmos_sig", "SIG"),
    ("dnsmos_bak", "BAK"),
    ("dnsmos_p808", "P.808"),
    ("utmos_score", "UTMOS"),
    ("nisqa_mos", "NISQA"),
    ("nisqa_noisiness", "Noise"),
    ("nisqa_coloration", "Color"),
    ("nisqa_discontinuity", "Discont"),
    ("nisqa_loudness", "Loud"),
]


def _bootstrap_ci(
    data, n_bootstrap: int = 10000, alpha: float = 0.05, seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap percentile confidence interval for the mean.

    Returns (ci_lo, ci_hi) for the (1-alpha) CI.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(data)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def run_statistical_tests(
    segments: list[dict], pipeline_names: list[str],
) -> dict:
    """Run multi-metric statistical analysis: Friedman + Wilcoxon + CIs."""
    result: dict = {"n_segments": len(segments), "per_metric": {}}

    for metric_key, metric_label in ANALYSIS_METRICS:
        metric_result = run_tests_for_metric(
            segments, pipeline_names, metric_key, metric_label,
        )
        if metric_result is not None:
            result["per_metric"][metric_key] = metric_result

    # Cross-metric agreement
    result["cross_metric"] = cross_metric_agreement(segments, pipeline_names)

    # Backward compat: alias primary metric fields at top level
    primary = result["per_metric"].get("dnsmos_ovrl", {})
    result["metric"] = "dnsmos_ovrl"
    result["friedman"] = primary.get("friedman", {})
    result["mean_ranks"] = primary.get("mean_ranks", {})
    result["descriptive"] = primary.get("descriptive", {})
    result["pairwise"] = primary.get("pairwise", {})
    result["alpha_corrected"] = primary.get("alpha_corrected", 0)
    result["per_stratum"] = primary.get("per_stratum", {})
    result["per_format"] = primary.get("per_format", {})

    return result


def run_tests_for_metric(
    segments: list[dict],
    pipeline_names: list[str],
    metric_key: str,
    metric_label: str,
) -> dict | None:
    """Run Friedman test and post-hoc Wilcoxon for a single metric."""
    import numpy as np
    from scipy import stats as sp_stats
    from scipy.stats import rankdata

    n_pipes = len(pipeline_names)

    # Build score matrix: segments x pipelines (only complete rows)
    score_matrix = []
    valid_segments = []
    for seg in segments:
        row = []
        all_present = True
        for pipe in pipeline_names:
            val = seg["scores"].get(pipe, {}).get(metric_key)
            if val is None:
                all_present = False
                break
            row.append(val)
        if all_present:
            score_matrix.append(row)
            valid_segments.append(seg)

    if len(score_matrix) < 3:
        return None

    matrix = np.array(score_matrix)
    result: dict = {
        "metric_key": metric_key,
        "metric_label": metric_label,
        "n_segments": len(score_matrix),
    }

    # Friedman test
    friedman_stat, friedman_p = sp_stats.friedmanchisquare(
        *[matrix[:, i] for i in range(n_pipes)]
    )
    result["friedman"] = {
        "statistic": float(friedman_stat),
        "p_value": float(friedman_p),
        "significant": friedman_p < 0.05,
    }

    # Mean ranks
    ranks = np.apply_along_axis(rankdata, 1, matrix)
    mean_ranks = ranks.mean(axis=0)
    result["mean_ranks"] = {
        pipe: float(mean_ranks[i]) for i, pipe in enumerate(pipeline_names)
    }

    # Descriptive stats with bootstrap CIs
    result["descriptive"] = {}
    for i, pipe in enumerate(pipeline_names):
        col = matrix[:, i]
        ci_lo, ci_hi = _bootstrap_ci(col)
        result["descriptive"][pipe] = {
            "mean": float(np.mean(col)),
            "median": float(np.median(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "ci_95": [ci_lo, ci_hi],
        }

    # Post-hoc Wilcoxon signed-rank tests (Bonferroni corrected)
    n_pairs = n_pipes * (n_pipes - 1) // 2
    alpha_corrected = 0.05 / n_pairs if n_pairs > 0 else 0.05
    pairwise = {}

    for (i, pipe_a), (j, pipe_b) in combinations(enumerate(pipeline_names), 2):
        diff = matrix[:, j] - matrix[:, i]
        if np.all(diff == 0):
            pairwise[f"{pipe_a}_vs_{pipe_b}"] = {
                "statistic": 0, "p_value": 1.0,
                "significant": False, "effect_size": 0.0,
            }
            continue

        try:
            stat, p = sp_stats.wilcoxon(matrix[:, i], matrix[:, j])
            n = len(diff[diff != 0])
            r = 1 - (2 * stat) / (n * (n + 1)) if n > 0 else 0.0
            pairwise[f"{pipe_a}_vs_{pipe_b}"] = {
                "statistic": float(stat),
                "p_value": float(p),
                "significant": p < alpha_corrected,
                "effect_size": float(r),
            }
        except Exception as e:
            pairwise[f"{pipe_a}_vs_{pipe_b}"] = {"error": str(e)}

    result["pairwise"] = pairwise
    result["alpha_corrected"] = alpha_corrected

    # Per-stratum analysis
    result["per_stratum"] = per_stratum_analysis(
        valid_segments, matrix, pipeline_names,
    )

    # Per-format (content type) analysis
    result["per_format"] = per_format_analysis(
        valid_segments, matrix, pipeline_names,
    )

    return result


def per_stratum_analysis(
    segments: list[dict],
    matrix,
    pipeline_names: list[str],
) -> dict:
    """Run Friedman test per series_group stratum."""
    import numpy as np
    from scipy import stats as sp_stats

    strata_results: dict = {}
    groups: dict[str, list[int]] = {}
    for idx, seg in enumerate(segments):
        sg = seg["strata"].get("series_group", "unknown")
        groups.setdefault(sg, []).append(idx)

    for sg, indices in sorted(groups.items()):
        sub_matrix = matrix[indices]
        n = len(indices)
        if n < 3:
            strata_results[sg] = {
                "n": n,
                "note": "too few samples for Friedman test",
                "means": {
                    pipe: float(np.mean(sub_matrix[:, i]))
                    for i, pipe in enumerate(pipeline_names)
                },
            }
            continue

        try:
            stat, p = sp_stats.friedmanchisquare(
                *[sub_matrix[:, i] for i in range(len(pipeline_names))]
            )
            strata_results[sg] = {
                "n": n,
                "friedman_statistic": float(stat),
                "friedman_p": float(p),
                "significant": p < 0.05,
                "means": {
                    pipe: float(np.mean(sub_matrix[:, i]))
                    for i, pipe in enumerate(pipeline_names)
                },
            }
        except Exception as e:
            strata_results[sg] = {"n": n, "error": str(e)}

    return strata_results


def per_format_analysis(
    segments: list[dict],
    matrix,
    pipeline_names: list[str],
) -> dict:
    """Run Friedman test per format_group (content type) stratum.

    Also computes best_pipeline per format group (highest mean OVRL
    excluding original) to support content-type-aware pipeline selection.
    """
    import numpy as np
    from scipy import stats as sp_stats

    format_results: dict = {}
    groups: dict[str, list[int]] = {}
    for idx, seg in enumerate(segments):
        fg = seg["strata"].get("format_group", "lecture")
        groups.setdefault(fg, []).append(idx)

    enhanced_indices = [
        i for i, p in enumerate(pipeline_names) if p != "original"
    ]

    for fg, indices in sorted(groups.items()):
        sub_matrix = matrix[indices]
        n = len(indices)
        means = {
            pipe: float(np.mean(sub_matrix[:, i]))
            for i, pipe in enumerate(pipeline_names)
        }

        # Best pipeline (highest mean, excluding original)
        if enhanced_indices:
            best_idx = max(
                enhanced_indices, key=lambda i: float(np.mean(sub_matrix[:, i])),
            )
            best_pipeline = pipeline_names[best_idx]
            best_mean = float(np.mean(sub_matrix[:, best_idx]))
        else:
            best_pipeline = ""
            best_mean = 0.0

        if n < 3:
            format_results[fg] = {
                "n": n,
                "note": "too few samples for Friedman test",
                "means": means,
                "best_pipeline": best_pipeline,
                "best_mean": best_mean,
            }
            continue

        try:
            stat, p = sp_stats.friedmanchisquare(
                *[sub_matrix[:, i] for i in range(len(pipeline_names))]
            )
            format_results[fg] = {
                "n": n,
                "friedman_statistic": float(stat),
                "friedman_p": float(p),
                "significant": p < 0.05,
                "means": means,
                "best_pipeline": best_pipeline,
                "best_mean": best_mean,
            }
        except Exception as e:
            format_results[fg] = {
                "n": n,
                "error": str(e),
                "means": means,
                "best_pipeline": best_pipeline,
                "best_mean": best_mean,
            }

    return format_results


def cross_metric_agreement(
    segments: list[dict], pipeline_names: list[str],
) -> dict:
    """Compute Spearman correlations between improvement deltas across metrics.

    For each segment, compute improvement delta (enhanced - original) for each
    metric. Then correlate these deltas across metric pairs. Weak agreement
    (|rho| < 0.4) flags potential metric disagreement.
    """
    import numpy as np
    from scipy import stats as sp_stats

    metric_keys = [m[0] for m in ANALYSIS_METRICS]
    enhanced_pipes = [p for p in pipeline_names if p != "original"]

    # Collect per-metric improvement deltas: {metric: list of deltas}
    metric_deltas: dict[str, list[float | None]] = {m: [] for m in metric_keys}

    for seg in segments:
        orig_scores = seg["scores"].get("original", {})
        for pipe in enhanced_pipes:
            pipe_scores = seg["scores"].get(pipe, {})
            for mk in metric_keys:
                orig_val = orig_scores.get(mk)
                pipe_val = pipe_scores.get(mk)
                if orig_val is not None and pipe_val is not None:
                    metric_deltas[mk].append(pipe_val - orig_val)
                else:
                    metric_deltas[mk].append(None)

    # Filter to metrics with enough data
    available = {
        mk: vals for mk, vals in metric_deltas.items()
        if sum(1 for v in vals if v is not None) >= 5
    }

    results: dict = {}
    avail_keys = sorted(available.keys())
    for i, mk_a in enumerate(avail_keys):
        for mk_b in avail_keys[i + 1:]:
            vals_a = available[mk_a]
            vals_b = available[mk_b]
            # Pair only where both are non-None
            paired = [
                (a, b) for a, b in zip(vals_a, vals_b)
                if a is not None and b is not None
            ]
            if len(paired) < 5:
                continue
            arr_a = np.array([p[0] for p in paired])
            arr_b = np.array([p[1] for p in paired])
            rho, p_val = sp_stats.spearmanr(arr_a, arr_b)
            label_a = next(lbl for k, lbl in ANALYSIS_METRICS if k == mk_a)
            label_b = next(lbl for k, lbl in ANALYSIS_METRICS if k == mk_b)
            results[f"{mk_a}_vs_{mk_b}"] = {
                "labels": f"{label_a} vs {label_b}",
                "spearman_rho": float(rho),
                "p_value": float(p_val),
                "n_pairs": len(paired),
                "weak_agreement": abs(rho) < 0.4,
            }

    return results
