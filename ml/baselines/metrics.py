"""Shared metrics for baseline comparison: RMSE, MAE, correlation, plus the event-based scores the project
guide's section 6 asks for (POD/FAR/CSI at IMD's own rainfall category thresholds, 99th-percentile bias).

valid_mask excludes the known 6 static coastal-artifact cells (see DEV_LOG) from every metric - including them
would score a permanently-empty patch of ocean as "correctly predicted zero rain", silently inflating every
baseline's apparent accuracy.

NaN policy: a stray NaN in pred/truth (within the valid mask) silently poisons an averaged metric like RMSE to
`nan`, with no error - found via testing (2026-09-25): this could easily be misread later as "not computed yet"
rather than "the model produced garbage," especially once the U-Net is in the picture and NaN predictions during
unstable training are a real, common failure mode. All entry points below fail loudly instead.
"""
import numpy as np

# IMD's own official rainfall category thresholds (mm/day), per the project guide section 6
THRESHOLDS = {"light": 2.5, "moderate": 15.6, "heavy": 64.5}


def _check_no_nan(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
    if np.isnan(pred[:, mask]).any():
        raise ValueError("pred contains NaN within the valid mask - metrics would silently report NaN, refusing")
    if np.isnan(truth[:, mask]).any():
        raise ValueError("truth contains NaN within the valid mask - metrics would silently report NaN, refusing")


def rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    _check_no_nan(pred, truth, mask)
    diff = (pred - truth)[:, mask]
    return float(np.sqrt(np.mean(diff**2)))


def mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    _check_no_nan(pred, truth, mask)
    diff = (pred - truth)[:, mask]
    return float(np.mean(np.abs(diff)))


def correlation(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    _check_no_nan(pred, truth, mask)
    p = pred[:, mask].flatten()
    t = truth[:, mask].flatten()
    if np.std(p) == 0 or np.std(t) == 0:
        return float("nan")  # legitimate: undefined correlation for a constant (e.g. all-zero-rain) array, not a NaN-input error
    return float(np.corrcoef(p, t)[0, 1])


def event_scores(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, threshold: float) -> dict:
    """POD (probability of detection), FAR (false alarm ratio), CSI (critical success index) at a threshold."""
    _check_no_nan(pred, truth, mask)
    p = pred[:, mask] >= threshold
    t = truth[:, mask] >= threshold
    hits = np.sum(p & t)
    misses = np.sum(~p & t)
    false_alarms = np.sum(p & ~t)

    pod = hits / (hits + misses) if (hits + misses) > 0 else float("nan")
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else float("nan")
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else float("nan")
    return {"pod": float(pod), "far": float(far), "csi": float(csi), "hits": int(hits), "misses": int(misses), "false_alarms": int(false_alarms)}


def percentile_99_bias(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Bias at the 99th percentile - are extremes smoothed away? Closer to 0 is better."""
    _check_no_nan(pred, truth, mask)
    p99_pred = np.percentile(pred[:, mask], 99)
    p99_truth = np.percentile(truth[:, mask], 99)
    return float(p99_pred - p99_truth)


def full_report(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    report = {
        "rmse": rmse(pred, truth, mask),
        "mae": mae(pred, truth, mask),
        "correlation": correlation(pred, truth, mask),
        "p99_bias": percentile_99_bias(pred, truth, mask),
    }
    for name, thresh in THRESHOLDS.items():
        report[f"event_{name}"] = event_scores(pred, truth, mask, thresh)
    return report
