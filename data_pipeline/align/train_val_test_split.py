"""Time-based train/val/test split for the training pairs, per the project guide's own rule (section 6):
split by time, never randomly by day (neighbouring days are nearly identical and would leak into the test set).

Range decided and documented in DEV_LOG.md (2026-09-24 "Training period locked to 1981-2026" entry):
train 1981-2018, val 2019-2020, test 2021-2026.
"""
from pathlib import Path

import numpy as np

TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"

TRAIN_YEARS = range(1981, 2019)  # 1981-2018 inclusive
VAL_YEARS = range(2019, 2021)  # 2019-2020 inclusive
TEST_YEARS = range(2021, 2027)  # 2021-2026 inclusive


def year_of(date_str: str) -> int:
    return int(date_str[:4])


def compute_split_indices(dates: np.ndarray) -> dict[str, np.ndarray]:
    years = np.array([year_of(d) for d in dates])
    train_idx = np.where(np.isin(years, list(TRAIN_YEARS)))[0]
    val_idx = np.where(np.isin(years, list(VAL_YEARS)))[0]
    test_idx = np.where(np.isin(years, list(TEST_YEARS)))[0]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def main() -> None:
    dates = np.load(TRAINING_PAIRS_DIR / "dates.npy")
    splits = compute_split_indices(dates)

    total = len(dates)
    assigned = sum(len(v) for v in splits.values())
    print(f"Total samples: {total}, assigned to a split: {assigned}")
    if assigned != total:
        unassigned_years = sorted(set(year_of(d) for d in dates) - set(TRAIN_YEARS) - set(VAL_YEARS) - set(TEST_YEARS))
        raise RuntimeError(f"{total - assigned} samples not assigned to any split - unassigned years: {unassigned_years}")

    overlap_tv = set(splits["train"]) & set(splits["val"])
    overlap_vt = set(splits["val"]) & set(splits["test"])
    overlap_tt = set(splits["train"]) & set(splits["test"])
    if overlap_tv or overlap_vt or overlap_tt:
        raise RuntimeError(f"Split overlap detected - train/val: {len(overlap_tv)}, val/test: {len(overlap_vt)}, train/test: {len(overlap_tt)}")

    for name, idx in splits.items():
        years_in_split = sorted(set(year_of(dates[i]) for i in idx))
        print(f"{name}: {len(idx)} samples, years {years_in_split[0]}-{years_in_split[-1]} ({len(years_in_split)} years)")

    np.savez(
        TRAINING_PAIRS_DIR / "split_indices.npz",
        train=splits["train"], val=splits["val"], test=splits["test"],
    )
    print(f"\nSaved: {TRAINING_PAIRS_DIR / 'split_indices.npz'}")
    print("No overlap between splits confirmed. Every sample assigned to exactly one split.")


if __name__ == "__main__":
    main()
