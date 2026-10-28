"""Build rectangular train/test Inputs and DNP configs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPTS_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SCRIPTS_DIR.parents[0]
DATASETS_DIR = PAPER_ROOT / "Datasets"
LSTM_INPUTS_DIR = PAPER_ROOT / "Models" / "LSTM_SplitTest" / "Inputs"

SIMILARITY_META_DIR = PAPER_ROOT / "Models" / "Similarity"

PAPER_INPUTS_DIR = PAPER_ROOT / "Inputs"
ATTRIBUTE_INPUT = DATASETS_DIR / "attribute_117_V1.parquet"
LSTM_CONFIG_TEMPLATE = PAPER_ROOT / "Models" / "LSTM_SplitTest" / "lstm_config.template.yaml"


SPLIT_MODE = "test_b_random_blocks_20pct"
TEST_FRACTION = 0.20
SPLIT_RANDOM_SEED = 21


SPLIT_RANDOM_SEEDS: tuple[int, ...] | None = None
MIN_OBS_TO_SPLIT = 50
MIN_TRAIN_OBS_AFTER_SPLIT = 40
MAX_BLOCK_TRIES = 500
MIN_FULL_BLOCKS_FOR_TEST = 3
CONTIGUOUS_INSUFFICIENT_POLICY = "exclude"
TEMPORAL_STRATIFY_BLOCKS = True
TEMPORAL_STRATIFY_N_BINS = 10
WRITE_LSTM_CONFIG = True

LSTM_CONFIG_FILENAME_TMPL = "lstm_config_{suffix}.yaml"


TEST_C_Q_LOW = 0.05
TEST_C_Q_HIGH = 0.95
TEST_C_Q_MID_LO = 0.45
TEST_C_Q_MID_HI = 0.55
TEST_C_FRAC_LOW = 0.05
TEST_C_FRAC_HIGH = 0.05
TEST_C_FRAC_MID = 0.10
HOLDOUT_YEARS = 1
GRAB_BIWEEKLY_DAYS = 14
GRAB_MONTHLY_PICK = "random"


PLOT_SITE_SPLITS = True
PLOT_DIRNAME = "split_plots"
PLOT_DPI = 150
PLOT_TRAIN_COLOR = "#009E73"
PLOT_TEST_COLOR = "#D55E00"
PLOT_TEXT_COLOR = "#001E44"
PLOT_FONT_FAMILY = "Segoe UI"
PLOT_FONT_SIZE = 14


RAW_DATE = "date"
RAW_SITE = "site_no"
RAW_TARGET = "nitrate_mg_l"
RAW_SPLIT = "split"
FORCING_COLUMNS_DROP = frozenset({RAW_SPLIT, "site_no_norm"})


LSTM_FORCING_META_COLS = frozenset(
    {RAW_DATE, RAW_SITE, RAW_TARGET, RAW_SPLIT, "site_no_norm"}
)

LSTM_ATTR_EXCLUDE_COLS = frozenset({"site_no", "tile_drain_percent_nodata"})


LSTM_NORM_DEFAULT = "zscore"
LSTM_NORM_FORCINGS: dict[str, str] = {
    "discharge_cfs": "log1p",
    "discharge_interpolated": "none",
    "pr": "log1p",
    "tmmx": "zscore",
    "tmmn": "zscore",
    "sph": "zscore",
    "srad": "zscore",
    "Corn_sqkm": "zscore",
    "Soybean_sqkm": "zscore",
    "Other_Agriculture_sqkm": "zscore",
    "Forest_sqkm": "zscore",
    "Urban_sqkm": "zscore",
    "Other_Unclassified_sqkm": "zscore",
    "Agriculture_Fertilizer": "zscore",
    "Atmospheric_Oxidized": "zscore",
    "Atmospheric_Reduced": "zscore",
    "Domestic_Fertilizer": "log1p",
    "Human": "log1p",
    "Lvst_Sum": "log1p",
    "Temperature": "zscore",
    "Dissolved_Oxygen": "zscore",
    "pH": "zscore",
    "Specific_Conductance": "log1p",
    "Turbidity": "log1p",

    "regional_donor": "log1p",
    "biophysical_donor": "log1p",
    "signature_donor": "log1p",
    "regional_donor_obs_mask": "none",
    "biophysical_donor_obs_mask": "none",
    "signature_donor_obs_mask": "none",
}

LSTM_NORM_GLOBAL_SCOPE = frozenset(
    {
        "pr",
        "tmmn",
        "tmmx",
        "sph",
        "srad",
        "Corn_sqkm",
        "Soybean_sqkm",
        "Other_Agriculture_sqkm",
        "Forest_sqkm",
        "Urban_sqkm",
        "Other_Unclassified_sqkm",
        "Agriculture_Fertilizer",
        "Atmospheric_Oxidized",
        "Atmospheric_Reduced",
        "Domestic_Fertilizer",
        "Human",
        "Lvst_Sum",
    }
)


ACTIVE_COHORT = "50_compare"

RUN_COHORTS: tuple[str, ...] | None = None

SITE_LIST_50 = DATASETS_DIR / "forcing_BASE_50_V1.parquet"
FORCING_117 = DATASETS_DIR / "forcing_117_V1.parquet"
FORCING_NODONOR_50 = DATASETS_DIR / "forcing_NODONOR_50_V1.parquet"
FORCING_REDUCED_50 = DATASETS_DIR / "forcing_REDUCED_50_V1.parquet"
FORCING_ALL_50 = DATASETS_DIR / "forcing_ALL_50_V1.parquet"

COHORTS: dict[str, dict] = {
    "117_donor": {
        "gap_groups": {
            "R20B14": {"block_days": 14, "n_sites": 117, "expected_testable": None},
            "R20B60": {"block_days": 60, "n_sites": 117, "expected_testable": None},
            "R20B365": {"block_days": 365, "n_sites": 117, "expected_testable": None},
        },
        "scenarios": ("NODONOR", "SIGDONOR", "REGDONOR"),
        "split_master_suffix": "NODONOR",

        "forcing_by_suffix": {
            "NODONOR": FORCING_117,
            "SIGDONOR": FORCING_117,
            "REGDONOR": FORCING_117,
        },
    },
    "50_compare": {
        "gap_groups": {
            "R20B14": {"block_days": 14, "n_sites": 50, "expected_testable": None},
            "R20B60": {"block_days": 60, "n_sites": 50, "expected_testable": None},
            "R20B365": {"block_days": 365, "n_sites": 50, "expected_testable": None},
        },

        "scenarios": (
            "NODONOR",
            "SIGDONOR",
            "REGDONOR",
            "REDUCEDSENSOR",
            "ALLSENSOR",
        ),
        "split_master_suffix": "NODONOR",
        "forcing_by_suffix": {

            "NODONOR": FORCING_NODONOR_50,
            "SIGDONOR": FORCING_NODONOR_50,
            "REGDONOR": FORCING_NODONOR_50,
            "REDUCEDSENSOR": FORCING_REDUCED_50,
            "ALLSENSOR": FORCING_ALL_50,
        },
    },
}

if ACTIVE_COHORT not in COHORTS:
    raise SystemExit(f"Unknown ACTIVE_COHORT={ACTIVE_COHORT!r}; choose from {list(COHORTS)}")

_COHORT = COHORTS[ACTIVE_COHORT]
GAP_GROUPS: dict[str, dict] = _COHORT["gap_groups"]
DONOR_SCENARIOS: tuple[str, ...] = tuple(_COHORT["scenarios"])
SENSOR_SCENARIOS = DONOR_SCENARIOS
SPLIT_MASTER_SUFFIX: str = str(_COHORT["split_master_suffix"])
FORCING_BY_SUFFIX: dict[str, Path] = dict(_COHORT["forcing_by_suffix"])


RUN_CASES: list[str] | None = None


def activate_cohort(cohort_name: str) -> None:
    """Switch module-level cohort controls (GAP_GROUPS, scenarios, forcings)."""
    global ACTIVE_COHORT, _COHORT, GAP_GROUPS, DONOR_SCENARIOS, SENSOR_SCENARIOS
    global SPLIT_MASTER_SUFFIX, FORCING_BY_SUFFIX, CASES
    if cohort_name not in COHORTS:
        raise SystemExit(f"Unknown cohort={cohort_name!r}; choose from {list(COHORTS)}")
    ACTIVE_COHORT = cohort_name
    _COHORT = COHORTS[cohort_name]
    GAP_GROUPS = _COHORT["gap_groups"]
    DONOR_SCENARIOS = tuple(_COHORT["scenarios"])
    SENSOR_SCENARIOS = DONOR_SCENARIOS
    SPLIT_MASTER_SUFFIX = str(_COHORT["split_master_suffix"])
    FORCING_BY_SUFFIX = dict(_COHORT["forcing_by_suffix"])
    CASES = build_case_specs(seed=SPLIT_RANDOM_SEED)


def ensure_nodonor_50_forcing(
    *,
    src_117: Path = FORCING_117,
    site_list_parquet: Path = SITE_LIST_50,
    out_path: Path = FORCING_NODONOR_50,
) -> Path:
    """Write forcing_NODONOR_50_V1.parquet = forcing_117 restricted to soft-sensor 50 sites."""
    if out_path.is_file():
        return out_path
    if not src_117.is_file():
        raise FileNotFoundError(f"Missing {src_117}")
    if not site_list_parquet.is_file():
        raise FileNotFoundError(f"Missing site list {site_list_parquet}")
    sites = (
        pd.read_parquet(site_list_parquet, columns=["site_no"])["site_no"]
        .astype(str)
        .str.strip()
        .unique()
    )
    df = pd.read_parquet(src_117)
    df["site_no"] = df["site_no"].astype(str).str.strip()
    out = df.loc[df["site_no"].isin(set(sites))].sort_values(
        ["site_no", "date"], kind="mergesort"
    ).reset_index(drop=True)
    n = int(out["site_no"].nunique())
    if n != 50:
        raise ValueError(f"Expected 50 sites in {out_path.name}, got {n}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}  sites={n}  rows={len(out):,}", flush=True)
    return out_path


def _forcing_path(suffix: str, n_sites: int) -> Path:
    """Resolve forcing parquet for a scenario suffix in the active cohort."""
    _ = n_sites
    if suffix not in FORCING_BY_SUFFIX:
        raise KeyError(
            f"No forcing mapped for suffix {suffix!r} in cohort {ACTIVE_COHORT!r}. "
            f"Known: {list(FORCING_BY_SUFFIX)}"
        )
    path = FORCING_BY_SUFFIX[suffix]
    if path == FORCING_NODONOR_50:
        ensure_nodonor_50_forcing()
    return path


def build_case_specs(
    *,
    seed: int = SPLIT_RANDOM_SEED,
    gap_groups: dict[str, dict] | None = None,
    donor_scenarios: tuple[str, ...] | None = None,
    sensor_scenarios: tuple[str, ...] | None = None,
    split_master_suffix: str | None = None,
) -> dict[str, dict]:
    """
    Build the case control dictionary for the active cohort.

    Case name: {RUN_EXPERIMENT}_{N_SITES}_Seed{seed}_{RUN_SUFFIX}
    e.g. R20B14_50_Seed21_ALLSENSOR

    Split is assigned once per gap group on the master suffix (NODONOR), then merged
    into the other scenarios so gap dates match.
    """
    gaps = gap_groups if gap_groups is not None else GAP_GROUPS
    donors = (
        donor_scenarios
        if donor_scenarios is not None
        else (sensor_scenarios if sensor_scenarios is not None else DONOR_SCENARIOS)
    )
    master_sfx = (
        split_master_suffix if split_master_suffix is not None else SPLIT_MASTER_SUFFIX
    )
    cases: dict[str, dict] = {}
    for run_experiment, gap in gaps.items():
        block_days = int(gap["block_days"])
        n_sites = int(gap["n_sites"])
        for suffix in donors:
            name = f"{run_experiment}_{n_sites}_Seed{int(seed)}_{suffix}"
            cases[name] = {
                "case_name": name,
                "run_experiment": run_experiment,
                "n_sites": n_sites,
                "run_suffix": suffix,
                "block_days": block_days,
                "dnp_testable_min_obs": block_days,
                "expected_testable": gap.get("expected_testable"),
                "split_random_seed": int(seed),
                "split_mode": SPLIT_MODE,
                "test_fraction": TEST_FRACTION,
                "forcing_input": _forcing_path(suffix, n_sites),
                "attribute_input": ATTRIBUTE_INPUT,
                "is_split_master": suffix == master_sfx,
                "gap_group_key": f"{run_experiment}_{n_sites}_Seed{int(seed)}",
                "cohort": ACTIVE_COHORT,
            }
    return cases


CASES: dict[str, dict] = build_case_specs()


BLOCK_DAYS = int(GAP_GROUPS["R20B60"]["block_days"])
DNP_TESTABLE_MIN_OBS = BLOCK_DAYS
N_SITES = int(GAP_GROUPS["R20B60"]["n_sites"])
RUN_EXPERIMENT = "R20B60"
RUN_SUFFIX = SPLIT_MASTER_SUFFIX
FORCING_INPUT = _forcing_path(RUN_SUFFIX, N_SITES)


def run_tag_for_case(case: dict) -> str:
    if case.get("case_name"):
        return str(case["case_name"])
    return (
        f"{case['run_experiment']}_{case['n_sites']}_"
        f"Seed{case['split_random_seed']}_{case['run_suffix']}"
    )


def output_dir_for_case(case: dict) -> Path:
    return LSTM_INPUTS_DIR / run_tag_for_case(case)


def select_cases(
    run_cases: list[str] | None = RUN_CASES,
    *,
    cases: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Filter CASES by explicit names, gap keys (e.g. R20B60), or donor suffixes."""
    table = cases if cases is not None else CASES
    if not run_cases:
        return dict(table)
    selected: dict[str, dict] = {}
    for token in run_cases:
        t = str(token).strip()
        if t in table:
            selected[t] = table[t]
            continue
        matched = False
        for name, case in table.items():
            if (
                t == case["run_experiment"]
                or t == case["gap_group_key"]
                or t == case["run_suffix"]
            ):
                selected[name] = case
                matched = True
        if not matched:
            raise SystemExit(
                f"Unknown RUN_CASES entry {t!r}. "
                f"Use a case name, gap key ({', '.join(GAP_GROUPS)}), or "
                f"donor suffix ({', '.join(DONOR_SCENARIOS)})."
            )
    return selected


def _discover_forcing_input_columns(
    df: pd.DataFrame,
    *,
    date_col: str,
    site_col: str,
    target_col: str,
) -> List[str]:
    fixed = {date_col, site_col, target_col}
    missing = [c for c in (date_col, site_col, target_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Forcing data must include columns {date_col!r}, {site_col!r}, {target_col!r}. "
            f"Missing: {missing}"
        )
    return [c for c in df.columns if c not in fixed and c not in FORCING_COLUMNS_DROP]


def _rectangularize_and_fill_inputs(
    df: pd.DataFrame,
    *,
    site_col: str,
    date_col: str,
    input_cols: List[str],
    target_col: str,
    global_dates: pd.DatetimeIndex,
    sites: List[str],
) -> pd.DataFrame:
    full_index = pd.MultiIndex.from_product([sites, global_dates], names=[site_col, date_col])
    df2 = df.set_index([site_col, date_col]).reindex(full_index).reset_index()

    for c in input_cols:
        if c not in df2.columns:
            df2[c] = np.nan
        s = df2[c]
        if pd.api.types.is_numeric_dtype(s):
            df2[c] = s.fillna(0.0)
        elif pd.api.types.is_bool_dtype(s):
            df2[c] = s.fillna(False)
        elif pd.api.types.is_datetime64_any_dtype(s):
            pass
        else:
            filled = df2.groupby(site_col, sort=False)[c].transform(lambda x: x.ffill().bfill())
            df2[c] = filled.astype("string")

    if target_col not in df2.columns:
        df2[target_col] = np.nan

    return df2.sort_values([site_col, date_col], kind="stable").reset_index(drop=True)


def _mask_train_and_gaptruth(
    df: pd.DataFrame,
    *,
    date_col: str,
    site_col: str,
    nitrate_col: str,
    split_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train: nitrate NaN on test split rows. Gap-truth: nitrate NaN on train rows."""
    required = {date_col, site_col, nitrate_col, split_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for split masking: {sorted(missing)}")

    d = df.copy()
    if not np.issubdtype(d[date_col].dtype, np.datetime64):
        d[date_col] = pd.to_datetime(d[date_col])
    d[date_col] = d[date_col].dt.normalize()
    d = d.sort_values([site_col, date_col], kind="stable")
    d = d.drop_duplicates(subset=[site_col, date_col], keep="first")

    split_norm = d[split_col].astype("string").str.lower().str.strip()
    is_test = split_norm.eq("test").to_numpy()
    unknown = ~(split_norm.eq("train") | split_norm.eq("test") | split_norm.isna())
    if bool(unknown.any()):
        bad = d.loc[unknown, split_col].dropna().unique()[:10]
        raise ValueError(f"Unexpected split labels in {split_col!r}: {bad!r}")

    df_train = d.copy()
    df_truth = d.copy()
    df_train.loc[is_test, nitrate_col] = np.nan
    df_truth.loc[~is_test, nitrate_col] = np.nan
    return df_train, df_truth


def _floor_target_at_zero(df: pd.DataFrame, col: str) -> None:
    """Set negative target values to 0; NaNs unchanged (coerce non-numeric to NaN)."""
    if col not in df.columns:
        return
    df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)


def _n_test_from_n_obs(
    n_obs: int,
    *,
    min_obs_to_split: int,
    min_train_obs_after_split: int,
) -> int:
    """Replicate the 'how many test points' logic from the provided snippet."""
    if n_obs < int(min_obs_to_split):
        return 0
    if 30 <= n_obs < 100:
        n_test = 10
    elif 100 <= n_obs < 300:
        n_test = int(round(0.15 * n_obs))
    else:
        n_test = int(round(0.20 * n_obs))

    n_test = min(int(n_test), n_obs - int(min_train_obs_after_split))
    return int(max(n_test, 0))


def _n_test_from_fraction(
    n_obs: int,
    *,
    fraction: float,
    min_obs_to_split: int,
    min_train_obs_after_split: int,
) -> int:
    if n_obs < int(min_obs_to_split):
        return 0
    n_test = int(round(float(fraction) * float(n_obs)))
    n_test = min(int(n_test), n_obs - int(min_train_obs_after_split))
    return int(max(n_test, 0))


def _select_random_points(
    obs_idx: np.ndarray,
    *,
    n_test: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_test <= 0 or obs_idx.size == 0:
        return np.asarray([], dtype=int)
    n_test = int(min(n_test, obs_idx.size))
    return rng.choice(obs_idx, size=n_test, replace=False).astype(int)


def _site_nonoverlap_block_count_and_latest(
    g: pd.DataFrame,
    *,
    obs_idx: np.ndarray,
    date_col: str,
    block_days: int,
) -> tuple[int, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Count non-overlapping fully-observed blocks of length `block_days` and return the latest block.

    Mirrors your 365-day script: within each continuous observed run, blocks are packed
    back-to-back from the run start. Total blocks = sum(floor(run_length / block_days)).
    Latest block is the last packed block among all runs.
    """
    if obs_idx.size == 0:
        return 0, None, None
    if date_col not in g.columns:
        raise ValueError(f"block selection requires {date_col!r} column")

    dt = pd.to_datetime(g.loc[obs_idx, date_col], errors="coerce").dt.normalize()
    dt = dt.dropna()
    if dt.empty:
        return 0, None, None

    obs_dates = pd.DatetimeIndex(dt).sort_values().unique()
    if len(obs_dates) == 0:
        return 0, None, None

    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    run_start = pd.Timestamp(obs_dates[0]).normalize()
    prev = run_start
    run_len = 1
    for d0 in obs_dates[1:]:
        d0 = pd.Timestamp(d0).normalize()
        if d0 - prev == pd.Timedelta(days=1):
            run_len += 1
        else:
            runs.append((run_start, prev, int(run_len)))
            run_start = d0
            run_len = 1
        prev = d0
    runs.append((run_start, prev, int(run_len)))

    total_blocks = 0
    latest_start: Optional[pd.Timestamp] = None
    latest_end: Optional[pd.Timestamp] = None

    for rs, re, L in runs:
        n_blocks = int(L // int(block_days))
        if n_blocks <= 0:
            continue
        total_blocks += n_blocks
        k_last = n_blocks - 1
        bs = pd.Timestamp(rs).normalize() + pd.Timedelta(days=int(k_last) * int(block_days))
        be = bs + pd.Timedelta(days=int(block_days) - 1)
        if latest_end is None or be > latest_end or (be == latest_end and bs > (latest_start or bs)):
            latest_start, latest_end = bs, be

    return int(total_blocks), latest_start, latest_end


def _indices_for_date_window(
    g: pd.DataFrame,
    *,
    obs_idx: np.ndarray,
    date_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    """Return one observed row-index per day in [start,end]."""
    g_obs = g.loc[obs_idx, [date_col]].copy()
    g_obs = g_obs.dropna(subset=[date_col])
    if g_obs.empty:
        return np.asarray([], dtype=int)

    dates_obs = pd.to_datetime(g_obs[date_col]).dt.normalize()
    g_obs2 = (
        pd.DataFrame({"idx": g_obs.index.to_numpy(), "date": dates_obs.to_numpy()})
        .sort_values("idx")
        .drop_duplicates(subset=["date"], keep="first")
    )

    win_dates = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    idxs: list[int] = []
    for dt0 in win_dates:
        m = (g_obs2["date"] == np.datetime64(pd.Timestamp(dt0).normalize())).to_numpy()
        if not bool(np.any(m)):
            return np.asarray([], dtype=int)
        idxs.append(int(g_obs2.loc[m, "idx"].iloc[0]))
    return np.asarray(idxs, dtype=int)


def _iter_temporally_stratified_starts(
    candidates: pd.DatetimeIndex,
    *,
    rng: np.random.Generator,
    n_bins: int,
) -> list[pd.Timestamp]:
    """Return candidate starts interleaved across time bins (random within bin)."""
    if len(candidates) == 0:
        return []
    n_bins = int(max(1, n_bins))
    if n_bins <= 1 or len(candidates) <= 2:
        arr = candidates.to_numpy()
        return [pd.Timestamp(arr[i]).normalize() for i in rng.permutation(len(arr))]

    n_bins = int(max(2, min(n_bins, len(candidates))))
    chunks = np.array_split(candidates.to_numpy(), n_bins)
    bins: list[list[pd.Timestamp]] = []
    for ch in chunks:
        ch2 = [pd.Timestamp(x).normalize() for x in ch.tolist()]
        if len(ch2) > 1:
            ch2 = [ch2[i] for i in rng.permutation(len(ch2))]
        bins.append(ch2)

    out: list[pd.Timestamp] = []
    remaining = True
    while remaining:
        remaining = False
        for b in rng.permutation(len(bins)):
            if bins[b]:
                out.append(bins[b].pop())
                remaining = True
    return out


def _block_window_allowed_months(
    start: pd.Timestamp,
    block_days: int,
    allowed_months: Optional[set[int] | frozenset[int]],
) -> bool:
    if not allowed_months:
        return True
    s0 = pd.Timestamp(start).normalize()
    for k in range(int(block_days)):
        if (s0 + pd.Timedelta(days=k)).month not in allowed_months:
            return False
    return True


def _select_fully_observed_blocks(
    g: pd.DataFrame,
    *,
    site: str,
    obs_idx: np.ndarray,
    date_col: str,
    block_days: int,
    n_test_target: int,
    rng: np.random.Generator,
    min_train_obs_after_split: int,
    max_block_tries: int,
    temporal_stratify: bool = TEMPORAL_STRATIFY_BLOCKS,
    temporal_n_bins: int = TEMPORAL_STRATIFY_N_BINS,
    block_allowed_months: Optional[set[int] | frozenset[int]] = None,
) -> tuple[np.ndarray, int, int]:
    """
    Return selected row indices (subset of obs_idx) corresponding to non-overlapping
    fully-observed consecutive-day windows of length block_days.
    """
    if date_col not in g.columns:
        raise ValueError(f"block selection requires {date_col!r} column")
    if n_test_target <= 0 or obs_idx.size == 0:
        return np.asarray([], dtype=int), 0, 0

    g_obs = g.loc[obs_idx, [date_col]].copy()
    g_obs = g_obs.dropna(subset=[date_col])
    if g_obs.empty:
        return np.asarray([], dtype=int), 0, 0

    dates_obs = pd.to_datetime(g_obs[date_col]).dt.normalize()
    g_obs2 = (
        pd.DataFrame({"idx": g_obs.index.to_numpy(), "date": dates_obs.to_numpy()})
        .sort_values("idx")
        .drop_duplicates(subset=["date"], keep="first")
    )
    obs_dates = pd.DatetimeIndex(g_obs2["date"]).sort_values()
    if len(obs_dates) == 0:
        return np.asarray([], dtype=int), 0, 0

    obs_set = set(obs_dates.to_pydatetime())
    candidates = []
    for s0 in obs_dates:
        ok = True
        for k in range(int(block_days)):
            if (s0 + pd.Timedelta(days=k)).to_pydatetime() not in obs_set:
                ok = False
                break
        if ok and _block_window_allowed_months(s0, block_days, block_allowed_months):
            candidates.append(pd.Timestamp(s0).normalize())
    candidates = pd.DatetimeIndex(candidates).sort_values()


    n_full_blocks_avail = 0
    if len(candidates) > 0:
        used_tmp = set()
        for s0 in candidates:
            win_dates = [pd.Timestamp(s0).normalize() + pd.Timedelta(days=k) for k in range(int(block_days))]
            if any(dt.to_pydatetime() in used_tmp for dt in win_dates):
                continue
            for dt in win_dates:
                used_tmp.add(dt.to_pydatetime())
            n_full_blocks_avail += 1


    if int(block_days) == 365:
        enough_train_remaining = (
            len(obs_idx) - int(block_days) >= int(min_train_obs_after_split)
        )
        blocks_to_take = (
            1
            if n_full_blocks_avail >= 1 and enough_train_remaining
            else 0
        )
    else:
        blocks_to_take = int(n_test_target // int(block_days))
        blocks_to_take = int(min(blocks_to_take, n_full_blocks_avail))
    if blocks_to_take <= 0:
        return np.asarray([], dtype=int), int(n_full_blocks_avail), 0

    chosen_idx: list[int] = []
    used_dates: set = set()
    n_blocks_selected = 0

    if bool(temporal_stratify):
        ordered_starts = _iter_temporally_stratified_starts(candidates, rng=rng, n_bins=int(temporal_n_bins))
    else:
        cand_arr = candidates.to_numpy()
        ordered_starts = [pd.Timestamp(cand_arr[i]).normalize() for i in rng.permutation(len(cand_arr))]

    n_try = int(min(max_block_tries, len(ordered_starts)))
    for s0 in ordered_starts[:n_try]:
        if n_blocks_selected >= blocks_to_take:
            break
        win_dates = [s0 + pd.Timedelta(days=k) for k in range(int(block_days))]
        if any(dt.to_pydatetime() in used_dates for dt in win_dates):
            continue
        idxs = []
        for dt in win_dates:
            m = (g_obs2["date"] == np.datetime64(dt)).to_numpy()
            if not bool(np.any(m)):
                idxs = []
                break
            idxs.append(int(g_obs2.loc[m, "idx"].iloc[0]))
        if not idxs:
            continue

        n_in = len(idxs)
        n_obs = int(len(obs_idx))
        if (n_obs - (len(chosen_idx) + n_in)) < int(min_train_obs_after_split):
            continue
        chosen_idx.extend(idxs)
        for dt in win_dates:
            used_dates.add(dt.to_pydatetime())
        n_blocks_selected += 1

    return np.asarray(chosen_idx, dtype=int), int(n_full_blocks_avail), int(n_blocks_selected)


def _select_regime_blocks_stratified(
    g: pd.DataFrame,
    *,
    obs_idx: np.ndarray,
    date_col: str,
    target_col: str,
    block_days: int,
    rng: np.random.Generator,
    p_lo: float,
    p_hi: float,
    n_peak_target: int,
    n_trough_target: int,
    n_mid_target: int,
    max_tries: int,
) -> tuple[np.ndarray, dict]:
    """
    Test C strategy: select non-overlapping windows of length `block_days` by regime.

    Regimes (site-specific thresholds on observed nitrate):
      - peak: window has >=1 observed day with y >= p_hi
      - trough: window has >=1 observed day with y <= p_lo
      - mid: window has >=1 observed day with p_lo < y < p_hi

    Counting rule:
      - windows must be FULLY observed (non-NaN nitrate on every day in the window)
      - thus each chosen window contributes exactly `block_days` observations
      - no point top-up; we stop early if we can't hit targets exactly (no overshoot).
    """
    if obs_idx.size == 0:
        return np.asarray([], dtype=int), {"n_blocks_peak": 0, "n_blocks_trough": 0, "n_blocks_mid": 0}
    if date_col not in g.columns:
        raise ValueError(f"regime block selection requires {date_col!r} column")

    y = pd.to_numeric(g.loc[obs_idx, target_col], errors="coerce")
    dt = pd.to_datetime(g.loc[obs_idx, date_col], errors="coerce").dt.normalize()
    ok = y.notna() & dt.notna()
    if not bool(ok.any()):
        return np.asarray([], dtype=int), {"n_blocks_peak": 0, "n_blocks_trough": 0, "n_blocks_mid": 0}

    obs_tbl = pd.DataFrame({"idx": obs_idx.astype(int), "date": dt.to_numpy(), "y": y.to_numpy()}).loc[
        ok.to_numpy()
    ].copy()
    obs_tbl = obs_tbl.sort_values("idx").drop_duplicates(subset=["date"], keep="first")
    obs_tbl = obs_tbl.sort_values("date", kind="stable").reset_index(drop=True)
    if obs_tbl.empty:
        return np.asarray([], dtype=int), {"n_blocks_peak": 0, "n_blocks_trough": 0, "n_blocks_mid": 0}

    lo = float(obs_tbl["y"].quantile(float(p_lo)))
    hi = float(obs_tbl["y"].quantile(float(p_hi)))

    obs_dates = pd.DatetimeIndex(obs_tbl["date"]).sort_values()
    obs_set = set(obs_dates.to_pydatetime())
    candidates = []
    for s0 in obs_dates:
        ok_win = True
        s0n = pd.Timestamp(s0).normalize()
        for k in range(int(block_days)):
            if (s0n + pd.Timedelta(days=k)).to_pydatetime() not in obs_set:
                ok_win = False
                break
        if ok_win:
            candidates.append(s0n)
    candidates = pd.DatetimeIndex(candidates).sort_values()
    if len(candidates) == 0:
        return np.asarray([], dtype=int), {"n_blocks_peak": 0, "n_blocks_trough": 0, "n_blocks_mid": 0}

    if bool(TEMPORAL_STRATIFY_BLOCKS):
        ordered_starts = _iter_temporally_stratified_starts(candidates, rng=rng, n_bins=int(TEMPORAL_STRATIFY_N_BINS))
    else:
        cand_arr = candidates.to_numpy()
        ordered_starts = [pd.Timestamp(cand_arr[i]).normalize() for i in rng.permutation(len(cand_arr))]
    ordered_starts = ordered_starts[: int(min(len(ordered_starts), max_tries))]

    used_dates: set = set()
    chosen: list[int] = []
    chosen_set: set = set()

    def window_obs(s0: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:

        win_dates = [pd.Timestamp(s0).normalize() + pd.Timedelta(days=k) for k in range(int(block_days))]
        idxs: list[int] = []
        ys: list[float] = []
        for dt0 in win_dates:
            m = (obs_tbl["date"] == np.datetime64(dt0)).to_numpy()
            if not bool(np.any(m)):
                return np.asarray([], dtype=int), np.asarray([], dtype=float)
            idxs.append(int(obs_tbl.loc[m, "idx"].iloc[0]))
            ys.append(float(obs_tbl.loc[m, "y"].iloc[0]))
        if chosen_set:
            keep = [i not in chosen_set for i in idxs]
            idxs = [i for i, k in zip(idxs, keep) if k]
            ys = [v for v, k in zip(ys, keep) if k]
        return np.asarray(idxs, dtype=int), np.asarray(ys, dtype=float)

    def eligible_peak(ys: np.ndarray) -> bool:
        return bool(np.any(np.isfinite(ys) & (ys >= hi)))

    def eligible_trough(ys: np.ndarray) -> bool:
        return bool(np.any(np.isfinite(ys) & (ys <= lo)))

    def eligible_mid(ys: np.ndarray) -> bool:
        return bool(np.any(np.isfinite(ys) & (ys > lo) & (ys < hi)))


    n_blocks_peak = 0
    n_blocks_trough = 0
    n_blocks_mid = 0
    got_peak = 0
    got_trough = 0
    got_mid = 0


    def run_select(want_obs: int, predicate) -> tuple[int, int]:
        got = 0
        n_blocks = 0
        for s0 in ordered_starts:
            if got >= int(want_obs):
                break
            win_dates = [s0 + pd.Timedelta(days=k) for k in range(int(block_days))]
            if any(dt0.to_pydatetime() in used_dates for dt0 in win_dates):
                continue
            idxs, ys = window_obs(s0)
            n_in = int(idxs.size)
            if n_in != int(block_days):
                continue
            if got + n_in > int(want_obs):
                continue
            if not predicate(ys):
                continue
            for dt0 in win_dates:
                used_dates.add(dt0.to_pydatetime())
            n_blocks += 1
            got += n_in
            chosen_set.update(idxs.tolist())
            chosen.extend(idxs.astype(int).tolist())
        return int(got), int(n_blocks)

    got_peak, n_blocks_peak = run_select(int(n_peak_target), eligible_peak)
    got_trough, n_blocks_trough = run_select(int(n_trough_target), eligible_trough)
    got_mid, n_blocks_mid = run_select(int(n_mid_target), eligible_mid)

    meta = {
        "p10": float(lo),
        "p90": float(hi),
        "got_peak_obs": int(got_peak),
        "got_trough_obs": int(got_trough),
        "got_mid_obs": int(got_mid),
        "n_blocks_peak": int(n_blocks_peak),
        "n_blocks_trough": int(n_blocks_trough),
        "n_blocks_mid": int(n_blocks_mid),
    }
    return np.asarray(chosen, dtype=int), meta


def _select_holdout_calendar_years(
    g: pd.DataFrame,
    *,
    obs_idx: np.ndarray,
    date_col: str,
    holdout_years: int,
    position: str,
) -> np.ndarray:
    """
    Label observed rows in the first/last ``holdout_years`` calendar years of a site's series as test.

    - start: date < min(obs_date) + holdout_years
    - end:   date >= max(obs_date) - holdout_years + 1 day  (equivalently date > max - holdout_years)
    """
    if obs_idx.size == 0 or int(holdout_years) <= 0:
        return np.asarray([], dtype=int)
    if date_col not in g.columns:
        raise ValueError(f"year holdout requires {date_col!r} column")

    dt = pd.to_datetime(g.loc[obs_idx, date_col], errors="coerce").dt.normalize()
    ok = dt.notna()
    if not bool(ok.any()):
        return np.asarray([], dtype=int)

    obs_tbl = pd.DataFrame({"idx": obs_idx.astype(int), "date": dt.to_numpy()}).loc[ok.to_numpy()].copy()
    obs_tbl = obs_tbl.sort_values("idx").drop_duplicates(subset=["date"], keep="first")
    obs_tbl = obs_tbl.sort_values("date", kind="stable").reset_index(drop=True)
    if obs_tbl.empty:
        return np.asarray([], dtype=int)

    min_d = pd.Timestamp(obs_tbl["date"].min()).normalize()
    max_d = pd.Timestamp(obs_tbl["date"].max()).normalize()
    pos = str(position).strip().lower()
    if pos == "start":
        cutoff = min_d + pd.DateOffset(years=int(holdout_years))
        mask = obs_tbl["date"] < np.datetime64(cutoff)
    elif pos == "end":
        cutoff = max_d - pd.DateOffset(years=int(holdout_years))
        mask = obs_tbl["date"] > np.datetime64(cutoff)
    else:
        raise ValueError(f"holdout position must be 'start' or 'end', got {position!r}")

    return obs_tbl.loc[mask.to_numpy(), "idx"].to_numpy(dtype=int)


def _select_grab_monthly_train(
    g: pd.DataFrame,
    *,
    obs_idx: np.ndarray,
    date_col: str,
    pick: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """One training grab per calendar month (first or random observed day in month)."""
    if obs_idx.size == 0:
        return np.asarray([], dtype=int)
    if date_col not in g.columns:
        raise ValueError(f"grab monthly selection requires {date_col!r} column")

    dt = pd.to_datetime(g.loc[obs_idx, date_col], errors="coerce").dt.normalize()
    ok = dt.notna()
    if not bool(ok.any()):
        return np.asarray([], dtype=int)

    obs_tbl = pd.DataFrame({"idx": obs_idx.astype(int), "date": dt.to_numpy()}).loc[ok.to_numpy()].copy()
    obs_tbl = obs_tbl.sort_values("idx").drop_duplicates(subset=["date"], keep="first")
    obs_tbl = obs_tbl.sort_values("date", kind="stable").reset_index(drop=True)
    obs_tbl["ym"] = pd.to_datetime(obs_tbl["date"]).dt.to_period("M")

    pick_mode = str(pick).strip().lower()
    chosen: list[int] = []
    for _, grp in obs_tbl.groupby("ym", sort=True):
        if grp.empty:
            continue
        if pick_mode == "random":
            row = grp.iloc[int(rng.integers(0, len(grp)))]
            chosen.append(int(row["idx"]))
        else:
            chosen.append(int(grp.iloc[0]["idx"]))
    return np.asarray(chosen, dtype=int)


def _select_grab_biweekly_train(
    g: pd.DataFrame,
    *,
    obs_idx: np.ndarray,
    date_col: str,
    interval_days: int,
) -> np.ndarray:
    """
    One training grab per ``interval_days``-day anchor from the site's first observed date.

    For each anchor, take the earliest observed nitrate on or after that anchor (no reuse).
    """
    if obs_idx.size == 0:
        return np.asarray([], dtype=int)
    if date_col not in g.columns:
        raise ValueError(f"grab biweekly selection requires {date_col!r} column")
    interval_days = int(max(1, interval_days))

    dt = pd.to_datetime(g.loc[obs_idx, date_col], errors="coerce").dt.normalize()
    ok = dt.notna()
    if not bool(ok.any()):
        return np.asarray([], dtype=int)

    obs_tbl = pd.DataFrame({"idx": obs_idx.astype(int), "date": dt.to_numpy()}).loc[ok.to_numpy()].copy()
    obs_tbl = obs_tbl.sort_values("idx").drop_duplicates(subset=["date"], keep="first")
    obs_tbl = obs_tbl.sort_values("date", kind="stable").reset_index(drop=True)
    if obs_tbl.empty:
        return np.asarray([], dtype=int)

    min_d = pd.Timestamp(obs_tbl["date"].min()).normalize()
    max_d = pd.Timestamp(obs_tbl["date"].max()).normalize()
    anchors = pd.date_range(min_d, max_d, freq=f"{interval_days}D")

    chosen: list[int] = []
    used_dates: set = set()
    obs_dates = obs_tbl["date"].to_numpy()
    obs_idxs = obs_tbl["idx"].to_numpy(dtype=int)

    for anchor in anchors:
        a = np.datetime64(pd.Timestamp(anchor).normalize())
        mask = (obs_dates >= a) & (~np.isin(obs_dates, list(used_dates)))
        if not bool(np.any(mask)):
            continue
        j = int(np.where(mask)[0][0])
        chosen.append(int(obs_idxs[j]))
        used_dates.add(obs_dates[j])

    return np.asarray(chosen, dtype=int)


def assign_split(
    df: pd.DataFrame,
    *,
    site_col: str,
    date_col: str,
    target_col: str,
    mode: str = SPLIT_MODE,
    seed: int = SPLIT_RANDOM_SEED,
    min_obs_to_split: int = MIN_OBS_TO_SPLIT,
    min_train_obs_after_split: int = MIN_TRAIN_OBS_AFTER_SPLIT,
    block_days: int = BLOCK_DAYS,
    max_block_tries: int = MAX_BLOCK_TRIES,
    test_fraction: float = TEST_FRACTION,
    test_c_q_low: float = TEST_C_Q_LOW,
    test_c_q_high: float = TEST_C_Q_HIGH,
    test_c_q_mid_lo: float = TEST_C_Q_MID_LO,
    test_c_q_mid_hi: float = TEST_C_Q_MID_HI,
    test_c_frac_low: float = TEST_C_FRAC_LOW,
    test_c_frac_high: float = TEST_C_FRAC_HIGH,
    test_c_frac_mid: float = TEST_C_FRAC_MID,
    holdout_years: int = HOLDOUT_YEARS,
    grab_biweekly_days: int = GRAB_BIWEEKLY_DAYS,
    grab_monthly_pick: str = GRAB_MONTHLY_PICK,
    block_allowed_months: Optional[set[int] | frozenset[int]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a per-site split on *observed* target rows only.

    See the module docstring at the top of this file for the full list of ``mode`` values
    and what each testing scheme does.

    Returns:
      df_out: dataframe with ``split`` column (``train`` / ``test``)
      site_summary: one row per site with counts and ``testable``
    """
    d = df.copy()
    if date_col in d.columns and (not np.issubdtype(d[date_col].dtype, np.datetime64)):
        try:
            d[date_col] = pd.to_datetime(d[date_col])
        except Exception:
            pass
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col]).dt.tz_localize(None).dt.normalize()

    d["_target_obs"] = d[target_col].notna() if target_col in d.columns else False
    d[RAW_SPLIT] = "train"

    rng = np.random.default_rng(int(seed))
    rows = []
    mode = str(mode).strip().lower()
    allowed = {
        "random_points",
        "contiguous_block",
        "test_a_random_points_20pct",
        "test_b_random_blocks_20pct",
        "test_c_peak_trough_20pct",
        "site_latest_block_test",
        "test_d_holdout_start_years",
        "test_d_holdout_end_years",
        "test_e_grab_monthly_train",
        "test_e_grab_biweekly_train",
    }
    if mode not in allowed:
        raise ValueError(f"Unknown split mode {mode!r} (expected one of {sorted(allowed)})")

    for site, g in d.groupby(site_col, sort=False):
        obs_idx = g.index[g["_target_obs"]].to_numpy()
        n_obs = int(len(obs_idx))


        eligible_site = False
        test_block_start = pd.NaT
        test_block_end = pd.NaT

        n_full_blocks_avail = 0
        n_blocks_selected = 0
        holdout_position = ""
        n_grab_train = 0

        scheme_d_e = mode in {
            "test_d_holdout_start_years",
            "test_d_holdout_end_years",
            "test_e_grab_monthly_train",
            "test_e_grab_biweekly_train",
        }
        if scheme_d_e:
            if n_obs < int(min_obs_to_split):
                rows.append(
                    {
                        site_col: site,
                        "n_obs": n_obs,
                        "n_test": 0,
                        "n_train_obs": n_obs,
                        "testable": False,
                        "mode": mode,
                        "n_full_blocks_avail": 0,
                        "n_blocks_selected": 0,
                        "holdout_years": int(holdout_years) if mode.startswith("test_d_") else 0,
                        "holdout_position": holdout_position,
                        "n_grab_train": 0,
                        "excluded": False,
                        "excluded_reason": "insufficient_obs",
                    }
                )
                continue
            desired_n_test = -1
        elif mode in {"test_a_random_points_20pct", "test_b_random_blocks_20pct", "test_c_peak_trough_20pct"}:
            desired_n_test = _n_test_from_fraction(
                n_obs,
                fraction=float(test_fraction),
                min_obs_to_split=min_obs_to_split,
                min_train_obs_after_split=min_train_obs_after_split,
            )
        elif mode == "site_latest_block_test":
            desired_n_test = int(block_days)
        else:
            desired_n_test = _n_test_from_n_obs(
                n_obs,
                min_obs_to_split=min_obs_to_split,
                min_train_obs_after_split=min_train_obs_after_split,
            )
        if not scheme_d_e and desired_n_test <= 0:
            rows.append(
                {
                    site_col: site,
                    "n_obs": n_obs,
                    "n_test": 0,
                    "n_train_obs": n_obs,
                    "testable": False,
                    "mode": mode,
                    "n_full_blocks_avail": 0,
                    "n_blocks_selected": 0,
                    "holdout_years": 0,
                    "holdout_position": "",
                    "n_grab_train": 0,
                    "excluded": False,
                    "excluded_reason": "",
                }
            )
            continue

        n_test: int = 0
        test_idx: Optional[np.ndarray] = None
        excluded_reason = ""

        if mode in {"random_points", "test_a_random_points_20pct"}:
            test_idx = _select_random_points(obs_idx, n_test=int(desired_n_test), rng=rng)
            n_test = int(test_idx.size)
            if n_test > 0:
                d.loc[test_idx, RAW_SPLIT] = "test"

        elif mode == "site_latest_block_test":

            n_blocks_avail, bs, be = _site_nonoverlap_block_count_and_latest(
                g,
                obs_idx=obs_idx,
                date_col=date_col,
                block_days=int(block_days),
            )
            n_full_blocks_avail = int(n_blocks_avail)
            n_blocks_selected = 0
            chosen_idx = np.asarray([], dtype=int)
            if n_full_blocks_avail >= 2 and bs is not None and be is not None:
                eligible_site = True
                test_block_start = pd.Timestamp(bs).normalize()
                test_block_end = pd.Timestamp(be).normalize()
                chosen_idx = _indices_for_date_window(
                    g,
                    obs_idx=obs_idx,
                    date_col=date_col,
                    start=bs,
                    end=be,
                )
                if chosen_idx.size == int(block_days):
                    d.loc[chosen_idx, RAW_SPLIT] = "test"
                    n_blocks_selected = 1
            test_idx = chosen_idx.astype(int)
            n_test = int(test_idx.size)

        elif mode in {"contiguous_block", "test_b_random_blocks_20pct"}:

            chosen_idx, n_full_blocks_avail, n_blocks_selected = _select_fully_observed_blocks(
                g,
                site=str(site),
                obs_idx=obs_idx,
                date_col=date_col,
                block_days=int(block_days),
                n_test_target=int(desired_n_test),
                rng=rng,
                min_train_obs_after_split=int(min_train_obs_after_split),
                max_block_tries=int(max_block_tries),
                block_allowed_months=block_allowed_months,
            )


            if mode == "contiguous_block":
                desired_blocks = int(desired_n_test // int(block_days))
                excluded = False
                excluded_reason = ""
                if n_full_blocks_avail < int(MIN_FULL_BLOCKS_FOR_TEST) or desired_blocks <= 0:
                    if str(CONTIGUOUS_INSUFFICIENT_POLICY).strip().lower() == "exclude":
                        excluded = True
                        excluded_reason = f"insufficient_full_blocks(<{MIN_FULL_BLOCKS_FOR_TEST})"
                    rows.append(
                        {
                            site_col: site,
                            "n_obs": n_obs,
                            "n_test": 0,
                            "n_train_obs": n_obs,
                            "testable": False,
                            "mode": mode,
                            "n_full_blocks_avail": int(n_full_blocks_avail),
                            "n_blocks_selected": 0,
                            "excluded": bool(excluded),
                            "excluded_reason": excluded_reason,
                        }
                    )
                    continue
                blocks_to_take = int(min(desired_blocks, n_full_blocks_avail))
                if blocks_to_take < int(MIN_FULL_BLOCKS_FOR_TEST):
                    if str(CONTIGUOUS_INSUFFICIENT_POLICY).strip().lower() == "exclude":
                        excluded = True
                        excluded_reason = f"blocks_selected(<{MIN_FULL_BLOCKS_FOR_TEST})"
                    rows.append(
                        {
                            site_col: site,
                            "n_obs": n_obs,
                            "n_test": 0,
                            "n_train_obs": n_obs,
                            "testable": False,
                            "mode": mode,
                            "n_full_blocks_avail": int(n_full_blocks_avail),
                            "n_blocks_selected": 0,
                            "excluded": bool(excluded),
                            "excluded_reason": excluded_reason,
                        }
                    )
                    continue


            if chosen_idx.size > 0:
                d.loc[chosen_idx, RAW_SPLIT] = "test"


            test_idx = chosen_idx.astype(int)
            n_test = int(test_idx.size)

        elif mode in {"test_d_holdout_start_years", "test_d_holdout_end_years"}:
            holdout_position = "start" if mode.endswith("_start_years") else "end"
            test_idx = _select_holdout_calendar_years(
                g,
                obs_idx=obs_idx,
                date_col=date_col,
                holdout_years=int(holdout_years),
                position=holdout_position,
            )
            n_test = int(test_idx.size)
            n_train_after = int(n_obs - n_test)
            if n_test <= 0:
                excluded_reason = "no_holdout_obs"
            elif n_train_after < int(min_train_obs_after_split):
                excluded_reason = "insufficient_train_after_holdout"
                test_idx = np.asarray([], dtype=int)
                n_test = 0
            elif n_test > 0:
                d.loc[test_idx, RAW_SPLIT] = "test"

        elif mode == "test_e_grab_monthly_train":
            train_idx = _select_grab_monthly_train(
                g,
                obs_idx=obs_idx,
                date_col=date_col,
                pick=str(grab_monthly_pick),
                rng=rng,
            )
            n_grab_train = int(train_idx.size)
            if n_grab_train <= 0:
                test_idx = np.asarray([], dtype=int)
                excluded_reason = "no_grab_train"
            else:
                train_set = set(train_idx.astype(int).tolist())
                test_idx = np.asarray([i for i in obs_idx.astype(int).tolist() if i not in train_set], dtype=int)
                n_test = int(test_idx.size)
                if n_test <= 0:
                    excluded_reason = "no_test_after_grab"
                elif n_grab_train < int(min_train_obs_after_split):

                    excluded_reason = "insufficient_monthly_grabs"
                    test_idx = np.asarray([], dtype=int)
                    n_test = 0
                else:
                    d.loc[test_idx, RAW_SPLIT] = "test"

        elif mode == "test_e_grab_biweekly_train":
            train_idx = _select_grab_biweekly_train(
                g,
                obs_idx=obs_idx,
                date_col=date_col,
                interval_days=int(grab_biweekly_days),
            )
            n_grab_train = int(train_idx.size)
            if n_grab_train <= 0:
                test_idx = np.asarray([], dtype=int)
                excluded_reason = "no_grab_train"
            else:
                train_set = set(train_idx.astype(int).tolist())
                test_idx = np.asarray([i for i in obs_idx.astype(int).tolist() if i not in train_set], dtype=int)
                n_test = int(test_idx.size)
                if n_test <= 0:
                    excluded_reason = "no_test_after_grab"
                elif n_grab_train < int(min_train_obs_after_split):
                    excluded_reason = "insufficient_train_grabs"
                    test_idx = np.asarray([], dtype=int)
                    n_test = 0
                    n_grab_train = 0
                else:
                    d.loc[test_idx, RAW_SPLIT] = "test"

        else:

            n_peak_target = int(round(0.05 * float(n_obs)))
            n_trough_target = int(round(0.05 * float(n_obs)))
            n_mid_target = int(round(0.10 * float(n_obs)))

            n_peak_target = int(min(n_peak_target, desired_n_test))
            n_trough_target = int(min(n_trough_target, desired_n_test - n_peak_target))
            n_mid_target = int(min(n_mid_target, desired_n_test - n_peak_target - n_trough_target))

            test_idx, _meta = _select_regime_blocks_stratified(
                g,
                obs_idx=obs_idx,
                date_col=date_col,
                target_col=target_col,
                block_days=int(block_days),
                rng=rng,
                p_lo=0.10,
                p_hi=0.90,
                n_peak_target=int(n_peak_target),
                n_trough_target=int(n_trough_target),
                n_mid_target=int(n_mid_target),
                max_tries=int(max_block_tries),
            )
            n_test = int(test_idx.size)
            if n_test > 0:
                d.loc[test_idx, RAW_SPLIT] = "test"

        rows.append(
            {
                site_col: site,
                "n_obs": n_obs,
                "n_test": int(n_test),
                "n_train_obs": int(n_obs - n_test),
                "testable": bool(n_test > 0),
                "mode": mode,
                "n_full_blocks_avail": int(n_full_blocks_avail) if mode in {"contiguous_block", "test_b_random_blocks_20pct"} else 0,
                "n_blocks_selected": int(n_blocks_selected) if mode in {"contiguous_block", "test_b_random_blocks_20pct"} else 0,
                "eligible_site": bool(eligible_site) if mode == "site_latest_block_test" else False,
                "test_block_start": test_block_start if mode == "site_latest_block_test" else pd.NaT,
                "test_block_end": test_block_end if mode == "site_latest_block_test" else pd.NaT,
                "holdout_years": int(holdout_years) if mode.startswith("test_d_") else 0,
                "holdout_position": holdout_position if mode.startswith("test_d_") else "",
                "n_grab_train": int(n_grab_train) if mode.startswith("test_e_") else 0,
                "excluded": bool(excluded_reason),
                "excluded_reason": excluded_reason,
            }
        )

    site_summary = pd.DataFrame(rows)
    d.drop(columns=["_target_obs"], inplace=True, errors="ignore")
    return d, site_summary


def _normalize_split(s: pd.Series) -> pd.Series:
    return s.astype("string").str.lower().str.strip()


def build_dnp_with_start_end(
    df: pd.DataFrame,
    *,
    site_col: str,
    date_col: str,
    target_col: str,
    split_col: str,
    testable_min_obs: int = DNP_TESTABLE_MIN_OBS,
) -> pd.DataFrame:
    """One row per site; ``site_no`` matches the Adaptive-TimeSeries convention.

    ``Start_Date`` / ``End_Date`` span the first and last non-NaN target (any split).
    ``S_Training`` / ``E_Training`` bound train-split rows with non-NaN target.
    ``S_Testing`` / ``E_Testing`` bound test-split rows with non-NaN target when any exist;
    otherwise they fall back to the train window (gap-truth schemes with no separate test
    calendar block).
    """
    required = {site_col, date_col, target_col, split_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for DNP: {sorted(missing)}")

    d = df[[site_col, date_col, target_col, split_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    d = d.sort_values([site_col, date_col], kind="stable")
    d = d.drop_duplicates(subset=[site_col, date_col], keep="first")
    sp = _normalize_split(d[split_col])
    if not (sp.eq("train") | sp.eq("test") | sp.isna()).all():
        bad = d.loc[~(sp.eq("train") | sp.eq("test") | sp.isna()), split_col].dropna().unique()[:5]
        raise ValueError(f"Unexpected {split_col!r} labels (expected train/test): {bad!r}")

    rows = []
    skipped = []
    for site_no, g in d.groupby(site_col, sort=True):
        sp_g = _normalize_split(g[split_col])
        tgt_ok = g[target_col].notna()
        if not bool(tgt_ok.any()):
            skipped.append(site_no)
            continue


        start_date = g.loc[tgt_ok, date_col].min()
        end_date = g.loc[tgt_ok, date_col].max()

        mask_tr = sp_g.eq("train") & tgt_ok
        mask_te = sp_g.eq("test") & tgt_ok

        if mask_tr.any():
            s_tr = g.loc[mask_tr, date_col].min()
            e_tr = g.loc[mask_tr, date_col].max()
            last_train_obs_date = g.loc[mask_tr, date_col].max()
        else:

            s_tr = start_date
            e_tr = end_date
            last_train_obs_date = pd.NaT

        if mask_te.any():
            s_te = g.loc[mask_te, date_col].min()
            e_te = g.loc[mask_te, date_col].max()
        else:

            s_te = s_tr
            e_te = e_tr

        n_target = int(tgt_ok.sum())
        n_tr = int(mask_tr.sum())
        n_te = int(mask_te.sum())
        testable = bool(n_te >= int(testable_min_obs))
        rows.append(
            {
                "DataType": "FullPeriodPerSite",
                "site_no": site_no,
                "Start_Date": start_date,
                "End_Date": end_date,
                "S_Training": s_tr,
                "E_Training": e_tr,
                "S_Testing": s_te,
                "E_Testing": e_te,
                "last_train_obs_date": last_train_obs_date,
                "n_obs_target": n_target,
                "n_obs_train": n_tr,
                "n_obs_test": n_te,
                "testable": testable,
            }
        )

    if skipped:
        msg = ", ".join(map(str, skipped[:10]))
        raise ValueError(f"Sites with no non-missing target values: {msg}")

    col_order = [
        "DataType",
        "site_no",
        "Start_Date",
        "End_Date",
        "S_Training",
        "E_Training",
        "S_Testing",
        "E_Testing",
        "last_train_obs_date",
        "n_obs_target",
        "n_obs_train",
        "n_obs_test",
        "testable",
    ]
    return pd.DataFrame(rows)[col_order]


def split_experiment_label(
    mode: str = SPLIT_MODE,
    block_days: int = BLOCK_DAYS,
) -> str:
    """Human-readable experiment label for split plots."""
    mode = str(mode).strip().lower()
    labels = {
        "random_points": "Random split",
        "contiguous_block": f"Contiguous block split, {block_days}-day blocks",
        "site_latest_block_test": f"Latest block test, {block_days}-day blocks",
        "test_a_random_points_20pct": "Random split, 20% test",
        "test_b_random_blocks_20pct": f"Random split, {block_days}-day blocks",
        "test_c_peak_trough_20pct": f"Peak/trough split, {block_days}-day blocks",
        "test_d_holdout_start_years": f"Holdout start, {HOLDOUT_YEARS} year(s)",
        "test_d_holdout_end_years": f"Holdout end, {HOLDOUT_YEARS} year(s)",
        "test_e_grab_monthly_train": "Grab monthly train split",
        "test_e_grab_biweekly_train": f"Grab biweekly train split, {GRAB_BIWEEKLY_DAYS}-day anchors",
    }
    return labels.get(mode, mode)


def _site_plot_label(site) -> str:
    site_str = str(site).strip().replace("USGS_", "USGS-")
    if not site_str.startswith("USGS-"):
        site_str = f"USGS-{site_str}"
    return site_str


def save_site_split_plots(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    site_col: str,
    date_col: str,
    target_col: str,
    split_col: str,
    dirname: str = PLOT_DIRNAME,
    dpi: int = PLOT_DPI,
    experiment_label: str | None = None,
) -> None:
    """
    For each site, save a scatter plot of observed target colored by split.
    Only non-NaN target rows are plotted.
    """
    out = Path(output_dir) / str(dirname)
    out.mkdir(parents=True, exist_ok=True)

    d = df[[site_col, date_col, target_col, split_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    d = d.dropna(subset=[target_col, date_col])
    if d.empty:
        return

    sp = _normalize_split(d[split_col])
    d["_sp"] = sp.where(sp.isin(["train", "test"]), other="train")
    label = experiment_label or split_experiment_label()

    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.size": PLOT_FONT_SIZE,
            "axes.titlesize": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_FONT_SIZE - 2,
            "text.color": PLOT_TEXT_COLOR,
            "axes.labelcolor": PLOT_TEXT_COLOR,
            "xtick.color": PLOT_TEXT_COLOR,
            "ytick.color": PLOT_TEXT_COLOR,
            "axes.edgecolor": PLOT_TEXT_COLOR,
        }
    )

    for site, g in d.groupby(site_col, sort=True):
        g = g.sort_values(date_col, kind="stable")
        tr = g[g["_sp"].eq("train")]
        te = g[g["_sp"].eq("test")]

        fig, ax = plt.subplots(figsize=(10, 3.5))
        if not tr.empty:
            ax.scatter(
                tr[date_col],
                tr[target_col],
                s=10,
                c=PLOT_TRAIN_COLOR,
                alpha=0.85,
                label=f"train (n={len(tr)})",
                zorder=2,
            )
        if not te.empty:
            ax.scatter(
                te[date_col],
                te[target_col],
                s=14,
                c=PLOT_TEST_COLOR,
                alpha=0.9,
                label=f"test gap (n={len(te)})",
                zorder=3,
            )

        ax.set_title(f"{_site_plot_label(site)} | {label}", color=PLOT_TEXT_COLOR)
        ax.set_xlabel("Date")
        ax.set_ylabel("Nitrate (mg/L as N)")
        ax.legend(loc="best", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.autofmt_xdate()
        fig.tight_layout()

        site_str = str(site).replace(":", "_").replace("/", "_").replace("\\", "_").strip()
        fpath = out / f"{site_str}_split.png"
        fig.savefig(fpath, dpi=int(dpi), bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _relative_path_as_posix(target: Path, start: Path) -> str:
    """Path from ``start`` to ``target`` for YAML (forward slashes)."""
    rel = os.path.relpath(target.resolve(), start.resolve())
    return Path(rel).as_posix()


def _lstm_norm_method(var: str) -> str:
    """Normalization transform for a forcing column."""
    if var in LSTM_NORM_FORCINGS:
        return LSTM_NORM_FORCINGS[var]

    if var.endswith("_interpolated") or var.endswith("_time_interpolated") or var.endswith(
        "_doy_interpolated"
    ):
        return "none"
    return LSTM_NORM_DEFAULT


def _lstm_norm_scope(var: str, method: str) -> str | None:
    """Return scope for normalized forcings; ``None`` for method ``none`` (no scope entry)."""
    if method == "none":
        return None
    return "global" if var in LSTM_NORM_GLOBAL_SCOPE else "per_site"


def _forcings_from_train_parquet(train_path: Path) -> list[str]:
    """Dynamic inputs = all train columns except date/site/target/split."""
    cols = list(pd.read_parquet(train_path).columns)
    return [c for c in cols if c not in LSTM_FORCING_META_COLS]


def _attributes_from_parquet(attr_path: Path) -> list[str]:
    """Static covariates from attribute parquet (exclude id / QC columns)."""
    cols = list(pd.read_parquet(attr_path).columns)
    return [c for c in cols if c not in LSTM_ATTR_EXCLUDE_COLS]


def write_lstm_config(
    *,
    output_dir: Path,
    attribute_input: Path,
    train_parquet: str,
    test_parquet: str,
    dnp_xlsx: str,
    split_seed: int,
    run_suffix: str,
    output_model: str | None = None,
) -> Path:
    """
    Write scenario-specific ``lstm_config_{suffix}.yaml`` into ``output_dir``.

    Forcings are taken from the train parquet columns (donor cols appear when added later).
    Attributes come from ``attribute_input``. Paths are relative to ``output_dir``.
    """
    if not LSTM_CONFIG_TEMPLATE.is_file():
        raise FileNotFoundError(f"LSTM config template not found: {LSTM_CONFIG_TEMPLATE}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(LSTM_CONFIG_TEMPLATE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid template YAML: {LSTM_CONFIG_TEMPLATE}")

    attr_file = Path(attribute_input)
    if not attr_file.is_file():
        raise FileNotFoundError(f"Attribute parquet not found: {attr_file}")

    train_path = output_dir / train_parquet
    if not train_path.is_file():
        raise FileNotFoundError(f"Train forcing not found: {train_path}")

    forcings = _forcings_from_train_parquet(train_path)
    attributes = _attributes_from_parquet(attr_file)
    if not forcings:
        raise ValueError(f"No forcing columns found in {train_path}")
    if not attributes:
        raise ValueError(f"No attribute columns found in {attr_file}")

    norm_forcings = {f: _lstm_norm_method(f) for f in forcings}
    norm_scope = {
        f: scope
        for f in forcings
        if (scope := _lstm_norm_scope(f, norm_forcings[f])) is not None
    }

    cfg["forcing_path_train"] = train_parquet
    cfg["forcing_path_test"] = test_parquet
    cfg["D_N_P_path"] = dnp_xlsx
    cfg["attr_path"] = _relative_path_as_posix(attr_file, output_dir)
    cfg["output_model"] = output_model or output_dir.name
    cfg["randomseed"] = [int(split_seed)]
    cfg["forcings"] = forcings
    cfg["attributes"] = attributes

    norm = cfg.setdefault("normalization", {})
    if not isinstance(norm, dict):
        raise ValueError("Template normalization block must be a mapping")
    norm["targets"] = {"nitrate_mg_l": "log1p"}
    norm["forcings"] = norm_forcings
    scope = norm.setdefault("scope", {})
    if not isinstance(scope, dict):
        raise ValueError("Template normalization.scope must be a mapping")
    scope["targets"] = {"nitrate_mg_l": "per_site"}
    scope["forcings"] = norm_scope

    suffix = str(run_suffix).strip().upper()
    out_path = output_dir / LSTM_CONFIG_FILENAME_TMPL.format(suffix=suffix)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    return out_path


def _load_forcing_parquet(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Forcing parquet not found: {path}")
    df = pd.read_parquet(path)
    if "site_id" in df.columns and RAW_SITE not in df.columns:
        df = df.rename(columns={"site_id": RAW_SITE})
    if RAW_SITE not in df.columns:
        raise ValueError(f"Expected site column {RAW_SITE!r} (or 'site_id') in {path}")
    df[RAW_SITE] = df[RAW_SITE].astype(str).str.strip()
    df[RAW_DATE] = pd.to_datetime(df[RAW_DATE], errors="coerce").dt.normalize()
    if df[RAW_DATE].isna().any():
        raise ValueError(f"Invalid dates in {path}")
    _floor_target_at_zero(df, RAW_TARGET)
    return df


def _split_lookup_table(df_with_split: pd.DataFrame) -> pd.DataFrame:
    """Compact (site_no, date, split) map for merging onto other sensor forcings."""
    out = df_with_split[[RAW_SITE, RAW_DATE, RAW_SPLIT]].copy()
    out[RAW_SITE] = out[RAW_SITE].astype(str).str.strip()
    out[RAW_DATE] = pd.to_datetime(out[RAW_DATE]).dt.normalize()
    out[RAW_SPLIT] = _normalize_split(out[RAW_SPLIT])
    out = out.drop_duplicates(subset=[RAW_SITE, RAW_DATE], keep="first")
    return out


def _apply_split_lookup(df: pd.DataFrame, split_lookup: pd.DataFrame) -> pd.DataFrame:
    """Merge a shared split onto another forcing; unmatched days default to train."""
    out = df.drop(columns=[RAW_SPLIT], errors="ignore").copy()
    out = out.merge(split_lookup, on=[RAW_SITE, RAW_DATE], how="left", validate="m:1")
    out[RAW_SPLIT] = out[RAW_SPLIT].fillna("train")

    return out


def meta_dir_for_case(tag: str) -> Path:
    """Case-level meta (split summary, plots, forcing_with_split) under Models/Similarity."""
    d = SIMILARITY_META_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _archive_case_to_paper_inputs(out_dir: Path, tag: str) -> Path:
    """Copy only LSTM run inputs into Paper_1A/Inputs/{tag}."""
    PAPER_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = PAPER_INPUTS_DIR / tag
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for pattern in (
        "forcing_*_train.parquet",
        "forcing_*_test.parquet",
        "lstm_config_*.yaml",
        "D_N_P_nitrate_*.xlsx",
    ):
        for src in out_dir.glob(pattern):
            shutil.copy2(src, dest / src.name)
            n += 1
    print(f"  Archived -> {dest}  ({n} model files)", flush=True)
    return dest


def _write_case_outputs(
    df: pd.DataFrame,
    case: dict,
    *,
    site_summary: pd.DataFrame | None,
    write_plots: bool = True,
    plot_df: pd.DataFrame | None = None,
) -> None:
    tag = run_tag_for_case(case)
    out_dir = output_dir_for_case(case)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = meta_dir_for_case(tag)


    cs = str(case.get("case_set") or case["n_sites"])
    train_name = f"forcing_{case['run_experiment']}_{cs}_train.parquet"
    test_name = f"forcing_{case['run_experiment']}_{cs}_test.parquet"
    dnp_name = f"D_N_P_nitrate_{case['run_experiment']}_{cs}.xlsx"
    summary_name = f"site_split_{case['run_experiment']}_{cs}_summary.csv"

    print(f"\n=== {tag} ===")
    print(f"  forcing: {case['forcing_input']}")
    print(f"  output:  {out_dir}")
    print(f"  meta:    {meta_dir}", flush=True)
    print(
        f"  block_days={case['block_days']}  dnp_testable_min_obs={case['dnp_testable_min_obs']}  "
        f"n_sites={case['n_sites']}",
        flush=True,
    )

    if not Path(case["attribute_input"]).is_file():
        print(f"  (warn) attribute parquet not found: {case['attribute_input']}", flush=True)

    if site_summary is not None:
        summary_path = meta_dir / summary_name
        site_summary.to_csv(summary_path, index=False)
        print("  Wrote", summary_path, "| sites:", len(site_summary), flush=True)


    forcing_split_path = meta_dir / f"forcing_with_split_{case['run_suffix']}.parquet"
    df_split_out = df.copy()
    if RAW_SPLIT not in df_split_out.columns:
        raise ValueError("forcing for case outputs is missing split column")
    df_split_out.to_parquet(forcing_split_path, index=False)

    df_train_raw, df_truth_raw = _mask_train_and_gaptruth(
        df,
        date_col=RAW_DATE,
        site_col=RAW_SITE,
        nitrate_col=RAW_TARGET,
        split_col=RAW_SPLIT,
    )
    for dframe in (df_train_raw, df_truth_raw):
        if RAW_SPLIT in dframe.columns:
            dframe.drop(columns=[RAW_SPLIT], inplace=True)

    input_cols = _discover_forcing_input_columns(
        df_train_raw, date_col=RAW_DATE, site_col=RAW_SITE, target_col=RAW_TARGET
    )
    global_dates = pd.date_range(df_train_raw[RAW_DATE].min(), df_train_raw[RAW_DATE].max(), freq="D")
    sites = sorted(df_train_raw[RAW_SITE].drop_duplicates().tolist())
    n_sites_actual = len(sites)
    if n_sites_actual != int(case["n_sites"]):
        print(
            f"  (warn) case n_sites={case['n_sites']} but forcing has {n_sites_actual} sites",
            flush=True,
        )

    train_df = _rectangularize_and_fill_inputs(
        df_train_raw,
        site_col=RAW_SITE,
        date_col=RAW_DATE,
        input_cols=input_cols,
        target_col=RAW_TARGET,
        global_dates=global_dates,
        sites=sites,
    )
    test_df = _rectangularize_and_fill_inputs(
        df_truth_raw,
        site_col=RAW_SITE,
        date_col=RAW_DATE,
        input_cols=input_cols,
        target_col=RAW_TARGET,
        global_dates=global_dates,
        sites=sites,
    )

    train_path = out_dir / train_name
    test_path = out_dir / test_name
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    dnp_df = build_dnp_with_start_end(
        df,
        site_col=RAW_SITE,
        date_col=RAW_DATE,
        target_col=RAW_TARGET,
        split_col=RAW_SPLIT,
        testable_min_obs=int(case["dnp_testable_min_obs"]),
    )
    dnp_path = out_dir / dnp_name
    dnp_df.to_excel(dnp_path, index=False)

    if write_plots and bool(PLOT_SITE_SPLITS):
        plot_source = plot_df if plot_df is not None else df
        save_site_split_plots(
            plot_source,
            output_dir=meta_dir,
            site_col=RAW_SITE,
            date_col=RAW_DATE,
            target_col=RAW_TARGET,
            split_col=RAW_SPLIT,
            experiment_label=split_experiment_label(
                mode=str(case["split_mode"]),
                block_days=int(case["block_days"]),
            ),
        )
        n_plots = len(list((meta_dir / PLOT_DIRNAME).glob("*_split.png")))
        print(f"  Wrote {meta_dir / PLOT_DIRNAME} | plots: {n_plots}", flush=True)

    print("  Wrote", forcing_split_path, flush=True)
    print("  Wrote", train_path, "| rows:", len(train_df), flush=True)
    print("  Wrote", test_path, "| rows:", len(test_df), flush=True)
    print("  Wrote", dnp_path, "| sites:", len(dnp_df), flush=True)

    if WRITE_LSTM_CONFIG:
        cfg_path = write_lstm_config(
            output_dir=out_dir,
            attribute_input=Path(case["attribute_input"]),
            train_parquet=train_name,
            test_parquet=test_name,
            dnp_xlsx=dnp_name,
            split_seed=int(case["split_random_seed"]),
            run_suffix=str(case["run_suffix"]),
            output_model=tag,
        )
        print("  Wrote", cfg_path, flush=True)

    _archive_case_to_paper_inputs(out_dir, tag)


def _master_case_for_gap_group(
    gap_group_key: str,
    *,
    cases: dict[str, dict] | None = None,
) -> dict:
    """Always resolve NODONOR for this gap group from the full CASES table."""
    table = cases if cases is not None else CASES
    for case in table.values():
        if case["gap_group_key"] == gap_group_key and case["is_split_master"]:
            return case
    raise KeyError(
        f"No {SPLIT_MASTER_SUFFIX} master case for gap group {gap_group_key!r}. "
        f"Check GAP_GROUPS / DONOR_SCENARIOS."
    )


def _count_testable_sites(site_summary: pd.DataFrame, *, min_test_obs: int) -> int:
    """Sites with enough test nitrate obs to pass DNP testable (n_test >= min_test_obs)."""
    if site_summary is None or site_summary.empty or "n_test" not in site_summary.columns:
        return 0
    return int((site_summary["n_test"].astype(int) >= int(min_test_obs)).sum())


def _report_testable_sites(case: dict, site_summary: pd.DataFrame) -> int:
    n_testable = _count_testable_sites(
        site_summary, min_test_obs=int(case["dnp_testable_min_obs"])
    )
    n_sites = int(case["n_sites"])
    expected = case.get("expected_testable")
    msg = (
        f"  testable sites (n_test >= {case['dnp_testable_min_obs']}): "
        f"{n_testable} / {n_sites}"
    )
    if expected is not None:
        msg += f"  (expected ~{int(expected)})"
    print(msg, flush=True)
    if expected is not None and n_testable != int(expected):
        print(
            f"  WARNING: testable site count {n_testable} differs from expected "
            f"{int(expected)} for {case['run_experiment']} "
            f"(block_days={case['block_days']}).",
            flush=True,
        )
    return n_testable


def main_plot_splits_only(
    *,
    seed: int | None = None,
    write_plots: bool = True,
) -> None:
    """Assign shared splits per gap group from NODONOR and write split plots only."""
    global CASES
    seed_i = int(SPLIT_RANDOM_SEED if seed is None else seed)
    CASES = build_case_specs(seed=seed_i)
    selected = select_cases(RUN_CASES, cases=CASES)
    by_group: dict[str, list[dict]] = {}
    for case in selected.values():
        by_group.setdefault(case["gap_group_key"], []).append(case)

    for gap_key, group_cases in by_group.items():
        master = _master_case_for_gap_group(gap_key, cases=CASES)
        print(
            f"Split plots | {gap_key} | seed={master['split_random_seed']} | "
            f"block_days={master['block_days']} | split master={master['forcing_input'].name}",
            flush=True,
        )
        df = _load_forcing_parquet(master["forcing_input"])
        df, site_summary = assign_split(
            df,
            site_col=RAW_SITE,
            date_col=RAW_DATE,
            target_col=RAW_TARGET,
            mode=str(master["split_mode"]),
            seed=int(master["split_random_seed"]),
            min_obs_to_split=MIN_OBS_TO_SPLIT,
            min_train_obs_after_split=MIN_TRAIN_OBS_AFTER_SPLIT,
            block_days=int(master["block_days"]),
            max_block_tries=MAX_BLOCK_TRIES,
            test_fraction=float(master["test_fraction"]),
        )
        _report_testable_sites(master, site_summary)

        for plot_case in group_cases:
            tag = run_tag_for_case(plot_case)
            out_dir = output_dir_for_case(plot_case)
            out_dir.mkdir(parents=True, exist_ok=True)
            meta_dir = meta_dir_for_case(tag)
            summary_path = meta_dir / (
                f"site_split_{master['run_experiment']}_{master['n_sites']}_summary.csv"
            )
            site_summary.to_csv(summary_path, index=False)
            if write_plots:
                save_site_split_plots(
                    df,
                    output_dir=meta_dir,
                    site_col=RAW_SITE,
                    date_col=RAW_DATE,
                    target_col=RAW_TARGET,
                    split_col=RAW_SPLIT,
                    experiment_label=split_experiment_label(
                        mode=str(master["split_mode"]),
                        block_days=int(master["block_days"]),
                    ),
                )
                plot_dir = meta_dir / PLOT_DIRNAME
                n_plots = len(list(plot_dir.glob("*_split.png"))) if plot_dir.is_dir() else 0
                print("Wrote", plot_dir, "| plots:", n_plots, flush=True)
            _archive_case_to_paper_inputs(out_dir, tag)


def main(
    *,
    seed: int | None = None,
    write_plots: bool | None = None,
) -> None:
    """
    For each gap-duration group, assign the split once on NODONOR (always), then merge
    that split into the other active-cohort suffixes so gap dates match.
    """
    global CASES
    seed_i = int(SPLIT_RANDOM_SEED if seed is None else seed)
    CASES = build_case_specs(seed=seed_i)
    do_plots = bool(PLOT_SITE_SPLITS if write_plots is None else write_plots)

    selected = select_cases(RUN_CASES, cases=CASES)
    print(f"ACTIVE_COHORT={ACTIVE_COHORT}", flush=True)
    print(f"SPLIT_RANDOM_SEED={seed_i}", flush=True)
    print(f"Cases to run ({len(selected)}): {', '.join(selected)}", flush=True)

    by_group: dict[str, list[dict]] = {}
    for case in selected.values():
        by_group.setdefault(case["gap_group_key"], []).append(case)

    for gap_key, group_cases in by_group.items():
        master = _master_case_for_gap_group(gap_key, cases=CASES)

        print(
            f"\n######## Gap group {gap_key} | block_days={master['block_days']} | "
            f"seed={master['split_random_seed']} | "
            f"split master={master['forcing_input'].name} ########",
            flush=True,
        )
        df_master = _load_forcing_parquet(master["forcing_input"])
        n_sites_actual = int(df_master[RAW_SITE].nunique())
        if n_sites_actual != int(master["n_sites"]):
            print(
                f"  WARNING: forcing has {n_sites_actual} sites but case label "
                f"n_sites={master['n_sites']}",
                flush=True,
            )
        df_master, site_summary = assign_split(
            df_master,
            site_col=RAW_SITE,
            date_col=RAW_DATE,
            target_col=RAW_TARGET,
            mode=str(master["split_mode"]),
            seed=int(master["split_random_seed"]),
            min_obs_to_split=MIN_OBS_TO_SPLIT,
            min_train_obs_after_split=MIN_TRAIN_OBS_AFTER_SPLIT,
            block_days=int(master["block_days"]),
            max_block_tries=MAX_BLOCK_TRIES,
            test_fraction=float(master["test_fraction"]),
        )
        split_lookup = _split_lookup_table(df_master)
        n_test_labels = int(_normalize_split(split_lookup[RAW_SPLIT]).eq("test").sum())
        print(
            f"  shared split: {n_test_labels:,} test site-days on "
            f"{master['forcing_input'].name}",
            flush=True,
        )
        _report_testable_sites(master, site_summary)

        ordered = sorted(
            group_cases,
            key=lambda c: (0 if c["case_name"] == master["case_name"] else 1, c["run_suffix"]),
        )
        for case in ordered:
            if case["case_name"] == master["case_name"]:
                df = df_master
            else:
                df = _apply_split_lookup(
                    _load_forcing_parquet(case["forcing_input"]),
                    split_lookup,
                )
            _write_case_outputs(
                df,
                case,
                site_summary=site_summary,
                write_plots=do_plots,
                plot_df=df_master,
            )


def _resolve_seeds(cli_seeds: list[int] | None) -> list[int]:
    if cli_seeds:
        return [int(s) for s in cli_seeds]
    if SPLIT_RANDOM_SEEDS:
        return [int(s) for s in SPLIT_RANDOM_SEEDS]
    return [int(SPLIT_RANDOM_SEED)]


def _resolve_cohorts(cli_cohorts: list[str] | None) -> list[str]:
    if cli_cohorts:
        return list(cli_cohorts)
    if RUN_COHORTS:
        return list(RUN_COHORTS)
    return [ACTIVE_COHORT]


def main_case_sets(
    *,
    case_sets: list[str] | None = None,
    gaps: list[str] | None = None,
    seeds: list[int] | None = None,
    write_plots: bool = False,
    force_populations: bool = False,
) -> None:
    """
    Build all experiment cases with fixed populations per (case_set, gap).

    Population (≥300 nitrate obs, ≥1 full block, and REG∩SIG for donor sets)
    is computed once per (case_set, gap) and reused for every seed/scenario.
    """
    import sys

    aux = SCRIPTS_DIR / "Auxilliary_Scripts"
    if str(aux) not in sys.path:
        sys.path.insert(0, str(aux))
    import experiment_case_sets as ecs

    ecs.assert_case_set_design()


    requested = list(case_sets) if case_sets else list(ecs.CASE_SETS.keys())
    for cs in requested:
        if cs not in ecs.CASE_SETS:
            raise SystemExit(f"Unknown case set {cs!r}; choose from {list(ecs.CASE_SETS)}")
    sets = [cs for cs in requested if not ecs.CASE_SETS[cs].get("single_site")]
    gap_list = list(gaps) if gaps else list(ecs.GAP_SPECS.keys())
    seed_list = list(seeds) if seeds else list(ecs.SPLIT_SEEDS)
    for g in gap_list:
        if g not in ecs.GAP_SPECS:
            raise SystemExit(f"Unknown gap {g!r}; choose from {list(ecs.GAP_SPECS)}")


    print("\n======== Building base site populations ========", flush=True)
    populations: dict[tuple[str, str], list[str]] = {}
    for cs in sets:
        for gap in gap_list:
            populations[(cs, gap)] = ecs.build_population_for_gap(
                cs,
                gap,
                block_days=int(ecs.GAP_SPECS[gap]),
                min_nitrate_obs=int(ecs.MIN_NITRATE_OBS),
                min_train_obs_after_split=int(ecs.MIN_TRAIN_OBS_AFTER_SPLIT),
                force_recompute=force_populations,
            )


    for cs in sets:
        spec = ecs.CASE_SETS[cs]
        for gap in gap_list:
            block_days = int(ecs.GAP_SPECS[gap])
            allowed_months = ecs.gap_allowed_months(gap)
            site_list = populations[(cs, gap)]
            site_set = set(site_list)
            n_sites = len(site_list)
            if n_sites == 0:
                print(f"SKIP {cs}/{gap}: empty population", flush=True)
                continue

            for seed in seed_list:
                print(
                    f"\n{'=' * 72}\nCASE_SET={cs}  GAP={gap}  SEED={seed}  "
                    f"n_sites={n_sites}\n{'=' * 72}",
                    flush=True,
                )

                base_path = Path(spec["forcing_by_scenario"]["BASELINE"])
                df_master = _load_forcing_parquet(base_path)
                df_master = df_master.loc[
                    df_master[RAW_SITE].astype(str).str.strip().isin(site_set)
                ].copy()
                if allowed_months:
                    print(
                        f"  block month filter: Oct–May only "
                        f"(months={sorted(allowed_months)})",
                        flush=True,
                    )
                df_master, site_summary = assign_split(
                    df_master,
                    site_col=RAW_SITE,
                    date_col=RAW_DATE,
                    target_col=RAW_TARGET,
                    mode=SPLIT_MODE,
                    seed=int(seed),
                    min_obs_to_split=MIN_OBS_TO_SPLIT,
                    min_train_obs_after_split=MIN_TRAIN_OBS_AFTER_SPLIT,
                    block_days=block_days,
                    max_block_tries=MAX_BLOCK_TRIES,
                    test_fraction=TEST_FRACTION,
                    block_allowed_months=allowed_months,
                )
                if allowed_months:
                    test_dates = pd.to_datetime(
                        df_master.loc[
                            _normalize_split(df_master[RAW_SPLIT]).eq("test"),
                            RAW_DATE,
                        ]
                    )
                    bad_month = test_dates[~test_dates.dt.month.isin(allowed_months)]
                    if not bad_month.empty:
                        raise AssertionError(
                            f"{cs}/{gap}/Seed{seed}: {len(bad_month)} test site-days "
                            f"fall outside allowed months {sorted(allowed_months)}"
                        )
                split_lookup = _split_lookup_table(df_master)
                print(
                    f"  shared split: "
                    f"{int(_normalize_split(split_lookup[RAW_SPLIT]).eq('test').sum()):,} "
                    f"test site-days",
                    flush=True,
                )

                for scenario in spec["scenarios"]:
                    tag = ecs.case_tag(gap, cs, seed, scenario)
                    fpath = Path(spec["forcing_by_scenario"][scenario])
                    case = {
                        "case_name": tag,
                        "case_set": cs,
                        "run_experiment": gap,
                        "n_sites": n_sites,
                        "run_suffix": scenario,
                        "block_days": block_days,
                        "dnp_testable_min_obs": block_days,
                        "expected_testable": None,
                        "split_random_seed": int(seed),
                        "split_mode": SPLIT_MODE,
                        "test_fraction": TEST_FRACTION,
                        "forcing_input": fpath,
                        "attribute_input": ATTRIBUTE_INPUT,
                        "is_split_master": scenario == "BASELINE",
                        "gap_group_key": ecs.gap_group_key(gap, cs, seed),
                        "cohort": cs,
                    }
                    if scenario == "BASELINE":
                        df = df_master
                    else:
                        df_other = _load_forcing_parquet(fpath)
                        df_other = df_other.loc[
                            df_other[RAW_SITE].astype(str).str.strip().isin(site_set)
                        ].copy()

                        master_keys = (
                            df_master[[RAW_SITE, RAW_DATE]]
                            .drop_duplicates()
                            .copy()
                        )
                        scenario_keys = (
                            df_other[[RAW_SITE, RAW_DATE]]
                            .drop_duplicates()
                            .copy()
                        )
                        master_keys[RAW_SITE] = (
                            master_keys[RAW_SITE].astype(str).str.strip()
                        )
                        scenario_keys[RAW_SITE] = (
                            scenario_keys[RAW_SITE].astype(str).str.strip()
                        )
                        master_keys[RAW_DATE] = pd.to_datetime(
                            master_keys[RAW_DATE]
                        ).dt.normalize()
                        scenario_keys[RAW_DATE] = pd.to_datetime(
                            scenario_keys[RAW_DATE]
                        ).dt.normalize()
                        master_keys = master_keys.sort_values(
                            [RAW_SITE, RAW_DATE]
                        ).reset_index(drop=True)
                        scenario_keys = scenario_keys.sort_values(
                            [RAW_SITE, RAW_DATE]
                        ).reset_index(drop=True)
                        if not master_keys.equals(scenario_keys):
                            raise ValueError(
                                f"{tag}: complete site-date keys differ from BASELINE "
                                f"(master={len(master_keys)} other={len(scenario_keys)} "
                                f"forcing={fpath.name})"
                            )
                        df = _apply_split_lookup(df_other, split_lookup)

                    expected_sites = set(site_list)
                    actual_sites = set(df[RAW_SITE].astype(str).str.strip().unique())
                    if actual_sites != expected_sites:
                        raise AssertionError(
                            f"{tag}: site set mismatch "
                            f"(extra={sorted(actual_sites - expected_sites)[:5]} "
                            f"missing={sorted(expected_sites - actual_sites)[:5]})"
                        )

                    _write_case_outputs(
                        df,
                        case,
                        site_summary=site_summary,
                        write_plots=write_plots and scenario == "BASELINE",
                        plot_df=df_master,
                    )


if __name__ == "__main__":
    import argparse
    import sys

    aux = SCRIPTS_DIR / "Auxilliary_Scripts"
    if str(aux) not in sys.path:
        sys.path.insert(0, str(aux))
    import experiment_case_sets as ecs

    parser = argparse.ArgumentParser(
        description=(
            "Build LSTM train/test cases for C1/C2/C3 (pooled) with fixed site "
            "populations and split seeds 21/42/84. C4 single-site cases are built "
            "by 15_c4_single_site.py (also via --case-sets C4)."
        )
    )
    parser.add_argument(
        "--case-sets",
        nargs="+",
        default=None,
        choices=list(ecs.CASE_SETS.keys()),
        help="Case sets to build (default: pooled C1 C2 C3; C4 via --case-sets C4).",
    )
    parser.add_argument(
        "--gaps",
        nargs="+",
        default=None,
        choices=list(ecs.GAP_SPECS.keys()),
        help="Gap keys (default: R20B14 R20B60 R20B365).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Split seeds (default: 21 42 84).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip per-site split PNGs (default for speed).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write split PNGs for BASELINE cases.",
    )
    parser.add_argument(
        "--force-populations",
        action="store_true",
        help="Recompute site population CSVs even if cached.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run the old ACTIVE_COHORT / NODONOR pipeline instead.",
    )
    args = parser.parse_args()

    if args.legacy:
        seeds = _resolve_seeds(args.seeds)
        cohorts = _resolve_cohorts(None)
        do_plots = bool(args.plots) and bool(PLOT_SITE_SPLITS)
        for cohort in cohorts:
            activate_cohort(cohort)
            for seed in seeds:
                print(
                    f"\n{'=' * 72}\nCOHORT={cohort}  SEED={seed}\n{'=' * 72}",
                    flush=True,
                )
                main(seed=seed, write_plots=do_plots)
    else:
        requested = list(args.case_sets) if args.case_sets else None
        single = [
            cs
            for cs in (requested or [])
            if cs in ecs.CASE_SETS and ecs.CASE_SETS[cs].get("single_site")
        ]
        pooled = None if requested is None else [cs for cs in requested if cs not in single]
        if pooled is None or pooled:
            main_case_sets(
                case_sets=pooled,
                gaps=args.gaps,
                seeds=args.seeds,
                write_plots=bool(args.plots),
                force_populations=bool(args.force_populations),
            )
        if single:
            import importlib.util

            c4_path = SCRIPTS_DIR / "15_c4_single_site.py"
            spec = importlib.util.spec_from_file_location("c4_single_site", c4_path)
            if spec is None or spec.loader is None:
                raise SystemExit(f"Cannot load {c4_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.build_c4_cases(gaps=args.gaps, seeds=args.seeds)
