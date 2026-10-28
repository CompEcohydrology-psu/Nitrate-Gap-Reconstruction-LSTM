#!/usr/bin/env python3
"""Shared C1/C2/C3 case-set definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PAPER_ROOT = SCRIPTS_DIR.parent
DATASETS = PAPER_ROOT / "Datasets"
SIMILARITY = PAPER_ROOT / "Models" / "Similarity"
POP_DIR = SIMILARITY / "populations"

MIN_NITRATE_OBS = 300
MIN_TRAIN_OBS_AFTER_SPLIT = 40
SPLIT_SEEDS = (21, 42, 84)

FORCING_117 = DATASETS / "forcing_117_V1.parquet"
FORCING_SS = DATASETS / "forcing_SS_52_V1.parquet"
FORCING_BASE_50 = DATASETS / "forcing_BASE_50_V1.parquet"
FORCING_REDUCED_50 = DATASETS / "forcing_REDUCED_50_V1.parquet"
FORCING_ALL_50 = DATASETS / "forcing_ALL_50_V1.parquet"

CASE_SETS: Dict[str, dict] = {
    "C1": {
        "title": "network_117_donors",
        "source": FORCING_117,
        "needs_donors": True,
        "scenarios": ("BASELINE", "REGDONOR", "SIGDONOR", "REGSIGDONOR"),
        "forcing_by_scenario": {
            "BASELINE": FORCING_117,
            "REGDONOR": FORCING_117,
            "SIGDONOR": FORCING_117,
            "REGSIGDONOR": FORCING_117,
        },
    },
    "C2": {
        "title": "ss_sensor_only",
        "source": FORCING_SS,
        "needs_donors": False,
        "scenarios": ("BASELINE", "ALLSENSOR", "REDUCEDSENSOR"),
        "forcing_by_scenario": {
            "BASELINE": FORCING_BASE_50,
            "ALLSENSOR": FORCING_ALL_50,
            "REDUCEDSENSOR": FORCING_REDUCED_50,
        },
    },
    "C3": {
        "title": "ss_full_compare",
        "source": FORCING_SS,
        "needs_donors": True,
        "scenarios": (
            "BASELINE",
            "REGDONOR",
            "SIGDONOR",
            "ALLSENSOR",
            "REDUCEDSENSOR",
            "REGSIGDONOR",
            "REGSIGALLSENSOR",
        ),
        "forcing_by_scenario": {
            "BASELINE": FORCING_BASE_50,
            "REGDONOR": FORCING_BASE_50,
            "SIGDONOR": FORCING_BASE_50,
            "ALLSENSOR": FORCING_ALL_50,
            "REDUCEDSENSOR": FORCING_REDUCED_50,
            "REGSIGDONOR": FORCING_BASE_50,
            "REGSIGALLSENSOR": FORCING_ALL_50,
        },
    },
    "C4": {
        "title": "network_117_single_site",
        "source": FORCING_117,
        "needs_donors": False,
        "single_site": True,
        "mirror_case_set": "C1",
        "seeds": (21,),
        "scenarios": ("BASELINE",),
        "forcing_by_scenario": {
            "BASELINE": FORCING_117,
        },
    },
}

GAP_SPECS = {
    "R20B14": 14,
    "R20B60": 60,
    "R20B120": 120,
    "R20B365": 365,
}


OCT_MAY_MONTHS: FrozenSet[int] = frozenset({10, 11, 12, 1, 2, 3, 4, 5})

GAP_BLOCK_RULES: Dict[str, dict] = {
    "R20B120": {
        "allowed_months": OCT_MAY_MONTHS,
        "description": "Test blocks must fall entirely within Oct–May",
    },
}


def gap_allowed_months(gap: str) -> Optional[FrozenSet[int]]:
    """Calendar months every day of a test block must fall in, or None if unrestricted."""
    rules = GAP_BLOCK_RULES.get(gap)
    if not rules:
        return None
    return rules.get("allowed_months")


def block_window_in_allowed_months(
    start: pd.Timestamp,
    block_days: int,
    allowed_months: Optional[Iterable[int]],
) -> bool:
    """True if every calendar day in [start, start+block_days-1] is in allowed_months."""
    if not allowed_months:
        return True
    allowed = {int(m) for m in allowed_months}
    s0 = pd.Timestamp(start).normalize()
    for k in range(int(block_days)):
        if (s0 + pd.Timedelta(days=k)).month not in allowed:
            return False
    return True


def case_tag(gap: str, case_set: str, seed: int, scenario: str) -> str:
    return f"{gap}_{case_set}_Seed{int(seed)}_{scenario}"


def gap_group_key(gap: str, case_set: str, seed: int) -> str:
    return f"{gap}_{case_set}_Seed{int(seed)}"


def population_csv(case_set: str, gap: str) -> Path:
    return POP_DIR / f"sites_{case_set}_{gap}.csv"


def load_population(case_set: str, gap: str) -> List[str]:
    path = population_csv(case_set, gap)
    if not path.is_file():
        raise FileNotFoundError(f"Missing population list: {path}")
    df = pd.read_csv(path)
    col = "site_no" if "site_no" in df.columns else df.columns[0]
    return [str(x).strip() for x in df[col].tolist() if str(x).strip()]


def filter_min_nitrate_obs(df: pd.DataFrame, min_obs: int = MIN_NITRATE_OBS) -> pd.DataFrame:
    counts = (
        df.groupby(df["site_no"].astype(str).str.strip())["nitrate_mg_l"]
        .apply(lambda s: int(s.notna().sum()))
    )
    keep = set(counts[counts >= int(min_obs)].index.astype(str))
    return df.loc[df["site_no"].astype(str).str.strip().isin(keep)].copy()


def sites_eligible_for_gap(
    df: pd.DataFrame,
    *,
    block_days: int,
    min_train_obs_after_split: int = MIN_TRAIN_OBS_AFTER_SPLIT,
    allowed_months: Optional[Iterable[int]] = None,
    site_col: str = "site_no",
    date_col: str = "date",
    target_col: str = "nitrate_mg_l",
) -> Set[str]:
    """
    Sites with >=1 fully-observed contiguous nitrate window of length block_days.

    For 365-day gaps, also require n_obs - 365 >= min_train_obs_after_split so a
    single annual test block can leave enough training observations.

    When ``allowed_months`` is set (e.g. Oct–May for R20B120), the window must also
    fall entirely within those calendar months.
    """
    out: Set[str] = set()
    d = df.copy()
    d[site_col] = d[site_col].astype(str).str.strip()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    bd = int(block_days)
    for sid, g in d.groupby(site_col, sort=False):
        obs = g.loc[g[target_col].notna(), date_col]
        if obs.empty:
            continue
        dates = pd.DatetimeIndex(obs).unique().sort_values()
        n_obs = len(dates)
        if n_obs < bd:
            continue
        if bd == 365 and (n_obs - bd) < int(min_train_obs_after_split):
            continue
        if allowed_months:
            obs_set = set(pd.to_datetime(dates).to_pydatetime())
            found = False
            for s0 in dates:
                if not block_window_in_allowed_months(s0, bd, allowed_months):
                    continue
                ok = True
                for k in range(bd):
                    if (pd.Timestamp(s0) + pd.Timedelta(days=k)).to_pydatetime() not in obs_set:
                        ok = False
                        break
                if ok:
                    found = True
                    break
            if found:
                out.add(str(sid))
            continue
        full = pd.date_range(dates.min(), dates.max(), freq="D")
        present = pd.Series(0, index=full)
        present.loc[present.index.isin(dates)] = 1
        roll = present.rolling(bd, min_periods=bd).sum()
        if (roll >= bd).any():
            out.add(str(sid))
    return out


def sites_with_full_block(*args, **kwargs) -> Set[str]:
    return sites_eligible_for_gap(*args, **kwargs)


def build_population_for_gap(
    case_set: str,
    gap: str,
    block_days: int,
    *,
    min_nitrate_obs: int = MIN_NITRATE_OBS,
    min_train_obs_after_split: int = MIN_TRAIN_OBS_AFTER_SPLIT,
    force_recompute: bool = False,
) -> List[str]:
    """Build and cache the base site list for (case_set, gap) — no donor selection."""
    out_path = population_csv(case_set, gap)
    if out_path.is_file() and not force_recompute:
        sites = load_population(case_set, gap)
        print(
            f"  Population {case_set}/{gap}: cached n={len(sites)} ({out_path.name})",
            flush=True,
        )
        return sites

    spec = CASE_SETS[case_set]

    if case_set in ("C2", "C3") and FORCING_BASE_50.is_file():
        df = pd.read_parquet(FORCING_BASE_50)
    else:
        df = pd.read_parquet(Path(spec["source"]))
    df["site_no"] = df["site_no"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    df = filter_min_nitrate_obs(df, min_nitrate_obs)
    n_after_300 = int(df["site_no"].nunique())
    gap_ok = sites_eligible_for_gap(
        df,
        block_days=block_days,
        min_train_obs_after_split=min_train_obs_after_split,
        allowed_months=gap_allowed_months(gap),
    )
    sites = sorted(gap_ok)

    POP_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"site_no": sites}).to_csv(out_path, index=False)
    pd.DataFrame(
        [
            {
                "case_set": case_set,
                "gap": gap,
                "block_days": block_days,
                "min_nitrate_obs": min_nitrate_obs,
                "min_train_obs_after_split": min_train_obs_after_split,
                "n_after_300": n_after_300,
                "n_final": len(sites),
                "needs_donors": bool(spec["needs_donors"]),
            }
        ]
    ).to_csv(POP_DIR / f"meta_{case_set}_{gap}.csv", index=False)
    print(
        f"  Population {case_set}/{gap}: "
        f">={min_nitrate_obs}->{n_after_300}, "
        f"gap_eligible(block={block_days})->{len(sites)} -> {out_path.name}",
        flush=True,
    )
    return sites


def assert_case_set_design() -> None:
    """Fail fast if scenario / forcing maps drift from the paper design."""
    assert len(CASE_SETS["C1"]["scenarios"]) == 4
    assert len(CASE_SETS["C2"]["scenarios"]) == 3
    assert len(CASE_SETS["C3"]["scenarios"]) == 7
    assert CASE_SETS["C1"]["scenarios"] == (
        "BASELINE",
        "REGDONOR",
        "SIGDONOR",
        "REGSIGDONOR",
    )
    assert CASE_SETS["C2"]["scenarios"] == (
        "BASELINE",
        "ALLSENSOR",
        "REDUCEDSENSOR",
    )
    for cs, spec in CASE_SETS.items():
        for sc in spec["scenarios"]:
            assert sc in spec["forcing_by_scenario"], f"{cs}/{sc} missing forcing"
            path = Path(spec["forcing_by_scenario"][sc])
            assert path.name, f"{cs}/{sc} empty path"

    assert CASE_SETS["C1"]["forcing_by_scenario"]["BASELINE"] == FORCING_117
    assert CASE_SETS["C2"]["forcing_by_scenario"]["BASELINE"] == FORCING_BASE_50
    assert CASE_SETS["C3"]["forcing_by_scenario"]["BASELINE"] == FORCING_BASE_50
    assert CASE_SETS["C3"]["forcing_by_scenario"]["REGSIGALLSENSOR"] == FORCING_ALL_50
    assert CASE_SETS["C4"]["single_site"] is True
    assert CASE_SETS["C4"]["needs_donors"] is False
    assert CASE_SETS["C4"]["mirror_case_set"] == "C1"
    assert CASE_SETS["C4"]["scenarios"] == ("BASELINE",)
    assert CASE_SETS["C4"]["seeds"] == (21,)
    assert CASE_SETS["C4"]["forcing_by_scenario"]["BASELINE"] == FORCING_117
