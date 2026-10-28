#!/usr/bin/env python3
"""Attach regional and signature donor nitrate channels."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yaml


SCRIPTS_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SCRIPTS_DIR.parent
INPUTS_ROOT = PAPER_ROOT / "Models" / "LSTM_SplitTest" / "Inputs"
SIMILARITY_META_ROOT = PAPER_ROOT / "Models" / "Similarity"
PAPER_INPUTS_ARCHIVE = PAPER_ROOT / "Inputs"
DATASETS = PAPER_ROOT / "Datasets"

NLDI_CANDIDATES_CSV = DATASETS / "Discharge_0060" / "nitrate_site_nldi_candidates.csv"
RAW_NITRATE_CANDIDATES = (
    PAPER_ROOT / "Similarity" / "USGS_raw_nitrate_99133.parquet",
    PAPER_ROOT.parent / "Paper1_Final" / "Similarity" / "USGS_raw_nitrate_99133.parquet",
    PAPER_ROOT.parent / "Data_Processing_V2" / "USGS_raw_nitrate_99133.parquet",
    DATASETS / "Nitrate_99133" / "USGS_MRBCB_raw_nitrate_154_99133.parquet",
)
RAW_NITRATE_PARQUET = next((p for p in RAW_NITRATE_CANDIDATES if p.is_file()), RAW_NITRATE_CANDIDATES[0])
ATTRIBUTE_PARQUET = DATASETS / "attribute_117_V1.parquet"

SIGNATURE_CANDIDATE_FORCING = DATASETS / "forcing_117_V1.parquet"
EXPECTED_SEEDS = {21, 42, 84}
METADATA_CSV_CANDIDATES = (
    PAPER_ROOT.parent / "Paper1_Final" / "Datasets" / "sites_metadata_99133.csv",
    DATASETS / "sites_metadata_99133.csv",
    PAPER_ROOT / "Similarity" / "sites_metadata_99133.csv",
)

MIN_COVERAGE = 0.75
MAX_SHORT_GAP_DAYS = 14
DOY_WINDOW_DAYS = 15
HUC_EXPAND_LEVELS: Tuple[int, ...] = (12, 10, 8, 6, 4)
HUC_CONTRACT_LEVELS: Tuple[int, ...] = (4, 6, 8, 10, 12)


MIN_TOTAL_PAIRS = 60
MIN_FLOW_GROUP_PAIRS = 30
CQ_FEATURES = [
    "cq_slope_all",
    "cq_slope_low",
    "cq_slope_high",
    "cq_r2_all",
    "cq_variability_ratio",
]

CQ_SLOPE_EPS = 0.10
CQ_VAR_CHEMOSTATIC = 0.50

SIGNATURE_FEATURES: Dict[str, List[str]] = {
    "nitrate": [
        "cv_NO3",
        "NO3_autocorr_7day",
        "mean_log1p_NO3",
        *CQ_FEATURES,
    ],
    "hydrologic": [
        "cv_Q",
        "flashiness_index",
        "log_mean_Q",
        "baseflow_index",
        "log_drainage_area",
        "tile_drain_fraction",
    ],
    "watershed": [
        "basinatlas_pre_mm_syr",
        "basinatlas_ari_ix_sav",
        "basinatlas_crp_pc_sse",
        "basinatlas_for_pc_sse",
        "basinatlas_urb_pc_sse",
        "basinatlas_cly_pc_sav",
        "basinatlas_gwt_cm_sav",
        "basinatlas_ele_mt_sav",
        "basinatlas_slp_dg_sav",
    ],
}


DYNAMIC_NITRATE_FEATURES = {"cv_NO3", "NO3_autocorr_7day", "mean_log1p_NO3"}
DYNAMIC_Q_FEATURES = {"cv_Q", "flashiness_index", "log_mean_Q", "baseflow_index"}
STATIC_FEATURES = (
    set(SIGNATURE_FEATURES["watershed"])
    | {"log_drainage_area", "tile_drain_fraction"}
)
GROUP_WEIGHTS = {"nitrate": 1.0, "hydrologic": 1.0, "watershed": 1.0}

REG_VAL, REG_MASK, REG_ID = "regional_donor", "regional_donor_obs_mask", "regional_donor_id"
SIG_VAL, SIG_MASK, SIG_ID = "signature_donor", "signature_donor_obs_mask", "signature_donor_id"
OBS_MASK_TRUSTED = frozenset({"observed", "interp"})


def canonical_usgs_site_id(site_id: object) -> str:
    s = str(site_id).strip()
    if not s or s.lower() == "nan":
        return ""
    m = re.match(r"^USGS[-_]?(.+)$", s, re.I)
    if m:
        return f"USGS-{m.group(1)}"
    digits = re.sub(r"\D", "", s)
    if digits:
        return f"USGS-{digits}"
    return s


def resolve_metadata_csv() -> Path:
    for p in METADATA_CSV_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "sites_metadata_99133.csv not found. Tried:\n"
        + "\n".join(f"  - {p}" for p in METADATA_CSV_CANDIDATES)
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r_km * math.asin(math.sqrt(min(1.0, a)))


def parse_point_wkt(point_wkt: object) -> Tuple[float, float]:
    s = str(point_wkt).strip()
    if not s.upper().startswith("POINT"):
        raise ValueError(f"Unexpected geometry: {point_wkt!r}")
    inner = s[s.find("(") + 1 : s.rfind(")")].strip()
    lon_s, lat_s = inner.split()
    return float(lat_s), float(lon_s)


def normalize_huc12(huc: object) -> str:
    if pd.isna(huc):
        return ""
    s = "".join(ch for ch in str(huc).strip() if ch.isdigit())
    return s.zfill(12)[:12] if s else ""


def huc_prefix(huc12: str, level: int) -> str:
    return huc12[: int(level)] if huc12 else ""


@dataclass(frozen=True)
class DonorCase:
    case_dir: Path
    track: str
    tag: str
    train_parquet: Path
    test_parquet: Path


@dataclass(frozen=True)
class DonorFamilyJob:
    family: str
    cases: Tuple[DonorCase, ...]
    inputs_root: Path
    meta_root: Path
    nldi_csv: Path
    raw_nitrate: Path
    attributes: Path
    write_plots: bool
    max_plot_sites: Optional[int]


@dataclass(frozen=True)
class DualDonorSitePlotJob:
    site: str
    site_dir: Path
    dates_iso: Tuple[str, ...]
    regional_vals: Tuple[float, ...]
    signature_vals: Tuple[float, ...]
    train_obs_dates: Tuple[str, ...]
    train_obs_vals: Tuple[float, ...]
    test_obs_dates: Tuple[str, ...]
    test_obs_vals: Tuple[float, ...]
    target_q_dates: Tuple[str, ...]
    target_q_vals: Tuple[float, ...]
    regional_q_dates: Tuple[str, ...]
    regional_q_vals: Tuple[float, ...]
    signature_q_dates: Tuple[str, ...]
    signature_q_vals: Tuple[float, ...]
    regional_donor_id: str
    signature_donor_id: str


@dataclass(frozen=True)
class PlotPairJob:
    pair_key: str
    reg_case: DonorCase
    sig_case: DonorCase
    meta_root: Path
    max_plot_sites: Optional[int]
    site_workers: int


def discover_donor_cases(
    inputs_root: Path,
    *,
    tracks: Sequence[str],
    case_filters: Optional[Sequence[str]] = None,
    gap_filters: Optional[Sequence[str]] = None,
) -> List[DonorCase]:
    out: List[DonorCase] = []
    for case_dir in sorted(p for p in inputs_root.iterdir() if p.is_dir()):
        name = case_dir.name
        if case_filters and not any(tok in name for tok in case_filters):
            continue
        if gap_filters and not any(name.startswith(g) for g in gap_filters):
            continue

        if name.endswith(("_REGSIGDONOR", "_REGSIGALLSENSOR")):
            continue
        if name.endswith("_REGDONOR") and "regional" in tracks:
            track = "regional"
        elif name.endswith("_SIGDONOR") and "signature" in tracks:
            track = "signature"
        else:
            continue
        trains = sorted(case_dir.glob("forcing_*_train.parquet"))
        tests = sorted(case_dir.glob("forcing_*_test.parquet"))
        if not trains or not tests:
            print(f"SKIP (missing train/test): {name}", flush=True)
            continue
        out.append(
            DonorCase(
                case_dir=case_dir,
                track=track,
                tag=name,
                train_parquet=trains[0],
                test_parquet=tests[0],
            )
        )
    return out


def load_site_meta(path: Path) -> Dict[str, dict]:
    df = pd.read_csv(path)
    id_col = "monitoring_location_id" if "monitoring_location_id" in df.columns else "site_no"
    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        sid = canonical_usgs_site_id(row[id_col])
        if not sid:
            continue
        try:
            lat, lon = parse_point_wkt(row["geometry"])
        except Exception:
            continue
        out[sid] = {
            "lat": lat,
            "lon": lon,
            "huc12": normalize_huc12(row.get("hydrologic_unit_code", "")),
        }
    return out


def load_nldi_candidates(path: Path) -> pd.DataFrame:
    """NLDI UM/DM topology: identifier=candidate, target_site=target."""
    df = pd.read_csv(path)
    df["candidate_id"] = df["identifier"].map(canonical_usgs_site_id)
    df["target_id"] = df["target_site"].map(canonical_usgs_site_id)
    df = df.loc[df["candidate_id"].ne("") & df["target_id"].ne("")].copy()
    return df.loc[df["candidate_id"] != df["target_id"]]


def load_forcing_no3_and_q(
    path: Path,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    df = pd.read_parquet(path)
    df["site_no"] = df["site_no"].map(canonical_usgs_site_id)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    no3: Dict[str, pd.Series] = {}
    q: Dict[str, pd.Series] = {}
    for sid, g in df.groupby("site_no", sort=True):
        g = g.sort_values("date")
        idx = pd.DatetimeIndex(g["date"])
        s_no3 = pd.Series(g["nitrate_mg_l"].to_numpy(dtype=float), index=idx).dropna()
        no3[str(sid)] = s_no3.groupby(level=0).mean().sort_index()
        if "discharge_cfs" in g.columns:
            s_q = pd.Series(g["discharge_cfs"].to_numpy(dtype=float), index=idx)
            q[str(sid)] = s_q.groupby(level=0).mean().sort_index()
    return no3, q


_SIGNATURE_CANDIDATE_CACHE: Optional[
    Tuple[List[str], Dict[str, pd.Series], Dict[str, pd.Series]]
] = None


def load_signature_candidate_series(
    forcing_path: Path = SIGNATURE_CANDIDATE_FORCING,
) -> Tuple[List[str], Dict[str, pd.Series], Dict[str, pd.Series]]:
    """
    Signature-donor candidate population: all sites in the 117-network forcing
    with at least one nitrate and one discharge observation.

    Independent of any case's retained target / common-pool subset.
    """
    global _SIGNATURE_CANDIDATE_CACHE
    if (
        _SIGNATURE_CANDIDATE_CACHE is not None
        and forcing_path.resolve() == SIGNATURE_CANDIDATE_FORCING.resolve()
    ):
        return _SIGNATURE_CANDIDATE_CACHE
    if not forcing_path.is_file():
        raise FileNotFoundError(f"Signature candidate forcing not found: {forcing_path}")
    no3, q = load_forcing_no3_and_q(forcing_path)
    sites = sorted(
        sid
        for sid in no3
        if len(no3[sid].dropna()) > 0 and sid in q and len(q[sid].dropna()) > 0
    )
    result = (sites, no3, q)
    if forcing_path.resolve() == SIGNATURE_CANDIDATE_FORCING.resolve():
        _SIGNATURE_CANDIDATE_CACHE = result
    return result


def target_nitrate_calendar(no3: pd.Series) -> pd.DatetimeIndex:
    s = no3.dropna()
    if s.empty:
        return pd.DatetimeIndex([])
    return pd.date_range(s.index.min(), s.index.max(), freq="D")


def stream_daily_from_raw(
    raw_parquet: Path,
    site_ids: Iterable[str],
    *,
    cache: Dict[str, pd.Series],
) -> Dict[str, pd.Series]:
    needed = sorted({canonical_usgs_site_id(s) for s in site_ids if canonical_usgs_site_id(s)})
    missing = [s for s in needed if s not in cache]
    if not missing:
        return {s: cache[s] for s in needed if s in cache}

    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
    except ImportError:
        pa = pc = ds = None

    if ds is not None:
        dataset = ds.dataset(raw_parquet, format="parquet")
        for i in range(0, len(missing), 40):
            chunk = missing[i : i + 40]
            filt = pc.is_in(pc.field("monitoring_location_id"), pa.array(chunk, type=pa.string()))
            table = dataset.to_table(
                filter=filt, columns=["monitoring_location_id", "time", "value"]
            )
            if table.num_rows == 0:
                continue
            df = table.to_pandas()
            df["monitoring_location_id"] = df["monitoring_location_id"].map(canonical_usgs_site_id)
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
            df = df.dropna(subset=["time", "value"])
            for sid, g in df.groupby("monitoring_location_id", sort=False):
                cache[str(sid)] = g.groupby("time")["value"].mean().sort_index().astype(float)
    else:
        df = pd.read_parquet(raw_parquet, columns=["monitoring_location_id", "time", "value"])
        df["monitoring_location_id"] = df["monitoring_location_id"].map(canonical_usgs_site_id)
        df = df.loc[df["monitoring_location_id"].isin(missing)]
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=["time", "value"])
        for sid, g in df.groupby("monitoring_location_id", sort=False):
            cache[str(sid)] = g.groupby("time")["value"].mean().sort_index().astype(float)

    for s in missing:
        cache.setdefault(s, pd.Series(dtype=float))
    return {s: cache[s] for s in needed}


def coverage_on_calendar(donor_obs: pd.Series, calendar: pd.DatetimeIndex) -> float:
    if len(calendar) == 0:
        return 0.0
    return float(donor_obs.reindex(calendar).notna().mean())


def _doy_climatology(obs: pd.Series) -> pd.Series:
    if obs.empty:
        return pd.Series(dtype=float)
    tmp = pd.Series(obs.to_numpy(dtype=float), index=obs.index.dayofyear)
    return tmp.groupby(level=0).mean().sort_index()


def _doy_window_mean(obs: pd.Series, doy: int, *, window: int) -> float:
    if obs.empty:
        return float("nan")
    doys = obs.index.dayofyear.to_numpy()
    vals = obs.to_numpy(dtype=float)
    deltas = np.minimum(np.abs(doys - doy), 366 - np.abs(doys - doy))
    mask = deltas <= window
    return float(np.nanmean(vals[mask])) if mask.any() else float("nan")


def fill_donor_series(
    observed: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    donor_id: str = "",
    report_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> Tuple[pd.Series, pd.Series, dict]:
    """
    Fill donor values on ``calendar`` (may be the full rectangular forcing span).

    Fill fractions in ``stats`` are restricted to ``report_window`` when provided
    (typically DNP Start_Date–End_Date / train–test native window). Outside that
    window values are still filled; they just do not enter frac_*.
    """
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize()
    methods = pd.Series("unfilled", index=calendar, dtype=object)
    obs = observed.dropna().astype(float).sort_index()
    if len(obs) and getattr(obs.index, "tz", None) is not None:
        obs.index = obs.index.tz_localize(None)
    if len(obs):
        obs.index = pd.DatetimeIndex(obs.index).normalize()
        obs = obs.groupby(level=0).mean()

    s = obs.reindex(calendar)
    if obs.empty:
        mask = pd.Series(0, index=calendar, dtype=np.int8)
        stats = {
            "donor_id": donor_id,
            "n_calendar_fill": len(calendar),
            "n_calendar": 0,
            "report_start": pd.NaT,
            "report_end": pd.NaT,
            "frac_observed": 0.0,
            "frac_interp": 0.0,
            "frac_doy_mean": 0.0,
            "frac_doy_window": 0.0,
            "frac_median": 0.0,
            "frac_unfilled": 0.0,
        }
        return s.astype(float), mask, stats

    methods.loc[s.notna()] = "observed"
    doy_mean = _doy_climatology(obs)
    median = float(obs.median())
    filled = s.copy()


    candidate = filled.interpolate(method="time")
    missing = filled.isna()
    gap_group = missing.ne(missing.shift()).cumsum()
    gap_length = missing.groupby(gap_group).transform("sum")
    short_gap = missing & gap_length.le(int(MAX_SHORT_GAP_DAYS))
    filled.loc[short_gap] = candidate.loc[short_gap]
    methods.loc[short_gap & filled.notna()] = "interp"

    for dt in filled.index[filled.isna()]:
        doy = int(dt.dayofyear)
        val = doy_mean.get(doy, np.nan)
        method = "doy_mean"
        if not (pd.notna(val) and math.isfinite(float(val))):
            val = _doy_window_mean(obs, doy, window=DOY_WINDOW_DAYS)
            method = "doy_window"
        if not (pd.notna(val) and math.isfinite(float(val))):
            val = median
            method = "median"
        filled.loc[dt] = val
        methods.loc[dt] = method

    if report_window is not None:
        w0, w1 = pd.Timestamp(report_window[0]), pd.Timestamp(report_window[1])
        report_methods = methods.loc[(methods.index >= w0) & (methods.index <= w1)]
    else:
        w0 = w1 = pd.NaT
        report_methods = methods

    n = max(len(report_methods), 1)
    vc = report_methods.value_counts()
    stats = {
        "donor_id": donor_id,
        "n_calendar_fill": int(len(methods)),
        "n_calendar": int(len(report_methods)),
        "report_start": w0,
        "report_end": w1,
        "frac_observed": float(vc.get("observed", 0)) / n,
        "frac_interp": float(vc.get("interp", 0)) / n,
        "frac_doy_mean": float(vc.get("doy_mean", 0)) / n,
        "frac_doy_window": float(vc.get("doy_window", 0)) / n,
        "frac_median": float(vc.get("median", 0)) / n,
        "frac_unfilled": float(vc.get("unfilled", 0)) / n,
    }
    mask = methods.isin(OBS_MASK_TRUSTED).astype(np.int8)
    return filled.astype(float), mask, stats


def load_dnp_report_windows(case_dir: Path) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Per-site train–test reporting window from DNP Start_Date–End_Date
    (falls back to S_Training/E_Training ∪ S_Testing/E_Testing).
    """
    dnp_files = sorted(case_dir.glob("D_N_P_nitrate_*.xlsx"))
    if not dnp_files:
        return {}
    dnp = pd.read_excel(dnp_files[0])
    site_col = "site_no" if "site_no" in dnp.columns else dnp.columns[0]
    out: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for _, row in dnp.iterrows():
        sid = canonical_usgs_site_id(row[site_col])
        if not sid:
            continue
        starts = []
        ends = []
        for c in ("Start_Date", "S_Training", "S_Testing"):
            if c in dnp.columns and pd.notna(row.get(c)):
                starts.append(pd.Timestamp(row[c]).normalize())
        for c in ("End_Date", "E_Training", "E_Testing"):
            if c in dnp.columns and pd.notna(row.get(c)):
                ends.append(pd.Timestamp(row[c]).normalize())
        if not starts or not ends:
            continue
        out[sid] = (min(starts), max(ends))
    return out


def site_forcing_calendar(train: pd.DataFrame, test: pd.DataFrame, site: str) -> pd.DatetimeIndex:
    parts = []
    for df in (train, test):
        g = df.loc[df["site_no"].map(canonical_usgs_site_id) == site, "date"]
        if len(g):
            parts.append(pd.to_datetime(g).dt.normalize())
    if not parts:
        return pd.DatetimeIndex([])
    all_dates = pd.DatetimeIndex(pd.concat(parts)).unique().sort_values()
    return pd.date_range(all_dates.min(), all_dates.max(), freq="D")


def attach_donor_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    selection: pd.DataFrame,
    val_col: str,
    mask_col: str,
    id_col: str,
    raw_cache: Dict[str, pd.Series],
    raw_parquet: Path,
    keep_sites: Optional[Set[str]] = None,
    report_windows: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for df in (train, test):
        df["site_no"] = df["site_no"].map(canonical_usgs_site_id)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    ok = selection.loc[selection["status"].astype(str) == "ok"].copy()
    ok["target_id"] = ok["target_id"].map(canonical_usgs_site_id)
    if keep_sites is not None:
        ok = ok.loc[ok["target_id"].isin(keep_sites)]
    donor_map = dict(zip(ok["target_id"].astype(str), ok["donor_id"].astype(str)))
    stream_daily_from_raw(raw_parquet, list(donor_map.values()), cache=raw_cache)

    fill_rows: List[dict] = []
    filled_by_target: Dict[str, Tuple[pd.Series, pd.Series]] = {}
    for tid, did in donor_map.items():
        cal = site_forcing_calendar(train, test, tid)
        if len(cal) == 0:
            continue
        window = None if report_windows is None else report_windows.get(tid)
        series, mask, stats = fill_donor_series(
            raw_cache.get(did, pd.Series(dtype=float)),
            cal,
            donor_id=did,
            report_window=window,
        )
        filled_by_target[tid] = (series, mask)
        fill_rows.append({"target_id": tid, **stats})

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[val_col] = np.nan
        out[mask_col] = np.int8(0)
        out[id_col] = ""
        for tid, (series, mask) in filled_by_target.items():
            m = out["site_no"] == tid
            if not m.any():
                continue
            dates = out.loc[m, "date"]
            out.loc[m, val_col] = series.reindex(dates).to_numpy()
            out.loc[m, mask_col] = mask.reindex(dates).fillna(0).astype(np.int8).to_numpy()
            out.loc[m, id_col] = donor_map[tid]
        keep = out["site_no"].isin(set(donor_map.keys()))
        return out.loc[keep].reset_index(drop=True)

    return _apply(train), _apply(test), pd.DataFrame(fill_rows)


def geographic_distance_km(
    target_id: str,
    cand_id: str,
    meta: Dict[str, dict],
    cand_geom: Optional[object] = None,
) -> float:
    """Straight-line haversine km (same ranking metric as discharge NLDI filter)."""
    if target_id not in meta:
        return float("inf")
    t = meta[target_id]
    if cand_id in meta:
        c = meta[cand_id]
        return haversine_km(t["lat"], t["lon"], c["lat"], c["lon"])
    if cand_geom is not None and pd.notna(cand_geom):
        try:
            lat, lon = parse_point_wkt(cand_geom)
            return haversine_km(t["lat"], t["lon"], lat, lon)
        except Exception:
            return float("inf")
    return float("inf")


def _pick_nearest_eligible(
    candidates: Sequence[Tuple[str, float]],
    *,
    calendar: pd.DatetimeIndex,
    raw_cache: Dict[str, pd.Series],
    raw_parquet: Path,
) -> Optional[Tuple[str, float, float]]:
    """Among (site_id, geo_km) pairs, return nearest that passes the coverage gate."""
    if not candidates:
        return None
    stream_daily_from_raw(raw_parquet, [c for c, _ in candidates], cache=raw_cache)
    ok: List[Tuple[str, float, float]] = []
    for sid, dkm in candidates:
        cov = coverage_on_calendar(raw_cache.get(sid, pd.Series(dtype=float)), calendar)
        if cov >= MIN_COVERAGE:
            ok.append((sid, dkm, cov))
    if not ok:
        return None
    ok.sort(key=lambda t: t[1])
    return ok[0]


def select_regional_donor(
    target_id: str,
    *,
    nldi: pd.DataFrame,
    meta: Dict[str, dict],
    calendar: pd.DatetimeIndex,
    raw_cache: Dict[str, pd.Series],
    raw_parquet: Path,
    donor_pool: Set[str],
) -> dict:
    """
    Regional donor = flow-connected nearest-geo, else finest HUC with eligible
    donors then nearest-geo within that HUC level.
    """
    base = {
        "target_id": target_id,
        "donor_id": "",
        "status": "fail",
        "source": "",
        "distance_km": np.nan,
        "coverage": np.nan,
        "huc_level": np.nan,
        "n_candidates_considered": 0,
        "message": "",
    }
    if len(calendar) == 0:
        base["message"] = "empty_target_calendar"
        return base
    if target_id not in meta:
        base["message"] = "target_missing_metadata"
        return base


    sub = nldi.loc[nldi["target_id"] == target_id]
    nldi_dist: Dict[str, float] = {}
    for _, row in sub.iterrows():
        cid = str(row["candidate_id"])
        if cid not in donor_pool or cid == target_id:
            continue
        dkm = geographic_distance_km(target_id, cid, meta, row.get("geometry"))
        if cid not in nldi_dist or dkm < nldi_dist[cid]:
            nldi_dist[cid] = dkm
    nldi_cands = list(nldi_dist.items())
    base["n_candidates_considered"] = len(nldi_cands)
    picked = _pick_nearest_eligible(
        nldi_cands, calendar=calendar, raw_cache=raw_cache, raw_parquet=raw_parquet
    )
    if picked is not None:
        sid, dkm, cov = picked
        return {
            **base,
            "donor_id": sid,
            "status": "ok",
            "source": "nldi_geographic",
            "distance_km": float(dkm) if math.isfinite(dkm) else np.nan,
            "coverage": float(cov),
            "message": "nldi",
        }


    thuc = meta[target_id]["huc12"]
    if not thuc:
        base["message"] = "nldi_fail_no_huc"
        return base

    def _cands_at_level(level: int) -> List[Tuple[str, float]]:
        pref = huc_prefix(thuc, level)
        if not pref:
            return []
        hits: List[Tuple[str, float]] = []
        for sid, m in meta.items():
            if sid == target_id or sid not in donor_pool:
                continue
            if not str(m["huc12"]).startswith(pref):
                continue
            hits.append((sid, geographic_distance_km(target_id, sid, meta)))
        return hits

    for level in HUC_EXPAND_LEVELS:
        picked = _pick_nearest_eligible(
            _cands_at_level(level),
            calendar=calendar,
            raw_cache=raw_cache,
            raw_parquet=raw_parquet,
        )
        if picked is not None:
            sid, dkm, cov = picked
            return {
                **base,
                "donor_id": sid,
                "status": "ok",
                "source": "huc_expand",
                "distance_km": float(dkm) if math.isfinite(dkm) else np.nan,
                "coverage": float(cov),
                "huc_level": int(level),
                "message": f"huc_expand_{level}",
            }

    for level in HUC_CONTRACT_LEVELS:
        picked = _pick_nearest_eligible(
            _cands_at_level(level),
            calendar=calendar,
            raw_cache=raw_cache,
            raw_parquet=raw_parquet,
        )
        if picked is not None:
            sid, dkm, cov = picked
            return {
                **base,
                "donor_id": sid,
                "status": "ok",
                "source": "huc_contract",
                "distance_km": float(dkm) if math.isfinite(dkm) else np.nan,
                "coverage": float(cov),
                "huc_level": int(level),
                "message": f"huc_contract_{level}",
            }

    base["message"] = "no_eligible_donor"
    return base


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0) and math.isfinite(b) else float("nan")


def _autocorr_lag(s: pd.Series, lag_days: int = 7) -> float:
    s = s.dropna()
    if len(s) < lag_days + 5:
        return float("nan")
    a = s.to_numpy(dtype=float)
    x, y = a[:-lag_days], a[lag_days:]
    if len(x) < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _flashiness(q: pd.Series) -> float:
    q = q.dropna().astype(float)
    if len(q) < 3:
        return float("nan")
    dq = np.abs(np.diff(q.to_numpy()))
    denom = float(np.nansum(q.to_numpy()))
    return float(np.nansum(dq) / denom) if denom > 0 else float("nan")


def _lyne_hollick_bfi(q: pd.Series, alpha: float = 0.925) -> float:
    vals = q.to_numpy(dtype=float)
    if len(vals) < 10 or not np.isfinite(vals).any():
        return float("nan")
    qf = np.full_like(vals, np.nan, dtype=float)
    bf = np.full_like(vals, np.nan, dtype=float)
    for i in range(1, len(vals)):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i - 1]):
            continue
        prev_qf = qf[i - 1] if np.isfinite(qf[i - 1]) else 0.0
        qf[i] = alpha * prev_qf + (1 + alpha) / 2.0 * (vals[i] - vals[i - 1])
        if qf[i] < 0:
            qf[i] = 0.0
        bf[i] = vals[i] - qf[i]
        if bf[i] < 0:
            bf[i] = 0.0
            qf[i] = vals[i]
    q_sum = float(np.nansum(vals))
    return float(np.nansum(bf) / q_sum) if q_sum > 0 else float("nan")


def _clip_series_to_period(
    s: pd.Series, period: Optional[Tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    out = s.dropna().astype(float).sort_index()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = pd.DatetimeIndex(out.index).normalize()
    out = out.groupby(level=0).mean()
    if period is None:
        return out
    start, end = period
    return out.loc[(out.index >= start) & (out.index <= end)]


def target_train_nitrate_dates(no3: pd.Series) -> pd.DatetimeIndex:
    """Exact dates with non-NaN train nitrate (not the filled daily span)."""
    s = no3.dropna().astype(float)
    if s.empty:
        return pd.DatetimeIndex([])
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return pd.DatetimeIndex(s.index).normalize().unique().sort_values()


def _huber_fit_slope_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y = a + b x with Huber regression (sklearn if available)."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    try:
        from sklearn.linear_model import HuberRegressor

        model = HuberRegressor()
        model.fit(x, y)
        return float(model.intercept_), float(model.coef_[0])
    except Exception:

        xb = np.column_stack([np.ones(len(x)), x.ravel()])
        coef, _, _, _ = np.linalg.lstsq(xb, y, rcond=None)
        return float(coef[0]), float(coef[1])


def _fit_cq_slope(df: pd.DataFrame) -> float:
    if len(df) < MIN_FLOW_GROUP_PAIRS:
        return float("nan")
    if float(df["x"].std()) < 1e-8 or float(df["y"].std()) < 1e-8:
        return 0.0
    _intercept, slope = _huber_fit_slope_intercept(df["x"].to_numpy(), df["y"].to_numpy())
    return float(slope)


def assign_cq_regime(
    *,
    b_all: float,
    b_low: float,
    b_high: float,
    variability_ratio: float,
) -> str:
    """
    Interpretable c–Q regime label (not used in Euclidean distance).

    Priority:
    1. flow_dependent — low- and high-flow slopes have opposite sign and
       both |slope| >= CQ_SLOPE_EPS
    2. chemostatic — |overall slope| < eps and variability ratio low/missing
    3. chemodynamic_weak_slope — |overall slope| < eps and variability high
    4. dilution / enrichment — overall slope sign with |slope| >= eps
    5. weak_response — residual near-zero cases
    """
    if not math.isfinite(b_all):
        return "insufficient_data"
    if (
        math.isfinite(b_low)
        and math.isfinite(b_high)
        and b_low * b_high < 0
        and abs(b_low) >= CQ_SLOPE_EPS
        and abs(b_high) >= CQ_SLOPE_EPS
    ):
        return "flow_dependent"
    if abs(b_all) < CQ_SLOPE_EPS and (
        not math.isfinite(variability_ratio) or variability_ratio <= CQ_VAR_CHEMOSTATIC
    ):
        return "chemostatic"
    if (
        abs(b_all) < CQ_SLOPE_EPS
        and math.isfinite(variability_ratio)
        and variability_ratio > CQ_VAR_CHEMOSTATIC
    ):
        return "chemodynamic_weak_slope"
    if b_all <= -CQ_SLOPE_EPS:
        return "dilution"
    if b_all >= CQ_SLOPE_EPS:
        return "enrichment"
    return "weak_response"


def compute_cq_features(
    donor_no3: pd.Series,
    donor_q: pd.Series,
    target_train_dates: pd.DatetimeIndex,
) -> dict:
    """
    Target-specific composite c–Q descriptors on the target's exact train nitrate dates.

    Returns continuous CQ_FEATURES plus cq_regime / cq_n_pairs for meta.
    """
    nan_out = {
        "cq_slope_all": float("nan"),
        "cq_slope_low": float("nan"),
        "cq_slope_high": float("nan"),
        "cq_r2_all": float("nan"),
        "cq_variability_ratio": float("nan"),
        "cq_regime": "insufficient_data",
        "cq_n_pairs": 0,
        "cq_q50": float("nan"),
    }
    if len(target_train_dates) == 0:
        return nan_out

    dates = pd.DatetimeIndex(pd.to_datetime(target_train_dates)).normalize().unique()
    c = _clip_series_to_period(donor_no3, None).reindex(dates)
    q = _clip_series_to_period(donor_q, None).reindex(dates)
    paired = pd.concat([c.rename("C"), q.rename("Q")], axis=1).dropna()
    paired = paired.loc[(paired["C"] >= 0) & (paired["Q"] > 0)]
    n_pairs = int(len(paired))
    if n_pairs < MIN_TOTAL_PAIRS:
        out = dict(nan_out)
        out["cq_n_pairs"] = n_pairs
        return out

    q50 = float(paired["Q"].median())
    if not math.isfinite(q50) or q50 <= 0:
        out = dict(nan_out)
        out["cq_n_pairs"] = n_pairs
        return out

    paired = paired.copy()
    paired["x"] = np.log(paired["Q"] / q50)
    paired["y"] = np.log1p(paired["C"])

    intercept, b_all = _huber_fit_slope_intercept(paired["x"].to_numpy(), paired["y"].to_numpy())
    pred = intercept + b_all * paired["x"].to_numpy()
    ss_res = float(np.sum((paired["y"].to_numpy() - pred) ** 2))
    ss_tot = float(np.sum((paired["y"].to_numpy() - float(paired["y"].mean())) ** 2))
    r2_all = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    low = paired.loc[paired["Q"] <= q50]
    high = paired.loc[paired["Q"] > q50]
    b_low = _fit_cq_slope(low)
    b_high = _fit_cq_slope(high)

    iqr_y = float(paired["y"].quantile(0.75) - paired["y"].quantile(0.25))
    iqr_x = float(paired["x"].quantile(0.75) - paired["x"].quantile(0.25))
    variability_ratio = iqr_y / iqr_x if iqr_x > 0 else float("nan")

    regime = assign_cq_regime(
        b_all=b_all, b_low=b_low, b_high=b_high, variability_ratio=variability_ratio
    )
    return {
        "cq_slope_all": float(b_all),
        "cq_slope_low": float(b_low) if math.isfinite(b_low) else float("nan"),
        "cq_slope_high": float(b_high) if math.isfinite(b_high) else float("nan"),
        "cq_r2_all": float(r2_all) if math.isfinite(r2_all) else float("nan"),
        "cq_variability_ratio": (
            float(variability_ratio) if math.isfinite(variability_ratio) else float("nan")
        ),
        "cq_regime": regime,
        "cq_n_pairs": n_pairs,
        "cq_q50": q50,
    }


def build_signature_feature_table(
    *,
    site_ids: Sequence[str],
    no3: Dict[str, pd.Series],
    q: Dict[str, pd.Series],
    attr: pd.DataFrame,
    period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    target_train_dates: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Build signature features for site_ids.

    If period is set, nitrate and dynamic Q features use series clipped to that window.
    If target_train_dates is set, c–Q features are computed by reindexing each site's
    C and Q onto those exact dates (target-specific composite c–Q).
    Static watershed/drainage/tile attrs always come from the full attribute table.
    """
    attr = attr.copy()
    attr["site_no"] = attr["site_no"].map(canonical_usgs_site_id)
    attr = attr.drop_duplicates("site_no", keep="last").set_index("site_no")
    rows = []
    for sid in site_ids:
        n = _clip_series_to_period(no3.get(sid, pd.Series(dtype=float)), period)
        qq_full = q.get(sid, pd.Series(dtype=float))
        qq = _clip_series_to_period(qq_full, period)

        mean_n = float(n.mean()) if len(n) else float("nan")
        std_n = float(n.std()) if len(n) else float("nan")
        mean_log = float(np.log1p(n.clip(lower=0)).mean()) if len(n) else float("nan")
        q_obs = qq.dropna()
        mean_q = float(q_obs.mean()) if len(q_obs) else float("nan")
        area = (
            float(attr.loc[sid, "basinatlas_area"])
            if sid in attr.index and "basinatlas_area" in attr.columns
            else float("nan")
        )
        area_km2 = area / 1e6 if math.isfinite(area) and area > 1e6 else area

        if target_train_dates is not None and len(target_train_dates):
            cq = compute_cq_features(
                no3.get(sid, pd.Series(dtype=float)),
                qq_full,
                target_train_dates,
            )
        else:
            cq = {
                "cq_slope_all": float("nan"),
                "cq_slope_low": float("nan"),
                "cq_slope_high": float("nan"),
                "cq_r2_all": float("nan"),
                "cq_variability_ratio": float("nan"),
                "cq_regime": "insufficient_data",
                "cq_n_pairs": 0,
                "cq_q50": float("nan"),
            }

        row = {
            "site_no": sid,
            "cv_NO3": _safe_div(std_n, mean_n),
            "NO3_autocorr_7day": _autocorr_lag(n),
            "mean_log1p_NO3": mean_log,
            "cq_slope_all": cq["cq_slope_all"],
            "cq_slope_low": cq["cq_slope_low"],
            "cq_slope_high": cq["cq_slope_high"],
            "cq_r2_all": cq["cq_r2_all"],
            "cq_variability_ratio": cq["cq_variability_ratio"],
            "cq_regime": cq["cq_regime"],
            "cq_n_pairs": cq["cq_n_pairs"],
            "cq_q50": cq["cq_q50"],
            "cv_Q": _safe_div(float(q_obs.std()) if len(q_obs) else float("nan"), mean_q),
            "flashiness_index": _flashiness(qq),
            "log_mean_Q": float(np.log1p(max(mean_q, 0))) if math.isfinite(mean_q) else float("nan"),
            "baseflow_index": _lyne_hollick_bfi(qq),
            "log_drainage_area": (
                float(np.log1p(max(area_km2, 0))) if math.isfinite(area_km2) else float("nan")
            ),
            "tile_drain_fraction": (
                float(attr.loc[sid, "tile_drain_fraction"])
                if sid in attr.index and "tile_drain_fraction" in attr.columns
                else float("nan")
            ),
        }
        for feat in SIGNATURE_FEATURES["watershed"]:
            row[feat] = (
                float(attr.loc[sid, feat])
                if sid in attr.index and feat in attr.columns
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("site_no")


def _zscore_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        mu = float(x.mean(skipna=True))
        sd = float(x.std(skipna=True))
        out[c] = 0.0 if (not math.isfinite(sd) or sd < 1e-12) else (x - mu) / sd
    return out


def pairwise_signature_distance(zdf: pd.DataFrame, sid_a: str, sid_b: str) -> float:
    if sid_a not in zdf.index or sid_b not in zdf.index:
        return float("inf")
    total_w = 0.0
    acc = 0.0
    for group, feats in SIGNATURE_FEATURES.items():
        w = float(GROUP_WEIGHTS.get(group, 1.0))
        xa = zdf.loc[sid_a, feats].to_numpy(dtype=float)
        xb = zdf.loc[sid_b, feats].to_numpy(dtype=float)
        mask = np.isfinite(xa) & np.isfinite(xb)
        if not mask.any():
            continue
        d = float(np.sqrt(np.nansum((xa[mask] - xb[mask]) ** 2)))
        d = d / math.sqrt(float(mask.sum()))
        acc += w * d
        total_w += w
    return acc / total_w if total_w > 0 else float("inf")


def select_signature_donor(
    target_id: str,
    *,
    zdf: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    raw_cache: Dict[str, pd.Series],
    raw_parquet: Path,
    donor_pool: Set[str],
) -> dict:
    base = {
        "target_id": target_id,
        "donor_id": "",
        "status": "fail",
        "source": "signature",
        "signature_distance": np.nan,
        "coverage": np.nan,
        "n_candidates_considered": 0,
        "period_start": calendar.min() if len(calendar) else pd.NaT,
        "period_end": calendar.max() if len(calendar) else pd.NaT,
        "message": "",
    }
    if len(calendar) == 0:
        base["message"] = "empty_target_calendar"
        return base
    if target_id not in zdf.index:
        base["message"] = "target_missing_features"
        return base

    cands = [s for s in zdf.index.tolist() if s in donor_pool and s != target_id]
    ranked = sorted(
        ((s, pairwise_signature_distance(zdf, target_id, s)) for s in cands),
        key=lambda t: t[1],
    )
    base["n_candidates_considered"] = len(ranked)
    stream_daily_from_raw(raw_parquet, [s for s, _ in ranked], cache=raw_cache)
    for sid, dist in ranked:
        cov = coverage_on_calendar(raw_cache.get(sid, pd.Series(dtype=float)), calendar)
        if cov >= MIN_COVERAGE:
            return {
                **base,
                "donor_id": sid,
                "status": "ok",
                "signature_distance": float(dist) if math.isfinite(dist) else np.nan,
                "coverage": float(cov),
                "message": "signature_euclidean",
            }
    base["message"] = "no_eligible_donor"
    return base


def patch_lstm_config(case_dir: Path, *, track: str, train_name: str, test_name: str) -> None:
    cfgs = sorted(case_dir.glob("lstm_config_*.yaml"))
    if track == "regional":
        extra = [REG_VAL, REG_MASK]
        norms = {REG_VAL: "log1p", REG_MASK: "none"}
        scopes = {REG_VAL: "per_site"}
        drop = {SIG_VAL, SIG_MASK}
    else:
        extra = [SIG_VAL, SIG_MASK]
        norms = {SIG_VAL: "log1p", SIG_MASK: "none"}
        scopes = {SIG_VAL: "per_site"}
        drop = {REG_VAL, REG_MASK}

    for cfg_path in cfgs:
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if not isinstance(cfg, dict):
            continue
        cfg["forcing_path_train"] = train_name
        cfg["forcing_path_test"] = test_name
        forcings = [c for c in list(cfg.get("forcings") or []) if c not in drop]
        for c in extra:
            if c not in forcings:
                forcings.append(c)
        cfg["forcings"] = forcings
        norm = cfg.setdefault("normalization", {})
        fn = norm.setdefault("forcings", {})
        sc = norm.setdefault("scope", {}).setdefault("forcings", {})
        for k, v in norms.items():
            fn[k] = v
        for k, v in scopes.items():
            sc[k] = v
        for k in drop:
            fn.pop(k, None)
            sc.pop(k, None)
        with cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)


def archive_case(case_dir: Path) -> None:
    """Copy model inputs (train/test/yaml/DNP) to Paper_1A/Inputs; refresh dest."""
    if not PAPER_INPUTS_ARCHIVE.is_dir():
        return
    dest = PAPER_INPUTS_ARCHIVE / case_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "forcing_*_train.parquet",
        "forcing_*_test.parquet",
        "lstm_config_*.yaml",
        "D_N_P_nitrate_*.xlsx",
    ):
        for src in case_dir.glob(pattern):
            shutil.copy2(src, dest / src.name)


def meta_dir_for_pair(pair_key: str, meta_root: Optional[Path] = None) -> Path:
    root = meta_root if meta_root is not None else SIMILARITY_META_ROOT
    d = root / pair_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def case_pair_key(tag: str) -> str:
    """Strip scenario suffix so siblings share one comparison key."""
    for suf in (
        "_REGSIGALLSENSOR",
        "_REGSIGDONOR",
        "_REDUCEDSENSOR",
        "_ALLSENSOR",
        "_REGDONOR",
        "_SIGDONOR",
        "_BASELINE",
        "_NODONOR",
    ):
        if tag.endswith(suf):
            return tag[: -len(suf)]
    return tag


def parse_case_set_and_gap(pair_key: str) -> Tuple[str, str]:
    """Extract (case_set, gap) from pair keys like R20B14_C1_Seed21."""
    parts = pair_key.split("_")
    gap = parts[0] if parts else ""
    case_set = "C1"
    for p in parts:
        if p in {"C1", "C2", "C3", "C4"}:
            case_set = p
            break
    return case_set, gap


def resolve_sibling_case(inputs_root: Path, pair_key: str, scenario: str) -> Optional[DonorCase]:
    case_dir = inputs_root / f"{pair_key}_{scenario}"
    if not case_dir.is_dir():
        return None
    trains = sorted(case_dir.glob("forcing_*_train.parquet"))
    tests = sorted(case_dir.glob("forcing_*_test.parquet"))
    if not trains or not tests:
        return None
    track = "regional" if "REG" in scenario and "SIG" not in scenario.replace("REGSIG", "") else (
        "signature" if scenario.startswith("SIG") else "other"
    )
    if scenario in {"REGDONOR"}:
        track = "regional"
    elif scenario in {"SIGDONOR"}:
        track = "signature"
    else:
        track = "other"
    return DonorCase(
        case_dir=case_dir,
        track=track,
        tag=case_dir.name,
        train_parquet=trains[0],
        test_parquet=tests[0],
    )


def selection_with_population_fallback(
    train_sel: pd.DataFrame,
    pop_sel: Optional[pd.DataFrame],
    keep_sites: Set[str],
) -> pd.DataFrame:
    """Prefer train-only ok rows; fill missing keep_sites from seed-independent pop sel."""
    rows = []
    train_ok = {
        canonical_usgs_site_id(r["target_id"]): r
        for _, r in train_sel.iterrows()
        if str(r.get("status", "")) == "ok"
    }
    pop_ok = {}
    if pop_sel is not None and not pop_sel.empty:
        pop_ok = {
            canonical_usgs_site_id(r["target_id"]): r
            for _, r in pop_sel.iterrows()
            if str(r.get("status", "")) == "ok"
        }
    for sid in sorted(keep_sites):
        if sid in train_ok:
            rows.append(dict(train_ok[sid]))
        elif sid in pop_ok:
            r = dict(pop_ok[sid])
            r["message"] = f"fallback_population:{r.get('message', '')}"
            rows.append(r)
        else:
            rows.append(
                {
                    "target_id": sid,
                    "donor_id": "",
                    "status": "fail",
                    "message": "missing_train_and_population",
                }
            )
    return pd.DataFrame(rows)


def rewrite_case_to_sites(case: DonorCase, keep_sites: Set[str]) -> None:
    """Filter train/test/DNP to keep_sites without attaching donors."""
    train = pd.read_parquet(case.train_parquet)
    test = pd.read_parquet(case.test_parquet)
    for df in (train, test):
        df["site_no"] = df["site_no"].map(canonical_usgs_site_id)
    train = train.loc[train["site_no"].isin(keep_sites)].reset_index(drop=True)
    test = test.loc[test["site_no"].isin(keep_sites)].reset_index(drop=True)
    train.to_parquet(case.train_parquet, index=False)
    test.to_parquet(case.test_parquet, index=False)
    dnp_files = sorted(case.case_dir.glob("D_N_P_nitrate_*.xlsx"))
    if dnp_files:
        dnp = pd.read_excel(dnp_files[0])
        site_col = "site_no" if "site_no" in dnp.columns else dnp.columns[0]
        dnp[site_col] = dnp[site_col].map(canonical_usgs_site_id)
        dnp.loc[dnp[site_col].isin(keep_sites)].reset_index(drop=True).to_excel(
            dnp_files[0], index=False
        )
    print(
        f"  [{case.tag}] site-filter -> {train['site_no'].nunique()} sites",
        flush=True,
    )


def patch_lstm_config_dual(case_dir: Path, *, train_name: str, test_name: str) -> None:
    """Ensure both regional and signature donor forcings are listed."""
    extra = [REG_VAL, REG_MASK, SIG_VAL, SIG_MASK]
    norms = {REG_VAL: "log1p", REG_MASK: "none", SIG_VAL: "log1p", SIG_MASK: "none"}
    scopes = {REG_VAL: "per_site", SIG_VAL: "per_site"}
    for cfg_path in sorted(case_dir.glob("lstm_config_*.yaml")):
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if not isinstance(cfg, dict):
            continue
        cfg["forcing_path_train"] = train_name
        cfg["forcing_path_test"] = test_name
        forcings = list(cfg.get("forcings") or [])
        for c in extra:
            if c not in forcings:
                forcings.append(c)
        cfg["forcings"] = forcings
        norm = cfg.setdefault("normalization", {})
        fn = norm.setdefault("forcings", {})
        sc = norm.setdefault("scope", {}).setdefault("forcings", {})
        for k, v in norms.items():
            fn[k] = v
        for k, v in scopes.items():
            sc[k] = v
        with cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)


def write_dual_donor_attach(
    case: DonorCase,
    *,
    reg_selection: pd.DataFrame,
    sig_selection: pd.DataFrame,
    keep_sites: Set[str],
    raw_parquet: Path,
    raw_cache: Dict[str, pd.Series],
    meta_out: Path,
) -> None:
    """Attach both REG and SIG donor columns; keep keep_sites only."""
    report_windows = load_dnp_report_windows(case.case_dir)
    train = pd.read_parquet(case.train_parquet)
    test = pd.read_parquet(case.test_parquet)
    train, test, fill_reg = attach_donor_columns(
        train,
        test,
        selection=reg_selection,
        val_col=REG_VAL,
        mask_col=REG_MASK,
        id_col=REG_ID,
        raw_cache=raw_cache,
        raw_parquet=raw_parquet,
        keep_sites=keep_sites,
        report_windows=report_windows,
    )


    reg_cols = [REG_VAL, REG_MASK, REG_ID]
    reg_train_stash = train[["site_no", "date"] + reg_cols].copy()
    reg_test_stash = test[["site_no", "date"] + reg_cols].copy()
    train2, test2, fill_sig = attach_donor_columns(
        train.drop(columns=reg_cols, errors="ignore"),
        test.drop(columns=reg_cols, errors="ignore"),
        selection=sig_selection,
        val_col=SIG_VAL,
        mask_col=SIG_MASK,
        id_col=SIG_ID,
        raw_cache=raw_cache,
        raw_parquet=raw_parquet,
        keep_sites=keep_sites,
        report_windows=report_windows,
    )
    train2 = train2.drop(columns=reg_cols, errors="ignore").merge(
        reg_train_stash, on=["site_no", "date"], how="left"
    )
    test2 = test2.drop(columns=reg_cols, errors="ignore").merge(
        reg_test_stash, on=["site_no", "date"], how="left"
    )
    train2.to_parquet(case.train_parquet, index=False)
    test2.to_parquet(case.test_parquet, index=False)
    fill_reg.to_csv(meta_out / f"donor_fill_stats_regional_{case.tag}.csv", index=False)
    fill_sig.to_csv(meta_out / f"donor_fill_stats_signature_{case.tag}.csv", index=False)
    dnp_files = sorted(case.case_dir.glob("D_N_P_nitrate_*.xlsx"))
    if dnp_files:
        dnp = pd.read_excel(dnp_files[0])
        site_col = "site_no" if "site_no" in dnp.columns else dnp.columns[0]
        dnp[site_col] = dnp[site_col].map(canonical_usgs_site_id)
        dnp.loc[dnp[site_col].isin(keep_sites)].reset_index(drop=True).to_excel(
            dnp_files[0], index=False
        )
    patch_lstm_config_dual(
        case.case_dir,
        train_name=case.train_parquet.name,
        test_name=case.test_parquet.name,
    )
    archive_case(case.case_dir)
    print(
        f"  [{case.tag}] dual donors attached  sites={train2['site_no'].nunique()}",
        flush=True,
    )


def ok_donor_sites(selection: pd.DataFrame) -> Set[str]:
    ok = selection.loc[selection["status"].astype(str) == "ok", "target_id"]
    return {canonical_usgs_site_id(x) for x in ok if canonical_usgs_site_id(x)}


def resolve_partner_case(inputs_root: Path, pair_key: str, track: str) -> Optional[DonorCase]:
    suffix = "_REGDONOR" if track == "regional" else "_SIGDONOR"
    partner_dir = inputs_root / f"{pair_key}{suffix}"
    if not partner_dir.is_dir():
        return None
    trains = sorted(partner_dir.glob("forcing_*_train.parquet"))
    tests = sorted(partner_dir.glob("forcing_*_test.parquet"))
    if not trains or not tests:
        return None
    return DonorCase(
        case_dir=partner_dir,
        track=track,
        tag=partner_dir.name,
        train_parquet=trains[0],
        test_parquet=tests[0],
    )


def select_regional_for_case(
    case: DonorCase,
    *,
    nldi: pd.DataFrame,
    meta: Dict[str, dict],
    raw_parquet: Path,
    raw_cache: Dict[str, pd.Series],
    donor_pool: Set[str],
    meta_out: Path,
) -> pd.DataFrame:
    no3, _q = load_forcing_no3_and_q(case.train_parquet)
    targets = sorted(s for s, ser in no3.items() if len(ser.dropna()))
    print(f"  [{case.tag}] regional selection targets={len(targets)}", flush=True)
    rows = [
        select_regional_donor(
            tid,
            nldi=nldi,
            meta=meta,
            calendar=target_nitrate_calendar(no3[tid]),
            raw_cache=raw_cache,
            raw_parquet=raw_parquet,
            donor_pool=donor_pool,
        )
        for tid in targets
    ]
    selection = pd.DataFrame(rows)
    sel_name = f"donor_selection_regional_{case.tag}.csv"
    selection.to_csv(meta_out / sel_name, index=False)
    print(
        f"  [{case.tag}] regional ok={len(ok_donor_sites(selection))} / {len(selection)} -> "
        f"{meta_out.name}/{sel_name}",
        flush=True,
    )
    return selection


def select_signature_for_case(
    case: DonorCase,
    *,
    attr: pd.DataFrame,
    raw_parquet: Path,
    raw_cache: Dict[str, pd.Series],
    donor_pool: Set[str],
    meta_out: Path,
) -> pd.DataFrame:
    """
    Signature selection with period-restricted dynamic features and target-specific c–Q.

    Targets come from ``case.train_parquet``. Signature *candidates* come from
    ``forcing_117_V1.parquet`` (full network), not from the case's retained target
    subset — so C3 can still draw donors from the 117-site pool.

    For each target T:
      * Target nitrate features use ``no3_train[T]`` (masked train record).
      * Candidate nitrate uses **raw** donor nitrate restricted to T's train period.
      * Dynamic Q features use network/train Q clipped to T's train period;
        c–Q reindexes C,Q onto T's exact train nitrate dates.
      * Static attrs unrestricted.
    Coverage gate and gap-fill still use observed (raw) donor nitrate on the
    target calendar.
    """
    no3_train, q_train = load_forcing_no3_and_q(case.train_parquet)
    targets = sorted(s for s, ser in no3_train.items() if len(ser.dropna()))

    cand_ids, _cand_no3, cand_q = load_signature_candidate_series()
    cohort = sorted(set(cand_ids) | set(targets))
    print(
        f"  [{case.tag}] signature selection targets={len(targets)}  "
        f"candidates={len(cand_ids)} (from {SIGNATURE_CANDIDATE_FORCING.name})  "
        f"(target NO3=train; candidate NO3=raw on target train dates)",
        flush=True,
    )


    stream_daily_from_raw(raw_parquet, cohort, cache=raw_cache)
    q_all: Dict[str, pd.Series] = dict(cand_q)
    q_all.update(q_train)
    for sid in cohort:
        q_all.setdefault(sid, pd.Series(dtype=float))

    all_feats = [f for feats in SIGNATURE_FEATURES.values() for f in feats]
    sel_rows: List[dict] = []
    feat_snapshots: List[pd.DataFrame] = []

    for tid in targets:
        calendar = target_nitrate_calendar(no3_train[tid])
        train_dates = target_train_nitrate_dates(no3_train[tid])
        if len(calendar) == 0 or len(train_dates) == 0:
            sel_rows.append(
                {
                    "target_id": tid,
                    "donor_id": "",
                    "status": "fail",
                    "source": "signature",
                    "signature_distance": np.nan,
                    "coverage": np.nan,
                    "n_candidates_considered": 0,
                    "period_start": pd.NaT,
                    "period_end": pd.NaT,
                    "n_train_nitrate_dates": int(len(train_dates)),
                    "message": "empty_target_calendar",
                }
            )
            continue


        no3_for_feats: Dict[str, pd.Series] = {}
        for sid in cohort:
            if sid == tid:
                no3_for_feats[sid] = no3_train[tid]
            else:
                no3_for_feats[sid] = raw_cache.get(sid, pd.Series(dtype=float))

        period = (pd.Timestamp(calendar.min()), pd.Timestamp(calendar.max()))
        ftab = build_signature_feature_table(
            site_ids=cohort,
            no3=no3_for_feats,
            q=q_all,
            attr=attr,
            period=period,
            target_train_dates=train_dates,
        )
        zdf = _zscore_cols(ftab, all_feats)
        snap = ftab.reset_index()
        snap.insert(0, "feature_target_id", tid)
        snap.insert(1, "period_start", period[0])
        snap.insert(2, "period_end", period[1])
        snap.insert(3, "n_train_nitrate_dates", int(len(train_dates)))
        feat_snapshots.append(snap)
        sel = select_signature_donor(
            tid,
            zdf=zdf,
            calendar=calendar,
            raw_cache=raw_cache,
            raw_parquet=raw_parquet,
            donor_pool=donor_pool,
        )
        sel["n_train_nitrate_dates"] = int(len(train_dates))
        if sel.get("status") == "ok" and sel.get("donor_id") in ftab.index:
            did = str(sel["donor_id"])
            sel["donor_cq_regime"] = ftab.loc[did, "cq_regime"]
            sel["target_cq_regime"] = ftab.loc[tid, "cq_regime"] if tid in ftab.index else ""
            sel["donor_cq_n_pairs"] = ftab.loc[did, "cq_n_pairs"]
        sel_rows.append(sel)

    selection = pd.DataFrame(sel_rows)
    sel_name = f"donor_selection_signature_{case.tag}.csv"
    selection.to_csv(meta_out / sel_name, index=False)
    if feat_snapshots:
        feats_path = meta_out / f"signature_features_{case.tag}.parquet"
        pd.concat(feat_snapshots, ignore_index=True).to_parquet(feats_path, index=False)
        print(f"  [{case.tag}] signature features -> {meta_out.name}/{feats_path.name}", flush=True)
    print(
        f"  [{case.tag}] signature ok={len(ok_donor_sites(selection))} / {len(selection)} -> "
        f"{meta_out.name}/{sel_name}",
        flush=True,
    )
    return selection


def write_common_site_reports(
    *,
    pair_key: str,
    meta_out: Path,
    reg_sites: Set[str],
    sig_sites: Set[str],
    common: Sequence[str],
) -> None:
    common_df = pd.DataFrame({"site_no": list(common)})
    out_name = f"donor_common_sites_{pair_key}.csv"
    common_df.to_csv(meta_out / out_name, index=False)

    detail = pd.DataFrame({"site_no": sorted(set(reg_sites) | set(sig_sites) | set(common))})
    detail["in_regional"] = detail["site_no"].isin(reg_sites)
    detail["in_signature"] = detail["site_no"].isin(sig_sites)
    detail["in_seed_common"] = detail["in_regional"] & detail["in_signature"]
    detail["in_fixed_common"] = detail["site_no"].isin(set(common))
    detail_name = f"donor_site_membership_{pair_key}.csv"
    detail.to_csv(meta_out / detail_name, index=False)

    print(
        f"  common pool: {len(common)}  "
        f"(regional_ok={len(reg_sites)}  signature_ok={len(sig_sites)}  "
        f"reg_only={len(reg_sites - sig_sites)}  sig_only={len(sig_sites - reg_sites)})",
        flush=True,
    )
    print(f"  Wrote {meta_out}/{out_name} and {detail_name}", flush=True)


def _safe_plot_filename(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", str(s))


def _target_obs_span(
    train_obs: pd.DataFrame, test_obs: pd.DataFrame
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    dates: List[pd.Timestamp] = []
    for df in (train_obs, test_obs):
        if df is None or df.empty:
            continue
        dates.extend(pd.to_datetime(df["date"]).tolist())
    if not dates:
        return None
    return min(dates), max(dates)


def _pooled_zscore_params(*arrays: np.ndarray) -> Tuple[float, float]:
    """Mean/std from pooled finite values (target + donors) for plot z-scoring."""
    chunks = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
    if not chunks:
        return 0.0, 1.0
    vals = np.concatenate(chunks)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 1.0
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0))
    if not math.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return mu, sd


def _zscore(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (np.asarray(x, dtype=float) - mu) / sd


def _site_q_series(df: pd.DataFrame, site: str) -> pd.Series:
    g = df.loc[df["site_no"] == site, ["date", "discharge_cfs"]].dropna()
    if g.empty:
        return pd.Series(dtype=float)
    g = g.sort_values("date")
    return pd.Series(
        g["discharge_cfs"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(g["date"]).normalize(),
    ).groupby(level=0).mean()


def _series_to_iso_pairs(s: pd.Series) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    if s is None or s.empty:
        return (), ()
    s = s.sort_index()
    idx = pd.DatetimeIndex(s.index).normalize()
    return (
        tuple(ts.isoformat() for ts in idx),
        tuple(float(x) if np.isfinite(x) else np.nan for x in s.to_numpy(dtype=float)),
    )


def _obs_df_to_iso_pairs(df: pd.DataFrame) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    if df is None or df.empty:
        return (), ()
    g = df.sort_values("date")
    dates = pd.to_datetime(g["date"]).dt.normalize()
    vals = g["nitrate_mg_l"].to_numpy(dtype=float)
    return (tuple(ts.isoformat() for ts in dates), tuple(float(x) for x in vals))


def _dual_donor_site_plot_worker(job: DualDonorSitePlotJob) -> int:
    dates = pd.DatetimeIndex(job.dates_iso)
    regional = pd.Series(job.regional_vals, index=dates)
    signature = pd.Series(job.signature_vals, index=dates)
    train_obs = (
        pd.DataFrame({"date": pd.to_datetime(list(job.train_obs_dates)), "nitrate_mg_l": list(job.train_obs_vals)})
        if job.train_obs_dates
        else pd.DataFrame(columns=["date", "nitrate_mg_l"])
    )
    test_obs = (
        pd.DataFrame({"date": pd.to_datetime(list(job.test_obs_dates)), "nitrate_mg_l": list(job.test_obs_vals)})
        if job.test_obs_dates
        else pd.DataFrame(columns=["date", "nitrate_mg_l"])
    )
    target_q = pd.Series(
        job.target_q_vals,
        index=pd.DatetimeIndex(pd.to_datetime(list(job.target_q_dates))),
    )
    regional_q = pd.Series(
        job.regional_q_vals,
        index=pd.DatetimeIndex(pd.to_datetime(list(job.regional_q_dates))),
    )
    signature_q = pd.Series(
        job.signature_q_vals,
        index=pd.DatetimeIndex(pd.to_datetime(list(job.signature_q_dates))),
    )
    return plot_dual_donor_site(
        site_no=job.site,
        dates=dates,
        regional=regional,
        signature=signature,
        train_obs=train_obs,
        test_obs=test_obs,
        target_q=target_q,
        regional_q=regional_q,
        signature_q=signature_q,
        out_dir=job.site_dir,
        regional_donor_id=job.regional_donor_id,
        signature_donor_id=job.signature_donor_id,
    )


def plot_dual_donor_site(
    *,
    site_no: str,
    dates: pd.DatetimeIndex,
    regional: pd.Series,
    signature: pd.Series,
    train_obs: pd.DataFrame,
    test_obs: pd.DataFrame,
    target_q: pd.Series,
    regional_q: pd.Series,
    signature_q: pd.Series,
    out_dir: Path,
    regional_donor_id: str = "",
    signature_donor_id: str = "",
) -> int:
    """
    Write two separate PNGs under ``out_dir`` (typically .../donor_nitrate_plots/{site}/):
      * nitrate_timeseries.png — raw C (mg/L)
      * cq_scatter.png — z-scored C vs z-scored Q (pooled μ,σ over target + donors)
    Returns number of files written (0–2).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span = _target_obs_span(train_obs, test_obs)
    if span is None:
        return 0
    t0, t1 = span
    mask = (dates >= t0) & (dates <= t1)
    if not mask.any():
        return 0
    d = dates[mask]
    reg = regional.reindex(d)
    sig = signature.reindex(d)

    color_reg = "#1f77b4"
    color_sig = "#8c2d2d"
    color_target = "#2ca02c"
    color_train = "#2ca02c"
    color_test = "#ff7f0e"

    tr = train_obs.sort_values("date") if not train_obs.empty else train_obs
    te = test_obs.sort_values("date") if not test_obs.empty else test_obs
    tr_c = tr["nitrate_mg_l"].to_numpy(dtype=float) if not tr.empty else np.array([])
    te_c = te["nitrate_mg_l"].to_numpy(dtype=float) if not te.empty else np.array([])

    reg_lab = (regional_donor_id or "regional").strip() or "regional"
    sig_lab = (signature_donor_id or "signature").strip() or "signature"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0


    fig, ax_ts = plt.subplots(figsize=(12, 4.8))
    ax_ts.plot(
        d,
        reg.to_numpy(dtype=float),
        color=color_reg,
        lw=1.6,
        label=f"{reg_lab} (regional)",
        zorder=2,
    )
    ax_ts.plot(
        d,
        sig.to_numpy(dtype=float),
        color=color_sig,
        lw=1.6,
        label=f"{sig_lab} (signature)",
        zorder=2,
    )
    if not tr.empty:
        ax_ts.scatter(
            pd.to_datetime(tr["date"]),
            tr_c,
            s=14,
            c=color_train,
            alpha=0.85,
            label=f"train (n={len(tr)})",
            zorder=4,
            edgecolors="none",
        )
    if not te.empty:
        ax_ts.scatter(
            pd.to_datetime(te["date"]),
            te_c,
            s=16,
            c=color_test,
            alpha=0.9,
            label=f"test gap (n={len(te)})",
            zorder=5,
            edgecolors="none",
        )
    ax_ts.set_title(site_no, fontsize=11)
    ax_ts.set_xlim(t0, t1)
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("Nitrate (mg/L as N)")
    ax_ts.grid(True, alpha=0.25)
    ax_ts.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    ts_path = out_dir / "nitrate_timeseries.png"
    fig.savefig(ts_path, dpi=150)
    plt.close(fig)
    n_written += 1


    def _cq_xy(c_series: pd.Series, q_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        if c_series is None or q_series is None or len(c_series) == 0 or len(q_series) == 0:
            return np.array([]), np.array([])
        aligned = pd.DataFrame({"C": c_series.astype(float), "Q": q_series.astype(float)}).dropna()
        aligned = aligned.loc[(aligned["C"] >= 0) & (aligned["Q"] > 0)]
        aligned = aligned.loc[(aligned.index >= t0) & (aligned.index <= t1)]
        if aligned.empty:
            return np.array([]), np.array([])
        return aligned["Q"].to_numpy(dtype=float), aligned["C"].to_numpy(dtype=float)

    tgt_parts = []
    for part in (tr, te):
        if part is None or part.empty:
            continue
        tmp = part.copy()
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()
        tmp = tmp.set_index("date")
        tmp["Q"] = target_q.reindex(tmp.index)
        tgt_parts.append(tmp[["nitrate_mg_l", "Q"]].rename(columns={"nitrate_mg_l": "C"}))
    if tgt_parts:
        tgt = pd.concat(tgt_parts).dropna()
        tgt = tgt.loc[(tgt["C"] >= 0) & (tgt["Q"] > 0)]
        q_t, c_t = tgt["Q"].to_numpy(dtype=float), tgt["C"].to_numpy(dtype=float)
    else:
        q_t, c_t = np.array([]), np.array([])

    q_r, c_r = _cq_xy(reg, regional_q)
    q_s, c_s = _cq_xy(sig, signature_q)

    mu_c, sd_c = _pooled_zscore_params(c_t, c_r, c_s)
    mu_q, sd_q = _pooled_zscore_params(q_t, q_r, q_s)

    fig, ax_cq = plt.subplots(figsize=(8.5, 6.2))
    if len(q_t):
        ax_cq.scatter(
            _zscore(q_t, mu_q, sd_q),
            _zscore(c_t, mu_c, sd_c),
            s=18,
            c=color_target,
            alpha=0.75,
            label=f"target (n={len(q_t)})",
            zorder=4,
            edgecolors="none",
        )
    if len(q_r):
        ax_cq.scatter(
            _zscore(q_r, mu_q, sd_q),
            _zscore(c_r, mu_c, sd_c),
            s=14,
            c=color_reg,
            alpha=0.55,
            label=f"{reg_lab} regional (n={len(q_r)})",
            zorder=3,
            edgecolors="none",
        )
    if len(q_s):
        ax_cq.scatter(
            _zscore(q_s, mu_q, sd_q),
            _zscore(c_s, mu_c, sd_c),
            s=14,
            c=color_sig,
            alpha=0.55,
            label=f"{sig_lab} signature (n={len(q_s)})",
            zorder=3,
            edgecolors="none",
        )
    ax_cq.axhline(0.0, color="0.5", lw=0.8, alpha=0.5)
    ax_cq.axvline(0.0, color="0.5", lw=0.8, alpha=0.5)
    ax_cq.set_xlabel("Discharge Q z-score\n(pooled target + donors)")
    ax_cq.set_ylabel("Nitrate C z-score\n(pooled target + donors)")
    ax_cq.set_title(f"{site_no}  |  c–Q (target period, z-scored)", fontsize=11)
    ax_cq.grid(True, alpha=0.25)
    ax_cq.legend(loc="best", fontsize=8)
    fig.tight_layout()
    cq_path = out_dir / "cq_scatter.png"
    fig.savefig(cq_path, dpi=150)
    plt.close(fig)
    n_written += 1

    return n_written


def plot_pair_dual_donors(
    *,
    reg_case: DonorCase,
    sig_case: DonorCase,
    meta_out: Path,
    common_sites: Sequence[str],
    max_sites: Optional[int] = None,
    workers: int = 1,
) -> int:
    """
    Per common site, write plots under
    ``Models/Similarity/{pair}/donor_nitrate_plots/{site_no}/``:
      nitrate_timeseries.png, cq_scatter.png
    """
    if not common_sites:
        return 0
    try:
        import matplotlib
    except ImportError as exc:
        print(f"  SKIP plots (matplotlib unavailable: {exc})", flush=True)
        return 0

    plot_root = meta_out / "donor_nitrate_plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    need_cols = ["site_no", "date", "nitrate_mg_l", "discharge_cfs"]
    reg_cols = need_cols + [REG_VAL, REG_ID]
    sig_cols = need_cols + [SIG_VAL, SIG_ID]
    reg_train = pd.read_parquet(reg_case.train_parquet, columns=reg_cols)
    reg_test = pd.read_parquet(
        reg_case.test_parquet, columns=["site_no", "date", "nitrate_mg_l", "discharge_cfs"]
    )
    sig_train = pd.read_parquet(sig_case.train_parquet, columns=sig_cols)

    for df in (reg_train, reg_test, sig_train):
        df["site_no"] = df["site_no"].map(canonical_usgs_site_id)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()


    q_by_site: Dict[str, pd.Series] = {}

    def _ingest_q(df: pd.DataFrame) -> None:
        if "discharge_cfs" not in df.columns:
            return
        for sid, g in df.groupby("site_no", sort=False):
            sid = str(sid)
            s = pd.Series(
                g["discharge_cfs"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(pd.to_datetime(g["date"]).dt.normalize()),
            ).groupby(level=0).mean()
            q_by_site[sid] = q_by_site[sid].combine_first(s) if sid in q_by_site else s

    for df in (reg_train, sig_train):
        _ingest_q(df)

    pair_key = case_pair_key(reg_case.tag)
    q_sources: List[Path] = []
    nodonor_dir = reg_case.case_dir.parent / f"{pair_key}_NODONOR"
    if nodonor_dir.is_dir():
        q_sources.extend(sorted(nodonor_dir.glob("forcing_*_train.parquet"))[:1])
    for cand in (
        DATASETS / "forcing_117_V1.parquet",
        DATASETS / "forcing_NODONOR_50_V1.parquet",
        DATASETS / "forcing_BASE_50_V1.parquet",
    ):
        if cand.is_file():
            q_sources.append(cand)
    for path in q_sources:
        try:
            extra = pd.read_parquet(path, columns=["site_no", "date", "discharge_cfs"])
            extra["site_no"] = extra["site_no"].map(canonical_usgs_site_id)
            extra["date"] = pd.to_datetime(extra["date"]).dt.normalize()
            _ingest_q(extra)
        except Exception:
            continue

    sites = [canonical_usgs_site_id(s) for s in common_sites]
    if max_sites is not None:
        sites = sites[: int(max_sites)]

    print(f"  Dual-donor plots: {len(sites)} sites -> {plot_root}/{{site}}/", flush=True)
    site_jobs: List[DualDonorSitePlotJob] = []
    for site in sites:
        g_reg = reg_train.loc[reg_train["site_no"] == site].sort_values("date")
        g_sig = sig_train.loc[sig_train["site_no"] == site].sort_values("date")
        if g_reg.empty or g_sig.empty:
            continue
        merged = g_reg[["date", REG_VAL, REG_ID]].merge(
            g_sig[["date", SIG_VAL, SIG_ID]], on="date", how="inner"
        )
        if merged.empty:
            continue
        dates = pd.DatetimeIndex(merged["date"])
        reg_id = str(merged[REG_ID].iloc[0]) if REG_ID in merged.columns else ""
        sig_id = str(merged[SIG_ID].iloc[0]) if SIG_ID in merged.columns else ""
        if reg_id.lower() in ("", "nan"):
            reg_id = ""
        if sig_id.lower() in ("", "nan"):
            sig_id = ""

        tr_obs = g_reg.loc[g_reg["nitrate_mg_l"].notna(), ["date", "nitrate_mg_l"]]
        te_obs = reg_test.loc[
            (reg_test["site_no"] == site) & reg_test["nitrate_mg_l"].notna(),
            ["date", "nitrate_mg_l"],
        ]
        if _target_obs_span(tr_obs, te_obs) is None:
            continue

        tgt_q_train = _site_q_series(reg_train, site)
        tgt_q_test = _site_q_series(reg_test, site)
        target_q = tgt_q_train.combine_first(tgt_q_test)
        regional_q = q_by_site.get(reg_id, pd.Series(dtype=float))
        signature_q = q_by_site.get(sig_id, pd.Series(dtype=float))

        tr_d, tr_v = _obs_df_to_iso_pairs(tr_obs)
        te_d, te_v = _obs_df_to_iso_pairs(te_obs)
        tq_d, tq_v = _series_to_iso_pairs(target_q)
        rq_d, rq_v = _series_to_iso_pairs(regional_q)
        sq_d, sq_v = _series_to_iso_pairs(signature_q)
        site_jobs.append(
            DualDonorSitePlotJob(
                site=site,
                site_dir=plot_root / _safe_plot_filename(site),
                dates_iso=tuple(ts.isoformat() for ts in pd.DatetimeIndex(dates)),
                regional_vals=tuple(float(x) for x in merged[REG_VAL].to_numpy(dtype=float)),
                signature_vals=tuple(float(x) for x in merged[SIG_VAL].to_numpy(dtype=float)),
                train_obs_dates=tr_d,
                train_obs_vals=tr_v,
                test_obs_dates=te_d,
                test_obs_vals=te_v,
                target_q_dates=tq_d,
                target_q_vals=tq_v,
                regional_q_dates=rq_d,
                regional_q_vals=rq_v,
                signature_q_dates=sq_d,
                signature_q_vals=sq_v,
                regional_donor_id=reg_id,
                signature_donor_id=sig_id,
            )
        )

    n_workers = max(1, int(workers))
    n_written = 0
    if n_workers == 1 or len(site_jobs) <= 1:
        for i, job in enumerate(site_jobs, start=1):
            n_written += _dual_donor_site_plot_worker(job)
            if i == 1 or i % 25 == 0 or i == len(site_jobs):
                print(f"    [{i}/{len(site_jobs)}] {job.site}", flush=True)
    else:
        print(f"  Site-parallel plots: {len(site_jobs)} sites, {n_workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for i, n in enumerate(pool.map(_dual_donor_site_plot_worker, site_jobs), start=1):
                n_written += n
                if i == 1 or i % 25 == 0 or i == len(site_jobs):
                    print(f"    [{i}/{len(site_jobs)}] {site_jobs[i - 1].site}", flush=True)

    print(f"  Wrote {n_written} plot files under {plot_root}", flush=True)
    return n_written


def load_common_sites_for_pair(
    *,
    pair_key: str,
    meta_out: Path,
    reg_case: DonorCase,
    sig_case: DonorCase,
) -> List[str]:
    """Prefer Similarity common-sites CSV; else intersect sites that have both donor cols."""
    csv_path = meta_out / f"donor_common_sites_{pair_key}.csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        col = "site_no" if "site_no" in df.columns else df.columns[0]
        sites = [canonical_usgs_site_id(x) for x in df[col].tolist()]
        return [s for s in sites if s]

    reg = pd.read_parquet(reg_case.train_parquet, columns=["site_no", REG_VAL])
    sig = pd.read_parquet(sig_case.train_parquet, columns=["site_no", SIG_VAL])
    reg["site_no"] = reg["site_no"].map(canonical_usgs_site_id)
    sig["site_no"] = sig["site_no"].map(canonical_usgs_site_id)
    reg_ok = set(reg.loc[reg[REG_VAL].notna(), "site_no"])
    sig_ok = set(sig.loc[sig[SIG_VAL].notna(), "site_no"])
    return sorted(reg_ok & sig_ok)


def _plot_single_pair(
    pair_key: str,
    reg_case: DonorCase,
    sig_case: DonorCase,
    *,
    meta_root: Path,
    max_plot_sites: Optional[int],
    site_workers: int = 1,
) -> int:
    try:
        import pyarrow.parquet as pq

        reg_cols = set(pq.ParquetFile(str(reg_case.train_parquet)).schema.names)
        sig_cols = set(pq.ParquetFile(str(sig_case.train_parquet)).schema.names)
    except Exception as exc:
        print(f"SKIP plots {pair_key}: cannot read train parquet schema ({exc})", flush=True)
        return 0
    if REG_VAL not in reg_cols or SIG_VAL not in sig_cols:
        print(
            f"SKIP plots {pair_key}: donor columns missing "
            f"(need {REG_VAL} / {SIG_VAL} — run full assign first)",
            flush=True,
        )
        return 0

    meta_out = meta_dir_for_pair(pair_key, meta_root)
    common = load_common_sites_for_pair(
        pair_key=pair_key,
        meta_out=meta_out,
        reg_case=reg_case,
        sig_case=sig_case,
    )
    if not common:
        print(f"SKIP plots {pair_key}: no common sites", flush=True)
        return 0

    plot_root = meta_out / "donor_nitrate_plots"
    if plot_root.is_dir():
        for stale in plot_root.rglob("nitrate_zscore_timeseries.png"):
            try:
                stale.unlink()
            except OSError:
                pass

    print(f"\n==== Plots only: {pair_key} ({len(common)} common sites) ====", flush=True)
    return plot_pair_dual_donors(
        reg_case=reg_case,
        sig_case=sig_case,
        meta_out=meta_out,
        common_sites=common,
        max_sites=max_plot_sites,
        workers=site_workers,
    )


def _plot_pair_worker(job: PlotPairJob) -> int:
    return _plot_single_pair(
        job.pair_key,
        job.reg_case,
        job.sig_case,
        meta_root=job.meta_root,
        max_plot_sites=job.max_plot_sites,
        site_workers=job.site_workers,
    )


def run_plots_only(
    cases: Sequence[DonorCase],
    *,
    inputs_root: Path,
    meta_root: Path,
    max_plot_sites: Optional[int] = None,
    workers: int = 1,
) -> int:
    """
    Regenerate dual-donor PNGs from already-attached REG/SIG train/test parquets.
    Does not run selection, attach, config patch, or archive.

    When workers > 1, parallelizes across pair keys (or across sites when only
    one pair is selected).
    """
    meta_root.mkdir(parents=True, exist_ok=True)
    by_key: Dict[str, Dict[str, DonorCase]] = {}
    for case in cases:
        by_key.setdefault(case_pair_key(case.tag), {})[case.track] = case

    for key, group in list(by_key.items()):
        for track in ("regional", "signature"):
            if track not in group:
                partner = resolve_partner_case(inputs_root, key, track)
                if partner is not None:
                    group[track] = partner

    ready_pairs: List[Tuple[str, DonorCase, DonorCase]] = []
    for key in sorted(by_key):
        group = by_key[key]
        if "regional" not in group or "signature" not in group:
            print(f"SKIP plots {key}: need both REGDONOR and SIGDONOR", flush=True)
            continue
        ready_pairs.append((key, group["regional"], group["signature"]))

    n_workers = max(1, int(workers))
    total = 0

    if n_workers > 1 and len(ready_pairs) > 1:
        print(
            f"Parallel plots-only: {len(ready_pairs)} pairs, {n_workers} workers",
            flush=True,
        )
        jobs = [
            PlotPairJob(
                pair_key=key,
                reg_case=reg_case,
                sig_case=sig_case,
                meta_root=meta_root,
                max_plot_sites=max_plot_sites,
                site_workers=1,
            )
            for key, reg_case, sig_case in ready_pairs
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_plot_pair_worker, job): job.pair_key for job in jobs}
            for fut in as_completed(futures):
                pair_key = futures[fut]
                n = fut.result()
                total += n
                print(f"  Completed plots {pair_key} ({n} files)", flush=True)
    else:
        site_workers = n_workers if len(ready_pairs) <= 1 else 1
        for key, reg_case, sig_case in ready_pairs:
            total += _plot_single_pair(
                key,
                reg_case,
                sig_case,
                meta_root=meta_root,
                max_plot_sites=max_plot_sites,
                site_workers=site_workers,
            )

    print(f"\nPlots-only done: {total} files written.", flush=True)
    return total


def write_donor_attach(
    case: DonorCase,
    *,
    selection: pd.DataFrame,
    keep_sites: Set[str],
    raw_parquet: Path,
    raw_cache: Dict[str, pd.Series],
    meta_out: Path,
) -> None:
    """Attach donors and rewrite train/test for keep_sites only; meta -> Similarity."""
    if case.track == "regional":
        val_col, mask_col, id_col = REG_VAL, REG_MASK, REG_ID
    else:
        val_col, mask_col, id_col = SIG_VAL, SIG_MASK, SIG_ID

    print(
        f"  [{case.tag}] attach/rewrite on common pool ({len(keep_sites)} sites)",
        flush=True,
    )
    report_windows = load_dnp_report_windows(case.case_dir)
    train = pd.read_parquet(case.train_parquet)
    test = pd.read_parquet(case.test_parquet)
    train_o, test_o, fill_df = attach_donor_columns(
        train,
        test,
        selection=selection,
        val_col=val_col,
        mask_col=mask_col,
        id_col=id_col,
        raw_cache=raw_cache,
        raw_parquet=raw_parquet,
        keep_sites=keep_sites,
        report_windows=report_windows,
    )
    train_o.to_parquet(case.train_parquet, index=False)
    test_o.to_parquet(case.test_parquet, index=False)
    print(
        f"  [{case.tag}] wrote {val_col}  sites={train_o['site_no'].nunique()}  "
        f"rows={len(train_o):,}",
        flush=True,
    )

    fill_path = meta_out / f"donor_fill_stats_{case.track}_{case.tag}.csv"
    fill_df.to_csv(fill_path, index=False)
    if not fill_df.empty:
        summary = (
            fill_df[
                ["frac_observed", "frac_interp", "frac_doy_mean", "frac_doy_window", "frac_median"]
            ]
            .mean()
            .rename("mean_fraction")
            .reset_index()
            .rename(columns={"index": "metric"})
        )
        summary_path = meta_out / f"donor_fill_summary_{case.track}_{case.tag}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  [{case.tag}] fill summary:\n{summary.to_string(index=False)}", flush=True)


    dnp_files = sorted(case.case_dir.glob("D_N_P_nitrate_*.xlsx"))
    if dnp_files:
        dnp = pd.read_excel(dnp_files[0])
        site_col = "site_no" if "site_no" in dnp.columns else dnp.columns[0]
        dnp[site_col] = dnp[site_col].map(canonical_usgs_site_id)
        dnp.loc[dnp[site_col].isin(keep_sites)].reset_index(drop=True).to_excel(
            dnp_files[0], index=False
        )

    patch_lstm_config(
        case.case_dir,
        track=case.track,
        train_name=case.train_parquet.name,
        test_name=case.test_parquet.name,
    )
    archive_case(case.case_dir)


def family_key_from_pair(pair_key: str) -> str:
    """R20B14_C1_Seed21 -> R20B14_C1 (strip Seed* token)."""
    return "_".join(tok for tok in pair_key.split("_") if not tok.startswith("Seed"))


def seed_from_pair_key(pair_key: str) -> int:
    match = re.search(r"_Seed(\d+)$", pair_key)
    if not match:
        raise ValueError(f"Cannot parse seed from {pair_key}")
    return int(match.group(1))


def _run_single_donor_family(
    family: str,
    cases: Sequence[DonorCase],
    *,
    inputs_root: Path,
    meta_root: Path,
    nldi: pd.DataFrame,
    meta: Dict[str, dict],
    attr: pd.DataFrame,
    raw_parquet: Path,
    raw_cache: Dict[str, pd.Series],
    donor_pool: Set[str],
    write_plots: bool,
    max_plot_sites: Optional[int],
) -> List[dict]:
    import sys

    aux = SCRIPTS_DIR / "Auxilliary_Scripts"
    if str(aux) not in sys.path:
        sys.path.insert(0, str(aux))
    try:
        import experiment_case_sets as ecs
    except ImportError:
        ecs = None

    by_key: Dict[str, Dict[str, DonorCase]] = {}
    for case in cases:
        by_key.setdefault(case_pair_key(case.tag), {})[case.track] = case

    for key, group in list(by_key.items()):
        for track in ("regional", "signature"):
            if track not in group:
                partner = resolve_partner_case(inputs_root, key, track)
                if partner is not None:
                    group[track] = partner

    pair_keys = sorted(by_key.keys())
    summary_rows: List[dict] = []

    print(
        f"\n{'#' * 72}\nFAMILY: {family}  seeds/pairs={len(pair_keys)}\n{'#' * 72}",
        flush=True,
    )
    found_seeds = {seed_from_pair_key(k) for k in pair_keys}
    if found_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            f"{family}: expected seeds {sorted(EXPECTED_SEEDS)}, "
            f"found {sorted(found_seeds)} "
            f"(pair keys: {sorted(pair_keys)})"
        )

    selections: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    ok_by_pair: Dict[str, Set[str]] = {}
    for key in pair_keys:
        group = by_key[key]
        reg_case = group.get("regional")
        sig_case = group.get("signature")
        meta_out = meta_dir_for_pair(key, meta_root)
        print(f"\n---- Pass1 select: {key} ----", flush=True)
        if reg_case is None or sig_case is None:
            print("  SKIP — missing REG or SIG case", flush=True)
            continue
        reg_sel = select_regional_for_case(
            reg_case,
            nldi=nldi,
            meta=meta,
            raw_parquet=raw_parquet,
            raw_cache=raw_cache,
            donor_pool=donor_pool,
            meta_out=meta_out,
        )
        sig_sel = select_signature_for_case(
            sig_case,
            attr=attr,
            raw_parquet=raw_parquet,
            raw_cache=raw_cache,
            donor_pool=donor_pool,
            meta_out=meta_out,
        )
        selections[key] = (reg_sel, sig_sel)
        ok_by_pair[key] = ok_donor_sites(reg_sel) & ok_donor_sites(sig_sel)
        print(f"  train-only REG∩SIG ok={len(ok_by_pair[key])}", flush=True)

    missing_sel = [k for k in pair_keys if k not in selections]
    if missing_sel:
        raise RuntimeError(
            f"{family}: incomplete Pass1 selection; missing REG/SIG for: "
            f"{missing_sel}"
        )
    if {seed_from_pair_key(k) for k in ok_by_pair} != EXPECTED_SEEDS:
        raise RuntimeError(
            f"{family}: expected seeds {sorted(EXPECTED_SEEDS)} after Pass1, "
            f"found {sorted(seed_from_pair_key(k) for k in ok_by_pair)}"
        )

    fixed_pop = set.intersection(*ok_by_pair.values())
    if ecs is not None and pair_keys:
        case_set, gap = parse_case_set_and_gap(pair_keys[0])
        try:
            base_pop = {
                canonical_usgs_site_id(s) for s in ecs.load_population(case_set, gap)
            }
            base_pop.discard("")
            fixed_pop &= base_pop
        except FileNotFoundError:
            pass

    fixed_list = sorted(fixed_pop)
    fam_meta = meta_root / f"fixed_population_{family}.csv"
    pd.DataFrame({"site_no": fixed_list}).to_csv(fam_meta, index=False)
    print(
        f"\n  FIXED population {family}: n={len(fixed_list)} "
        f"(intersection over {len(ok_by_pair)} seeds) -> {fam_meta.name}",
        flush=True,
    )

    for key in pair_keys:
        if key not in selections:
            summary_rows.append(
                {
                    "pair_key": key,
                    "family": family,
                    "n_common": 0,
                    "n_fixed": len(fixed_list),
                    "has_both_tracks": False,
                    "attached": False,
                }
            )
            continue
        group = by_key[key]
        reg_case = group["regional"]
        sig_case = group["signature"]
        meta_out = meta_dir_for_pair(key, meta_root)
        reg_sel, sig_sel = selections[key]
        print(f"\n---- Pass2 attach: {key} ----", flush=True)

        write_common_site_reports(
            pair_key=key,
            meta_out=meta_out,
            reg_sites=ok_donor_sites(reg_sel),
            sig_sites=ok_donor_sites(sig_sel),
            common=fixed_list,
        )

        attached = False
        if fixed_list:
            keep_set = set(fixed_list)
            write_donor_attach(
                reg_case,
                selection=reg_sel,
                keep_sites=keep_set,
                raw_parquet=raw_parquet,
                raw_cache=raw_cache,
                meta_out=meta_out,
            )
            write_donor_attach(
                sig_case,
                selection=sig_sel,
                keep_sites=keep_set,
                raw_parquet=raw_parquet,
                raw_cache=raw_cache,
                meta_out=meta_out,
            )
            for scenario in (
                "REGSIGDONOR",
                "REGSIGALLSENSOR",
                "BASELINE",
                "ALLSENSOR",
                "REDUCEDSENSOR",
            ):
                sib = resolve_sibling_case(inputs_root, key, scenario)
                if sib is None:
                    continue
                if scenario in {"REGSIGDONOR", "REGSIGALLSENSOR"}:
                    write_dual_donor_attach(
                        sib,
                        reg_selection=reg_sel,
                        sig_selection=sig_sel,
                        keep_sites=keep_set,
                        raw_parquet=raw_parquet,
                        raw_cache=raw_cache,
                        meta_out=meta_out,
                    )
                else:
                    rewrite_case_to_sites(sib, keep_set)
                    archive_case(sib.case_dir)
            attached = True
            if write_plots:
                plot_pair_dual_donors(
                    reg_case=reg_case,
                    sig_case=sig_case,
                    meta_out=meta_out,
                    common_sites=fixed_list,
                    max_sites=max_plot_sites,
                )
        else:
            print("  No fixed common sites — skipped attach", flush=True)

        summary_rows.append(
            {
                "pair_key": key,
                "family": family,
                "n_regional_ok": len(ok_donor_sites(reg_sel)),
                "n_signature_ok": len(ok_donor_sites(sig_sel)),
                "n_seed_intersection": len(ok_by_pair.get(key, set())),
                "n_fixed": len(fixed_list),
                "n_common": len(fixed_list),
                "has_both_tracks": True,
                "attached": attached,
            }
        )

    return summary_rows


def _donor_family_worker(job: DonorFamilyJob) -> List[dict]:
    meta = load_site_meta(resolve_metadata_csv())
    nldi = load_nldi_candidates(job.nldi_csv)
    attr = pd.read_parquet(job.attributes)
    attr["site_no"] = attr["site_no"].map(canonical_usgs_site_id)
    raw_sites = pd.read_parquet(job.raw_nitrate, columns=["monitoring_location_id"])
    donor_pool = {
        canonical_usgs_site_id(x) for x in raw_sites["monitoring_location_id"].unique()
    }
    raw_cache: Dict[str, pd.Series] = {}
    return _run_single_donor_family(
        job.family,
        job.cases,
        inputs_root=job.inputs_root,
        meta_root=job.meta_root,
        nldi=nldi,
        meta=meta,
        attr=attr,
        raw_parquet=job.raw_nitrate,
        raw_cache=raw_cache,
        donor_pool=donor_pool,
        write_plots=job.write_plots,
        max_plot_sites=job.max_plot_sites,
    )


def run_paired_donor_assign(
    cases: Sequence[DonorCase],
    *,
    inputs_root: Path,
    meta_root: Path,
    nldi: pd.DataFrame,
    meta: Dict[str, dict],
    attr: pd.DataFrame,
    raw_parquet: Path,
    raw_cache: Dict[str, pd.Series],
    donor_pool: Set[str],
    nldi_csv: Path,
    attributes: Path,
    write_plots: bool = True,
    max_plot_sites: Optional[int] = None,
    workers: int = 1,
) -> pd.DataFrame:
    """
    Two-pass donor attach for fair cross-seed populations.

    Pass 1 — train-only REG/SIG selection for every seed (no attach).
    Pass 2 — fixed_pop = intersection of REG∩SIG over all seeds for that
    (case_set, gap); attach and rewrite every sibling scenario to fixed_pop.

    When workers > 1, each (case_set, gap) family runs in a separate process.
    """
    meta_root.mkdir(parents=True, exist_ok=True)

    by_family_cases: Dict[str, List[DonorCase]] = {}
    for case in cases:
        fam = family_key_from_pair(case_pair_key(case.tag))
        by_family_cases.setdefault(fam, []).append(case)

    jobs = [
        DonorFamilyJob(
            family=fam,
            cases=tuple(by_family_cases[fam]),
            inputs_root=inputs_root,
            meta_root=meta_root,
            nldi_csv=nldi_csv,
            raw_nitrate=raw_parquet,
            attributes=attributes,
            write_plots=write_plots,
            max_plot_sites=max_plot_sites,
        )
        for fam in sorted(by_family_cases.keys())
    ]

    summary_rows: List[dict] = []
    n_workers = max(1, int(workers))

    if n_workers == 1 or len(jobs) <= 1:
        for job in jobs:
            summary_rows.extend(
                _run_single_donor_family(
                    job.family,
                    job.cases,
                    inputs_root=inputs_root,
                    meta_root=meta_root,
                    nldi=nldi,
                    meta=meta,
                    attr=attr,
                    raw_parquet=raw_parquet,
                    raw_cache=raw_cache,
                    donor_pool=donor_pool,
                    write_plots=write_plots,
                    max_plot_sites=max_plot_sites,
                )
            )
    else:
        print(
            f"Parallel donor assign: {len(jobs)} families, {n_workers} workers",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_donor_family_worker, job): job.family for job in jobs}
            for fut in as_completed(futures):
                fam = futures[fut]
                rows = fut.result()
                summary_rows.extend(rows)
                print(f"  Completed family {fam} ({len(rows)} pair(s))", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary_path = meta_root / "donor_common_sites_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}", flush=True)
    if not summary.empty:
        print(summary.to_string(index=False), flush=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs-root", type=Path, default=INPUTS_ROOT)
    p.add_argument("--cases", nargs="+", default=None, help="Substring filters on case folder names")
    p.add_argument("--gaps", nargs="+", default=None, help="Gap prefixes e.g. R20B14 R20B60")
    p.add_argument(
        "--track",
        choices=["regional", "signature", "both"],
        default="both",
        help="Which case folders to discover; attach always requires both tracks for the pair",
    )
    p.add_argument("--nldi-csv", type=Path, default=NLDI_CANDIDATES_CSV)
    p.add_argument("--raw-nitrate", type=Path, default=RAW_NITRATE_PARQUET)
    p.add_argument("--attributes", type=Path, default=ATTRIBUTE_PARQUET)
    p.add_argument(
        "--similarity-meta",
        type=Path,
        default=SIMILARITY_META_ROOT,
        help="Directory for selection/fill/common-site meta outputs",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip dual-donor nitrate PNGs under Models/Similarity/{pair}/donor_nitrate_plots/",
    )
    p.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "Only regenerate dual-donor PNGs from already-attached REG/SIG parquets "
            "(no selection/attach/config rewrite)"
        ),
    )
    p.add_argument(
        "--max-plot-sites",
        type=int,
        default=None,
        help="Limit dual-donor plots per pair (debug)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=12,
        help=(
            "Parallel worker processes: donor assign uses one per (case_set, gap) "
            "family; --plots-only uses one per pair (or per site when a single pair "
            "is selected)"
        ),
    )
    args = p.parse_args(argv)

    tracks = ["regional", "signature"] if args.track == "both" else [args.track]
    cases = discover_donor_cases(
        args.inputs_root,
        tracks=tracks,
        case_filters=args.cases,
        gap_filters=args.gaps,
    )
    if not cases:
        print(f"No REGDONOR/SIGDONOR cases under {args.inputs_root}", flush=True)
        return 1

    if args.plots_only:
        print(
            f"Plots-only mode | cases={len(cases)} | workers={args.workers} | "
            f"meta -> {args.similarity_meta}",
            flush=True,
        )
        n = run_plots_only(
            cases,
            inputs_root=args.inputs_root,
            meta_root=args.similarity_meta,
            max_plot_sites=args.max_plot_sites,
            workers=args.workers,
        )
        return 0 if n > 0 else 1

    for path in (args.nldi_csv, args.raw_nitrate, args.attributes):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")

    print(f"Cases: {len(cases)}", flush=True)
    print(f"NLDI candidates CSV: {args.nldi_csv}", flush=True)
    print(f"Meta/summaries -> {args.similarity_meta}", flush=True)
    print(f"Final train/test stay under {args.inputs_root}", flush=True)
    print(
        "Note: this CSV is the precomputed NLDI UM/DM topology list "
        "(identifier=candidate connected to target_site) — correct substitute for a live NLDI cache.",
        flush=True,
    )
    print(
        "Attach/rewrite uses only REG ∩ SIG common sites for each matched pair.",
        flush=True,
    )

    meta = load_site_meta(resolve_metadata_csv())
    nldi = load_nldi_candidates(args.nldi_csv)
    attr = pd.read_parquet(args.attributes)
    attr["site_no"] = attr["site_no"].map(canonical_usgs_site_id)
    raw_sites = pd.read_parquet(args.raw_nitrate, columns=["monitoring_location_id"])
    donor_pool = {canonical_usgs_site_id(x) for x in raw_sites["monitoring_location_id"].unique()}
    print(f"Donor pool (raw nitrate sites): {len(donor_pool)}", flush=True)

    raw_cache: Dict[str, pd.Series] = {}
    run_paired_donor_assign(
        cases,
        inputs_root=args.inputs_root,
        meta_root=args.similarity_meta,
        nldi=nldi,
        meta=meta,
        attr=attr,
        raw_parquet=args.raw_nitrate,
        raw_cache=raw_cache,
        donor_pool=donor_pool,
        nldi_csv=args.nldi_csv,
        attributes=args.attributes,
        write_plots=not args.no_plots,
        max_plot_sites=args.max_plot_sites,
        workers=args.workers,
    )
    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
