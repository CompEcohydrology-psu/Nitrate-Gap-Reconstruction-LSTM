#!/usr/bin/env python3
"""Build BASE/REDUCED/ALL soft-sensor forcings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT_ROOT / "Datasets"

SS_PARAMS = [
    "Temperature",
    "Specific_Conductance",
    "Dissolved_Oxygen",
    "pH",
    "Turbidity",
]


def _ss_related_columns(columns: list[str], params: list[str]) -> list[str]:
    """Value + *_interpolated / *_time_interpolated / *_doy_interpolated for params."""
    out: list[str] = []
    for c in columns:
        for p in params:
            if c == p or c.startswith(f"{p}_"):
                out.append(c)
                break
    return out


def _filter_min_nitrate(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    if min_obs <= 0:
        return df
    counts = (
        df.groupby(df["site_no"].astype(str).str.strip())["nitrate_mg_l"]
        .apply(lambda s: int(s.notna().sum()))
    )
    keep = counts[counts >= int(min_obs)].index
    drop = counts[counts < int(min_obs)]
    out = df.loc[df["site_no"].astype(str).str.strip().isin(keep)].copy()
    if len(drop):
        detail = ", ".join(f"{sid}={n}" for sid, n in drop.sort_values().items())
        print(f"  dropped {len(drop)} site(s) with < {min_obs} nitrate obs ({detail})", flush=True)
    return out


def build_scenarios(
    src: Path,
    *,
    out_dir: Path,
    min_nitrate_obs: int = 300,
) -> dict[str, Path]:
    df = pd.read_parquet(src)
    df["site_no"] = df["site_no"].astype(str).str.strip()
    print(f"Source: {src}  sites={df['site_no'].nunique()}  rows={len(df):,}", flush=True)

    df = _filter_min_nitrate(df, min_nitrate_obs)
    n_sites = int(df["site_no"].nunique())
    print(f"  after filter: {n_sites} sites, {len(df):,} rows", flush=True)

    all_ss_cols = _ss_related_columns(list(df.columns), SS_PARAMS)
    missing = [p for p in SS_PARAMS if p not in df.columns]
    if missing:
        raise KeyError(f"Source missing soft-sensor columns: {missing}")

    ec_cols = _ss_related_columns(list(df.columns), ["Specific_Conductance"])
    base_cols = [c for c in df.columns if c not in all_ss_cols]

    scenarios = {
        "BASE": base_cols,
        "REDUCED": base_cols + ec_cols,
        "ALL": list(df.columns),
    }

    written: dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, cols in scenarios.items():

        ordered = [c for c in df.columns if c in set(cols)]
        out = df[ordered].sort_values(["site_no", "date"], kind="mergesort").reset_index(drop=True)
        path = out_dir / f"forcing_{suffix}_{n_sites}_V1.parquet"
        out.to_parquet(path, index=False)
        written[suffix] = path
        extra = [c for c in ordered if c not in base_cols]
        print(
            f"  Wrote {path.name}  cols={out.shape[1]}  "
            f"SS extras={extra if extra else 'none'}",
            flush=True,
        )
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build BASE/REDUCED/ALL forcings from SS forcing.")
    p.add_argument(
        "--src",
        type=Path,
        default=DATASETS / "forcing_SS_52_V1.parquet",
    )
    p.add_argument("--out-dir", type=Path, default=DATASETS)
    p.add_argument(
        "--min-nitrate-obs",
        type=int,
        default=300,
        help="Drop sites with fewer non-null nitrate values (default 300 → 50 sites from SS_52).",
    )
    args = p.parse_args(argv)
    if not args.src.is_file():
        raise SystemExit(f"Missing source: {args.src}")
    build_scenarios(args.src, out_dir=args.out_dir, min_nitrate_obs=args.min_nitrate_obs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
