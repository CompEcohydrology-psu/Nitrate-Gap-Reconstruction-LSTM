"""Config I/O, data load, and normalization helpers."""

from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml


def create_outdir(base_dir, args):
    folder_name = (
        f"E{args['EPOCHS']}_"
        f"H{args['hidden_size']}_"
        f"R{args['rho']}_"
        f"L{args['LSTM_layer']}"
    )

    out_root = Path(base_dir) / "Outputs"
    sub = str(args.get("output_subdir") or "").strip().replace("\\", "/")
    if sub:
        sub_path = Path(sub)
        if sub_path.is_absolute() or ".." in sub_path.parts:
            raise ValueError("output_subdir must be a relative path under Outputs/")
        out_root = out_root / sub_path
    out_dir = out_root / args["output_model"] / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def read_config(dir, config_path=None):
    """
    Load YAML config.

    Args:
        dir: project root (used when config_path is None).
        config_path: optional path to a config file (absolute or relative to dir).
    """
    if config_path is None:
        file_path = (Path(dir) / "config.yaml").resolve()
    else:
        file_path = Path(config_path)
        if not file_path.is_absolute():
            file_path = (Path(dir) / file_path).resolve()
        else:
            file_path = file_path.resolve()

    if not file_path.is_file():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["config_dir"] = str(file_path.parent)
    data["config_path"] = str(file_path)
    return data


def resolve_data_path(path_str: str, args: dict, base_dir: Path) -> Path:
    """
    Resolve a data path from config.

    Relative paths are resolved against ``args['config_dir']`` (directory
    containing the YAML config), e.g.
    ``Inputs/R20B60_102_Seed21_SS/lstm_config.yaml`` → parquets in that folder.

    Falls back to ``<project>/Inputs/<path>`` only for backward compatibility.
    """
    if not path_str:
        raise ValueError("Empty data path in config")
    p = Path(path_str)
    if p.is_absolute():
        if not p.is_file():
            raise FileNotFoundError(f"Data file not found: {p}")
        return p

    config_dir = Path(args.get("config_dir") or (Path(base_dir) / "Inputs"))
    candidate = (config_dir / p).resolve()
    if candidate.is_file():
        return candidate

    legacy = (Path(base_dir) / "Inputs" / p).resolve()
    if legacy.is_file():
        return legacy

    raise FileNotFoundError(
        "Data file not found. Tried:\n"
        f"  1) {candidate}  (relative to config_dir)\n"
        f"  2) {legacy}  (legacy Inputs/ layout)\n"
        f"config_dir={config_dir}\n"
        f"path in yaml={path_str!r}\n"
        "Use --config Inputs/<run_folder>/lstm_config.yaml and copy updated Auxilliary.py to HPC."
    )


def _sort_by_site_date(
    df: pd.DataFrame,
    *,
    site_col: str = "site_no",
    date_col: str = "date",
) -> pd.DataFrame:
    """Stable sort by site then date when those columns exist."""
    key = [c for c in (site_col, date_col) if c in df.columns]
    if not key:
        return df
    return df.sort_values(key, kind="mergesort").reset_index(drop=True)


def load_data(base_dir, args):
    site_col = args.get("site_col", "site_no")
    date_col = args.get("date_col", "date")
    base_dir = Path(base_dir)

    train_forcing = _sort_by_site_date(
        pd.read_parquet(resolve_data_path(args["forcing_path_train"], args, base_dir)),
        site_col=site_col,
        date_col=date_col,
    )
    test_forcing = _sort_by_site_date(
        pd.read_parquet(resolve_data_path(args["forcing_path_test"], args, base_dir)),
        site_col=site_col,
        date_col=date_col,
    )
    attribute = _sort_by_site_date(
        pd.read_parquet(resolve_data_path(args["attr_path"], args, base_dir)),
        site_col=site_col,
        date_col=date_col,
    )
    DNP = _sort_by_site_date(
        pd.read_excel(resolve_data_path(args["D_N_P_path"], args, base_dir)),
        site_col=site_col,
        date_col=date_col,
    )

    data_dict = {
        "train_forcing": train_forcing,
        "test_forcing": test_forcing,
        "attribute": attribute,
        "DNP": DNP,
    }

    return data_dict


def _series_for_norm(s: pd.Series):
    """Return a float series for mean/std, or None if there are no numeric values."""
    if pd.api.types.is_numeric_dtype(s):
        ser = s
    else:
        ser = pd.to_numeric(s, errors="coerce")
    if not ser.notna().any():
        return None
    return ser.astype(float)


def _norm_stats_entry(mean_val, std_val, ser: pd.Series) -> dict:
    """mean/std/min/max from the pre-scaling numeric series (train or static attrs)."""
    min_v = ser.min(skipna=True)
    max_v = ser.max(skipna=True)
    return {
        "mean": float(mean_val) if pd.notna(mean_val) else None,
        "std": float(std_val) if pd.notna(std_val) else None,
        "min": float(min_v) if pd.notna(min_v) else None,
        "max": float(max_v) if pd.notna(max_v) else None,
    }


def _get_norm_method(args: dict, section: str, var: str) -> str:
    """
    Per-variable normalization selector.

    Optional config shape:
      normalization:
        forcings:
          pr: log1p
          tmmx: zscore
        targets:
          nitrate_mg_l: log1p

    Supported methods:
      - "zscore": (x - mean) / std
      - "log1p": log1p(max(x, 0)) then z-score in log space
      - "none": no scaling (categorical / reliability flags, e.g. donor_obs_mask)
    """
    norm_cfg = args.get("normalization", {}) or {}
    sec = norm_cfg.get(section, {}) or {}
    method = (sec.get(var) or "zscore")
    method = str(method).strip().lower()
    if method not in {"zscore", "log1p", "none"}:
        raise ValueError(f"Unknown normalization method {method!r} for {section}.{var}")
    return method


def _get_norm_scope(args: dict, section: str, var: str) -> str:
    """
    Per-variable normalization *scope* selector.

    Supported scopes:
      - "global": one mean/std across all sites (current behavior)
      - "per_site": compute mean/std per site (from TRAIN only) and apply to both train/test

    Config options (either style works):

    1) Mapping form:
      normalization:
        scope:
          forcings:
            discharge_cfs: per_site
            pr: global
          targets:
            nitrate_mg_l: per_site

    2) List form:
      normalization:
        per_site:
          forcings: [discharge_cfs, local_NO3_input]
          targets: [nitrate_mg_l]
    """
    norm_cfg = args.get("normalization", {}) or {}


    scope_cfg = (norm_cfg.get("scope", {}) or {}).get(section, {}) or {}
    if isinstance(scope_cfg, dict) and var in scope_cfg:
        scope = str(scope_cfg.get(var) or "global").strip().lower()
        if scope not in {"global", "per_site"}:
            raise ValueError(f"Unknown normalization scope {scope!r} for {section}.{var}")
        return scope


    per_site_cfg = norm_cfg.get("per_site", {}) or {}
    if isinstance(per_site_cfg, dict):
        vars_list = per_site_cfg.get(section, []) or []
        if isinstance(vars_list, (list, tuple, set)) and var in set(vars_list):
            return "per_site"

    return "global"


def _apply_transform_series(ser: pd.Series, method: str) -> pd.Series:
    """Apply the non-linear part of normalization before z-scoring."""
    if method == "none":
        return ser.astype(float)
    if method == "log1p":
        x = ser.astype(float).clip(lower=0.0)
        return np.log1p(x)
    return ser.astype(float)


def _dnp_window_bounds(DNP: pd.DataFrame, site_col: str) -> pd.DataFrame:
    """
    Per-site calendar windows from DNP.

    native: Start_Date–End_Date (site's observed-target span; excludes rectangular padding)
    train:  S_Training–E_Training (supervised train window), intersected with native
    """
    d = DNP.copy()
    d[site_col] = d[site_col].astype(str).str.strip()
    need = ["Start_Date", "End_Date", "S_Training", "E_Training"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise ValueError(f"DNP missing columns required for norm windows: {missing}")

    out = pd.DataFrame(
        {
            site_col: d[site_col],
            "native_start": pd.to_datetime(d["Start_Date"], errors="coerce").dt.normalize(),
            "native_end": pd.to_datetime(d["End_Date"], errors="coerce").dt.normalize(),
            "train_start": pd.to_datetime(d["S_Training"], errors="coerce").dt.normalize(),
            "train_end": pd.to_datetime(d["E_Training"], errors="coerce").dt.normalize(),
        }
    )

    out["train_start"] = out[["train_start", "native_start"]].max(axis=1)
    out["train_end"] = out[["train_end", "native_end"]].min(axis=1)
    return out.drop_duplicates(subset=[site_col], keep="first")


def _row_mask_dnp_window(
    df: pd.DataFrame,
    bounds: pd.DataFrame,
    *,
    site_col: str,
    date_col: str,
    window: str,
) -> pd.Series:
    """
    Boolean mask aligned to ``df.index`` for rows inside the requested DNP window.

    window:
      - \"native\": Start_Date–End_Date (exclude zero-padded rectangular days)
      - \"train\":  S_Training–E_Training ∩ native (rows used for norm statistics)
    """
    if site_col not in df.columns or date_col not in df.columns:
        return pd.Series(True, index=df.index)

    start_col, end_col = ("native_start", "native_end") if window == "native" else ("train_start", "train_end")
    left = df[[site_col]].copy()
    left[site_col] = left[site_col].astype(str).str.strip()
    left["_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    m = left.merge(bounds[[site_col, start_col, end_col]], on=site_col, how="left", validate="m:1")
    ok = (
        m["_date"].notna()
        & m[start_col].notna()
        & m[end_col].notna()
        & (m["_date"] >= m[start_col])
        & (m["_date"] <= m[end_col])
    )
    return pd.Series(ok.to_numpy(), index=df.index)


def norm_data(raw_data, out_dir, args):
    """
    Normalize forcings / targets / attributes.

    Forcing stats are computed only on valid training rows:
      DNP train window ∩ native site span ∩ finite values.
    After z-score, days outside the native span (rectangular zero-padding) are set to
    0.0 so a filled zero means \"training mean\", not a physical zero observation.
    """

    train_df = raw_data["train_forcing"].copy()
    test_df = raw_data["test_forcing"].copy()
    attr_df = raw_data["attribute"].copy()
    DNP = raw_data["DNP"]

    forcings = args["forcings"]
    attributes = args["attributes"]

    eps = 1e-8
    site_col = args.get("site_col", "site_no")
    date_col = args.get("date_col", "date")

    norm_stats = {
        "forcings": {},
        "attributes": {},
        "target": {},
        "norm_policy": {
            "forcing_stats": "DNP_train_window_intersect_native_span_finite_values",
            "forcing_pad_after_zscore": "zero_outside_native_span",
        },
    }
    if site_col in DNP.columns:
        norm_stats["site_order"] = [str(s).strip() for s in DNP[site_col].tolist()]
    else:
        norm_stats["site_order"] = (
            train_df[site_col].astype(str).str.strip().drop_duplicates().tolist()
            if site_col in train_df.columns
            else []
        )

    bounds = _dnp_window_bounds(DNP, site_col)
    train_native_mask = _row_mask_dnp_window(
        train_df, bounds, site_col=site_col, date_col=date_col, window="native"
    )
    train_stats_mask = _row_mask_dnp_window(
        train_df, bounds, site_col=site_col, date_col=date_col, window="train"
    )
    test_native_mask = _row_mask_dnp_window(
        test_df, bounds, site_col=site_col, date_col=date_col, window="native"
    )

    def _zscore_per_site(
        *,
        df_apply: pd.DataFrame,
        var: str,
        method: str,
        means_by_site: pd.Series,
        stds_by_site: pd.Series,
        native_mask: pd.Series,
    ) -> pd.Series:
        if site_col not in df_apply.columns:
            raise ValueError(f"Per-site normalization requires {site_col!r} column")
        s0 = _series_for_norm(df_apply[var])
        if s0 is None:
            return df_apply[var]
        s = _apply_transform_series(s0, method)
        site_vals = df_apply[site_col].astype(str).str.strip()
        mu = site_vals.map(means_by_site).astype(float)
        sd = site_vals.map(stds_by_site).astype(float)
        miss = (~np.isfinite(mu.to_numpy())) | (~np.isfinite(sd.to_numpy())) | (sd.to_numpy() < eps)
        if bool(np.any(miss)):
            mu_f = float(np.nanmean(means_by_site.to_numpy(dtype=float))) if len(means_by_site) else 0.0
            sd_f = float(np.nanmean(stds_by_site.to_numpy(dtype=float))) if len(stds_by_site) else 1.0
            sd_f = sd_f if sd_f >= eps else 1.0
            mu = mu.where(~miss, mu_f)
            sd = sd.where(~miss, sd_f)
        out = (s - mu.to_numpy()) / sd.to_numpy()

        out = out.where(native_mask.to_numpy() & s0.notna().to_numpy(), 0.0)
        return out.fillna(0.0)

    def _per_site_stats_rows(
        *,
        sites: pd.Series,
        ser_transformed: pd.Series,
        site_order: list,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, list]:
        g = pd.DataFrame({"_site": sites.astype(str).str.strip(), "_v": ser_transformed}).groupby(
            "_site", sort=False
        )["_v"]
        means = g.mean()
        stds = g.std().fillna(1.0).where(lambda x: x >= eps, 1.0)
        mins = g.min()
        maxs = g.max()

        rows = []
        for s in site_order:
            rows.append(
                {
                    "site_no": s,
                    "mean": float(means.get(s, np.nan)),
                    "std": float(stds.get(s, np.nan)),
                    "min": float(mins.get(s, np.nan)),
                    "max": float(maxs.get(s, np.nan)),
                }
            )
        return means, stds, mins, maxs, rows


    for var in forcings:
        if var not in train_df.columns:
            continue
        method = _get_norm_method(args, "forcings", var)
        tr0 = _series_for_norm(train_df[var])
        if tr0 is None:
            continue

        scope = _get_norm_scope(args, "forcings", var)
        if method == "none":

            norm_stats["forcings"][var] = {"scope": scope, "transform": "none"}
            continue

        tr = _apply_transform_series(tr0, method)

        stats_ok = train_stats_mask.to_numpy() & tr0.notna().to_numpy()
        tr_stats = tr.where(stats_ok)

        if scope == "per_site":
            if site_col not in train_df.columns:
                raise ValueError(
                    f"Per-site normalization requested for {var!r} but {site_col!r} missing in train_forcing"
                )
            site_order = norm_stats.get("site_order", [])
            means, stds, mins, maxs, rows = _per_site_stats_rows(
                sites=train_df[site_col],
                ser_transformed=tr_stats,
                site_order=site_order,
            )

            train_df[var] = _zscore_per_site(
                df_apply=train_df,
                var=var,
                method=method,
                means_by_site=means,
                stds_by_site=stds,
                native_mask=train_native_mask,
            )
            if var in test_df.columns:
                test_df[var] = _zscore_per_site(
                    df_apply=test_df,
                    var=var,
                    method=method,
                    means_by_site=means,
                    stds_by_site=stds,
                    native_mask=test_native_mask,
                )

            entry = {
                "scope": "per_site",
                "transform": method,
                "site_col": site_col,
                "site_order": site_order,
                "mean_by_site": [float(means.get(s, np.nan)) for s in site_order],
                "std_by_site": [float(stds.get(s, np.nan)) for s in site_order],
                "min_by_site": [float(mins.get(s, np.nan)) for s in site_order],
                "max_by_site": [float(maxs.get(s, np.nan)) for s in site_order],
                "by_site": rows,
                "n_stats_rows": int(stats_ok.sum()),
            }
            norm_stats["forcings"][var] = entry
        else:
            mean_val = tr_stats.mean(skipna=True)
            std_val = tr_stats.std(skipna=True)
            if pd.isna(std_val) or std_val < eps:
                std_val = 1.0
            if pd.isna(mean_val):
                mean_val = 0.0

            out_tr = (tr - mean_val) / std_val
            out_tr = out_tr.where(train_native_mask.to_numpy() & tr0.notna().to_numpy(), 0.0)
            train_df[var] = out_tr.fillna(0.0)

            if var in test_df.columns:
                te0 = _series_for_norm(test_df[var])
                if te0 is not None:
                    te = _apply_transform_series(te0, method)
                    out_te = (te - mean_val) / std_val
                    out_te = out_te.where(test_native_mask.to_numpy() & te0.notna().to_numpy(), 0.0)
                    test_df[var] = out_te.fillna(0.0)

            entry = _norm_stats_entry(mean_val, std_val, tr_stats)
            entry["transform"] = method
            entry["scope"] = "global"
            entry["n_stats_rows"] = int(stats_ok.sum())
            norm_stats["forcings"][var] = entry


    target_cols = args["targets"]

    for target_col in target_cols:
        if target_col not in train_df.columns:
            continue
        method = _get_norm_method(args, "targets", target_col)
        tr0 = _series_for_norm(train_df[target_col])
        if tr0 is None:
            continue

        scope = _get_norm_scope(args, "targets", target_col)
        tr = _apply_transform_series(tr0, method)

        stats_ok = train_stats_mask.to_numpy() & tr0.notna().to_numpy()
        tr_stats = tr.where(stats_ok)

        if scope == "per_site":
            if site_col not in train_df.columns:
                raise ValueError(
                    f"Per-site normalization requested for target {target_col!r} but {site_col!r} missing"
                )
            site_order = norm_stats.get("site_order", [])
            means, stds, mins, maxs, rows = _per_site_stats_rows(
                sites=train_df[site_col],
                ser_transformed=tr_stats,
                site_order=site_order,
            )


            s = _apply_transform_series(tr0, method)
            site_vals = train_df[site_col].astype(str).str.strip()
            mu = site_vals.map(means).astype(float)
            sd = site_vals.map(stds).astype(float)
            train_df[target_col] = ((s - mu.to_numpy()) / sd.to_numpy()).where(tr0.notna())

            if target_col in test_df.columns:
                te0 = _series_for_norm(test_df[target_col])
                if te0 is not None:
                    te = _apply_transform_series(te0, method)
                    site_vals_te = test_df[site_col].astype(str).str.strip()
                    mu_te = site_vals_te.map(means).astype(float)
                    sd_te = site_vals_te.map(stds).astype(float)
                    test_df[target_col] = ((te - mu_te.to_numpy()) / sd_te.to_numpy()).where(te0.notna())

            entry = {
                "scope": "per_site",
                "transform": method,
                "site_col": site_col,
                "site_order": site_order,
                "mean_by_site": [float(means.get(s, np.nan)) for s in site_order],
                "std_by_site": [float(stds.get(s, np.nan)) for s in site_order],
                "min_by_site": [float(mins.get(s, np.nan)) for s in site_order],
                "max_by_site": [float(maxs.get(s, np.nan)) for s in site_order],
                "by_site": rows,
                "n_stats_rows": int(stats_ok.sum()),
            }
            norm_stats["target"][target_col] = entry
        else:
            target_mean = tr_stats.mean(skipna=True)
            target_std = tr_stats.std(skipna=True)
            if pd.isna(target_std) or target_std < eps:
                target_std = 1.0
            if pd.isna(target_mean):
                target_mean = 0.0
            train_df[target_col] = ((tr - target_mean) / target_std).where(tr0.notna())
            if target_col in test_df.columns:
                te0 = _series_for_norm(test_df[target_col])
                if te0 is not None:
                    te = _apply_transform_series(te0, method)
                    test_df[target_col] = ((te - target_mean) / target_std).where(te0.notna())
            entry = _norm_stats_entry(target_mean, target_std, tr_stats)
            entry["transform"] = method
            entry["scope"] = "global"
            entry["n_stats_rows"] = int(stats_ok.sum())
            norm_stats["target"][target_col] = entry


    for var in attributes:
        if var not in attr_df.columns:
            continue

        s = _series_for_norm(attr_df[var])
        if s is None:
            continue

        mean_val = s.mean(skipna=True)
        std_val = s.std(skipna=True)

        if pd.isna(std_val) or std_val < eps:
            std_val = 1.0

        attr_df[var] = (s - mean_val) / std_val

        entry = _norm_stats_entry(mean_val, std_val, s)
        entry["transform"] = "zscore"
        norm_stats["attributes"][var] = entry


    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "norm_stats.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(norm_stats, f, sort_keys=False)

    data_dict = {
        "train_forcing": train_df,
        "test_forcing": test_df,
        "attribute": attr_df,
        "DNP": DNP,
        "norm_stats": norm_stats,
    }

    return data_dict


def denorm_targets_array(y_scaled: np.ndarray, args: dict, norm_stats: dict) -> np.ndarray:
    """Inverse scaling for each target channel using ``norm_stats['target']``."""
    out = np.array(y_scaled, dtype=np.float64, copy=True)
    for vi, name in enumerate(args["targets"]):
        st = norm_stats.get("target", {}).get(name)
        if st is None:
            continue

        transform = str((st.get("transform") or "zscore")).strip().lower()
        scope = str((st.get("scope") or "global")).strip().lower()

        if scope == "per_site":
            mean_by_site = st.get("mean_by_site") or []
            std_by_site = st.get("std_by_site") or []

            if (not mean_by_site or not std_by_site) and st.get("by_site"):
                by_site = st.get("by_site") or []
                mean_by_site = [r.get("mean") for r in by_site]
                std_by_site = [r.get("std") for r in by_site]
            if len(mean_by_site) == 0 or len(std_by_site) == 0:
                continue
            mean_arr = np.asarray(mean_by_site, dtype=float)
            std_arr = np.asarray(std_by_site, dtype=float)
            if mean_arr.ndim != 1 or std_arr.ndim != 1:
                raise ValueError(f"Per-site norm stats for target {name!r} must be 1D arrays")
            if y_scaled.shape[1] != mean_arr.shape[0]:
                raise ValueError(
                    f"Per-site norm stats for target {name!r} has {mean_arr.shape[0]} sites, "
                    f"but y_scaled has ngrid={y_scaled.shape[1]}"
                )
            x = y_scaled[..., vi] * std_arr[None, :] + mean_arr[None, :]
            if transform == "log1p":
                out[..., vi] = np.expm1(x)
            else:
                out[..., vi] = x
        else:
            if st.get("std") is None or st.get("mean") is None:
                continue
            std = float(st["std"])
            mean = float(st["mean"])
            x = y_scaled[..., vi] * std + mean
            if transform == "log1p":
                out[..., vi] = np.expm1(x)
            else:
                out[..., vi] = x
    return out
