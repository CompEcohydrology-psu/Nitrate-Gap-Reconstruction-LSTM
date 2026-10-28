"""Train/test batch sampling for gap evaluation."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def _t2dt(t, hr=False):
    import datetime as dt

    t_out = None
    if type(t) is int:
        if t < 30000000 and t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            t_out = t if hr is False else t

    if type(t) is dt.date:
        t_out = t if hr is False else t

    if type(t) is dt.datetime:
        t_out = t.date() if hr is False else t

    if t_out is None:
        raise ValueError("t2dt: unsupported time type")
    return t_out


def t_range2_array(t_range, *, step=np.timedelta64(1, "D")):
    sd = _t2dt(t_range[0])
    ed = _t2dt(t_range[1])
    return np.arange(sd, ed, step)


def percentage_day_cal(D_N_P: pd.DataFrame, args) -> pd.DataFrame:
    """Per-basin training length and offset; rows follow D_N_P row order (matches tensor columns)."""
    D_N_P = D_N_P.copy()
    first_tr_date = pd.to_datetime(str(args["t_train"][0]))
    d1_general = datetime.datetime(
        first_tr_date.year, first_tr_date.month, first_tr_date.day
    )
    temp_data = []
    offset = []
    for _, row in D_N_P.iterrows():
        st = pd.Timestamp(row["S_Training"]).to_pydatetime()
        et = pd.Timestamp(row["E_Training"]).to_pydatetime()
        d1 = datetime.datetime(st.year, st.month, st.day)
        d2 = datetime.datetime(et.year, et.month, et.day)
        temp_data.append((d2 - d1).days)
        offset.append((d1 - d1_general).days)

    D_N_P["no_days"] = temp_data
    D_N_P["offset"] = offset
    total_days = float(np.sum(temp_data))
    if total_days <= 0:
        n = max(len(D_N_P), 1)
        D_N_P["day_percent"] = np.full(len(D_N_P), 1.0 / n, dtype=float)
    else:
        D_N_P["day_percent"] = D_N_P["no_days"] / total_days
    return D_N_P


def no_iter_nt_ngrid(set_name: str, args, x: np.ndarray, D_N_P_new: pd.DataFrame):
    """Iterations per epoch and batch metadata; matches Adaptive_TimeSeries logic."""
    nt, ngrid, nx = x.shape
    if len(D_N_P_new) != ngrid:
        raise ValueError(
            f"D_N_P rows ({len(D_N_P_new)}) != ngrid from tensor ({ngrid})"
        )
    nt_new = D_N_P_new["no_days"].sum() / ngrid
    denom = float(nt_new) - float(args["warm_up"])
    if denom <= 0:
        n_iter_ep = 1
    else:
        p = (args["batch_size"] * args["rho"]) / (ngrid * denom)
        if p >= 1.0:
            n_iter_ep = 1
        elif p <= 0:
            n_iter_ep = 1
        else:
            n_iter_ep = int(
                np.ceil(np.log(0.01) / np.log(1.0 - p))
            )
    return ngrid, n_iter_ep, nt_new, args["batch_size"], D_N_P_new


def select_subset(
    args,
    x: np.ndarray,
    i_grid: np.ndarray,
    i_t,
    rho: int,
    *,
    c=None,
    tuple_out=False,
    has_grad=False,
    warm_up=0,
):
    nx = x.shape[-1]
    if i_t is not None:
        batch_size = i_grid.shape[0]
        x_tensor = torch.zeros([rho + warm_up, batch_size, nx], requires_grad=has_grad)
        for k in range(batch_size):
            idx = np.arange(i_t[k] - warm_up, i_t[k] + rho)
            temp = x[idx, i_grid[k] : i_grid[k] + 1, :]
            x_tensor[:, k : k + 1, :] = torch.from_numpy(np.ascontiguousarray(temp))
    else:
        if len(x.shape) == 2:
            x_tensor = torch.from_numpy(x[i_grid, :]).float()
        else:
            x_tensor = torch.from_numpy(x[:, i_grid, :]).float()
            rho = x_tensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho + warm_up, axis=1
        )
        c_tensor = torch.from_numpy(temp).float()
        if tuple_out:
            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()
                c_tensor = c_tensor.cuda()
            out = (x_tensor, c_tensor)
        else:
            out = torch.cat((x_tensor, c_tensor), 2)
    else:
        out = x_tensor

    if type(out) is not tuple:
        out = out.to(args["device"])
    return out


def random_index_percentage(ngrid, dim_subset, D_N_P_new: pd.DataFrame, warm_up=0):
    batch_size, rho = dim_subset
    no_days = D_N_P_new["no_days"].to_numpy()
    eligible = np.where(no_days > (rho + warm_up))[0]
    if eligible.size == 0:
        eligible = np.arange(ngrid)

    p = np.array(D_N_P_new["day_percent"].to_numpy(), dtype=float)
    p = np.where(np.isfinite(p) & (p > 0), p, 0.0)
    p_masked = p.copy()
    p_masked[np.setdiff1d(np.arange(ngrid), eligible)] = 0.0
    if p_masked.sum() <= 0:
        p_masked = np.zeros(ngrid, dtype=float)
        p_masked[eligible] = 1.0 / float(eligible.size)
    else:
        p_masked = p_masked / p_masked.sum()

    i_grid = np.random.choice(np.arange(0, ngrid), size=batch_size, p=p_masked)
    i_t = []
    for i in i_grid:
        nt_b = D_N_P_new.iloc[i]["no_days"]
        offset = D_N_P_new.iloc[i]["offset"]
        low = 0 + warm_up
        high = int(nt_b - rho)
        if high <= low:
            T = offset + low
        else:
            T = offset + int(np.random.randint(low, high, size=1)[0])
        i_t.append(T)
    return i_grid, i_t


def take_sample_train(
    args, dataset_dictionary: dict, ngrid_train: int, nt: int, batch_size: int, D_N_P_new
):
    dim_subset = [batch_size, args["rho"]]
    i_grid, i_t = random_index_percentage(
        ngrid_train, dim_subset, D_N_P_new, warm_up=args["warm_up"]
    )
    sample = {"iGrid": i_grid}
    sample["inputs_NN_scaled"] = select_subset(
        args,
        dataset_dictionary["inputs_NN_scaled"],
        i_grid,
        i_t,
        args["rho"],
        has_grad=False,
        warm_up=args["warm_up"],
    )
    sample["c_NN"] = torch.tensor(
        dataset_dictionary["c_NN"][i_grid],
        device=args["device"],
        dtype=torch.float32,
    )
    w = args["warm_up"]
    sample["obs"] = select_subset(
        args,
        dataset_dictionary["obs"],
        i_grid,
        i_t,
        args["rho"],
        has_grad=False,
        warm_up=w,
    )[w:, :, :]
    sample["obs_scaled"] = select_subset(
        args,
        dataset_dictionary["obs_scaled"],
        i_grid,
        i_t,
        args["rho"],
        has_grad=False,
        warm_up=w,
    )[w:, :, :]
    return sample


def _build_numpy_tensors_for_period(
    scaled_data: dict,
    args: dict,
    *,
    forcing_key: str,
    tr_key: str,
) -> dict:
    """Build [nt, ngrid, nx] dict from a normalized forcing frame and ``args[tr_key]`` calendar."""
    forcing_df = scaled_data[forcing_key].copy()
    attr_df = scaled_data["attribute"].copy()
    dnp = scaled_data["DNP"].copy()

    site_col = args.get("site_col", "site_no")
    date_col = args.get("date_col", "date")
    forcing_df[date_col] = pd.to_datetime(forcing_df[date_col]).dt.tz_localize(None).dt.normalize()

    sites = dnp[site_col].tolist()
    ngrid = len(sites)
    t_all = t_range2_array(args[tr_key])
    t_index = pd.DatetimeIndex(pd.to_datetime(t_all).normalize())

    var_dyn = list(args["forcings"])
    var_attr = list(args["attributes"])
    targets = list(args["targets"])
    nt = len(t_index)
    nx_dyn = len(var_dyn)
    nx_attr = len(var_attr)

    x_arr = np.zeros((nt, ngrid, nx_dyn), dtype=np.float64)
    for vi, var in enumerate(var_dyn):
        pv = forcing_df.pivot_table(
            index=date_col, columns=site_col, values=var, aggfunc="first"
        )
        pv = pv.reindex(t_index)
        pv = pv.reindex(columns=sites)
        block = pv.to_numpy(dtype=float)
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        x_arr[:, :, vi] = block

    c_arr = np.zeros((ngrid, nx_attr), dtype=np.float64)
    for j, site in enumerate(sites):
        sub = attr_df[attr_df[site_col] == site]
        if len(sub) >= 1:
            c_arr[j, :] = sub.iloc[0][var_attr].to_numpy(dtype=float)

    n_tar = len(targets)
    obs_arr = np.full((nt, ngrid, n_tar), np.nan, dtype=np.float64)
    for vi, var in enumerate(targets):
        pv = forcing_df.pivot_table(
            index=date_col, columns=site_col, values=var, aggfunc="first"
        )
        pv = pv.reindex(t_index).reindex(columns=sites)
        obs_arr[:, :, vi] = pv.to_numpy(dtype=float)

    c_exp = np.repeat(np.expand_dims(c_arr, 0), nt, axis=0)
    inputs_nn = np.concatenate([x_arr, c_exp], axis=2).astype(np.float32)

    return {
        "inputs_NN_scaled": inputs_nn,
        "obs": obs_arr.astype(np.float32),
        "obs_scaled": obs_arr.astype(np.float32),
        "c_NN": c_arr.astype(np.float32),
    }


def build_numpy_training_tensors(scaled_data: dict, args, base_dir: Path) -> tuple[dict, pd.DataFrame]:
    """
    Build [nt, ngrid, nx] tensors from long-form parquets (sorted site/date),
    matching LSTM_Adaptive_TimeSeries layout for sampling.
    """
    _ = base_dir
    dnp = scaled_data["DNP"].copy()
    dataset_dict = _build_numpy_tensors_for_period(
        scaled_data, args, forcing_key="train_forcing", tr_key="t_train"
    )
    D_N_P_new = percentage_day_cal(dnp, args)
    return dataset_dict, D_N_P_new


def build_numpy_test_tensors(scaled_data: dict, args, base_dir: Path) -> dict:
    """Gap-truth test period tensors from ``test_forcing`` over ``t_test`` (same layout as train)."""
    _ = base_dir
    return _build_numpy_tensors_for_period(
        scaled_data, args, forcing_key="test_forcing", tr_key="t_test"
    )


def get_gap_test_row_indices(D_N_P: pd.DataFrame, args, basin_index: int) -> np.ndarray:
    """Indices along time dimension (rows of ``[nt, ngrid, :]`` tensors) for gap-test window."""
    if "S_Testing" not in D_N_P.columns or "E_Testing" not in D_N_P.columns:
        raise KeyError("D_N_P must define S_Testing and E_Testing for gap evaluation.")
    t_full = pd.DatetimeIndex(pd.to_datetime(t_range2_array(args["t_test"])).normalize())
    tstart = pd.Timestamp(D_N_P.iloc[basin_index]["S_Testing"]).normalize()
    tend = pd.Timestamp(D_N_P.iloc[basin_index]["E_Testing"]).normalize()
    mask = (t_full >= tstart) & (t_full <= tend)
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def take_sample_test(args, dataset_dictionary: dict, i_s: int, i_e: int, ind_time: np.ndarray) -> dict:
    """Slice one or more basins and the test time rows; move tensors to ``args['device']``."""
    out: dict = {}
    for key, arr in dataset_dictionary.items():
        if not hasattr(arr, "shape"):
            continue
        if len(arr.shape) == 3:
            sl = np.ascontiguousarray(arr[ind_time, i_s:i_e, :])
            out[key] = torch.from_numpy(sl).float().to(args["device"])
        elif len(arr.shape) == 2 and key == "c_NN":
            sl = np.ascontiguousarray(arr[i_s:i_e, :])
            out[key] = torch.from_numpy(sl).float().to(args["device"])
    return out


def save_train_sampling_meta(out_dir: Path, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_sampling_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=False)
