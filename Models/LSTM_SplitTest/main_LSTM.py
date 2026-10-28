"""Train and evaluate the gap-filling LSTM."""

import warnings


warnings.filterwarnings(
    "ignore",
    message=r".*RNN module weights are not part of single contiguous chunk of memory.*",
    category=UserWarning,
)

from Auxilliary import (
    read_config,
    create_outdir,
    load_data,
    norm_data,
    denorm_targets_array,
    resolve_data_path,
)
from train_functions import (
    prepare_train_sampling,
    prepare_test_sampling,
    create_NN_models,
    take_sample_train,
)
from loss import getlossfn
from sampling import get_gap_test_row_indices, take_sample_test, t_range2_array
from pathlib import Path
import argparse
import csv
import random
import numpy as np
import time
import yaml
import pandas as pd
import matplotlib.pyplot as plt

import torch


PLOT_DPI = 600
PLOT_FONT_FAMILY = "DejaVu Sans"
PLOT_FONT_SIZE = 8
PLOT_TICK_FONT_SIZE = 7.5
PLOT_LEGEND_FONT_SIZE = 7.5
PLOT_TITLE_FONT_SIZE = 9
PLOT_FIG_WIDTH_IN = 6.7
PLOT_FIG_HEIGHT_IN = 3.25
PLOT_TRAIN_COLOR = "#127475"
PLOT_TEST_COLOR = "#FA8B26"
PLOT_PRED_COLOR = "#99031E"

PLOT_TRAIN_SCATTER_SIZE = 5
PLOT_TEST_SCATTER_SIZE = 7


def _parse_randomseed(args: dict) -> int:
    """Read ``randomseed`` from config (int or one-element list from YAML)."""
    raw = args.get("randomseed", 0)
    if isinstance(raw, (list, tuple)):
        if not raw:
            return 0
        return int(raw[0])
    return int(raw)


def set_random_seed(seed: int, *, cudnn_deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU/GPU) for reproducible training."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _parse_cli():
    parser = argparse.ArgumentParser(description="Train / evaluate LSTM gap model.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML path (default: <project>/config.yaml). "
        "Relative paths are resolved from the project directory.",
    )
    parser.add_argument(
        "--save-epoch",
        type=int,
        default=None,
        help="Override config saveEpoch (checkpoint interval).",
    )
    parser.add_argument(
        "--log-epoch",
        type=int,
        default=None,
        help="Override config logEpoch (train_epoch_log.csv interval). Default 10.",
    )
    parser.add_argument(
        "--log-eval-nse",
        action="store_true",
        help="At each logEpoch, write median gap NSE (by site) to train_epoch_log.csv.",
    )
    parser.add_argument(
        "--skip-test-gap",
        action="store_true",
        help="Drop Test_Gap from Action (train + epoch log only).",
    )
    parser.add_argument(
        "--no-eval-all-saved",
        action="store_true",
        help="Test_Gap only at EPOCH_testing (ignore eval_gap_truth_all_saved).",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help="Place run outputs under Outputs/<subdir>/ instead of Outputs/ "
        "(e.g. Diagnostic_Runs).",
    )
    return parser.parse_args()


def apply_cli_overrides(args: dict, cli) -> dict:
    """Apply CLI flags onto the loaded YAML config (does not rewrite the YAML)."""
    if cli.save_epoch is not None:
        if cli.save_epoch < 1:
            raise ValueError("--save-epoch must be >= 1")
        args["saveEpoch"] = int(cli.save_epoch)
    if cli.log_epoch is not None:
        if cli.log_epoch < 1:
            raise ValueError("--log-epoch must be >= 1")
        args["logEpoch"] = int(cli.log_epoch)
    elif args.get("logEpoch") in (None, 0, ""):
        args["logEpoch"] = 10
    if cli.log_eval_nse:
        args["log_eval_nse"] = True
    if cli.skip_test_gap:
        actions = args.get("Action") or []
        if isinstance(actions, str):
            actions = [actions]
        args["Action"] = [a for a in actions if a != "Test_Gap"]
        if not args["Action"]:
            args["Action"] = ["Train"]
    if cli.no_eval_all_saved:
        args["eval_gap_truth_all_saved"] = False
    if cli.out_subdir:
        args["output_subdir"] = str(cli.out_subdir).strip()
    return args


def dump_run_configs(*, base_dir: Path, out_dir: Path, args: dict) -> Path:
    """
    Dump the main config + generated run metadata into one YAML for reproducibility.

    Writes: <out_dir>/run_configs.yaml
    Includes (when present):
      - base config.yaml (parsed)
      - args used in this run (with device resolved)
      - norm_stats.yaml
      - train_sampling_meta.yaml
    """
    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _read_yaml(p: Path):
        if not p.is_file():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _yaml_safe(obj):
        """Convert common runtime objects (torch.device/Path/numpy scalars) into YAML-safe types."""

        try:
            import torch

            if isinstance(obj, torch.device):
                return str(obj)
        except Exception:
            pass


        if isinstance(obj, Path):
            return str(obj)


        try:
            import numpy as np

            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except Exception:
            pass


        if isinstance(obj, dict):
            return {str(k): _yaml_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_yaml_safe(v) for v in obj]


        return obj

    cfg_src = args.get("config_path") or str(base_dir / "config.yaml")
    snap = {
        "config_yaml": _read_yaml(Path(cfg_src)),
        "args_runtime": _yaml_safe(args),
        "norm_stats": _read_yaml(out_dir / "norm_stats.yaml"),
        "train_sampling_meta": _read_yaml(out_dir / "train_sampling_meta.yaml"),
    }

    out_path = out_dir / "run_configs.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(snap, f, sort_keys=False)
    return out_path


def predict_sitewise(
    args: dict,
    model,
    ds: dict,
    *,
    dnp_df=None,
    gap_only: bool = False,
) -> np.ndarray:
    """
    Run inference site-by-site to keep memory bounded.

    Returns predictions in **scaled** space with shape [nt, ngrid, n_targets].

    - If gap_only=True, only predicts within each basin's [S_Testing, E_Testing] window
      (requires dnp_df). Other times remain NaN.
    - If gap_only=False, predicts for all times for all basins.
    """
    nt, ngrid, _ = ds["inputs_NN_scaled"].shape
    n_tar = len(args["targets"])
    y_pred = np.full((nt, ngrid, n_tar), np.nan, dtype=np.float64)

    if gap_only and dnp_df is None:
        raise ValueError("dnp_df is required when gap_only=True")

    for ii in range(ngrid):
        if gap_only:
            ind_time = get_gap_test_row_indices(dnp_df, args, ii)
            if ind_time.size == 0:
                continue
        else:
            ind_time = np.arange(nt, dtype=int)

        sample = take_sample_test(args, ds, ii, ii + 1, ind_time)
        with torch.no_grad():
            out = model(sample)
        out_np = out.detach().cpu().numpy()
        y_pred[ind_time, ii, :] = out_np[:, 0, :]

    return y_pred


def save_pred_obs_wide(
    *,
    out_parquet_path: Path,
    dates,
    sites,
    site_col: str,
    targets,
    y_pred_phys: np.ndarray,
    y_obs_phys: np.ndarray,
) -> pd.DataFrame:
    """
    Save predictions + observations into one parquet (all sites together).

    Output columns:
      [date, <site_col>, pred_<target...>, obs_<target...>]
    """
    out_parquet_path = Path(out_parquet_path)
    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    nt, ngrid, n_tar = y_pred_phys.shape
    if y_obs_phys.shape != (nt, ngrid, n_tar):
        raise ValueError("y_obs_phys must match y_pred_phys shape")
    if len(targets) != n_tar:
        raise ValueError("targets length must match last dim of predictions")
    if len(dates) != nt:
        raise ValueError("dates length must match nt")
    if len(sites) != ngrid:
        raise ValueError("sites length must match ngrid")

    n_rows = nt * ngrid
    date_rep = np.repeat(np.asarray(dates), ngrid)
    site_rep = np.tile(np.asarray(sites), nt)

    df = pd.DataFrame({"date": pd.to_datetime(date_rep), site_col: site_rep})
    for vi, name in enumerate(targets):
        df[f"pred_{name}"] = y_pred_phys[..., vi].reshape(n_rows)
        df[f"obs_{name}"] = y_obs_phys[..., vi].reshape(n_rows)

    df.to_parquet(out_parquet_path, index=False)
    return df


def filter_pred_obs_wide_by_mask(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """
    Filter a dataframe returned by `save_pred_obs_wide` using a boolean mask [nt, ngrid].

    Assumes row order is (t major, site minor): row_index = t*ngrid + gi.
    """
    keep = np.asarray(mask, dtype=bool).reshape(-1)
    if keep.shape[0] != len(df):
        raise ValueError("mask size must match number of rows in df")
    return df.loc[keep].reset_index(drop=True)


def save_data(
    *,
    out_parquet_path: Path,
    dates,
    sites,
    site_col: str,
    targets,
    y_pred_phys: np.ndarray,
    y_obs_phys: np.ndarray,
    is_test: np.ndarray,
) -> pd.DataFrame:
    """
    Save predictions + observations for both train/test points into one parquet.

    Output is in "wide" form:
      columns: [date, site_no, split, pred_<target...>, obs_<target...>]
      split:
        - "test": held-out truth points used for gap evaluation
        - "train": other rows with non-NaN observations
        - "missing": rows where observations are NaN (no truth)

    Shapes:
      y_pred_phys, y_obs_phys: [nt, ngrid, n_targets]
      is_test: [nt, ngrid] boolean
    """
    out_parquet_path = Path(out_parquet_path)
    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    nt, ngrid, n_tar = y_pred_phys.shape
    if y_obs_phys.shape != (nt, ngrid, n_tar):
        raise ValueError("y_obs_phys must match y_pred_phys shape")
    if is_test.shape != (nt, ngrid):
        raise ValueError("is_test must be [nt, ngrid]")
    if len(targets) != n_tar:
        raise ValueError("targets length must match last dim of predictions")
    if len(dates) != nt:
        raise ValueError("dates length must match nt")
    if len(sites) != ngrid:
        raise ValueError("sites length must match ngrid")


    n_rows = nt * ngrid
    date_rep = np.repeat(np.asarray(dates), ngrid)
    site_rep = np.tile(np.asarray(sites), nt)


    has_truth = np.any(np.isfinite(y_obs_phys), axis=2)
    split = np.full((nt, ngrid), "missing", dtype=object)
    split[has_truth] = "train"
    split[is_test] = "test"
    split_rep = split.reshape(-1)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(date_rep),
            site_col: site_rep,
            "split": split_rep,
        }
    )


    for vi, name in enumerate(targets):
        df[f"pred_{name}"] = y_pred_phys[..., vi].reshape(n_rows)
        df[f"obs_{name}"] = y_obs_phys[..., vi].reshape(n_rows)

    df.to_parquet(out_parquet_path, index=False)
    return df


def plot_data_site_series(
    *,
    out_dir: Path,
    df: pd.DataFrame,
    site_col: str,
    date_col: str,
    targets,
) -> None:
    """Create per-site PNG plots with different colors for train vs test points."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    style = _matplotlib_plot_style()

    for site, sdf in df.groupby(site_col, sort=False):
        with plt.rc_context(style):
            sdf = sdf.sort_values(date_col)
            n_tar = len(targets)
            fig, axes = plt.subplots(
                n_tar,
                1,
                figsize=(PLOT_FIG_WIDTH_IN, max(PLOT_FIG_HEIGHT_IN, PLOT_FIG_HEIGHT_IN * n_tar)),
                sharex=True,
            )
            if n_tar == 1:
                axes = [axes]

            is_train = sdf["split"].to_numpy() == "train"
            is_test = sdf["split"].to_numpy() == "test"


            for ax, name in zip(axes, targets):
                y_obs = sdf[f"obs_{name}"].to_numpy()
                y_pred = sdf[f"pred_{name}"].to_numpy()
                x = sdf[date_col]


                mp = np.isfinite(y_pred)
                if np.any(mp):
                    ax.plot(
                        x[mp],
                        y_pred[mp],
                        lw=0.9,
                        alpha=0.95,
                        c=PLOT_PRED_COLOR,
                        label="Predicted data",
                        zorder=2,
                    )
                ax.scatter(
                    x[is_train],
                    y_obs[is_train],
                    s=PLOT_TRAIN_SCATTER_SIZE,
                    alpha=0.85,
                    c=PLOT_TRAIN_COLOR,
                    label="Observed data (training)",
                    zorder=3,
                )
                ax.scatter(
                    x[is_test],
                    y_obs[is_test],
                    s=PLOT_TEST_SCATTER_SIZE,
                    alpha=0.9,
                    c=PLOT_TEST_COLOR,
                    label="Observed data (testing)",
                    zorder=4,
                )

                ax.set_title(f"{site} - {name}")
                ax.grid(True, alpha=0.3)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=3,
                frameon=False,
            )
            fig.tight_layout()
            fig.savefig(out_dir / f"{site}.png", dpi=PLOT_DPI, bbox_inches="tight")
            plt.close(fig)


def _matplotlib_plot_style() -> dict:
    """Matplotlib rcParams for gap-eval site plots."""
    return {
        "font.family": PLOT_FONT_FAMILY,
        "font.size": PLOT_FONT_SIZE,
        "axes.titlesize": PLOT_TITLE_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_TICK_FONT_SIZE,
        "ytick.labelsize": PLOT_TICK_FONT_SIZE,
        "legend.fontsize": PLOT_LEGEND_FONT_SIZE,
    }


def plot_train_test_site_series(
    *,
    out_dir: Path,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    site_col: str,
    date_col: str,
    targets,
) -> None:
    """Per-site PNGs overlaying train vs test periods (obs + pred)."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sites = pd.Index(df_train[site_col].unique()).union(pd.Index(df_test[site_col].unique()))

    style = _matplotlib_plot_style()

    for site in sites:
        with plt.rc_context(style):
            tr0 = df_train[df_train[site_col] == site].sort_values(date_col)
            te0 = df_test[df_test[site_col] == site].sort_values(date_col)


            obs_cols = [f"obs_{t}" for t in targets]

            def _finite_obs_dates(d: pd.DataFrame) -> pd.Series:
                if d.empty:
                    return pd.Series([], dtype="datetime64[ns]")
                m = np.zeros(len(d), dtype=bool)
                for c in obs_cols:
                    if c in d.columns:
                        m |= np.isfinite(d[c].to_numpy())
                return d.loc[m, date_col]

            dt_tr = _finite_obs_dates(tr0)
            dt_te = _finite_obs_dates(te0)
            dt_all = pd.concat([dt_tr, dt_te], ignore_index=True)
            if len(dt_all) > 0:
                dmin = dt_all.min()
                dmax = dt_all.max()
                tr = tr0[(tr0[date_col] >= dmin) & (tr0[date_col] <= dmax)]
                te = te0[(te0[date_col] >= dmin) & (te0[date_col] <= dmax)]
            else:
                tr, te = tr0, te0

            n_tar = len(targets)
            fig, axes = plt.subplots(
                n_tar,
                1,
                figsize=(PLOT_FIG_WIDTH_IN, max(PLOT_FIG_HEIGHT_IN, PLOT_FIG_HEIGHT_IN * n_tar)),
                sharex=True,
                constrained_layout=True,
            )
            if n_tar == 1:
                axes = [axes]

            fig.suptitle(str(site), fontsize=PLOT_TITLE_FONT_SIZE)

            for ax, name in zip(axes, targets):
                xtr = tr[date_col]
                obs_tr_col = f"obs_{name}"
                xte = te[date_col]
                obs_te_col = f"obs_{name}"
                pred_te_col = f"pred_{name}"


                if pred_te_col in te.columns:
                    yte_p = te[pred_te_col].to_numpy()
                    mte_p = np.isfinite(yte_p)
                    if np.any(mte_p):
                        ax.plot(
                            xte[mte_p],
                            yte_p[mte_p],
                            lw=0.9,
                            alpha=0.95,
                            c=PLOT_PRED_COLOR,
                            label="Predicted data",
                            zorder=2,
                        )
                if obs_tr_col in tr.columns:
                    ytr_o = tr[obs_tr_col].to_numpy()
                    mtr_o = np.isfinite(ytr_o)
                    if np.any(mtr_o):
                        ax.scatter(
                            xtr[mtr_o],
                            ytr_o[mtr_o],
                            s=PLOT_TRAIN_SCATTER_SIZE,
                            alpha=0.85,
                            c=PLOT_TRAIN_COLOR,
                            label="Observed data (training)",
                            zorder=3,
                        )
                if obs_te_col in te.columns:
                    yte_o = te[obs_te_col].to_numpy()
                    mte_o = np.isfinite(yte_o)
                    if np.any(mte_o):
                        ax.scatter(
                            xte[mte_o],
                            yte_o[mte_o],
                            s=PLOT_TEST_SCATTER_SIZE,
                            alpha=0.9,
                            c=PLOT_TEST_COLOR,
                            label="Observed data (testing)",
                            zorder=4,
                        )

                ax.grid(True, alpha=0.3)
                ax.set_ylabel("Nitrate (mg/L as N)")

            axes[-1].set_xlabel("Date")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=3,
                frameon=False,
            )
            try:
                fig.tight_layout()
            except Exception:
                pass

            fig.savefig(out_dir / f"{site}.png", dpi=PLOT_DPI, bbox_inches="tight")
            plt.close(fig)

def _gap_nse_summary(args: dict, model, data: dict) -> tuple[float, float, int]:
    """
    Gap-window NSE for epoch logging.

    Returns (median site NSE, pooled overall NSE, n sites with finite NSE)
    for the first target in ``args["targets"]``.
    """
    test_ds = data["test_dataset"]
    dnp_df = data["DNP"]
    norm_stats = data["norm_stats"]
    site_col = args.get("site_col", "site_no")
    ngrid = test_ds["inputs_NN_scaled"].shape[1]
    sites = dnp_df[site_col].tolist() if site_col in dnp_df.columns else list(range(ngrid))
    tname = args["targets"][0]

    y_obs_phys = denorm_targets_array(test_ds["obs"], args, norm_stats)
    y_pred_scaled = predict_sitewise(args, model, test_ds, dnp_df=dnp_df, gap_only=True)
    y_pred_phys = denorm_targets_array(y_pred_scaled, args, norm_stats)
    metrics = gap_metrics(y_pred_phys, y_obs_phys, args, sites=sites)

    nses = []
    for site_dict in metrics.get("by_site", {}).values():
        nses.append(float(site_dict.get(tname, {}).get("nse", float("nan"))))
    nses = np.asarray(nses, dtype=float)
    n_ok = int(np.isfinite(nses).sum())
    median_nse = float(np.nanmedian(nses)) if n_ok else float("nan")
    overall_nse = float(metrics.get("overall", {}).get(tname, {}).get("nse", float("nan")))
    return median_nse, overall_nse, n_ok


def _init_epoch_log(path: Path, *, log_nse: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["epoch", "train_loss", "epoch_time_sec"]
    if log_nse:
        header += ["median_nse", "overall_nse", "n_sites_nse", "eval_time_sec"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def _append_epoch_log(path: Path, row: list) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
        f.flush()


def train_model(data, dir, args):

    niter=data['train_sampling_meta']['nIterEp']
    model=create_NN_models(args)
    lr = float(args.get("optimizer_lr", 1e-3))
    wd = float(args.get("optimizer_weight_decay", 1e-2))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossfn=getlossfn(args)
    model = model.to(args["device"])
    lossFun = lossfn.to(args["device"])

    log_every = int(args.get("logEpoch") or 0)
    save_every = int(args.get("saveEpoch") or 0)
    log_nse = bool(args.get("log_eval_nse", False)) and ("test_dataset" in data)
    log_path = Path(args["out_dir"]) / "train_epoch_log.csv"
    if log_every > 0:
        _init_epoch_log(log_path, log_nse=log_nse)
        print(
            f"Epoch log every {log_every}: {log_path}"
            + (" (train loss + gap median NSE)" if log_nse else " (train loss)")
        )
        if bool(args.get("log_eval_nse", False)) and not log_nse:
            print("  (warn) log_eval_nse set but test_dataset missing; NSE columns omitted")

    model.train()
    for epoch in range(1, args["EPOCHS"] + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(1, niter + 1):
            dataset_dictionary_sample = take_sample_train(args, data['train_dataset'], data['train_sampling_meta']['ngrid'], data['train_sampling_meta']['nt_tensor'], data['train_sampling_meta']['batch_size'],
                                                          data['D_N_P_processed'])

            out_model = model(dataset_dictionary_sample)


            w = int(args.get("warm_up", 0) or 0)
            if w > 0:
                out_model = out_model[w:, :, :]

            loss = lossFun(args, out_model,
                           dataset_dictionary_sample["obs_scaled"],
                           igrid=dataset_dictionary_sample["iGrid"])
            optim.zero_grad()
            loss.backward()
            clip = float(args.get("grad_clip_norm", 0.0) or 0.0)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optim.step()
            lossEp = lossEp + loss.item()

        lossEp = lossEp / niter
        t_epoch = time.time() - t0

        if args["device"].type == "cuda":
            mem_kb = int(torch.cuda.memory_allocated(device=args["device"]) * 0.001)
            logStr = "Epoch {} Loss {:.6f}, time {:.2f} sec, {} Kb allocated GPU memory".format(
                epoch, lossEp, t_epoch, mem_kb)
        else:
            logStr = "Epoch {} Loss {:.6f}, time {:.2f} sec (CPU)".format(
                epoch, lossEp, t_epoch)
        print(logStr)

        if log_every > 0 and (epoch % log_every == 0 or epoch == args["EPOCHS"]):
            row = [int(epoch), float(lossEp), float(t_epoch)]
            if log_nse:
                was_training = model.training
                model.eval()
                t1 = time.time()
                try:
                    with torch.no_grad():
                        med, overall, n_ok = _gap_nse_summary(args, model, data)
                except Exception as e:
                    print(f"  (warn) epoch {epoch} NSE log failed: {e}")
                    med, overall, n_ok = float("nan"), float("nan"), 0
                t_eval = time.time() - t1
                if was_training:
                    model.train()
                row += [med, overall, n_ok, t_eval]
                print(
                    "  gap NSE log: median={:.4f} overall={:.4f} n_sites={} ({:.1f}s)".format(
                        med, overall, n_ok, t_eval
                    )
                )
            _append_epoch_log(log_path, row)

        if save_every > 0 and epoch % save_every == 0:

            modelFile = Path(args["out_dir"], "model_Ep" + str(epoch) + ".pt")
            torch.save(model, modelFile)
        if epoch == args["EPOCHS"]:
            print("last epoch")
    print("Training ended")


def gap_metrics(
    pred_phys: np.ndarray,
    obs_phys: np.ndarray,
    args: dict,
    *,
    sites=None,
) -> dict:
    """
    Compute metrics for gap-evaluation.

    Returns a dict with:
      - overall: metrics aggregated across all sites (per target)
      - by_site: per-site metrics (per target)

    Inputs:
      pred_phys, obs_phys: [nt, ngrid, n_targets]
    """
    nt, ngrid, n_tar = pred_phys.shape
    if obs_phys.shape != (nt, ngrid, n_tar):
        raise ValueError("obs_phys must match pred_phys shape")

    if sites is None:
        sites = list(range(ngrid))
    if len(sites) != ngrid:
        raise ValueError("sites length must match ngrid")


    overall = {}
    for vi, name in enumerate(args["targets"]):
        p = pred_phys[..., vi].ravel()
        o = obs_phys[..., vi].ravel()
        m = np.isfinite(o) & np.isfinite(p)
        n = int(m.sum())
        if n < 2:
            overall[name] = {
                "nse": float("nan"),
                "rmse": float("nan"),
                "kge": float("nan"),
                "n": n,
            }
            continue
        po, oo = p[m], o[m]
        rmse = float(np.sqrt(np.mean((po - oo) ** 2)))
        sst = float(np.sum((oo - np.mean(oo)) ** 2))
        nse = (
            float(1.0 - np.sum((po - oo) ** 2) / sst)
            if sst > 1e-12
            else float("nan")
        )
        kge = _kge(po, oo)
        overall[name] = {"nse": nse, "rmse": rmse, "kge": kge, "n": n}


    by_site = {}
    for gi, site in enumerate(sites):
        site_out = {}
        for vi, name in enumerate(args["targets"]):
            p = pred_phys[:, gi, vi]
            o = obs_phys[:, gi, vi]
            m = np.isfinite(o) & np.isfinite(p)
            n = int(m.sum())
            if n < 2:
                site_out[name] = {
                    "nse": float("nan"),
                    "rmse": float("nan"),
                    "kge": float("nan"),
                    "n": n,
                }
                continue
            po, oo = p[m], o[m]
            rmse = float(np.sqrt(np.mean((po - oo) ** 2)))
            sst = float(np.sum((oo - np.mean(oo)) ** 2))
            nse = (
                float(1.0 - np.sum((po - oo) ** 2) / sst)
                if sst > 1e-12
                else float("nan")
            )
            kge = _kge(po, oo)
            site_out[name] = {"nse": nse, "rmse": rmse, "kge": kge, "n": n}
        by_site[str(site)] = site_out

    return {"overall": overall, "by_site": by_site}


def _kge(p: np.ndarray, o: np.ndarray) -> float:
    """
    Kling-Gupta Efficiency (KGE, 2009):
      KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 )
      r     = corr(p, o)
      alpha = std(p) / std(o)
      beta  = mean(p) / mean(o)
    """
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    if p.size < 2:
        return float("nan")
    mp = float(np.mean(p))
    mo = float(np.mean(o))
    sp = float(np.std(p))
    so = float(np.std(o))
    if not np.isfinite(sp) or not np.isfinite(so) or so <= 1e-12:
        return float("nan")
    if not np.isfinite(mp) or not np.isfinite(mo) or abs(mo) <= 1e-12:
        return float("nan")
    r = float(np.corrcoef(p, o)[0, 1])
    if not np.isfinite(r):
        return float("nan")
    alpha = sp / so
    beta = mp / mo
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def metrics_by_site_split(
    *,
    y_pred_phys: np.ndarray,
    y_obs_phys: np.ndarray,
    split: np.ndarray,
    sites,
    targets,
) -> dict:
    """
    Compute per-site metrics for each split ("train"/"test").

    Inputs:
      y_pred_phys, y_obs_phys: [nt, ngrid, n_targets]
      split: [nt, ngrid] object/str with values in {"train","test","missing"}
    """
    nt, ngrid, n_tar = y_pred_phys.shape
    out: dict = {}
    for gi, site in enumerate(sites):
        site_out = {}
        for split_name in ("train", "test"):
            mask_split = (split[:, gi] == split_name)
            per_target = {}
            for vi, tname in enumerate(targets):
                p = y_pred_phys[:, gi, vi]
                o = y_obs_phys[:, gi, vi]
                m = mask_split & np.isfinite(p) & np.isfinite(o)
                n = int(np.sum(m))
                if n < 2:
                    per_target[tname] = {
                        "n": n,
                        "rmse": float("nan"),
                        "nse": float("nan"),
                        "kge": float("nan"),
                    }
                    continue
                pp = p[m]
                oo = o[m]
                rmse = float(np.sqrt(np.mean((pp - oo) ** 2)))
                sst = float(np.sum((oo - np.mean(oo)) ** 2))
                nse = float(1.0 - np.sum((pp - oo) ** 2) / sst) if sst > 1e-12 else float("nan")
                kge = _kge(pp, oo)
                per_target[tname] = {"n": n, "rmse": rmse, "nse": nse, "kge": kge}
            site_out[split_name] = per_target
        out[str(site)] = site_out
    return out


def test_model(data, dir, args):
    if "train_dataset" not in data:
        raise KeyError(
            "train_dataset is missing. Build it (e.g., call prepare_train_sampling) "
            "before running Test_Gap so we can run inference on train_df too."
        )

    train_ds = data["train_dataset"]
    test_ds = data["test_dataset"]
    dnp_df = data["DNP"]
    norm_stats = data["norm_stats"]


    y_obs_test_phys = denorm_targets_array(test_ds["obs"], args, norm_stats)
    y_obs_train_phys = denorm_targets_array(train_ds["obs"], args, norm_stats)


    nt, ngrid, _ = test_ds["inputs_NN_scaled"].shape
    is_test = np.zeros((nt, ngrid), dtype=bool)
    for ii in range(ngrid):
        ind_time = get_gap_test_row_indices(dnp_df, args, ii)
        if ind_time.size:

            obs_block = test_ds["obs"][ind_time, ii, :]
            has_truth = np.any(np.isfinite(obs_block), axis=1)
            is_test[ind_time[has_truth], ii] = True

    if args.get("eval_gap_truth_all_saved"):
        epochs = list(range(args["saveEpoch"], args["EPOCHS"] + 1, args["saveEpoch"]))
    else:
        epochs = [int(args["EPOCH_testing"])]

    eval_root = Path(args["out_dir"]) / "gap_eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    site_col = args.get("site_col", "site_no")
    sites = dnp_df[site_col].tolist() if site_col in dnp_df.columns else list(range(ngrid))

    for epoch in epochs:
        model_path = Path(args["out_dir"]) / ("model_Ep" + str(epoch) + ".pt")
        if not model_path.is_file():
            print(f"Test_Gap: skip epoch {epoch}, missing {model_path}")
            continue
        model = torch.load(
            str(model_path),
            map_location=args["device"],
            weights_only=False,
        )
        model = model.to(args["device"])
        model.eval()
        print(f"Test_Gap: evaluating epoch {epoch}")


        y_pred_test_scaled_full = predict_sitewise(args, model, test_ds, gap_only=False)
        y_pred_train_scaled_full = predict_sitewise(args, model, train_ds, gap_only=False)
        y_pred_test_phys_full = denorm_targets_array(y_pred_test_scaled_full, args, norm_stats)
        y_pred_train_phys_full = denorm_targets_array(y_pred_train_scaled_full, args, norm_stats)


        y_pred_test_scaled_gap = predict_sitewise(args, model, test_ds, dnp_df=dnp_df, gap_only=True)
        y_pred_test_phys_gap = denorm_targets_array(y_pred_test_scaled_gap, args, norm_stats)
        metrics = gap_metrics(y_pred_test_phys_gap, y_obs_test_phys, args, sites=sites)
        ep_dir = eval_root / f"ep_{epoch}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        with open(ep_dir / "metrics.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)


        rows = []
        overall = metrics.get("overall", {})
        for tname, m in overall.items():
            rows.append(
                {
                    "epoch": int(epoch),
                    "scope": "overall",
                    "site": "",
                    "target": tname,
                    "n": m.get("n"),
                    "rmse": m.get("rmse"),
                    "nse": m.get("nse"),
                    "kge": m.get("kge"),
                }
            )
        by_site = metrics.get("by_site", {})
        for site, site_dict in by_site.items():
            for tname, m in site_dict.items():
                rows.append(
                    {
                        "epoch": int(epoch),
                        "scope": "site",
                        "site": site,
                        "target": tname,
                        "n": m.get("n"),
                        "rmse": m.get("rmse"),
                        "nse": m.get("nse"),
                        "kge": m.get("kge"),
                    }
                )
        pd.DataFrame(rows).to_csv(ep_dir / "metrics_sitewise.csv", index=False)
        np.savez_compressed(
            ep_dir / "pred_obs_gap.npz",
            y_pred=y_pred_test_phys_gap,
            y_obs=y_obs_test_phys,
        )


        test_dates = pd.to_datetime(np.asarray(t_range2_array(args["t_test"]))).normalize()
        train_dates = pd.to_datetime(np.asarray(t_range2_array(args["t_train"]))).normalize()

        df_test = save_pred_obs_wide(
            out_parquet_path=ep_dir / "pred_obs_test_all.parquet",
            dates=test_dates,
            sites=sites,
            site_col=site_col,
            targets=list(args["targets"]),
            y_pred_phys=y_pred_test_phys_full,
            y_obs_phys=y_obs_test_phys,
        )

        gap_mask = np.any(np.isfinite(y_pred_test_phys_gap), axis=2)
        df_test_gap = filter_pred_obs_wide_by_mask(df_test, gap_mask)
        df_test_gap.to_parquet(ep_dir / "pred_obs_test_gap.parquet", index=False)
        df_train = save_pred_obs_wide(
            out_parquet_path=ep_dir / "pred_obs_train_all.parquet",
            dates=train_dates,
            sites=sites,
            site_col=site_col,
            targets=list(args["targets"]),
            y_pred_phys=y_pred_train_phys_full,
            y_obs_phys=y_obs_train_phys,
        )


        plot_train_test_site_series(
            out_dir=ep_dir / "plots_by_site",
            df_train=df_train,
            df_test=df_test,
            site_col=site_col,
            date_col="date",
            targets=list(args["targets"]),
        )

        print(f"  wrote {ep_dir}  metrics={metrics}")

    print("Test_Gap finished")


def main():
    file_dir = Path(__file__).parent
    cli = _parse_cli()
    config_path = cli.config
    if config_path is not None:
        cfg_file = Path(config_path)
        if not cfg_file.is_absolute():
            cfg_file = file_dir / cfg_file
        args = read_config(file_dir, config_path=cfg_file)
    else:
        cfg_file = file_dir / "config.yaml"
        args = read_config(file_dir)
    args = apply_cli_overrides(args, cli)
    save_ep = int(args.get("saveEpoch") or 0)
    if args.get("eval_gap_truth_all_saved") and 0 < save_ep < 50:
        print(
            "WARNING: eval_gap_truth_all_saved=true with saveEpoch="
            f"{save_ep} will run full Test_Gap (plots+parquets) on every "
            "checkpoint. Use --no-eval-all-saved for epoch diagnostics."
        )
    train_seed = _parse_randomseed(args)
    cudnn_det = bool(args.get("cudnn_deterministic", False))
    set_random_seed(train_seed, cudnn_deterministic=cudnn_det)
    print(f"LSTM random seed: {train_seed}  (cudnn_deterministic={cudnn_det})")

    args["out_dir"] = create_outdir(file_dir, args)

    if args.get("device", "cpu") == "cuda" and torch.cuda.is_available():
        args["device"] = torch.device("cuda")
    else:
        args["device"] = torch.device("cpu")


    try:
        cfg_path = Path(args.get("config_path", file_dir / "config.yaml"))
        print(f"\n===== LSTM_SplitTest: {cfg_path.name} =====")
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                print(f.read().rstrip())
        else:
            print(f"(missing) {cfg_path}")
    except Exception as e:
        print(f"(warn) failed to print config.yaml: {e}")

    try:

        print("\n===== LSTM_GapTest: resolved args (runtime) =====")
        a = dict(args)
        if "device" in a:
            a["device"] = str(a["device"])
        if "out_dir" in a:
            a["out_dir"] = str(a["out_dir"])
        print(yaml.safe_dump(a, sort_keys=False).rstrip())
    except Exception:

        print({"out_dir": str(args.get("out_dir")), "device": str(args.get("device"))})

    print("\n===== LSTM_SplitTest: output paths =====")
    print(f"out_dir: {args['out_dir']}")
    print(f"device: {args['device']}")
    print(f"config_dir: {args.get('config_dir')}")
    print(f"train_forcing: {resolve_data_path(args['forcing_path_train'], args, file_dir)}")

    raw_data = load_data(file_dir, args)
    scaled_data = norm_data(raw_data, args["out_dir"], args)

    want_nse_log = bool(args.get("log_eval_nse")) and ("Train" in args["Action"])
    if "Train" in args["Action"] or "Test_Gap" in args["Action"]:

        scaled_data = prepare_train_sampling(scaled_data, args, file_dir)
    if "Test_Gap" in args["Action"] or want_nse_log:
        scaled_data = prepare_test_sampling(scaled_data, args, file_dir)


    dump_run_configs(base_dir=file_dir, out_dir=Path(args["out_dir"]), args=args)

    if "Train" in args["Action"]:
        train_model(scaled_data, file_dir, args)
    if "Test_Gap" in args["Action"]:
        test_model(scaled_data, file_dir, args)

if __name__=="__main__":
    main()
