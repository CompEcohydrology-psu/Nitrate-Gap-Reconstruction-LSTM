"""Training loss factory."""

import torch
import torch.nn.functional as F


class HuberLoss_nitrate(torch.nn.Module):
    """Smooth L1 (Huber) loss on nitrate (masked to finite observations)."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, args, y_sim, y_obs, igrid):
        varTar_NN = args["targets"]
        obs = y_obs[:, :, varTar_NN.index("nitrate_mg_l")]
        sim = y_sim.squeeze()
        mask = obs == obs
        if not mask.any():
            return y_sim.sum() * 0.0
        p = sim[mask]
        t = obs[mask]
        return F.smooth_l1_loss(p, t, beta=self.delta, reduction="mean")


class NSELoss_nitrate(torch.nn.Module):
    """
    Plain Nash-Sutcliffe as a minimization objective on scaled nitrate.

    For finite (obs, sim) pairs in the batch, minimizes SS_res / (SST + eps),
    which is (1 - NSE) when SST = sum((t - mean(t))^2) and SS_res = sum((p-t)^2).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()

        self.eps = float(eps)

    def forward(self, args, y_sim, y_obs, igrid):
        varTar_NN = args["targets"]
        obs = y_obs[:, :, varTar_NN.index("nitrate_mg_l")]
        sim = y_sim.squeeze()
        mask = obs == obs
        if not mask.any():
            return y_sim.sum() * 0.0
        t = obs[mask]
        p = sim[mask]
        mean_t = t.mean()
        sst = ((t - mean_t) ** 2).sum()
        if sst < self.eps:
            return y_sim.sum() * 0.0
        ss_res = ((p - t) ** 2).sum()
        return ss_res / (sst + self.eps)

class RmseLoss_comb(torch.nn.Module):
    def __init__(self, alpha=0.25, beta=1e-6):
        super(RmseLoss_comb, self).__init__()


        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, args, y_sim, y_obs, igrid):
        varTar_NN = args["targets"]
        obs_flow = y_obs[:, :, varTar_NN.index("nitrate_mg_l")]
        sim_flow = y_sim.squeeze()
        if len(obs_flow[obs_flow == obs_flow]) > 0:
            mask_flow1 = obs_flow == obs_flow
            p = sim_flow[mask_flow1]
            t = obs_flow[mask_flow1]
            p_log = torch.clamp(p, min=0.0)
            t_log = torch.clamp(t, min=0.0)
            rmse = torch.sqrt(((p - t) ** 2).mean())
            rmse_log = torch.sqrt(((torch.log(p_log + self.beta) - torch.log(t_log + self.beta)) ** 2).mean())
            loss_flow_total = (1 - self.alpha) * rmse + self.alpha * rmse_log

        else:
            loss_flow_total = y_sim.sum() * 0.0
        return loss_flow_total


def getlossfn(args):

    if args['loss_function']=='Huber':
        delta_param=args['loss_function_weights']['huber_delta']
        lossfn=HuberLoss_nitrate(delta=delta_param)

    elif args['loss_function']=='NSE':
        eps_param = args['loss_function_weights']['nse_eps']
        lossfn = NSELoss_nitrate(eps=float(eps_param))

    elif args['loss_function']=='Combined_RMSE':
        alpha_param = float(args['loss_function_weights']['rmse_alpha'])
        beta_param = float(args['loss_function_weights']['rmse_beta'])
        lossfn = RmseLoss_comb(alpha=alpha_param, beta=beta_param)

    else:
        raise ValueError(f"Unknown loss_function: {args['loss_function']}")

    return lossfn
