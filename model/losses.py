"""
Loss functions from L1DeepMETv2/model/net.py, adapted to Keras ops.
Same args and formulas; torch -> keras.ops; scatter_add replaced by sum over axis=1 when weights/particles_vis are (batch, num_particles).
"""
import keras
from keras import ops


@keras.saving.register_keras_serializable(package="transformet")
class ResponseTunedLoss(keras.losses.Loss):
    """MSE + L1 response correction term, adapted from L1DeepMETv2/model/net.py.

    y_true, y_pred: (B, 2) predicted and true MET vectors (GeV).

    Key differences from the original L1DeepMETv2 version:
      - signature is call(y_true, y_pred) → compatible with model.compile(loss=...)
      - uses ops.mean (not sum) for the response term → c is batch-size agnostic
      - works on the output MET vector directly, no per-particle intermediates needed
    """

    def __init__(self, c=5000.0, pt_threshold=50.0, name="response_tuned_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.c = float(c)
        self.pt_threshold = float(pt_threshold)

    def call(self, y_true, y_pred):
        # MSE term
        diff = y_pred - y_true
        mse = 0.5 * ops.mean(ops.sum(diff ** 2, axis=-1))

        # Response: |pred_MET| / |gen_MET|
        # eps^2 inside sqrt to avoid nan gradient at zero; pt_threshold >> eps so filtering unaffected
        eps = 1e-7
        scale_true = ops.sqrt(ops.sum(y_true ** 2, axis=-1) + eps ** 2)  # (B,)
        scale_pred = ops.sqrt(ops.sum(y_pred ** 2, axis=-1) + eps ** 2)  # (B,)
        response = scale_pred / scale_true  # (B,)

        above_thr = scale_true > self.pt_threshold  # (B,) — only penalize high-MET events

        under = ops.where(
            ops.logical_and(response < 1.0, above_thr),
            1.0 - response,
            ops.zeros_like(response),
        )
        over = ops.where(
            ops.logical_and(response > 1.0, above_thr),
            response - 1.0,
            ops.zeros_like(response),
        )
        response_term = self.c * ops.mean(under + over)

        return mse + response_term

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"c": self.c, "pt_threshold": self.pt_threshold})
        return cfg

@keras.saving.register_keras_serializable(package="transformet")
class L1METMLRecoilLoss(keras.losses.Loss):
    """Exact L1METMLv1 (Poon et al.) recoil-response loss.

    Faithful keras.ops port of their `custom_loss` `baseline_loss` branch:

        upar     = |MET_pred| - |MET_true|                 (absolute-response residual)
        dev      = ( sum_bins |sum_{i in bin} upar_i| ) / sum_i |MET_true_i|
        mae_loss = mae_weight * mean(|dpx| + |dpy|)
        mse_loss = mse_weight * mean(dpx^2 + dpy^2) / 1000
        loss     = mae_loss + mse_loss + respcorr_factor * dev   (dev only if add_respcorr)

    The `dev` term balances over- vs. under-response *within* genMET-pT bins
    [50-100, 100-200, 200-300, 300-400, >400] GeV / normFac: per bin it sums the
    signed residual upar, so over-response in some events cancels under-response in
    others, and only the *net* imbalance per bin is penalized. Events below
    50/normFac are excluded from the response term (but still enter MAE/MSE).

    Note: this differs from ResponseTunedLoss, which uses a per-event clipped
    response ratio with a single hard pt threshold. Here the response correction is
    a binned net-deviation (the original L1METML form the paper plots come from).

    Mathematical equivalence to the reference: the reference splits each bin's upar
    into >0 and <0 parts and sums them; sum(pos)+sum(neg) over a bin equals
    sum(all) over that bin, so the masked sum below is identical.

    y_true, y_pred: (B, 2) MET vectors, in the same GeV/normFac units the model
    outputs (set normFac to match your scale_momentum if MET is scaled).
    """
    _BIN_EDGES = ((50., 100.), (100., 200.), (200., 300.), (300., 400.), (400., 1e12))

    def __init__(self, normFac=1.0, mse_weight=1.0, mae_weight=1.0,
                 add_respcorr=True, respcorr_factor=1000.0,
                 name="l1metml_recoil_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.normFac = float(normFac)
        self.mse_weight = float(mse_weight)
        self.mae_weight = float(mae_weight)
        self.add_respcorr = bool(add_respcorr)
        self.respcorr_factor = float(respcorr_factor)

    def call(self, y_true, y_pred):
        px_truth = y_true[:, 0]
        py_truth = y_true[:, 1]
        px_pred = y_pred[:, 0]
        py_pred = y_pred[:, 1]

        eps = 1e-7  # inside sqrt to avoid nan gradient at zero; << 50/normFac so binning unaffected
        pt_truth = ops.sqrt(px_truth ** 2 + py_truth ** 2 + eps ** 2)   # (B,)
        pt_pred = ops.sqrt(px_pred ** 2 + py_pred ** 2 + eps ** 2)      # (B,)

        # absolute-response residual: |MET_pred| - |MET_true|
        upar = pt_pred - pt_truth   # (B,)

        nf = self.normFac
        norm = ops.sum(pt_truth) + eps
        bin_devs = []
        for lo, hi in self._BIN_EDGES:
            m = ops.logical_and(pt_truth > lo / nf, pt_truth < hi / nf)
            bin_sum = ops.sum(ops.where(m, upar, ops.zeros_like(upar)))
            bin_devs.append(ops.abs(bin_sum))
        dev = sum(bin_devs) / norm

        mae_loss = self.mae_weight * ops.mean(ops.abs(px_pred - px_truth) + ops.abs(py_pred - py_truth))
        mse_loss = self.mse_weight * ops.mean((px_pred - px_truth) ** 2 + (py_pred - py_truth) ** 2) / 1000.0
        loss = mae_loss + mse_loss
        if self.add_respcorr:
            loss = loss + self.respcorr_factor * dev
        return loss

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "normFac": self.normFac,
            "mse_weight": self.mse_weight,
            "mae_weight": self.mae_weight,
            "add_respcorr": self.add_respcorr,
            "respcorr_factor": self.respcorr_factor,
        })
        return cfg


def getdot(vx, vy):
    return ops.sum(vx * vy, axis=1)

def getscale(vx):
    return ops.sqrt(getdot(vx, vx))

# loss function for direct MET regression (model outputs MET): same formula as loss_fn, (y_true=genMET, y_pred=MET).
def loss_fn_met(y_true, y_pred, scale_momentum=1.):
    # gen MET = (px, py); uT = (-1)*genMET. Predicted MET -> pred uT = (-1)*y_pred.
    true_px = (-1) * y_true[:, 0] / scale_momentum
    true_py = (-1) * y_true[:, 1] / scale_momentum
    pred_px = (-1) * y_pred[:, 0] / scale_momentum
    pred_py = (-1) * y_pred[:, 1] / scale_momentum
    loss = 0.5 * ((pred_px - true_px) ** 2 + (pred_py - true_py) ** 2)
    return ops.mean(loss)

# loss function for weights only (same as L1 loss_fn)
#weights=y_pred, particles_vis=x, genMET=y_true
def loss_fn_weights(y_true, y_pred, particles_vis, scale_momentum=1.):
    # particles_vis: (batch, num_particles, nfeat) with px at 1, py at 2
    # gen MET = (px, py) of genMET; uT = (-1)*genMET
    genMET = y_true 
    weights = y_pred     

    px = particles_vis[:, :, 1]
    py = particles_vis[:, :, 2]

    true_px = (-1) * genMET[:, 0] / scale_momentum
    true_py = (-1) * genMET[:, 1] / scale_momentum

    # regressed uT: MET = (-1)*uT; ML weights are [0,1]
    uTx = ops.sum(weights * px, axis=1)
    uTy = ops.sum(weights * py, axis=1)

    loss = 0.5 * ((uTx - true_px) ** 2 + (uTy - true_py) ** 2)
    loss = ops.mean(loss)
    return loss


# loss function with response tune
def loss_fn_response_tune(weights, particles_vis, genMET, batch, c=5000, scale_momentum=1.):
    px = particles_vis[:, :, 1]
    py = particles_vis[:, :, 2]

    true_px = (-1) * genMET[:, 0] / scale_momentum
    true_py = (-1) * genMET[:, 1] / scale_momentum

    uTx = ops.sum(weights * px, axis=1)
    uTy = ops.sum(weights * py, axis=1)

    loss = 0.5 * ((uTx - true_px) ** 2 + (uTy - true_py) ** 2)
    loss = ops.mean(loss)

    # response correction
    v_true = ops.stack([true_px, true_py], axis=1)
    v_regressed = ops.stack([uTx, uTy], axis=1)

    # response = getdot(v_true, v_regressed) / getdot(v_true, v_true)  # dot product
    response = getscale(v_regressed) / getscale(v_true)  # ratio of the MET scale

    pT_thres = 50. / scale_momentum
    resp_pos = (response > 1.) & (getscale(v_true) > pT_thres)
    resp_neg = (response < 1.) & (getscale(v_true) > pT_thres)

    c = c / scale_momentum

    response_term = c * (ops.sum(ops.where(resp_neg, 1. - response, 0.)) + ops.sum(ops.where(resp_pos, response - 1., 0.)))

    loss = loss + response_term
    return loss


# loss function; loss normalized with genMETx and genMETy
def loss_fn_relative(weights, particles_vis, genMET, batch, c=5000):
    px = particles_vis[:, :, 1]
    py = particles_vis[:, :, 2]

    true_px = (-1) * genMET[:, 0]
    true_py = (-1) * genMET[:, 1]

    uTx = ops.sum(weights * px, axis=1)
    uTy = ops.sum(weights * py, axis=1)

    loss = 0.5 * (((uTx - true_px) / true_px) ** 2 + ((uTy - true_py) / true_py) ** 2)
    loss = ops.mean(loss)
    return loss


# loss function; loss normalized with genMET scale
def loss_fn_relative_genMET(weights, particles_vis, genMET, batch, c=5000):
    px = particles_vis[:, :, 1]
    py = particles_vis[:, :, 2]

    true_px = (-1) * genMET[:, 0]
    true_py = (-1) * genMET[:, 1]

    v_true = ops.stack([true_px, true_py], axis=1)
    true_uT = getscale(v_true)

    uTx = ops.sum(weights * px, axis=1)
    uTy = ops.sum(weights * py, axis=1)

    loss = 0.5 * (((uTx - true_px) ** 2 + (uTy - true_py) ** 2) / (true_uT ** 2))
    loss = ops.mean(loss)
    return loss


@keras.saving.register_keras_serializable(package="transformet")
class FlattenMETLoss(keras.losses.Loss):
    """MSE with per-event weights that flatten the genMET distribution.

    Bin weights computed from TTBar MC so each 20 GeV bin in [0, 160] GeV
    contributes equally to the gradient. Addresses the MSE bias toward
    high-MET events that dominate TTBar.
    """
    _BINS    = [0., 20., 40., 60., 80., 100., 120., 140., 160., 1e9]
    _WEIGHTS = [0.14890485, 0.05692836, 0.04364244, 0.04413395, 0.05471598,
                0.08185655, 0.13920728, 0.25549501, 0.17511559]

    def __init__(self, name="flatten_met_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        sq_err  = ops.sum((y_pred - y_true) ** 2, axis=-1)           # (B,)
        true_met = ops.sqrt(ops.sum(y_true ** 2, axis=-1) + 1e-14)   # (B,)
        w = ops.zeros_like(true_met)
        for i in range(len(self._BINS) - 1):
            mask = (true_met > self._BINS[i]) & (true_met <= self._BINS[i + 1])
            w = ops.where(mask, float(self._WEIGHTS[i]), w)
        return 0.5 * ops.mean(sq_err * w)

    def get_config(self):
        return super().get_config()


# loss function flatten MET
def loss_fn_flattenMET(weights, particles_vis, genMET, batch, sample_weight=None):
    px = particles_vis[:, :, 1]
    py = particles_vis[:, :, 2]

    true_px = (-1) * genMET[:, 0]
    true_py = (-1) * genMET[:, 1]

    uTx = ops.sum(weights * px, axis=1)
    uTy = ops.sum(weights * py, axis=1)

    if sample_weight is not None:
        binnings = [0, 20, 40, 60, 80, 100, 120, 140, 160, 1000]

        per_genMET_bin_weight = [0.14890485, 0.05692836, 0.04364244, 0.04413395, 0.05471598,
                                 0.08185655, 0.13920728, 0.25549501, 0.17511559]

        v_true = ops.stack([true_px, true_py], axis=1)
        true_uT = getscale(v_true)

        w = ops.zeros_like(true_uT)
        for idx in range(len(binnings) - 1):
            mask_uT = (true_uT > binnings[idx]) & (true_uT <= binnings[idx + 1])
            w = ops.where(mask_uT, per_genMET_bin_weight[idx], w)

        loss = 0.5 * (((uTx - true_px) ** 2 + (uTy - true_py) ** 2) * w)
        loss = ops.mean(loss)
    else:
        loss = 0.5 * ((uTx - true_px) ** 2 + (uTy - true_py) ** 2)
        loss = ops.mean(loss)

    return loss