"""
EdgeLayer #3 - GPTQ demo
Naive round-to-nearest (RTN) vs GPTQ-style error compensation on one synthetic layer.

Everything here is synthetic. The layer is small so it runs on any laptop in seconds.
The point is the mechanism, not the exact numbers.

Run:  python edgelayer-03-gptq-demo.py
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. A synthetic layer: weights W (out x in) and inputs X (in x tokens)
#    Real activations are correlated across channels, so we build them that way.
# ---------------------------------------------------------------------------
d_out, d_in, n_cal, n_test = 256, 512, 2048, 2048

W = rng.normal(0, 0.02, size=(d_out, d_in))

rank = 64                                              # inputs share 64 hidden "themes"
mix = rng.normal(size=(d_in, rank)) / np.sqrt(rank)
chan_scale = np.exp(rng.normal(0, 0.5, size=d_in))    # some channels louder than others


def make_inputs(n):
    shared = mix @ rng.normal(size=(rank, n))          # correlated part
    noise = 0.3 * rng.normal(size=(d_in, n))           # independent part
    return chan_scale[:, None] * (shared + noise)


X_cal = make_inputs(n_cal)    # calibration set (what GPTQ gets to look at)
X_test = make_inputs(n_test)  # held-out inputs (what we score on)


# ---------------------------------------------------------------------------
# 2. Quantizer: asymmetric, one scale + zero-point per row (per group of columns)
# ---------------------------------------------------------------------------
def quant_params(w, bits):
    """w: (rows, cols) slice -> per-row scale and zero-point."""
    qmax = 2**bits - 1
    lo = np.minimum(w.min(axis=1), 0)
    hi = np.maximum(w.max(axis=1), 0)
    scale = (hi - lo) / qmax
    scale[scale == 0] = 1e-8
    zero = np.round(-lo / scale)
    return scale, zero, qmax


def quantize(w, scale, zero, qmax):
    q = np.clip(np.round(w / scale + zero), 0, qmax)
    return scale * (q - zero)


# ---------------------------------------------------------------------------
# 3. RTN: round every weight on its own
# ---------------------------------------------------------------------------
def rtn(W, bits, group=128):
    Q = np.empty_like(W)
    for g in range(0, W.shape[1], group):
        blk = W[:, g:g + group]
        s, z, qmax = quant_params(blk, bits)
        Q[:, g:g + group] = quantize(blk, s[:, None], z[:, None], qmax)
    return Q


# ---------------------------------------------------------------------------
# 4. GPTQ: round one column, push its error onto the columns not yet rounded
# ---------------------------------------------------------------------------
def gptq(W, X, bits, group=128, damp=0.01):
    W = W.copy()
    d_in = W.shape[1]
    H = 2 * X @ X.T                                    # how inputs co-vary
    H += damp * np.mean(np.diag(H)) * np.eye(d_in)     # small damping for stability
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T      # upper Cholesky of H^-1

    Q = np.zeros_like(W)
    for i in range(d_in):
        if i % group == 0:                             # new group: fresh scale
            s, z, qmax = quant_params(W[:, i:i + group], bits)
        w = W[:, i]
        q = quantize(w, s, z, qmax)
        Q[:, i] = q
        err = (w - q) / Hinv[i, i]
        W[:, i + 1:] -= np.outer(err, Hinv[i, i + 1:])  # the "nudge"
    return Q


# ---------------------------------------------------------------------------
# 5. Score: how far is the layer's OUTPUT from the original?
# ---------------------------------------------------------------------------
def out_err(Q, X):
    return np.linalg.norm(W @ X - Q @ X) / np.linalg.norm(W @ X)


def w_err(Q):
    return np.linalg.norm(W - Q) / np.linalg.norm(W)


if __name__ == "__main__":
    print(f"{'method':<10}{'bits':>5}{'output error':>15}{'weight error':>15}")
    for bits in (4, 3):
        Q_rtn = rtn(W, bits)
        Q_gptq = gptq(W, X_cal, bits)
        for name, Q in (("RTN", Q_rtn), ("GPTQ", Q_gptq)):
            print(f"{name:<10}{bits:>5}{out_err(Q, X_test):>14.1%}{w_err(Q):>15.1%}")

    # Same test with independent inputs: no correlation for GPTQ to use.
    ind = np.random.default_rng(5)
    make_ind = lambda n: chan_scale[:, None] * ind.normal(size=(d_in, n))
    Xc_ind, Xt_ind = make_ind(n_cal), make_ind(n_test)
    print("\nindependent inputs (output error):")
    for bits in (4, 3):
        e_rtn = out_err(rtn(W, bits), Xt_ind)
        e_gptq = out_err(gptq(W, Xc_ind, bits), Xt_ind)
        print(f"  {bits}-bit  RTN {e_rtn:.1%}   GPTQ {e_gptq:.1%}")
