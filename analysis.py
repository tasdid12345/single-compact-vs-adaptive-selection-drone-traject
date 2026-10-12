"""Adaptive model selection vs one compact LightGBM for maneuvering-drone trajectory prediction."""
import json, os, sys, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from lightgbm import LGBMRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, DT, T, NTRAJ, W = 42, 0.1, 600, 300, 10   # 10 Hz, 60 s tracks, 1 s window
STEPS = [1, 5, 10]            # step 1 only used for model selection; 5/10 = 0.5 s / 1 s
SIG, K, SHARP = 0.3, 5, 3.0   # sensor noise (m), selection memory (steps), sharp threshold (m/s^2)
u = lambda x: x / np.linalg.norm(x)


def simulate(rng):
    """One trajectory: cruise / sharp turn / hard acceleration / stop segments. Returns noisy, true pos, |accel|."""
    p, v = rng.normal(0, 50, 3), u(rng.normal(size=3)) * rng.uniform(5, 15)
    Q, A = [], []
    while len(Q) < T:
        kind = rng.choice(4, p=[.6, .2, .12, .08])
        if np.linalg.norm(v) < .5: kind = 2          # restart after a stop
        ax, w = u(rng.normal(size=3) * [.2, .2, 1]), rng.choice([-1, 1]) * rng.uniform(.5, 1.6)
        acc = u(rng.normal(size=3)) * rng.uniform(4, 8)
        for _ in range(rng.integers(10, 40)):
            sp = np.linalg.norm(v)
            if kind == 0: vn = v + rng.normal(0, .5, 3) * DT
            elif kind == 1:
                th = w * DT
                vn = v * np.cos(th) + np.cross(ax, v) * np.sin(th) + ax * (ax @ v) * (1 - np.cos(th))
            elif kind == 2: vn = v + acc * DT if np.linalg.norm(v + acc * DT) < 25 else v
            else: vn = v - u(v) * min(6 * DT, sp) if sp > 1e-9 else v
            Q.append(p); A.append(np.linalg.norm(vn - v) / DT)
            p, v = p + (v + vn) / 2 * DT, vn
    Q = np.array(Q[:T])
    return Q + rng.normal(0, SIG, Q.shape), Q, np.array(A[:T])


def build(trajs):
    """Sliding windows of noisy positions + future true positions, next observation, sharp flag, traj id."""
    n = T - W - max(STEPS) + 1
    e = np.arange(n) + W - 1
    Ws, Ys, nx, sh, tid = [], [], [], [], []
    for k, (P, Q, A) in enumerate(trajs):
        Ws.append(swv(P, (W, 3))[:n, 0]); Ys.append(Q[e[:, None] + STEPS]); nx.append(P[e + 1])
        sh.append(swv(A, 10).mean(1)[W:W + n] > SHARP); tid.append(np.full(n, k))
    return [np.concatenate(x) for x in (Ws, Ys, nx, sh, tid)]


def poly_fit(seg, deg):
    """Least-squares polynomial in time (origin = last sample) for each window: (N, deg+1, 3)."""
    t = np.arange(-(seg.shape[1] - 1), 1) * DT
    return np.einsum("cl,nld->ncd", np.linalg.pinv(np.vander(t, deg + 1, increasing=True)), seg)


def poly_pred(Wn, L, deg, steps, damp=None):
    """Extrapolate positions at the given steps; optional exponentially damped velocity."""
    C, tau = poly_fit(Wn[:, -L:], deg), np.array(steps) * DT
    B = (np.stack([np.ones_like(tau), damp * (1 - np.exp(-tau / damp))], 1) if damp
         else tau[:, None] ** np.arange(deg + 1))
    return np.einsum("sk,nkd->nsd", B, C)


def rot(x, a):
    """Rotate (N,...,3) arrays about z by per-row angle a."""
    c, s = np.cos(a).reshape(-1, *[1] * (x.ndim - 2)), np.sin(a).reshape(-1, *[1] * (x.ndim - 2))
    return np.stack([c * x[..., 0] - s * x[..., 1], s * x[..., 0] + c * x[..., 1], x[..., 2]], -1)


def ridge_X(Wn): return (Wn - Wn[:, -1:])[:, :-1].reshape(len(Wn), -1)


def baseline_predict(ridge, Wn, tid, nx):
    """Pool = CV, CA, damped velocity, ridge AR; pick per window the model with lowest recent 1-step error."""
    out = np.stack([poly_pred(Wn, 5, 1, STEPS), poly_pred(Wn, 10, 2, STEPS),
                    poly_pred(Wn, 5, 1, STEPS, damp=1.5),
                    ridge.predict(ridge_X(Wn)).reshape(-1, 3, 3) + Wn[:, -1:]])      # (M,N,3,3)
    e = np.linalg.norm(out[:, :, 0] - nx, axis=-1)       # past 1-step errors vs observed positions
    s, c = np.zeros_like(e), np.zeros_like(e)
    for k in range(1, K + 1):                            # window j-k's target P[.+1] is already observed at j
        ok = tid[k:] == tid[:-k]
        s[:, k:] += np.where(ok, e[:, :-k], 0); c[:, k:] += ok
    pick = np.argmin(s / np.maximum(c, 1), axis=0)       # ties / no history -> constant velocity
    return out[pick, np.arange(len(pick))][:, 1:], out[:, :, 1:]


def prep(Wn):
    """Local frame: relative to last point, rotated about z so recent velocity points along +x."""
    last, rel = Wn[:, -1], Wn - Wn[:, -1:]
    yaw = np.arctan2(*poly_fit(rel[:, -5:], 1)[:, 1, [1, 0]].T)
    return last, yaw, rot(rel, -yaw)


def feats(L):
    """Velocity, acceleration, turn-rate features of the local window plus raw relative positions."""
    v3, v5, q = poly_fit(L[:, -3:], 1)[:, 1], poly_fit(L[:, -5:], 1)[:, 1], poly_fit(L, 2)
    v1, v2 = poly_fit(L[:, :5], 1)[:, 1], poly_fit(L[:, 5:], 1)[:, 1]
    nrm = lambda x: np.linalg.norm(x, axis=1, keepdims=True)
    turn = np.arccos(np.clip((v1 * v2).sum(1, keepdims=True) / (nrm(v1) * nrm(v2) + 1e-6), -1, 1)) / (5 * DT)
    return np.concatenate([L[:, :-1].reshape(len(L), -1), v3, v5, q[:, 1], q[:, 2], q[:, 0], v1, v2, v2 - v1,
                           np.cross(v1, v2), nrm(v1), nrm(v2), nrm(v5), turn,
                           poly_pred(L, 5, 1, STEPS[1:]).reshape(len(L), -1)], 1)


def improved_data(Wn, Y):
    last, yaw, L = prep(Wn)
    base = poly_pred(L, 5, 1, STEPS[1:])                  # constant-velocity prior in local frame
    return feats(L), (rot(Y[:, 1:] - last[:, None], -yaw) - base).reshape(len(Wn), 6)


def improved_predict(models, Wn):
    last, yaw, L = prep(Wn)
    res = np.stack([m.predict(feats(L)) for m in models], 1).reshape(-1, 2, 3)
    return last[:, None] + rot(poly_pred(L, 5, 1, STEPS[1:]) + res, yaw)


def evaluate(pred, Y, sharp):
    d = np.linalg.norm(pred - Y[:, 1:], axis=-1)          # (N,2) position error at 0.5 s, 1 s
    return {"all": d.mean(), "sharp": d[sharp].mean(), "h05": d[:, 0].mean(), "h10": d[:, 1].mean()}


def main():
    rng = np.random.default_rng(SEED)
    print("Simulating trajectories...", flush=True)
    trajs = [simulate(rng) for _ in range(NTRAJ)]
    tr_id, te_id = train_test_split(np.arange(NTRAJ), test_size=0.25, random_state=SEED)  # split by trajectory
    Wtr, Ytr, _, _, gtr = build([trajs[i] for i in tr_id])
    Wte, Yte, nxte, shte, gte = build([trajs[i] for i in te_id])
    print(f"train windows {len(Wtr)}, test windows {len(Wte)}, sharp share {shte.mean():.2f}", flush=True)

    # Baseline: ridge AR (alpha by grouped CV on train) inside the adaptive pool
    gs = GridSearchCV(Ridge(), {"alpha": [.1, 1, 10, 100, 1000]}, cv=GroupKFold(3), scoring="neg_mean_squared_error")
    gs.fit(ridge_X(Wtr), (Ytr - Wtr[:, -1:]).reshape(len(Wtr), -1), groups=gtr)
    ridge = gs.best_estimator_
    print("Ridge alpha", gs.best_params_, flush=True)

    # Improved: LightGBM hyper-parameters chosen by grouped CV on train (one representative target)
    sub = slice(None, None, 2)
    Xtr, Ttr = improved_data(Wtr[sub], Ytr[sub])
    base_p = dict(learning_rate=.05, min_child_samples=50, subsample=.8, subsample_freq=1,
                  colsample_bytree=.8, n_jobs=2, verbose=-1)
    gl = GridSearchCV(LGBMRegressor(random_state=SEED, **base_p), {"num_leaves": [7, 15, 31], "n_estimators": [150, 300]},
                      cv=GroupKFold(3), scoring="neg_mean_squared_error")
    gl.fit(Xtr, Ttr[:, 3], groups=gtr[sub])               # target: 1 s horizon, local forward axis
    print("LightGBM params", gl.best_params_, flush=True)

    B, I, lat_b, lat_i = [], [], [], []
    for seed in (42, 43, 44):                             # test set used once, at the end
        models = [LGBMRegressor(random_state=seed, **{**base_p, **gl.best_params_}).fit(Xtr, Ttr[:, j]) for j in range(6)]
        t0 = time.perf_counter(); pb, pool = baseline_predict(ridge, Wte, gte, nxte); lat_b.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); pi = improved_predict(models, Wte); lat_i.append(time.perf_counter() - t0)
        B.append(evaluate(pb, Yte, shte)); I.append(evaluate(pi, Yte, shte))
        print(f"seed {seed}: baseline {B[-1]['all']:.3f} m, improved {I[-1]['all']:.3f} m", flush=True)

    mean = lambda R, k: float(np.mean([r[k] for r in R])); std = lambda R, k: float(np.std([r[k] for r in R]))
    m = {"baseline_score": mean(B, "all"), "improved_score": mean(I, "all"),
         "baseline_score_std": std(B, "all"), "improved_score_std": std(I, "all")}
    m["improvement"] = m["baseline_score"] - m["improved_score"]
    for k in ("sharp", "h05", "h10"):
        m[f"baseline_{k}"], m[f"improved_{k}"] = mean(B, k), mean(I, k)
        m[f"baseline_{k}_std"], m[f"improved_{k}_std"] = std(B, k), std(I, k)
    m["baseline_latency_ms"] = float(np.mean(lat_b) / len(Wte) * 1e3)
    m["improved_latency_ms"] = float(np.mean(lat_i) / len(Wte) * 1e3)
    for name, o in zip(["cv", "ca", "damped", "ridge_ar"], pool):   # single pool members, for context
        r = evaluate(o, Yte, shte); m[f"pool_{name}_score"], m[f"pool_{name}_sharp"] = float(r["all"]), float(r["sharp"])
    m["n_sharp_test_windows"] = int(shte.sum())
    if not all(np.isfinite(list(m.values()))) or shte.sum() < 100:
        sys.exit("No meaningful metrics")

    os.makedirs("figures", exist_ok=True)
    x = np.arange(3); lab = ["all windows", "sharp windows", "1 s horizon"]
    plt.bar(x - .2, [m["baseline_score"], m["baseline_sharp"], m["baseline_h10"]], .4, label="adaptive selection")
    plt.bar(x + .2, [m["improved_score"], m["improved_sharp"], m["improved_h10"]], .4, label="LightGBM")
    plt.xticks(x, lab); plt.ylabel("mean position error (m)"); plt.legend(); plt.tight_layout()
    plt.savefig("figures/errors.png", dpi=120)

    res = {
        "question": "Can one compact gradient-boosting trajectory predictor beat an adaptive select-the-best-model-by-recent-error scheme at predicting a maneuvering target drone's future position, especially during sharp maneuvers?",
        "method": "Simulated 10 Hz 3D drone tracks with noise, split by whole trajectory (75/25). Baseline: adaptive pool (CV, CA, damped velocity, ridge AR) choosing the model with lowest error over the last 5 steps. Improved: compact LightGBM on velocity/acceleration/turn-rate features in a heading-aligned frame, predicting the residual over constant velocity. Hyper-parameters tuned by grouped CV on train only.",
        "dataset": {"name": "Simulated 3D drone trajectories", "source_url": "generated in analysis.py with numpy (no download)",
                    "n_rows": int(len(Wtr) + len(Wte)), "n_train": int(len(Wtr)), "n_test": int(len(Wte)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "Mean 3D position error (m) averaged over 0.5 s and 1 s horizons on test trajectories",
        "higher_is_better": False,
        "baseline": "Adaptive model selection over constant-velocity, constant-acceleration, damped-velocity and ridge-AR predictors (not the paper's exact code).",
        "improved": "LightGBM regressors (6 small trees ensembles: 2 horizons x 3 axes) on engineered kinematic features in a heading-aligned frame.",
        "metrics": m,
        "notes": "Synthetic data: conclusions depend on the simulator's maneuver statistics and noise (0.3 m). Targets are true (noise-free) future positions; inputs are noisy. Sharp windows = mean true acceleration over the next 1 s above 3 m/s^2 (used only for evaluation). 'One' LightGBM model is a set of 6 output-wise regressors with shared hyper-parameters tuned on one target. Latency is batch-amortised per prediction on the same test windows (features included), 2 CPU threads for LightGBM. Baseline is deterministic (std 0); LightGBM seeds 42-44 differ only in subsampling. Windows overlap within a trajectory, so errors are correlated.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
