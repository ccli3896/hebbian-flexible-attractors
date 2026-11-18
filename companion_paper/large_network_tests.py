"""Preliminary tests of the §3-§5 results in the large (1-50-2) PC network (companion paper, §6).

Same model and operating point as large_network_line.py. For each reward s, 200 seeds x T trials;
records theta every 100 trials and (theta, choice) every 10 trials. Writes big_tests.npz.
"""
import numpy as np, sys, time


def run(S=200, T=150_000, alpha=0.0164, beta=0.0072, eta=0.01, H=50, r=1.0, seed0=0):
    rng = np.random.default_rng(seed0)
    x0 = rng.uniform(0, 1, S); h = rng.uniform(0, 1, (S, H)); y = rng.uniform(0, 1, (S, 2))
    Wh = rng.uniform(0, 1, (S, H)); Wy = rng.uniform(0, 1, (S, 2, H))
    act = np.argmax(y, 1)
    th100, beh = [], []
    for t in range(T):
        if t % 10 == 0:
            theta = np.einsum('sh,sh->s', Wy[:, 0] - Wy[:, 1], Wh)
            beh.append(np.stack([theta, (act == 0).astype(float)], 1))
            if t % 100 == 0:
                th100.append(theta)
        s = np.where(act == 0, r, 0.0)
        ph = Wh * x0[:, None]; py = np.einsum('skh,sh->sk', Wy, h)
        nx0 = x0 + 2 * alpha * np.clip(s - x0, -5, 5); nh = h + 2 * alpha * np.clip(ph - h, -5, 5)
        ny = y + 2 * alpha * np.clip(py - y, -5, 5)
        x0 = np.maximum(0, nx0 + eta * rng.standard_normal(S)); h = np.maximum(0, nh + eta * rng.standard_normal((S, H)))
        y = np.maximum(0, ny + eta * rng.standard_normal((S, 2)))
        eh = h - Wh * x0[:, None]; ey = y - np.einsum('skh,sh->sk', Wy, h)
        Wh = np.maximum(0, Wh + beta * np.clip(eh * x0[:, None], -10, 10))
        Wy = np.maximum(0, Wy + beta * np.clip(ey[:, :, None] * h[:, None, :], -10, 10))
        act = np.argmax(y, 1)
    return np.array(th100), np.array(beh)


if __name__ == '__main__':
    out = {}
    for r in (0.2, 0.5, 1.0):
        t0 = time.time(); th, beh = run(r=r)
        out[f'th_{r}'] = th; out[f'beh_{r}'] = beh.astype(np.float32)
        print(f's={r}: {time.time()-t0:.0f}s', flush=True)
    t0 = time.time(); th, beh = run(S=48, T=50_000, eta=0.0, r=1.0)
    out['th_noiseless'] = th; out['beh_noiseless'] = beh.astype(np.float32)
    print(f'noiseless: {time.time()-t0:.0f}s', flush=True)
    np.savez('big_tests.npz', **out)
