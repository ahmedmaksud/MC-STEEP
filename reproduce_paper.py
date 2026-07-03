"""Code for the figures of Secure Multi-Carrier Communication using STEEP

The main code lives in lib/OFDM_STEEP_2.py and lib/OFDM_STEEP_3.py;
this script is a thin parallel wrapper that evaluates each curve point in a separate process and redraws the figures into results/.
"""

import os

# Pin BLAS to one thread per worker before numpy is imported, so that process-level parallelism does not oversubscribe the cores.
for _v in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "1"

import argparse
import contextlib
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from lib import OFDM_STEEP_2 as m2
from lib import OFDM_STEEP_3 as m3

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
BASE_SEED = 12345

# alpha, beta, nE are common to every figure in the chapter.
ALPHA, BETA, NE = 3, 5, 4

# preset -> (R_upper, R_lower, R_bottom, N, point_stride, curve_stride)
# R_lower / R_bottom of None keep each module's own default.
PRESETS = {
    "smoke": (3, 5, 5, 8, 12, 3),
    "quick": (60, 15, 25, 50, 3, 2),
    "full": (2000, None, None, 50, 1, 1),
    "rich": (5000, None, None, 50, 1, 1),
}

MODULES = {"m2": m2, "m3": m3}

DUMP = False


def _dump(name, **arrays):
    if DUMP:
        np.savez(os.path.join(RESULTS, f"data_{name}.npz"), **arrays)


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            yield


def _prepare(mod, ru, rl, rb, seed):
    mod.R_upper = ru
    if rl is not None:
        mod.R_lower = rl
    if rb is not None:
        mod.R_bottom = rb
    np.random.seed(seed)


# --- workers (top level so they pickle) ---------------------------------


def _w_nopower(job):
    key, sab, sba, N, R, seed = job
    mod = MODULES[key]
    _prepare(mod, *R, seed)
    with _quiet():
        return mod.get_rate_upper_nopower(
            mod.in_db(sab), mod.in_db(sba), ALPHA, BETA, N, NE
        )


def _w_both(job):
    key, sab, sba, N, R, seed = job
    mod = MODULES[key]
    _prepare(mod, *R, seed)
    with _quiet():
        return mod.get_rate_upper_both(
            mod.in_db(sab), mod.in_db(sba), ALPHA, BETA, N, NE
        )


def _w_p5(job):
    key, sab, sba, N, R, seed = job
    mod = MODULES[key]
    _prepare(mod, *R, seed)
    with _quiet():
        return mod.get_rate_upper_5(mod.in_db(sab), mod.in_db(sba), ALPHA, BETA, NE, N)


def _w_50(job):
    key, sab, sba, N, R, seed = job
    mod = MODULES[key]
    _prepare(mod, *R, seed)
    with _quiet():
        return mod.get_rate_upper_50(mod.in_db(sab), mod.in_db(sba), ALPHA, BETA, NE, N)


def _w_5689(job):
    key, sab, sba, N, R, seed = job
    mod = MODULES[key]
    _prepare(mod, *R, seed)
    with _quiet():
        return mod.get_rate_upper_5689(
            mod.in_db(sab), mod.in_db(sba), ALPHA, BETA, NE, N
        )


# --- helpers ------------------------------------------------------------


def _stride(arr, k):
    if k <= 1:
        return arr
    idx = list(range(0, len(arr), k))
    if idx[-1] != len(arr) - 1:
        idx.append(len(arr) - 1)
    return arr[idx]


def _run(ex, worker, jobs, desc):
    out = [None] * len(jobs)
    futs = {ex.submit(worker, job): i for i, job in enumerate(jobs)}
    for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
        out[futs[fut]] = fut.result()
    return out


def _newfig():
    sns.set_style("whitegrid")
    return sns.color_palette("muted")[:4]


def _finish(name, xlabel, ylabel, title, legend_out=True):
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles and legend_out:
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    elif handles:
        plt.legend(loc="upper right")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title)
    path = os.path.join(RESULTS, name)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def _title():
    return f"$n_E={NE}$" + f"; $\\alpha={ALPHA}$" + f"; $\\beta={BETA}$"


# --- phases -------------------------------------------------------------


def phase_pplot1(ex, R, N, pstride, cstride, off):
    SABs = [15, 30]
    SBAs = _stride(np.linspace(-10, 20, 61), pstride)
    styles = ["--", "-"]
    colors = _newfig()
    markers = ["o", "*", "^"]
    seed = 0
    for si, SAB in enumerate(SABs):
        jobs = [
            ("m2", SAB, sba, N, R, BASE_SEED + off + seed + j)
            for j, sba in enumerate(SBAs)
        ]
        seed += len(jobs)
        res = np.array(_run(ex, _w_nopower, jobs, f"pplot1 SAB={SAB}"))
        _dump(f"pplot1_SAB{SAB}", SBAs=SBAs, p123=res)
        for p in range(3):
            plt.plot(
                SBAs,
                res[:, p],
                linestyle=styles[si],
                color=colors[p],
                marker=markers[p],
                label=f"policy-{p + 1};$S_{{AB}}=${SAB} dB",
            )
    _finish(
        "pplot1.png",
        "$S_{BA}$ in dB",
        "avg secrecy in single carrier",
        f"$n_E={NE}$; $N={N}$; $\\alpha={ALPHA}$; $\\beta={BETA}$",
    )


def phase_pplot2(ex, R, N, pstride, cstride, off):
    SAB = 20
    Ns = [50, 100] if N >= 50 else [N, N * 2]
    SBAs = _stride(np.linspace(-10, 20, 31), pstride)
    styles = ["-", "--"]
    colors = _newfig()
    markers = ["o", "*", "^"]
    seed = 0
    for si, Ncur in enumerate(Ns):
        jobs = [
            ("m2", SAB, sba, Ncur, R, BASE_SEED + off + seed + j)
            for j, sba in enumerate(SBAs)
        ]
        seed += len(jobs)
        res = np.array(_run(ex, _w_nopower, jobs, f"pplot2 N={Ncur}"))
        for p in range(3):
            plt.plot(
                SBAs,
                res[:, p],
                linestyle=styles[si],
                color=colors[p],
                marker=markers[p],
                label=f"policy-{p + 1};$N=${Ncur}",
            )
    _finish(
        "pplot2.png",
        "$S_{BA}$ in dB",
        "avg secrecy in single carrier",
        f"$n_E={NE}$; $S_{{AB}}=${SAB} dB; $\\alpha={ALPHA}$; " f"$\\beta={BETA}$",
    )


def phase_pplot3(ex, R, N, pstride, cstride, off):
    SABs = [15, 20]
    SBAs = _stride(np.linspace(-10, 20, 31), pstride)
    styles = ["--", "-"]
    colors = _newfig()
    seed = 0
    for si, SAB in enumerate(SABs):
        jobs = [
            ("m2", SAB, sba, N, R, BASE_SEED + off + seed + j)
            for j, sba in enumerate(SBAs)
        ]
        seed += len(jobs)
        res = np.array(_run(ex, _w_both, jobs, f"pplot3 SAB={SAB}"))
        plt.plot(
            SBAs,
            res[:, 0],
            linestyle=styles[si],
            color=colors[0],
            marker="o",
            label=f"policy-2; $S_{{AB}}=${SAB} dB",
        )
        plt.plot(
            SBAs,
            res[:, 1],
            linestyle=styles[si],
            color=colors[1],
            marker="*",
            label=f"policy-4; $S_{{AB}}=${SAB} dB",
        )
    _finish(
        "pplot3.png",
        "$S_{BA}$ in dB",
        "avg secrecy in single carrier",
        f"$n_E={NE}$; $N={N}$; $\\alpha={ALPHA}$; $\\beta={BETA}$",
    )


def phase_pplot4(ex, R, N, pstride, cstride, off):
    SABs = [10, 20]
    SBAs = _stride(np.linspace(-15, 15, 31), pstride)
    styles = ["-", "--"]
    colors = _newfig()
    markers = ["o", "*", "d"]
    seed = 0
    for si, SAB in enumerate(SABs):
        jobs = [
            ("m2", SAB, sba, N, R, BASE_SEED + off + seed + j)
            for j, sba in enumerate(SBAs)
        ]
        seed += len(jobs)
        res = np.array(_run(ex, _w_both, jobs, f"pplot4 SAB={SAB}"))
        _dump(f"pplot4_SAB{SAB}", SBAs=SBAs, p245=res)
        for p, name in enumerate(["policy-2", "policy-4", "policy-5"]):
            plt.plot(
                SBAs,
                res[:, p],
                linestyle=styles[si],
                color=colors[p],
                marker=markers[p],
                label=f"{name}; $S_{{AB}}=${SAB} dB",
            )
    _finish(
        "pplot4.png",
        "$S_{BA}$ in dB",
        "avg secrecy in single carrier",
        f"$n_E={NE}$; $N={N}$; $\\alpha={ALPHA}$; $\\beta={BETA}$",
    )


def phase_plot5(ex, R, N, pstride, cstride, off):
    SABs = _stride(np.linspace(5, 30, 11), cstride)
    SBAs = _stride(np.linspace(-10, 20, 31), pstride)
    colors = _newfig()
    cpx = np.zeros(len(SABs))
    cpy = np.zeros(len(SABs))
    seed = 0
    for i, SAB in enumerate(SABs):
        jobs = [
            ("m3", SAB, sba, N, R, BASE_SEED + off + seed + j)
            for j, sba in enumerate(SBAs)
        ]
        seed += len(jobs)
        p5 = np.array(_run(ex, _w_p5, jobs, f"plot5 SAB={SAB:.0f}"))
        plt.plot(SBAs, p5, linestyle="-", color=colors[0])
        cpy[i] = np.max(p5)
        cpx[i] = SBAs[np.argmax(p5)]
    plt.plot(cpx, cpy, linestyle="-", marker="*", color=colors[1])
    _finish(
        "plot5.png",
        "$S_{BA}$ in dB",
        "avg secrecy in single carrier",
        f"$n_E={NE}$; $N={N}$; $\\alpha={ALPHA}$; $\\beta={BETA}$",
    )


def phase_plot50(ex, R, N, pstride, cstride, off):
    SABs = [10, 20]
    SBAs = _stride(np.linspace(-10, 20, 31), pstride)
    styles = ["-", "--"]
    colors = _newfig()
    seed = 0
    for si, SAB in enumerate(SABs):
        jobs = [
            ("m3", SAB, sba, N, R, BASE_SEED + off + seed + j)
            for j, sba in enumerate(SBAs)
        ]
        seed += len(jobs)
        res = np.array(_run(ex, _w_50, jobs, f"plot50 SAB={SAB}"))
        _dump(f"plot50_SAB{SAB}", SBAs=SBAs, p5=res[:, 0], classic=res[:, 1])
        plt.plot(
            SBAs,
            res[:, 0],
            linestyle=styles[si],
            color=colors[1],
            marker="*",
            label=f"policy-5; $p_{{B}}=${SAB} dB",
        )
        plt.plot(
            SBAs,
            res[:, 1],
            linestyle=styles[si],
            color=colors[2],
            marker="d",
            label=f"classic WTC; $p_{{B}}=${SAB} dB",
        )
    _finish(
        "plot50.png",
        "$p_{A}$ in dB",
        "mean $C_s$ per carrier",
        f"$n_E={NE}$; $N={N}$; $\\alpha={ALPHA}$; $\\beta={BETA}$",
        legend_out=False,
    )


def phase_plot5689(ex, R, N, pstride, cstride, off):
    SABs = _stride(np.array([15, 25, 35]), cstride)
    SBAs = _stride(np.linspace(-20, 20, 41), pstride)
    ls = ["-", "--", "-."]
    colors = _newfig()
    peak_sba = np.zeros(len(SABs))
    last8 = last9 = None
    seed = 0
    for j, SAB in enumerate(SABs):
        jobs = [
            ("m3", SAB, sba, N, R, BASE_SEED + off + seed + k)
            for k, sba in enumerate(SBAs)
        ]
        seed += len(jobs)
        res = _run(ex, _w_5689, jobs, f"plot5689 SAB={SAB}")
        p5 = np.array([r[0] for r in res])
        p6 = np.array([r[2] for r in res])
        last8 = np.array([r[4] for r in res])
        last9 = np.array([r[6] for r in res])
        _dump(
            f"plot5689_SAB{SAB}",
            SBAs=SBAs,
            p5=p5,
            p6=p6,
            ckey_uniform=last8,
            ckey_wf=last9,
        )
        plt.plot(
            SBAs,
            p5,
            linestyle=ls[j % len(ls)],
            color=colors[0],
            marker="o",
            label=f"policy-5; $S_{{AB}}=${SAB} dB",
        )
        plt.plot(
            SBAs,
            p6,
            linestyle=ls[j % len(ls)],
            color=colors[1],
            marker="*",
            label=f"policy-6; $S_{{AB}}=${SAB} dB",
        )
        peak_sba[j] = SBAs[np.argmax(p5)]
    plt.plot(
        SBAs,
        last8,
        linestyle="-",
        color=colors[2],
        marker="x",
        label="$C_{key}$; probing power: uniform",
    )
    plt.plot(
        SBAs,
        last9,
        linestyle="-",
        color=colors[3],
        marker="d",
        label="$C_{key}$; probing power: water-filling",
    )
    _finish(
        "plot5689.png",
        "$S_{BA}$ in dB",
        "mean AASR in bits per carrier per roundtrip",
        f"$n_E={NE}$; $N={N}$; $\\alpha={ALPHA}$; $\\beta={BETA}$",
    )

    # distributions at the probing power that maximises policy-5 AASR
    jobs = [
        ("m3", SAB, peak_sba[j], N, R, BASE_SEED + off + 90000 + j)
        for j, SAB in enumerate(SABs)
    ]
    dres = _run(ex, _w_5689, jobs, "plot5689 hist")
    for j, SAB in enumerate(SABs):
        _, d5, _, d6, _, d8, _, d9 = dres[j]
        _hist(SAB, peak_sba[j], d5, d6, d8, d9, colors)


def _hist(SAB, sba, d5, d6, d8, d9, colors):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    lo1, hi1 = min(d5.min(), d6.min()), max(d5.max(), d6.max())
    lo2, hi2 = min(d8.min(), d9.min()), max(d8.max(), d9.max())
    sba = np.round(sba, 1)
    for a, data, col, xl, ttl, (lo, hi) in [
        (
            ax[0, 0],
            d5,
            colors[0],
            "AASR",
            f"Dist. of AASR of policy-5; $S_{{AB}}={SAB}$ dB, " f"$S_{{BA}}={sba}$ dB",
            (lo1, hi1),
        ),
        (
            ax[1, 0],
            d6,
            colors[1],
            "AASR",
            f"Dist. of AASR of policy-6; $S_{{AB}}={SAB}$ dB, " f"$S_{{BA}}={sba}$ dB",
            (lo1, hi1),
        ),
        (
            ax[0, 1],
            d8,
            colors[2],
            "$C_{key}$",
            f"Dist. of $C_{{key}}$, uniform probing power; " f"$S_{{BA}}={sba}$ dB",
            (lo2, hi2),
        ),
        (
            ax[1, 1],
            d9,
            colors[3],
            "$C_{key}$",
            f"Dist. of $C_{{key}}$, water-filing probing power; "
            f"$S_{{BA}}={sba}$ dB",
            (lo2, hi2),
        ),
    ]:
        a.hist(data, bins=64, color=col, density=True)
        a.set_xlabel(xl)
        a.set_ylabel("Frequency")
        a.set_xlim(lo, hi)
        a.axvline(data.mean(), linestyle="--", color="black")
        a.set_title(ttl)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS, f"histogram_p5_p6_ckey_data_SAB_{SAB}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


PHASES = {
    "pplot1": (phase_pplot1, 0),
    "pplot2": (phase_pplot2, 1),
    "pplot3": (phase_pplot3, 2),
    "pplot4": (phase_pplot4, 3),
    "plot5": (phase_plot5, 4),
    "plot50": (phase_plot50, 5),
    "plot5689": (phase_plot5689, 6),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument(
        "--only", nargs="+", choices=list(PHASES), help="run only these phases"
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument(
        "--dump",
        action="store_true",
        help="also save the curve arrays as results/data_*.npz",
    )
    args = ap.parse_args()

    global DUMP
    DUMP = args.dump

    ru, rl, rb, N, pstride, cstride = PRESETS[args.preset]
    R = (ru, rl, rb)
    names = args.only if args.only else list(PHASES)
    os.makedirs(RESULTS, exist_ok=True)

    print(
        f"preset={args.preset} workers={args.workers} N={N} "
        f"R_upper={ru} phases={','.join(names)}"
    )

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name in names:
            fn, off = PHASES[name]
            fn(ex, R, N, pstride, cstride, off * 1000)
    print(f"done -> {RESULTS}")


if __name__ == "__main__":
    main()
