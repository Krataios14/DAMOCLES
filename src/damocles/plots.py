"""Report figures. All functions take a StudyResult and write a PNG."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_pof_curve(result, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    mask = result.pof_curve > 0
    ax.semilogy(result.pof_curve_cycles[mask], result.pof_curve[mask],
                color="tab:blue", lw=2, label="no inspection")
    if result.inspection is not None and result.inspection.times:
        for i, t in enumerate(result.inspection.times):
            ax.axvline(t, color="grey", ls=":", lw=1,
                       label="inspection" if i == 0 else None)
        ax.plot(result.service_cycles, result.inspection.pof_inspected, "v",
                color="tab:green", ms=10, label="inspected, end of service")
    if result.target_pof is not None:
        ax.axhline(result.target_pof, color="tab:red", ls="--", lw=1.5,
                   label="target")
    ax.set_xlabel("cycles")
    ax.set_ylabel("probability of failure")
    ax.set_title(result.name)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_life_histogram(result, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    finite = result.lives[np.isfinite(result.lives) & (result.lives > 0)]
    if finite.size:
        ax.hist(np.log10(finite), bins=80, color="tab:blue", alpha=0.75)
    ax.axvline(np.log10(result.service_cycles), color="tab:red", ls="--",
               lw=2, label="end of service")
    runouts = np.mean(~np.isfinite(result.lives))
    ax.set_xlabel("log10 cycles to failure")
    ax.set_ylabel("samples")
    ax.set_title(f"{result.name}  ({runouts:.1%} run-outs not shown)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pod(result, path):
    if result.inspection_plan is None:
        return
    pod = result.inspection_plan.pod
    fig, ax = plt.subplots(figsize=(8, 5))
    a = np.logspace(np.log10(pod.a50) - 1.5, np.log10(pod.a50) + 1.5, 300)
    ax.semilogx(a * 1e3, pod.pod(a), color="tab:blue", lw=2)
    ax.axhline(0.9, color="grey", ls=":", lw=1)
    ax.axvline(pod.a50 * 1e3, color="grey", ls=":", lw=1)
    ax.set_xlabel("crack size [mm]")
    ax.set_ylabel("probability of detection")
    ax.set_title("NDT capability")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_all(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for fn, name in ((plot_pof_curve, "pof_curve.png"),
                     (plot_life_histogram, "life_histogram.png"),
                     (plot_pod, "pod_curve.png")):
        path = os.path.join(out_dir, name)
        if fn is plot_pod and result.inspection_plan is None:
            continue
        fn(result, path)
        written.append(path)
    return written
