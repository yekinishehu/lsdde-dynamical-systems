# Preconditioned DC Algorithms via Discrete Dynamical Systems

Companion code for the manuscript

  Y. Shehu, Y. Yao, "Preconditioned DC Algorithms via Discrete Dynamical Systems:
  Stability, Phase Transitions, and Explicit Complexity", submitted to
  Journal of Scientific Computing (2026).

This repository reproduces the theory-validation experiments of Sections 5.4, 5.7 and 5.8
(variant comparison, period-2 line-search failure cycle, Moreau smoothing of nonsmooth
concave parts). The main large-scale application experiments of Section 5.1-5.3, 5.5-5.6
(SCAD-LS at m=2000/d=5000, Ginzburg-Landau segmentation, MRI segmentation) use the
authors' PyTorch implementation, to be added under `pytorch_experiments/`.

## Layout

    cycle_validation/simulate_cycle.py    Section 5.7, Table 7 (tab:cycle) and Figure fig_cycle;
                                          1D quadratic period-2 cycle, margin threshold,
                                          2D period-3 orbit, constant-step frozen-rate check.
    variant_comparison/variant_comparison.py   Section 5.4, Table 5 (tab:variant-comp),
                                          synthetic SCAD-LS m=50, d=100, kappa=1e2, 5 instances.
    variant_comparison/full_scale.py      Section 5.4, Table 6 (tab:variant-comp-full),
                                          SCAD-LS m=2000, d=5000, kappa=1e3, stationarity trajectories.
    moreau_smoothing/smoothed_capped_l1.py     Section 5.8, Table 8 (tab:smooth-g2),
                                          capped-l1-LS with nonsmooth concave part,
                                          Moreau envelope transfer inequality check.
    figures/                              generated figures (fig_cycle.pdf).
    pytorch_experiments/                  (to be added) PyTorch code for Tables 2-4, 6, 9-11.

## Requirements

    numpy >= 1.21, matplotlib >= 3.5

## Reproduction

    python cycle_validation/simulate_cycle.py
    python variant_comparison/variant_comparison.py
    python variant_comparison/full_scale.py          # ~2-5 minutes
    python moreau_smoothing/smoothed_capped_l1.py

All scripts are deterministic (fixed seeds) and print the values reported in the
corresponding tables. The acceptance test is scale-invariant in exact arithmetic; the
simulations renormalize iterates near machine underflow, which leaves the step sequence
and every reported ratio unchanged.
