# FRE-GY-9743 Assignment 7

**Chuyu Wu — November 12, 2026**

This assignment implements a monotone-convex interpolator for instantaneous forward rates (IFRs), plots the supplied SOFR and FF curves, and compares FRA, RFR swap, and overnight index basis swap values against the starter piecewise-constant method.

## Files changed

- `fixedincomelib/utilities/numerics.py`: interpolation, flat/linear extrapolation, one-node handling, and analytical integration.
- `hw7_monotone_convex_interpolator.ipynb`: curve plot, product comparisons, and short method discussion.

## Run

Install the listed dependencies with `python -m pip install -r requirements.txt`, then open `hw7_monotone_convex_interpolator.ipynb` from this repository directory. Restart the kernel and choose **Run All**. The notebook uses the local `fixedincomelib` package.
