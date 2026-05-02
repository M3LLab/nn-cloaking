"""Fit a Gaussian Mixture Model on (lambda, mu, rho) from a stiffness HDF5.

Saves a portable .npz that the optimisation loop can load on any machine to
apply a flat-top dataset-prior penalty: ``max(0, threshold - log p(C, rho))``.

The .npz contains:
    weights              (K,)            mixture weights
    means                (K, 3)          means in standardised (λ, μ, ρ) space
    covariances          (K, 3, 3)       full covariances (standardised space)
    precisions_cholesky  (K, 3, 3)       lower-triangular Cholesky of precision
                                         (sklearn convention; lets the JAX
                                         log-prob skip an inverse)
    feature_mean         (3,)            standardisation mean (raw → std)
    feature_std          (3,)            standardisation std
    threshold            scalar          τ used in the flat-top penalty
    threshold_percentile scalar          quantile of dataset log p chosen as τ
    n_components         int             K
    n_samples            int             dataset size used for the fit
    feature_order        ['lambda','mu','rho']
    bic                  scalar          for K-selection sanity
    log_p_quantiles      (5,)            (q1, q5, q25, q50, q75) of dataset log p

Usage
-----
    python -m dataset.cellular_chiral.fit_gmm \
        -i output/ca_bulk_squared/stiffness.h5 \
        -o output/ca_bulk_squared/gmm_lambda_mu_rho.npz \
        -K 16 \
        --threshold-percentile 0.25
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from sklearn.mixture import GaussianMixture

# Concurrent reads of an actively-written h5 are fine if locking is off.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--input", type=Path,
                    default=Path("output/ca_bulk_squared/stiffness.h5"),
                    help="Stiffness HDF5 produced by bulk_stiffness.py")
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("output/ca_bulk_squared/gmm_lambda_mu_rho.npz"),
                    help="Where to save the fitted GMM artifact")
    ap.add_argument("-K", "--n-components", type=int, default=16,
                    help="Number of GMM mixture components")
    ap.add_argument("--covariance-type", default="full",
                    choices=["full"],
                    help="Only 'full' is supported by the JAX-side log-prob.")
    ap.add_argument("--threshold-percentile", type=float, default=0.25,
                    help="Quantile of dataset log p below which the flat-top "
                         "penalty kicks in. 0.25 = bottom quartile, leaves a "
                         "comfortable margin so the optimiser is pushed away "
                         "from the borders of the manifold, not just its tails.")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bic-sweep", type=int, nargs="*", default=None,
                    help="Optional list of K values to evaluate BIC on, "
                         "before the final fit at -K. Just for diagnostics.")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"missing input: {args.input}")

    with h5py.File(args.input, "r") as f:
        lam = f["lambda_"][:]
        mu = f["mu"][:]
        rho = f["rho"][:]
    n = lam.size
    print(f"loaded {n} samples from {args.input}")

    X = np.column_stack([lam, mu, rho]).astype(np.float64)
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    Xs = (X - feature_mean) / feature_std
    print(f"standardisation: mean={feature_mean}, std={feature_std}")

    if args.bic_sweep:
        print("\nBIC sweep:")
        for k in args.bic_sweep:
            g = GaussianMixture(
                n_components=k, covariance_type=args.covariance_type,
                max_iter=args.max_iter, random_state=args.seed,
            ).fit(Xs)
            print(f"  K={k:3d}  BIC={g.bic(Xs):.1f}  AIC={g.aic(Xs):.1f}")

    print(f"\nfitting GMM (K={args.n_components}, cov={args.covariance_type}) ...")
    gmm = GaussianMixture(
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        max_iter=args.max_iter,
        random_state=args.seed,
        verbose=2,
    )
    gmm.fit(Xs)
    bic = float(gmm.bic(Xs))

    log_p = gmm.score_samples(Xs)
    quantiles = np.quantile(log_p, [0.01, 0.05, 0.25, 0.50, 0.75])
    threshold = float(np.quantile(log_p, args.threshold_percentile))
    print(
        f"\nlog_p stats:\n"
        f"  min={log_p.min():.3f}  q1={quantiles[0]:.3f}  q5={quantiles[1]:.3f}  "
        f"q25={quantiles[2]:.3f}  q50={quantiles[3]:.3f}  q75={quantiles[4]:.3f}  "
        f"max={log_p.max():.3f}"
    )
    print(f"threshold τ (q={args.threshold_percentile}): {threshold:.4f}")
    print(f"BIC: {bic:.1f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        weights=gmm.weights_.astype(np.float64),
        means=gmm.means_.astype(np.float64),
        covariances=gmm.covariances_.astype(np.float64),
        precisions_cholesky=gmm.precisions_cholesky_.astype(np.float64),
        feature_mean=feature_mean.astype(np.float64),
        feature_std=feature_std.astype(np.float64),
        threshold=np.float64(threshold),
        threshold_percentile=np.float64(args.threshold_percentile),
        n_components=np.int64(args.n_components),
        n_samples=np.int64(n),
        feature_order=np.array(["lambda", "mu", "rho"]),
        bic=np.float64(bic),
        log_p_quantiles=quantiles.astype(np.float64),
        covariance_type=np.array(args.covariance_type),
    )
    print(f"\nsaved GMM to {args.output}")


if __name__ == "__main__":
    main()
