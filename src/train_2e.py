from __future__ import annotations

"""
Skeleton for 2e PINN training (singlet/triplet) with Coulomb term and symmetry.
To be implemented after 1e validation.
"""

import argparse


def parse_args():
    ap = argparse.ArgumentParser(description="Train 2e PINN (skeleton)")
    ap.add_argument("--todo", action="store_true")
    return ap.parse_args()


def main():
    _ = parse_args()
    print("2e training skeleton: TODO (define symmetric/antisymmetric ansatz, quadrature in 4D, Coulomb weight gamma)")


if __name__ == "__main__":
    main()

