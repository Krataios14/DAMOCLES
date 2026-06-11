"""Command line entry point: run a study from a YAML file.

    damtol config.yaml
    damtol config.yaml --sensitivity --plot out/
"""

from __future__ import annotations

import argparse
import sys

import yaml

from .study import build_study


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="damtol",
        description="Probabilistic damage tolerance from a YAML study file.")
    parser.add_argument("config", help="study definition, see examples/")
    parser.add_argument("--sensitivity", action="store_true",
                        help="also compute Sobol indices on log-life")
    parser.add_argument("--plot", metavar="DIR",
                        help="write report figures to this directory")
    args = parser.parse_args(argv)

    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)
        study = build_study(spec)
    except (OSError, yaml.YAMLError, KeyError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = study.run(sensitivity=args.sensitivity)
    print(result.summary())

    if args.plot:
        from .plots import save_all
        for path in save_all(result, args.plot):
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
