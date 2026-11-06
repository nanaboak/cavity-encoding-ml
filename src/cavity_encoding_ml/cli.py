"""Command-line interface: ``cavity-encoding-ml``."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from .datasets import load_tip_example, tip_example_config
from .diagnostic import cavity_diagnostic
from .mapper import CavityConfig
from .workflow import CavityWorkflow


def _default_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler

    return make_pipeline(
        MinMaxScaler(),
        RandomForestRegressor(n_estimators=200, max_features="sqrt", max_depth=15, random_state=42, n_jobs=1),
    )


def _cmd_example(args):
    data, cfg = load_tip_example(), tip_example_config()
    res = CavityWorkflow(cfg, _default_model(), encoding=args.encoding, audit_kfold=True, response_unit="mm",
                       arms=("blind", "readings", "encoded")).run(data)
    print(res.report("mm"))
    return 0


def _cmd_diagnose(args):
    df = pd.read_csv(args.csv)
    d = cavity_diagnostic(df[args.response], df[args.shot], df[args.cavity])
    print(d.report())
    return 0


def _cmd_run(args):
    df = pd.read_csv(args.csv)
    cfg = CavityConfig.from_json(args.config)
    res = CavityWorkflow(cfg, _default_model(), encoding=args.encoding, margin=args.margin, audit_kfold=args.audit,
                       arms=("blind", "encoded") if args.no_readings else ("blind", "readings", "encoded")).run(df)
    print(res.report())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cavity-encoding-ml", description=__doc__)
    p.add_argument("--version", action="version", version=f"cavity-encoding-ml {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("example", help="run the whole workflow on the bundled synthetic tip data")
    e.add_argument("--encoding", default="onehot", choices=["onehot", "target", "integer", "auto"])
    e.set_defaults(func=_cmd_example)

    d = sub.add_parser("diagnose", help="step 1 only: F test of the cavity effect on a CSV file")
    d.add_argument("csv")
    d.add_argument("--shot", required=True)
    d.add_argument("--cavity", required=True)
    d.add_argument("--response", required=True)
    d.set_defaults(func=_cmd_diagnose)

    r = sub.add_parser("run", help="all four steps on a CSV file, with a random forest")
    r.add_argument("csv")
    r.add_argument("--config", required=True, help="JSON file written by CavityConfig.to_json")
    r.add_argument("--encoding", default="onehot", choices=["onehot", "target", "integer", "auto"])
    r.add_argument("--margin", type=float, default=2.0)
    r.add_argument("--readings", action="store_true", help="kept for compatibility, R1 is trained by default")
    r.add_argument("--no-readings", action="store_true", help="skip R1, the cavity readings without the label")
    r.add_argument("--audit", action="store_true", help="also measure the leakage of ordinary K-fold")
    r.set_defaults(func=_cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
