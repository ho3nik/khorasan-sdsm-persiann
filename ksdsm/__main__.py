"""Command line: python -m ksdsm {demo,run,persiann-download,persiann-cells}"""
import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ksdsm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="make synthetic data and run the full pipeline")
    d.add_argument("--dir", default="demo_run")
    r = sub.add_parser("run", help="run the pipeline from a TOML config")
    r.add_argument("config")
    pdl = sub.add_parser("persiann-download", help="download and crop PERSIANN-CDR years")
    pdl.add_argument("--years", nargs=2, type=int, default=[1983, 2005])
    pdl.add_argument("--out", default="data/persiann")
    pc = sub.add_parser("persiann-cells", help="select cells and write a predictand CSV")
    pc.add_argument("--dir", default="data/persiann")
    pc.add_argument("--years", nargs=2, type=int, default=[1983, 2005])
    pc.add_argument("--n", type=int, default=46)
    pc.add_argument("--out", default="data/persiann_predictand.csv")
    pc.add_argument("--sites-out", default="data/persiann_sites.csv")
    a = ap.parse_args(argv)

    if a.cmd == "demo":
        from .synthetic import make_demo
        from .pipeline import run, summarise
        root = Path(a.dir); (root / "data").mkdir(parents=True, exist_ok=True)
        make_demo(root / "data")
        cfg = dict(predictand=str(root / "data/predictand.csv"), sites=str(root / "data/sites.csv"),
                   predictor_root=str(root / "data/predictors"), out_dir=str(root / "results"),
                   scenarios=["rcp26", "rcp45", "rcp85"],
                   periods=dict(calibration=[1975, 1990], validation=[1991, 2005], base=[1975, 2005], future=[2020, 2050]),
                   model=dict(grouping="season", n_ensemble=10, n_predictors=3, tune_bias=True))
        res = run(cfg)
        print(summarise(res["metrics"]).round(3).to_string())
        print(res["changes"].round(2).to_string(index=False))
        print(f"\nOutputs in {root / 'results'}")
    elif a.cmd == "run":
        from .pipeline import load_config, run, summarise
        res = run(load_config(a.config))
        print(summarise(res["metrics"]).round(3).to_string())
    elif a.cmd == "persiann-download":
        from .persiann import download_year
        for y in range(a.years[0], a.years[1] + 1):
            print("saved", download_year(y, a.out))
    elif a.cmd == "persiann-cells":
        from .persiann import load_years, select_cells, extract_cells
        from .io import write_predictand
        da = load_years(a.dir, range(a.years[0], a.years[1] + 1))
        cells = select_cells(da, a.n)
        cells.to_csv(a.sites_out, index=False)
        write_predictand(extract_cells(da, cells), a.out)
        print(f"wrote {a.out} and {a.sites_out}")


if __name__ == "__main__":
    main()
