from ksdsm.synthetic import make_demo
from ksdsm.pipeline import run


def test_demo_pipeline_runs(tmp_path):
    make_demo(tmp_path, n_sites=3)
    cfg = dict(predictand=str(tmp_path / "predictand.csv"), sites=str(tmp_path / "sites.csv"),
               predictor_root=str(tmp_path / "predictors"), out_dir=str(tmp_path / "out"),
               scenarios=["rcp45"], maps=False,
               periods=dict(calibration=[1975, 1990], validation=[1991, 2005], base=[1975, 2005], future=[2020, 2050]),
               model=dict(grouping="season", n_ensemble=3, n_predictors=3))
    res = run(cfg)
    met = res["metrics"]
    assert set(met.stage) == {"validation_ncep", "base_gcm"}
    v = met[(met.stage == "validation_ncep") & (met.statistic == "mean")]
    assert (v.nse > 0.5).all()               # synthetic data follow the model, so fit is good
    assert len(res["changes"]) == 3
    assert (tmp_path / "out" / "metrics.csv").exists()
