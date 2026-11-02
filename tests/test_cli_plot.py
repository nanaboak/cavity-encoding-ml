import pytest

from cavity_encoding_ml import cli


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "0.1.0" in capsys.readouterr().out


def test_example_runs(capsys):
    assert cli.main(["example"]) == 0
    out = capsys.readouterr().out
    assert "STEP 1" in out and "STEP 4" in out and "worth keeping" in out


def test_diagnose(tmp_path, capsys, toy):
    p = tmp_path / "d.csv"
    toy.to_csv(p, index=False)
    assert cli.main(["diagnose", str(p), "--shot", "shot", "--cavity", "cav", "--response", "y"]) == 0
    assert "PASSED" in capsys.readouterr().out


def test_run_with_config(tmp_path, capsys, toy, toy_cfg):
    toy.to_csv(tmp_path / "d.csv", index=False)
    toy_cfg.to_json(str(tmp_path / "c.json"))
    assert cli.main(["run", str(tmp_path / "d.csv"), "--config", str(tmp_path / "c.json")]) == 0
    assert "STEP 3" in capsys.readouterr().out


def test_plot_decision(toy, toy_cfg, ridge):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from cavity_encoding_ml import CavityWorkflow
    from cavity_encoding_ml.plotting import plot_decision

    ax = plot_decision(CavityWorkflow(toy_cfg, ridge).run(toy))
    assert ax.get_title() == "use cavity"
