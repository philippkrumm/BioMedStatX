import json
import os
import pandas as pd
import pytest
from analysis.emm_posthoc import split_plot_strata, UnsupportedDesignError

_DATA = os.path.join("tests", "golden", "mixed_dunnett_emmeans_dataset.csv")


def _df():
    return pd.read_csv(_DATA)


def test_split_plot_strata_matches_classical_values():
    s = split_plot_strata(_df(), dv="Value", subject="Subject",
                          between="Group", within="Time")
    assert s.n_per_group == 8
    assert s.W == 3 and s.G == 3
    assert s.df_sg == 21 and s.df_res == 42
    assert s.ms_sg == pytest.approx(4.559091, abs=1e-4)
    assert s.ms_res == pytest.approx(0.714811, abs=1e-4)


def test_unbalanced_design_raises():
    df = _df()
    df = df[~((df.Subject == "S01") & (df.Time == "T3"))]  # missing cell
    with pytest.raises(UnsupportedDesignError):
        split_plot_strata(df, dv="Value", subject="Subject",
                          between="Group", within="Time")


from analysis.emm_posthoc import contrast_se_df


def test_contrast_se_and_satterthwaite_df_match_golden():
    s = split_plot_strata(_df(), dv="Value", subject="Subject",
                          between="Group", within="Time")
    se, df = contrast_se_df(s)
    assert se == pytest.approx(0.706441, abs=1e-5)
    assert df == pytest.approx(34.5371, abs=1e-3)


from analysis.emm_posthoc import mixed_dunnett_emm_mvt

_GOLD = os.path.join("tests", "golden", "references_mixed_dunnett_emmeans.json")


def test_mixed_dunnett_emm_mvt_matches_emmeans_golden():
    gold = {(c["Time"], c["contrast"]): c for c in json.load(open(_GOLD))["contrasts"]}
    out = mixed_dunnett_emm_mvt(_df(), dv="Value", subject="Subject",
                                between="Group", within="Time",
                                control_group="Ctrl", alpha=0.05)
    assert len(out) == len(gold)
    for c in out:
        key = (c["within_level"], f"{c['treatment']} - Ctrl")
        g = gold[key]
        assert c["estimate"] == pytest.approx(g["estimate"], abs=1e-4)
        assert c["se"] == pytest.approx(g["SE"], abs=1e-4)
        assert c["df"] == pytest.approx(g["df"], abs=1e-2)
        assert c["t"] == pytest.approx(g["t_ratio"], abs=1e-3)
        assert c["p_value"] == pytest.approx(g["p_value"], abs=2e-3)
