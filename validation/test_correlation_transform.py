"""
test_correlation_transform.py

Tests for the correlation transform & retest feature:
  - _apply_transform() helper
  - CorrelationModel.fit() with x_transform / y_transform
  - Results dict has correct keys
  - Transformation → Pearson path when transformed data is normal
  - No transformation → Spearman path preserved
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from correlation_models import CorrelationModel, _apply_transform


# ---------------------------------------------------------------------------
# _apply_transform helper
# ---------------------------------------------------------------------------

def test_apply_transform_none_returns_unchanged():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _apply_transform(vals, 'none')
    np.testing.assert_array_equal(result, vals)


def test_apply_transform_log10():
    vals = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    result = _apply_transform(vals, 'log10')
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_apply_transform_log10_handles_nonpositive():
    # Values <= 0 must be shifted so min becomes 1
    vals = np.array([-2.0, 0.0, 3.0, 10.0, 20.0])
    result = _apply_transform(vals, 'log10')
    assert np.all(np.isfinite(result))


def test_apply_transform_sqrt():
    vals = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    result = _apply_transform(vals, 'sqrt')
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_apply_transform_sqrt_handles_negative():
    vals = np.array([-4.0, 0.0, 4.0, 9.0, 16.0])
    result = _apply_transform(vals, 'sqrt')
    assert np.all(np.isfinite(result))


def test_apply_transform_boxcox():
    rng = np.random.default_rng(42)
    vals = rng.lognormal(mean=0, sigma=0.5, size=50)  # strictly positive, right-skewed
    result = _apply_transform(vals, 'boxcox')
    assert result.shape == vals.shape
    assert np.all(np.isfinite(result))


def test_apply_transform_unknown_raises():
    with pytest.raises(ValueError, match="Unknown transformation"):
        _apply_transform(np.array([1.0, 2.0, 3.0]), 'magic')


# ---------------------------------------------------------------------------
# CorrelationModel with transforms
# ---------------------------------------------------------------------------

def _make_lognormal_df(n=80, seed=0):
    """Both variables log-normal (non-normal raw, normal after log10)."""
    rng = np.random.default_rng(seed)
    x = rng.lognormal(0, 1, n)
    y = 2.5 * x + rng.lognormal(0, 0.3, n)
    return pd.DataFrame({'x': x, 'y': y})


def test_no_transform_uses_spearman_for_lognormal():
    df = _make_lognormal_df()
    m = CorrelationModel()
    m.fit(df, 'x', 'y', method='auto', x_transform='none', y_transform='none')
    res = m.as_results_dict()
    assert res['method'] == 'spearman', "Raw log-normal data should trigger Spearman"


def test_log10_transform_switches_to_pearson():
    """After log10 transformation both variables should be normal -> Pearson."""
    df = _make_lognormal_df(n=80, seed=7)
    m = CorrelationModel()
    m.fit(df, 'x', 'y', method='auto', x_transform='log10', y_transform='log10')
    res = m.as_results_dict()
    assert res['method'] == 'pearson', (
        f"After log10 transform both variables should be normal -> Pearson. "
        f"Got method={res['method']!r}, normality_check={res.get('normality_check')}"
    )


def test_results_dict_has_transformation_key():
    df = _make_lognormal_df()
    m = CorrelationModel()
    m.fit(df, 'x', 'y', method='auto', x_transform='log10', y_transform='log10')
    res = m.as_results_dict()
    assert 'transformation' in res
    assert res['transformation'] != 'none'


def test_results_dict_has_pre_post_normality():
    df = _make_lognormal_df()
    m = CorrelationModel()
    m.fit(df, 'x', 'y', method='auto', x_transform='log10', y_transform='log10')
    res = m.as_results_dict()
    nc = res.get('normality_check', {})
    assert 'pre_transform' in nc, "normality_check must include pre_transform section"
    assert 'post_transform' in nc, "normality_check must include post_transform section"


def test_no_transform_results_has_none_transformation():
    df = _make_lognormal_df()
    m = CorrelationModel()
    m.fit(df, 'x', 'y', method='auto', x_transform='none', y_transform='none')
    res = m.as_results_dict()
    assert res.get('transformation') in ('none', None, 'None')


def test_sqrt_transform_runs_without_error():
    df = _make_lognormal_df()
    m = CorrelationModel()
    m.fit(df, 'x', 'y', method='auto', x_transform='sqrt', y_transform='none')
    res = m.as_results_dict()
    assert 'r' in res


def test_boxcox_transform_runs_without_error():
    df = _make_lognormal_df()
    m = CorrelationModel()
    m.fit(df, 'x', 'y', method='auto', x_transform='boxcox', y_transform='none')
    res = m.as_results_dict()
    assert 'r' in res


def test_original_df_not_mutated_by_transform():
    """fit() must not modify the caller's DataFrame."""
    df = _make_lognormal_df()
    original_x = df['x'].copy()
    m = CorrelationModel()
    m.fit(df, 'x', 'y', method='auto', x_transform='log10', y_transform='log10')
    pd.testing.assert_series_equal(df['x'], original_x)


def test_backward_compat_no_transform_args():
    """fit() without x_transform / y_transform must behave exactly as before."""
    df = _make_lognormal_df()
    m = CorrelationModel()
    m.fit(df, 'x', 'y', method='auto')   # no transform kwargs
    res = m.as_results_dict()
    assert 'method' in res
    assert 'r' in res
