"""Fixed-depth constraint applied on a rendered NLLoc control file."""

import os

import pytest

from dbclust.localization import LocalizationError
from dbclust.localization import apply_fixed_depth_to_conf

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "nll_template", "test")

CONF = """\
# comment LOCGRID 1 2 3
LOCSEARCH OCT 16 16 5 0.02 200000 10000 1 1
LOCGRID 6545 6041 201  0.00 0.00 0.00  0.20 0.20 0.20  PROB_DENSITY  SAVE
LOCMETH EDT_OT_WT 9999.0 4 -1 -1 -1 -1 -1 0
"""


def _line(text, key):
    return next(l for l in text.splitlines() if l.startswith(key)).split()


def test_locgrid_and_locsearch_rewritten():
    out = apply_fixed_depth_to_conf(CONF, 0.0)
    g = _line(out, "LOCGRID")
    assert g[1:3] == ["6545", "6041"]  # xy untouched
    assert g[3] == "2"  # numz
    assert float(g[6]) == 0.0  # z0
    assert float(g[9]) == pytest.approx(1.0)  # dz (default search thickness)
    assert g[-2:] == ["PROB_DENSITY", "SAVE"]
    s = _line(out, "LOCSEARCH")
    assert s[2:5] == ["32", "32", "2"]
    assert s[5:] == ["0.02", "200000", "10000", "1", "1"]
    # other lines and comments are preserved
    assert "# comment LOCGRID 1 2 3" in out
    assert "LOCMETH EDT_OT_WT" in out


def test_search_thickness_configurable():
    out = apply_fixed_depth_to_conf(CONF, 0.0, search_thickness_km=0.5)
    assert float(_line(out, "LOCGRID")[9]) == pytest.approx(0.5)


def test_depth_and_refine_factor():
    out = apply_fixed_depth_to_conf(CONF, 1.5, xy_refine=3)
    assert float(_line(out, "LOCGRID")[6]) == 1.5
    assert _line(out, "LOCSEARCH")[2:5] == ["48", "48", "2"]


def test_missing_locgrid_raises():
    with pytest.raises(LocalizationError):
        apply_fixed_depth_to_conf("LOCSEARCH OCT 16 16 5 0.02 1 1 1 1\n", 0.0)


def test_malformed_locgrid_raises():
    with pytest.raises(LocalizationError):
        apply_fixed_depth_to_conf("LOCGRID 10 10 2\n", 0.0)


def test_non_oct_locsearch_untouched():
    conf = "LOCSEARCH GRID 1000\nLOCGRID 10 10 11 0 0 0 1 1 1 PROB_DENSITY SAVE\n"
    out = apply_fixed_depth_to_conf(conf, 0.0)
    assert "LOCSEARCH GRID 1000" in out


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(TEMPLATE_DIR, "nll_haslach-0.2_template.conf")
    ),
    reason="NLL test templates not available",
)
def test_3d_template_equals_surface_template_search_lines():
    """3D template + fixed depth 0 reproduces the historical surface template."""
    with open(os.path.join(TEMPLATE_DIR, "nll_haslach-0.2_template.conf")) as f:
        t3d = f.read()
    with open(os.path.join(TEMPLATE_DIR, "nll_haslach-0.2-surface_template.conf")) as f:
        tsurf = f.read()
    out = apply_fixed_depth_to_conf(t3d, 0.0, search_thickness_km=0.2)
    assert _line(out, "LOCSEARCH") == _line(tsurf, "LOCSEARCH")
    g_out, g_surf = _line(out, "LOCGRID"), _line(tsurf, "LOCGRID")
    assert g_out[1:4] == g_surf[1:4]  # nx ny numz
    assert float(g_out[6]) == float(g_surf[6]) == 0.0
