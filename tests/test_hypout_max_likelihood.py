"""nll.hypout = SAVE_NLLOC_EXPECTATION: dloch/dz stay "maximum likelihood vs
expectation" whichever of the two the origin holds (BUGS_COHERENCE_REVIEW_2026-11-15.md, B4)."""
import pytest
from obspy.core.event import Comment
from obspy.core.event import Origin

from dbclust.inject_spatialite import get_max_likelihood_localization
from dbclust.localization import read_nll_max_likelihood
from dbclust.localization_quality import nll_ml_expectation_offset

ML = (43.0000, 0.1200, 8.0)  # lat, lon, depth km
EXPECTATION = (43.0030, 0.1220, 9.0)

EXPECTATION_COMMENT = '{"expectation": {"latitude": %s, "longitude": %s, "depth": %s}}'
ML_COMMENT = '{"maximum_likelihood": {"latitude": %s, "longitude": %s, "depth": %s}}'


def _origin(hypocenter, comments):
    lat, lon, depth_km = hypocenter
    return Origin(
        latitude=lat,
        longitude=lon,
        depth=depth_km * 1000.0,
        comments=[Comment(text=c) for c in comments],
    )


def test_read_max_likelihood_line(tmp_path):
    hyp = tmp_path / "last.hyp"
    hyp.write_text(
        "NLLOC \"x\" \"LOCATED\" \"Location completed.\"\n"
        "STAT_GEOG  ExpectLat 43.003000 Long 0.122000 Depth 9.000000\n"
        "MAXIMUM_LIKELIHOOD  MaxLikeLat 43.000000 Long 0.120000 Depth 8.000000 OT 17.392377\n"
        "END_NLLOC\n"
    )
    assert read_nll_max_likelihood(str(hyp)) == ML


def test_no_max_likelihood_line_in_save_nlloc_all(tmp_path):
    hyp = tmp_path / "last.hyp"
    hyp.write_text("STAT_GEOG  ExpectLat 43.003000 Long 0.122000 Depth 9.000000\n")
    assert read_nll_max_likelihood(str(hyp)) is None
    assert read_nll_max_likelihood(str(tmp_path / "missing.hyp")) is None


def test_offset_is_the_same_in_both_modes():
    # SAVE_NLLOC_ALL: the origin is the maximum likelihood
    origin_ml = _origin(ML, [EXPECTATION_COMMENT % EXPECTATION])
    # SAVE_NLLOC_EXPECTATION: the origin is the expectation, ML is in a comment
    origin_exp = _origin(
        EXPECTATION, [EXPECTATION_COMMENT % EXPECTATION, ML_COMMENT % ML]
    )

    dloch_ml, dz_ml = nll_ml_expectation_offset(origin_ml)
    dloch_exp, dz_exp = nll_ml_expectation_offset(origin_exp)

    assert dloch_ml > 0.3  # ~0.37 km, not collapsed to 0
    assert dloch_exp == pytest.approx(dloch_ml)
    assert dz_exp == pytest.approx(dz_ml) == pytest.approx(-1.0)


def test_offset_needs_the_expectation():
    assert nll_ml_expectation_offset(_origin(ML, [])) is None


def test_max_likelihood_comment_for_database():
    origin_exp = _origin(
        EXPECTATION, [EXPECTATION_COMMENT % EXPECTATION, ML_COMMENT % ML]
    )
    assert get_max_likelihood_localization(origin_exp) == (43.0, 0.12, 8000.0)
    assert get_max_likelihood_localization(_origin(ML, [])) == (None, None, None)
