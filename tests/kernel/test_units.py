"""Tests for the unit system.

Unit bugs are silent and expensive - a part comes out 25.4 times too big - so
these check exact conversion factors rather than round-trip consistency alone.
A round trip through a wrong factor still round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.kernel.units import (
    UnitSystem,
    convert_angle,
    convert_length,
    format_length,
    parse_length,
)


class TestLengthConversion:
    @pytest.mark.parametrize(
        ("value", "source", "target", "expected"),
        [
            (1.0, "in", "mm", 25.4),
            (1.0, "mm", "in", 1.0 / 25.4),
            (1.0, "m", "mm", 1000.0),
            (1.0, "cm", "mm", 10.0),
            (1.0, "ft", "mm", 304.8),
            (1.0, "ft", "in", 12.0),
            (1000.0, "um", "mm", 1.0),
            (1.0, "thou", "in", 0.001),
            (1.0, "mil", "mm", 0.0254),
            (2.5, "mm", "mm", 2.5),
            (1.0, "yd", "ft", 3.0),
        ],
    )
    def test_known_factors(self, value, source, target, expected):
        assert convert_length(value, source, target) == pytest.approx(expected)

    def test_conversion_is_invertible(self):
        assert convert_length(convert_length(7.3, "mm", "in"), "in", "mm") == pytest.approx(7.3)

    def test_works_elementwise_on_arrays(self):
        values = np.array([1.0, 2.0, 3.0])
        assert convert_length(values, "in", "mm") == pytest.approx([25.4, 50.8, 76.2])

    def test_unit_names_are_case_and_plural_tolerant(self):
        assert convert_length(1.0, "IN", "mm") == pytest.approx(25.4)
        assert convert_length(1.0, "inches", "mm") == pytest.approx(25.4)
        assert convert_length(1.0, "Inch", "mm") == pytest.approx(25.4)

    def test_unknown_unit_is_reported_clearly(self):
        with pytest.raises(ValueError, match="Unknown length unit"):
            convert_length(1.0, "furlong", "mm")


class TestAngleConversion:
    @pytest.mark.parametrize(
        ("value", "source", "target", "expected"),
        [
            (180.0, "deg", "rad", np.pi),
            (np.pi, "rad", "deg", 180.0),
            (1.0, "turn", "deg", 360.0),
            (100.0, "grad", "deg", 90.0),
            (0.5, "rev", "deg", 180.0),
        ],
    )
    def test_known_factors(self, value, source, target, expected):
        assert convert_angle(value, source, target) == pytest.approx(expected)

    def test_unknown_unit_is_reported_clearly(self):
        with pytest.raises(ValueError, match="Unknown angle unit"):
            convert_angle(1.0, "quarter", "deg")


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "expected_mm"),
        [
            ("12", 12.0),
            ("12mm", 12.0),
            ("12 mm", 12.0),
            ("1in", 25.4),
            ('2"', 50.8),
            ("-3.5 cm", -35.0),
            ("1e3 um", 1.0),
            (".5m", 500.0),
            ("+4", 4.0),
            ("  7  ", 7.0),
        ],
    )
    def test_parses_common_forms(self, text, expected_mm):
        assert parse_length(text) == pytest.approx(expected_mm)

    def test_bare_number_uses_the_default_unit(self):
        assert parse_length("3", default_unit="in") == pytest.approx(76.2)

    def test_explicit_unit_beats_the_default(self):
        assert parse_length("3mm", default_unit="in") == pytest.approx(3.0)

    def test_numeric_input_passes_through(self):
        assert parse_length(5, default_unit="cm") == pytest.approx(50.0)

    @pytest.mark.parametrize("text", ["", "abc", "12 34", "mm", "1.2.3", "12 mm mm"])
    def test_garbage_raises(self, text):
        with pytest.raises(ValueError):
            parse_length(text)


class TestFormatting:
    def test_trailing_zeros_are_trimmed(self):
        assert format_length(25.4, "mm") == "25.4 mm"
        assert format_length(25.0, "mm") == "25 mm"

    def test_unit_suffix_is_optional(self):
        assert format_length(25.4, "mm", with_unit=False) == "25.4"

    def test_converts_before_formatting(self):
        assert format_length(25.4, "in", precision=4) == "1 in"

    def test_tiny_values_do_not_collapse_to_zero(self):
        """Rounding 1e-6 to '0.000 mm' would hide a real, if small, dimension."""
        text = format_length(1e-6, "mm", precision=3)
        assert "e-" in text
        assert not text.startswith("0 ")

    def test_exact_zero_stays_zero(self):
        assert format_length(0.0, "mm") == "0 mm"

    def test_negative_values_keep_their_sign(self):
        assert format_length(-2.5, "mm") == "-2.5 mm"


class TestUnitSystem:
    def test_millimetre_default_is_a_no_op(self):
        system = UnitSystem.millimetres()
        assert system.to_display(10.0) == pytest.approx(10.0)
        assert system.to_base(10.0) == pytest.approx(10.0)
        assert system.format_length(10.0) == "10 mm"

    def test_inch_system_converts_for_display(self):
        system = UnitSystem.inches()
        assert system.to_display(25.4) == pytest.approx(1.0)
        assert system.to_base(1.0) == pytest.approx(25.4)
        assert system.format_length(25.4) == "1 in"

    def test_area_and_volume_use_squared_and_cubed_factors(self):
        system = UnitSystem.inches()
        # One square inch is 25.4^2 mm^2; one cubic inch is 25.4^3 mm^3.
        assert system.format_area(25.4**2) == "1 in^2"
        assert system.format_volume(25.4**3) == "1 in^3"

    def test_centimetre_volume_factor(self):
        system = UnitSystem.centimetres()
        assert system.format_volume(1000.0) == "1 cm^3"

    def test_parse_uses_the_document_unit_as_default(self):
        system = UnitSystem.inches()
        assert system.parse("2") == pytest.approx(50.8)
        assert system.parse("2mm") == pytest.approx(2.0)

    def test_angle_formatting(self):
        system = UnitSystem.millimetres()
        assert system.format_angle(90.0) == "90°"
        radians = UnitSystem("Rad", "mm", "rad", 4)
        assert radians.format_angle(180.0).startswith("3.1416")

    def test_point_formatting(self):
        system = UnitSystem.millimetres()
        assert system.format_point([1.0, 2.5, -3.0]) == "1, 2.5, -3"

    def test_round_trips_through_a_dict(self):
        system = UnitSystem.inches()
        restored = UnitSystem.from_dict(system.to_dict())
        assert restored == system
        assert restored.length_unit == "in"

    def test_from_dict_tolerates_missing_keys(self):
        assert UnitSystem.from_dict({}).length_unit == "mm"
        assert UnitSystem.from_dict(None).length_unit == "mm"

    def test_invalid_unit_rejected_at_construction(self):
        with pytest.raises(ValueError):
            UnitSystem("Bad", "parsec")
