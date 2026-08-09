"""Tests for the safe expression evaluator and the parametric parameter table.

Two kinds of ground truth are asserted here.  Arithmetic and the function table
are checked against closed-form values (``hypot(3, 4) == 5``, ``sind(30) == 0.5``,
``2 ** 3 ** 2 == 512``), never against recorded output.  Safety is checked by
attacking the evaluator: every construct that could reach ``__builtins__`` or
burn unbounded memory gets its own test, because "it did not crash" is not
evidence that a sandbox holds.
"""

from __future__ import annotations

import json
import math
import time

import pytest

from src.kernel.expressions import (
    DEFAULT_CONSTANTS,
    DEFAULT_FUNCTIONS,
    MAX_EXPONENT,
    CircularReferenceError,
    ExpressionError,
    Parameter,
    ParameterTable,
    UnknownNameError,
    evaluate,
    parse_expression,
    references,
)


class TestArithmetic:
    def test_result_is_always_a_float(self):
        assert isinstance(evaluate("1"), float)
        assert evaluate("1") == 1.0

    def test_a_bare_number_is_a_valid_expression(self):
        assert evaluate(2.5) == pytest.approx(2.5)
        assert evaluate(7) == pytest.approx(7.0)

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("2 + 3 * 4", 14.0),
            ("(2 + 3) * 4", 20.0),
            ("10 - 3 - 2", 5.0),  # subtraction is left-associative
            ("2 ** 3 ** 2", 512.0),  # exponentiation is right-associative
            ("-2 ** 2", -4.0),  # unary minus binds looser than **
            ("(-2) ** 2", 4.0),
            ("2 * 3 % 4", 2.0),
            ("7 / 2", 3.5),
            ("7 // 2", 3.0),
            ("-7 // 2", -4.0),  # floor division rounds toward -inf
            ("7 % 3", 1.0),
            ("-7 % 3", 2.0),  # modulo takes the sign of the divisor
            ("1e-3", 0.001),
            ("  4   +   4  ", 8.0),
            ("+3", 3.0),
            ("--3", 3.0),
        ],
    )
    def test_precedence_and_operators(self, expression, expected):
        assert evaluate(expression) == pytest.approx(expected)

    def test_variables_are_substituted(self):
        assert evaluate("x + y * 2", {"x": 3, "y": 4}) == pytest.approx(11.0)

    def test_variables_shadow_constants(self):
        assert evaluate("pi") == pytest.approx(math.pi)
        assert evaluate("pi", {"pi": 3.0}) == pytest.approx(3.0)

    @pytest.mark.parametrize(
        ("name", "expected"), [("pi", math.pi), ("tau", math.tau), ("e", math.e)]
    )
    def test_constants(self, name, expected):
        assert evaluate(name) == pytest.approx(expected)

    def test_booleans_are_numbers(self):
        assert evaluate("True + True") == pytest.approx(2.0)
        assert evaluate("False") == pytest.approx(0.0)


class TestLogic:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("1 < 2", 1.0),
            ("2 < 1", 0.0),
            ("1 + 2 == 3", 1.0),
            ("3 != 3", 0.0),
            ("2 >= 2", 1.0),
            ("2 > 2", 0.0),
            ("1 <= 0", 0.0),
            ("1 < 2 < 3", 1.0),  # chained comparison
            ("1 < 3 < 2", 0.0),
            ("not 0", 1.0),
            ("not 5", 0.0),
            ("1 and 2", 2.0),
            ("0 and 2", 0.0),
            ("0 or 7", 7.0),
            ("5 or 7", 5.0),
            ("10 if 2 > 1 else 20", 10.0),
            ("10 if 2 < 1 else 20", 20.0),
        ],
    )
    def test_comparisons_and_boolean_operators(self, expression, expected):
        assert evaluate(expression) == pytest.approx(expected)

    def test_boolean_operators_short_circuit(self):
        # If the right-hand side were evaluated these would raise on 1/0.
        assert evaluate("0 and 1 / 0") == pytest.approx(0.0)
        assert evaluate("1 or 1 / 0") == pytest.approx(1.0)

    def test_ternary_does_not_evaluate_the_dead_branch(self):
        assert evaluate("1 if 1 else 1 / 0") == pytest.approx(1.0)


# One case per entry in DEFAULT_FUNCTIONS; the coverage test below fails if a
# function is ever added without a matching closed-form assertion.
FUNCTION_CASES = {
    "sin": ("sin(pi / 2)", 1.0),
    "cos": ("cos(0)", 1.0),
    "tan": ("tan(pi / 4)", 1.0),
    "asin": ("asin(1)", math.pi / 2),
    "acos": ("acos(0)", math.pi / 2),
    "atan": ("atan(1)", math.pi / 4),
    "atan2": ("atan2(1, 1)", math.pi / 4),
    "sind": ("sind(30)", 0.5),
    "cosd": ("cosd(60)", 0.5),
    "tand": ("tand(45)", 1.0),
    "asind": ("asind(0.5)", 30.0),
    "acosd": ("acosd(0.5)", 60.0),
    "atand": ("atand(1)", 45.0),
    "atan2d": ("atan2d(1, 1)", 45.0),
    "degrees": ("degrees(pi)", 180.0),
    "radians": ("radians(180)", math.pi),
    "sqrt": ("sqrt(9)", 3.0),
    "abs": ("abs(-3.5)", 3.5),
    "min": ("min(3, 1, 2)", 1.0),
    "max": ("max(3, 1, 2)", 3.0),
    "floor": ("floor(2.7)", 2.0),
    "ceil": ("ceil(2.1)", 3.0),
    "round": ("round(2.567, 2)", 2.57),
    "exp": ("exp(1)", math.e),
    "log": ("log(e)", 1.0),
    "log10": ("log10(1000)", 3.0),
    "pow": ("pow(2, 10)", 1024.0),
    "hypot": ("hypot(3, 4)", 5.0),
    "clamp": ("clamp(12, 0, 10)", 10.0),
    "sign": ("sign(-4)", -1.0),
}


class TestFunctionTable:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        list(FUNCTION_CASES.values()),
        ids=list(FUNCTION_CASES),
    )
    def test_default_function(self, expression, expected):
        assert evaluate(expression) == pytest.approx(expected, abs=1e-12)

    def test_every_default_function_is_covered(self):
        assert set(FUNCTION_CASES) == set(DEFAULT_FUNCTIONS)

    def test_variadic_arities(self):
        assert evaluate("min(4)") == pytest.approx(4.0)
        assert evaluate("max(1, 9, 2, 8)") == pytest.approx(9.0)
        assert evaluate("hypot(1, 2, 2)") == pytest.approx(3.0)

    def test_log_accepts_an_optional_base(self):
        assert evaluate("log(8, 2)") == pytest.approx(3.0)
        assert evaluate("log(exp(2))") == pytest.approx(2.0)

    def test_round_uses_half_to_even(self):
        # Documented IEEE-754 behaviour, not a defect - pin it so nobody
        # "fixes" it into half-up without noticing.
        assert evaluate("round(2.5)") == pytest.approx(2.0)
        assert evaluate("round(3.5)") == pytest.approx(4.0)

    def test_clamp_bounds_both_ends(self):
        assert evaluate("clamp(-5, 0, 10)") == pytest.approx(0.0)
        assert evaluate("clamp(5, 0, 10)") == pytest.approx(5.0)
        assert evaluate("clamp(50, 0, 10)") == pytest.approx(10.0)

    def test_sign_of_zero_is_zero(self):
        assert evaluate("sign(0)") == pytest.approx(0.0)
        assert evaluate("sign(2.5)") == pytest.approx(1.0)

    def test_custom_function_table_replaces_the_default(self):
        table = {"double": lambda value: value * 2.0}
        assert evaluate("double(4)", functions=table) == pytest.approx(8.0)
        with pytest.raises(ExpressionError):
            evaluate("sin(0)", functions=table)

    def test_constants_survive_a_custom_function_table(self):
        assert evaluate("pi", functions={}) == pytest.approx(math.pi)


class TestDegreesVersusRadians:
    def test_plain_trig_is_radians(self):
        assert evaluate("sin(30)") == pytest.approx(math.sin(30.0))
        assert evaluate("sin(30)") != pytest.approx(0.5)

    def test_suffixed_trig_is_degrees(self):
        assert evaluate("sind(30)") == pytest.approx(0.5)
        assert evaluate("cosd(180)") == pytest.approx(-1.0)
        assert evaluate("tand(45)") == pytest.approx(1.0)

    def test_inverse_trig_matches_its_forward_partner(self):
        assert evaluate("asind(sind(37))") == pytest.approx(37.0)
        assert evaluate("asin(sin(0.6))") == pytest.approx(0.6)

    def test_atan2_argument_order_is_y_then_x(self):
        assert evaluate("atan2d(1, 0)") == pytest.approx(90.0)
        assert evaluate("atan2d(0, 1)") == pytest.approx(0.0)


class TestSandbox:
    """One test per escape route.  Each must raise, not merely misbehave."""

    def test_attribute_access_is_refused(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("(1).real")
        assert "attribute access" in str(info.value)

    def test_dunder_traversal_is_refused(self):
        with pytest.raises(ExpressionError):
            evaluate("().__class__")
        # The classic sandbox escape. What matters is that it is refused and
        # says why; the parser rejects the attribute-based *call* before it ever
        # looks at which attribute was named, so do not pin the token here.
        with pytest.raises(ExpressionError) as info:
            evaluate("(1).__class__.__base__.__subclasses__()")
        assert "attribute" in str(info.value).lower()

    def test_bare_dunder_name_is_refused(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("__builtins__")
        assert "__" in str(info.value)

    def test_builtins_are_unreachable_even_as_a_variable(self):
        # Supplying the name as a variable must not help: the '__' rule is
        # enforced during parsing, before any lookup happens.
        with pytest.raises(ExpressionError):
            evaluate("__builtins__", {"__builtins__": 1})

    def test_subscripting_is_refused(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("a[0]", {"a": 1})
        assert "subscripting" in str(info.value)

    def test_lambda_is_refused(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("lambda: 1")
        assert "lambda" in str(info.value)
        with pytest.raises(ExpressionError):
            evaluate("(lambda: 1)()")

    def test_comprehensions_are_refused(self):
        for source in ("[i for i in a]", "{i for i in a}", "{i: i for i in a}", "(i for i in a)"):
            with pytest.raises(ExpressionError):
                evaluate(source, {"a": 1})

    def test_import_is_refused(self):
        with pytest.raises(ExpressionError):
            evaluate("import os")  # not even valid in eval mode
        with pytest.raises(ExpressionError) as info:
            evaluate("__import__('os')")
        assert info.value.token == "__import__"

    def test_calling_a_non_whitelisted_name_is_refused(self):
        for source in ("open('f')", "eval('1')", "exec('1')", "globals()"):
            with pytest.raises(ExpressionError) as info:
                evaluate(source)
            assert "Unknown function" in str(info.value)

    def test_calling_a_variable_is_refused(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("shell(1)", {"shell": print})
        assert info.value.token == "shell"

    def test_string_and_container_literals_are_refused(self):
        for source in ("'abc'", "[1, 2]", "(1, 2)", "{1: 2}", "{1, 2}", "None", "1j"):
            with pytest.raises(ExpressionError):
                evaluate(source)

    def test_fstrings_are_refused(self):
        with pytest.raises(ExpressionError) as info:
            evaluate('f"{1}"')
        assert "f-string" in str(info.value)

    def test_starred_arguments_are_refused(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("max(*a)", {"a": 1})
        assert "starred" in str(info.value)

    def test_keyword_arguments_are_refused(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("round(2.5, digits=1)")
        assert "keyword" in str(info.value)

    def test_walrus_is_refused(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("(x := 1)")
        assert "walrus" in str(info.value)

    def test_bitwise_and_identity_operators_are_refused(self):
        for source in ("1 | 2", "1 & 2", "1 ^ 2", "1 << 2", "~1", "1 is 1", "1 in a"):
            with pytest.raises(ExpressionError):
                evaluate(source, {"a": 1})

    def test_huge_exponent_is_refused_instead_of_computed(self):
        started = time.perf_counter()
        with pytest.raises(ExpressionError) as info:
            evaluate("9**9**9")
        elapsed = time.perf_counter() - started
        assert "safety limit" in str(info.value)
        # A real evaluation would allocate a number with ~370 million digits.
        assert elapsed < 1.0

    def test_exponent_cap_is_the_documented_value(self):
        assert evaluate(f"2 ** {MAX_EXPONENT - 24:g}") == pytest.approx(2.0**1000)
        with pytest.raises(ExpressionError):
            evaluate(f"2 ** {MAX_EXPONENT + 1:g}")

    def test_overlong_expressions_are_refused(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("1+" * 5000 + "1")
        assert "limit" in str(info.value)


class TestErrorReporting:
    def test_expression_error_is_a_value_error(self):
        assert issubclass(ExpressionError, ValueError)
        assert issubclass(UnknownNameError, ExpressionError)
        assert issubclass(CircularReferenceError, ExpressionError)

    def test_unknown_name_is_named_and_located(self):
        with pytest.raises(UnknownNameError) as info:
            evaluate("1 + foo")
        assert info.value.token == "foo"
        assert info.value.column == 5  # 1-based, as an editor counts
        assert "foo" in str(info.value)

    @pytest.mark.parametrize("expression", ["1 / 0", "1 // 0", "1 % 0", "1 / (2 - 2)"])
    def test_division_by_zero_surfaces_as_an_expression_error(self, expression):
        with pytest.raises(ExpressionError) as info:
            evaluate(expression)
        assert "zero" in str(info.value).lower()
        assert not isinstance(info.value, ZeroDivisionError)

    def test_zero_to_a_negative_power(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("0 ** -1")
        assert "negative power" in str(info.value)

    def test_complex_results_are_refused(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("(-8) ** 0.5")
        assert "real" in str(info.value)

    @pytest.mark.parametrize("expression", ["sqrt(-1)", "log(0)", "asin(2)", "clamp(1, 10, 0)"])
    def test_domain_errors_name_the_function(self, expression):
        name = expression.split("(")[0]
        with pytest.raises(ExpressionError) as info:
            evaluate(expression)
        assert name in str(info.value)

    def test_missing_arguments_are_reported(self):
        with pytest.raises(ExpressionError):
            evaluate("min()")
        with pytest.raises(ExpressionError):
            evaluate("hypot(1, 2, 3, 4) + atan2(1)")

    @pytest.mark.parametrize("expression", ["1e400", "1e308 * 10", "exp(1000)"])
    def test_non_finite_results_are_refused(self, expression):
        with pytest.raises(ExpressionError):
            evaluate(expression)

    @pytest.mark.parametrize("expression", ["", "   ", "1 +", "((1)", "1 2"])
    def test_malformed_input_is_reported(self, expression):
        with pytest.raises(ExpressionError):
            evaluate(expression)

    def test_non_numeric_variable_is_reported(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("x + 1", {"x": "wide"})
        assert "'x'" in str(info.value)

    def test_error_message_quotes_the_expression(self):
        with pytest.raises(ExpressionError) as info:
            evaluate("3 * unknown_thing")
        assert "3 * unknown_thing" in str(info.value)


class TestHelpers:
    def test_parse_expression_returns_a_tree_without_needing_values(self):
        tree = parse_expression("a + b * 2")
        assert tree.__class__.__name__ == "Expression"

    def test_references_lists_names_in_first_appearance_order(self):
        assert references("sin(theta) + pi * radius + theta") == ("theta", "radius")

    def test_references_ignores_functions_and_constants(self):
        assert references("2 + 2") == ()
        assert references("hypot(3, 4) * tau") == ()

    def test_reserved_names_are_the_union_of_functions_and_constants(self):
        from src.kernel.expressions import RESERVED_NAMES

        assert set(DEFAULT_FUNCTIONS) | set(DEFAULT_CONSTANTS) == RESERVED_NAMES


class TestParameterTableBasics:
    def test_empty_table(self):
        table = ParameterTable()
        assert len(table) == 0
        assert table.names() == ()
        assert table.values() == {}
        assert table.validate() == []
        assert "wall" not in table

    def test_set_get_and_iterate(self):
        table = ParameterTable()
        table.set("wall", 2, unit="mm", comment="shell thickness")
        table.set("bore", "wall * 3")

        assert len(table) == 2
        assert "wall" in table
        assert list(table) == ["wall", "bore"]
        assert table.names() == ("wall", "bore")
        assert [name for name, _ in table.items()] == ["wall", "bore"]

        wall = table.get("wall")
        assert isinstance(wall, Parameter)
        assert wall.unit == "mm"
        assert wall.comment == "shell thickness"
        assert wall.expression == "2.0"  # a literal is just a constant expression

    def test_get_missing_raises_key_error(self):
        table = ParameterTable()
        with pytest.raises(KeyError):
            table.get("nope")
        with pytest.raises(KeyError):
            table.value("nope")

    def test_values_are_evaluated(self):
        table = ParameterTable()
        table.set("a", "2")
        table.set("b", "a * 3")
        assert table.value("b") == pytest.approx(6.0)
        assert table.values() == pytest.approx({"a": 2.0, "b": 6.0})

    def test_redefining_a_parameter_updates_dependents(self):
        table = ParameterTable()
        table.set("a", 2)
        table.set("b", "a * 3")
        assert table.value("b") == pytest.approx(6.0)
        table.set("a", 5)
        assert table.value("b") == pytest.approx(15.0)

    def test_expressions_are_not_evaluated_until_asked(self):
        # Deferring evaluation is what lets a formula be typed before the
        # parameters it names exist.
        table = ParameterTable()
        table.set("later", "1 / divisor")
        table.set("divisor", 4)
        assert table.value("later") == pytest.approx(0.25)

    def test_table_evaluate_resolves_against_current_values(self):
        table = ParameterTable()
        table.set("bore", 12.2)
        table.set("wall", 2)
        assert table.evaluate("bore * 2 + wall") == pytest.approx(26.4)

    def test_copy_is_independent(self):
        table = ParameterTable()
        table.set("a", 1)
        clone = table.copy()
        clone.set("a", 99)
        assert table.value("a") == pytest.approx(1.0)
        assert clone.value("a") == pytest.approx(99.0)

    @pytest.mark.parametrize(
        "name", ["2x", "my-name", "wall size", "class", "if", "pi", "e", "sin", "clamp", "a__b", ""]
    )
    def test_invalid_names_are_refused(self, name):
        table = ParameterTable()
        with pytest.raises(ExpressionError):
            table.set(name, 1)
        assert len(table) == 0

    def test_a_cad_parameter_chain(self):
        table = ParameterTable()
        table.set("shaft_d", 12, unit="mm")
        table.set("clearance", 0.2, unit="mm")
        table.set("bore", "shaft_d + clearance", unit="mm")
        table.set("wall", 2, unit="mm")
        table.set("plate_w", "3 * bore", unit="mm")
        table.set("plate_area", "plate_w * (bore + 2 * wall)", unit="mm2")

        values = table.values()
        assert values["bore"] == pytest.approx(12.2)
        assert values["plate_w"] == pytest.approx(36.6)
        assert values["plate_area"] == pytest.approx(36.6 * 16.2)

        table.set("shaft_d", 20)
        assert table.value("plate_w") == pytest.approx(60.6)


class TestDependencies:
    @staticmethod
    def chain():
        table = ParameterTable()
        table.set("a", 2)
        table.set("b", "a * 3")
        table.set("c", "b + 1")
        return table

    def test_chain_evaluates_in_dependency_order(self):
        table = self.chain()
        assert table.value("c") == pytest.approx(7.0)
        assert table.values() == pytest.approx({"a": 2.0, "b": 6.0, "c": 7.0})

    def test_dependency_order_ignores_insertion_order(self):
        table = ParameterTable()
        table.set("c", "b + 1")
        table.set("b", "a * 3")
        table.set("a", 2)
        assert table.names() == ("c", "b", "a")  # insertion order is preserved
        assert table.value("c") == pytest.approx(7.0)

    def test_direct_dependencies_and_dependents(self):
        table = self.chain()
        assert table.dependencies("c") == ("b",)
        assert table.dependencies("a") == ()
        assert table.dependents("a") == ("b",)
        assert table.dependents("c") == ()

    def test_transitive_dependencies_and_dependents(self):
        table = self.chain()
        assert table.dependencies("c", transitive=True) == ("a", "b")
        assert table.dependents("a", transitive=True) == ("b", "c")

    def test_dependencies_include_names_that_do_not_exist_yet(self):
        table = ParameterTable()
        table.set("a", "ghost + 1")
        assert table.dependencies("a") == ("ghost",)

    def test_functions_and_constants_are_not_dependencies(self):
        table = ParameterTable()
        table.set("r", 3)
        table.set("area", "pi * pow(r, 2)")
        assert table.dependencies("area") == ("r",)
        assert table.value("area") == pytest.approx(math.pi * 9.0)


class TestCycles:
    def test_self_reference_is_refused(self):
        table = ParameterTable()
        with pytest.raises(CircularReferenceError) as info:
            table.set("a", "a + 1")
        assert info.value.cycle == ("a", "a")
        assert "a" not in table
        assert len(table) == 0

    def test_redefining_a_parameter_in_terms_of_itself_is_refused(self):
        table = ParameterTable()
        table.set("a", 4)
        with pytest.raises(CircularReferenceError):
            table.set("a", "a * 2")
        assert table.get("a").expression == "4.0"
        assert table.value("a") == pytest.approx(4.0)

    def test_three_node_cycle_is_refused_and_the_table_is_unchanged(self):
        table = ParameterTable()
        table.set("a", 1)
        table.set("b", "a + 1")
        table.set("c", "b + 1")
        before = table.to_dict()
        before_values = table.values()

        with pytest.raises(CircularReferenceError) as info:
            table.set("a", "c + 1")

        # The cycle must be reported as a closed loop over exactly these three
        # names. The traversal direction is an implementation detail - walking
        # dependencies and walking dependents both describe the same cycle.
        cycle = info.value.cycle
        assert cycle[0] == cycle[-1] == "a"
        assert set(cycle) == {"a", "b", "c"}
        assert len(cycle) == 4
        assert " -> ".join(cycle) in str(info.value)
        # The table must be byte-for-byte what it was, and still evaluable.
        assert table.to_dict() == before
        assert table.values() == pytest.approx(before_values)

    def test_two_node_cycle_through_a_forward_reference(self):
        table = ParameterTable()
        table.set("a", "b + 1")  # b does not exist yet - allowed
        with pytest.raises(CircularReferenceError):
            table.set("b", "a + 1")
        assert table.names() == ("a",)


class TestUnknownReferences:
    def test_value_names_the_missing_parameter_and_its_owner(self):
        table = ParameterTable()
        table.set("plate_w", "3 * bore")
        with pytest.raises(UnknownNameError) as info:
            table.value("plate_w")
        message = str(info.value)
        assert "bore" in message
        assert "plate_w" in message

    def test_removing_a_parameter_leaves_a_reported_dangling_reference(self):
        table = ParameterTable()
        table.set("shaft_d", 12)
        table.set("bore", "shaft_d + 0.2")
        assert table.value("bore") == pytest.approx(12.2)

        removed = table.remove("shaft_d")
        assert removed.name == "shaft_d"
        assert "shaft_d" not in table
        with pytest.raises(UnknownNameError):
            table.value("bore")
        assert any("shaft_d" in problem for problem in table.validate())

    def test_remove_missing_raises_key_error(self):
        with pytest.raises(KeyError):
            ParameterTable().remove("nope")

    def test_validate_reports_arithmetic_failures_too(self):
        table = ParameterTable()
        table.set("bad", "1 / 0")
        problems = table.validate()
        assert len(problems) == 1
        assert "bad" in problems[0]

    def test_validate_is_empty_for_a_healthy_table(self):
        table = ParameterTable()
        table.set("a", 1)
        table.set("b", "a + 1")
        assert table.validate() == []


class TestRename:
    def test_rename_updates_the_key_and_the_record(self):
        table = ParameterTable()
        table.set("a", 3, unit="mm", comment="note")
        table.rename("a", "b")
        assert "a" not in table
        assert table.get("b").unit == "mm"
        assert table.get("b").comment == "note"
        assert table.value("b") == pytest.approx(3.0)

    def test_rename_rewrites_references(self):
        table = ParameterTable()
        table.set("bore", 10)
        table.set("plate", "bore * 3")
        table.rename("bore", "hole")
        assert table.get("plate").expression == "hole * 3"
        assert table.value("plate") == pytest.approx(30.0)

    def test_rename_does_not_touch_names_that_merely_contain_the_old_name(self):
        # A naive str.replace would turn 'bore_depth' into 'hole_depth' here.
        table = ParameterTable()
        table.set("bore", 10)
        table.set("bore_depth", "bore * 2")
        table.set("prebore", 1)
        table.set("total", "bore + bore_depth + prebore")

        table.rename("bore", "hole")

        assert table.names() == ("hole", "bore_depth", "prebore", "total")
        assert table.get("bore_depth").expression == "hole * 2"
        assert table.get("prebore").expression == "1.0"
        assert table.get("total").expression == "hole + bore_depth + prebore"
        assert table.value("total") == pytest.approx(10.0 + 20.0 + 1.0)

    def test_rename_leaves_unrelated_expressions_byte_identical(self):
        table = ParameterTable()
        table.set("a", 1)
        table.set("untouched", "2 + 3*4")
        table.rename("a", "z")
        assert table.get("untouched").expression == "2 + 3*4"

    def test_rename_preserves_insertion_order(self):
        table = ParameterTable()
        for name in ("first", "second", "third"):
            table.set(name, 1)
        table.rename("second", "middle")
        assert table.names() == ("first", "middle", "third")

    def test_rename_to_an_existing_name_is_refused(self):
        table = ParameterTable()
        table.set("a", 1)
        table.set("b", 2)
        with pytest.raises(ExpressionError):
            table.rename("a", "b")
        assert table.names() == ("a", "b")

    def test_rename_to_an_invalid_name_is_refused(self):
        table = ParameterTable()
        table.set("a", 1)
        for bad in ("pi", "2b", "class"):
            with pytest.raises(ExpressionError):
                table.rename("a", bad)
        assert table.names() == ("a",)

    def test_rename_missing_raises_key_error(self):
        with pytest.raises(KeyError):
            ParameterTable().rename("a", "b")

    def test_rename_to_the_same_name_is_a_no_op(self):
        table = ParameterTable()
        table.set("a", 1)
        table.rename("a", "a")
        assert table.names() == ("a",)


class TestSerialization:
    @staticmethod
    def populated():
        table = ParameterTable()
        table.set("shaft_d", 12, unit="mm", comment="from the bearing spec")
        table.set("clearance", 0.2, unit="mm")
        table.set("bore", "shaft_d + clearance", unit="mm")
        return table

    def test_round_trip_preserves_everything(self):
        table = self.populated()
        payload = table.to_dict()
        clone = ParameterTable.from_dict(payload)

        assert clone.to_dict() == payload
        assert clone.names() == table.names()
        assert clone.items() == table.items()
        assert clone.values() == pytest.approx(table.values())

    def test_payload_is_json_safe(self):
        payload = self.populated().to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_payload_shape(self):
        payload = self.populated().to_dict()
        assert payload["version"] == 1
        assert [entry["name"] for entry in payload["parameters"]] == [
            "shaft_d",
            "clearance",
            "bore",
        ]
        # Absent metadata is omitted rather than written as null.
        assert "comment" not in payload["parameters"][1]

    def test_from_dict_accepts_a_plain_mapping(self):
        table = ParameterTable.from_dict({"a": 2, "b": "a * 3"})
        assert table.values() == pytest.approx({"a": 2.0, "b": 6.0})

    def test_from_dict_accepts_forward_references(self):
        table = ParameterTable.from_dict({"b": "a * 3", "a": 2})
        assert table.value("b") == pytest.approx(6.0)

    def test_from_dict_of_empty_payload(self):
        assert len(ParameterTable.from_dict(None)) == 0
        assert len(ParameterTable.from_dict({})) == 0
        assert len(ParameterTable.from_dict({"version": 1, "parameters": []})) == 0

    def test_from_dict_rejects_malformed_entries(self):
        with pytest.raises(ExpressionError):
            ParameterTable.from_dict({"version": 1, "parameters": [{"expression": "1"}]})
        with pytest.raises(ExpressionError):
            ParameterTable.from_dict({"version": 1, "parameters": ["a = 1"]})

    def test_from_dict_rejects_a_cyclic_payload(self):
        payload = {"version": 1, "parameters": [
            {"name": "a", "expression": "b"},
            {"name": "b", "expression": "a"},
        ]}
        with pytest.raises(CircularReferenceError):
            ParameterTable.from_dict(payload)

    def test_parameter_equality_and_repr(self):
        first = Parameter("a", "1.0", "mm", "note")
        assert first == Parameter("a", "1.0", "mm", "note")
        assert first != Parameter("a", "2.0", "mm", "note")
        assert "a" in repr(first)
