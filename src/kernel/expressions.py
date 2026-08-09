"""Safe expression evaluation and a dependency-aware parameter table.

CAD dimensions want to be *driven*, not typed twice.  A parametric model reads
``wall = 2``, ``bore = shaft_d + clearance``, ``plate_w = 3 * bore``: edit one
number and the whole part follows.  That needs two things this module provides -
an evaluator that runs arbitrary user text without handing the user a Python
shell, and a table that knows which parameter depends on which.

Safety
------
Expressions are parsed with :func:`ast.parse` and checked against an explicit
allow-list of node types.  Nothing is ever handed to :func:`eval` or
:func:`compile`; the tree is walked by the small interpreter in this module, so
no code object and no execution frame exists in which ``__builtins__`` could be
reached.  Attribute access, subscripting, lambdas, comprehensions, f-strings,
starred arguments and the walrus operator are rejected outright, which closes
the usual ``().__class__.__base__.__subclasses__()`` escape route.  ``**``
exponents are capped so ``9**9**9`` fails in microseconds instead of trying to
materialise a number with hundreds of millions of digits.

Conventions
-----------
- Every value is a Python ``float``.  Comparisons and boolean operators yield
  ``1.0`` / ``0.0`` rather than ``True`` / ``False``, so a formula can feed a
  result straight into arithmetic.
- Trigonometry comes in two flavours: ``sin``/``cos``/``tan``/``asin``/``acos``/
  ``atan``/``atan2`` take and return **radians**, while ``sind``/``cosd``/
  ``tand``/``asind``/``acosd``/``atand``/``atan2d`` take and return **degrees**.
  CAD users mostly think in degrees; both are provided so neither camp has to
  convert by hand.
- Results must be finite.  An expression that overflows to infinity is an error
  because an infinite dimension is never what the user meant.
- Names containing ``__`` are refused everywhere, belt-and-braces on top of the
  node allow-list.
"""

from __future__ import annotations

import ast
import keyword
import math
import operator

__all__ = [
    "DEFAULT_CONSTANTS",
    "DEFAULT_FUNCTIONS",
    "MAX_EXPONENT",
    "MAX_EXPRESSION_LENGTH",
    "RESERVED_NAMES",
    "CircularReferenceError",
    "ExpressionError",
    "Parameter",
    "ParameterTable",
    "UnknownNameError",
    "evaluate",
    "parse_expression",
    "references",
]

#: Largest magnitude allowed for a ``**`` exponent.  Anything above this cannot
#: produce a finite double anyway, so refusing it costs no expressive power and
#: turns ``9**9**9`` from a memory-exhaustion attack into an instant error.
MAX_EXPONENT = 1024.0

#: Expressions longer than this are refused unparsed.  Parsing cost grows with
#: input size and no honest dimension formula is a kilobyte long.
MAX_EXPRESSION_LENGTH = 4096

_SCHEMA_VERSION = 1


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
class ExpressionError(ValueError):
    """An expression could not be parsed, was unsafe, or failed to evaluate.

    The offending ``token`` and its 1-based ``column`` are carried alongside the
    message whenever they are known, so a UI can underline the exact character
    the user got wrong instead of showing a bare traceback.  Subclasses
    :class:`ValueError`, so callers that already funnel bad input through
    ``except ValueError`` keep working.
    """

    def __init__(self, message, expression=None, column=None, token=None):
        self.message = str(message)
        self.expression = expression
        self.column = None if column is None else int(column)
        self.token = token
        super().__init__(self._describe())

    def _describe(self):
        parts = [self.message]
        if self.column is not None:
            parts.append(f"at column {self.column}")
        if self.expression:
            parts.append(f'in "{self.expression}"')
        return " ".join(parts)


class UnknownNameError(ExpressionError):
    """An expression referenced a name that no variable or parameter defines."""


class CircularReferenceError(ExpressionError):
    """A parameter would depend, directly or transitively, on itself.

    ``cycle`` lists the names on the loop, starting and ending with the
    parameter that was being assigned, so the UI can show ``a -> b -> c -> a``.
    """

    def __init__(self, message, cycle=(), expression=None):
        self.cycle = tuple(cycle)
        super().__init__(message, expression=expression)


# ----------------------------------------------------------------------
# The default function table
# ----------------------------------------------------------------------
def _power(base, exponent):
    """``base ** exponent`` with the exponent cap and a real-result guarantee.

    Python returns a *complex* number for a negative base raised to a fractional
    power; a CAD dimension has no use for that, so it is reported as an error
    rather than silently coerced.
    """
    if not math.isfinite(exponent) or abs(exponent) > MAX_EXPONENT:
        raise ValueError(
            f"exponent {exponent:g} exceeds the safety limit of {MAX_EXPONENT:g}"
        )
    result = base**exponent
    if isinstance(result, complex):
        raise ValueError("a negative base with a fractional exponent has no real result")
    return float(result)


def _minimum(*values):
    if not values:
        raise ValueError("min() needs at least one argument")
    return min(values)


def _maximum(*values):
    if not values:
        raise ValueError("max() needs at least one argument")
    return max(values)


def _clamp(value, low, high):
    """Constrain ``value`` to ``[low, high]``; an inverted range is an error.

    Silently swapping the bounds would hide a genuine authoring mistake, so a
    reversed range raises instead.
    """
    if low > high:
        raise ValueError(f"clamp() lower bound {low:g} is above upper bound {high:g}")
    return min(max(value, low), high)


def _round(value, digits=0.0):
    """Round to ``digits`` decimal places.

    Uses Python's round-half-to-even, so ``round(2.5)`` is ``2.0`` and
    ``round(3.5)`` is ``4.0``.  That is IEEE-754 behaviour, not a bug, but it
    surprises people who expect half-up - hence this note.
    """
    return float(round(value, int(digits)))


def _log(value, base=None):
    """Natural logarithm, or logarithm to ``base`` when a second argument is given."""
    if base is None:
        return math.log(value)
    return math.log(value, base)


def _sign(value):
    """``-1``, ``0`` or ``1``.  Zero returns ``0`` for both signed zeroes."""
    if value == 0.0:
        return 0.0
    return math.copysign(1.0, value)


def _sind(angle_degrees):
    return math.sin(math.radians(angle_degrees))


def _cosd(angle_degrees):
    return math.cos(math.radians(angle_degrees))


def _tand(angle_degrees):
    return math.tan(math.radians(angle_degrees))


def _asind(ratio):
    return math.degrees(math.asin(ratio))


def _acosd(ratio):
    return math.degrees(math.acos(ratio))


def _atand(ratio):
    return math.degrees(math.atan(ratio))


def _atan2d(y, x):
    return math.degrees(math.atan2(y, x))


#: Callables an expression may invoke.  Pass a different mapping to
#: :func:`evaluate` to replace it wholesale - build on top of this one with
#: ``{**DEFAULT_FUNCTIONS, "my_fn": my_fn}`` rather than mutating it in place.
DEFAULT_FUNCTIONS = {
    # Radians.
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # Degrees - the CAD-native flavour.
    "sind": _sind,
    "cosd": _cosd,
    "tand": _tand,
    "asind": _asind,
    "acosd": _acosd,
    "atand": _atand,
    "atan2d": _atan2d,
    "degrees": math.degrees,
    "radians": math.radians,
    # General arithmetic.
    "sqrt": math.sqrt,
    "abs": abs,
    "min": _minimum,
    "max": _maximum,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": _round,
    "exp": math.exp,
    "log": _log,
    "log10": math.log10,
    "pow": _power,
    "hypot": math.hypot,
    "clamp": _clamp,
    "sign": _sign,
}

#: Bare names that always resolve, unless an explicit variable shadows them.
DEFAULT_CONSTANTS = {
    "pi": math.pi,
    "tau": math.tau,
    "e": math.e,
}

#: Names a parameter may not take, because an expression could no longer tell
#: the parameter apart from the built-in of the same name.
RESERVED_NAMES = frozenset(DEFAULT_FUNCTIONS) | frozenset(DEFAULT_CONSTANTS)


# ----------------------------------------------------------------------
# The node allow-list
# ----------------------------------------------------------------------
_ALLOWED_BINARY = frozenset(
    {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow}
)
_ALLOWED_UNARY = frozenset({ast.UAdd, ast.USub, ast.Not})
_ALLOWED_BOOL = frozenset({ast.And, ast.Or})
_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_OPERATOR_SYMBOLS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.MatMult: "@",
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "not",
    ast.Invert: "~",
    ast.And: "and",
    ast.Or: "or",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.In: "in",
    ast.NotIn: "not in",
}

# Human-readable labels for the constructs that are deliberately refused.
_REJECTED_LABELS = {
    ast.Attribute: "attribute access",
    ast.Subscript: "subscripting",
    ast.Lambda: "lambda expressions",
    ast.ListComp: "list comprehensions",
    ast.SetComp: "set comprehensions",
    ast.DictComp: "dict comprehensions",
    ast.GeneratorExp: "generator expressions",
    ast.JoinedStr: "f-strings",
    ast.FormattedValue: "f-strings",
    ast.Starred: "starred arguments",
    ast.NamedExpr: "the walrus operator",
    ast.List: "list literals",
    ast.Tuple: "tuple literals",
    ast.Dict: "dict literals",
    ast.Set: "set literals",
    ast.Slice: "slices",
    ast.Await: "await",
    ast.Yield: "yield",
    ast.YieldFrom: "yield from",
}


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------
def parse_expression(expression, functions=None):
    """Parse and safety-check ``expression``, returning the validated AST.

    Useful on its own for live syntax checking in a UI: it raises exactly the
    errors :func:`evaluate` would raise for anything structural, without needing
    values for the names involved.
    """
    text = _normalise_expression(expression)
    table = DEFAULT_FUNCTIONS if functions is None else functions
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as err:
        raise ExpressionError(
            f"Could not parse the expression: {err.msg}", expression=text, column=err.offset
        ) from err
    except (ValueError, MemoryError, RecursionError) as err:
        raise ExpressionError(
            f"Could not parse the expression: {err}", expression=text
        ) from err
    try:
        _validate(tree, text, table)
    except RecursionError as err:
        raise ExpressionError("Expression is nested too deeply", expression=text) from err
    return tree


def references(expression, functions=None):
    """Names an expression reads, in order of first appearance.

    Function and constant names (``sin``, ``pi``, ...) are *not* references -
    they resolve from the function table, never from the parameter table - so
    the result is exactly the dependency set of a parametric formula.
    """
    text = _normalise_expression(expression)
    table = DEFAULT_FUNCTIONS if functions is None else functions
    tree = parse_expression(text, functions=table)
    names, seen = [], set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id in seen:
            continue
        seen.add(node.id)
        if node.id in table or node.id in DEFAULT_CONSTANTS:
            continue
        names.append(node.id)
    return tuple(names)


def evaluate(expression, variables=None, functions=None):
    """Evaluate ``expression`` to a finite float.

    ``variables`` maps names to numbers and shadows :data:`DEFAULT_CONSTANTS`;
    ``functions`` replaces :data:`DEFAULT_FUNCTIONS` wholesale when given.  A
    bare number is accepted as an expression so callers do not have to stringify
    constants first.  Anything outside the allow-list - or any arithmetic that
    fails or overflows - raises :class:`ExpressionError`.
    """
    text = _normalise_expression(expression)
    table = DEFAULT_FUNCTIONS if functions is None else functions
    scope = {} if variables is None else variables
    tree = parse_expression(text, functions=table)
    try:
        result = _Interpreter(text, scope, table).run(tree)
    except RecursionError as err:
        raise ExpressionError("Expression is nested too deeply", expression=text) from err
    if not math.isfinite(result):
        raise ExpressionError(
            "Expression does not evaluate to a finite number", expression=text
        )
    return float(result)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def _column(node):
    """1-based column of a node, matching what a text editor shows."""
    offset = getattr(node, "col_offset", None)
    return None if offset is None else offset + 1


def _normalise_expression(expression):
    """Coerce user input into expression source text."""
    if isinstance(expression, str):
        text = expression.strip()
        if not text:
            raise ExpressionError("Expression is empty")
        if len(text) > MAX_EXPRESSION_LENGTH:
            raise ExpressionError(
                f"Expression is longer than the {MAX_EXPRESSION_LENGTH} character limit"
            )
        return text
    try:
        number = float(expression)
    except (TypeError, ValueError) as err:
        raise ExpressionError(f"Cannot use {expression!r} as an expression") from err
    if not math.isfinite(number):
        raise ExpressionError(f"Cannot use {expression!r} as an expression")
    return repr(number)


def _reject(node, expression, label, token=None):
    raise ExpressionError(
        f"Expressions may not use {label}",
        expression=expression,
        column=_column(node),
        token=token,
    )


def _check_identifier(name, node, expression):
    """Refuse dunder names outright - the classic sandbox-escape vocabulary."""
    if "__" in name:
        raise ExpressionError(
            f"Names may not contain '__': '{name}'",
            expression=expression,
            column=_column(node),
            token=name,
        )


def _validate(node, expression, functions):
    """Walk the tree, allowing only the node types this module can interpret."""
    if isinstance(node, ast.Expression):
        _validate(node.body, expression, functions)
        return

    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so True/False are admitted deliberately.
        if isinstance(node.value, (int, float)):
            return
        _reject(node, expression, f"the literal {node.value!r}", token=repr(node.value))

    if isinstance(node, ast.Name):
        _check_identifier(node.id, node, expression)
        return

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARY:
            _reject(node, expression, f"the '{_symbol(node.op)}' operator")
        _validate(node.operand, expression, functions)
        return

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINARY:
            _reject(node, expression, f"the '{_symbol(node.op)}' operator")
        _validate(node.left, expression, functions)
        _validate(node.right, expression, functions)
        return

    if isinstance(node, ast.BoolOp):
        if type(node.op) not in _ALLOWED_BOOL:
            _reject(node, expression, f"the '{_symbol(node.op)}' operator")
        for value in node.values:
            _validate(value, expression, functions)
        return

    if isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _COMPARISONS:
                _reject(node, expression, f"the '{_symbol(op)}' operator")
        _validate(node.left, expression, functions)
        for comparator in node.comparators:
            _validate(comparator, expression, functions)
        return

    if isinstance(node, ast.IfExp):
        _validate(node.test, expression, functions)
        _validate(node.body, expression, functions)
        _validate(node.orelse, expression, functions)
        return

    if isinstance(node, ast.Call):
        _validate_call(node, expression, functions)
        return

    label = _REJECTED_LABELS.get(type(node), type(node).__name__)
    token = getattr(node, "attr", None) or getattr(node, "id", None)
    _reject(node, expression, label, token=token)


def _validate_call(node, expression, functions):
    if not isinstance(node.func, ast.Name):
        _reject(node.func, expression, "computed or attribute-based function calls")
    name = node.func.id
    _check_identifier(name, node.func, expression)
    if node.keywords:
        raise ExpressionError(
            f"Function '{name}' may not be called with keyword arguments",
            expression=expression,
            column=_column(node.func),
            token=node.keywords[0].arg or "**",
        )
    if name not in functions:
        raise ExpressionError(
            f"Unknown function '{name}'",
            expression=expression,
            column=_column(node.func),
            token=name,
        )
    if not callable(functions[name]):
        raise ExpressionError(
            f"'{name}' is not a function",
            expression=expression,
            column=_column(node.func),
            token=name,
        )
    for argument in node.args:
        _validate(argument, expression, functions)


def _symbol(op):
    return _OPERATOR_SYMBOLS.get(type(op), type(op).__name__)


# ----------------------------------------------------------------------
# Interpretation
# ----------------------------------------------------------------------
class _Interpreter:
    """Walks a validated AST and produces floats.

    Kept separate from validation so :func:`parse_expression` can check a
    formula the moment the user stops typing, before any value exists for it.
    """

    __slots__ = ("expression", "variables", "functions")

    def __init__(self, expression, variables, functions):
        self.expression = expression
        self.variables = variables
        self.functions = functions

    def run(self, node):
        if isinstance(node, ast.Expression):
            return self.run(node.body)
        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, ast.BoolOp):
            return self._boolean(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.IfExp):
            return self.run(node.body) if self.run(node.test) else self.run(node.orelse)
        if isinstance(node, ast.Call):
            return self._call(node)
        # Unreachable while _validate runs first; kept so a future node type
        # cannot silently become executable.
        _reject(node, self.expression, type(node).__name__)
        return 0.0  # pragma: no cover

    def _constant(self, node):
        # A literal integer can be arbitrarily long; converting it to a double
        # is where that stops being free.
        try:
            return float(node.value)
        except (OverflowError, ValueError) as err:
            raise ExpressionError(
                "Numeric literal is too large to represent",
                expression=self.expression,
                column=_column(node),
            ) from err

    def _name(self, node):
        name = node.id
        if name in self.variables:
            raw = self.variables[name]
        elif name in DEFAULT_CONSTANTS:
            raw = DEFAULT_CONSTANTS[name]
        else:
            raise UnknownNameError(
                f"Unknown name '{name}'",
                expression=self.expression,
                column=_column(node),
                token=name,
            )
        try:
            return float(raw)
        except (TypeError, ValueError) as err:
            raise ExpressionError(
                f"Value of '{name}' is not a number: {raw!r}",
                expression=self.expression,
                column=_column(node),
                token=name,
            ) from err

    def _unary(self, node):
        value = self.run(node.operand)
        op = type(node.op)
        if op is ast.UAdd:
            return +value
        if op is ast.USub:
            return -value
        return 0.0 if value else 1.0  # ast.Not

    def _binary(self, node):
        left = self.run(node.left)
        right = self.run(node.right)
        op = type(node.op)
        if op is ast.Add:
            return left + right
        if op is ast.Sub:
            return left - right
        if op is ast.Mult:
            return left * right
        if op in (ast.Div, ast.FloorDiv, ast.Mod):
            if right == 0.0:
                raise ExpressionError(
                    f"Division by zero in '{_symbol(node.op)}'",
                    expression=self.expression,
                    column=_column(node.right),
                    token=_symbol(node.op),
                )
            if op is ast.Div:
                return left / right
            if op is ast.FloorDiv:
                return float(left // right)
            return float(left % right)
        # ast.Pow
        try:
            return _power(left, right)
        except ZeroDivisionError as err:
            raise ExpressionError(
                "Zero cannot be raised to a negative power",
                expression=self.expression,
                column=_column(node.right),
                token="**",
            ) from err
        except (ArithmeticError, ValueError) as err:
            raise ExpressionError(
                f"Cannot evaluate '**': {err}",
                expression=self.expression,
                column=_column(node.right),
                token="**",
            ) from err

    def _boolean(self, node):
        wants_truth = isinstance(node.op, ast.Or)
        value = 0.0
        for child in node.values:
            value = self.run(child)
            if bool(value) is wants_truth:
                return value
        return value

    def _compare(self, node):
        left = self.run(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.run(comparator)
            if not _COMPARISONS[type(op)](left, right):
                return 0.0
            left = right
        return 1.0

    def _call(self, node):
        name = node.func.id
        function = self.functions[name]
        arguments = [self.run(argument) for argument in node.args]
        column = _column(node.func)
        try:
            result = function(*arguments)
        except ExpressionError:
            raise
        except ZeroDivisionError as err:
            raise ExpressionError(
                f"{name}() divided by zero",
                expression=self.expression,
                column=column,
                token=name,
            ) from err
        except (ArithmeticError, TypeError, ValueError) as err:
            raise ExpressionError(
                f"{name}(): {err}",
                expression=self.expression,
                column=column,
                token=name,
            ) from err
        try:
            return float(result)
        except (TypeError, ValueError) as err:
            raise ExpressionError(
                f"{name}() did not return a number",
                expression=self.expression,
                column=column,
                token=name,
            ) from err


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
class Parameter:
    """One named entry in a :class:`ParameterTable`.

    The expression is always stored as source text - a plain constant is just
    the expression ``"2.0"`` - so the table has a single evaluation path and a
    project file never has to distinguish "value" from "formula".
    """

    __slots__ = ("name", "expression", "unit", "comment")

    def __init__(self, name, expression, unit=None, comment=None):
        self.name = str(name)
        self.expression = str(expression)
        self.unit = None if unit is None else str(unit)
        self.comment = None if comment is None else str(comment)

    def as_dict(self):
        """JSON-safe mapping; empty unit/comment are omitted to keep files tidy."""
        payload = {"name": self.name, "expression": self.expression}
        if self.unit is not None:
            payload["unit"] = self.unit
        if self.comment is not None:
            payload["comment"] = self.comment
        return payload

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            return NotImplemented
        return (self.name, self.expression, self.unit, self.comment) == (
            other.name,
            other.expression,
            other.unit,
            other.comment,
        )

    def __hash__(self):
        return hash((self.name, self.expression, self.unit, self.comment))

    def __repr__(self):
        unit = f" {self.unit}" if self.unit else ""
        return f"Parameter({self.name} = {self.expression}{unit})"


_VISITING = 0
_DONE = 1


class ParameterTable:
    """An ordered, dependency-aware set of named parametric values.

    Parameters may be entered in any order: an expression is allowed to name a
    parameter that does not exist yet, which is what makes ``from_dict`` and
    ordinary typing-as-you-go work.  The dangling reference is reported when the
    value is asked for (:class:`UnknownNameError`) or by :meth:`validate`.

    A reference cycle, by contrast, is rejected at :meth:`set` time and leaves
    the table byte-for-byte unchanged, because a cyclic table has no defined
    state to recover from.

    Lookups of a name that is not in the table raise :class:`KeyError`; problems
    *inside* an expression raise :class:`ExpressionError`.
    """

    def __init__(self, functions=None):
        self._parameters = {}
        self._functions = dict(DEFAULT_FUNCTIONS if functions is None else functions)
        self._reserved = frozenset(self._functions) | frozenset(DEFAULT_CONSTANTS)
        self._cache = {}

    # ------------------------------------------------------------------
    # Mapping-ish access
    # ------------------------------------------------------------------
    def __contains__(self, name):
        return name in self._parameters

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters)

    def names(self):
        """Parameter names in insertion order."""
        return tuple(self._parameters)

    def items(self):
        """``(name, Parameter)`` pairs in insertion order."""
        return tuple(self._parameters.items())

    def get(self, name):
        """The :class:`Parameter` record, or ``KeyError`` if there is none."""
        self._require(name)
        return self._parameters[name]

    def __repr__(self):
        return f"ParameterTable({len(self._parameters)} parameters)"

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def set(self, name, expression, unit=None, comment=None):
        """Define or redefine a parameter and return its record.

        The expression is parsed and safety-checked before anything is stored,
        and a change that would close a reference loop raises
        :class:`CircularReferenceError` without touching the table.
        """
        self._check_name(name)
        text = _normalise_expression(expression)
        refs = references(text, functions=self._functions)
        self._check_cycle(name, refs, text)
        parameter = Parameter(name, text, unit, comment)
        self._parameters[name] = parameter
        self._cache.clear()
        return parameter

    def remove(self, name):
        """Delete a parameter and return it.

        Expressions that still name it become dangling references rather than
        being rewritten - :meth:`validate` reports them so the user can decide.
        """
        self._require(name)
        parameter = self._parameters.pop(name)
        self._cache.clear()
        return parameter

    def rename(self, old, new):
        """Rename a parameter and rewrite every reference to it.

        References are rewritten through the AST, not by string substitution, so
        renaming ``bore`` leaves ``bore_depth`` and the string-like fragments of
        neighbouring identifiers alone.  Expressions that do not mention ``old``
        keep their original text untouched; the ones that do are re-emitted by
        :func:`ast.unparse` and may come back normalised.
        """
        self._require(old)
        if new == old:
            return self._parameters[old]
        self._check_name(new)
        if new in self._parameters:
            raise ExpressionError(f"A parameter named '{new}' already exists", token=new)

        rebuilt = {}
        for key, parameter in self._parameters.items():
            expression = _rewrite_name(parameter.expression, old, new, self._functions)
            target = new if key == old else key
            rebuilt[target] = Parameter(
                target, expression, parameter.unit, parameter.comment
            )
        self._parameters = rebuilt
        self._cache.clear()
        return self._parameters[new]

    def copy(self):
        """An independent table with the same parameters and function set."""
        clone = ParameterTable(functions=self._functions)
        clone._parameters = {
            name: Parameter(p.name, p.expression, p.unit, p.comment)
            for name, p in self._parameters.items()
        }
        return clone

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def value(self, name):
        """Evaluated value of one parameter, resolving its dependencies first."""
        self._require(name)
        return self._resolve([name])[name]

    def values(self):
        """Every parameter's value, keyed in table order."""
        resolved = self._resolve(list(self._parameters))
        return {name: resolved[name] for name in self._parameters}

    def evaluate(self, expression):
        """Evaluate a free expression against the table's current values."""
        text = _normalise_expression(expression)
        refs = references(text, functions=self._functions)
        scope = self._resolve([ref for ref in refs if ref in self._parameters])
        # The module-level evaluate(), not this method.
        return evaluate(text, variables=scope, functions=self._functions)

    # ------------------------------------------------------------------
    # Dependency graph
    # ------------------------------------------------------------------
    def dependencies(self, name, transitive=False):
        """Names this parameter reads - direct by default, the full cone if asked.

        Dangling references are included: they are real references, and the UI
        needs to show them.
        """
        self._require(name)
        graph = self._graph()
        if not transitive:
            return self._ordered(graph[name])
        return self._ordered(_reachable(graph, graph[name]))

    def dependents(self, name, transitive=False):
        """Parameters that read this one - direct by default, the full cone if asked."""
        self._require(name)
        graph = self._graph()
        reverse = {key: [] for key in graph}
        for key, refs in graph.items():
            for ref in refs:
                if ref in reverse:
                    reverse[ref].append(key)
        direct = [key for key, refs in graph.items() if name in refs]
        if not transitive:
            return self._ordered(direct)
        return self._ordered(_reachable(reverse, direct))

    def validate(self):
        """Human-readable problems with the table (an empty list is good)."""
        problems = []
        for name in self._parameters:
            try:
                self.value(name)
            except ExpressionError as err:
                message = str(err)
                if message not in problems:
                    problems.append(message)
        return problems

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self):
        """JSON-safe payload preserving order, expressions, units and comments."""
        return {
            "version": _SCHEMA_VERSION,
            "parameters": [p.as_dict() for p in self._parameters.values()],
        }

    @classmethod
    def from_dict(cls, payload, functions=None):
        """Rebuild a table from :meth:`to_dict`, or from a plain name -> expression map."""
        table = cls(functions=functions)
        if not payload:
            return table
        entries = payload
        if isinstance(payload, dict) and "parameters" in payload:
            entries = payload["parameters"]
        if isinstance(entries, dict):
            entries = [
                {**value, "name": key}
                if isinstance(value, dict)
                else {"name": key, "expression": value}
                for key, value in entries.items()
            ]
        for entry in entries:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ExpressionError(f"Cannot read parameter entry {entry!r}")
            table.set(
                entry["name"],
                entry.get("expression", 0.0),
                entry.get("unit"),
                entry.get("comment"),
            )
        return table

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require(self, name):
        if name not in self._parameters:
            raise KeyError(f"No parameter named '{name}'.")

    def _check_name(self, name):
        if not isinstance(name, str) or not name:
            raise ExpressionError("Parameter names must be non-empty strings")
        if not name.isidentifier():
            raise ExpressionError(f"'{name}' is not a valid parameter name", token=name)
        if keyword.iskeyword(name):
            raise ExpressionError(f"'{name}' is a Python keyword", token=name)
        if "__" in name:
            raise ExpressionError(f"Names may not contain '__': '{name}'", token=name)
        if name in self._reserved:
            raise ExpressionError(
                f"'{name}' would shadow the built-in function or constant of the same name",
                token=name,
            )

    def _graph(self):
        return {
            name: references(parameter.expression, functions=self._functions)
            for name, parameter in self._parameters.items()
        }

    def _ordered(self, names):
        """Sort names by table position, with undefined ones last and alphabetical."""
        position = {key: index for index, key in enumerate(self._parameters)}
        limit = len(position)
        return tuple(sorted(set(names), key=lambda n: (position.get(n, limit), n)))

    def _check_cycle(self, name, refs, expression):
        """Refuse an assignment that would let ``name`` reach itself."""
        graph = self._graph()
        graph[name] = tuple(refs)
        parents = {}
        visited = {name}
        stack = [name]
        while stack:
            current = stack.pop()
            for ref in graph.get(current, ()):
                if ref == name:
                    chain = [current]
                    while chain[-1] != name:
                        chain.append(parents[chain[-1]])
                    chain.reverse()
                    chain.append(name)
                    raise CircularReferenceError(
                        "Circular reference: " + " -> ".join(chain),
                        cycle=chain,
                        expression=expression,
                    )
                if ref in graph and ref not in visited:
                    visited.add(ref)
                    parents[ref] = current
                    stack.append(ref)

    def _dependency_order(self, graph, targets):
        """Defined names reachable from ``targets``, dependencies first."""
        order, state = [], {}
        for target in targets:
            if state.get(target) == _DONE:
                continue
            state[target] = _VISITING
            stack = [(target, iter(graph.get(target, ())))]
            while stack:
                node, children = stack[-1]
                descended = False
                for child in children:
                    if child not in self._parameters:
                        continue  # dangling; evaluate() reports it by name
                    status = state.get(child)
                    if status == _VISITING:
                        raise CircularReferenceError(
                            f"Circular reference: {node} -> {child}", cycle=(node, child)
                        )
                    if status == _DONE:
                        continue
                    state[child] = _VISITING
                    stack.append((child, iter(graph.get(child, ()))))
                    descended = True
                    break
                if descended:
                    continue
                stack.pop()
                state[node] = _DONE
                order.append(node)
        return order

    def _resolve(self, targets):
        graph = self._graph()
        values = dict(self._cache)
        for name in self._dependency_order(graph, targets):
            if name in values:
                continue
            expression = self._parameters[name].expression
            try:
                # The module-level evaluate(), not the method of the same name.
                values[name] = evaluate(
                    expression, variables=values, functions=self._functions
                )
            except CircularReferenceError:
                raise
            except ExpressionError as err:
                kind = UnknownNameError if isinstance(err, UnknownNameError) else ExpressionError
                raise kind(
                    f"Parameter '{name}': {err.message}",
                    expression=expression,
                    column=err.column,
                    token=err.token,
                ) from err
        self._cache.update(values)
        return values


def _reachable(graph, seeds):
    """Every node reachable from ``seeds`` (the seeds included)."""
    seen, stack = set(), list(seeds)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(graph.get(current, ()))
    return seen


def _rewrite_name(expression, old, new, functions):
    """Replace every reference to ``old`` with ``new`` via the AST.

    String replacement would corrupt ``bore_depth`` when renaming ``bore``;
    rewriting :class:`ast.Name` nodes only touches whole identifiers.
    """
    tree = parse_expression(expression, functions=functions)
    changed = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == old:
            node.id = new
            changed = True
    if not changed:
        return expression
    return ast.unparse(tree)
