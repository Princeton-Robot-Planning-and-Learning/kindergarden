"""Pylint checker that bans direct use of np.random (global RNG state).

Allowed: np.random.default_rng, np.random.Generator Banned:  everything else under
np.random
"""

from astroid import nodes  # type: ignore[import-untyped]
from pylint.checkers import BaseChecker
from pylint.lint import PyLinter

_ALLOWED_ATTRS = frozenset({"default_rng", "Generator"})


class NoNpRandomChecker(BaseChecker):
    """Flag ``np.random.<X>`` unless *X* is ``default_rng`` or ``Generator``."""

    name = "no-np-random"
    msgs = {
        "C9900": (
            "Use np.random.default_rng() instead of np.random.%s",
            "np-random-disallowed",
            "Direct use of np.random functions/classes relies on global "
            "state and is non-reproducible. Use np.random.default_rng() "
            "to create a local Generator instance instead.",
        ),
    }

    def visit_attribute(self, node: nodes.Attribute) -> None:
        """Check every attribute access for banned np.random usage."""
        if node.attrname in _ALLOWED_ATTRS:
            return
        expr = node.expr
        if not isinstance(expr, nodes.Attribute):
            return
        if (
            expr.attrname == "random"
            and isinstance(expr.expr, nodes.Name)
            and expr.expr.name == "np"
        ):
            self.add_message("np-random-disallowed", node=node, args=(node.attrname,))


def register(linter: PyLinter) -> None:
    """Register the checker with pylint."""
    linter.register_checker(NoNpRandomChecker(linter))
