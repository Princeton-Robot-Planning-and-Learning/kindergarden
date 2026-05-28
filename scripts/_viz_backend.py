"""Shared matplotlib backend selection for visualisation scripts.

Must be imported *before* ``matplotlib.pyplot`` is first imported.
Call :func:`select_backend` with ``no_show=True`` to force the non-interactive
``Agg`` backend (e.g. for headless / CI runs), or ``no_show=False`` to probe
for an interactive backend in preference order.
"""

from __future__ import annotations

import sys

import matplotlib


def select_backend(no_show: bool) -> None:
    """Switch matplotlib to an appropriate backend.

    Tries each candidate backend in preference order.  Falls back to ``Agg``
    (which still supports ``savefig``) with a warning if none of the
    interactive backends are available.

    Args:
        no_show: If ``True``, use ``Agg`` unconditionally (non-interactive).
    """
    if no_show:
        matplotlib.use("Agg")
        return

    candidates = ["Qt5Agg", "Qt6Agg", "TkAgg", "GTK3Agg", "WXAgg", "WebAgg"]
    for backend in candidates:
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as _plt  # pylint: disable=import-outside-toplevel

            _plt.figure()
            _plt.close("all")
            return
        except Exception:  # pylint: disable=broad-except
            continue

    matplotlib.use("Agg")
    print(
        "Warning: no interactive matplotlib backend found — "
        "plots will not be shown interactively.\n"
        "  Install PyQt5 for interactive display:  pip install PyQt5",
        file=sys.stderr,
    )
