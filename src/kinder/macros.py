"""Macros and configuration constants for KinDER.

Environment variable macros
---------------------------
DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD
    Set to "1", "true", "yes", or "on" to prevent kinder.make() from
    automatically downloading MimicLabs scene assets on the first call
    to a Dynamic3D environment.  This is set automatically during CI.
"""

import os

# ---------------------------------------------------------------------------
# MimicLabs asset download macro
# ---------------------------------------------------------------------------

# Name of the environment variable that disables automatic asset downloads.
DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD = "DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD"

def is_truthy_env_var(name: str) -> bool:
    """Return True when an environment variable is set to a truthy value.

    Accepted values (case-insensitive): "1", "true", "yes", "on".
    """
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
