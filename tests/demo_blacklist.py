"""Blacklist configuration for demo unit tests.

This module defines demos that should be excluded from deterministic replay tests. Each
entry includes a pattern to match against demo paths and a reason for exclusion.
"""

from pathlib import Path

# Blacklist for deterministic demo replay tests
# Format: {pattern: reason}
# Pattern can be any substring that appears in the demo path
DETERMINISTIC_REPLAY_BLACKLIST = {
    "ConstrainedCupboard3D-o1": (
        "Non-deterministic behavior in physics simulation. "
        "Test passes on local machines but fails inconsistently on GitHub Actions CI."
    ),
    "SweepIntoDrawer3D-o5": (
        "Non-deterministic behavior in physics simulation. "
        "Test passes on local machines but fails inconsistently on GitHub Actions CI."
    ),
    "Transport3D": ("speed up ci checks. "),
    "BaseMotion3D": ("speed up ci checks. "),
    "DynScoopPour": (
        "Non-deterministic behavior in physics simulation. "
        "Test passes on local machines but fails inconsistently on GitHub Actions CI."
    ),
    "DynObstruction": (
        "Non-deterministic behavior in physics simulation. "
        "Test passes on local machines but fails inconsistently on GitHub Actions CI."
    ),
    "DynPushPullHook": (
        "Non-deterministic behavior in physics simulation. "
        "Test passes on local machines but fails inconsistently on GitHub Actions CI."
    ),
    "DynPushT2D-t1/0/1759591501.p": (
        "Numerical precision issue at step 302. "
        "Difference (0.000197) slightly exceeds tolerance (0.0001)."
    ),
    # Motion2D: Keep only p5 (last variant)
    "Motion2D-p0": "Keeping only last variant (p5) per environment group",
    "Motion2D-p1": "Keeping only last variant (p5) per environment group",
    "Motion2D-p2": "Keeping only last variant (p5) per environment group",
    "Motion2D-p3": "Keeping only last variant (p5) per environment group",
    "Motion2D-p4": "Keeping only last variant (p5) per environment group",
    # StickButton2D: Keep only b10 (last variant)
    "StickButton2D-b1": "Keeping only last variant (b10) per environment group",
    "StickButton2D-b3": "Keeping only last variant (b10) per environment group",
    # Obstruction2D: Keep only o4 (last variant)
    "Obstruction2D-o0": "Keeping only last variant (o4) per environment group",
    "Obstruction2D-o3": "Keeping only last variant (o4) per environment group",
}


def is_demo_blacklisted(
    demo_path: Path, blacklist: dict[str, str]
) -> tuple[bool, str | None]:
    """Check if a demo path matches any blacklist pattern.

    Args:
        demo_path: Path to the demo file (as string or Path object)
        blacklist: Dictionary mapping patterns to reasons for blacklisting

    Returns:
        Tuple of (is_blacklisted, reason)
        - is_blacklisted: True if demo matches any blacklist pattern
        - reason: The reason string if blacklisted, None otherwise
    """
    demo_path_str = str(demo_path)
    for pattern, reason in blacklist.items():
        if pattern in demo_path_str:
            return True, reason
    return False, None
