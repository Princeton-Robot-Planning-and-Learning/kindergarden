"""Configure pytest."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # typing-only; safe across pytest versions
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser

# Global variable that gets set with the command line --make-videos flag.
MAKE_VIDEOS = False

# Global variable that gets set with the command line --visualize flag.
VISUALIZE = False


def pytest_addoption(parser: "Parser") -> None:
    """Register custom command-line options for pytest."""
    parser.addoption(
        "--make-videos",
        action="store_true",
        default=False,
        help="Enable video generation during tests",
    )
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Show matplotlib visualizations during tests (e.g. point clouds)",
    )
    parser.addoption(
        "--save-point-cloud",
        metavar="PATH",
        default=None,
        help=(
            "Save the generated point cloud to a .npz file at PATH "
            "(e.g. --save-point-cloud /tmp/scene_pc.npz)"
        ),
    )


def pytest_configure(config: "Config") -> None:
    """Set global configuration values after command-line options are parsed."""
    global MAKE_VIDEOS, VISUALIZE  # pylint:disable=global-statement
    MAKE_VIDEOS = config.getoption("--make-videos")
    VISUALIZE = config.getoption("--visualize")
