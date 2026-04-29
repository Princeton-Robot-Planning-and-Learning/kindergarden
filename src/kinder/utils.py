"""Utility functions for KinDER."""

import os
import shutil
from pathlib import Path
from typing import List

import dill as pkl
import gdown

# Get the path to the kinder assets directory
_PACKAGE_DIR = Path(__file__).parent
_ASSETS_DIR = _PACKAGE_DIR / "envs" / "dynamic3d" / "models" / "assets"
_MIMICLABS_SCENES_DIR = _ASSETS_DIR / "mimiclabs_scenes"

# Google Drive URL for MimicLabs assets
_MIMICLABS_ASSETS_URL = (
    "https://drive.google.com/file/d/1k0dsJXFrqzlR1nPy8zo0vu9BnnezfkAm/view?usp=sharing"
)


def _download_file_from_gdrive(
    url: str,
    download_dir: Path,
    dst_filename: str,
    non_interactive: bool = False,
) -> None:
    """Download a file from Google Drive using gdown."""
    indent_str = "    " if non_interactive else ""

    tmp_dir = download_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    curr_dir = os.getcwd()
    os.chdir(tmp_dir)
    print(f"{indent_str}Downloading from Google Drive to {tmp_dir}")
    gdown.download(url, str(tmp_dir), quiet=False, fuzzy=True)
    tmp_files = list(tmp_dir.iterdir())
    if not tmp_files:
        raise FileNotFoundError("No file downloaded from Google Drive")
    tmp_path = tmp_files[0]
    os.chdir(curr_dir)

    dst_path = download_dir / dst_filename
    if dst_path.exists():
        if non_interactive:
            os.remove(dst_path)
            shutil.move(str(tmp_path), str(dst_path))
            print(f"{indent_str}Overwritten {dst_path}")
        else:
            inp = input(
                f"{indent_str}File {dst_path} already exists. "
                "Would you like to overwrite it? y/n\n"
            )
            if inp.lower() in ["y", "yes"]:
                shutil.move(str(tmp_path), str(dst_path))
                print(f"{indent_str}Overwritten {dst_path}")
            else:
                print(f"{indent_str}File {dst_path} not overwritten.")
    else:
        shutil.move(str(tmp_path), str(dst_path))
        print(f"{indent_str}Downloaded to {dst_path}")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def download_mimiclabs_assets(non_interactive: bool = False) -> None:
    """Download the MimicLabs scene assets from Google Drive.

    This will:
    1. Download assets.zip from Google Drive
    2. Extract it to kinder/envs/dynamic3d/models/assets/mimiclabs_scenes/
    3. Clean up the zip file

    Args:
        non_interactive: If True, avoid prompts and keep existing scene directory.
    """
    _ASSETS_DIR.mkdir(exist_ok=True, parents=True)

    indent_str = "    " if non_interactive else ""

    if _MIMICLABS_SCENES_DIR.exists():
        if non_interactive:
            shutil.rmtree(_MIMICLABS_SCENES_DIR)
            print(f"{indent_str}Removed existing {_MIMICLABS_SCENES_DIR}")
        else:
            print(f"\nWarning: Directory {_MIMICLABS_SCENES_DIR} already exists.")
            inp = input(
                f"{indent_str}Would you like to remove it and re-download? y/n\n"
            )
            if inp.lower() in ["y", "yes"]:
                shutil.rmtree(_MIMICLABS_SCENES_DIR)
                print(f"{indent_str}Removed existing {_MIMICLABS_SCENES_DIR}")
            else:
                print(f"{indent_str}Keeping existing assets. Exiting.")
                return

    print(f"\n{indent_str}Downloading MimicLabs scene assets to {_MIMICLABS_SCENES_DIR}")
    print(f"{indent_str}This may take a few minutes (assets are ~1GB)...\n")

    zip_filename = "assets.zip"
    _download_file_from_gdrive(
        _MIMICLABS_ASSETS_URL,
        _ASSETS_DIR,
        zip_filename,
        non_interactive=non_interactive,
    )

    zip_path = _ASSETS_DIR / zip_filename
    print(f"\n{indent_str}Extracting {zip_path}...")
    shutil.unpack_archive(str(zip_path), str(_ASSETS_DIR))

    unzipped_folder = _ASSETS_DIR / "assets"
    if unzipped_folder.exists():
        scenes_folder = unzipped_folder / "scenes" / "mimiclabs_scenes"
        if scenes_folder.exists():
            _MIMICLABS_SCENES_DIR.mkdir(exist_ok=True, parents=True)
            for item in scenes_folder.iterdir():
                shutil.move(str(item), str(_MIMICLABS_SCENES_DIR / item.name))
            print(f"{indent_str}Extracted assets to {_MIMICLABS_SCENES_DIR}")
        else:
            print(
                f"{indent_str}Warning: Expected scenes/mimiclabs_scenes not found in "
                f"{unzipped_folder}"
            )
        shutil.rmtree(unzipped_folder)
    else:
        print(f"{indent_str}Warning: Unzipped folder 'assets' not found")

    if zip_path.exists():
        os.remove(zip_path)
        print(f"{indent_str}Removed {zip_filename}")

    print(f"\n{indent_str}✓ MimicLabs scene assets successfully downloaded to:")
    print(f"{indent_str}  {_MIMICLABS_SCENES_DIR}")
    print(
        f"\n{indent_str}Available scenes: lab2.xml, lab3.xml, lab4.xml, lab5.xml, "
        f"lab6.xml, lab7.xml, lab8.xml"
    )


def load_demo(demo_path: Path) -> dict:
    """Load a demonstration from a pickle file."""
    try:
        with open(demo_path, "rb") as f:
            demo_data = pkl.load(f)

        # Validate demo data structure.
        required_keys = ["env_id", "observations", "actions"]
        for key in required_keys:
            if key not in demo_data:
                raise ValueError(f"Demo data missing required key: {key}")

        if not demo_data["actions"]:
            raise ValueError("Demo contains no actions")

        if len(demo_data["observations"]) != len(demo_data["actions"]) + 1:
            print(
                f"Warning: Expected {len(demo_data['actions']) + 1} observations, "
                f"got {len(demo_data['observations'])}"
            )

        if "seed" not in demo_data:
            raise ValueError(" Demo does not contain seed information.")

        return demo_data
    except Exception as e:
        # Don't exit, just raise the exception to be handled by caller
        raise ValueError(f"Error loading demo from {demo_path}: {e}") from e


def find_all_demo_files() -> List[Path]:
    """Find all demo files in the demos directory."""
    demos_dir = Path(__file__).parent.parent.parent / "demos"
    demo_files = list(demos_dir.glob("**/*.p"))
    return sorted(demo_files)


def get_env_id_from_demo_path(demo_path: Path) -> str:
    """Extract environment ID from demo file path structure."""
    # Demo path structure: demos/{env_name}/{instance}/{timestamp}.p
    env_name = demo_path.parent.parent.name
    # Convert from demo directory name to environment ID
    env_id = f"kinder/{env_name}-v0"
    return env_id
