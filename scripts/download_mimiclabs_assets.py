"""Script to download MimicLabs scene assets from Google Drive for KinDER.

This downloads the MimicLabs realistic background scenes (lab2-lab8) and places them
in the KinDER assets directory.

Example usage:
    python scripts/download_mimiclabs_assets.py
"""

import os
import shutil
from pathlib import Path

import gdown

# Get the path to the kinder assets directory
SCRIPT_DIR = Path(__file__).parent
PRBENCH_ROOT = SCRIPT_DIR.parent
ASSETS_DIR = (
    PRBENCH_ROOT / "src" / "kinder" / "envs" / "dynamic3d" / "models" / "assets"
)
MIMICLABS_SCENES_DIR = ASSETS_DIR / "mimiclabs_scenes"

# Google Drive URL for MimicLabs assets
ASSETS_URL = (
    "https://drive.google.com/file/d/1k0dsJXFrqzlR1nPy8zo0vu9BnnezfkAm/view?usp=sharing"
)


def download_file_from_gdrive(url: str, download_dir: Path, dst_filename: str) -> None:
    """Download a file from Google Drive using gdown.

    Args:
        url: Google Drive sharing URL
        download_dir: Directory to download to
        dst_filename: Destination filename
    """
    tmp_dir = download_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    # Download file
    curr_dir = os.getcwd()
    os.chdir(tmp_dir)
    print(f"Downloading from Google Drive to {tmp_dir}")
    gdown.download(url, str(tmp_dir), quiet=False, fuzzy=True)
    tmp_files = list(tmp_dir.iterdir())
    if not tmp_files:
        raise FileNotFoundError("No file downloaded from Google Drive")
    tmp_path = tmp_files[0]
    os.chdir(curr_dir)

    # Move downloaded file to destination
    dst_path = download_dir / dst_filename
    if dst_path.exists():
        inp = input(
            f"File {dst_path} already exists. Would you like to overwrite it? y/n\n"
        )
        if inp.lower() in ["y", "yes"]:
            shutil.move(str(tmp_path), str(dst_path))
            print(f"Overwritten {dst_path}")
        else:
            print(f"File {dst_path} not overwritten.")
    else:
        shutil.move(str(tmp_path), str(dst_path))
        print(f"Downloaded to {dst_path}")

    # Clean up tmp directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def download_mimiclabs_assets() -> None:
    """Download the MimicLabs scene assets from Google Drive.

    This will:
    1. Download assets.zip from Google Drive
    2. Extract it to kinder/src/kinder/envs/dynamic3d/models/assets/mimiclabs_scenes/
    3. Clean up the zip file
    """
    # Ensure assets directory exists
    ASSETS_DIR.mkdir(exist_ok=True, parents=True)

    # Check if mimiclabs_scenes already exists
    if MIMICLABS_SCENES_DIR.exists():
        print(f"\nWarning: Directory {MIMICLABS_SCENES_DIR} already exists.")
        inp = input("Would you like to remove it and re-download? y/n\n")
        if inp.lower() in ["y", "yes"]:
            shutil.rmtree(MIMICLABS_SCENES_DIR)
            print(f"Removed existing {MIMICLABS_SCENES_DIR}")
        else:
            print("Keeping existing assets. Exiting.")
            return

    print(f"\nDownloading MimicLabs scene assets to {MIMICLABS_SCENES_DIR}")
    print("This may take a few minutes (assets are ~1GB)...\n")

    # Download the assets zip file
    zip_filename = "assets.zip"
    download_file_from_gdrive(ASSETS_URL, ASSETS_DIR, zip_filename)

    # Unzip the assets
    zip_path = ASSETS_DIR / zip_filename
    print(f"\nExtracting {zip_path}...")
    shutil.unpack_archive(str(zip_path), str(ASSETS_DIR))

    # The unzipped folder is named "assets"
    unzipped_folder = ASSETS_DIR / "assets"
    if unzipped_folder.exists():
        # Move scenes/mimiclabs_scenes to the final location
        scenes_folder = unzipped_folder / "scenes" / "mimiclabs_scenes"
        if scenes_folder.exists():
            MIMICLABS_SCENES_DIR.mkdir(exist_ok=True, parents=True)
            # Move all contents
            for item in scenes_folder.iterdir():
                shutil.move(str(item), str(MIMICLABS_SCENES_DIR / item.name))
            print(f"Extracted assets to {MIMICLABS_SCENES_DIR}")
        else:
            print(
                f"Warning: Expected scenes/mimiclabs_scenes not found in "
                f"{unzipped_folder}"
            )

        # Clean up unzipped folder
        shutil.rmtree(unzipped_folder)
    else:
        print("Warning: Unzipped folder 'assets' not found")

    # Remove the zip file
    if zip_path.exists():
        os.remove(zip_path)
        print(f"Removed {zip_filename}")

    print("\n✓ MimicLabs scene assets successfully downloaded to:")
    print(f"  {MIMICLABS_SCENES_DIR}")
    print(
        "\nAvailable scenes: lab2.xml, lab3.xml, lab4.xml, lab5.xml, "
        "lab6.xml, lab7.xml, lab8.xml"
    )


if __name__ == "__main__":
    download_mimiclabs_assets()
