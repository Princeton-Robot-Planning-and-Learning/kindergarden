"""Fix demo pickle files corrupted by the prbench -> kinder binary rename.

ROOT CAUSE
----------
When the package was renamed from "prbench" to "kinder", a raw find-and-replace
was done on the binary pickle files in demos/. Because "kinder" (6 chars) is one
byte shorter than "prbench" (7 chars), this broke the pickle format in two ways:

  1. The SHORT_BINUNICODE length byte for the env_id value (e.g.
     "prbench/BaseMotion3D-v0" -> "kinder/BaseMotion3D-v0") still declares the
     old string length (off by 1).
  2. The pickle protocol-4 frame length in the file header no longer matches the
     actual data size.

FIX
---
For each corrupted file:
  - Decrement the env_id string length byte (position 24) by 1.
  - Rewrite the frame length (bytes 3-10) to match the actual data size.

This restores all 1200 affected files. Three files were created after the rename
and are already correct (their env_id string length byte matches the actual
string length).

DELETE THIS SCRIPT once all demo files have been fixed and the fixes are
committed. It is kept temporarily so the fix is documented and reproducible.
"""

import glob
import io
import struct
import sys
from pathlib import Path

import dill

DEMO_DIR = Path(__file__).resolve().parent.parent / "demos"


def is_corrupted(data: bytes) -> bool:
    """Check if a pickle file has the off-by-one rename corruption.

    The rename replaced "prbench" with "kinder" in the raw bytes, making the env_id
    string 1 byte shorter. The corruption signature is: the SHORT_BINUNICODE length byte
    at position 24 is exactly 1 more than the actual env_id string length (i.e. it still
    reflects the old "prbench/" prefix).
    """
    if len(data) < 25:
        return False
    declared_str_len = data[24]
    # Read the actual string that follows (starts at byte 25)
    actual_str = data[25 : 25 + declared_str_len - 1]  # what it would be if 1 shorter
    # The string must start with "kinder/" to be a rename victim
    if not actual_str.startswith(b"kinder/"):
        return False
    # Verify the byte right after the shorter string is MEMOIZE (0x94),
    # which is what got consumed as part of the string due to the off-by-one
    if data[25 + declared_str_len - 1] != 0x94:
        return False
    return True


def fix_file(data: bytes) -> bytearray:
    """Fix the string length byte and frame length header."""
    fixed = bytearray(data)
    actual = len(fixed) - 11
    fixed[24] -= 1  # fix SHORT_BINUNICODE length for env_id value
    fixed[3:11] = struct.pack("<Q", actual)  # fix frame length
    return fixed


def main() -> None:
    """Fix all corrupted demo pickle files in the demos directory."""
    files = sorted(glob.glob(str(DEMO_DIR / "**" / "*.p"), recursive=True))
    if not files:
        print(f"No demo files found in {DEMO_DIR}")
        sys.exit(1)

    fixed = 0
    skipped = 0
    failed = []

    for path in files:
        with open(path, "rb") as f:
            data = f.read()

        if not is_corrupted(data):
            skipped += 1
            continue

        repaired = fix_file(data)

        try:
            dill.load(io.BytesIO(bytes(repaired)))
        except Exception as e:
            failed.append((path, str(e)))
            continue

        with open(path, "wb") as f:
            f.write(repaired)
        fixed += 1

    print(f"Fixed:   {fixed}")
    print(f"Skipped: {skipped} (already correct)")
    if failed:
        print(f"Failed:  {len(failed)}")
        for path, err in failed:
            print(f"  {path}: {err}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
