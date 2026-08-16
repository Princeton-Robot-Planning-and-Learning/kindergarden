# Maintaining Kinder

## Package Names

- **PyPI name**: `kindergarden` (`pip install kindergarden`)
- **Import name**: `kinder` (`import kinder`)
- **Repository**: `kinder/`

## Dependencies from prpl-mono

Kinder depends on five packages from the [prpl-mono](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono) monorepo, all published to PyPI:

| PyPI name | Import name | prpl-mono directory | Used by |
|-----------|-------------|---------------------|---------|
| `prpl_utils` | `prpl_utils` | `prpl-utils/` | core |
| `relational_structs` | `relational_structs` | `relational-structs/` | core |
| `tomsgeoms2d` | `tomsgeoms2d` | `toms-geoms-2d/` | dynamic2d, kinematic2d |
| `pybullet_helpers` | `pybullet_helpers` | `pybullet-helpers/` | kinematic3d |
| `prpl_kinematics` | `prpl_kinematics` | `prpl-kinematics/` | kinematic3d_v2 |

`prpl_kinematics` is an optional dependency, installed via the `prpl-kinematics` extra
(`pip install "kindergarden[prpl-kinematics]"`) or `requirements/kinematic3d_v2.txt`. It
overlaps heavily with `pybullet_helpers` and both are large, so a base install pulls in
neither pair member beyond what the other environments need. The `develop` extra installs
it so the kinematic3d_v2 environments are linted, type checked, and tested.

## Releasing a New Version

### 1. Release prpl-mono dependencies first (if changed)

If you changed any of the five prpl-mono packages above, publish them before publishing kinder. The publish order matters because of inter-dependencies:

1. `prpl_utils` (no prpl-mono deps)
2. `tomsgeoms2d` (no prpl-mono deps)
3. `relational_structs` (depends on `prpl_utils`)
4. `pybullet_helpers` (depends on `prpl_utils`)
5. `prpl_kinematics` (depends on `prpl_utils`)

To check whether a package has unpublished changes, compare its `version` in
`pyproject.toml` against PyPI, and look for commits touching its directory since that
version was set:

```bash
cd prpl-mono
bump=$(git log -1 --format=%H -- <package-dir>/pyproject.toml)
git log --oneline $bump..HEAD -- <package-dir>/
```

A matching version number does not by itself mean the published artifact is current: the
version may have been bumped without a publish, or commits may have landed after it. When
it matters, diff the published wheel against your working tree.

For each package that changed:

```bash
cd prpl-mono/<package-dir>
# bump version in pyproject.toml
rm -rf dist/ build/ src/*.egg-info/
uv build
uv publish dist/*
```

`uv publish` needs a PyPI token. Set it via `export UV_PUBLISH_TOKEN=pypi-...`.

### 2. Update kinder's dependency versions (if needed)

If you published new versions of the prpl-mono packages, update the version pins in `kinder/pyproject.toml`. For example, change `prpl_utils>=0.0.1` to `prpl_utils>=0.0.2`.

### 3. Release kinder

```bash
cd kinder/
# bump version in pyproject.toml
rm -rf dist/ build/ src/*.egg-info/
uv build
uv publish dist/*
```

### 4. Create a GitHub release

After publishing to PyPI, create a matching GitHub release so the repo stays in sync:

```bash
gh release create v0.0.X --title "v0.0.X" --generate-notes
```

### Version numbering

Bump the patch version (`0.2.1` → `0.2.2`) for fixes and additive changes. Bump the minor
version (`0.2.x` → `0.3.0`) when an environment is removed or renamed, when a registration
or state-space API changes, or when a new environment family arrives with its own extra.
The major version stays at `0` for now.

Keep the PyPI publish and the GitHub release together. If a publish happens without a tag,
`--generate-notes` on the next release spans from the last tag and picks up the skipped
commits, but the intermediate version has no reachable release notes.

## CI

CI runs on GitHub Actions (`.github/workflows/ci.yml`) with five jobs: autoformat, linting, static type checking, unit tests, and notebook tests. The setup action installs with `uv pip install -e ".[develop]"`, which pulls all dependencies from PyPI.

## Local Development

```bash
cd kinder/
uv venv
uv pip install -e ".[develop]"
./run_ci_checks.sh
```

If you're also developing the prpl-mono dependencies locally, install them in editable mode and they will take precedence over the PyPI versions:

```bash
uv pip install -e ../prpl-mono/prpl-utils -e ../prpl-mono/relational-structs -e ../prpl-mono/toms-geoms-2d -e ../prpl-mono/pybullet-helpers -e ../prpl-mono/prpl-kinematics
```
