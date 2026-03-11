# Maintaining Kinder

## Package Names

- **PyPI name**: `kindergarden` (`pip install kindergarden`)
- **Import name**: `kinder` (`import kinder`)
- **Repository**: `kinder/`

## Dependencies from prpl-mono

Kinder depends on four packages from the [prpl-mono](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono) monorepo, all published to PyPI:

| PyPI name | Import name | prpl-mono directory | Used by |
|-----------|-------------|---------------------|---------|
| `prpl_utils` | `prpl_utils` | `prpl-utils/` | core |
| `relational_structs` | `relational_structs` | `relational-structs/` | core |
| `tomsgeoms2d` | `tomsgeoms2d` | `toms-geoms-2d/` | dynamic2d, kinematic2d |
| `pybullet_helpers` | `pybullet_helpers` | `pybullet-helpers/` | kinematic3d |

## Releasing a New Version

### 1. Release prpl-mono dependencies first (if changed)

If you changed any of the four prpl-mono packages above, publish them before publishing kinder. The publish order matters because of inter-dependencies:

1. `prpl_utils` (no prpl-mono deps)
2. `tomsgeoms2d` (no prpl-mono deps)
3. `relational_structs` (depends on `prpl_utils`)
4. `pybullet_helpers` (depends on `prpl_utils`)

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

### Version numbering

Use `0.0.x` for early development. Bump the patch version for each release.

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
uv pip install -e ../prpl-mono/prpl-utils -e ../prpl-mono/relational-structs -e ../prpl-mono/toms-geoms-2d -e ../prpl-mono/pybullet-helpers
```
