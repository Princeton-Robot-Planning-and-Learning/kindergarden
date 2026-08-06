# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

KinDER is a physical-reasoning benchmark: a set of Gymnasium environments across four
backends (`kinematic2d`, `dynamic2d`, `kinematic3d`, `dynamic3d`), all with
object-centric states. `README.md` is the user-facing tour; `docs/maintaining.md`
covers packaging and releases. This file covers the things that are easy to get wrong
while *changing* the code.

## Names: the distribution is `kindergarden`, the import package is `kinder`

`pip install kindergarden`, then `import kinder`. `import kindergarden` raises
`ModuleNotFoundError`. Anything that resolves a path into the installed package uses
`kinder`, e.g. `Path(kinder.__path__[0]) / "envs" / "dynamic3d" / "tasks"`.

## Setup

`uv` is the supported path (see `README.md`), and CI uses it:

```bash
uv venv --python=3.10
uv pip install -e ".[develop]"
```

Python is pinned to `>=3.10,<3.13`. The Dynamic3D tests additionally need OSMesa/GL and
BLAS/LAPACK system libraries; CI installs
`liblapack-dev libblas-dev libosmesa6-dev libgl1 libglx-mesa0` before the linting,
unit-test and notebook jobs, and a local Linux box needs the same.

## Running the checks

`./run_ci_checks.sh` runs everything CI runs, in the same order. It exports
`DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1` first — do the same when running the pieces
by hand, or a lint run will try to fetch multi-gigabyte scene assets.

```bash
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
./run_autoformat.sh                                     # black, docformatter, isort
mypy .
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc --ignore=notebooks
pytest tests/
pytest notebooks/ --nbmake --nbmake-timeout=120
```

Two things about the format step that will otherwise cost you a confusing hour:

- **The CI `autoformat` job runs the formatters; it does not assert a clean diff.** A
  repo-wide `./run_autoformat.sh` therefore reformats files nobody has touched — as of
  this writing, 28 of them on a clean `main`, 27 from `docformatter` and 3 from `black`,
  purely from drift between whatever formatter versions you happen to have installed and
  the ones the tree was last formatted with. Nothing pins them.
- So: **format the files you actually changed, and revert the rest.** Run
  `black`/`docformatter`/`isort` on your own paths, then `git checkout --` everything
  else before committing. A diff full of unrelated rewrapped docstrings is not
  reviewable.

Line length is 88 (`black`, `docformatter` `wrap-summaries`/`wrap-descriptions`), but
`pylint`'s `max-line-length` is 89 — `black` is the authority, and the extra column just
means pylint never fights it. `isort` uses `profile = "black"` with
`multi_line_output = 2` and `split_on_trailing_comma`, so import blocks wrap
hanging-indent style, not one-per-line-in-parens.

`mypy` runs over the **whole** repo including `tests/` and `scripts/`, with
`strict_equality`, `disallow_untyped_calls` and `warn_unreachable`. Untyped *definitions*
are still allowed, which is why test functions have no annotations (see below), but a
call into an untyped function from typed code is an error.

## `np.random` is banned by a custom pylint checker

`pylint_plugins/no_np_random.py` (loaded via `.pylintrc`'s `load-plugins`) rejects every
`np.random.<x>` except `np.random.default_rng` and `np.random.Generator`. The global RNG
is non-reproducible; construct a local `Generator`, or take the `np_random` that
Gymnasium already threads through `reset(seed=...)`.

Most other pylint noise is disabled in `.pylintrc` (`invalid-name`, the whole `DESIGN`
group, and so on). Docstrings are *not* disabled: every public module, class and function
needs one, and `docformatter` will reflow it.

## Dynamic3D: the headless-rendering trap

`register_all_environments()` (`src/kinder/__init__.py:67-74`) inspects `DISPLAY` and, if
it is unset, forces `MUJOCO_GL`/`PYOPENGL_PLATFORM` to `osmesa` on Linux (`glfw` on
macOS) — overwriting whatever you set. Registration then probes each backend through
`_check_deps`, which catches **every** exception, not just `ImportError`:

```python
    for mod in modules:
        try:
            __import__(mod)
        except Exception:  # pylint: disable=broad-except
            return False
```

That is deliberate — `mujoco` can import-then-fail during initialization when OpenGL is
unavailable — but it means that on a machine with no OSMesa libraries installed, *every*
Dynamic3D environment is skipped **in silence**, and the only symptom is a
`gymnasium.error.NameNotFound` from a much later `kinder.make("kinder/Tossing3D-o1-v0")`.

CI avoids this by installing `libosmesa6-dev` and setting `MUJOCO_GL=osmesa` explicitly.
On a workstation with a GPU but no display, prefer EGL:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

and be aware `register_all_environments()` will clobber both if `DISPLAY` is unset — set
them again *after* that call, and import a dynamic3d **module**
(e.g. `kinder.envs.dynamic3d.envs`) rather than the `kinder.envs.dynamic3d` package,
which does not pull in `mujoco` and so does not surface the failure.

If a whole category of environments has vanished from the registry, this is why. Check
`kinder.get_env_categories()` before debugging anything else.

## Dynamic3D: MimicLabs scene assets

`kinder.make()` calls `_ensure_assets_for_env()`, which — only for Dynamic3D env ids,
and only when `models/assets/mimiclabs_scenes/meshes` is missing — downloads the
MimicLabs scene bundle from Google Drive via `gdown`. It is a few minutes and roughly
2 GB unpacked into the checkout.

- `DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1` suppresses it. CI sets this globally in
  `.github/workflows/ci.yml`, and `run_ci_checks.sh` sets it too.
- `python scripts/download_mimiclabs_assets.py` fetches them explicitly.
- Tests that need a MimicLabs background scene guard themselves rather than assuming:

```python
@pytest.mark.skipif(
    not MIMICLABS_SCENES_DIR.exists(),
    reason="MimicLabs scenes not downloaded. "
    "Run: python scripts/download_mimiclabs_assets.py",
)
```

Each such test file defines its own module-level `MIMICLABS_SCENES_DIR` constant. A test
constructing an env with `scene_bg=False` does not touch these assets and needs no guard.

## Regions, and why they are invisible

Regions are how task JSON declares "where objects may spawn" and "where they count as
goal-satisfying". Two details bite repeatedly.

**Range length.** `MujocoGround._create_regions` and `MujocoGround.sample_pose_in_region`
(`src/kinder/envs/dynamic3d/objects/base.py`) each accept a range of **4** values
(`[x_start, y_start, x_end, y_end]`, with z implied) or **6**
(`[x_start, y_start, z_start, x_end, y_end, z_end]`) and raise `ValueError` otherwise.
`MujocoObject._create_regions` (same file, higher up) accepts **only** the 4-value form —
regions on an object surface get a hardcoded ±0.01 m tolerance instead.

**Ground regions are inflated.** `MujocoGround.ground_placement_threshold` is `0.05`, and
the bounding box grows by that much *per side* on x, y and z, with the floor clamped:

```python
                bbox = [
                    x_start - self.ground_placement_threshold,
                    y_start - self.ground_placement_threshold,
                    max(0.0, z_start - self.ground_placement_threshold),
                    ...
```

So a nominally paper-thin ground region is a 10 cm-wider box in x/y and spans z from 0 to
0.10 by the time anything is tested against it. Membership checks compare against this
inflated bbox, not against the raw JSON numbers — a test asserting on region geometry
must read `region.bbox`, and a goal region that reaches the floor **cannot distinguish a
height**, only an x/y footprint. Two live inconsistencies to be careful of: the implied z
range differs between construction (`0 .. threshold`) and sampling
(`0 .. 2 * threshold`), and `sample_pose_in_region`'s `ValueError` message states the
6-value order as `[x_start, y_start, x_end, y_end, z_start, z_end]`, which is not the
order the code actually unpacks.

**Every region is invisible by default.** The visualisation site's `rgba` defaults to
`[1.0, 0.0, 0.0, 0.0]` — alpha `0`. If you are rendering a scene to check where a region
is, set an `rgba` with a nonzero alpha in the task JSON; otherwise you will see nothing
and conclude the region does not exist.

## Task configs

Dynamic3D tasks are JSON under `src/kinder/envs/dynamic3d/tasks/<EnvName>/`, passed as
`task_config_path=`. Tests either resolve them out of the installed package —

```python
_TASK_CONFIG_PATH = Path(kinder.__path__[0]) / "envs" / "dynamic3d" / "tasks" / ...
```

— or, for fixtures that exist only to exercise mechanics, use the small configs in
`tests/envs/dynamic3d/test_tasks/`. Both are house idioms; pick by whether the config is
a shipped task or a test rig.

Task JSON is *behaviour*: changing a region range changes what counts as success, so it
needs a test asserting the new semantics, and it may invalidate committed demos and docs
GIFs (below).

## Tests

`tests/` mirrors `src/kinder/`. Under `tests/envs/dynamic3d/`, a file covering a task on
a specific robot carries that robot's prefix — `test_tidybot3d_*.py`,
`test_franka3d_pickplace.py`. Unprefixed names are cross-cutting machinery
(`test_controller.py`, `test_objects.py`, `test_placement_samplers.py`) or are named for
the module under test (`test_tidybot_rewards.py` tests
`kinder/envs/dynamic3d/tidybot_rewards.py`). A `_tasks` suffix exists only to
disambiguate a task-level file from a same-named file about scene mechanics
(`test_tidybot3d_cupboard.py` vs `test_tidybot3d_cupboard_tasks.py`).

Conventions in those files, all of which a new test should match:

- **No return annotations on `test_*` functions.** Zero of the ~266 test functions under
  `tests/envs/dynamic3d/` are annotated. Module-level `_helper` functions *are* annotated.
- Module docstring is `"""Tests for <the thing>."""`; test docstrings start with `Test `.
- Build environments by constructing the `ObjectCentric*Env` class directly rather than
  through `kinder.make()` — 44 direct constructions against 5 `make()` calls — and pass
  `seed=` to `reset()` so failures reproduce. Reach for `make()` only when the test is
  specifically about the registered id or the Gymnasium wrapper stack.
- Reach into privates with an inline `# pylint: disable=protected-access` **at the call
  site** — `env._check_goals()  # pylint: disable=protected-access`. There are ~121 of
  these across `tests/`. Do not add a private-access wrapper helper to hide them; the
  noise is the point.
- Call `env.close()` at the end of a test that constructed one. Dynamic3D envs hold
  simulator resources.
- Video/visualisation is opt-in through the `tests/conftest.py` flags: import
  `from tests.conftest import MAKE_VIDEOS` and guard with `if MAKE_VIDEOS:`. Never render
  unconditionally — `pytest tests/` runs on CI without a display.

`tests/test_deterministic_demo_replay.py` and `test_deterministic_demo_resettable.py`
replay every demo under `demos/`; `tests/demo_blacklist.py` is where a genuinely
flaky-in-CI or slow demo gets excluded, with a written reason.

## Adding an environment

Per `README.md`: implement it under `src/kinder/envs/<category>/`, register it in
`src/kinder/__init__.py`, collect at least one demonstration with
`scripts/collect_demos.py`, and make a video with `scripts/generate_demo_video.py`.

Registration is not the same job in every category. For 2D and Kinematic3D you write an
explicit block in the matching `_register_*()` function: ids **must** start with
`kinder/`, and you register both the variant ids and the class (via
`_register_env_class`, which is what `docs/` and `get_env_classes()` read).

**Dynamic3D is different — it is derived from the filesystem.** `_register_dynamic3d()`
walks `src/kinder/envs/dynamic3d/tasks/`, treats each subdirectory as a class and each
JSON inside it as a variant, and picks the env class from the config's **first `robots`
key** (`tidybot` → `TidyBot3DEnv`, `fr3` → `Franka3DEnv`, `rby1a` → `RBY1A3DEnv`).
Consequences:

- Dropping a task JSON into an existing task folder registers a new environment id. No
  code change needed.
- A malformed task JSON — bad JSON, missing `robots`, or an unrecognised robot key —
  raises `RuntimeError` out of `register_all_environments()`, which breaks *every*
  environment for everyone, not just that task.
- Variant order follows `Path.iterdir()`, so it is filesystem order, not sorted. Anything
  that indexes into `variants` (`generate_env_docs.py` picks `variants[len(variants) // 2]`
  as the representative for a class-level GIF) is picking a variant you cannot predict
  from the filenames. Do not write code that assumes `-o1` comes first.

## Docs assets are committed, and regeneration is not automatic

`docs/envs/**` — the per-environment markdown plus the GIFs under
`docs/envs/assets/` — is generated by `scripts/docs/generate_env_docs.py` and checked in.
The script `git add`s its output itself.

By default it regenerates only environments it thinks changed, and the change test is
narrower than it looks: `get_changed_files()` takes `git diff origin/main --name-only`
and keeps paths under `src/kinder/`, but `is_env_changed()` then compares that set
against `inspect.getfile(env.unwrapped.__class__)` — **the environment class's module
file**. So editing a task JSON, an asset, or a scene will *not* trigger regeneration even
though the rendered GIFs are now stale. Force it:

```bash
python scripts/docs/generate_env_docs.py --env Tossing3D
python scripts/docs/generate_env_docs.py --force          # everything, slow
```

If a change alters what a scene looks like, say so explicitly in the PR and either
regenerate the affected environment or note that the assets are stale.

## Pull requests

`README.md`'s contributing section is the whole process: everything goes through review,
and all of `./run_ci_checks.sh` must pass before merge. Run it locally rather than
discovering a docformatter complaint on CI.
