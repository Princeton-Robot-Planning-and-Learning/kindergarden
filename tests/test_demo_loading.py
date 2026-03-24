"""Fast smoke test that demo files can be loaded without corruption."""

from kinder.utils import find_all_demo_files, load_demo


def test_load_sample_of_demos():
    """Load a sample of demos across different environments to catch corruption.

    Picks the first demo from each environment directory so we get broad coverage
    without loading all 1200+ files.
    """
    all_files = find_all_demo_files()
    assert len(all_files) > 0, "No demo files found"

    # Pick one demo per environment (parent.parent is the env dir)
    seen_envs: set[str] = set()
    sample: list = []
    for f in all_files:
        env_name = f.parent.parent.name
        if env_name not in seen_envs:
            seen_envs.add(env_name)
            sample.append(f)

    assert len(sample) >= 5, f"Expected demos from >=5 envs, got {len(sample)}"

    for path in sample:
        demo = load_demo(path)
        assert demo["env_id"].startswith(
            "kinder/"
        ), f"{path}: env_id {demo['env_id']!r} has wrong prefix"
        assert len(demo["observations"]) > 0
        assert len(demo["actions"]) > 0
