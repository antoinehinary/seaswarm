from importlib.metadata import version

import seaswarm


def test_installed_version_matches_the_package() -> None:
    assert version("seaswarm") == seaswarm.__version__
