import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def make_cfg():
    def _make(*overrides):
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            return compose("config", overrides=list(overrides))

    return _make
