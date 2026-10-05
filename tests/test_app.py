"""Every dashboard page renders without exceptions before the pipeline has run (empty states)."""
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

APP_DIR = str(Path(__file__).resolve().parents[1] / "app")


def _page(app_dir: str, cfg_path: str, module: str):
    import sys

    import yaml

    sys.path.insert(0, app_dir)
    import importlib

    from ui.theme import inject_css

    inject_css()
    importlib.import_module(f"ui.{module}").page(yaml.safe_load(open(cfg_path)))


@pytest.mark.parametrize("module", ["home", "analyze", "compare", "jury_page", "insights", "explore", "method"])
def test_page_renders_without_artifacts(tmp_path, cfg, module):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    at = AppTest.from_function(_page, args=(APP_DIR, str(path), module), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def _real_page(app_dir: str, module: str):
    import importlib
    import sys

    sys.path.insert(0, app_dir)
    from viralsense.utils import load_config

    importlib.import_module(f"ui.{module}").page(load_config())


REAL = Path(__file__).resolve().parents[1] / "reports" / "tables" / "classification_test.csv"


@pytest.mark.skipif(not REAL.exists(), reason="real pipeline results not present")
@pytest.mark.parametrize("module", ["home", "insights", "jury_page", "explore", "method", "compare", "analyze"])
def test_page_renders_with_real_results(module):
    at = AppTest.from_function(_real_page, args=(APP_DIR, module), default_timeout=300)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_form_keys_never_reused_as_session_state_keys():
    """Regression: st.form("ab") + st.session_state["ab"] = ... raises StreamlitWidgetAlreadyInstantiatedError."""
    import re

    for f in (Path(APP_DIR) / "ui").glob("*.py"):
        src = f.read_text()
        forms = set(re.findall(r'st\.form\(\s*"([^"]+)"', src))
        state = set(re.findall(r'session_state\[\s*"([^"]+)"\s*\]', src)) | set(re.findall(r'session_state\.get\(\s*"([^"]+)"', src))
        assert not forms & state, f"{f.name}: form key reused in session_state: {forms & state}"
