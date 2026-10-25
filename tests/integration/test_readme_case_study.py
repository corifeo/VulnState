"""The README's PHP2Shell case study must run and match what the screenshot shows."""

import ast
import re
from datetime import datetime
from pathlib import Path

import pytest

from vulnstate import CVDArray

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def php2shell():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = [b for b in re.findall(r"```python\n(.*?)```", readme, re.S) if "php2shell =" in b]
    assert len(blocks) == 1, "README should contain exactly one PHP2Shell code block"
    namespace: dict = {}
    exec(blocks[0], namespace)
    return namespace["php2shell"]


def test_readme_claims(php2shell):
    assert php2shell.is_coordinated
    assert php2shell.is_premature_disclosure
    assert not php2shell.has_fix_before_exploit
    assert not php2shell.is_zero_day


def test_desiderata_match_readme_text(php2shell):
    arr = CVDArray([php2shell])
    arr.transform()
    assert len(arr.get_violated_desiderata()[0]) == 8  # "8 of the 12 desiderata are violated"
    assert arr.desiderata_score[0] == pytest.approx(4 / 12)  # "score 0.33"
    assert "Attacks Before Disclosure" in arr.get_violated_desiderata()[0]


def test_screenshot_uses_readme_timeline(php2shell):
    """scripts/make_readme_assets.py needs matplotlib, so read its timeline without importing."""
    tree = ast.parse((ROOT / "scripts" / "make_readme_assets.py").read_text(encoding="utf-8"))
    node = next(
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "PHP2SHELL_TIMELINE" for t in n.targets)
    )
    script = {
        row.elts[0].attr: datetime(*(ast.literal_eval(a) for a in row.elts[1].args))
        for row in node.elts
    }
    readme = {event.name: ts for event, ts in php2shell.events.items() if ts is not None}
    assert script == readme
