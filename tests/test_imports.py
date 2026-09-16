import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "src",
        "src.config",
        "src.runtime",
        "src.data",
        "src.models",
        "src.baselines",
        "src.evaluation",
        "src.services",
        "src.visualization",
        "pages",
        "app",
    ],
)
def test_project_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
