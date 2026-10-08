import importlib


def test_package_imports_and_has_version():
    pkg = importlib.import_module("autogui_anno")
    assert hasattr(pkg, "__version__")
    assert isinstance(pkg.__version__, str) and pkg.__version__
