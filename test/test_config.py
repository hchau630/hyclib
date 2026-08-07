import pathlib

import pytest

import hyclib as lib


@pytest.mark.parametrize(
    "filename",
    [
        ("test_load_config.toml"),
        ("test_load_config.json"),
    ],
)
def test_load(filename, pytestconfig):
    data_path = pathlib.Path(pytestconfig.rootdir) / "test" / "data"
    config = lib.config.load(data_path / filename)
    assert config == {"a": 1, "b": "hi", "c": True, "d": [10, 11]}


def test_dump(tmp_path):
    config = {"a": 1, "b": "hi", "c": True, "d": [10, 11]}
    filename = tmp_path / "config.json"
    lib.config.dump(config, filename)
    loaded_config = lib.config.load(filename)
    assert loaded_config == config
