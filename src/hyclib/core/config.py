import json
import logging
import os
import pathlib
from importlib import resources

import platformdirs
import tomli

from . import itertools

logger = logging.getLogger(__name__)


def load(filename):
    filename = str(filename)
    if filename.endswith(".toml"):
        with open(filename, "rb") as f:
            config = tomli.load(f)
    elif filename.endswith(".json"):
        with open(filename, "r") as f:
            config = json.load(f)
    else:
        raise NotImplementedError()

    return config


def dump(config, filename):
    filename = str(filename)
    if filename.endswith(".json"):
        with open(filename, "w") as f:
            json.dump(config, f, indent=4)
    else:
        raise NotImplementedError()


def package_config_locs(
    package_name,
    package_author=None,
    package_version=None,
    default_config_path="config.toml",
):
    default_config_filename = resources.files(package_name).joinpath(
        default_config_path
    )

    user_config_filenames = []

    user_config_filenames.append(pathlib.Path(f"{package_name}_config.toml"))
    try:
        path = pathlib.Path(os.environ[f"{package_name.upper()}_CONFIG"])
    except KeyError:
        pass
    else:
        user_config_filenames.append(
            path if path.is_file() else path / f"{package_name}_config.toml"
        )
    user_config_filenames.append(
        pathlib.Path(
            platformdirs.user_config_dir(
                package_name, appauthor=package_author, version=package_version
            )
        )
        / "config.toml"
    )

    return {
        "default_config": default_config_filename,
        "user_configs": user_config_filenames,
    }


def load_package_config(*args, **kwargs):
    """
    Loads configs from various config files to be imported and used anywhere in a
    package. This is meant to be used in the top-level __init__.py file in the package.

    It first loads default package configs at default_config_path
    (relative to the top level directory of the package) if such a file exists.

    Next, it loads user-defined configs in the following priority, from highest to
    lowest:
        1. f'{package_name}_config.toml' in the directory in which the top level code is
           run
        2. f'${package_name.upper()}_CONFIG' if it is a file,
           f'${package_name.upper()}_CONFIG/{package_name}_config.toml' otherwise
        3. f'{platformdirs.user_config_dir(package_name, appauthor=package_author,
              version=package_version)}/config.toml'

    If there are overlapping configs, the configs from the config file with the higher
    priority will be used.

    Reminder: If you want to add a default config file to your package, remember to set
    include_package_data=True in setup.py to include it in your package distribution.
    """
    filenames = package_config_locs(*args, **kwargs)

    default_config_filename = filenames["default_config"]
    if default_config_filename.is_file():
        with resources.as_file(default_config_filename) as default_config_file:
            config = load(default_config_file)

        logger.debug(f"Loaded default config file at {default_config_filename}.")
    else:
        config = {}

    user_config_filenames = [
        filename for filename in filenames["user_configs"] if filename.is_file()
    ]
    for filename in reversed(user_config_filenames):
        itertools.dict_update(config, load(filename))

    if len(user_config_filenames) > 0:
        logger.debug(
            "Loaded user config files at the following locations from highest to"
            f" lowest priority: {user_config_filenames}"
        )

    return config
