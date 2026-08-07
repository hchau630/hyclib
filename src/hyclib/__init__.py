import importlib
import logging

from .core import config, functools, itertools, pprint, random

logger = logging.getLogger(__name__)

modules = [".np", ".pt", ".sp"]

for module in modules:
    try:
        importlib.import_module(module, package="hyclib")
    except ImportError as err:
        logger.info(f"Did not import {module=} due to ImportError {err!s}.")

del importlib, modules, module, logging
