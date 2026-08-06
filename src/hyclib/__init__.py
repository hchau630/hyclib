from .core import pprint, functools, itertools, random

import importlib
import logging

logger = logging.getLogger(__name__)

modules = [".np", ".pt", ".sp"]

for module in modules:
    try:
        importlib.import_module(module, package="hyclib")
    except ImportError as err:
        logger.info(f"Did not import {module=} due to ImportError {str(err)}.")

del importlib, modules, module, logging
