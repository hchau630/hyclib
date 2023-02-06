from time import perf_counter
import logging

logger = logging.getLogger(__name__)

# copied from https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
class timeit:
    def __init__(self, float_format='{:.4f}'):
        self.float_format = float_format
    
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'Time: {self.float_format.format(self.time)} seconds'
        logger.info(self.readout)