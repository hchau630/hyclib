import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # bucketize currently does not support MPS

def pytest_make_parametrize_id(config, val, argname):
    if config.option.verbose >= 2:  # -vv or -vvv
        return f"{argname}={val}"
    return repr(val)  # the default
