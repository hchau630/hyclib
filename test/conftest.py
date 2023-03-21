import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # bucketize currently does not support MPS