from __future__ import division

import numpy as np
import pandas as pd

def mem_usage(df):
    if isinstance(df, pd.DataFrame):
        usage_bytes = df.memory_usage(deep=True).sum()
    else:
        # series
        usage_bytes = df.memory_usage(deep=True)
    usage_mbytes = usage_bytes / (1024 ** 2)
    return "{:03.2f} MB".format(usage_mbytes)
