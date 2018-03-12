# source link from: https://zhuanlan.zhihu.com/p/34301270
from __future__ import division

import numpy as np
import pandas as pd
import hashlib

def ab_split(key, salt, control_group_ratio):
    """
    The object is to split users or entities (identified by key) 
    into treatment group and control group
    Parameters:
        key: identification of user or entity
        salt: different values in different tests
        control_group_ratio: the ratio bewteen treatment and control group
    Returns:
        c: control group
        t: treatment group
    """

    full_id = "{}-{}".format(key, salt)

    # md5(id) -> get the first six digits -> convert into hex number
    hashed_id = hashlib.md5(full_id.encode("ascii")).hexdigest()
    hashed_id = hashed_id[:6]
    hashed_id = int(hashed_id, 16)

    result_ratio = hashed_id / 0xFFFFFF

    if result_ratio > control_group_ratio:
        return 't'
    else:
        return 'c'

if __name__ == "__main__":
    users = pd.DataFrame({"id": np.arange(10000)})
    users["group"] = users.id.apply(lambda x: ab_split(x, "test", 0.7))

    print sum(users.group == 'c') / users.count()

