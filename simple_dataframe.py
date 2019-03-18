#! /usr/bin/env python

import pandas as pd
import numpy as np


x = np.random.randint(1, 100, (5, 3))
y = pd.DataFrame(x, columns=['a', 'b', 'c'])

