#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tqdm_test.py    
@Contact :   910660298@qq.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/22 14:05   daxiongpro    1.0         None
'''

# import lib
import time

from tqdm import tqdm

for i in tqdm(range(100)):
    time.sleep(0.05)