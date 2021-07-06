#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config_test.py
@Contact :   910660298@qq.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/2 20:59   daxiongpro    1.0         None
'''

# import lib
from easydict import EasyDict as edict
d = edict()
d.n = edict()
d.n.m = 'hello'
d.n.h = [1,2.3]
print(d.n.m)
print(d.n.h)