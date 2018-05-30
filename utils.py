# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

from __future__ import division
import numpy as np


class Diff(object):
    def __init__(self, v=0.):
        self.value = v
        self.diff = 0.

    def update(self, v):
        if self.value:
            diff = (v / self.value - 1)
            self.diff = diff
        self.value = v


def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


# TODO general pretty print
def pretty_list(list):
    return ', '.join(list)


def pretty_dict(dict):
    return '\n'.join('{} : {}'.format(k, v) for k, v in dict.items())


def closest_power_of_two(target):
    if target > 1:
        for i in range(1, int(target)):
            if (2 ** i >= target):
                pwr = 2 ** i
                break
        if abs(pwr - target) < abs(pwr/2 - target):
            return pwr
        else:
            return int(pwr / 2)
    else:
        return 1
