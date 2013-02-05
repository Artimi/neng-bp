#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class FormatError(RuntimeError):
    """Signalize erorr with format of input string"""
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
        