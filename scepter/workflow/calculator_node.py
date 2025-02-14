# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from .constant import WORKFLOW_CONFIG


class CalculatorNode:
    def __init__(self):
        self.cfg = WORKFLOW_CONFIG.workflow_config

    CATEGORY = '🪄 ComfyUI-Scepter'

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'parameter': ('INT',),
                'type': (list(s().cfg['CALCULATOR']['TYPE']),),
                'value': ('FLOAT',),
                'round_method': (list(s().cfg['CALCULATOR']['ROUND']),)
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ('INT',)
    RETURN_NAMES = ('INT',)
    FUNCTION = 'execute'

    def execute(self, parameter, type, value, round_method):
        _OPERATIONS = {
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'div': lambda a, b: a / b if b != 0 else _raise_zero_division()
        }

        _ROUND_METHODS = {
            'ceil': math.ceil,
            'floor': math.floor,
            'round': round
        }

        def _raise_zero_division():
            raise ValueError("Division by zero is not allowed.")

        if not isinstance(parameter, (int, float)) or not isinstance(value, (int, float)):
            raise TypeError("Parameters must be int or float.")

        try:
            operation = _OPERATIONS[type]
        except KeyError:
            raise ValueError(f"Invalid type: {type}") from None

        try:
            res = operation(parameter, value)
        except ZeroDivisionError:
            raise ValueError("Division by zero is not allowed") from None

        try:
            round_func = _ROUND_METHODS[round_method]
        except KeyError:
            raise ValueError(f"Invalid rounding method: {round_method}") from None

        return (round_func(res),)
