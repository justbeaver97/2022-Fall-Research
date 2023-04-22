"""
reference:
    https://dodonam.tistory.com/185
"""

import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Arguemtn \"%s\" is not a list" % (s))
    return v
