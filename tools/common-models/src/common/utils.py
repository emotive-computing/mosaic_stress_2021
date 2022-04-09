import errno
import os
import re
import shutil

import numpy as np


def concatenate(a, b):
    a = np.asarray(a)
    if a.size == 0:
        a = np.asarray(b)
    else:
        a = np.concatenate((a, b), axis=0)

    return a

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_inverse_binary_values(arr):
    return [x^1 for x in arr]


def capitalize(s):
    return s[0].upper() + s[1:]

def to_camelcase(s):
    return capitalize(re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), s))


def flatten(l):
    return [item for sublist in l for item in sublist]

def combine_if_list(fn, x):
    return x if not isinstance(x, list) else fn(x)

def remove_value_from_array(arr, value):
    while value in arr: arr.remove(value)
    return arr

def class_has_method(kls, m):
    return hasattr(kls, m) and callable(getattr(kls, m))


def ensure_directory(dirname):
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def ensure_directory_for_file(filename):
    directory = os.path.dirname(filename)
    ensure_directory(directory)


def parse_from_file(value):
    value = parse_float(value)
    value = parse_tuple(value)
    value = parse_bool(value)
    return value

def parse_bool(s):
    if s == "None":
        return None
    elif s == "True":
        return True
    elif s == "False":
        return False
    else:
        return s

def parse_tuple(string):
    try:
        s = eval(string)
        if type(s) == tuple:
            return s
        return string
    except:
        return string

def parse_float(value):
    try:
        return float(value)
    except ValueError:
        return value


class try_without_failure(object):

    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        try:
            self.f(*args)
        except:
            pass

@try_without_failure
def try_delete_file(filename):
    os.remove(filename)

@try_without_failure
def try_delete_directory(directory):
    shutil.rmtree(directory)
