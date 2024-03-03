import pandas as pd
from enum import Enum

CATEGORY = 'FINANCE     '
SHEET_NAMES = ['M3Year', 'M3Quart', 'M3Month', 'M3Other']


class Freq(Enum):
    """
    Enum for labeling different sampling frequencies
    """
    Y = 0  # year
    Q = 1  # quart
    M = 2  # month
    O = 3  # other


def load_df(freq: Freq, path='./data/'):
    """
    Loads FINANCE data frame according to provided sampling frequency
    :param freq: enum of the series type
    :param path: base path to the data directory
    :return:
    """
    return pd.read_pickle(path + SHEET_NAMES[freq.value] + '.pkl')
