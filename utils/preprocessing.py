import pandas as pd

CATEGORY = 'FINANCE     '
SHEET_NAMES = ['M3Year', 'M3Quart', 'M3Month', 'M3Other']


def load_df(freq, path='./data/'):
    """
    Loads FINANCE data frame according to provided sampling frequency
    :param freq: enum of the series type
    :param path: base path to the data directory
    :return:
    """
    print("Loading", SHEET_NAMES[freq], "data...")
    return pd.read_pickle(path + SHEET_NAMES[freq] + '.pkl')
