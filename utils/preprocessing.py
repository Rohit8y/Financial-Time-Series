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


def get_train_eval_df(df):
    nf = df.iloc[0, 2]  # number of points ['NF'] which are meant to use for evaluation

    # Extract the train set by removing the last nf points and ignoring the first 6 columns
    df_train = df.iloc[0:len(df) - nf, 6:]
    print("Length of train set:", len(df_train))

    # Extract the val set by taking last nf points and ignoring the first 6 columns
    df_val = df.iloc[len(df) - nf:, 6:]
    print("Length of val set:", len(df_val))

    return df_train, df_val
