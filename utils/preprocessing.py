import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.scaler import Scaler

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


def get_evaluation_df(df):
    nf = df.iloc[0, 2]  # number of points ['NF'] which are meant to use for evaluation
    for i in range(nf):
        dfDataEvalSeries = df.iloc[len(df) - nf + i:len(df) - nf + i + 1, 6:]
        evalSeries = df.iloc[len(df) - nf + i:len(df) - nf + i + 1, 0:1].values.tolist()[0][0]


def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n


def generateFeatures(dfData, window_size):
    df_features = pd.DataFrame()
    for counter in range(len(dfData)):
        dataList = []
        for i in range(len(dfData.iloc[counter])):
            if not np.isnan(dfData.iloc[counter, i]):
                dataList.append(dfData.iloc[counter, i])
        dataListDf = pd.DataFrame(dataList, columns=['value'])
        # Generate time lag features of window_size
        df_generated = generate_time_lags(dataListDf, window_size)
        # Merge the DF
        frames = [df_features, df_generated]
        df_features = pd.concat(frames)
    return df_features


def feature_label_split(df, target_col):
    y = df[[target_col]]
    x = df.drop(columns=[target_col])
    return x, y


def scaleData(args, X_train, X_val, Y_train, Y_val):
    args.xScaler = Scaler()
    args.yScaler = Scaler()

    X_train = args.xScaler.fit(X_train)
    Y_train = args.yScaler.fit(Y_train)

    X_val = args.xScaler.transform(X_val)
    Y_val = args.yScaler.transform(Y_val)

    return X_train, X_val, Y_train, Y_val


def inverse_transform_visualise(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform_visualise(scaler, df_result, [["value", "prediction"]])
    return df_result
