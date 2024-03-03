import argparse

from utils.loader import get_train_loader, get_val_loader
from utils.preprocessing import load_df, get_train_eval_df, generateFeatures, feature_label_split, scaleData

parser = argparse.ArgumentParser(description="M3C time series forecasting")
parser.add_argument('--freq', default=0, type=int,
                    help='Defines the range of data to be processed, 0->Year, 1->Quarter, 2->Month')
parser.add_argument('--window_size', default=5, type=int,
                    help='The window size refers to the duration of observations to consider for training')
parser.add_argument('--arch', default='rnn', type=str,
                    help='Choose one of the following models (rnn | gru)')
# |--------------------------------------------- Training Hyperparameters ---------------------------------------------|
parser.add_argument('--epochs', default=100, type=int,
                    help='number of epochs to train')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size to use for fine tuning')
parser.add_argument('--learning_rate', default=0.001, type=float,
                    help='learning rate for the optimizer')
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='type of optimizer to use [sgd,adam]')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')

if __name__ == "__main__":
    args = parser.parse_args()

    # Assertion checks
    assert 0 <= args.freq <= 3
    assert args.arch in ['rnn', 'gru']

    print("Assertions complete...")

    # Load data and get train, val sets
    df = load_df(args.freq)
    df_train, df_val = get_train_eval_df(df)

    # Get features for train and val set using lag of window_size
    df_train_features = generateFeatures(df_train, args.window_size)
    df_val_features = generateFeatures(df_val, args.window_size)

    print("Size of TRAIN set after splitting in windows: ", len(df_train_features))
    print("Size of VAL set after splitting in windows: ", len(df_val_features))

    # Split features and target label from data frame
    X_train, Y_train = feature_label_split(df_train_features, 'value')
    X_val, Y_val = feature_label_split(df_val_features, 'value')

    # Scale dataset
    X_train, X_val, Y_train, Y_val = scaleData(args, X_train, X_val, Y_train, Y_val)

    # Get data loaders
    train_loader = get_train_loader(X_train, Y_train, args.batch_size, shuffle=False, drop_last=True)
    val_loader = get_val_loader(X_val, Y_val, args.batch_size, shuffle=False, drop_last=True)

    print("Length of train loader: ", len(train_loader))
    print("Length of val loader: ", len(val_loader))
