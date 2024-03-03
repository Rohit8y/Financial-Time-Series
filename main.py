import argparse

from utils.preprocessing import load_df, get_train_eval_df

parser = argparse.ArgumentParser(description="M3C time series forecasting")
parser.add_argument('--freq', default=0, type=int,
                    help='Defines the range of data to be processed, 0->Year, 1->Quarter, 2->Month')
parser.add_argument('--arch', default='rnn', type=str,
                    help='Choose one of the following models (rnn | gru)')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of epochs to train')
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
