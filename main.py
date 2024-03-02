import argparse

parser = argparse.ArgumentParser(description="M3C time series forecasting")
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
