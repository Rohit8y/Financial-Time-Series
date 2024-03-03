import argparse
import os

from torch import nn, optim

from models.arch import get_model, save_checkpoint
from utils.loader import get_train_loader, get_val_loader
from utils.metrics import sMAPE
from utils.opt import Optimization
from utils.preprocessing import load_df, get_train_eval_df, generateFeatures, feature_label_split, scaleData

parser = argparse.ArgumentParser(description="M3C time series forecasting")
parser.add_argument('--freq', default=0, type=int,
                    help='Defines the range of data to be processed, 0->Year, 1->Quarter, 2->Month')
parser.add_argument('--window_size', default=5, type=int,
                    help='The window size refers to the duration of observations to consider for training')
# |------------------------------------------------ Model configuration -----------------------------------------------|
parser.add_argument('--arch', default='rnn', type=str,
                    help='Choose one of the following models (rnn | gru)')
parser.add_argument('--input_size', default=5, type=int,
                    help='The number of expected features in the input x')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='The number of features in the hidden state h')
parser.add_argument('--num_layers', default=2, type=int,
                    help='Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together'
                         ' to form a stacked RNN,')
parser.add_argument('--output_dim', default=1, type=int,
                    help='The number of features in the output')
parser.add_argument('--dropout', default=0.2, type=float,
                    help='dropout probability')
parser.add_argument('--result_path', default="output", type=str,
                    help='the path where models are saved')
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
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

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

    all_sMape = 0
    iteration = 1
    for i in range(iteration):
        # Build Model
        args.model_params = {'input_dim': args.input_size,
                             'hidden_dim': args.hidden_size,
                             'layer_dim': args.num_layers,
                             'output_dim': args.output_dim,
                             'dropout_prob': args.dropout}
        model = get_model(args.model_params, args.arch)

        # Training essentials
        loss_fn = nn.MSELoss(reduction="mean")
        optimizer = None
        if args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

        # Training
        opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
        opt.train(train_loader, val_loader, batch_size=args.batch_size, n_epochs=args.epochs,
                  n_features=args.window_size)

        opt.plot_losses()

        # Evaluation
        val_one_loader = get_val_loader(X_val, Y_val, 1, shuffle=False, drop_last=True)
        predictions, values = opt.evaluate(val_one_loader, batch_size=1, n_features=args.window_size)
        sMAPE_score = sMAPE(predictions, values)
        all_sMape += sMAPE_score
        print(i, ": sMape for validation set: ", sMAPE_score)

        # Saving model
        if i == 0:
            save_checkpoint(args, model)

    print("Average sMape:", all_sMape / iteration)
