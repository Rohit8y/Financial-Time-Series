import argparse

import torch
from torch import nn, optim

from models.arch import get_model
from utils.loader import get_val_loader
from utils.metrics import sMAPE
from utils.opt import Optimization
from utils.plot import plotEvalPrediction
from utils.preprocessing import load_df, generateFeatures, feature_label_split, format_predictions

parser = argparse.ArgumentParser(description="M3C time series visualising")
parser.add_argument('--model_path', default='output/rnn_model.pth', type=str,
                    help="path of trained model to visualise time series")
# |--------------------------------------------- Training Hyperparameters ---------------------------------------------|
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

    checkpoint_dict = torch.load(args.model_path, map_location=torch.device('cpu'))

    # Get model specific configurations
    state_dict = checkpoint_dict['state_dict']
    model_params = checkpoint_dict['model_params']
    arch = checkpoint_dict['arch']
    freq = checkpoint_dict['freq']
    xScaler = checkpoint_dict['xScaler']
    yScaler = checkpoint_dict['yScaler']
    window_size = checkpoint_dict['window_size']

    # Build Model
    model = get_model(model_params, arch)
    model.load_state_dict(state_dict)
    model.eval()
    print("Checkpoint loaded...")

    # Model essentials
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = None
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.learning_rate,
                              weight_decay=args.weight_decay)
    opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)

    # Get val series one at a time and evaluate
    df = load_df(freq)
    nf = df.iloc[0, 2]  # number of points ['NF'] which are meant to use for evaluation
    for i in range(nf):
        df_series = df.iloc[len(df) - nf + i:len(df) - nf + i + 1, 6:]
        evalSeries = df.iloc[len(df) - nf + i:len(df) - nf + i + 1, 0:1].values.tolist()[0][0]

        print("Evaluating series: ", evalSeries)

        df_series_features = generateFeatures(df_series, window_size)
        X_val, Y_val = feature_label_split(df_series_features, 'value')

        X_val_scaled = xScaler.transform(X_val)
        Y_val_scaled = yScaler.transform(Y_val)

        val_loader = get_val_loader(X_val_scaled, Y_val_scaled, 1, shuffle=False, drop_last=True)

        predictions, values = opt.evaluate(val_loader, batch_size=1, n_features=window_size)
        df_result = format_predictions(predictions, values, X_val, yScaler)
        sMAPE_score = sMAPE(df_result.iloc[0].values.tolist(), df_result.iloc[:, 1].values.tolist())
        print('sMAPE ', arch, ': ', sMAPE_score)
        plotEvalPrediction(df_result, arch, title=evalSeries)
