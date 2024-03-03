from matplotlib import pyplot as plt


def plotEvalPrediction(df_result, arch, title):
    valueList = df_result.iloc[:, 0].values.tolist()
    predicList = df_result.iloc[:, 1].values.tolist()
    xAxis = list(range(len(df_result)))

    plt.rcParams["figure.figsize"] = (15, 5)
    plt.plot(xAxis, valueList, label="True Value")
    plt.plot(xAxis, predicList, label=arch)

    plt.title(title)
    plt.legend()
    plt.savefig(title + '.png')
    plt.clf()
