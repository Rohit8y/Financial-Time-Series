def sMAPE(trueValues, predictedValues):
    score = 0
    for i in range(len(trueValues)):
        real = trueValues[i]
        predicted = predictedValues[i]
        smape = (abs(real - predicted) / ((real + predicted) / 2)) * 100
        score += smape
    return score / len(trueValues)
