### M3 Financial Time Series Forecasting

### [**Contents**](#)
1. [Description](#descr)
2. [Installation](#install)
3. [Data Preparation](#prepare)
4. [Training](#train)
5. [Metrics](#metrics)
6. [References](#ref)

---

### [**Description**](#) <a name="descr"></a>
The [M3 time series competition](https://forecasters.org/resources/time-series-data/m3-competition/), organized by the [International Institute of Forecasters (IIF)](https://forecasters.org/), stands as one of the most influential and comprehensive forecasting competitions in the field of time series analysis. The competition was designed to address the limitations and challenges faced by forecasters in accurately predicting various types of time series data, including economic indicators, financial market trends, demographic patterns, and industrial demand, among others.

The focus of this project is on the financial dimension of the M3 competition due to its intricacy and scarcity of available data. The table below illustrates how the financial data is partitioned into intervals, along with the corresponding average length and prediction horizon for each interval:

Details of the financial data from M3 forecasting competition
| Interval | # | Length (avg) | Prediction Horizon |
| --- | --- | --- | --- |
| Yearly | 58 | 36 | 6 |
| Quarterly | 76 | 52 | 8 |
| Monthly | 145 | 124 | 18 |
| Other | 29 | 95 | 8 |
| Total | 308 | | |

Due to the lack of consistency in the 'Other' interval, we opted to exclude it and instead focus on training the models for three distinct intervals: Yearly, Quarterly, and Monthly.

---

### [**Installation**](#) <a name="install"></a>

**1.** Clone the repository:

``` shell
$ git clone git@github.com:Rohit8y/Financial-Time-Series.git
$ cd Financial-Time-Series
```

**2.** Create a new Python environment and activate it:

``` shell
$ python3 -m venv py_env
$ source py_env/bin/activate
```

**3.** Install necessary packages:

``` shell
$ pip install -r requirements.txt
```

---

### [***Data Preparation***](#) <a name="prepare"></a>

The raw data from the official competition can be accessed and downloaded from the [IIF website](https://forecasters.org/data/m3comp/M3C.xls). However, as our focus is solely on the financial data, we have processed the raw data and stored it in pickle format, categorized according to the specified intervals. These pickle files are available in the [data/](https://github.com/Rohit8y/Time-Series-Forecasting/tree/main/data) folder.

---

### [***Training***](#) <a name="train"></a>
We can train our models using different time periods for the data: yearly, quarterly, or monthly intervals. Additionally, we have the choice of two model architectures: Recurrent Neural Networks (RNNs) or Gated Recurrent Units (GRUs). To initiate the training process, execute the following command:
```
python main.py -h

usage: main.py [-h] [--freq FREQ] [--window_size WIN_SIZE] [--arch ARCH] [--epochs EPOCHS] [--lr LR]
               [--batch-size BS] [--wd WD] [--optimizer OPT] [--result_path RESULT]
general options:
  --help                show this help message and exit
  --epochs              number of total epochs to run (default: 100)
  --lr                  initial learning rate (default: 0.001)
  --batch-size          mini-batch size (default: 16)
  --wd                  weight decay (default: 1e-4)

|----------------------------------------------------------------------------------------------------------------------|
model configuration:
  --freq                defines the range of data to be processed (0->Year | 1->Quarter | 2->Month)
  --window_size         the window size refers to the duration of observations to consider for training (default: 5)
  --arch                choose one of the following model architectures (rnn | gru)
  --input_size          the number of expected features in the input x, should be the same as window_size (default: 5)
  --hidden_size         the number of features in the hidden state h (default: 512)
  --num_layers          number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to
                        form a stacked RNN (default: 2)
  --output_dim          the number of features in the output (default: 1)
  --dropout             dropout probability (default: 0.2)
  --result_path         the path where the final output models are saved (default: output)
```

---

### [**Metrics**](#) <a name="metrics"></a>

In line with the accuracy measures of the M3-Competition, we selected symmetric mean absolute percentage error (sMAPE) and the mean absolute percentage error (MAPE) for our metrics[[1]](#1). The sMAPE and MAPE are calculated as below, with n representing the length of the data sequence, $F_t$ the forecast at time $t$ and $A_t$ the actual value at time $t$. Both measures form an assessment of the percentage of errors:


$$\text{sMAPE} = \frac{100\%}{n} \sum^{n}_{t=1} {\frac{|F(t) - A(t)|}{(|A(t)| + |F(t)|)/2}}$$

$$\text{MAPE} = \frac{100\%}{n} \sum^{n}_{t=1} \left| {\frac{A(t) - F(t)}{A(t)}} \right|$$

These measures are biased towards larger values, meaning that if $A_t > F_t$ by a given distance d, it yields a more promising MAPE than if $F_t > A_t$ by the same distance d.

---

### [**References**](#) <a name="ref"></a>

<a id="1">[1]</a> 
Makridakis, Spyros. "Accuracy measures: theoretical and practical concerns." *International journal of forecasting 9, no. 4 (1993): 527-529.*

