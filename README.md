### M3 Finanacial Time Series Forecasting

### [**Contents**](#)
1. [Description](#descr)
2. [Installation](#install)
3. [Data Preparation](#prepare)
4. [Training](#train)
5. [References](#ref)

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
git clone git@github.com:Rohit8y/Time-Series-Forecasting.git
cd Time-Series-Forecasting
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



---

### [**References**](#) <a name="ref"></a>

