# Stock predictor using Recurrent Neural Network

[Link to Final Project Presentation](https://www.canva.com/design/DAFfHN1wFoo/9yFiEDgxaCdU4FkWLmVd4A/edit)

[Link to python code](https://github.com/aryan1113/Beat-The-Market/blob/main/predictor.py)

### Table of contents

[TOC]

## Instructions to run file

<ol>
<li>Download the predictor.py file
<li>Check if xgboost is installed locally, if not type "pip install xgboost" on command window.
<li>Close the matplotlib figure to see the ratios calculated.
</ol>

## Libraries and versions
Library | Version
---|---|
matplotlib|3.6.2
seaborn|0.11.2
numpy|1.23.2
pandas|1.4.3
sklearn|1.1.2
keras|2.10.0
xgboost|1.7.4
<hr>

## Methodology
<ol>
<li>Removing null values
<li>Removing data from days where volume traded was below a certain threshold, to avoid days when stock wasn't actively traded.
<li> Using a stacked LSTM to build relation between the target feature and 4input features.

</ol>

### Key Decisions 

1. Adj Close was chosen as the target variable as Adj Close represents the value of share accounting for **Corporate Actions** , using "Close" is not a good idea as it only represents the cash/in-hand value of the share. The predictor had to be trained on long term data so we chose to analyze the Adj Close value of a particular share.
2. Why only a Recurrent Neural Network ?
As we had to regress over past values, some sort of network was required that made use of past memory. Settled upon LSTM's as using gates control of information can be controlled.
2. Used MinMaxScaler instead of StandardScalar as the "Adj Close" feature did not follow a normal distribution.
3. TimeSeriesSplit provides train/test indices to split time series data samples that are observed at fixed time intervals, in train/test sets.
4. Choice of Activation Function
ReLU is chosen as the activation function to avoid underflow error due to vanishing gradient problem.
5. Stacked LSTM or not
Stacked LSTM is prefferable for sequence prediction problems, these allow for more complex models .
6. Choice of Evaluation Metric
For a numeric regression problem, we have limited evaluation metrics. The chosen parameters are Root Mean Square Error and Mean Square Error.

$RMSE = \sqrt{Σ ( Yᵢ - Ŷᵢ ) ²\over N }$

$MSE = {Σ ( Yᵢ - Ŷᵢ ) ²\over N}$

<hr>

## Network Architecture

2 LSTM layers are used with 2 dropout layers with dropout ratio =0.5, which indicates that 50% of neurons are randomly dropped during training, to avoid overfitting.<br>

![model_summary](https://user-images.githubusercontent.com/87320561/229703257-8ee313d4-8634-4098-a37d-20d8aa03bf56.jpg)

<hr>

## What do the ratios mean ?

Since the ratios are based on historical data, it is important to note that this does not necessarily indicate future performance, and one ratio should not be the only factor relied upon for investing decisions.

### Sharpe Ratio

$Sₐ = {R_{p}-r_{f}\over\sigma_{a}}$

Where $R_{p}$ is the return on portfolio,  $r_{f}$ is the risk free return of portfolio and $\sigma_{a}$ is the standard deivation of expected returns.

Sharpe ratio is defined as difference over time between expected returns and benchmark, which is usually the government bond rate divided by the standard deviation in returns.

### Sortino Ratio

$Sortino = {R_{p}-r_{f}\over\sigma_{d}}$

Where $\sigma_{d}$ is the negative downside of portfolio
Difference over time between expected returns and benchmark, which is usually the government bond rate divided by only the negative standard deviation in returns.

Because the Sortino ratio focuses only on the negative deviation of a portfolio's returns from the mean, it is thought to give a better view of a portfolio's risk-adjusted performance since positive volatility is a benefit.

### Treynor Ratio

$Treynor Ratio = {r_{p}-r{f}\overβ_{p}}$

Also known as Reward-to-Volatility ratio, it is the risk-adjusted measure of return based on systematic risk. </br>
Beta measures the tendency of a portfolio's return to change in response to changes in return for the overall market.

### Cumulative returns

$Cumulative Return = {Current Price-Original Price\over Original Price}$

Total Change in investment price over a set time.
Taxes are ignored in these calculations.

### Max Drawdown

$MDD={Trough Value -Peak Value\over Peak Value}$

A maximum drawdown (MDD) is the maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained




## Results

We have trained the network on data from Stock A (historical data present for around 10years, 3000+ data points) for 50epochs and from the table below we can decipher the signifance of dropout layers in preventing overfitting.

### Tabular Comparsion of Ratios

Ratio |Without Dropout|With Dropout
--- | --- | --- |
MSE Train|9.85|497.65
RMSE Train |3.14|22.31
Sharpe|0.5371|0.5194
Sortino |0.0510| 0.0492
Treynor |-0.0090| -0.0067
Cumulative Returns |5.1008| 3.8568
Max Drawdown |1.9066| 1.5081
Highest Actual Return | 0.1378| 0.1330


---


### Graph depicting model performance
Graph showing prediction (in Blue) by LSTM with Dropout Layers  and True Value of stock (in yellow)

![Stock A LSTM with dropout](https://user-images.githubusercontent.com/87320561/229703187-56851281-94ab-4877-9c2e-dc9bf01873c5.png)

Graph showing the prediction by LSTM without Droupout (shows overfitting)

![Stock A LSTM without dropout](https://user-images.githubusercontent.com/87320561/229703111-f6d4cce2-c9d2-4b40-9358-078a9f576dc8.png)

### Issues with the model
1. Dataset size is quite small (3000rows)
2. The model prediction shows negative bias for all instances.
3. LSTM's are slow to run as we cannot parallelize the training due to recurrance relation.
4. Stock D only has 748rows.

