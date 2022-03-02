# COVID Cases/Deaths Predictor

Author: Nathan Waltz

I am creating an LSTM-RNN with 8 hidden layers to predict how many deaths there will be on a given day in various states/counties using this [dataset](https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv). 

Usage of this program requires installation of the `Darts` package as well as `Pytorch`. You can do this either via Anaconda or pip!

### Run Instructions

#### Install Requirements

You only need to do this once. Make sure that you have `python3` and `pip3` installed on your system.

```
$ pip install -r requirements.txt
```

#### Run Program

```
$ python3 ./analyze.py
```

## Projected COVID-19 Deaths (as of March 02, 2021)

![COVID-19 Deaths](COVID-19-Deaths-Forecast-Washington-Pierce.png)

![COVID-19 Deaths](COVID-19-Deaths-Forecast-Washington-Whitman.png)

![COVID-19 Deaths](COVID-19-Deaths-Forecast-Idaho-Ada.png)

## Plot of COVID-19 Cases (as of March 02, 2021)

![COVID-19 Cases](COVID-19-Cases-Forecast-Washington-Pierce.png)

![COVID-19 Cases](COVID-19-Cases-Forecast-Washington-Whitman.png)

![COVID-19 Deaths](COVID-19-Cases-Forecast-Idaho-Ada.png)
