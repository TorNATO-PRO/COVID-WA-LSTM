"""
Predicting the number of COVID-19 deaths given time series data.

Dataset being used: https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

from darts import TimeSeries
from darts.dataprocessing.transformers import (
    Scaler,
    MissingValuesFiller
)
from darts.models import RNNModel

# define the pipeline
filler = MissingValuesFiller()
scaler = Scaler()

# define the datatypes that the dataset can contain
datatypes = {
    'date': str,
    'county': 'category',
    'state': 'category',
    'fips': 'category',
    'cases': int,
    'deaths': int
}


def read_csv() -> pd.DataFrame:
    # read the csv containing covid data
    loc = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    if os.path.exists('/home/waltz/Documents/Personal Projects/COVID-WA-LSTM/assets/us-counties.csv'):
        loc = '/home/waltz/Documents/Personal Projects/COVID-WA-LSTM/assets/us-counties.csv'

    temp_csv_data = pd.read_csv(loc)
    temp_csv_data['date'] = pd.to_datetime(temp_csv_data['date'])
    return temp_csv_data

def convert_to_timeseries(df: pd.DataFrame, time: str = 'date', value: str = 'deaths') -> TimeSeries:
    return TimeSeries.from_dataframe(df, time, value, freq='D', fill_missing_dates=True)

def filter_by_county(df: pd.DataFrame, state: str = 'Washington', county: str = 'Pierce', value: str = 'deaths') -> pd.DataFrame:
    df = df[(df['county'] == county) & (df['state'] == state)]
    df.sort_values('date', inplace=True, ascending=True)
    df.drop(df.columns.difference(['date', value]), axis=1, inplace=True)
    return df

def retrieve_time_series_data(state: str, county: str, value: str, df: pd.DataFrame = None):
    if df is None:
        df = read_csv()
    return convert_to_timeseries(filter_by_county(df, state, county, value), 'date', value)

possible_features = ['cases', 'deaths']

for observation in possible_features:
    series = retrieve_time_series_data('Washington', 'Pierce', observation)

    training_cutoff = pd.Timestamp('20220201')
    train, test = series.split_after(training_cutoff)
    train_transformed = scaler.fit_transform(train)
    test_transformed = scaler.transform(test)

    # covariates actually make the performance worse? no seasonality or something :P
    # covariates = datetime_attribute_timeseries(series, attribute='year', one_hot=False)
    # covariates = covariates.stack(datetime_attribute_timeseries(series, attribute='month', one_hot=False))
    # covariates = covariates.stack(
    #     TimeSeries.from_times_and_values(
    #         times=series.time_index,
    #         values=np.arange(len(series)),
    #         columns=['linear_increase']
    #     )
    # )
    # covariates = covariates.astype(np.float32)

    # covariant_scaler = Scaler()
    # covariant_train, covariant_test = covariates.split_after(training_cutoff)
    # covariant_scaler.fit(covariant_train)
    # covariates_transformed = covariant_scaler.transform(covariates)

    # define RNN model
    my_model = RNNModel(
        model="LSTM",
        model_name="COVID-LSTM",
        hidden_dim=8,
        dropout=0.1,
        batch_size=14,
        n_epochs=80,
        optimizer_kwargs={"lr": 1e-3},
        log_tensorboard=True,
        random_state=42,
        training_length=15,
        input_chunk_length=7,
        force_reset=True,
        save_checkpoints=True,
    )

    my_model.fit(series=train_transformed,
                 val_series=test_transformed,
                 verbose=True)
    pred = my_model.predict(30)
    pred = scaler.inverse_transform(pred)
    observation_pp = observation.capitalize()
    series.plot(label=f"Actual # of {observation_pp}")
    pred.plot(label=f"{observation_pp} forecast")
    plt.title(f"COVID-19 {observation_pp} Forecast - Pierce County")
    plt.savefig(f"COVID-19-{observation_pp}-Forecast.png")
    plt.close()
