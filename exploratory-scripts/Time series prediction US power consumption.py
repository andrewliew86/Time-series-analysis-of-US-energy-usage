# Time series prediction of US energy consumption over time
# Dataset was taken from here https://catalog.data.gov/dataset/monthly-energy-consumption-by-sector
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from math import sqrt
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima

df = pd.read_excel('data-raw/Energy_consumption_US_data.xlsx', parse_dates=['Date'], index_col='Date')


df_residential = df[['Residential']]
# Check the data
print(df_residential.head(50))
print(df_residential.info())
print(df_residential.describe())
print(df_residential.isnull().sum())

# Do a quick check of the number of entries for missing dates(e.g. skipped dates).
import datetime
start_date = datetime.datetime(1973, 1, 1)
end_date = datetime.datetime(2019, 12, 1)
num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
print('Number of months in our dataset: {}'. format(num_months+1))
# We know that the dates in our dateset are between 1973-01-01 to 2019-12-01
# According to this formula, there are 564 months between our dates which matches exactly the number of rows in our dataframe. So this is good!


# Plot the data
df_residential.plot(title='US residential energy consumption over time', marker='o')
plt.ylabel('Energy units')
plt.xlabel('Date')
plt.show()

# Lets check if there is a trend in the dataset using the adf_residential adfuller method
result = adfuller(df_residential['Residential'])
print("The adf_residentialuller test statistic for dataset is {}". format(result))
print("The adf_residentialuller p-value for dataset is {}". format(result[1]))
# The adf_residentialuller says that the p-value is 0.05 (rounded) so it is >= 0.05 so it is not stationary so lets take the diffference

df_residential_diff = df_residential.diff().dropna()
result = adfuller(df_residential_diff['Residential'])
print("The adf_residentialuller p-value for dataset (after diff) is {}". format(result[1]))
# The adf_residentialuller says that the p-value is now less than 0.05 (rounded) so its stationary now!

# Lets look for the best AR and MA orders
# Lets check for seasonality and trend using seasonal decomposition
decomposition = sm.tsa.seasonal_decompose(df_residential_diff)
decomposition.plot()
plt.show()
# You can clearly see that there is a yearly seasonality trend!

# Lets take a seasonal diff and then plot the acf and pacf to determine the P and Q for the sesonal SARIMA component
df_residential_seasonal_diff = df_residential_diff.diff(12).dropna()
fig, ax = plt.subplots(2, 1)
plot_acf(df_residential_seasonal_diff, lags=[12, 24, 36, 48, 60], ax=ax[0])
plot_pacf(df_residential_seasonal_diff, lags=[12, 24, 36, 48, 60], ax=ax[1])
plt.show()
# Here is a handy guide for selecting models and the order of the models for a single time series.
# If the time series shows these characteristics, select the appropriate model and order!
# Note: Make sure you make your time series stationary before calling the PACF or ACF method on it!
'''
     AR(p)                    MA(q)                      ARMA(p,q)
ACF  Tails off                Cuts off after lag q       Tails off
PACF Cuts off after lag p     Tails off                  Tails off
'''
# I noticed that both pacf and acf are 'tailing off'
# So for the seasonal component of the SARIMA, the order should be (1,1,1,12)


# Lets try out different ARMA model orders using a 'gridsearch' method to determine optimal orders of the model
# Create empty list to store search results
order_aic_bic = []
# Loop over p values from 0-2
for p in range(0, 3):
    # Loop over q values from 0-2
    for q in range(0, 3):
        # create and fit ARMA(p,q) model
        model = sm.tsa.statespace.SARIMAX(df_residential, order=(p, 1, q), seasonal_order=(1, 1, 1, 12))
        results = model.fit()
        # Append order and results tuple
        order_aic_bic.append((p, q, results.aic, results.bic))

# Construct DataFrame from order_aic_bic
order_df_residential = pd.DataFrame(order_aic_bic, columns=['p', 'q', 'AIC', 'BIC'])

# Print order_df_residential in order of increasing AIC
print(order_df_residential.sort_values('AIC'))
# The systematic method says that p = 1 and q = 1 is best!


# Lets also plot the acf and pacf to corroborate our 'gridsearch' results and determine the AR and MA model order for SARIMA
fig, ax = plt.subplots(2, 1)
plot_acf(df_residential_seasonal_diff, zero=False, ax=ax[0])
plot_pacf(df_residential_seasonal_diff, zero=False, ax=ax[1])
plt.show()
# I noticed that both pacf and acf are 'tailing off' so p=1, q=1 should be ok! This is also consistent with the gridsearch method!
# So for the non-seasonal component of the SARIMA, the order will be (1,1,1)


# You should also check diagnostic features of your model
model = sm.tsa.statespace.SARIMAX(df_residential, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()
results.plot_diagnostics()
plt.show()
# Most of the datapoints in the Q-Q plot lie on the red line so this means that the residuals are probably normally distributed = good!
# The histogram also shows (almost) normal distribution which is good and
# The residuals all look quite evenly distributed and no obvious pattern!
# The collegram shows that >95% of correlations for lag greater than one are NOT be significant... This means that there isn't much 'correlation' in the residuals to explain the model... so that is good!

# Lets use the PACF and ACF lag data to do a forecast!
# Lets check the model summary!
print(results.summary())

# Generate predictions for the last 50 days of your data using get_prediction
# Note that you can use the get_forecast method to get 'out of sample predictions' only! just state the number of steps i.e number of out-of-sample data
one_step_forecast = results.get_prediction(start=-12, end='2020-06-01')

# Extract prediction mean
mean_forecast = one_step_forecast.predicted_mean
print(mean_forecast)

# Get confidence intervals of  predictions
confidence_intervals = one_step_forecast.conf_int()
print(confidence_intervals)

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:, 'lower Residential']
upper_limits = confidence_intervals.loc[:, 'upper Residential']

# Print best estimate  predictions
print(mean_forecast)

# Lets make a smaller subset so we can see our predictions better. I am going to use loc to subset the data to a smaller window!
df_residential_subset_predict = df_residential.loc['2018-01-01':, :]
# Lets project out our one-step-ahead prediction data on the actual values
plt.plot(df_residential_subset_predict.index, df_residential_subset_predict, label='observed', marker='o')  # note that df_residential.index refers to the dates (used for the x argument)
mean_forecast.plot(x=mean_forecast.index, y=mean_forecast, color='r', label='forecast', marker='v')
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color='pink')

# set labels, legends and show plot
plt.title('One-step-ahead prediction for energy consumption')
plt.xlabel('Date')
plt.ylabel('Residential energy consumption')
plt.legend()
plt.show()

# You can see from the data that there is a prediction that energy consumption peaks in January then falls all the way until July before picking up again!
# The forecast is not too bad!
# We can also use calculate the errors of the model using RMSE
actual = df_residential_subset_predict.iloc[-12:]  # The 12 refers to the fact that I only selected the last 12 datapoints in my data for predictions... So I am using these to calculate my RMSE!
forecast = mean_forecast.iloc[0:12]
rmse_df = pd.concat([actual, forecast], axis=1)  # Make the two pandas series into a dataframe for display
rmse_df.columns = ['Actual', 'Forecast']
print(rmse_df)
rms = sqrt(mean_squared_error(actual, forecast))
print('Root mean squared error: ')
print(rms)


# We can also try and use auto arima to see if we end up getting the same model orders!
# auto arima is an automated to get the best model order based on the aic
# Note that you still need to tell auto arima whether:
# a) there should be any seasonal differencing (D) that we need to take into account... Use the seasonal_decomposition method to determine this!
# b) there should be any non-seasonal differencing (d) that we need to take into account... Use the seasonal_decomposition method to determine this or just look at the raw plot!
stepwise_model = auto_arima(df_residential, start_p=1, start_q=1,
                            max_p=4, max_q=4, m=12,
                            start_P=0, seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
print(stepwise_model.aic())
# According to autoarima the best (lowest AIC) model is this:
# Fit ARIMA: order=(1, 1, 1) seasonal_order=(1, 1, 2, 12); AIC=6004.552, BIC=6034.734, Fit time=26.653 seconds
# This is quite similar to our predictions!

# We can also look at forecasting energy usage of the other sectors
