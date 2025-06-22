# Time-series-analysis-of-US-energy-usage
Background: Forecasting energy usage is an important step in ensuring that the energy needs of the community. Energy usage often follows a seasonal pattern (i.e. usage peaks during the winter months) and future energy use can therefore be forecasted using time series analysis tools and ML tools (e.g. xgboost).

Results: I performed exploratory data analysis and was able to forecast energy usage by US residents over time using a SARIMA model (see figure below). I also trained an xgboost regression model to predict energy usage using engineered features such as month, year and lags.

The results can inform decision makers and ensure appropriate resources are avaialble during peak periods.

Python libraries/tools: Pandas, Statsmodel (tsa models), matplotlib, pmdarima, xgboost

Dataset was taken from here: https://catalog.data.gov/dataset/monthly-energy-consumption-by-sector


![alt tag](https://github.com/andrewliew86/Time-series-analysis-of-US-energy-usage/blob/master/Forecast_time%20series.PNG)
