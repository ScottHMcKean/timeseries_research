# %% [markdown]
# # A Whirlwind Time Series Tour With Nixtla
# 
# <img src="../resources/nixtla_logo.png" width="300"/>
# 
# First, create and activate a virtual environment (run these in terminal/command prompt):
# - `python -m venv nixtla_env`
# - `source nixtla_env/bin/activate  # On Windows use: nixtla_env\Scripts\activate`
# 
# Install requirements:
# - `pip install -r requirements.txt`

# %% [markdown]
# Here is the agenda for this tutorial:
# - We start by loading a dataset using the datasetsforecast library. 
# - We then look at the basics, seasonality, autocorrelation, and stationarity.
# - We move on to use traditional forecasting models.
# - We close out using machine learning based forecasting models.

# %%
from datasetsforecast.long_horizon import LongHorizon
import pandas as pd

# Load the Weather dataset (downloads all long horizon datasets at first, takes a while)
y_df, x_df, _ = LongHorizon.load(directory='../data', group='Weather')
y_df.ds = pd.to_datetime(y_df.ds)
x_df.ds = pd.to_datetime(x_df.ds)

# %%
# Note the specific dataset format here, ds, unique_id, value
display(y_df.sample(10))

# %%
from utilsforecast.plotting import plot_series
plot_series(y_df, engine='plotly')

# %% [markdown]
# You'll notice that there are a lot of erroneous values in this dataset. I wouldn't even call them outliers, they're just bad data, especially in the case of max PAR column. But let's use a simple IQR-based outlier removal technique to clean it up.

# %%
def remove_outliers_iqr(group, excluded_groups=['rain (mm)']):
    if group['unique_id'].iloc[0] in excluded_groups:
        return group
    Q1 = group['y'].quantile(0.05)
    Q3 = group['y'].quantile(0.95)
    IQR = Q3 - Q1
    group.loc[(group['y'] < Q1 - 1.5 * IQR) | (group['y'] > Q3 + 1.5 * IQR), 'y'] = float('nan')
    return group

# Apply IQR-based outlier removal to each group
y_df = y_df.groupby('unique_id').apply(remove_outliers_iqr).reset_index(drop=True)

# Fill gaps in time series data uisng bfill and ffill
y_df = (
    y_df
    .groupby('unique_id')
    .apply(lambda x: x.set_index('ds').asfreq('10min').ffill().bfill())
    .drop("unique_id", axis=1)
    .reset_index()
)

# %%
plot_series(y_df, engine='plotly')

# %% [markdown]
# Before we start forecasting, let's look at some basic things about time series data - time series data is composed of a trend, a seasonal component, and a residual component. It is fundamentally defined by autocorrelation and we leverage that autocorrelation a LOT when forecasting.

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get a single time series
name = 'T (degC)'
group = y_df[y_df['unique_id'] == name]
ts = group.set_index('ds')['y']

# Perform seasonal decomposition
decomposition = seasonal_decompose(ts, period=144)

# Calculate PACF
pacf_values = pacf(ts.dropna(), nlags=14)

# Create subplots
fig = make_subplots(
    rows=5, cols=1,
    subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual', 'PACF'),
    vertical_spacing=0.1,
)

# Add decomposition components
fig.add_trace(
    go.Scatter(x=ts.index, y=ts.values, name='Original'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=ts.index, y=decomposition.trend, name='Trend'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=ts.index, y=decomposition.seasonal, name='Seasonal'),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=ts.index, y=decomposition.resid, name='Residual'),
    row=4, col=1
)
fig.add_trace(
    go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF'),
    row=5, col=1
)

# Update layout
fig.update_layout(
    height=1000,
    title_text=f"Time Series Decomposition and PACF for {name}",
    showlegend=False
)

fig.show()

# %% [markdown]
# The next two important things are stationarity and homeoskedacity. Stationarity is the property of a time series where the mean and variance are constant over time. Homeoskedacity is the property of a time series where the variance is constant over time.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get a single time series (using temperature as an example)
name = 'T (degC)'
group = y_df[y_df['unique_id'] == name]
ts = group.set_index('ds')['y']

# Calculate rolling statistics
window_sizes = [144, 720]  # 1 and 5 day window (with 10-min data)
rolling_stats = {}

for window in window_sizes:
    rolling_stats[window] = {
        'mean': ts.rolling(window=window).mean(),
        'std': ts.rolling(window=window).std()
    }

# Create subplots
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Rolling Mean', 'Rolling Standard Deviation'),
    vertical_spacing=0.15
)

# Add original series and rolling means
fig.add_trace(
    go.Scatter(x=ts.index, y=ts.values, name='Original', opacity=0.3),
    row=1, col=1
)

colors = ['red', 'blue']
for (window, color) in zip(window_sizes, colors):
    # Add rolling mean
    fig.add_trace(
        go.Scatter(
            x=rolling_stats[window]['mean'].index,
            y=rolling_stats[window]['mean'].values,
            name=f'{window/144:.0f}d Mean',
            line=dict(color=color)
        ),
        row=1, col=1
    )
    
    # Add rolling std
    fig.add_trace(
        go.Scatter(
            x=rolling_stats[window]['std'].index,
            y=rolling_stats[window]['std'].values,
            name=f'{window/144:.0f}d Std',
            line=dict(color=color)
        ),
        row=2, col=1
    )

# Update layout
fig.update_layout(
    height=800,
    title_text=f"Rolling Statistics for {name} (Visualizing Stationarity)",
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.05
    )
)

# Update y-axes labels
fig.update_yaxes(title_text="Temperature", row=1, col=1)
fig.update_yaxes(title_text="Standard Deviation", row=2, col=1)

fig.show()

# %% [markdown]
# ## Traditional Forecasting
# 
# Let's do some traditional forecasting! We are going to take three common models and use them to forecast the temperature.
# 
# ### ARIMA
# ARIMA (AutoRegressive Integrated Moving Average) is a statistical model that combines autoregression, differencing, and moving average components to analyze and forecast time series data. AutoARIMA automatically determines the optimal parameters (p,d,q) for the ARIMA model by testing different combinations and selecting the one that best fits the data based on information criteria like AIC or BIC. ARIMA models are particularly useful for time series that exhibit trends and seasonality, making them well-suited for temperature forecasting.
# 
# ### Exponential Smoothing
# Exponential Smoothing is a forecasting technique that uses a weighted average of past observations, with more weight given to recent observations. It is particularly useful for time series with trends and seasonality, as it can adapt to changing patterns over time. We use a slightly more advanced method for exponential smoothing called Holt Winters
# 
# ### Seasonal Naive
# Seasonal Naive is a simple forecasting method that uses the last observed value from the same season to predict future values. It is effective for time series with strong seasonal patterns, such as temperature data, where the same pattern repeats regularly.

# %% [markdown]
# Let's split the data into train and test sets. We are going to use the last 30 days (144*30 observations) as test set.

# %%
test_size = 144 * 30  # 30 days of 10-minute intervals

y_train = y_df.groupby('unique_id').apply(
    lambda x: x.iloc[:-test_size]
).reset_index(drop=True)

y_test = y_df.groupby('unique_id').apply(
    lambda x: x.iloc[-test_size:]
).reset_index(drop=True)

# %%
from statsforecast import StatsForecast
from statsforecast.models import (
    SeasonalNaive,
    HoltWinters,
    ARIMA,
)

models = [
    HoltWinters(season_length=144),
    ARIMA(order=(1,1,1), season_length=144),
    SeasonalNaive(season_length=144)
]

sf = StatsForecast(
    models=models, 
    freq='10min', 
    n_jobs=-1)

# %% [markdown]
# It is worth nothing why I didn't call sf.fit() here. This is because the StatsForecast library is designed to be used in a pipeline where you fit the models on a training set and then use the same models to forecast on a test set. In this case, we are just going to fit the models on the entire dataset and then use the same models to forecast on the entire dataset.

# %%
series = ["T (degC)", "rain (mm)"]

forecasts_df = sf.forecast(
    df=y_train.query("unique_id in @series"),
    h=144*30, 
    level=[80]
    )
forecasts_df.head()

# %% [markdown]
# Lets' have a peek!

# %%
plt = sf.plot(
    df=y_df[y_df['ds'] > '20201101'], 
    forecasts_df=forecasts_df, 
    unique_ids=["T (degC)"], 
    level=[80], 
    engine='plotly',
)
plt.update_traces(visible=True, selector=dict(name='y'))  # Then show only y trace
plt.show()

# %% [markdown]
# Let's evaluate how our forecasts performed against our holdout set. We use the evaluator in the utilsforecast library for this purpose. It's worth nothing here that metrics matter a lot in time series forecasting, for example if you are predicting a large or small value, or one that is frequently zero.

# %%
from utilsforecast.losses import mae, mape, rmse, smape
from utilsforecast.evaluation import evaluate

eval_df = forecasts_df.merge(y_test, on=['unique_id', 'ds'], how='inner')

evaluation = evaluate(
    df = eval_df,
    metrics=[mae, mape, rmse, smape],
    models=['HoltWinters', 'ARIMA', 'SeasonalNaive']
    )
evaluation.drop(columns=['unique_id']).style.background_gradient(cmap='RdYlGn_r', axis=1)

# %% [markdown]
# Okay, the example above showed seasonality and classical autoregressive forecasting. Now let's dive into machine learning based forecasting! In this example, we are going to use lag features, differencing, and a couple other time series

# %%

from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean

# %%
# Pivot the data to get unique_ids as columns
pivot_df = y_df.pivot(index='ds', columns='unique_id', values='y')

# Calculate correlation matrix
corr_matrix = pivot_df.corr()

# Create correlation heatmap
import plotly.express as px

fig = px.imshow(
    corr_matrix,
    labels=dict(x="Variable", y="Variable", color="Correlation"),
    color_continuous_scale="RdBu",
    aspect="auto"
)

fig.update_layout(
    title="Correlation Matrix Between Variables",
    xaxis_title="",
    yaxis_title=""
)

fig.show()


# %% [markdown]
# We select two features here, VPmax (mbar) and H2OC (mmol/mol). We avoid some of the higher correlated features because they are collinear with the above features. We want independent information.

# %%
corr_matrix['T (degC)'].abs().sort_values(ascending=False).head(10)
# VPmax (mbar)            0.968229
# H2OC (mmol/mol)         0.759763

# %%
pivot_df = (y_df
    .pivot(index='ds', columns='unique_id', values='y')
    .reset_index()
    [['ds', 'T (degC)', 'VPmax (mbar)', 'H2OC (mmol/mol)']]
    .assign(unique_id='T (degC)')
    .rename(columns={'T (degC)': 'y'})
)
pivot_df.head(5)

# %% [markdown]
# Let's leverage the amazing power of the MLForecast library to create lag features, define two models, and fit them to our dataframe of lagged features and exogenous variables.

# %%
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

models={
        'lightgbm': lgb.LGBMRegressor(verbosity=-1),
        'knn': KNeighborsRegressor(),
        'mlp': MLPRegressor(),
        'linear': LinearRegression()
    }
# Let's create lag features
fcst = MLForecast(
    models=models,
    freq='10min',
    lags = [6, 48, 144],
    lag_transforms={
        1: [ExpandingMean()],
        24: [RollingMean(window_size=48)],
        144: [RollingMean(window_size=144)],
    }   
)
prep = fcst.preprocess(pivot_df, static_features=[])
display(prep.head(5))

y_train = pivot_df.groupby('unique_id').apply(
    lambda x: x.iloc[:-test_size]
).reset_index(drop=True)

y_test = pivot_df.groupby('unique_id').apply(
    lambda x: x.iloc[-test_size:]
).reset_index(drop=True)

# %%
crossvalidation_df = fcst.cross_validation(
    df=y_train,
    h=144*30,
    n_windows=4,
    refit=False,
    static_features=[]
)
crossvalidation_df.head()

# %%
from utilsforecast.losses import mae, mape, rmse, smape
from utilsforecast.evaluation import evaluate

def evaluate_crossvalidation(crossvalidation_df, models):
    evaluations = []
    for c in crossvalidation_df['cutoff'].unique():
        df_cv = crossvalidation_df.query('cutoff == @c')
        evaluation = evaluate(
            df = df_cv,
            metrics=[mae, mape, rmse, smape],
            models=list(models.keys())
            )
        evaluations.append(evaluation)
    evaluations = pd.concat(evaluations, ignore_index=True).drop(columns='unique_id')
    evaluations = evaluations.groupby('metric').mean()
    return evaluations.style.background_gradient(cmap='RdYlGn_r', axis=1)

evaluate_crossvalidation(crossvalidation_df, models)

# %%
y_test

# %%
fcst_fit = fcst.fit(
    df=y_train,
    static_features=[]
)

forecasts_df = fcst_fit.predict(
    X_df=y_test,
    h=144*30
)

# %%
plt = sf.plot(
    df=y_df[y_df['ds'] > '20201101'], 
    forecasts_df=forecasts_df, 
    unique_ids=["T (degC)"],
    engine='plotly',
)
plt.update_traces(visible=True, selector=dict(name='y'))  # Then show only y trace
plt.show()


