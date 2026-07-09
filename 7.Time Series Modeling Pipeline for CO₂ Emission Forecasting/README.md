<div style="border:1px solid #ccc; border-radius:8px; padding:25px; font-family:Arial, sans-serif; line-height:1.7;">

<h1> Time Series Modeling Pipeline for CO₂ Emissions Forecasting</h1>

<h3>Predicting CO₂ Emissions Using Statistical & Deep Learning Time Series Models</h3>

<hr>

<h2> Project Overview</h2>

<p style="text-align:justify;">
Climate change mitigation requires accurate forecasting of greenhouse gas emissions to support
energy planning, policy development, and decarbonization initiatives. This project develops an
end-to-end time series forecasting pipeline for <b>monthly CO₂ emissions from electricity generation
(2013–2025)</b>. <a href="https://www.eia.gov/electricity/data.php#elecenv" target="_blank">dataset</a>.
</p>

<p style="text-align:justify;">
The project combines <b>Exploratory Data Analysis (EDA)</b>, feature engineering,
statistical forecasting techniques, and deep learning architectures to understand historical
emission patterns and generate reliable future forecasts.
</p>

<p style="text-align:justify;">
Multiple forecasting approaches are compared, including
<b>ARIMA, SARIMA, Holt-Winters Exponential Smoothing, Prophet,
LSTM, GRU, and RNN</b>, providing a comprehensive evaluation of
traditional statistical models and modern deep learning methods.
</p>

<hr>

<h2> Engineering / Business Problem</h2>

<p>
Electricity generation remains one of the largest contributors to global greenhouse gas emissions.
Governments, utility companies, and environmental agencies require reliable forecasting models to:
</p>

<ul>
<li>Forecast future CO₂ emissions</li>
<li>Evaluate decarbonization strategies</li>
<li>Understand seasonal demand fluctuations</li>
<li>Support carbon reduction policies</li>
<li>Improve long-term energy planning</li>
</ul>

<p>
Traditional statistical approaches often struggle with complex nonlinear relationships, whereas
deep learning models require extensive preprocessing and feature engineering.
This project evaluates both methodologies to determine the most suitable forecasting solution.
</p>

<hr>
<h2> Objectives</h2>

<ul>
<li>Analyze historical CO₂ emission patterns</li>
<li>Identify long-term trends</li>
<li>Detect seasonal behavior</li>
<li>Detect volatility and emission spikes</li>
<li>Engineer temporal features</li>
<li>Compare statistical forecasting models</li>
<li>Compare deep learning forecasting models</li>
<li>Evaluate forecasting accuracy</li>
<li>Predict future CO₂ emissions</li>
</ul>

<hr>

<h2> Dataset Description</h2>

<table style="border-collapse:collapse; width:100%;">
<tr>
<th style="border:1px solid #ccc; padding:8px;">Property</th>
<th style="border:1px solid #ccc; padding:8px;">Description</th>
</tr>

<tr><td style="border:1px solid #ccc;padding:8px;">Source</td><td style="border:1px solid #ccc;padding:8px;">Energy Information Administration (EIA)</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Dataset</td><td style="border:1px solid #ccc;padding:8px;">Electric Power Monthly</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Period</td><td style="border:1px solid #ccc;padding:8px;">January 2013 – December 2025</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Frequency</td><td style="border:1px solid #ccc;padding:8px;">Monthly</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Target Variable</td><td style="border:1px solid #ccc;padding:8px;">Total CO₂ Emissions</td></tr>

</table>

<hr>

<h2> Data Cleaning & Preprocessing</h2>

<ul>
<li>Duplicate detection</li>
<li>Missing value analysis</li>
<li>Data type validation</li>
<li>Datetime conversion</li>
<li>Datetime indexing</li>
<li>Chronological ordering</li>
<li>Forward/Backward Fill</li>
<li>Interpolation</li>
<li>Mean & Median Imputation</li>
</ul>

<pre>
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
</pre>

<hr>

<h2> Feature Engineering</h2>

<table style="border-collapse:collapse;width:100%;">
<tr>
<th style="border:1px solid #ccc;padding:8px;">Feature</th>
<th style="border:1px solid #ccc;padding:8px;">Purpose</th>
</tr>

<tr><td style="border:1px solid #ccc;padding:8px;">Year</td><td style="border:1px solid #ccc;padding:8px;">Long-term trend</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Month</td><td style="border:1px solid #ccc;padding:8px;">Seasonality</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Day of Year</td><td style="border:1px solid #ccc;padding:8px;">Seasonal progression</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Day of Week</td><td style="border:1px solid #ccc;padding:8px;">Weekly behavior</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Hour</td><td style="border:1px solid #ccc;padding:8px;">Hourly variation</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Weekend</td><td style="border:1px solid #ccc;padding:8px;">Weekend effect</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Season</td><td style="border:1px solid #ccc;padding:8px;">Winter / Spring / Summer / Fall</td></tr>

</table>

<hr>

<h2> EDA & Models Results</h2>

<ul>
<li>Monthly Trend → assets/monthly_trend.png</li>
<li>Seasonal Analysis → assets/seasonal_pattern.png</li>
<li>Outlier Detection → assets/outlier_boxplot.png</li>
<li>Distribution → assets/distribution.png</li>
<li>Rolling Mean & Volatility → assets/volatility.png</li>
<li>Spike Detection → assets/spikes.png</li>
<li>Fuel Contribution → assets/fuel_distribution.png</li>
<li>LSTM Training → assets/lstm_training.png</li>
<li>LSTM Prediction → assets/lstm_prediction.png</li>
</ul>

<hr>

<h2> Time Series Workflow</h2>

<pre>
Load Dataset
    ↓
Data Cleaning
    ↓
EDA
    ↓
Feature Engineering
    ↓
Stationarity Testing
    ↓
Train/Test Split
    ↓
Model Training
    ↓
Hyperparameter Tuning
    ↓
Forecasting
    ↓
Model Evaluation
</pre>

<hr>
<h2> Forecasting Models</h2>

<b>Statistical Models</b>

<ul>
<li>ARIMA</li>
<li>SARIMA</li>
<li>Holt-Winters</li>
<li>Prophet</li>
</ul>

<b>Deep Learning Models</b>

<ul>
<li>Simple RNN</li>
<li>LSTM</li>
<li>GRU</li>
</ul>

<hr>

<h2> Evaluation Metrics</h2>

<ul>
<li>RMSE</li>
<li>MAE</li>
<li>MAPE</li>
<li>R² Score</li>
</ul>

<hr>

<h2>Technologies</h2>

<p>
Python • Pandas • NumPy • Matplotlib • Seaborn • Plotly • Statsmodels • Scikit-learn • TensorFlow • Keras • Prophet
</p>

<hr>

<h2> Key Results</h2>

<ul>
<li>Clear annual seasonality identified.</li>
<li>Long-term decline in emissions observed.</li>
<li>Highest emissions occur during summer.</li>
<li>Coal and Natural Gas remain dominant contributors.</li>
<li>Feature engineering improved forecasting performance.</li>
<li>Statistical and deep learning models were comprehensively compared.</li>
</ul>

<hr>

<h2> Business Insights</h2>

<ul>
<li>Supports long-term energy planning.</li>
<li>Enables seasonal demand forecasting.</li>
<li>Assists decarbonization strategy evaluation.</li>
<li>Improves capacity planning.</li>
<li>Detects abnormal emission spikes.</li>
<li>Identifies high-emission fuel sources for policy development.</li>
</ul>

</div>