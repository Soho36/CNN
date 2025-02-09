import pandas as pd
import mplfinance as mpf

# Load OHLC data (example)
path = "h1.csv"
df_csv = pd.read_csv(path, parse_dates=True, delimiter='\t')

new_column_names = {
    '<DATE>': 'Date',
    '<TIME>': 'Time',
    '<OPEN>': 'Open',
    '<HIGH>': 'High',
    '<LOW>': 'Low',
    '<CLOSE>': 'Close'
}

df_csv = df_csv.rename(columns=new_column_names)
df_csv['Datetime'] = pd.to_datetime(df_csv['Date'] + ' ' + df_csv['Time'])

# Filter by date range
start_date = '2025-01-01 00:00:00'
end_date = '2025-01-01 23:00:00'
df_filtered_by_date = df_csv[(df_csv['Datetime'] >= start_date) & (df_csv['Datetime'] <= end_date)]
df_filtered_by_date.set_index('Datetime', inplace=True)

# Define buy/sell signals
signals = {
    "buy": [
        "2025-01-01 11:00:00", "2025-01-01 18:00:00"
    ],
    "sell": [
        "2025-01-01 04:00:00", "2025-01-01 07:00:00"
    ]
}
signals = {k: pd.to_datetime(v) for k, v in signals.items()}

# Align signals with DataFrame index
aligned_buy = df_filtered_by_date.index.intersection(signals["buy"])
aligned_sell = df_filtered_by_date.index.intersection(signals["sell"])

print('aligned_buy', aligned_buy)
print('aligned_sell', aligned_sell)

# Create full-length series for mplfinance
buy_prices = pd.Series(index=df_filtered_by_date.index, dtype=float)
sell_prices = pd.Series(index=df_filtered_by_date.index, dtype=float)

# Assign only to valid buy/sell points
buy_prices.loc[aligned_buy] = df_filtered_by_date.loc[aligned_buy, "High"]
sell_prices.loc[aligned_sell] = df_filtered_by_date.loc[aligned_sell, "Low"]

# Print to verify
print("buy_prices", buy_prices.dropna())  # Drop NaN for cleaner output
print("sell_prices", sell_prices.dropna())  # Drop NaN for cleaner output

# Add buy/sell markers
add_plot = [
    mpf.make_addplot(buy_prices - 0.02, type='scatter', markersize=100, color='green', marker='^'),
    mpf.make_addplot(sell_prices + 0.02, type='scatter', markersize=100, color='red', marker='v')
]

# Create a custom style for minimalist plotting
custom_style = mpf.make_mpf_style(
    base_mpf_style='classic',  # Use a dark theme as the base
    gridcolor='white',                # Remove grid lines
    facecolor='black',             # Set the background to black
    y_on_right=False               # Optional: Keep or remove y-axis
)

# Plot the chart
mpf.plot(
    df_filtered_by_date,
    type="candle",
    addplot=add_plot,
    style=custom_style,
    axisoff=True,  # Turn off the axes completely
    savefig="chart.png"
)
mpf.plot(df_filtered_by_date, type="candle", addplot=add_plot)

