import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
import requests
from datetime import datetime 




st.title("Agentic AI - Indian Stock Predictor")


symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS, TCS.NS)", "IOC.NS")
option = st.selectbox("Select Prediction Range", ("1 Day", "5 Days", "15 Days"))
model_choice = st.selectbox("Select Model", ["Random Forest", "Linear Regression"])
if model_choice == "Random Forest":
    model = RandomForestRegressor()
else:
    model = LinearRegression()

# Map prediction range
range_map = {
    "1 Day": 1,
    "5 Days": 5,
    "15 Days": 15,
}
future_days = range_map[option]

# Fetch data
try:
    stock = yf.Ticker(symbol)
    data = stock.history(period="max")
    if data.empty:
        st.warning("No data found. Please check the stock symbol.")
        st.stop()
    data = data[-(future_days + 65):]  # Keep enough data
    data['MA10'] = data['Close'].rolling(window=10).mean()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Features
features = data[['Close', 'Volume', 'Open', 'High', 'Low']].values

X = []
y = []

for i in range(5, len(features) - future_days):
    X.append(features[i-5:i].flatten())
    y.append(data['Close'].values[i:i+future_days])

X = np.array(X)
y = np.array(y)
if y.ndim == 1:
    y = y.reshape(-1, 1)

# Check data length
if len(X) == 0 or len(y) == 0:
    st.warning("Not enough data to make predictions.")
    st.stop()

# Train model
# Train model
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = LinearRegression()
model.fit(X, y)



# Predict
last_5 = features[-5:].flatten().reshape(1, -1)
predicted_price = model.predict(last_5)[0]
trained_prediction = model.predict(X)
mae = mean_absolute_error(y, trained_prediction)
current_price = data['Close'].values[-1]

# Display metrics
if hasattr(predicted_price, "__len__"):
    predicted_value = predicted_price[0]
else:
    predicted_value = predicted_price
st.metric("Predicted Next Price: ₹", round(predicted_value, 2))
action_value = predicted_value

st.metric("Current Price: ₹", round(current_price, 2))
st.metric("Mean Absolute Error: ₹", round(mae, 2))

# Action suggestion
if action_value > current_price:
    st.success("Action: Buy")
elif action_value < current_price:
    st.error("Action: Sell")
else:
    st.info("Action: Hold")

# Future Prices
st.subheader("Predicted Future Prices")
if future_days == 1:
    st.write(f"Day 1: ₹{round(predicted_price, 2)}")
else:
    for i, price in enumerate(predicted_price, 1):
        st.write(f"Day {i}: ₹{round(price, 2)}")

st.subheader("Actual vs Predicted Closing Prices")
fig, ax = plt.subplots(figsize=(10, 4))

# Ensure 1D for plotting
actual = y.ravel()
predicted = trained_prediction.ravel()

ax.plot(actual, label="Actual Prices", color='blue')
ax.plot(predicted, label="Predicted Prices", color='orange')

ma10 = data["MA10"].values[5:5 + len(actual)]
ax.plot(ma10, label='MA10', color='green')

ax.set_title("Actual Price vs Predicted Price")
ax.set_xlabel("Days")
ax.set_ylabel("Prices (INR)")
ax.legend()
st.pyplot(fig)




def fetch_news(symbol):
    api_key = "d13f3f1r01qs7glh6tagd13f3f1r01qs7glh6tb0"  
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-06-01&to=2024-06-06&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

def format_datetime(unix_timestamp):
    return datetime.utcfromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S')


# In your Streamlit UI
st.subheader("Latest News")
news_items = fetch_news(symbol.split('.')[0])  
if news_items:
    for n in news_items[:5]:
        readble_time = format_datetime(n["datetime"])
        st.markdown(f"- [{n['headline']}]({n['url']}) ({n['datetime']})\n*Published on:*{readable_time}")
else:
    st.info("No news available for this stock.")


st.subheader("Volume and Price Volatility")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data['Volume'], label='Volume', color='purple')
ax.set_ylabel("Volume")
ax2 = ax.twinx()
ax2.plot(data['Close'].rolling(window=5).std(), label='Volatility', color='red')
ax2.set_ylabel("Volatility (Std Dev)")
fig.legend()
st.pyplot(fig)


if future_days > 1:
    future_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(len(predicted_price))],
        "Predicted Price (₹)": predicted_price
    })

    st.subheader("Predicted Price Table")
    st.dataframe(future_df)



    csv = future_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Future Predictions",
        data=csv,
        file_name=f'{symbol}_future_predictions.csv',
        mime='text/csv'
    )

    
    
