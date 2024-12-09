import streamlit as st
import requests
from datetime import datetime, timedelta
import wordcloud
from wordcloud import WordCloud

from textblob import TextBlob
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

FINNHUB_API_KEY = "csuvn39r01qglf5p26a0csuvn39r01qglf5p26ag"


css=""" 
<style>
/* General Styles */
body {
    background: linear-gradient(135deg, #000, #111);
    font-family: 'Roboto', sans-serif;
    color: #fff;
    overflow-x: hidden;
}

/* Header */
header {
    width: 100%;
    position: fixed;
    top: 0;
    left: 0;
    background: linear-gradient(45deg, #0f0, #00f);
    animation: morph 10s infinite alternate;
    display: flex;
    justify-content: space-between;
    padding: 10px 20px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    z-index: 1000;
}

@keyframes morph {
    0% {border-radius: 0;}
    100% {border-radius: 50%;}
}

header nav a {
    color: #fff;
    text-decoration: none;
    padding: 15px;
    transition: color 0.3s ease, transform 0.3s ease;
}

header nav a:hover {
    color: #ff0;
    text-shadow: 0 0 5px #fff;
    transform: scale(1.1);
}

/* Hero Section */
.hero {
    position: relative;
    width: 100%;
    height: 100vh;
    background: url('path_to_your_video.mp4') no-repeat center center/cover;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 1;
}

.hero .card {
    position: relative;
    z-index: 2;
    background: rgba(0, 0, 0, 0.7);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    animation: float 6s ease-in-out infinite;
}

@keyframes float {
    0%, 100% {transform: translateY(0);}
    50% {transform: translateY(-20px);}
}

.hero .card h1 {
    font-size: 3rem;
    color: #fff;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.7);
}

.hero .card .cta {
    display: inline-block;
    padding: 10px 20px;
    margin-top: 20px;
    background: linear-gradient(45deg, #0ff, #0f0);
    border-radius: 5px;
    text-decoration: none;
    color: #000;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% {box-shadow: 0 0 0 0 rgba(0, 255, 255, 0.7);}
    70% {box-shadow: 0 0 0 20px rgba(0, 255, 255, 0);}
    100% {box-shadow: 0 0 0 0 rgba(0, 255, 255, 0);}
}

/* Content Cards */
.card {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 20px;
    margin: 20px;
    transition: transform 0.3s, background 0.3s;
    backdrop-filter: blur(10px);
}

.card:hover {
    transform: translateY(-10px);
    background: rgba(255, 255, 255, 0.3);
}

/* Footer */
footer {
    background: #000;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

footer::before {
    content: "";
    position: absolute;
    top: 0;
    left: 50%;
    width: 150%;
    height: 100%;
    background: url('path_to_starry_sky_image.jpg') no-repeat center center/cover;
    opacity: 0.3;
    transform: translateX(-50%);
}

footer .social-icons a {
    color: #fff;
    margin: 0 10px;
    font-size: 1.5rem;
    transition: transform 0.3s, color 0.3s;
}

footer .social-icons a:hover {
    color: #0ff;
    transform: scale(1.2);
}

/* Buttons */
button {
    background: linear-gradient(45deg, #ff0080, #ff00ff);
    color: #fff;
    border: none;
    padding: 10px 20px;
    cursor: pointer;
    border-radius: 5px;
    transition: all 0.3s;
}

button:active {
    animation: ripple 0.6s;
}

@keyframes ripple {
    0% {box-shadow: 0 0 0 0 rgba(255, 0, 255, 0.6);}
    100% {box-shadow: 0 0 0 40px rgba(255, 0, 255, 0);}
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', sans-serif;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.7);
}


</style>
"""

# Link to external CSS file
st.markdown(css, unsafe_allow_html=True)
# Load Nifty stocks dictionary
def get_nifty_stocks():
    nifty_stocks = {
        'Nifty 50': [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'HDFC.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'LT.NS',
            'BAJFINANCE.NS', 'HCLTECH.NS', 'TECHM.NS', 'AXISBANK.NS',
            'MARUTI.NS', 'SUNPHARMA.NS', 'WIPRO.NS', 'NTPC.NS',
            'ADANIGREEN.NS', 'ULTRACEMCO.NS', 'ONGC.NS', 'JSWSTEEL.NS',
            'CIPLA.NS', 'TITAN.NS', 'SHREECEM.NS', 'M&M.NS',
            'HINDALCO.NS', 'RECLTD.NS', 'COALINDIA.NS', 'BPCL.NS',
            'INDUSINDBK.NS', 'GRASIM.NS', 'EICHERMOT.NS', 'DIVISLAB.NS',
            'TATAMOTORS.NS', 'PVR.NS', 'SIEMENS.NS'
        ],
        'Nifty 100': [
            'ADANIGREEN.NS', 'ADANIPORTS.NS', 'AIAENG.NS', 'AMBUJACEM.NS',
            'APOLLOTYRE.NS', 'ASHOKLEY.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
            'BANDHANBNK.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'BHEL.NS',
            'BHARTIARTL.NS', 'BOSCHLTD.NS', 'BRITANNIA.NS', 'CANBK.NS',
            'CENTRALBK.NS', 'CHOLAFIN.NS', 'CUMMINSIND.NS', 'DABUR.NS',
            'DALBHARAT.NS', 'DLF.NS', 'EXIDEIND.NS', 'FEDERALBNK.NS',
            'GAIL.NS', 'GLAND.NS', 'GODREJCP.NS', 'GODREJIND.NS',
            'GSKCONS.NS', 'HAVELLS.NS', 'HEG.NS', 'HINDPETRO.NS',
            'HINDZINC.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'IDBI.NS',
            'IDFCFIRSTB.NS', 'IGL.NS', 'INDUSTOWER.NS', 'INFRATEL.NS',
            'INFY.NS', 'JINDALSTEL.NS', 'JUBLFOOD.NS', 'KARURVYSYA.NS',
            'LICHSGFIN.NS', 'M&MFIN.NS', 'MANAPPURAM.NS', 'MARICO.NS',
            'MCDOWELL-N.NS', 'MINDTREE.NS', 'NESTLEIND.NS', 'OIL.NS',
            'PERSISTENT.NS', 'PIDILITIND.NS', 'PIIND.NS', 'PNB.NS',
            'POLYMED.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHREECEM.NS',
            'SOUTHBANK.NS', 'SRF.NS', 'SYNGENE.NS', 'TATACHEM.NS',
            'TATAMOTORS.NS', 'TATAPOWER.NS', 'TCS.NS', 'TECHM.NS',
            'TORNTPOWER.NS', 'TRENT.NS', 'UBL.NS', 'WIPRO.NS'
        ],
        'Bank Nifty': [
            'AXISBANK.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS',
            'SBIN.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'CANBK.NS',
            'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'INDUSINDBK.NS', 'PNB.NS'
        ]
    }
    return nifty_stocks


# Function to fetch stock data
def fetch_stock_data(ticker):
    return yf.download(ticker, period='max')

# Function to preprocess data
def preprocess_data(data):
    data['Prediction'] = data['Close'].shift(-25)
    X = data[['Close']].values[:-25]
    y = data['Prediction'].values[:-25]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

# Function to train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to make predictions
def make_predictions(model, data):
    X_future = data[['Close']].values[-25:]
    return model.predict(X_future)

# Streamlit app layout
st.title("Stock Prediction App 📈")
st.subheader("Predict Nifty Stocks with Linear Regression")

# Nifty Index Selection
st.sidebar.header("Choose an Index and Stock")
nifty_stocks = get_nifty_stocks()
index_choice = st.sidebar.selectbox("Select Nifty Index", list(nifty_stocks.keys()))

# Stock Selection based on Index
selected_stock = st.sidebar.selectbox("Select Stock", nifty_stocks[index_choice])

# Display selected stock info
st.write(f"**Selected Stock**: {selected_stock}")

# Fetch stock data
if st.sidebar.button("Fetch Data and Predict"):
    with st.spinner("Fetching stock data and making predictions..."):
        stock_data = fetch_stock_data(selected_stock)
        
        # Preprocess and Train Model
        X_train, X_test, y_train, y_test = preprocess_data(stock_data)
        model = train_model(X_train, y_train)
        predictions = make_predictions(model, stock_data)
        
        # Plot Predictions
        st.subheader("Stock Price Prediction for Next 25 Days")
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot historical and predicted prices
        two_years_data = stock_data[-2*252:]  # Last 2 years
        ax.plot(two_years_data.index, two_years_data['Close'], label='Historical Prices', color='blue')
        
        valid = two_years_data[['Close']].copy()
        valid['Prediction'] = np.nan
        valid.iloc[-25:, valid.columns.get_loc('Prediction')] = predictions
        ax.plot(valid.index, valid['Prediction'], label='Predicted Prices', color='red', linestyle='--')
        
        # Customize plot
        ax.set_title("Historical and Predicted Stock Prices", fontsize=20)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Price (₹)", fontsize=14)
        ax.legend()
        st.pyplot(fig)
        
        # Display last 10 rows of stock data in a table
        st.subheader("Last 10 Rows of Historical Data")
        st.write(stock_data.tail(10))
        
        # Display last 25 days: Actual vs Predicted Prices
        st.subheader("Actual vs Predicted Close Prices for Last 25 Days")
        actual = stock_data['Close'].values[-25:]
        dates = stock_data.index[-25:]
        prediction_df = pd.DataFrame({
            'Date': dates,
            'Actual Close Price (₹)': actual,
            'Predicted Close Price (₹)': predictions
        })
        st.write(prediction_df)
        
        # Display Model Accuracy
        score = model.score(X_test, y_test)
        st.write(f"**Model Accuracy:** {score * 100:.2f}%")


# -------------------------------------

st.header(selected_stock)
selected_stock = selected_stock.replace(".NS", "")

# Function to fetch recent company-specific news

# Function to fetch recent company-specific news
def fetch_stock_news(selected_stock):
    stock = yf.Ticker(selected_stock)
    news = stock.news  # Fetch the latest news
    return news[:5]  # Return the latest 5 news articles

# Display Stock Overview and Basic Information
def display_stock_info(selected_stock):
    stock = yf.Ticker(selected_stock)
    stock_info = stock.info
    st.subheader("Stock Overview")
    st.write(f"**Company Name**: {stock_info.get('longName', 'Not available')}")
    st.write(f"**Current Price**: ₹{stock_info.get('currentPrice', 'Not available')}")
    st.write(f"**Market Cap**: ₹{stock_info.get('marketCap', 'Not available')}")
    st.write(f"**PE Ratio**: {stock_info.get('trailingPE', 'Not available')}")
    st.write(f"**52 Week High**: ₹{stock_info.get('fiftyTwoWeekHigh', 'Not available')}")
    st.write(f"**52 Week Low**: ₹{stock_info.get('fiftyTwoWeekLow', 'Not available')}")
    
# Display Sentiment Analysis
def display_sentiment_analysis(news):
    sentiments = []
    for article in news:
        headline = article["title"]
        blob = TextBlob(headline)
        sentiment = "Positive" if blob.sentiment.polarity > 0 else \
                    "Negative" if blob.sentiment.polarity < 0 else "Neutral"
        
        sentiments.append({
            "Headline": headline,
            "Sentiment": sentiment
        })
    
    sentiment_df = pd.DataFrame(sentiments)
    st.write("Sentiment analysis of recent news:")
    st.dataframe(sentiment_df)

# # Word Cloud for sentiment visualization
# def generate_wordcloud(news):
#     all_headlines = " ".join([article["title"] for article in news])
#     wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_headlines)

#     st.subheader("Word Cloud for News Headlines")
#     plt.figure(figsize=(8, 6))
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
#     st.pyplot()

# Display the news headlines as clickable cards
def display_news_cards(news):
    for article in news:
        headline = article["title"]
        url = article["link"]
        sentiment = "Positive" if TextBlob(headline).sentiment.polarity > 0 else \
                    "Negative" if TextBlob(headline).sentiment.polarity < 0 else "Neutral"
        
        with st.expander(f"{headline} - {sentiment}"):
            st.write(f"**Source**: [Read more]({url})")

# Streamlit UI components
st.header("Sentiment Analysis for Stock News")

# Select a stock
selected_stock = st.sidebar.text_input("Enter Stock Symbol", "EICHERMOT.NS").upper()

# Display stock info
display_stock_info(selected_stock)

# Fetch news for the selected stock
news = fetch_stock_news(selected_stock)

if news:
    # Display sentiment analysis results
    display_sentiment_analysis(news)
    
    # Display news headlines as clickable cards
    display_news_cards(news)
    
    # Display word cloud
    # generate_wordcloud(news)
else:
    st.write("No recent news available for sentiment analysis.")