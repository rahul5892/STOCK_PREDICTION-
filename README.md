# Stock Prediction and Visualization Application

## Overview

This is a **Streamlit-based stock prediction application** that uses **Linear Regression** to forecast stock prices for Nifty indices and their constituent companies. It provides an interactive UI to fetch data, predict stock prices for the next 25 days, and visualize historical and predicted prices.

---

## Features

- **Nifty Index Selection**: Choose from Nifty 50, Nifty 100, or Bank Nifty indices.
- **Stock Data Fetching**: Retrieve historical data for any stock in the selected index.
- **Stock Price Prediction**: Predict stock prices for the next 25 days using a trained Linear Regression model.
- **Visualization**: Display historical stock prices alongside predicted prices.
- **News Integration**: Fetch the latest company-specific news for informed decision-making.

---

## Technologies Used

- **Frontend**: Streamlit
- **Data Visualization**: Matplotlib
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Linear Regression)
- **API Integration**: Yahoo Finance API for stock data
- **News Fetching**: Integrated news feature for company-specific updates

---

## How to Run the Project

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/<your-repo>/stock-prediction.git
   cd stock-prediction
   ```

2. **Install Dependencies**:  
   Ensure Python is installed, then install required libraries using pip:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:  
   Start the Streamlit app with the command:  
   ```bash
   streamlit run app.py
   ```

4. **Access the App**:  
   Open the local URL displayed in the terminal (default: `http://localhost:8501`) to interact with the application.

---

## Folder Structure

```
.
├── app.py                 # Main application script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation

```

---

## Key Functionalities

### 1. **Stock Selection**
   - Select an index (e.g., Nifty 50) and a stock from the dropdown menu.

### 2. **Data Fetching and Prediction**
   - Retrieve stock data using the Yahoo Finance API.
   - Train a Linear Regression model using historical stock prices.
   - Predict the stock's closing prices for the next 25 days.

### 3. **Visualization**
   - Interactive plot of historical and predicted stock prices.

### 4. **News Display**
   - Fetch and display recent company-specific news articles.

---

## Example Visualization

- **Stock Price Graph**: Historical vs. Predicted prices for the last 2 years.
- **Predicted Data**: Display actual vs. predicted stock prices for the next 25 days.

---

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for any enhancements or bug fixes.

---

## License

This project is open-source and available under the MIT License.  
