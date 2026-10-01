# 📈 Stock Price Prediction Using LSTM

A machine learning-based stock price prediction application that uses
**Long Short-Term Memory (LSTM)** neural networks to forecast the next
stock price from historical market data.

The project combines **Python, TensorFlow/Keras, Yahoo Finance,
Scikit-learn, and Streamlit** to provide an interactive stock prediction
dashboard.

---

## 📌 Project Overview

Stock price prediction is a time-series forecasting problem where
historical market data can be used to learn patterns in price movements.

In this project, historical stock data is collected using
**Yahoo Finance**. The closing prices are preprocessed, normalized,
converted into 60-day sequences, and used to train LSTM neural networks.

A separate LSTM model is trained for each supported stock.

The trained models are then integrated into a **Streamlit web application**
where users can select a stock and generate a one-step-ahead prediction.

---

## 🎯 Objectives

The main objectives of this project are:

- Collect historical stock market data.
- Clean and preprocess the collected data.
- Normalize stock prices using MinMaxScaler.
- Create time-series sequences using a 60-day lookback window.
- Build and train LSTM neural networks.
- Evaluate model predictions.
- Save trained models and scalers.
- Build an interactive Streamlit dashboard.
- Deploy the machine learning application.
- Demonstrate an end-to-end machine learning workflow.

---

## 🏢 Supported Stocks

The application currently supports the following NSE-listed companies:

| Company | Yahoo Finance Ticker |
|---|---|
| Reliance Industries | `RELIANCE.NS` |
| Tata Consultancy Services | `TCS.NS` |
| Infosys | `INFY.NS` |
| HDFC Bank | `HDFCBANK.NS` |
| ICICI Bank | `ICICIBANK.NS` |
| State Bank of India | `SBIN.NS` |

Each stock has its own trained LSTM model and corresponding scaler.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Yahoo Finance | Historical stock market data |
| Scikit-learn | Data preprocessing and evaluation |
| TensorFlow | Deep learning framework |
| Keras | LSTM model development |
| Matplotlib | Data visualization |
| Streamlit | Interactive web application |
| Joblib | Saving and loading scalers |

---

# 🧠 Machine Learning Methodology

The project follows the workflow below:

```text
Historical Stock Data
        ↓
Data Collection
        ↓
Data Cleaning
        ↓
Train/Test Split
        ↓
MinMax Scaling
        ↓
60-Day Sequence Creation
        ↓
LSTM Model Training
        ↓
Model Evaluation
        ↓
Model & Scaler Saving
        ↓
Streamlit Application
        ↓
Next Price Prediction
````

---

## 1. Data Collection

Historical stock data is collected using the `yfinance` Python library.

Example:

```python
import yfinance as yf

data = yf.download(
    "RELIANCE.NS",
    start="2015-01-01",
    end="2026-01-01"
)
```

The project primarily uses the **daily closing price** for forecasting.

---

## 2. Data Preprocessing

The collected data is cleaned by removing missing values.

The dataset is then divided chronologically:

```text
80% → Training Data
20% → Testing Data
```

A chronological split is used because stock-price data is time-dependent.

---

## 3. Feature Scaling

Stock prices are normalized using:

```python
MinMaxScaler()
```

The values are transformed into the range:

```text
0 to 1
```

The scaler is fitted only on the training data to reduce the risk of
data leakage.

---

## 4. Time-Series Sequence Creation

The model uses the previous **60 trading days** to predict the next
stock price.

For example:

```text
Day 1 ───────── Day 60
                    ↓
              Predict Day 61
```

The process is repeated throughout the training dataset.

This creates the input sequences required by the LSTM network.

---

# 🧬 LSTM Model Architecture

The project uses a stacked LSTM architecture.

```text
Input
  ↓
LSTM — 50 Units
  ↓
Dropout — 20%
  ↓
LSTM — 50 Units
  ↓
Dropout — 20%
  ↓
Dense — 25 Units
  ↓
Dense — 1 Unit
  ↓
Predicted Price
```

### Model Configuration

| Parameter         | Value              |
| ----------------- | ------------------ |
| Architecture      | Stacked LSTM       |
| LSTM Layers       | 2                  |
| First LSTM Units  | 50                 |
| Second LSTM Units | 50                 |
| Dropout           | 20%                |
| Dense Layer       | 25 units           |
| Output Layer      | 1 unit             |
| Optimizer         | Adam               |
| Loss Function     | Mean Squared Error |
| Batch Size        | 32                 |
| Epochs            | 20                 |
| Lookback Window   | 60 trading days    |

---

# 📊 Model Evaluation

The trained models can be evaluated using standard regression metrics:

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted
prices.

```text
MAE = Average(|Actual - Predicted|)
```

### Root Mean Squared Error — RMSE

Measures prediction error while giving greater weight to larger errors.

```text
RMSE = √Mean((Actual - Predicted)²)
```

### R² Score

Measures how well the model explains the variation in the target values.

```text
R² = 1 - SS_res / SS_tot
```

These metrics are used during model evaluation to understand prediction
performance on the test data.

---

# 💾 Trained Models

A separate model is trained for each supported stock.

### LSTM Models

```text
stock_price_lstm.keras
tcs_price_lstm.keras
infy_price_lstm.keras
hdfcbank_price_lstm.keras
icicibank_price_lstm.keras
sbin_price_lstm.keras
```

### Scalers

```text
stock_scaler.pkl
tcs_scaler.pkl
infy_scaler.pkl
hdfcbank_scaler.pkl
icicibank_scaler.pkl
sbin_scaler.pkl
```

Each model is paired with its corresponding scaler.

This is important because each stock has a different price range and
requires its own preprocessing parameters.

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit dashboard.

The application allows users to:

* Select a supported stock.
* Retrieve recent market data.
* View the current stock price.
* Generate the next-price prediction.
* View the predicted percentage change.
* Visualize historical price trends.
* View the recent 60-day price trend.
* Review model information.

---

# 🎨 Dashboard

The application uses a simple professional interface based on:

* Forest Green
* Sage Green
* Cream / Off-White
* Coral accents

The dashboard is designed to provide a clean and easy-to-understand
machine learning demonstration.

---

# 📂 Project Structure

```text
Stock_Price_Prediction/
│
├── app.py
├── stock_prediction.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── stock_price_lstm.keras
├── stock_scaler.pkl
│
├── tcs_price_lstm.keras
├── tcs_scaler.pkl
│
├── infy_price_lstm.keras
├── infy_scaler.pkl
│
├── hdfcbank_price_lstm.keras
├── hdfcbank_scaler.pkl
│
├── icicibank_price_lstm.keras
├── icicibank_scaler.pkl
│
├── sbin_price_lstm.keras
└── sbin_scaler.pkl
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

Navigate to the project directory:

```bash
cd Stock_Price_Prediction
```

---

## 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
streamlit
yfinance
pandas
numpy
matplotlib
scikit-learn
tensorflow
joblib
```

---

# ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🔮 Prediction Workflow

When a user selects a stock, the application performs the following
steps:

```text
User Selects Stock
        ↓
Load Stock-Specific Model
        ↓
Load Stock-Specific Scaler
        ↓
Download Recent Market Data
        ↓
Extract Closing Prices
        ↓
Select Last 60 Trading Days
        ↓
Scale Input Data
        ↓
Pass Data to LSTM
        ↓
Generate Prediction
        ↓
Inverse Transform Prediction
        ↓
Display Predicted Price
```

---

# 📈 Application Features

### Stock Selection

Users can select from the six supported stocks.

### Current Price

Displays the latest available closing price retrieved from Yahoo Finance.

### Predicted Price

Displays the price predicted by the corresponding LSTM model.

### Predicted Change

Displays the numerical and percentage difference between the latest
price and model prediction.

### Historical Chart

Displays the historical stock-price trend.

### Recent Price Chart

Displays the most recent 60 trading days together with the model's
predicted next price.

---

# 🔐 Data and Model Considerations

Each stock uses a separate:

```text
LSTM Model
+
MinMaxScaler
```

This prevents a model trained on one company's price history from being
incorrectly applied to another company's price range.

The application therefore does not treat the six stocks as interchangeable
inputs to a single model.

---

# ⚠️ Limitations

This project is primarily an educational machine-learning and
time-series forecasting project.

Stock prices can be affected by many factors that are not included in
this model, including:

* Company announcements
* Market conditions
* Economic indicators
* Interest rates
* News events
* Trading volume
* Investor sentiment
* Global events

The current implementation primarily uses historical closing prices,
so it does not incorporate these additional variables.

---

# 🚀 Future Improvements

Potential improvements include:

* Add technical indicators such as SMA, EMA, RSI, and MACD.
* Include trading volume as an additional feature.
* Incorporate multiple market features.
* Experiment with GRU architectures.
* Experiment with Transformer-based time-series models.
* Perform hyperparameter tuning.
* Implement walk-forward validation.
* Compare multiple machine learning models.
* Add model performance comparison to the dashboard.
* Add more stocks.
* Implement automated model retraining.
* Improve error handling and logging.
* Add interactive date-range selection.

---

# 🧪 Example Machine Learning Pipeline

```python
# Scale data
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(
    train_data[["Close"]]
)

# Create sequences
for i in range(60, len(train_scaled)):

    X_train.append(
        train_scaled[i-60:i, 0]
    )

    y_train.append(
        train_scaled[i, 0]
    )
```

The resulting sequences are reshaped into the format required by the
LSTM:

```text
(samples, 60, 1)
```

---

# 🌐 Deployment

The application can be deployed using **Streamlit Community Cloud**.

Deployment workflow:

```text
GitHub Repository
        ↓
requirements.txt
        ↓
Streamlit Community Cloud
        ↓
Install Dependencies
        ↓
Load Models
        ↓
Run app.py
        ↓
Public Web Application
```

---

# 💡 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Time-series data preprocessing
* Data normalization
* Train/test splitting
* Sequence generation
* LSTM neural networks
* Deep learning model training
* Regression evaluation metrics
* Model serialization
* Streamlit application development
* Interactive data visualization
* Git and GitHub
* Machine learning deployment

---

# 📌 Disclaimer

This project is created for **educational and demonstration purposes**.

The predictions generated by the application are based on historical
data and the trained machine-learning models. They should not be
interpreted as guaranteed future prices or as financial or trading
advice.

---

# 👩‍💻 Author

## Bhumika GS

**AI / ML | Data Science | Data Analytics**

---

# ⭐ Acknowledgements

* Yahoo Finance for historical market data
* TensorFlow / Keras for deep learning tools
* Scikit-learn for preprocessing and evaluation
* Streamlit for application development
* Python open-source community

---
