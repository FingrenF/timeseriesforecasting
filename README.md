# GRU Stock Price Prediction with Fruit Fly Optimization Algorithm (FOA)

This repository contains my undergraduate thesis project on predicting the **LQ45 stock index** using a **Gated Recurrent Unit (GRU)** model combined with the **Fruit Fly Optimization Algorithm (FOA)** for hyperparameter tuning.  
The project compares the performance of GRU models trained using different optimizers — **RMSProp**, **Adam**, and **FOA** — to analyze their effect on prediction accuracy and generalization.

---

## 🧠 Background
Stock market data is highly volatile and nonlinear, making accurate prediction challenging.  
Recurrent Neural Network (RNN) architectures such as GRU are capable of capturing temporal dependencies in time-series data, making them suitable for stock price forecasting.  
To enhance performance, the **Fruit Fly Optimization Algorithm (FOA)** is applied to tune GRU hyperparameters automatically.

---

## ⚙️ Methodology
1. **Data Source:**  
   - LQ45 stock index data obtained via [Yahoo Finance](https://finance.yahoo.com) using the `yfinance` library  
   - Total data: ~6,000 daily records with price range between 600 – 1100 IDR  

2. **Preprocessing:**  
   - Feature used: *Closing Price*  
   - Normalization using **MinMaxScaler**  
   - Converted into sequential data for GRU input  

3. **Model Development:**  
   - Implemented GRU using TensorFlow/Keras  
   - Compared optimizers:
     - `RMSProp` (baseline)  
     - `Adam` (adaptive optimizer)  
     - `FOA` (metaheuristic optimization)  
   - FOA searches for best parameters including:
     - Number of units  
     - Learning rate  
     - Dropout rate  
     - Activation function  
     - Batch size  
     - Epochs  
     - Number of layers  

4. **Evaluation Metric:**  
   - Mean Squared Error (MSE)  
   - Root Mean Squared Error (RMSE)  
   - Mean Absolute Percentage Error (MAPE)  

---

## 📊 Results
- FOA-tuned GRU achieved **up to 98.9% prediction accuracy** with improved generalization over RMSProp and Adam baselines.  
- Analysis showed that optimizer selection and hyperparameter tuning have significant impact on model convergence and stability.
