import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


def scrape_nasdaq_stock_data(symbol, start_date, end_date):
    url = f"https://www.nasdaq.com/market-activity/stocks/{symbol}/historical"
    params = {
        "date": start_date,
        "endDate": end_date,
    }
    response = requests.get(url, params=params)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'class': 'historical-data__table'})
    data = []
    for row in table.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) == 6:
            date = cols[0].get_text()
            close_price = float(cols[1].get_text().replace(',', ''))
            data.append((date, close_price))
    df = pd.DataFrame(data, columns=['Date', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def prepare_data(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def predict_stock_prices(symbol, start_date, end_date, sequence_length, epochs=50):
    stock_data = scrape_nasdaq_stock_data(symbol, start_date, end_date)

    # Normalize the data using Min-Max Scaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data)

    # Split data into train and test sets
    X, y = prepare_data(scaled_data, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_data=(X_test, y_test))

    # Make predictions on the test set
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate mean absolute percentage error (MAPE)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return predictions, y_test, scaler

# Example usage:
if __name__ == "__main__":
    symbol = "AAPL"  # Replace with the stock symbol you want to predict
    start_date = "2020-01-01" 
    end_date = "2023-07-31" 
    sequence_length = 60
    epochs = 50  

    predictions, actual_values, scaler = predict_stock_prices(symbol, start_date, end_date, sequence_length, epochs)

 
    df_predictions = pd.DataFrame(predictions, columns=['Predicted'], index=stock_data.index[-len(predictions):])
    df_actual = pd.DataFrame(actual_values, columns=['Actual'], index=stock_data.index[-len(actual_values):])

   
    df_predictions = scaler.inverse_transform(df_predictions)
    df_actual = scaler.inverse_transform(df_actual)

   
    df_result = pd.concat([df_actual, df_predictions], axis=1)
    print(df_result)

