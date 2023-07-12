from flask import Flask, render_template,jsonify
from flask_cors import CORS
import ta 
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import datetime
from datetime import timedelta
import pytz
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask_apscheduler import APScheduler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import mysql.connector
sender_email = 'innosta74@gmail.com'
sender_password = 'smqmfpvgoyphnjip'
model=None
model = load_model('year4a.h5')
app = Flask(__name__)
CORS(app)
symbols = ["USDCHF", "USDCAD", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "AUDJPY","GBPJPY","CADJPY","EURJPY","USDJPY","CHFJPY","EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "GBPAUD", "AUDCAD", "AUDCHF", "GBPNZD", "CADCHF", "GBPCAD", "AUDNZD"]

 

def generate_dataframes():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    mt5.login(login = 81043660, password="584Muh12",server="Exness-MT5Trial10") # type: ignore
       
    timezone = pytz.timezone("Europe/Tallinn")
    now = datetime.datetime.now(timezone)
    start = datetime.datetime.now(timezone) - timedelta(days=50)
    utc_from = datetime.datetime(start.year, start.month, start.day)
    utc_to = datetime.datetime(now.year, now.month, now.day, now.hour, now.minute, now.second)

    
    df_list = []
    for symbol in symbols:
        df = pd.DataFrame(mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, utc_from, utc_to))
        df['time'] = pd.to_datetime(df['time'], unit='s')
        #add technical indicators 
        ema = ta.trend.EMAIndicator(close=df['close'], window=18, fillna=False)
        df['EMA'] = ema.ema_indicator()
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=18).rsi()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=24).average_true_range()
        bb = ta.volatility.BollingerBands(df['close'], window=24)
        df['upper_bb'] = bb.bollinger_hband()
        df['lower_bb'] = bb.bollinger_lband()
        # Calculate the Stochastic Oscillator using ta library
        stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=18)
        # Add %K and %D to dataframe
        df['%K'] = stoch.stoch()
        df['%D'] = stoch.stoch_signal()
        # macd
        ema_slow = df['close'].ewm(span=24).mean()
        ema_fast = df['close'].ewm(span=18).mean()

        # Calculate the MACD line
        macd_line = ema_fast - ema_slow

        # Calculate the signal line (EMA of MACD line)
        signal_line = macd_line.ewm(span=12).mean()

        # Calculate the histogram
        histogram = macd_line - signal_line

        # Add the MACD indicators to the DataFrame
        df['macd_line'] = macd_line
        df['signal_line'] = signal_line
        df['histogram'] = histogram
        # Define breakout signals
        df['breakout_above_upper'] = np.where(df['close'] > df['upper_bb'], 1, 0)
        df['breakout_below_lower'] = np.where(df['close'] < df['lower_bb'], 1, 0)
        # Calculate the rolling high, low, and diff values for the past 18 candles
        high = df['high'].rolling(window=48).max()
        low = df['low'].rolling(window=48).min()
        diff = high - low

        # Calculate the Fibonacci retracement levels for the rolling window
        fib_levels = [high - (x / 100) * diff for x in [23.6, 38.2, 50, 61.8, 76.4]]

        # Add the Fibonacci retracement levels to the dataframe
        df['fib_23.6'] = fib_levels[0]
        df['fib_38.2'] = fib_levels[1]
        df['fib_50'] = fib_levels[2]
        df['fib_61.8'] = fib_levels[3]
        df['fib_76.4'] = fib_levels[4]

        # Compute the rolling maximum and minimum prices
        window_size =48
        df['supply_zone'] = df['high'].rolling(window_size).max()
        df['demand_zone'] = df['low'].rolling(window_size).min()
    
        df=df.dropna() 
        df_list.append(df)
    return df_list


dataframes = []
copied_dataframes = []
last_values = []
last_times = []

def extract_data():
    global dataframes, copied_dataframes, last_values, last_times
    dataframes = generate_dataframes()
    copied_dataframes = [df.copy() for df in dataframes]

    tail_dataframes = []
    for copied_df in copied_dataframes:
        tail_df = copied_df.loc[:, ['time', 'close']].tail(1)
        tail_dataframes.append(tail_df)

        last_values = []
        last_times = []
    for tail_df in tail_dataframes:
        last_value = tail_df.iloc[0]['close']
        last_time = tail_df.iloc[0]['time']
        last_values.append(last_value)
        last_times.append(last_time)

    for i, df in enumerate(dataframes):
        df=df.set_axis(df['time'], copy=False)
        df.drop(['open','time','real_volume', 'tick_volume', 'spread'], axis=1, inplace=True)
        dataframes[i] = df

def calculate_predictions():
    global model
    predictions=[]
    N_past = 18

    for i in range(len(dataframes)):
        # create the scaler objects
        sc = MinMaxScaler()
        sc_Y = MinMaxScaler()

        # scale the dataframe
        scaled_data = sc.fit_transform(dataframes[i])
        sc_Y.fit(dataframes[i]['close'].values.reshape(-1,1))

        # extract the last N_past rows from the scaled dataframe
        df_past = scaled_data[-N_past:]
        df_past = df_past.reshape((1, N_past, scaled_data.shape[1]))
        predicted_close = model.predict(df_past)
        predicted_close = sc_Y.inverse_transform(predicted_close)
        predictions.append(np.round(predicted_close,5))
    return predictions

def analyze_market(symbols, dataframes):
    trends = {}
    for pair, df in zip(symbols, dataframes):
        # calculate the 20-period moving average
        ma_20 = np.mean(df['close'].rolling(window=12).mean())

        # calculate the 50-period moving average
        ma_40 = np.mean(df['close'].rolling(window=18).mean())

        # determine the trend based on the relative positions of the current price and the moving averages
        current_price = df['close'].iloc[-1]
        if current_price > ma_20 and current_price > ma_40:
            trend = "uptrend"
        elif current_price < ma_20 and current_price < ma_40:
            trend = "downtrend"
        else:
            trend = "sideways trend"
        
        trends[pair] = trend
    return trends

def calculate_pips():
    differences = []
    predictions = calculate_predictions()
    last_values = []
    last_times = []
    for copied_df in copied_dataframes:
        tail_df = copied_df.loc[:, ['time', 'close']].tail(1)
        last_value = tail_df.iloc[0]['close']
        last_time = tail_df.iloc[0]['time']
        last_values.append(np.round(last_value,5))
        last_times.append(last_time)
    for forecast, last_value, pair in zip(predictions, last_values, symbols):
        if pair in ["USDJPY","EURJPY","CHFJPY","AUDJPY","GBPJPY","CADJPY"]:
            diff = round(( forecast[0][0] - last_value) * 1000, 2)
        else:
            diff = round((forecast[0][0] - last_value) * 100000, 2)
        differences.append(diff)
    return differences, last_values, last_times

def generate_ohlc_charts(symbols, dataframes):
    charts = []

    for pair, df in zip(symbols, dataframes):
        # Convert the DataFrame to the required format for mplfinance
        ohlc_data = df[['time', 'open', 'high', 'low', 'close']]
        ohlc_data.set_index('time', inplace=True)
        ohlc_data.index = pd.to_datetime(ohlc_data.index)

        # Create the chart
        fig, ax = plt.subplots(figsize=(6, 3))

        # Specify colors for bearish and bullish candles
        colors = mpf.make_marketcolors(up='green', down='red')
        style = mpf.make_mpf_style(marketcolors=colors)

        # Plot the candlestick chart with the specified colors
        mpf.plot(ohlc_data, type='candle', ax=ax, warn_too_much_data=len(ohlc_data), style=style)

        # Calculate the moving averages
        ma_20 = df['close'].rolling(window=12).mean()
        ma_50 = df['close'].rolling(window=18).mean()

        # Plot the moving averages
        ax.plot(df.index, ma_20, label='MA 20', color='red')
        ax.plot(df.index, ma_50, label='MA 50', color='yellow')

        # Set the title and labels
        # ax.set_title(f'{pair} Price Trend')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        # ax.legend()

        # Convert the chart to a base64-encoded PNG string using canvas
        canvas = FigureCanvas(fig)
        img = BytesIO()
        canvas.print_png(img)
        chart_data = base64.b64encode(img.getvalue()).decode('utf-8')

        # Close the figure
        plt.close(fig)

        # Add the chart to the list
        charts.append(chart_data)

    # Return the list of charts
    return charts

# Function to fetch email addresses from the database
def get_user_emails():
    conn = mysql.connector.connect(host="localhost", database="forex", user="root", password="")
    cursor = conn.cursor()
    cursor.execute("SELECT email, fname, lname FROM f_clients")
    rows = cursor.fetchall()
    emails = [(row[0], row[1], row[2]) for row in rows]
    cursor.close()
    conn.close()
    return emails

def send_email_notification(sender_email, sender_password, recipient_email, subject, name, trend, previous_close,message):
    # Create a multipart message object
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Create the customized message
    #message = f"Dear {name},\n\nNew forecasts have been generated by FMTAF System.\n\nThe Pair: {subject}'s Current Trend is\n: {trend}\n and its Previous candle closed at: {previous_close}\n\n please Check out for more forex market forecasts with FMTAF System and trade accordingly. We are here to help you defeat the forex market. Stay Safe!"

    # Add the message body
    msg.attach(MIMEText(message, 'plain'))

    # SMTP settings for Gmail
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    try:
        # Create a secure connection with Gmail's SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()

        # Login to your Gmail account
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, recipient_email, msg.as_string())

        print("Notification sent successfully!")

    except Exception as e:
        print(f"Notification failed to {recipient_email} Error: {str(e)}")

    finally:
        # Close the connection to the SMTP server
        server.quit()

@app.route('/muhirefx')
def home():
        with app.app_context():
            return  render_template("index.html")
            #return jsonify(data)

@app.route("/test")
def test():
    with app.app_context():
        symbols = ["USDCHF", "USDCAD", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "GBPAUD", "AUDCAD", "AUDCHF", "GBPNZD", "CADCHF", "GBPCAD", "AUDNZD"]
        # Generate new data from the market
        dataframes = generate_dataframes()
        predicted_values = calculate_predictions()
        pips, last_values, last_times = calculate_pips()
        trends = analyze_market(symbols, dataframes)
        charts = generate_ohlc_charts(symbols, dataframes)

        # Get the recipient emails from the database
        recipient_emails = get_user_emails()

        # Iterate through the recipient emails and send notifications
        for email, first_name, last_name in recipient_emails:
            pair_data_list = []
            name = f"{first_name} {last_name}"
            message = f"Dear {name},\n\nNew forecasts have been generated by FMTAF System.\n\n"
            pair_messages = []
            for i, pair in enumerate(symbols):
                subject = 'FMTAF  Forecasts'
                trend = trends[pair]
                previous_close = last_values[i]
                pair_data = {
                    'pair': pair,
                    'predicted_value': predicted_values[i][0][0].tolist(),
                    'previous_close': last_values[i],
                    'previous_time': last_times[i],
                    'pips': pips[i],
                    'trend': trends[pair],
                    'chart': charts[i]
                }
                pair_data_list.append(pair_data)

                # Generate the message for the current pair
                pair_message = f"The Pair: {pair}'s Current Trend is: {trend}\n and its Previous candle closed at: {previous_close}\n\n"

                # Append the pair message to the list
                pair_messages.append(pair_message)

            # Combine all pair messages into a single string
            pairs_info = "\n".join(pair_messages)

            # Generate the final message
            final_message = "Please check out for more forex market forecasts with FMTAF System and trade accordingly. \n We are glad to help you make informed forex market trading decisions. Stay safe!"

            # Combine the individual parts into the email body
            message += pairs_info + "\n" + final_message

            # Send the email
            send_email_notification(sender_email, sender_password, email, subject, name, trend, previous_close, message)

        data = []
        for i, pair in enumerate(symbols):
            pair_data = {
                'pair': pair,
                'predicted_value': predicted_values[i][0][0].tolist(),
                'previous_close': last_values[i],
                'previous_time': last_times[i],
                'pips': pips[i],
                'trend': trends[pair],
                'chart': charts[i]
            }
            data.append(pair_data)

        for item in data:
            print(item)

        return jsonify(data)

if __name__ == '__main__':
    scheduler = APScheduler()
    scheduler.add_job(id="dataon", func=extract_data, trigger='interval', minutes=2)
    scheduler.add_job(id="prediction", func=test, trigger='interval', minutes=3)


    # Start the scheduler
    scheduler.start()
    app.run(host='localhost', port=4000)
