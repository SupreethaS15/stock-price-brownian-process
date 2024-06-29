import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy.stats import shapiro

def get_live_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def normality_test(returns):
    stat, p_value = shapiro(returns)
    return p_value

def calculate_mape(actual, predicted):
    actual, predicted = actual.align(predicted, join='inner')
    mask = (actual != 0) & (~actual.isna())
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted series must have the same length after filtering")

    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mape

def calculate_confidence_intervals(price_paths, confidence_level=0.95):
    lower_bound = np.percentile(price_paths, (1 - confidence_level) / 2 * 100, axis=1)
    upper_bound = np.percentile(price_paths, (1 + confidence_level) / 2 * 100, axis=1)
    return lower_bound, upper_bound

def predict_stock_prices(ticker_symbol, start_date, days_to_predict):
    df = get_live_data(ticker_symbol, start_date, pd.Timestamp.now().strftime('%Y-%m-%d'))
    df = df[['Close']]
    df['Returns'] = df['Close'].pct_change().dropna()

    p_value = normality_test(df['Returns'])
    normality_message = "Returns are normally distributed (p >= 0.05)" if p_value >= 0.05 else "Returns are not normally distributed (p < 0.05)"
    print(normality_message)

    mu = df['Returns'].mean()
    sigma = df['Returns'].std()

    S0 = df['Close'].iloc[-1]
    T = days_to_predict
    dt = 1
    N = int(T / dt)
    num_simulations = 1000
    price_paths = np.zeros((N, num_simulations))
    price_paths[0] = S0

    for t in range(1, N):
        Z = np.random.standard_normal(num_simulations)
        price_paths[t] = price_paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    predicted_prices_df = pd.DataFrame(price_paths.mean(axis=1), columns=['Close'])
    predicted_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=N)
    predicted_prices_df['Date'] = predicted_dates
    predicted_prices_df.set_index('Date', inplace=True)

    lower_bound, upper_bound = calculate_confidence_intervals(price_paths)
    predicted_prices_df['Lower Bound'] = lower_bound
    predicted_prices_df['Upper Bound'] = upper_bound

    return df, predicted_prices_df, normality_message

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction")
        self.window_width = root.winfo_screenwidth()
        self.window_height = root.winfo_screenheight()
        self.root.attributes('-fullscreen', True)
        self.root.bind("<Escape>", lambda e: self.root.quit())

        self.bg_image = Image.open("C:/COLLEGE/FIN lab/bg.jpg")
        self.bg_image = self.bg_image.resize((self.window_width, self.window_height))
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        self.logo_image = Image.open("C:/COLLEGE/FIN lab/logo.png")
        self.logo_image_resized = self.logo_image.resize((100, 100), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(self.logo_image_resized)

        self.image1 = Image.open("C:/COLLEGE/FIN lab/lock and key.jpeg.jpg")
        self.image1_resized = self.image1.resize((200, 200), Image.LANCZOS)
        self.image1_photo = ImageTk.PhotoImage(self.image1_resized)

        self.image2 = Image.open("C:/COLLEGE/FIN lab/money.jpeg.jpg")
        self.image2_resized = self.image2.resize((200, 200), Image.LANCZOS)
        self.image2_photo = ImageTk.PhotoImage(self.image2_resized)

        self.image3 = Image.open("C:/COLLEGE/FIN lab/heeramandi.jpg")
        self.image3_resized = self.image3.resize((200, 200), Image.LANCZOS)
        self.image3_photo = ImageTk.PhotoImage(self.image3_resized)

        self.image4 = Image.open("C:/COLLEGE/FIN lab/stranger things.jpeg.jpg")
        self.image4_resized = self.image4.resize((200, 200), Image.LANCZOS)
        self.image4_photo = ImageTk.PhotoImage(self.image4_resized)

        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
        self.logo_id = self.canvas.create_image(self.window_width // 2, 50, image=self.logo_photo, anchor="center")

        self.canvas.create_image(100, 100, image=self.image1_photo, anchor="nw")
        self.canvas.create_image(self.window_width - 100, 100, image=self.image2_photo, anchor="ne")
        self.canvas.create_image(100, self.window_height - 100, image=self.image3_photo, anchor="sw")
        self.canvas.create_image(self.window_width - 100, self.window_height - 100, image=self.image4_photo, anchor="se")

        self.main_frame = ttk.Frame(root, padding="10", style="MainFrame.TFrame")
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.prediction_label = ttk.Label(self.main_frame, text="Stock Price Prediction using Brownian Motion", font=("Helvetica", 18))
        self.prediction_label.pack(pady=20)

        self.predict_button = ttk.Button(self.main_frame, text="Predict" , command=self.ask_days_to_predict, style="TButton")
        self.predict_button.pack(pady=20)

        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 16), padding=10, background="black", foreground="black")
        self.style.configure("MainFrame.TFrame", background="")

    def ask_days_to_predict(self):
        self.days_to_predict = simpledialog.askinteger("Input", "Enter the number of days to predict:")
        if self.days_to_predict is not None:
            self.show_prediction_options()

    def show_prediction_options(self):
        self.clear_frame(self.main_frame)
        
        self.prediction_button = ttk.Button(self.main_frame, text="Show Predicted Values", command=self.show_predicted_values, style="TButton")
        self.prediction_button.pack(pady=20)

        self.graph_button = ttk.Button(self.main_frame, text="Show Predicted Graph", command=self.show_predicted_graph, style="TButton")
        self.graph_button.pack(pady=20)

        self.bollinger_button = ttk.Button(self.main_frame, text="Bollinger Bands", command=self.show_bollinger_bands, style="TButton")
        self.bollinger_button.pack(pady=20)
        
        self.back_button = ttk.Button(self.main_frame, text="Back", command=self.go_back_to_main, style="TButton")
        self.back_button.pack(pady=20)

    def show_predicted_values(self):
        self.clear_frame(self.main_frame)
        ticker_symbol = 'NFLX'
        start_date = '2024-03-01'
        actual_prices_df, predicted_prices_df, normality_message = predict_stock_prices(ticker_symbol, start_date, self.days_to_predict)

        text = tk.Text(self.main_frame, wrap="word", width=100, height=20)
        text.insert(tk.END, normality_message + "\n\n")
        text.insert(tk.END, predicted_prices_df.to_string())
        text.pack(pady=20)

        predicted_prices_df.to_excel("predicted_prices_with_chart.xlsx", sheet_name="Predicted Prices", index=True)

        wb = Workbook()
        ws = wb.active

        for row in dataframe_to_rows(predicted_prices_df, index=True, header=True):
            ws.append(row)

        chart = LineChart()
        chart.title = "Predicted Prices"
        chart.x_axis.title = "Date"
        chart.y_axis.title = "Price"
        data = Reference(ws, min_col=2, min_row=1, max_col=2, max_row=len(predicted_prices_df)+1)
        categories = Reference(ws, min_col=1, min_row=2, max_row=len(predicted_prices_df)+1)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(categories)
        ws.add_chart(chart, "E5")

        wb.save("predicted_prices_with_chart.xlsx")

        self.back_button = ttk.Button(self.main_frame, text="Back", command=self.show_prediction_options, style="TButton")
        self.back_button.pack(pady=20)

    def show_predicted_graph(self):
        self.clear_frame(self.main_frame)
        ticker_symbol = 'NFLX'
        start_date = '2024-03-01'
        actual_prices_df, predicted_prices_df, normality_message = predict_stock_prices(ticker_symbol, start_date, self.days_to_predict)

        fig, ax = plt.subplots()
        actual_prices_df['Close'].plot(ax=ax, label='Actual Prices', color='blue')
        predicted_prices_df['Close'].plot(ax=ax, label='Predicted Prices', color='orange')
        ax.fill_between(predicted_prices_df.index, predicted_prices_df['Lower Bound'], predicted_prices_df['Upper Bound'], color='orange', alpha=0.2)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=20)

        actual = actual_prices_df['Close'][-len(predicted_prices_df):]
        predicted = predicted_prices_df['Close']

        try:
            mape = calculate_mape(actual, predicted)
            messagebox.showinfo("MAPE", f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

        self.back_button = ttk.Button(self.main_frame, text="Back", command=self.show_prediction_options, style="TButton")
        self.back_button.pack(pady=20)

    def show_bollinger_bands(self):
        self.clear_frame(self.main_frame)
        ticker_symbol = 'NFLX'
        start_date = '2024-03-01'
        df = get_live_data(ticker_symbol, start_date, pd.Timestamp.now().strftime('%Y-%m-%d'))
        df['SMA'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper Band'] = df['SMA'] + (df['STD'] * 2)
        df['Lower Band'] = df['SMA'] - (df['STD'] * 2)

        fig, ax = plt.subplots()
        ax.plot(df.index, df['Close'], label='Close Price', color='blue')
        ax.plot(df.index, df['SMA'], label='20-Day SMA', color='orange')
        ax.plot(df.index, df['Upper Band'], label='Upper Band', color='green')
        ax.plot(df.index, df['Lower Band'], label='Lower Band', color='red')
        ax.fill_between(df.index, df['Upper Band'], df['Lower Band'], color='gray', alpha=0.1)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=20)

        self.back_button = ttk.Button(self.main_frame, text="Back", command=self.show_prediction_options, style="TButton")
        self.back_button.pack(pady=20)

    def go_back_to_main(self):
        self.clear_frame(self.main_frame)
        
        # Recreate the widgets before packing them
        self.prediction_label = ttk.Label(self.main_frame, text="Stock Price Prediction using Brownian Motion", font=("Helvetica", 18))
        self.prediction_label.pack(pady=20)
        
        self.predict_button = ttk.Button(self.main_frame, text="Predict", command=self.ask_days_to_predict, style="TButton")
        self.predict_button.pack(pady=20)

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()

