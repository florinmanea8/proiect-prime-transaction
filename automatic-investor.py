import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import time

from six import ensure_text

# Am configurat un buget de 100.000$ si am creat o lista cu primele 50 de companii din S&P500

BUDGET = 100000

STOCK_TICKERS = [
    'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'AVGO',
    'TSLA', 'BRK-B', 'LLY', 'JPM', 'WMT', 'V', 'ORCL', 'MA', 'XOM',
    'JNJ', 'PLTR', 'BAC', 'ABBV', 'NFLX', 'COST', 'AMD', 'HD', 'PG',
    'GE', 'MU', 'CSCO', 'CVX', 'KO', 'WFC', 'UNH', 'MS', 'IBM', 'GS',
    'CAT', 'MRK', 'AXP', 'PM', 'RTX', 'CRM', 'APP', 'LRCX', 'MCD',
    'TMUS', 'TMO', 'C', 'ABT', 'AMAT'
]

WEIGHT_EXPONENT = 3

CATEGORY_MULTIPLIERS = {
    'both_undervalued': 2.0,
    'pe_undervalued': 1.2,
    'pb_undervalued': 1.2,
    'both_overvalued': 1.0
}

def get_single_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    pe_ratio = info.get('trailingPE', None)
    pb_ratio = info.get('priceToBook', None)
    current_price = info.get('currentPrice', None)

    if pe_ratio and pb_ratio and current_price:
        return {
            'Ticker': ticker,
            'Price': current_price,
            'P/E': pe_ratio,
            'P/B': pb_ratio
            }

    return None

def fetch_stock_data(tickers):
    stock_data = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(get_single_stock_data, tickers)

        for result in results:
            if result:
                stock_data.append(result)
                print(f"Added {result['Ticker']}, Price: ${result['Price']:.2f}, P/E: {result['P/E']:.2f}, P/B: {result['P/B']:.2f}")

    return pd.DataFrame(stock_data)

def calculate_market_averages(df):
    avg_pe = df['P/E'].mean()
    avg_pb = df['P/B'].mean()

    print(f"\nMarket Averages:")
    print(f"    Average P/E: {avg_pe:.2f}")
    print(f"    Average P/B: {avg_pb:.2f}")

    return avg_pe, avg_pb

def get_classification(row, avg_pe, avg_pb):
    pe_under = row['P/E'] < avg_pe
    pb_under = row['P/B'] < avg_pb

    if pe_under and pb_under:
        return 'both_undervalued'
    elif pe_under and not pb_under:
        return 'pe_undervalued'
    elif not pe_under and pb_under:
        return 'pb_undervalued'
    else:
        return 'both_overvalued'

def classify_stock_type(df, avg_pe, avg_pb):
    df['Valuation_Type'] = df.apply(lambda row: get_classification(row, avg_pe, avg_pb), axis=1)

    print("\nStock Classification:")
    for val_type, multiplier in CATEGORY_MULTIPLIERS.items():
        count = len(df[df['Valuation_Type'] == val_type])

        type_name = {
            'both_undervalued': "Both P/E and P/B are Undervalued",
            'pe_undervalued': "P/E Undervalued, P/B Overvalued",
            'pb_undervalued': "P/E Overvalued, P/B Undervalued",
            'both_overvalued': "Both P/E and P/B are Overvalued"
        }[val_type]

        print(f"    {type_name}: {count} stocks (multiplier: {multiplier}x)")

    print('')

    return df

def main():
    df = fetch_stock_data(STOCK_TICKERS)
    if df.empty:
        print("No data available")
        return

    avg_pe, avg_pb = calculate_market_averages(df)

    df = classify_stock_type(df, avg_pe, avg_pb)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    display_cols = ['Ticker', 'Price', 'P/E', 'P/B', 'Valuation_Type']
    print(df[display_cols])

if __name__ == '__main__':
    main()