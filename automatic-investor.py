import yfinance as yf
import pandas as pd
import numpy as np

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

def fetch_stock_data(tickers):
    stock_data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            pe_ratio = info.get('trailingPE', None)
            pb_ratio = info.get('priceToBook', None)
            current_price = info.get('currentPrice', None)


            if pe_ratio and pb_ratio and current_price:
                stock_data.append({
                    'Ticker': ticker,
                    'Price': current_price,
                    'P/E': pe_ratio,
                    'P/B': pb_ratio
                })
                print(f"Added {ticker}: P/E={pe_ratio:.2f}, P/B={pb_ratio:.2f}, Price={current_price:.2f}")
            else:
                print(f"Can't add {ticker}: Missing data")

        except Exception as e:
            print(f"Can't add {ticker}: Error - {e}")

    return pd.DataFrame(stock_data)

def main():
    df = fetch_stock_data(STOCK_TICKERS)
    if not df.empty:
        print(df.head())
    else:
        print("No data found")


if __name__ == '__main__':
    main()