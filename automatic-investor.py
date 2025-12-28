import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures

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
                print(f"✓ Added {result['Ticker']}, Price: ${result['Price']:.2f}, P/E: {result['P/E']:.2f}, P/B: {result['P/B']:.2f}")
            else:
                print(f"✗ Can't add {result['Ticker']}: Missing Data")

    return pd.DataFrame(stock_data)

def calculate_market_averages(df):
    avg_pe = df['P/E'].mean()
    avg_pb = df['P/B'].mean()

    print(f"\n📊 Market Averages:")
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

    print("\nc Stock Classification:")
    for val_type, multiplier in CATEGORY_MULTIPLIERS.items():
        count = len(df[df['Valuation_Type'] == val_type])

        type_name = {
            'both_undervalued': "✅ Both P/E and P/B are Undervalued",
            'pe_undervalued': "⚠️ P/E Undervalued, P/B Overvalued",
            'pb_undervalued': "⚠️ P/E Overvalued, P/B Undervalued",
            'both_overvalued': "❌ Both P/E and P/B are Overvalued"
        }[val_type]

        print(f"    {type_name}: {count} stocks (multiplier: {multiplier}x)")

    return df

def calculate_raw_scores(df, avg_pe, avg_pb):
    df = classify_stock_type(df, avg_pe, avg_pb)
    df['PE_Distance'] = avg_pe - df['P/E']
    df['PB_Distance'] = avg_pb - df['P/B']

    df['Base_Score'] = df['PE_Distance'] + df['PB_Distance']

    df['Category_Multiplier'] = df['Valuation_Type'].map(CATEGORY_MULTIPLIERS)
    df['Raw_Score'] = df['Base_Score'] * df['Category_Multiplier']

    return df

def normalize_scores(df):
    min_score = df['Base_Score'].min()
    if min_score < 0:
        df['Adjusted_Score'] = df['Raw_Score'] - min_score + 1
    else:
        df['Adjusted_Score'] = df['Raw_Score'] + 1

    df['Weighted_Score'] = df['Adjusted_Score'] ** WEIGHT_EXPONENT

    return df

def calculate_allocation_amounts(df, budget):
    total_weighted_score = df['Weighted_Score'].sum()
    df['Allocation_Pct'] = df['Weighted_Score'] / total_weighted_score

    df['Allocation_$'] = df ['Allocation_Pct'] * budget

    df['Shares'] = np.floor(df['Allocation_$'] / df['Price'])

    df['Actual_Investment'] = df['Shares'] * df['Price']

    return df

def categorize_stocks(df):
    category_map = {
        'both_undervalued': "🌟 Both Undervalued",
        'pe_undervalued': "✅ P/E Undervalued",
        'pb_undervalued': "✅ P/B Undervalued",
        'both_overvalued': "❌ Both Overvalued"
    }

    df['Category'] = df['Valuation_Type'].map(category_map)

    return df

def calculate_scores_and_allocation(df, budget):
    avg_pe, avg_pb = calculate_market_averages(df)

    df = calculate_raw_scores(df, avg_pe, avg_pb)

    df = normalize_scores(df)

    df = calculate_allocation_amounts(df, budget)

    df = categorize_stocks(df)

    return df, avg_pe, avg_pb

def adjust_to_exact_budget(df, budget):
    total_invested = df['Actual_Investment'].sum()
    remaining = budget - total_invested

    if remaining > 0:
        while remaining > 0:
            df_sorted = df.sort_values('Weighted_Score', ascending=False)
            share_added = False

            for index in df_sorted.index:
                stock_price = df.loc[index, 'Price']

                if remaining >= stock_price:
                    df.loc[index, 'Shares'] += 1
                    df.loc[index, 'Actual_Investment'] += stock_price
                    remaining -= stock_price
                    share_added = True
                    break

            if not share_added:
                break

    if remaining > 0:
        max_iterations = 1000
        iteration = 0

        while remaining > 0 and iteration < max_iterations:
            iteration += 1
            swap_made = False

            df_sorted = df.sort_values('Weighted_Score', ascending=True)

            for sell_index in df_sorted.index:
                if df.loc[sell_index, 'Shares'] > 0:
                    sell_price = df.loc[sell_index, 'Price']
                    potential_budget = remaining + sell_price

                    df_buy_candidates = df.sort_values('Weighted_Score', ascending=False)

                    for buy_index in df_buy_candidates.index:
                        buy_price = df.loc[buy_index, 'Price']

                        if buy_price <= potential_budget:
                            new_remaining = potential_budget - buy_price

                            if new_remaining < remaining:
                                df.loc[sell_index, 'Shares'] -= 1
                                df.loc[sell_index, 'Actual_Investment'] -= sell_price
                                df.loc[sell_index, 'Shares'] += 1
                                df.loc[sell_index, 'Actual_Investment'] += buy_price
                                remaining = new_remaining
                                swap_made = True
                                break
                    if swap_made:
                        break
            if not swap_made:
                break

    final_total = df['Actual_Investment'].sum()
    final_remaining = budget - final_total

    print(f"\n💰 Final Investment: ${final_total:,.2f}")
    print(f"    Remaining: ${final_remaining:,.2f}")

    if final_remaining == 0:
        print("    🎯 Perfect! Exactly $0 remaining!")
    elif final_remaining < 1:
        print("    ⚠️ Unable to reach exactly $0 (closest possible with whole shares)")

    return df, remaining

def display_results(df):
    print("\n" + "="*90)
    print("📈 FINAL PORTFOLIO ALLOCATION")
    print("="*90)

    df_display = df.sort_values('Actual_Investment', ascending=False)

    categories = [
        '🌟 Both Undervalued',
        '✅ P/E Undervalued',
        '✅ P/B Undervalued',
        '❌ Both Overvalued'
    ]

    for category in categories:
        stocks_in_category = df_display[df_display['Category'] == category]

        if len(stocks_in_category) > 0:
            category_total = stocks_in_category['Actual_Investment'].sum()
            category_pct = (category_total / df['Actual_Investment'].sum()) * 100

            print(f"\n{category} - ${category_total:,.2f} ({category_pct:.1f}%)")
            print("-" * 90)

            for _, row in stocks_in_category.iterrows():
                print(f"{row['Ticker']:6} | ${row['Price']:8.2f}/share | "
                      f"P/E: {row['P/E']:6.2f} | P/B: {row['P/B']:6.2f} | "
                      f"Score: {row['Raw_Score']:7.2f}")
                print(f"        | Shares {int(row['Shares']):4} | "
                      f"Invested: ${row['Actual_Investment']:10,.2f} "
                      f"({row['Allocation_Pct']*100:5.2f}%)")

    print("\n" + "="*90)
    print(f"💰 TOTAL INVESTED: ${df['Actual_Investment'].sum():,.2f}")
    print(f"📋 Number of stocks in portfolio: {len(df[df['Shares'] > 0])}")
    print("="*90)

    print("\n📊 Portfolio Breakdown by Category:")
    for val_type, category_name in [
        ('both_undervalued', '🌟 Both Undervalued'),
        ('pe_undervalued', '✅ P/E Undervalued'),
        ('pb_undervalued', '✅ P/B Undervalued'),
        ('both_overvalued', '❌ Both Overvalued')
    ]:
        amount = df[df['Valuation_Type'] == val_type]['Actual_Investment'].sum()
        pct = (amount / df['Actual_Investment'].sum()) * 100
        count = len(df[df['Valuation_Type'] == val_type])
        print(f"    {category_name}: ${amount:,.2f} ({pct:.1f}%) across {count} stocks")

def main():
    print("🚀 Stock Portfolio Allocation Calculator")
    print(f"\nBudget: ${BUDGET:,.2f}\n")

    df = fetch_stock_data(STOCK_TICKERS)

    if len(df) == 0:
        print("❌ No stock data available")
        return

    allocated_df, avg_pe, avg_pb = calculate_scores_and_allocation(df, BUDGET)

    final_df, remaining = adjust_to_exact_budget(allocated_df, BUDGET)

    display_results(final_df)

    final_df.to_csv('portfolio_allocation.csv', index=False)
    print("\n💾 Results saved to 'portfolio_allocation.csv'")


if __name__ == '__main__':
    main()