import yfinance as yf

# We will value Apple for this project
ticker = 'AAPL'
stock = yf.Ticker(ticker)

print(f"--- Running Advanced DCF Valuation (Dynamic WACC) for {ticker} ---\n")

try:
    # 1. DYNAMIC WACC CALCULATION (CAPM)
    # Pull the live 10-Year Treasury Yield for the Risk-Free Rate (^TNX)
    treasury = yf.Ticker("^TNX")
    risk_free_rate = treasury.history(period="1d")['Close'].iloc[0] / 100
    
    # Pull Beta (Volatility) from Yahoo Finance
    beta = stock.info.get('beta', 1.1) 
    
    # Standard Equity Risk Premium (Historical market average)
    equity_risk_premium = 0.055 
    
    # Calculate Cost of Equity using CAPM
    cost_of_equity = risk_free_rate + (beta * equity_risk_premium)
    
    # Calculate Capital Structure Weights
    market_cap = stock.info.get('marketCap')
    total_debt = stock.info.get('totalDebt', 0)
    total_capital = market_cap + total_debt
    
    weight_of_equity = market_cap / total_capital
    weight_of_debt = total_debt / total_capital
    
    # Cost of Debt (Estimated 5%) and Corporate Tax Rate (21%)
    cost_of_debt = 0.05 
    tax_rate = 0.21
    
    # Final Dynamic WACC Formula
    wacc = (weight_of_equity * cost_of_equity) + (weight_of_debt * cost_of_debt * (1 - tax_rate))
    
    print(f"Live 10-Yr Treasury (Risk-Free Rate): {risk_free_rate:.2%}")
    print(f"{ticker} Live Beta: {beta}")
    print(f"Calculated Dynamic WACC (Discount Rate): {wacc:.2%}\n")

    # 2. FETCH FINANCIAL DATA FOR CASH FLOWS
    cash_flow = stock.cash_flow
    if 'Free Cash Flow' in cash_flow.index:
        recent_fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
    else:
        operating_cf = cash_flow.loc['Operating Cash Flow'].iloc[0]
        capex = cash_flow.loc['Capital Expenditure'].iloc[0]
        recent_fcf = operating_cf + capex 

    shares_out = stock.info.get('sharesOutstanding')
    total_cash = stock.info.get('totalCash', 0)
    current_price = stock.history(period="1d")['Close'].iloc[0]

    # 3. PROJECT FUTURE CASH FLOWS
    projected_growth_rate = 0.05   # 5% expected Free Cash Flow growth
    perpetual_growth_rate = 0.025  # 2.5% Terminal Growth (Average GDP growth)
    
    projected_fcf = []
    pv_fcf = 0 
    
    for year in range(1, 6):
        fcf = recent_fcf * ((1 + projected_growth_rate) ** year)
        projected_fcf.append(fcf)
        # Discount it back to today using our DYNAMIC WACC
        pv_fcf += fcf / ((1 + wacc) ** year)

    # 4. CALCULATE TERMINAL & ENTERPRISE VALUE
    terminal_value = (projected_fcf[-1] * (1 + perpetual_growth_rate)) / (wacc - perpetual_growth_rate)
    pv_terminal_value = terminal_value / ((1 + wacc) ** 5)
    
    enterprise_value = pv_fcf + pv_terminal_value
    equity_value = enterprise_value + total_cash - total_debt
    
    # 5. CALCULATE INTRINSIC VALUE
    intrinsic_value = equity_value / shares_out
    
    print(f"Current Stock Price: ${current_price:.2f}")
    print(f"Calculated Intrinsic Value: ${intrinsic_value:.2f}\n")
    
    if intrinsic_value > current_price:
        print("Verdict: UNDERVALUED (Buy Signal)")
    else:
        print("Verdict: OVERVALUED (Market is pricing in higher growth)")

except Exception as e:
    print(f"Data pull error: {e}")