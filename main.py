# Standard library imports
import datetime
import re
# Third-party library requirements
import pandas as pd
import QuantLib as ql
from yahoo_fin.options import get_calls
from yahoo_fin.stock_info import get_quote_table
# Hard-coded ticker and expiration for example purposes
ticker = "AAPL"
expiration = datetime.date(2021, 2, 12)
# Get the current stock price and dividend rate
info = get_quote_table(ticker)
current_price = info["Quote Price"]
yield_re = re.compile(r"\((?P<value>(\d+\.\d+))%\)")
try:
    dividend_yield = float(
        yield_re.search(info["Forward Dividend & Yield"])["value"]
    )
except (KeyError, ValueError, TypeError):
    dividend_yield = 0.0
# Fetch call option chain prices
calls = get_calls(ticker, expiration.strftime("%B %d, %Y"))
# Setup instruments for Black-Scholes pricing
today = ql.Date.todaysDate()
underlying = ql.SimpleQuote(current_price)
exercise = ql.AmericanExercise(
    today,
    ql.Date(expiration.day, expiration.month, expiration.year)
)
dividendYield = ql.FlatForward(
    today, dividend_yield, ql.Actual360()
)
riskFreeRate = ql.FlatForward(today, 0.0008913, ql.Actual360())
def create_option(row):
    volatility = ql.BlackConstantVol(
        today,
        ql.UnitedStates(),
        row["volatility"],
        ql.Business252()
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, row["Strike"]),
        exercise
    )
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(underlying),
        ql.YieldTermStructureHandle(dividendYield),
        ql.YieldTermStructureHandle(riskFreeRate),
        ql.BlackVolTermStructureHandle(volatility),
    )
    # Don't use the quoted implied vol
    # Calculate it out from the last price
    imp_vol = option.impliedVolatility(row["Last Price"], process)
    implied_volatility = ql.BlackConstantVol(
        today,
        ql.UnitedStates(),
        imp_vol,
        ql.Business252()
    )
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(underlying),
        ql.YieldTermStructureHandle(dividendYield),
        ql.YieldTermStructureHandle(riskFreeRate),
        ql.BlackVolTermStructureHandle(implied_volatility),
    )
    option.setPricingEngine(
        ql.FdBlackScholesVanillaEngine(process, 1000, 1000)
    )
    return {
        "Name": row["Contract Name"],
        "Strike": row["Strike"],
        "Last": row["Last Price"],
        "Bid": row["Bid"],
        "Ask": row["Ask"],
        "NPV": option.NPV(),
        "Delta": option.delta(),
        "Gamma": option.gamma(),
        "Theta": option.theta() / 365,
        "Volatility": imp_vol * 100,
    }
# Filter down to only OTM strikes
calls = calls[calls["Strike"] >= current_price * 1.025]
calls = calls[calls["Strike"] <= current_price * 1.10]
# Parse out implied volatility
calls = calls.assign(
    volatility=lambda x: x["Implied Volatility"].str.rstrip("%").astype("float") / 100,
)
# Price options and calculate greeks
options = calls.apply(create_option, axis=1, result_type="expand")

# This code is a continuation of the same file from above
# Pair up options with the next available option
pairs = pd.concat(
    [
        options.add_suffix("_1"),
        options.shift(-1).add_suffix("_2")
    ],
    axis=1
)
pairs = pairs[pairs["Name_2"].notna()]
def maximize_theta(row):
    bid, ask = row["Bid_1"], row["Ask_2"]
    strike_1, strike_2 = row["Strike_1"], row["Strike_2"]
    delta_1, delta_2 = row["Delta_1"], row["Delta_2"]
    gamma_1, gamma_2 = row["Gamma_1"], row["Gamma_2"]
    theta_1, theta_2 = row["Theta_1"], row["Theta_2"]
    def calculate_values(sell):
        buy = round(gamma_1 * sell / gamma_2)
        credit = (bid * sell * 100) - (ask * buy * 100) - ((sell + buy) * 0.65)
        shares = -1 * round(delta_2 * buy * 100 - delta_1 * sell * 100)
        delta = delta_2 * buy * 100 - delta_1 * sell * 100 + shares
        gamma = gamma_2 * buy * 100 - gamma_1 * sell * 100
        theta = theta_2 * buy * 100 - theta_1 * sell * 100
        share_cost = shares * current_price
        margin = strike_2 * buy * 100 - strike_1 * sell * 100
        return {
            "Sell Contract": f"{row['Strike_1']} @ {bid}",
            "Sell Amount": sell,
            "Buy Contract": f"{row['Strike_2']} @ {ask}",
            "Buy Amount": buy,
            "Shares": shares,
            "Share Cost": share_cost,
            "Margin": margin,
            "Credit": credit,
            "Cost Ratio": credit / (share_cost + margin),
            "Net Delta": delta,
            "Net Gamma": gamma,
            "Net Theta": theta,
            "Risk Ratio": theta / (delta ** 2 + gamma ** 2) ** 0.5,
        }
    trades = []
    sell = 1
    values = calculate_values(sell)
    while (abs(values["Share Cost"]) + abs(values["Margin"])) < 2500:
        if values["Shares"] > 0 and values["Credit"] > 0:
            trades.append(values)
        sell += 1
        values = calculate_values(sell)
    
    if trades:
        results = pd.DataFrame.from_records(trades)
        results.sort_values(
            by=["Risk Ratio", "Cost Ratio"],
            ascending=False,
            inplace=True
        )
        return results.iloc[0]
    return None
results = pairs.apply(maximize_theta, axis=1, result_type="expand")
results = results[results["Shares"].notna()]
print(results)
