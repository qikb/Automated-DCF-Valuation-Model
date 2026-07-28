"""
Microsoft (MSFT) Discounted Cash Flow (DCF) Valuation Model
===========================================================

An unlevered free cash flow (FCF) DCF model for Microsoft Corporation with a
**dynamic, CAPM-derived WACC** sourced from live market data.

The model:
  1. Engineers a live Weighted Average Cost of Capital (WACC) using CAPM:
     pulls the 10-Year Treasury yield (^TNX) for the risk-free rate and the
     equity's Beta, market cap, debt, cash, shares and price via Yahoo Finance.
  2. Loads historical financials from ``data/msft_historical_financials.csv``.
  3. Builds a 5-year operating forecast (revenue, EBIT, NOPAT, D&A, capex, NWC).
  4. Derives unlevered free cash flow and discounts it at the dynamic WACC.
  5. Computes enterprise value, equity value and the implied intrinsic share
     price, and prints an Undervalued / Overvalued verdict vs. the live price.
  6. Runs a Base / Downside / Upside scenario analysis.
  7. Produces a WACC vs. terminal-growth sensitivity table.
  8. Writes every result to ``outputs/*.csv`` and renders the summary charts.

If the live market data cannot be reached (no network / API change), the model
transparently falls back to fixed, documented assumptions so it always runs.

All monetary figures are in US$ millions unless stated otherwise.

Run:
    python msft_dcf_model.py            # attempt live CAPM WACC
    python msft_dcf_model.py --offline  # force the fixed-assumption fallback
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # head-less rendering so the script runs anywhere
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: E402  (registered after backend selection)
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_FILE = os.path.join(DATA_DIR, "msft_historical_financials.csv")

FORECAST_YEARS = [2025, 2026, 2027, 2028, 2029]

# --------------------------------------------------------------------------- #
# Live-data / CAPM configuration
# --------------------------------------------------------------------------- #
TICKER = "MSFT"
RISK_FREE_TICKER = "^TNX"           # CBOE 10-Year Treasury yield index
EQUITY_RISK_PREMIUM = 0.055         # historical US market equity risk premium
COST_OF_DEBT = 0.05                 # estimated pre-tax cost of debt
WACC_TAX_RATE = 0.21                # statutory rate for the debt tax shield

# Fallback values used when live market data is unavailable.
FALLBACK_WACC = 0.090
FALLBACK_MARKET_PRICE = 428.0


# --------------------------------------------------------------------------- #
# Assumptions
# --------------------------------------------------------------------------- #
@dataclass
class Assumptions:
    """Operating and valuation assumptions that drive a single DCF run."""

    revenue_growth: float          # annual revenue growth applied each year
    ebit_margin: float             # operating (EBIT) margin
    tax_rate: float                # cash tax rate applied to EBIT
    da_pct_revenue: float          # depreciation & amortization, % of revenue
    capex_pct_revenue: float       # capital expenditure, % of revenue
    nwc_change_pct_revenue: float  # change in net working capital, % of revenue
    wacc: float                    # weighted average cost of capital
    terminal_growth: float         # perpetual growth rate for the terminal value

    @property
    def fcf_margin(self) -> float:
        """Implied unlevered FCF margin as a share of revenue."""
        nopat_margin = self.ebit_margin * (1.0 - self.tax_rate)
        return (
            nopat_margin
            + self.da_pct_revenue
            - self.capex_pct_revenue
            - self.nwc_change_pct_revenue
        )


# Base-case assumptions. These reflect a continuation of Microsoft's cloud /
# AI-driven growth with margins holding near recent highs. The ``wacc`` field is
# a placeholder: it is overwritten at run time with the live CAPM-derived WACC
# (see ``fetch_market_data``) unless the model is run with ``--offline``.
BASE_ASSUMPTIONS = Assumptions(
    revenue_growth=0.090,
    ebit_margin=0.450,
    tax_rate=0.160,
    da_pct_revenue=0.070,
    capex_pct_revenue=0.130,
    nwc_change_pct_revenue=0.040,
    wacc=FALLBACK_WACC,
    terminal_growth=0.030,
)


# --------------------------------------------------------------------------- #
# Live market data & dynamic CAPM WACC
# --------------------------------------------------------------------------- #
@dataclass
class MarketData:
    """Live market inputs and the WACC engineered from them."""

    is_live: bool
    wacc: float
    current_price: float
    risk_free_rate: float | None = None
    beta: float | None = None
    cost_of_equity: float | None = None
    cost_of_debt: float | None = None
    weight_of_equity: float | None = None
    weight_of_debt: float | None = None
    market_cap: float | None = None
    total_debt: float | None = None      # $ millions
    cash: float | None = None            # $ millions
    shares_outstanding: float | None = None  # millions
    net_debt: float | None = None        # $ millions (debt − cash)
    source: str = "fallback"


def fetch_market_data(offline: bool = False) -> MarketData:
    """Engineer a dynamic CAPM WACC from live Yahoo Finance data.

    Returns a :class:`MarketData` instance. On any failure (no network, missing
    fields, API change) or when ``offline`` is set, returns a fallback with
    fixed assumptions so the rest of the model always runs.
    """
    if offline:
        return MarketData(
            is_live=False,
            wacc=FALLBACK_WACC,
            current_price=FALLBACK_MARKET_PRICE,
            source="offline (forced)",
        )

    try:
        import yfinance as yf

        # 1. Risk-free rate from the live 10-Year Treasury yield (^TNX).
        treasury = yf.Ticker(RISK_FREE_TICKER).history(period="5d")
        risk_free_rate = float(treasury["Close"].dropna().iloc[-1]) / 100.0

        stock = yf.Ticker(TICKER)
        info = stock.info

        beta = float(info.get("beta") or 1.0)
        market_cap = float(info["marketCap"])
        total_debt = float(info.get("totalDebt") or 0.0)
        cash = float(info.get("totalCash") or 0.0)
        shares_outstanding = float(info["sharesOutstanding"])
        current_price = float(
            info.get("currentPrice")
            or stock.history(period="1d")["Close"].iloc[-1]
        )

        # 2. CAPM cost of equity and capital-structure weights.
        cost_of_equity = risk_free_rate + beta * EQUITY_RISK_PREMIUM
        total_capital = market_cap + total_debt
        weight_of_equity = market_cap / total_capital
        weight_of_debt = total_debt / total_capital

        # 3. Dynamic WACC.
        wacc = weight_of_equity * cost_of_equity + weight_of_debt * COST_OF_DEBT * (
            1.0 - WACC_TAX_RATE
        )

        # Convert balance-sheet absolutes to $ millions to match the model.
        total_debt_m = total_debt / 1e6
        cash_m = cash / 1e6
        shares_m = shares_outstanding / 1e6

        return MarketData(
            is_live=True,
            wacc=wacc,
            current_price=current_price,
            risk_free_rate=risk_free_rate,
            beta=beta,
            cost_of_equity=cost_of_equity,
            cost_of_debt=COST_OF_DEBT,
            weight_of_equity=weight_of_equity,
            weight_of_debt=weight_of_debt,
            market_cap=market_cap / 1e6,
            total_debt=total_debt_m,
            cash=cash_m,
            shares_outstanding=shares_m,
            net_debt=total_debt_m - cash_m,
            source="live (Yahoo Finance + CAPM)",
        )
    except Exception as exc:  # noqa: BLE001 - any failure -> documented fallback
        print(f"[warn] live market data unavailable ({exc}); using fallback.")
        return MarketData(
            is_live=False,
            wacc=FALLBACK_WACC,
            current_price=FALLBACK_MARKET_PRICE,
            source="fallback (live fetch failed)",
        )


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_historical_financials() -> pd.DataFrame:
    """Load the historical income-statement / cash-flow data."""
    df = pd.read_csv(DATA_FILE)
    df["ebit_margin"] = df["ebit"] / df["revenue"]
    df["net_debt"] = df["total_debt"] - df["cash_and_investments"]
    return df


# --------------------------------------------------------------------------- #
# Core DCF mechanics
# --------------------------------------------------------------------------- #
@dataclass
class DCFResult:
    forecast: pd.DataFrame
    pv_fcf: float
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    equity_value: float
    implied_share_price: float
    assumptions: Assumptions = field(repr=False, default=None)


def build_forecast(
    last_revenue: float, assumptions: Assumptions
) -> pd.DataFrame:
    """Project the operating model forward over the forecast horizon."""
    rows = []
    revenue = last_revenue
    for offset, year in enumerate(FORECAST_YEARS, start=1):
        revenue = revenue * (1.0 + assumptions.revenue_growth)
        ebit = revenue * assumptions.ebit_margin
        nopat = ebit * (1.0 - assumptions.tax_rate)
        da = revenue * assumptions.da_pct_revenue
        capex = revenue * assumptions.capex_pct_revenue
        change_nwc = revenue * assumptions.nwc_change_pct_revenue
        unlevered_fcf = nopat + da - capex - change_nwc
        discount_factor = 1.0 / ((1.0 + assumptions.wacc) ** offset)
        rows.append(
            {
                "year": year,
                "period": offset,
                "revenue": revenue,
                "revenue_growth": assumptions.revenue_growth,
                "ebit": ebit,
                "ebit_margin": assumptions.ebit_margin,
                "nopat": nopat,
                "depreciation_amortization": da,
                "capex": capex,
                "change_nwc": change_nwc,
                "unlevered_fcf": unlevered_fcf,
                "discount_factor": discount_factor,
                "pv_unlevered_fcf": unlevered_fcf * discount_factor,
            }
        )
    return pd.DataFrame(rows)


def run_dcf(
    historicals: pd.DataFrame,
    assumptions: Assumptions,
    net_debt: float | None = None,
    shares_outstanding: float | None = None,
) -> DCFResult:
    """Run a single DCF valuation and return the full result set.

    ``net_debt`` and ``shares_outstanding`` default to the latest historical
    figures but can be overridden with live values from :class:`MarketData`.
    """
    last_row = historicals.iloc[-1]
    last_revenue = float(last_row["revenue"])
    if net_debt is None:
        net_debt = float(last_row["net_debt"])
    if shares_outstanding is None:
        shares_outstanding = float(last_row["shares_outstanding"])

    forecast = build_forecast(last_revenue, assumptions)

    pv_fcf = float(forecast["pv_unlevered_fcf"].sum())

    final_fcf = float(forecast["unlevered_fcf"].iloc[-1])
    terminal_value = (
        final_fcf
        * (1.0 + assumptions.terminal_growth)
        / (assumptions.wacc - assumptions.terminal_growth)
    )
    final_discount_factor = float(forecast["discount_factor"].iloc[-1])
    pv_terminal_value = terminal_value * final_discount_factor

    enterprise_value = pv_fcf + pv_terminal_value
    equity_value = enterprise_value - net_debt  # net_debt is negative -> net cash
    implied_share_price = equity_value / shares_outstanding

    return DCFResult(
        forecast=forecast,
        pv_fcf=pv_fcf,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        implied_share_price=implied_share_price,
        assumptions=assumptions,
    )


# --------------------------------------------------------------------------- #
# Scenario analysis
# --------------------------------------------------------------------------- #
def build_scenarios() -> dict[str, Assumptions]:
    """Define Downside / Base / Upside scenarios."""
    return {
        "Downside": Assumptions(
            revenue_growth=0.040,
            ebit_margin=0.430,
            tax_rate=0.170,
            da_pct_revenue=0.070,
            capex_pct_revenue=0.140,
            nwc_change_pct_revenue=0.030,
            wacc=0.095,
            terminal_growth=0.020,
        ),
        "Base": BASE_ASSUMPTIONS,
        "Upside": Assumptions(
            revenue_growth=0.120,
            ebit_margin=0.460,
            tax_rate=0.160,
            da_pct_revenue=0.070,
            capex_pct_revenue=0.120,
            nwc_change_pct_revenue=0.040,
            wacc=0.085,
            terminal_growth=0.035,
        ),
    }


def run_scenarios(
    historicals: pd.DataFrame,
    market: MarketData,
    base_wacc: float,
    net_debt: float | None = None,
    shares_outstanding: float | None = None,
) -> pd.DataFrame:
    """Run Downside / Base / Upside scenarios.

    The Base scenario uses the dynamic ``base_wacc``; Downside and Upside shift
    the WACC up / down around it to reflect tighter / looser conditions.
    """
    rows = []
    scenarios = build_scenarios()
    scenarios["Base"].wacc = base_wacc
    scenarios["Downside"].wacc = base_wacc + 0.005
    scenarios["Upside"].wacc = base_wacc - 0.005
    for name, assumptions in scenarios.items():
        result = run_dcf(historicals, assumptions, net_debt, shares_outstanding)
        rows.append(
            {
                "scenario": name,
                "revenue_growth": assumptions.revenue_growth,
                "fcf_margin": assumptions.fcf_margin,
                "wacc": assumptions.wacc,
                "terminal_growth": assumptions.terminal_growth,
                "enterprise_value": result.enterprise_value,
                "equity_value": result.equity_value,
                "implied_share_price": result.implied_share_price,
                "upside_vs_market": result.implied_share_price / market.current_price
                - 1.0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sensitivity analysis
# --------------------------------------------------------------------------- #
TERMINAL_GROWTH_GRID = [0.020, 0.025, 0.030, 0.035, 0.040]


def run_sensitivity(
    historicals: pd.DataFrame,
    base_wacc: float,
    net_debt: float | None = None,
    shares_outstanding: float | None = None,
) -> pd.DataFrame:
    """Implied share price across a WACC x terminal-growth grid (base ops).

    The WACC grid is centered on the dynamic ``base_wacc`` (±1.0% in 0.5% steps)
    so the live discount rate sits in the middle of the table.
    """
    wacc_grid = [base_wacc + step for step in (-0.010, -0.005, 0.0, 0.005, 0.010)]
    data = {}
    for tg in TERMINAL_GROWTH_GRID:
        col = []
        for wacc in wacc_grid:
            assumptions = Assumptions(
                revenue_growth=BASE_ASSUMPTIONS.revenue_growth,
                ebit_margin=BASE_ASSUMPTIONS.ebit_margin,
                tax_rate=BASE_ASSUMPTIONS.tax_rate,
                da_pct_revenue=BASE_ASSUMPTIONS.da_pct_revenue,
                capex_pct_revenue=BASE_ASSUMPTIONS.capex_pct_revenue,
                nwc_change_pct_revenue=BASE_ASSUMPTIONS.nwc_change_pct_revenue,
                wacc=wacc,
                terminal_growth=tg,
            )
            col.append(
                run_dcf(
                    historicals, assumptions, net_debt, shares_outstanding
                ).implied_share_price
            )
        data[f"{tg:.1%}"] = col
    df = pd.DataFrame(data, index=[f"{w:.1%}" for w in wacc_grid])
    df.index.name = "WACC"
    return df


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_historical(historicals: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    years = historicals["fiscal_year"].astype(int)
    ax1.bar(years, historicals["revenue"], color="#1f77b4", label="Revenue")
    ax1.set_ylabel("Revenue ($ millions)")
    ax1.set_xticks(years)

    ax2 = ax1.twinx()
    ax2.plot(
        years,
        historicals["ebit_margin"],
        color="#c0392b",
        marker="o",
        label="EBIT margin",
    )
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    )
    ax2.set_ylabel("EBIT margin")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Microsoft Historical Revenue and EBIT Margin")
    ax1.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "historical_revenue_margin.png"), dpi=110)
    plt.close(fig)


def plot_forecast(result: DCFResult) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    years = result.forecast["year"].astype(int)
    ax.bar(
        years,
        result.forecast["unlevered_fcf"],
        color="#2e8b57",
        label="Unlevered FCF",
    )
    ax.plot(
        years,
        result.forecast["revenue"],
        color="#1f77b4",
        marker="o",
        label="Revenue",
    )
    ax.set_ylabel("$ millions")
    ax.set_xticks(years)
    ax.set_title("Microsoft Forecast Revenue and Unlevered Free Cash Flow")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "forecast_revenue_fcf.png"), dpi=110)
    plt.close(fig)


def plot_scenarios(scenarios: pd.DataFrame, current_price: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["Downside", "Base", "Upside"]
    colors = {"Downside": "#c0504d", "Base": "#1f77b4", "Upside": "#2e8b57"}
    ordered = scenarios.set_index("scenario").loc[order]
    ax.bar(
        order,
        ordered["implied_share_price"],
        color=[colors[s] for s in order],
    )
    ax.axhline(
        current_price,
        color="#333333",
        linestyle="--",
        label="Current market price",
    )
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax.set_ylabel("Implied share price")
    ax.set_title("Microsoft Scenario Valuation")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "scenario_valuation.png"), dpi=110)
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = sensitivity.values
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(sensitivity.columns)))
    ax.set_xticklabels(sensitivity.columns)
    ax.set_yticks(range(len(sensitivity.index)))
    ax.set_yticklabels(sensitivity.index)
    ax.set_xlabel("Terminal growth")
    ax.set_ylabel("WACC")
    ax.set_title("Sensitivity: WACC vs. Terminal Growth")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                f"${data[i, j]:,.0f}",
                ha="center",
                va="center",
                color="#222222",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Implied share price")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "sensitivity_wacc_terminal_growth.png"), dpi=110
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_outputs(
    base_result: DCFResult,
    scenarios: pd.DataFrame,
    sensitivity: pd.DataFrame,
    market: MarketData,
    net_debt: float,
    shares_outstanding: float,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_result.forecast.round(2).to_csv(
        os.path.join(OUTPUT_DIR, "forecast_model.csv"), index=False
    )
    scenarios.round(4).to_csv(
        os.path.join(OUTPUT_DIR, "scenario_analysis.csv"), index=False
    )
    sensitivity.round(2).to_csv(
        os.path.join(OUTPUT_DIR, "sensitivity_wacc_terminal_growth.csv")
    )

    verdict = (
        "UNDERVALUED"
        if base_result.implied_share_price > market.current_price
        else "OVERVALUED"
    )
    summary = pd.DataFrame(
        [
            {"metric": "Data source", "value": market.source},
            {"metric": "Risk-free rate (10Y)", "value": _fmt(market.risk_free_rate)},
            {"metric": "Beta", "value": _fmt(market.beta)},
            {"metric": "Cost of equity (CAPM)", "value": _fmt(market.cost_of_equity)},
            {"metric": "Cost of debt (pre-tax)", "value": _fmt(market.cost_of_debt)},
            {"metric": "Weight of equity", "value": _fmt(market.weight_of_equity)},
            {"metric": "Weight of debt", "value": _fmt(market.weight_of_debt)},
            {"metric": "WACC", "value": round(base_result.assumptions.wacc, 4)},
            {
                "metric": "Terminal growth",
                "value": base_result.assumptions.terminal_growth,
            },
            {"metric": "PV of explicit FCF ($M)", "value": round(base_result.pv_fcf, 2)},
            {
                "metric": "Terminal value ($M)",
                "value": round(base_result.terminal_value, 2),
            },
            {
                "metric": "PV of terminal value ($M)",
                "value": round(base_result.pv_terminal_value, 2),
            },
            {
                "metric": "Enterprise value ($M)",
                "value": round(base_result.enterprise_value, 2),
            },
            {"metric": "Net debt ($M)", "value": round(net_debt, 2)},
            {
                "metric": "Equity value ($M)",
                "value": round(base_result.equity_value, 2),
            },
            {
                "metric": "Shares outstanding (M)",
                "value": round(shares_outstanding, 2),
            },
            {
                "metric": "Implied share price ($)",
                "value": round(base_result.implied_share_price, 2),
            },
            {
                "metric": "Current market price ($)",
                "value": round(market.current_price, 2),
            },
            {
                "metric": "Upside / (downside) vs. market",
                "value": round(
                    base_result.implied_share_price / market.current_price - 1.0, 4
                ),
            },
            {"metric": "Verdict", "value": verdict},
        ]
    )
    summary.to_csv(os.path.join(OUTPUT_DIR, "valuation_summary.csv"), index=False)


def _fmt(value: float | None) -> float | str:
    """Round a metric for the summary CSV, leaving blanks for missing data."""
    return round(value, 4) if value is not None else ""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip live data and use fixed fallback assumptions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historicals = load_historical_financials()

    market = fetch_market_data(offline=args.offline)
    base_wacc = market.wacc

    # Use live balance-sheet figures when available, else the latest historicals.
    last_row = historicals.iloc[-1]
    net_debt = (
        market.net_debt
        if market.net_debt is not None
        else float(last_row["net_debt"])
    )
    shares_outstanding = (
        market.shares_outstanding
        if market.shares_outstanding is not None
        else float(last_row["shares_outstanding"])
    )

    BASE_ASSUMPTIONS.wacc = base_wacc

    base_result = run_dcf(historicals, BASE_ASSUMPTIONS, net_debt, shares_outstanding)
    scenarios = run_scenarios(
        historicals, market, base_wacc, net_debt, shares_outstanding
    )
    sensitivity = run_sensitivity(
        historicals, base_wacc, net_debt, shares_outstanding
    )

    write_outputs(
        base_result, scenarios, sensitivity, market, net_debt, shares_outstanding
    )

    plot_historical(historicals)
    plot_forecast(base_result)
    plot_scenarios(scenarios, market.current_price)
    plot_sensitivity(sensitivity)

    verdict = (
        "UNDERVALUED (model > market)"
        if base_result.implied_share_price > market.current_price
        else "OVERVALUED (market > model)"
    )

    print("=== Microsoft DCF Valuation ===")
    print(f"Data source       : {market.source}")
    if market.is_live:
        print(f"Risk-free rate    : {market.risk_free_rate:.2%}")
        print(f"Beta              : {market.beta:.3f}")
        print(f"Cost of equity    : {market.cost_of_equity:.2%}")
        print(f"Dynamic WACC      : {base_wacc:.2%}")
    else:
        print(f"WACC (fallback)   : {base_wacc:.2%}")
    print(f"Enterprise value  : ${base_result.enterprise_value:,.0f}M")
    print(f"Equity value      : ${base_result.equity_value:,.0f}M")
    print(f"Implied price     : ${base_result.implied_share_price:,.2f}")
    print(f"Market price      : ${market.current_price:,.2f}")
    print(
        f"Upside/(downside) : "
        f"{base_result.implied_share_price / market.current_price - 1.0:+.1%}"
    )
    print(f"Verdict           : {verdict}")
    print("\nScenario implied prices:")
    for _, row in scenarios.iterrows():
        print(f"  {row['scenario']:<9}: ${row['implied_share_price']:,.2f}")
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
