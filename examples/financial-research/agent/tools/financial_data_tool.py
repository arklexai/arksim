from typing import Dict, Any


class FinancialDataClient:
    def __init__(self) -> None:
        # TODO: initialize MCP client / REST client here
        pass

    def get_income_statement(self, ticker: str) -> Dict[str, Any]:
        # TODO: replace with real MCP/REST call
        return {
            "ticker": ticker,
            "revenue_latest": "TODO",
            "gross_margin_latest": "TODO",
            "net_income_latest": "TODO"
        }

    def get_balance_sheet(self, ticker: str) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "cash_latest": "TODO",
            "debt_latest": "TODO"
        }

    def get_cash_flow(self, ticker: str) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "operating_cf_latest": "TODO",
            "fcf_latest": "TODO"
        }


def build_financial_context(ticker: str) -> str:
    client = FinancialDataClient()
    income = client.get_income_statement(ticker)
    balance = client.get_balance_sheet(ticker)
    cashflow = client.get_cash_flow(ticker)

    return f"""
Income statement:
{income}

Balance sheet:
{balance}

Cash flow:
{cashflow}
""".strip()