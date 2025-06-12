import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import json
import logging

logger = logging.getLogger(__name__)


class FinancialDataParser:
    """
    Enhanced Parser to convert Task Planner parameters into YFinance API calls
    """
    
    def __init__(self):
        # Mapping from your metric names to YFinance field names
        self.metric_mapping = {
            # Income Statement Metrics
            'net_income': 'Net Income',
            'revenue': 'Total Revenue',
            'gross_profit': 'Gross Profit',
            'operating_income': 'Operating Income',
            'ebitda': 'EBITDA',
            
            # Balance Sheet Metrics
            'total_assets': 'Total Assets',
            'total_debt': 'Total Debt',
            'cash': 'Cash And Cash Equivalents',
            'total_equity': 'Total Equity',
            
            # Cash Flow Metrics
            'operating_cash_flow': 'Operating Cash Flow',
            'free_cash_flow': 'Free Cash Flow',
            'capital_expenditures': 'Capital Expenditures',
            
            # Calculated/Ratio Metrics
            'profit_margin': 'calculated',
            'gross_margin': 'calculated',
            'operating_margin': 'calculated',
            'debt_to_equity': 'calculated',
            'return_on_equity': 'calculated',
            'return_on_assets': 'calculated',
            
            # Market Data & Ratios (from info)
            'pe_ratio': 'info_field',
            'forward_pe': 'info_field',
            'price_to_book': 'info_field',
            'beta': 'info_field',
            'dividend_yield': 'info_field',
            'market_cap': 'info_field',
            'enterprise_value': 'info_field',
            
            # Technical Indicators
            'current_price': 'info_field',
            'fifty_two_week_high': 'info_field',
            'fifty_two_week_low': 'info_field',
            'fifty_day_average': 'info_field',
            'two_hundred_day_average': 'info_field',
            
            # Company Information
            'company_info': 'company_profile',
            'business_summary': 'info_field',
            'sector': 'info_field',
            'industry': 'info_field',
            'employee_count': 'info_field',
            
            # News and Recommendations
            'analyst_recommendations': 'recommendations',
            'company_news': 'news'
        }
        
        # Info field mappings for market data
        self.info_field_mapping = {
            'pe_ratio': 'trailingPE',
            'forward_pe': 'forwardPE',
            'price_to_book': 'priceToBook',
            'beta': 'beta',
            'dividend_yield': 'dividendYield',
            'market_cap': 'marketCap',
            'enterprise_value': 'enterpriseValue',
            'current_price': ['regularMarketPrice', 'currentPrice'],  # Fallback options
            'fifty_two_week_high': 'fiftyTwoWeekHigh',
            'fifty_two_week_low': 'fiftyTwoWeekLow',
            'fifty_day_average': 'fiftyDayAverage',
            'two_hundred_day_average': 'twoHundredDayAverage',
            'business_summary': 'longBusinessSummary',
            'sector': 'sector',
            'industry': 'industry',
            'employee_count': 'fullTimeEmployees'
        }
    
    def parse_task_planner_request(self, parameters: Dict) -> Dict:
        """
        Main function to process Task Planner financial data requests
        
        Input format from Task Planner:
        {
            "ticker": ,
            "timeframe": a to b, 
            "metric": "net_income, profit_margin, pe_ratio"
        }
        """
        ticker_symbol = parameters.get('ticker', '').upper()
        metrics = [m.strip().lower() for m in parameters.get('metric', '').split(',')]
        timeframe = parameters.get('timeframe', '')
        
        # Initialize ticker
        ticker = yf.Ticker(ticker_symbol)
        
        # Ticker validation
        try:
            info = ticker.info
            if not info or len(info) < 3:
                return {"error": f"Invalid ticker symbol: {ticker_symbol}"}
        except Exception as e:
            return {"error": f"Failed to fetch ticker {ticker_symbol}: {str(e)}"}
        
        results = {
            "ticker": ticker_symbol,
            "timeframe": timeframe,
            "data": {},
            "metadata": {
                "company_name": info.get("longName", info.get("shortName", "Unknown")),
                "currency": info.get("currency", "USD"),
                "last_updated": datetime.now().isoformat()
            }
        }
        

        for metric in metrics:
            if not metric:  
                continue
                
            try:
                metric_data = self.get_metric_data(ticker, metric, timeframe)
                results["data"][metric] = metric_data
            except Exception as e:
                results["data"][metric] = {"error": f"Failed to get {metric}: {str(e)}"}
        
        return results
    
    def get_metric_data(self, ticker, metric: str, timeframe: str) -> Dict:
        """
        Extract specific metric data from YFinance
        """
        
        if metric in ['net_income', 'revenue', 'gross_profit', 'operating_income', 'ebitda']:
            return self._get_financials_metric(ticker, metric, timeframe)
        
        elif metric in ['total_assets', 'total_debt', 'cash', 'total_equity']:
            return self._get_balance_sheet_metric(ticker, metric, timeframe)
        
        elif metric in ['operating_cash_flow', 'free_cash_flow', 'capital_expenditures']:
            return self._get_cashflow_metric(ticker, metric, timeframe)
        
        elif metric in ['profit_margin', 'gross_margin', 'operating_margin', 'debt_to_equity', 'return_on_equity', 'return_on_assets']:
            return self._get_calculated_metric(ticker, metric, timeframe)
        
        elif metric == 'stock_price':
            return self._get_stock_price(ticker, timeframe)
        
        elif metric == 'company_info':
            return self._get_company_profile(ticker)
        
        elif metric == 'analyst_recommendations':
            return self._get_analyst_recommendations(ticker)
        
        elif metric == 'company_news':
            return self._get_company_news(ticker)
        
        elif metric in self.info_field_mapping:
            return self._get_info_field_metric(ticker, metric)
        else:
            return {"error": f"Unsupported metric: {metric}"}
    
    def _get_financials_metric(self, ticker, metric: str, timeframe: str) -> Dict:
        """Get data from quarterly_financials"""
        
        yf_field = self.metric_mapping[metric]
        
        try:
            quarterly_data = ticker.quarterly_financials
            
            if quarterly_data.empty or yf_field not in quarterly_data.index:
                return {"error": f"{yf_field} not available in financials"}
            
            data_series = quarterly_data.loc[yf_field]
            
            # Filter by timeframe
            if timeframe and "to" in timeframe:
                start_date, end_date = self._parse_timeframe(timeframe)
                if start_date and end_date:
                    data_series = data_series[
                        (data_series.index >= start_date) & 
                        (data_series.index <= end_date)
                    ]
            logger.info("finanial metric extracted successfully")

            return {
                "values": {str(k): float(v) if pd.notna(v) else None for k, v in data_series.to_dict().items()},
                "latest_value": float(data_series.iloc[0]) if len(data_series) > 0 and pd.notna(data_series.iloc[0]) else None,
                "unit": "USD",
                "quarters_count": len(data_series),
                "data_type": "quarterly_financials"
            }
            
        except Exception as e:
            return {"error": f"Error fetching {metric}: {str(e)}"}
    
    def _get_balance_sheet_metric(self, ticker, metric: str, timeframe: str) -> Dict:
        """Get data from quarterly_balance_sheet"""
        
        yf_field = self.metric_mapping[metric]
        
        try:
            quarterly_data = ticker.quarterly_balance_sheet
            
            if quarterly_data.empty or yf_field not in quarterly_data.index:
                return {"error": f"{yf_field} not available in balance sheet"}
            
            data_series = quarterly_data.loc[yf_field]
            
            return {
                "values": {str(k): float(v) if pd.notna(v) else None for k, v in data_series.to_dict().items()},
                "latest_value": float(data_series.iloc[0]) if len(data_series) > 0 and pd.notna(data_series.iloc[0]) else None,
                "unit": "USD",
                "quarters_count": len(data_series),
                "data_type": "quarterly_balance_sheet"
            }
            
        except Exception as e:
            return {"error": f"Error fetching {metric}: {str(e)}"}
    
    def _get_cashflow_metric(self, ticker, metric: str, timeframe: str) -> Dict:
        """Get data from quarterly_cashflow"""
        
        yf_field = self.metric_mapping[metric]
        
        try:
            quarterly_data = ticker.quarterly_cashflow
            
            if quarterly_data.empty or yf_field not in quarterly_data.index:
                return {"error": f"{yf_field} not available in cash flow"}
            
            data_series = quarterly_data.loc[yf_field]
            
            return {
                "values": {str(k): float(v) if pd.notna(v) else None for k, v in data_series.to_dict().items()},
                "latest_value": float(data_series.iloc[0]) if len(data_series) > 0 and pd.notna(data_series.iloc[0]) else None,
                "unit": "USD",
                "quarters_count": len(data_series),
                "data_type": "quarterly_cashflow"
            }
            
        except Exception as e:
            return {"error": f"Error fetching {metric}: {str(e)}"}
    
    def _get_calculated_metric(self, ticker, metric: str, timeframe: str) -> Dict:
        """Calculate derived metrics with calculations.
        Although it can be handled with seperate calculator tool"""
        
        try:
            if metric == 'profit_margin':
                financials = ticker.quarterly_financials
                
                if 'Net Income' not in financials.index or 'Total Revenue' not in financials.index:
                    return {"error": "Cannot calculate profit margin - missing data"}
                
                net_income = financials.loc['Net Income']
                revenue = financials.loc['Total Revenue']
                margin = (net_income / revenue * 100).round(2)
                
                return {
                    "values": {str(k): float(v) if pd.notna(v) else None for k, v in margin.to_dict().items()},
                    "latest_value": float(margin.iloc[0]) if len(margin) > 0 and pd.notna(margin.iloc[0]) else None,
                    "unit": "percentage",
                    "calculation": "Net Income / Total Revenue * 100"
                }
            
            elif metric == 'gross_margin':
                financials = ticker.quarterly_financials
                
                if 'Gross Profit' not in financials.index or 'Total Revenue' not in financials.index:
                    return {"error": "Cannot calculate gross margin - missing data"}
                
                gross_profit = financials.loc['Gross Profit']
                revenue = financials.loc['Total Revenue']
                margin = (gross_profit / revenue * 100).round(2)
                
                return {
                    "values": {str(k): float(v) if pd.notna(v) else None for k, v in margin.to_dict().items()},
                    "latest_value": float(margin.iloc[0]) if len(margin) > 0 and pd.notna(margin.iloc[0]) else None,
                    "unit": "percentage",
                    "calculation": "Gross Profit / Total Revenue * 100"
                }
            
            elif metric == 'debt_to_equity':
                balance_sheet = ticker.quarterly_balance_sheet
                
                if 'Total Debt' not in balance_sheet.index or 'Total Equity' not in balance_sheet.index:
                    return {"error": "Cannot calculate debt-to-equity - missing data"}
                
                debt = balance_sheet.loc['Total Debt']
                equity = balance_sheet.loc['Total Equity']
                ratio = (debt / equity).round(2)
                
                return {
                    "values": {str(k): float(v) if pd.notna(v) else None for k, v in ratio.to_dict().items()},
                    "latest_value": float(ratio.iloc[0]) if len(ratio) > 0 and pd.notna(ratio.iloc[0]) else None,
                    "unit": "ratio",
                    "calculation": "Total Debt / Total Equity"
                }
            
            elif metric == 'return_on_equity':
                financials = ticker.quarterly_financials
                balance_sheet = ticker.quarterly_balance_sheet
                
                if 'Net Income' not in financials.index or 'Total Equity' not in balance_sheet.index:
                    return {"error": "Cannot calculate ROE - missing data"}
                
                net_income = financials.loc['Net Income']
                equity = balance_sheet.loc['Total Equity']
                # Align dates
                common_dates = net_income.index.intersection(equity.index)
                if len(common_dates) == 0:
                    return {"error": "Cannot calculate ROE - no matching dates"}
                
                roe = (net_income[common_dates] / equity[common_dates] * 100).round(2)
                
                return {
                    "values": {str(k): float(v) if pd.notna(v) else None for k, v in roe.to_dict().items()},
                    "latest_value": float(roe.iloc[0]) if len(roe) > 0 and pd.notna(roe.iloc[0]) else None,
                    "unit": "percentage",
                    "calculation": "Net Income / Total Equity * 100"
                }
                
        except Exception as e:
            return {"error": f"Error calculating {metric}: {str(e)}"}
    
    def _get_info_field_metric(self, ticker, metric: str) -> Dict:
        """Get data from ticker.info"""
        
        try:
            info = ticker.info
            field_name = self.info_field_mapping[metric]
            
            if isinstance(field_name, list):
                value = None
                for field in field_name:
                    value = info.get(field)
                    if value is not None:
                        break
            else:
                value = info.get(field_name)
            
            # Determine unit based on metric type
            unit = "USD" if metric in ['market_cap', 'enterprise_value'] else \
                   "percentage" if metric in ['dividend_yield'] else \
                   "ratio" if metric in ['pe_ratio', 'forward_pe', 'price_to_book', 'beta'] else \
                   "number"
            
            return {
                "value": value,
                "unit": unit,
                "data_type": "company_info",
                "field_name": field_name if not isinstance(field_name, list) else field_name[0]
            }
            
        except Exception as e:
            return {"error": f"Error fetching {metric}: {str(e)}"}
    
    def _get_company_profile(self, ticker) -> Dict:
        """Get comprehensive company profile"""
        
        try:
            info = ticker.info
            
            profile = {
                "name": info.get("longName", info.get("shortName")),
                "symbol": info.get("symbol"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "business_summary": info.get("longBusinessSummary"),
                "website": info.get("website"),
                "headquarters": {
                    "address": info.get("address1"),
                    "city": info.get("city"),
                    "state": info.get("state"),
                    "country": info.get("country"),
                    "zip": info.get("zip")
                },
                "employees": info.get("fullTimeEmployees"),
                "market_data": {
                    "market_cap": info.get("marketCap"),
                    "currency": info.get("currency"),
                    "exchange": info.get("exchange")
                }
            }
            
            return {
                "profile": profile,
                "data_type": "company_profile"
            }
            
        except Exception as e:
            return {"error": f"Error fetching company profile: {str(e)}"}
    
    def _get_analyst_recommendations(self, ticker) -> Dict:
        """Get analyst recommendations"""
        
        try:
            recommendations = ticker.recommendations
            if recommendations is not None and not recommendations.empty:
                # Convert to serializable format
                rec_data = recommendations.tail(10).to_dict('records')  
                return {
                    "recommendations": rec_data,
                    "count": len(rec_data),
                    "data_type": "analyst_recommendations"
                }
            else:
                return {"error": "No analyst recommendations available"}
                
        except Exception as e:
            return {"error": f"Error fetching recommendations: {str(e)}"}
    
    def _get_company_news(self, ticker, num_stories: int = 5) -> Dict:
        """Get company news"""
        
        try:
            news = ticker.news
            if news:
                # Limit and clean news data
                news_data = news[:num_stories]
                return {
                    "news": news_data,
                    "count": len(news_data),
                    "data_type": "company_news"
                }
            else:
                return {"error": "No news available"}
                
        except Exception as e:
            return {"error": f"Error fetching news: {str(e)}"}
    
    def _get_stock_price(self, ticker, timeframe: str) -> Dict:
        """Get historical stock price data"""
        
        try:
            if timeframe and "to" in timeframe:
                start_date, end_date = self._parse_timeframe(timeframe)
                if start_date and end_date:
                    hist = ticker.history(start=start_date, end=end_date)
                else:
                    hist = ticker.history(period="1y")
            else:
                hist = ticker.history(period="1y")
            
            if hist.empty:
                return {"error": "No price data available"}
            
            return {
                "current_price": float(hist['Close'].iloc[-1]),
                "price_history": {str(k): float(v) for k, v in hist['Close'].to_dict().items()},
                "price_change": {
                    "absolute": float(hist['Close'].iloc[-1] - hist['Close'].iloc[0]),
                    "percentage": float((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100)
                },
                "high": float(hist['High'].max()),
                "low": float(hist['Low'].min()),
                "average_volume": float(hist['Volume'].mean()),
                "data_type": "stock_price"
            }
            
        except Exception as e:
            return {"error": f"Error fetching stock price: {str(e)}"}
    
    def _parse_timeframe(self, timeframe: str):
        """Parse timeframe string into start and end dates"""
        try:
            start_str, end_str = timeframe.split(" to ")
            start_date = pd.to_datetime(start_str.strip())
            end_date = pd.to_datetime(end_str.strip())
            return start_date, end_date
        except Exception:
            pass
        return None, None


def process_task_planner_subquery(subquery: Dict) -> Dict:
    """
    Process a single subquery from your Task Planner
    """
    parser = FinancialDataParser()
    
    if subquery["tool"] == "financial_data":
        parameters = subquery["parameters"]
        result = parser.parse_task_planner_request(parameters)
        
        return {
            "subquery_id": subquery["subquery_id"],
            "output_key": subquery["output_key"],
            "result": result,
            "status": "success" if not any("error" in str(v) for v in result.get("data", {}).values()) else "partial_success"
        }
    
    return {"error": "Not a financial_data subquery"}
import pprint


if __name__ == "__main__":
    sample_subquery = {
        "subquery_id": "q1",
        "subquery": "Get Apple comprehensive financial metrics including ratios and company info",
        "tool": "financial_data",
        "depends_on": [],
        "output_key": "apple_comprehensive_analysis",
        "priority": "high",
        "parameters": {
            "ticker": "TSLA",
            "timeframe": "2022-01-01 to 2023-12-31",
            "metric": "net_income,profit_margin,pe_ratio,beta,company_info,current_price"
        }
    }
    # a = FinancialDataParser()
    result = process_task_planner_subquery(sample_subquery)
    print(result)

    print(" Result Keys:", list(result['result']['data'].keys()))