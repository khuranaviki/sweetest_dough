import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from notion_client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NotionStockManager:
    """
    Manages Notion integration for stock analysis:
    - Reads stock list from Notion database
    - Creates analysis reports in Notion
    - Updates stock status
    """
    
    def __init__(self):
        """Initialize Notion client with API token"""
        self.notion_token = os.getenv("NOTION_TOKEN")
        if not self.notion_token:
            raise ValueError("NOTION_TOKEN not found in environment variables")
        
        self.notion = Client(auth=self.notion_token)
        
        # Page IDs - you'll need to update these with your actual page IDs
        self.stock_list_page_id = os.getenv("NOTION_STOCK_LIST_PAGE_ID")
        self.reports_page_id = os.getenv("NOTION_REPORTS_PAGE_ID")
        
        if not self.stock_list_page_id or not self.reports_page_id:
            raise ValueError("NOTION_STOCK_LIST_PAGE_ID and NOTION_REPORTS_PAGE_ID must be set in .env")
    
    def get_stock_list(self) -> List[Dict]:
        """
        Read stock list from Notion database
        Returns list of stocks with ticker, company_name, sector, category
        """
        try:
            stocks = []
            all_stocks_count = 0
            pending_stocks_count = 0
            has_more = True
            start_cursor = None
            
            print("📊 Fetching all stocks from Notion database (with pagination)...")
            
            while has_more:
                # Query parameters
                query_params = {
                    "database_id": self.stock_list_page_id,
                    "page_size": 100  # Maximum page size
                }
                
                # Add cursor for pagination
                if start_cursor:
                    query_params["start_cursor"] = start_cursor
                
                # Query the stock list database
                response = self.notion.databases.query(**query_params)
                
                # Process results
                for page in response["results"]:
                    properties = page["properties"]
                    all_stocks_count += 1
                    
                    # Extract stock data from Notion properties
                    ticker = self._get_property_value(properties, "Ticker", "title")
                    company_name = self._get_property_value(properties, "Company Name", "rich_text")
                    sector = self._get_property_value(properties, "Sector", "rich_text")
                    category = self._get_property_value(properties, "Category", "rich_text")  # Add category support
                    status = self._get_property_value(properties, "Status", "rich_text")
                    
                    print(f"📊 Found stock: {ticker} | Status: {status} | Category: {category}")
                    
                    # Only add if ticker exists and status is "Pending"
                    if ticker and status and status.lower() == "pending":
                        pending_stocks_count += 1
                        stocks.append({
                            "ticker": ticker,
                            "company_name": company_name or ticker,
                            "sector": sector or "Unknown",
                            "category": category or "Unknown",  # Include category
                            "page_id": page["id"]
                        })
                
                # Check if there are more pages
                has_more = response.get("has_more", False)
                if has_more:
                    start_cursor = response.get("next_cursor")
                    print(f"📄 Fetched page with {len(response['results'])} stocks, continuing...")
            
            print(f"📋 Total stocks in database: {all_stocks_count}")
            print(f"📋 Stocks with 'Pending' status: {pending_stocks_count}")
            print(f"📋 Found {len(stocks)} stocks to analyze from Notion")
            return stocks
            
        except Exception as e:
            print(f"❌ Error reading stock list from Notion: {e}")
            return []
    
    def update_stock_status(self, page_id: str, status: str, error_message: str = None):
        """
        Update stock status in Notion (Pending -> Analyzed/Error)
        """
        try:
            update_data = {
                "properties": {
                    "Status": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": status
                                }
                            }
                        ]
                    }
                }
            }
            
            self.notion.pages.update(page_id=page_id, **update_data)
            print(f"✅ Updated stock status to: {status}")
            
        except Exception as e:
            print(f"❌ Error updating stock status: {e}")
    
    def create_analysis_report(self, analysis_result: Dict):
        """
        Create a new analysis report in Notion reports database
        """
        try:
            # First, check if reports_page_id is a database or page
            try:
                # Try to access as database first
                db_response = self.notion.databases.retrieve(database_id=self.reports_page_id)
                is_database = True
                print("✅ Reports page is a database")
            except:
                # If not a database, try to find database in the page
                try:
                    children_response = self.notion.blocks.children.list(block_id=self.reports_page_id)
                    database_id = None
                    for child in children_response.get('results', []):
                        if child.get('type') == 'child_database':
                            database_id = child.get('id')
                            break
                    
                    if database_id:
                        is_database = True
                        self.reports_page_id = database_id
                        print(f"✅ Found database in reports page: {database_id}")
                    else:
                        is_database = False
                        print("❌ No database found in reports page")
                except:
                    is_database = False
                    print("❌ Reports page not accessible")
            
            if not is_database:
                print("❌ Reports page is not a database. Please create a database in the reports page.")
                return None
            
            # Prepare the report data - matching the actual database structure
            report_data = {
                "parent": {"database_id": self.reports_page_id},
                "properties": {
                    "Date": {
                        "title": [
                            {
                                "text": {
                                    "content": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                            }
                        ]
                    },
                    "Stock": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": analysis_result.get("ticker", "Unknown")
                                }
                            }
                        ]
                    },
                    "Action": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": analysis_result.get("action", "HOLD")
                                }
                            }
                        ]
                    },
                    "Entry Price": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": analysis_result.get("entry_price", "N/A")
                                }
                            }
                        ]
                    },
                    "Target": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": analysis_result.get("target", "N/A")
                                }
                            }
                        ]
                    },
                    "Strategy": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": analysis_result.get("strategy_used", "N/A")
                                }
                            }
                        ]
                    },
                    "Category": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": analysis_result.get("category", "Unknown")
                                }
                            }
                        ]
                    },
                    "Summary": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": analysis_result.get("summary", "No summary available")
                                }
                            }
                        ]
                    }
                }
            }
            
            # Create the page in Notion
            response = self.notion.pages.create(**report_data)
            print(f"📊 Created analysis report for {analysis_result.get('ticker', 'Unknown')}")
            return response["id"]
            
        except Exception as e:
            print(f"❌ Error creating analysis report: {e}")
            return None
    
    def create_summary_report(self, all_results: List[Dict]):
        """
        Create a summary report of all analyses
        """
        try:
            # Count actions
            buy_count = sum(1 for r in all_results if r.get("action") == "BUY")
            hold_count = sum(1 for r in all_results if r.get("action") == "HOLD")
            sell_count = sum(1 for r in all_results if r.get("action") == "SELL")
            
            # Create summary content
            summary_content = f"""
# 📈 Stock Analysis Summary Report
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Analysis Overview
- **Total Stocks Analyzed**: {len(all_results)}
- **BUY Recommendations**: {buy_count}
- **HOLD Recommendations**: {hold_count}
- **SELL Recommendations**: {sell_count}

## 🎯 Key Insights
"""
            
            # Add top recommendations
            buy_stocks = [r for r in all_results if r.get("action") == "BUY"]
            if buy_stocks:
                summary_content += "\n### 🚀 Top BUY Recommendations:\n"
                for stock in buy_stocks[:5]:  # Top 5
                    summary_content += f"- **{stock.get('ticker')}**: {stock.get('entry_price')} → {stock.get('target')} ({stock.get('strategy_used')})\n"
            
            # Add detailed results
            summary_content += "\n### 📋 Detailed Results:\n"
            for result in all_results:
                summary_content += f"- **{result.get('ticker')}**: {result.get('action')} | Entry: {result.get('entry_price')} | Target: {result.get('target')}\n"
            
            # Create the summary page
            summary_data = {
                "parent": {"database_id": self.reports_page_id},
                "properties": {
                    "Date": {
                        "title": [
                            {
                                "text": {
                                    "content": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                            }
                        ]
                    },
                    "Stock": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": f"Summary Report - {datetime.now().strftime('%Y-%m-%d')}"
                                }
                            }
                        ]
                    },
                    "Action": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": "SUMMARY"
                                }
                            }
                        ]
                    },
                    "Summary": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": summary_content
                                }
                            }
                        ]
                    }
                }
            }
            
            response = self.notion.pages.create(**summary_data)
            print(f"📋 Created summary report with {len(all_results)} analyses")
            return response["id"]
            
        except Exception as e:
            print(f"❌ Error creating summary report: {e}")
            return None
    
    def get_all_stocks(self) -> List[Dict]:
        """
        Read all stocks from Notion database regardless of status
        Returns list of stocks with ticker, company_name, sector, category
        """
        try:
            stocks = []
            all_stocks_count = 0
            has_more = True
            start_cursor = None
            
            print("📊 Fetching all stocks from Notion database (with pagination)...")
            
            while has_more:
                # Query parameters
                query_params = {
                    "database_id": self.stock_list_page_id,
                    "page_size": 100  # Maximum page size
                }
                
                # Add cursor for pagination
                if start_cursor:
                    query_params["start_cursor"] = start_cursor
                
                # Query the stock list database
                response = self.notion.databases.query(**query_params)
                
                # Process results
                for page in response["results"]:
                    properties = page["properties"]
                    all_stocks_count += 1
                    
                    # Extract stock data from Notion properties
                    ticker = self._get_property_value(properties, "Ticker", "title")
                    company_name = self._get_property_value(properties, "Company Name", "rich_text")
                    sector = self._get_property_value(properties, "Sector", "rich_text")
                    category = self._get_property_value(properties, "Category", "rich_text")
                    status = self._get_property_value(properties, "Status", "rich_text")
                    
                    # Add all stocks with valid ticker
                    if ticker:
                        stocks.append({
                            "ticker": ticker,
                            "company_name": company_name or ticker,
                            "sector": sector or "Unknown",
                            "category": category or "Unknown",
                            "status": status or "Unknown",
                            "page_id": page["id"]
                        })
                
                # Check if there are more pages
                has_more = response.get("has_more", False)
                if has_more:
                    start_cursor = response.get("next_cursor")
                    print(f"📄 Fetched page with {len(response['results'])} stocks, continuing...")
            
            print(f"📋 Total stocks in database: {all_stocks_count}")
            print(f"📋 Found {len(stocks)} stocks with valid tickers")
            return stocks
            
        except Exception as e:
            print(f"❌ Error reading all stocks from Notion: {e}")
            return []
    
    def _get_property_value(self, properties: Dict, property_name: str, property_type: str) -> Optional[str]:
        """
        Helper method to extract property values from Notion response
        """
        try:
            if property_name not in properties:
                return None
            
            prop = properties[property_name]
            
            if property_type == "rich_text":
                if prop["rich_text"]:
                    return prop["rich_text"][0]["text"]["content"]
                return None
            elif property_type == "title":
                if prop["title"]:
                    return prop["title"][0]["text"]["content"]
                return None
            elif property_type == "select":
                if prop["select"]:
                    return prop["select"]["name"]
                return None
            
            return None
            
        except Exception as e:
            print(f"❌ Error extracting property {property_name}: {e}")
            return None

    def _create_summary_text(self, analysis_result: Dict) -> str:
        """Create summary text for Notion report"""
        try:
            # Extract key information
            action = analysis_result.get('action', 'HOLD')
            strategy = analysis_result.get('strategy_used', 'Conservative Analysis')
            entry = analysis_result.get('entry_price', 'Not Available')
            target = analysis_result.get('target_price', 'Not Available')
            confidence = analysis_result.get('confidence_level', 'N/A')
            fundamental_reasons = analysis_result.get('fundamental_reasons', 'Not Available')
            
            # Create summary
            summary = f"**Action:** {action}\n"
            summary += f"**Strategy:** {strategy}\n"
            summary += f"**Entry Range:** {entry}\n"
            summary += f"**Target:** {target}\n"
            summary += f"**Confidence:** {confidence}\n"
            
            # Add fundamental reasoning if available
            if fundamental_reasons and fundamental_reasons != "Not Available":
                summary += f"**Fundamental Reasons:** {fundamental_reasons}\n"
            
            # Add strategy performance if available
            if 'strategy_performance' in analysis_result:
                strategy_perf = analysis_result['strategy_performance']
                if strategy_perf.get('eligible', False) and 'recommended_performance' in strategy_perf:
                    perf = strategy_perf['recommended_performance']
                    success_rate = perf.get('success_rate', 0)
                    total_signals = perf.get('total_signals', 0)
                    avg_gain = perf.get('avg_gain', 0)
                    avg_hold_days = perf.get('avg_hold_days', 0)
                    
                    summary += f"**Strategy Performance:** {success_rate:.1f}% success rate ({total_signals} signals)\n"
                    summary += f"**Avg Gain:** {avg_gain:.1f}% | **Avg Hold:** {avg_hold_days:.0f} days\n"
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Error creating summary text: {e}")
            return f"Analysis completed: {action} recommendation using {strategy}"

# Example usage
if __name__ == "__main__":
    try:
        # Test the Notion integration
        notion_manager = NotionStockManager()
        
        # Get stock list
        stocks = notion_manager.get_stock_list()
        print(f"Found {len(stocks)} stocks to analyze")
        
        if stocks:
            print("Sample stocks:")
            for i, stock in enumerate(stocks[:3], 1):
                print(f"  {i}. {stock['ticker']} - {stock['company_name']} ({stock['sector']})")
        
        # Example analysis result
        example_result = {
            "ticker": "RELIANCE.NS",
            "action": "BUY",
            "entry_price": "₹2,400-₹2,450",
            "target": "₹2,650",
            "strategy_used": "SMA Strategy",
            "summary": "Strong technical breakout with improving fundamentals"
        }
        
        # Create a report
        report_id = notion_manager.create_analysis_report(example_result)
        print(f"Created report with ID: {report_id}")
        
    except Exception as e:
        print(f"❌ Error testing Notion integration: {e}") 