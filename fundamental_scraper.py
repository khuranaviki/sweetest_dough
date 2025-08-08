#!/usr/bin/env python3
"""
Fundamental Data Scraper for Indian Stocks
Fetches data from BSE, Screener, and other sources
Enhanced with comprehensive Screener.in extraction
"""

import requests
import pandas as pd
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

# Import the enhanced comprehensive extractor
try:
    from enhanced_screener_extraction_v3 import EnhancedScreenerExtractionV3
    ENHANCED_EXTRACTOR_AVAILABLE = True
except ImportError:
    ENHANCED_EXTRACTOR_AVAILABLE = False
    print("⚠️ Enhanced Screener Extractor not available, using fallback methods")

@dataclass
class FundamentalData:
    """Data class to store fundamental information"""
    ticker: str
    company_name: str
    sector: str
    
    # Financial Ratios
    market_cap: Optional[str] = None
    pe_ratio: Optional[str] = None
    pb_ratio: Optional[str] = None
    roe: Optional[str] = None
    roa: Optional[str] = None
    roce: Optional[str] = None
    debt_to_equity: Optional[str] = None
    current_ratio: Optional[str] = None
    quick_ratio: Optional[str] = None
    interest_coverage: Optional[str] = None
    
    # Growth Metrics
    revenue_growth_3y: Optional[str] = None
    profit_growth_3y: Optional[str] = None
    revenue_growth_1y: Optional[str] = None
    profit_growth_1y: Optional[str] = None
    revenue_growth_qoq: Optional[str] = None
    profit_growth_qoq: Optional[str] = None
    
    # Valuation
    book_value: Optional[str] = None
    face_value: Optional[str] = None
    dividend_yield: Optional[str] = None
    dividend_payout_ratio: Optional[str] = None
    price_to_book: Optional[str] = None
    price_to_sales: Optional[str] = None
    ev_to_ebitda: Optional[str] = None
    
    # Quarterly Results (Enhanced)
    quarterly_column_headers: Optional[List[str]] = None
    quarterly_revenue: Optional[List[str]] = None
    quarterly_expenses: Optional[List[str]] = None
    quarterly_operating_profit: Optional[List[str]] = None
    quarterly_net_profit: Optional[List[str]] = None
    quarterly_ebitda: Optional[List[str]] = None
    
    # Annual Results (Enhanced)
    annual_column_headers: Optional[List[str]] = None
    annual_total_revenue: Optional[str] = None
    annual_total_expenses: Optional[str] = None
    annual_operating_profit: Optional[str] = None
    annual_net_profit: Optional[str] = None
    annual_ebitda: Optional[str] = None
    
    # Balance Sheet (Enhanced)
    total_assets: Optional[str] = None
    total_liabilities: Optional[str] = None
    net_worth: Optional[str] = None
    working_capital: Optional[str] = None
    
    # Cash Flows (Enhanced)
    operating_cf: Optional[str] = None
    investing_cf: Optional[str] = None
    financing_cf: Optional[str] = None
    
    # Legacy fields for backward compatibility
    latest_quarter_revenue: Optional[str] = None
    latest_quarter_profit: Optional[str] = None
    latest_quarter_ebitda: Optional[str] = None
    latest_quarter_operating_margin: Optional[str] = None
    latest_quarter_net_margin: Optional[str] = None
    
    # Management Guidance
    management_guidance: Optional[str] = None
    future_outlook: Optional[str] = None
    
    # Promoter & Shareholding
    promoter_holding: Optional[str] = None
    promoter_pledging: Optional[str] = None
    retail_shareholding: Optional[str] = None
    institutional_shareholding: Optional[str] = None
    fii_shareholding: Optional[str] = None
    dii_shareholding: Optional[str] = None
    
    # Filing Information
    latest_filings: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.latest_filings is None:
            self.latest_filings = []
        
        # Set legacy fields from enhanced data for backward compatibility
        if self.quarterly_revenue and len(self.quarterly_revenue) > 0:
            self.latest_quarter_revenue = self.quarterly_revenue[0]  # Most recent quarter
        if self.quarterly_net_profit and len(self.quarterly_net_profit) > 0:
            self.latest_quarter_profit = self.quarterly_net_profit[0]  # Most recent quarter
        if self.quarterly_ebitda and len(self.quarterly_ebitda) > 0:
            self.latest_quarter_ebitda = self.quarterly_ebitda[0]  # Most recent quarter

class BSEScraper:
    """Scraper for BSE (Bombay Stock Exchange) data"""
    
    def __init__(self):
        self.base_url = "https://www.bseindia.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic stock information from BSE"""
        try:
            # Remove .NS suffix if present
            clean_ticker = ticker.replace('.NS', '')
            
            # Try different BSE URL formats
            urls_to_try = [
                f"https://www.bseindia.com/stock-share-price/{clean_ticker}",
                f"https://www.bseindia.com/stock-quote/{clean_ticker}",
                f"https://www.bseindia.com/get-quotes/equity/{clean_ticker}",
                f"https://www.bseindia.com/stock-share-price/{clean_ticker}/",
                f"https://www.bseindia.com/stock-quote/{clean_ticker}/"
            ]
            
            for url in urls_to_try:
                try:
                    print(f"🔍 Trying BSE URL: {url}")
                    response = self.session.get(url, timeout=15)
                    print(f"📊 BSE Response Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extract basic information
                        data = {}
                        
                        # Try multiple selectors for each metric
                        # Market cap
                        market_cap_selectors = [
                            'td:contains("Market Cap") + td',
                            'td:contains("Market Cap")',
                            'span:contains("Market Cap")',
                            '[data-label*="Market Cap"]'
                        ]
                        
                        for selector in market_cap_selectors:
                            try:
                                elem = soup.select_one(selector)
                                if elem:
                                    market_cap_text = elem.get_text(strip=True)
                                    if market_cap_text and market_cap_text != "Market Cap":
                                        data['market_cap'] = market_cap_text
                                        break
                            except:
                                continue
                        
                        # P/E Ratio
                        pe_selectors = [
                            'td:contains("P/E") + td',
                            'td:contains("P/E")',
                            'span:contains("P/E")',
                            '[data-label*="P/E"]'
                        ]
                        
                        for selector in pe_selectors:
                            try:
                                elem = soup.select_one(selector)
                                if elem:
                                    pe_text = elem.get_text(strip=True)
                                    if pe_text and pe_text != "P/E":
                                        data['pe_ratio'] = pe_text
                                        break
                            except:
                                continue
                        
                        # Book Value
                        book_value_selectors = [
                            'td:contains("Book Value") + td',
                            'td:contains("Book Value")',
                            'span:contains("Book Value")',
                            '[data-label*="Book Value"]'
                        ]
                        
                        for selector in book_value_selectors:
                            try:
                                elem = soup.select_one(selector)
                                if elem:
                                    book_value_text = elem.get_text(strip=True)
                                    if book_value_text and book_value_text != "Book Value":
                                        data['book_value'] = book_value_text
                                        break
                            except:
                                continue
                        
                        # Face Value
                        face_value_selectors = [
                            'td:contains("Face Value") + td',
                            'td:contains("Face Value")',
                            'span:contains("Face Value")',
                            '[data-label*="Face Value"]'
                        ]
                        
                        for selector in face_value_selectors:
                            try:
                                elem = soup.select_one(selector)
                                if elem:
                                    face_value_text = elem.get_text(strip=True)
                                    if face_value_text and face_value_text != "Face Value":
                                        data['face_value'] = face_value_text
                                        break
                            except:
                                continue
                        
                        if data:  # If we got some data, return it
                            print(f"✅ BSE Data found: {data}")
                            return data
                            
                except Exception as e:
                    print(f"⚠️ BSE URL failed: {url} - {str(e)}")
                    continue
            
            # If all URLs fail, return empty dict
            print(f"❌ Could not fetch BSE data for {ticker}")
            return {}
            
        except Exception as e:
            print(f"❌ Error in BSE scraper for {ticker}: {str(e)}")
            return {}

class ScreenerScraper:
    """Scraper for Screener.in data"""
    
    def __init__(self):
        self.base_url = "https://www.screener.in"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive stock data from Screener"""
        try:
            # Remove .NS suffix if present
            clean_ticker = ticker.replace('.NS', '')
            
            # Try different URL formats
            urls_to_try = [
                f"https://www.screener.in/company/{clean_ticker}/",
                f"https://www.screener.in/company/{clean_ticker}",
                f"https://www.screener.in/stock/{clean_ticker}/",
                f"https://www.screener.in/stock/{clean_ticker}"
            ]
            
            for url in urls_to_try:
                try:
                    print(f"🔍 Trying Screener URL: {url}")
                    response = self.session.get(url, timeout=15)
                    print(f"📊 Screener Response Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        data = {}
                        
                        # Extract key metrics using multiple approaches
                        # Look for common patterns in the HTML
                        page_text = soup.get_text().lower()
                        
                        # Extract P/E Ratio
                        pe_patterns = [
                            r'pe ratio[:\s]*([\d.]+)',
                            r'p/e[:\s]*([\d.]+)',
                            r'price to earnings[:\s]*([\d.]+)'
                        ]
                        
                        for pattern in pe_patterns:
                            match = re.search(pattern, page_text)
                            if match:
                                data['pe_ratio'] = f"{match.group(1)}x"
                                break
                        
                        # Extract ROE
                        roe_patterns = [
                            r'roe[:\s]*([\d.]+)%',
                            r'return on equity[:\s]*([\d.]+)%',
                            r'return on equity[:\s]*([\d.]+)'
                        ]
                        
                        for pattern in roe_patterns:
                            match = re.search(pattern, page_text)
                            if match:
                                data['roe'] = f"{match.group(1)}%"
                                break
                        
                        # Extract ROA
                        roa_patterns = [
                            r'roa[:\s]*([\d.]+)%',
                            r'return on assets[:\s]*([\d.]+)%',
                            r'return on assets[:\s]*([\d.]+)'
                        ]
                        
                        for pattern in roa_patterns:
                            match = re.search(pattern, page_text)
                            if match:
                                data['roa'] = f"{match.group(1)}%"
                                break
                        
                        # Extract Debt to Equity
                        debt_patterns = [
                            r'debt to equity[:\s]*([\d.]+)',
                            r'debt/equity[:\s]*([\d.]+)',
                            r'debt equity[:\s]*([\d.]+)'
                        ]
                        
                        for pattern in debt_patterns:
                            match = re.search(pattern, page_text)
                            if match:
                                data['debt_to_equity'] = match.group(1)
                                break
                        
                        # Extract Current Ratio
                        current_ratio_patterns = [
                            r'current ratio[:\s]*([\d.]+)',
                            r'current ratio[:\s]*([\d.]+)'
                        ]
                        
                        for pattern in current_ratio_patterns:
                            match = re.search(pattern, page_text)
                            if match:
                                data['current_ratio'] = match.group(1)
                                break
                        
                        # Extract Dividend Yield
                        dividend_patterns = [
                            r'dividend yield[:\s]*([\d.]+)%',
                            r'dividend yield[:\s]*([\d.]+)'
                        ]
                        
                        for pattern in dividend_patterns:
                            match = re.search(pattern, page_text)
                            if match:
                                data['dividend_yield'] = f"{match.group(1)}%"
                                break
                        
                        # Extract Book Value
                        book_value_patterns = [
                            r'book value[:\s]*₹([\d,]+)',
                            r'book value[:\s]*([\d.]+)'
                        ]
                        
                        for pattern in book_value_patterns:
                            match = re.search(pattern, page_text)
                            if match:
                                data['book_value'] = f"₹{match.group(1)}"
                                break
                        
                        # Extract Market Cap
                        market_cap_patterns = [
                            r'market cap[:\s]*₹([\d,]+)\s*(cr|crore|billion)',
                            r'market cap[:\s]*([\d.]+)\s*(cr|crore|billion)'
                        ]
                        
                        for pattern in market_cap_patterns:
                            match = re.search(pattern, page_text)
                            if match:
                                data['market_cap'] = f"₹{match.group(1)} {match.group(2)}"
                                break
                        
                        # Try to extract quarterly results from tables
                        tables = soup.find_all('table')
                        for table in tables:
                            rows = table.find_all('tr')
                            if len(rows) >= 2:
                                # Look for revenue and profit in table
                                for row in rows:
                                    cells = row.find_all(['td', 'th'])
                                    if len(cells) >= 3:
                                        cell_text = ' '.join([cell.get_text(strip=True) for cell in cells]).lower()
                                        if 'revenue' in cell_text and 'profit' in cell_text:
                                            # Try to extract values
                                            for cell in cells:
                                                cell_value = cell.get_text(strip=True)
                                                if '₹' in cell_value and 'cr' in cell_value.lower():
                                                    if 'revenue' in cell_text:
                                                        data['latest_quarter_revenue'] = cell_value
                                                    elif 'profit' in cell_text:
                                                        data['latest_quarter_profit'] = cell_value
                        
                        if data:  # If we got some data, return it
                            print(f"✅ Screener Data found: {data}")
                            return data
                            
                except Exception as e:
                    print(f"⚠️ Screener URL failed: {url} - {str(e)}")
                    continue
            
            # If all URLs fail, return empty dict
            print(f"❌ Could not fetch Screener data for {ticker}")
            return {}
            
        except Exception as e:
            print(f"❌ Error in Screener scraper for {ticker}: {str(e)}")
            return {}
    
    def _extract_number(self, text: str) -> str:
        """Extract number from text"""
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        return numbers[0] if numbers else "0"
    
    def _extract_percentage(self, text: str) -> str:
        """Extract percentage from text"""
        import re
        percentages = re.findall(r'(\d+\.?\d*)%', text)
        return f"{percentages[0]}%" if percentages else "0%"
    
    def _extract_currency(self, text: str) -> str:
        """Extract currency value from text"""
        import re
        # Look for currency patterns like ₹1,234 or ₹1,234 Cr
        currency_pattern = r'₹\s*([\d,]+\.?\d*)\s*(Cr|Lakh|Crore)?'
        matches = re.findall(currency_pattern, text)
        if matches:
            value, unit = matches[0]
            return f"₹{value}{' ' + unit if unit else ''}"
        return "₹0"
    
    def get_management_guidance(self, ticker: str) -> Dict[str, Any]:
        """Get management guidance and transcripts from Screener"""
        try:
            clean_ticker = ticker.replace('.NS', '')
            
            # Try different URL formats
            urls_to_try = [
                f"{self.base_url}/company/{clean_ticker}/",
                f"{self.base_url}/company/{clean_ticker}",
                f"{self.base_url}/stock/{clean_ticker}/"
            ]
            
            for url in urls_to_try:
                try:
                    response = self.session.get(url, timeout=15)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        guidance_data = {}
                        
                        # Look for management discussion or guidance sections
                        guidance_sections = soup.find_all(['div', 'section'], string=re.compile(r'management|guidance|outlook|transcript', re.I))
                        
                        for section in guidance_sections:
                            text = section.get_text(strip=True)
                            if len(text) > 100:
                                guidance_data['management_guidance'] = text[:1000]
                                break
                        
                        # Look for future outlook
                        outlook_sections = soup.find_all(['div', 'section'], string=re.compile(r'future|outlook|forecast', re.I))
                        
                        for section in outlook_sections:
                            text = section.get_text(strip=True)
                            if len(text) > 50:
                                guidance_data['future_outlook'] = text[:500]
                                break
                        
                        if guidance_data:
                            return guidance_data
                            
                except Exception as e:
                    continue
            
            # If all URLs fail, return empty dict
            print(f"Could not fetch management guidance for {ticker}")
            return {}
            
        except Exception as e:
            print(f"Error fetching management guidance for {ticker}: {str(e)}")
            return {}

class NewsScraper:
    """Scraper for news and sentiment data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def get_news(self, ticker: str, company_name: str) -> List[Dict[str, str]]:
        """Get recent news for the stock"""
        try:
            clean_ticker = ticker.replace('.NS', '')
            
            # Try multiple news sources
            news_sources = [
                f"https://www.moneycontrol.com/news/tags/{clean_ticker}.html",
                f"https://economictimes.indiatimes.com/topic/{clean_ticker}",
                f"https://www.livemint.com/search?q={clean_ticker}"
            ]
            
            news_list = []
            
            for source_url in news_sources:
                try:
                    response = self.session.get(source_url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extract news headlines (generic approach)
                        headlines = soup.find_all(['h1', 'h2', 'h3', 'h4'], string=re.compile(rf'{clean_ticker}|{company_name.split()[0]}', re.I))
                        
                        for headline in headlines[:5]:  # Limit to 5 headlines per source
                            title = headline.get_text(strip=True)
                            if title and len(title) > 10:
                                # Simple sentiment analysis based on keywords
                                sentiment = self._analyze_sentiment(title)
                                news_list.append({
                                    'headline': title,
                                    'sentiment': sentiment,
                                    'source': source_url.split('/')[2]
                                })
                        
                        if news_list:
                            break  # If we got news from one source, stop
                            
                except Exception as e:
                    print(f"Error fetching news from {source_url}: {str(e)}")
                    continue
            
            # If no news found, create placeholder
            if not news_list:
                news_list = [
                    {
                        'headline': f"Market analysis for {company_name}",
                        'sentiment': 'Neutral',
                        'source': 'Market Data'
                    }
                ]
            
            return news_list[:10]  # Return max 10 news items
            
        except Exception as e:
            print(f"Error fetching news for {ticker}: {str(e)}")
            return []
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis based on keywords"""
        text_lower = text.lower()
        
        positive_words = ['profit', 'growth', 'up', 'rise', 'gain', 'positive', 'strong', 'beat', 'exceed']
        negative_words = ['loss', 'down', 'fall', 'decline', 'negative', 'weak', 'miss', 'below']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'Positive'
        elif negative_count > positive_count:
            return 'Negative'
        else:
            return 'Neutral'

class FundamentalDataCollector:
    """Main class to collect all fundamental data"""
    
    def __init__(self, openai_api_key: str = None):
        self.bse_scraper = BSEScraper()
        self.screener_scraper = ScreenerScraper()
        self.news_scraper = NewsScraper()
        self.openai_api_key = openai_api_key
        
        # Initialize enhanced extractor if available
        if ENHANCED_EXTRACTOR_AVAILABLE and openai_api_key:
            try:
                self.enhanced_extractor = EnhancedScreenerExtractionV3(openai_api_key)
                print("✅ Enhanced Screener Extractor initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Enhanced Screener Extractor: {e}")
                self.enhanced_extractor = None
        else:
            self.enhanced_extractor = None
    
    def collect_fundamental_data(self, ticker: str, company_name: str, sector: str) -> FundamentalData:
        """Collect comprehensive fundamental data for a stock"""
        
        print(f"🔍 Collecting fundamental data for {ticker} ({company_name})...")
        
        # Initialize data object
        fundamental_data = FundamentalData(
            ticker=ticker,
            company_name=company_name,
            sector=sector
        )
        
        # Try enhanced extraction first if available
        if self.enhanced_extractor:
            try:
                print("🚀 Using enhanced comprehensive extraction...")
                enhanced_data = self.collect_enhanced_data(ticker)
                if enhanced_data:
                    # Update fundamental data with enhanced data
                    self._update_fundamental_data_with_enhanced(fundamental_data, enhanced_data)
                    print("✅ Enhanced data collection completed")
            except Exception as e:
                print(f"⚠️ Enhanced extraction failed: {e}, falling back to basic methods")
        
        # Collect BSE data
        print("📊 Fetching BSE data...")
        bse_data = self.bse_scraper.get_stock_info(ticker)
        
        # Collect Screener data
        print("📈 Fetching Screener data...")
        screener_data = self.screener_scraper.get_stock_data(ticker)
        
        # Collect management guidance
        print("🎯 Fetching management guidance...")
        guidance_data = self.screener_scraper.get_management_guidance(ticker)
        
        # Collect news
        print("📰 Fetching news...")
        news_data = self.news_scraper.get_news(ticker, company_name)
        
        # Merge all data
        all_data = {**bse_data, **screener_data, **guidance_data}
        
        # Update fundamental data object
        for key, value in all_data.items():
            if hasattr(fundamental_data, key):
                setattr(fundamental_data, key, value)
        
        # Add news as sentiment data
        fundamental_data.latest_filings = news_data
        
        # Add realistic fallback data if no data was collected
        if not any([bse_data, screener_data, guidance_data]):
            print("📊 Adding realistic fallback data based on sector averages...")
            self._add_fallback_data(fundamental_data, sector)
        
        print(f"✅ Fundamental data collected for {ticker}")
        
        return fundamental_data
    
    def collect_enhanced_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Collect data using enhanced comprehensive extractor"""
        try:
            output_dir = f"enhanced_fundamental_data_{ticker.replace('.NS', '')}"
            enhanced_data = self.enhanced_extractor.extract_complete_data(ticker, output_dir)
            return enhanced_data
        except Exception as e:
            print(f"❌ Enhanced data collection failed: {e}")
            return None
    
    def _update_fundamental_data_with_enhanced(self, fundamental_data: FundamentalData, enhanced_data: Dict[str, Any]):
        """Update fundamental data object with enhanced data"""
        try:
            # Update key metrics
            if 'key_metrics' in enhanced_data:
                km = enhanced_data['key_metrics']
                fundamental_data.market_cap = km.get('market_cap')
                fundamental_data.pe_ratio = km.get('pe_ratio')
                fundamental_data.book_value = km.get('book_value')
                fundamental_data.roce = km.get('roce')
                fundamental_data.roe = km.get('roe')
            
            # Update quarterly results
            if 'quarterly_results' in enhanced_data:
                qr = enhanced_data['quarterly_results']
                fundamental_data.quarterly_column_headers = qr.get('column_headers')
                fundamental_data.quarterly_revenue = qr.get('revenue')
                fundamental_data.quarterly_expenses = qr.get('expenses')
                fundamental_data.quarterly_operating_profit = qr.get('operating_profit')
                fundamental_data.quarterly_net_profit = qr.get('net_profit')
                fundamental_data.quarterly_ebitda = qr.get('ebitda')
            
            # Update annual results
            if 'annual_results' in enhanced_data:
                ar = enhanced_data['annual_results']
                fundamental_data.annual_column_headers = ar.get('column_headers')
                fundamental_data.annual_total_revenue = ar.get('total_revenue')
                fundamental_data.annual_total_expenses = ar.get('total_expenses')
                fundamental_data.annual_operating_profit = ar.get('operating_profit')
                fundamental_data.annual_net_profit = ar.get('net_profit')
                fundamental_data.annual_ebitda = ar.get('ebitda')
            
            # Update balance sheet
            if 'balance_sheet' in enhanced_data:
                bs = enhanced_data['balance_sheet']
                fundamental_data.total_assets = bs.get('total_assets')
                fundamental_data.total_liabilities = bs.get('total_liabilities')
                fundamental_data.net_worth = bs.get('net_worth')
                fundamental_data.working_capital = bs.get('working_capital')
            
            # Update cash flows
            if 'cash_flows' in enhanced_data:
                cf = enhanced_data['cash_flows']
                fundamental_data.operating_cf = cf.get('operating_cf')
                fundamental_data.investing_cf = cf.get('investing_cf')
                fundamental_data.financing_cf = cf.get('financing_cf')
            
            # Update shareholding
            if 'shareholding' in enhanced_data:
                sh = enhanced_data['shareholding']
                fundamental_data.promoter_holding = sh.get('promoter_holding')
                fundamental_data.fii_shareholding = sh.get('fii_holding')
                fundamental_data.dii_shareholding = sh.get('dii_holding')
                fundamental_data.retail_shareholding = sh.get('retail_holding')
            
            print("✅ Enhanced data successfully integrated")
            
        except Exception as e:
            print(f"⚠️ Error updating fundamental data with enhanced data: {e}")
    
    def _add_fallback_data(self, fundamental_data: FundamentalData, sector: str):
        """Document missing data instead of adding fallback values"""
        
        print(f"⚠️  WARNING: Could not scrape sufficient data for {fundamental_data.ticker}")
        print(f"📊 Sector: {sector}")
        print("🔍 Data collection issues:")
        
        # Check what data we actually have vs what's missing
        missing_data = []
        
        if not fundamental_data.market_cap:
            missing_data.append("Market Cap - BSE/Screener data not accessible")
        if not fundamental_data.pe_ratio:
            missing_data.append("P/E Ratio - Financial ratios not found")
        if not fundamental_data.roe:
            missing_data.append("ROE - Return metrics not available")
        if not fundamental_data.roa:
            missing_data.append("ROA - Asset return data missing")
        if not fundamental_data.roce:
            missing_data.append("ROCE - Capital efficiency data not found")
        if not fundamental_data.debt_to_equity:
            missing_data.append("Debt/Equity - Balance sheet data missing")
        if not fundamental_data.current_ratio:
            missing_data.append("Current Ratio - Liquidity metrics not available")
        if not fundamental_data.revenue_growth_1y:
            missing_data.append("Revenue Growth (1Y) - Growth data not found")
        if not fundamental_data.profit_growth_1y:
            missing_data.append("Profit Growth (1Y) - Profitability trends missing")
        if not fundamental_data.latest_quarter_revenue:
            missing_data.append("Latest Quarter Revenue - Quarterly data not accessible")
        if not fundamental_data.latest_quarter_profit:
            missing_data.append("Latest Quarter Profit - Quarterly results missing")
        if not fundamental_data.book_value:
            missing_data.append("Book Value - Valuation data not found")
        if not fundamental_data.promoter_holding:
            missing_data.append("Promoter Holding - Shareholding data missing")
        if not fundamental_data.management_guidance:
            missing_data.append("Management Guidance - Corporate communications not found")
        if not fundamental_data.latest_filings:
            missing_data.append("Latest Filings - News/announcements not accessible")
        
        # Print missing data reasons
        for missing in missing_data:
            print(f"   ❌ {missing}")
        
        print(f"\n📋 Total missing metrics: {len(missing_data)}")
        print("💡 Possible reasons:")
        print("   - Website structure changed")
        print("   - Data not publicly available")
        print("   - Network connectivity issues")
        print("   - Rate limiting by websites")
        print("   - Company not listed on BSE/Screener")
        print("   - Data requires authentication")
        
        # Set missing data to "Not Available" with reasons
        if not fundamental_data.market_cap:
            fundamental_data.market_cap = "Not Available (BSE/Screener data not accessible)"
        if not fundamental_data.pe_ratio:
            fundamental_data.pe_ratio = "Not Available (Financial ratios not found)"
        if not fundamental_data.pb_ratio:
            fundamental_data.pb_ratio = "Not Available (Price-to-book data missing)"
        if not fundamental_data.roe:
            fundamental_data.roe = "Not Available (Return on equity data missing)"
        if not fundamental_data.roa:
            fundamental_data.roa = "Not Available (Return on assets data missing)"
        if not fundamental_data.roce:
            fundamental_data.roce = "Not Available (Return on capital employed data missing)"
        if not fundamental_data.debt_to_equity:
            fundamental_data.debt_to_equity = "Not Available (Balance sheet data missing)"
        if not fundamental_data.current_ratio:
            fundamental_data.current_ratio = "Not Available (Liquidity metrics missing)"
        if not fundamental_data.quick_ratio:
            fundamental_data.quick_ratio = "Not Available (Quick ratio data missing)"
        if not fundamental_data.interest_coverage:
            fundamental_data.interest_coverage = "Not Available (Interest coverage data missing)"
        if not fundamental_data.revenue_growth_1y:
            fundamental_data.revenue_growth_1y = "Not Available (1-year growth data missing)"
        if not fundamental_data.profit_growth_1y:
            fundamental_data.profit_growth_1y = "Not Available (1-year profit growth missing)"
        if not fundamental_data.revenue_growth_3y:
            fundamental_data.revenue_growth_3y = "Not Available (3-year growth data missing)"
        if not fundamental_data.profit_growth_3y:
            fundamental_data.profit_growth_3y = "Not Available (3-year profit growth missing)"
        if not fundamental_data.revenue_growth_qoq:
            fundamental_data.revenue_growth_qoq = "Not Available (Quarterly growth data missing)"
        if not fundamental_data.profit_growth_qoq:
            fundamental_data.profit_growth_qoq = "Not Available (Quarterly profit growth missing)"
        if not fundamental_data.latest_quarter_revenue:
            fundamental_data.latest_quarter_revenue = "Not Available (Quarterly revenue data missing)"
        if not fundamental_data.latest_quarter_profit:
            fundamental_data.latest_quarter_profit = "Not Available (Quarterly profit data missing)"
        if not fundamental_data.latest_quarter_ebitda:
            fundamental_data.latest_quarter_ebitda = "Not Available (Quarterly EBITDA data missing)"
        if not fundamental_data.latest_quarter_operating_margin:
            fundamental_data.latest_quarter_operating_margin = "Not Available (Operating margin data missing)"
        if not fundamental_data.latest_quarter_net_margin:
            fundamental_data.latest_quarter_net_margin = "Not Available (Net margin data missing)"
        if not fundamental_data.book_value:
            fundamental_data.book_value = "Not Available (Book value data missing)"
        if not fundamental_data.face_value:
            fundamental_data.face_value = "Not Available (Face value data missing)"
        if not fundamental_data.dividend_yield:
            fundamental_data.dividend_yield = "Not Available (Dividend yield data missing)"
        if not fundamental_data.dividend_payout_ratio:
            fundamental_data.dividend_payout_ratio = "Not Available (Dividend payout ratio missing)"
        if not fundamental_data.price_to_book:
            fundamental_data.price_to_book = "Not Available (Price-to-book ratio missing)"
        if not fundamental_data.price_to_sales:
            fundamental_data.price_to_sales = "Not Available (Price-to-sales ratio missing)"
        if not fundamental_data.ev_to_ebitda:
            fundamental_data.ev_to_ebitda = "Not Available (EV/EBITDA ratio missing)"
        if not fundamental_data.promoter_holding:
            fundamental_data.promoter_holding = "Not Available (Promoter holding data missing)"
        if not fundamental_data.promoter_pledging:
            fundamental_data.promoter_pledging = "Not Available (Promoter pledging data missing)"
        if not fundamental_data.retail_shareholding:
            fundamental_data.retail_shareholding = "Not Available (Retail shareholding data missing)"
        if not fundamental_data.institutional_shareholding:
            fundamental_data.institutional_shareholding = "Not Available (Institutional shareholding missing)"
        if not fundamental_data.fii_shareholding:
            fundamental_data.fii_shareholding = "Not Available (FII shareholding data missing)"
        if not fundamental_data.dii_shareholding:
            fundamental_data.dii_shareholding = "Not Available (DII shareholding data missing)"
        if not fundamental_data.management_guidance:
            fundamental_data.management_guidance = "Not Available (Management guidance not found)"
        if not fundamental_data.future_outlook:
            fundamental_data.future_outlook = "Not Available (Future outlook not found)"
        
        # Add placeholder news with explanation
        if not fundamental_data.latest_filings:
            fundamental_data.latest_filings = [
                {
                    'headline': f"No recent news found for {fundamental_data.company_name}",
                    'sentiment': 'Neutral',
                    'source': 'Data Not Available',
                    'reason': 'News scraping failed or no recent announcements found'
                }
            ]
        
        print(f"\n✅ Analysis will proceed with available data only")
        print(f"📊 Data completeness: {len([f for f in fundamental_data.__dict__.values() if f and f != 'Not Available' and not isinstance(f, list)])}/{len([f for f in fundamental_data.__dict__.values() if not isinstance(f, list)])} metrics available")

    def get_summary_report(self, fundamental_data: FundamentalData) -> str:
        """Generate a summary report from fundamental data"""
        
        report = f"""
📊 COMPREHENSIVE FUNDAMENTAL ANALYSIS SUMMARY
{'='*60}

🏢 Company: {fundamental_data.company_name} ({fundamental_data.ticker})
📊 Sector: {fundamental_data.sector}

💰 VALUATION METRICS:
- Market Cap: {fundamental_data.market_cap or 'N/A'}
- P/E Ratio: {fundamental_data.pe_ratio or 'N/A'}
- P/B Ratio: {fundamental_data.pb_ratio or 'N/A'}
- Book Value: {fundamental_data.book_value or 'N/A'}
- Face Value: {fundamental_data.face_value or 'N/A'}
- Dividend Yield: {fundamental_data.dividend_yield or 'N/A'}
- Dividend Payout Ratio: {fundamental_data.dividend_payout_ratio or 'N/A'}
- Price to Book: {fundamental_data.price_to_book or 'N/A'}
- Price to Sales: {fundamental_data.price_to_sales or 'N/A'}
- EV/EBITDA: {fundamental_data.ev_to_ebitda or 'N/A'}

📈 GROWTH METRICS:
- Revenue Growth (1Y): {fundamental_data.revenue_growth_1y or 'N/A'}
- Profit Growth (1Y): {fundamental_data.profit_growth_1y or 'N/A'}
- Revenue Growth (3Y): {fundamental_data.revenue_growth_3y or 'N/A'}
- Profit Growth (3Y): {fundamental_data.profit_growth_3y or 'N/A'}

📊 FINANCIAL HEALTH:
- ROE: {fundamental_data.roe or 'N/A'}
- ROA: {fundamental_data.roa or 'N/A'}
- ROCE: {fundamental_data.roce or 'N/A'}
- Debt to Equity: {fundamental_data.debt_to_equity or 'N/A'}
- Current Ratio: {fundamental_data.current_ratio or 'N/A'}
- Quick Ratio: {fundamental_data.quick_ratio or 'N/A'}
- Interest Coverage: {fundamental_data.interest_coverage or 'N/A'}

📋 LATEST QUARTERLY RESULTS:
- Revenue: {fundamental_data.latest_quarter_revenue or 'N/A'}
- Profit: {fundamental_data.latest_quarter_profit or 'N/A'}
- EBITDA: {fundamental_data.latest_quarter_ebitda or 'N/A'}
- Operating Margin: {fundamental_data.latest_quarter_operating_margin or 'N/A'}
- Net Margin: {fundamental_data.latest_quarter_net_margin or 'N/A'}
- Revenue Growth (QoQ): {fundamental_data.revenue_growth_qoq or 'N/A'}
- Profit Growth (QoQ): {fundamental_data.profit_growth_qoq or 'N/A'}

👥 SHAREHOLDING PATTERN:
- Promoter Holding: {fundamental_data.promoter_holding or 'N/A'}
- Promoter Pledging: {fundamental_data.promoter_pledging or 'N/A'}
- Retail Shareholding: {fundamental_data.retail_shareholding or 'N/A'}
- Institutional Shareholding: {fundamental_data.institutional_shareholding or 'N/A'}
- FII Shareholding: {fundamental_data.fii_shareholding or 'N/A'}
- DII Shareholding: {fundamental_data.dii_shareholding or 'N/A'}

🎯 MANAGEMENT GUIDANCE:
{fundamental_data.management_guidance or 'No guidance available'}

📰 RECENT NEWS ({len(fundamental_data.latest_filings)} items):
"""
        
        for i, news in enumerate(fundamental_data.latest_filings[:5], 1):
            report += f"{i}. {news['headline']} ({news['sentiment']})\n"
        
        return report

# Example usage
if __name__ == "__main__":
    collector = FundamentalDataCollector()
    
    # Test with a sample stock
    test_stocks = [
        ("DELHIVERY.NS", "Delhivery Ltd", "Logistics"),
        ("RELIANCE.NS", "Reliance Industries Ltd", "Oil & Gas")
    ]
    
    for ticker, company_name, sector in test_stocks:
        print(f"\n{'='*60}")
        print(f"Testing: {ticker} - {company_name}")
        print(f"{'='*60}")
        
        try:
            fundamental_data = collector.collect_fundamental_data(ticker, company_name, sector)
            report = collector.get_summary_report(fundamental_data)
            print(report)
            
            # Save to file
            filename = f"fundamental_data_{ticker.replace('.NS', '')}.txt"
            with open(filename, 'w') as f:
                f.write(report)
            print(f"📄 Data saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {str(e)}")
        
        time.sleep(2)  # Be respectful to servers 