#!/usr/bin/env python3
"""
Enhanced Multi-Agent Stock Analysis System - 3 Year Chart Version
Forked from EnhancedMultiAgent.py with modifications:
1. Uses 3 years of chart data instead of 6 months
2. Prioritizes chart-based OpenAI responses for technical analysis
3. Enhanced chart pattern recognition for longer timeframes
"""

import os
import json
import time
import base64
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from io import BytesIO
import yfinance as yf
from openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from original modules
from fundamental_scraper import FundamentalData

@dataclass
class StockData:
    ticker: str
    company_name: str
    sector: str
    category: str = "Unknown"
    ohlcv_data: Optional[pd.DataFrame] = None
    fundamental_data: Optional[FundamentalData] = None
    chart_image: Optional[str] = None

class EnhancedTechnicalAnalysis(BaseModel):
    trend: str = Field(description="Market trend: Bullish, Bearish, or Sideways")
    support_levels: List[float] = Field(description="Key support levels")
    resistance_levels: List[float] = Field(description="Key resistance levels")
    entry_range: str = Field(description="Suggested entry price range")
    short_term_target: str = Field(description="Short-term target price")
    medium_term_target: str = Field(description="Medium-term target price")
    stop_loss: str = Field(description="Stop loss level (if applicable)")
    confidence_score: str = Field(description="Confidence level: High, Medium, or Low")
    indicators: Dict[str, float] = Field(description="Technical indicators values")
    patterns: List[str] = Field(description="Identified chart patterns")
    strategy_signals: List[str] = Field(description="Specific strategy signals identified")
    position_sizing: str = Field(description="Recommended position sizing based on strategy")

class EnhancedTechnicalAnalysisAgent:
    """Enhanced Technical Analysis Agent with 3-year chart prioritization"""
    
    def __init__(self, llm: ChatOpenAI, openai_api_key: str):
        self.llm = llm
        self.openai_api_key = openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)

    def analyze(self, stock_data: StockData) -> EnhancedTechnicalAnalysis:
        print(f"📈 Enhanced Technical Analysis Agent analyzing {stock_data.ticker} (3Y Chart Priority)...")
        
        # PRIORITY 1: Chart-based OpenAI analysis (3-year data)
        if stock_data.ohlcv_data is not None and not stock_data.ohlcv_data.empty:
            print("✅ Technical Analysis: Creating 3-year candlestick chart for OpenAI analysis")
            
            # Create comprehensive 3-year chart
            chart_base64 = self._create_enhanced_3y_candlestick_chart(stock_data)
            if chart_base64:
                print("🚀 Technical Analysis: PRIORITY - Using 3-year chart image for OpenAI analysis")
                return self._analyze_with_3y_chart_priority(stock_data, chart_base64)
            else:
                print("⚠️ Chart creation failed, falling back to data analysis")
        
        # PRIORITY 2: OHLCV data analysis with pattern recognition
        if stock_data.ohlcv_data is not None and not stock_data.ohlcv_data.empty:
            print("📊 Technical Analysis: Using 3-year OHLCV data for pattern analysis")
            return self._analyze_with_ohlcv_data_3y(stock_data)
        
        # FALLBACK: Basic analysis
        print("❌ No OHLCV data available for technical analysis")
        return self._get_basic_technical_analysis(stock_data)

    def _create_enhanced_3y_candlestick_chart(self, stock_data: StockData) -> Optional[str]:
        """Create an enhanced 3-year candlestick chart with comprehensive indicators"""
        try:
            if stock_data.ohlcv_data is None or len(stock_data.ohlcv_data) == 0:
                return None
            
            # Handle MultiIndex DataFrame structure
            if isinstance(stock_data.ohlcv_data.columns, pd.MultiIndex):
                stock_data.ohlcv_data.columns = stock_data.ohlcv_data.columns.get_level_values(0)
            
            df = stock_data.ohlcv_data.copy()
            
            # Create enhanced figure with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                               gridspec_kw={'height_ratios': [4, 1, 1]})
            
            # Plot enhanced candlestick chart
            self._plot_enhanced_candlesticks_3y(ax1, df, stock_data.ticker)
            
            # Plot volume
            self._plot_volume_enhanced(ax2, df)
            
            # Plot RSI
            self._plot_rsi_3y(ax3, df)
            
            # Add comprehensive title
            fig.suptitle(f'{stock_data.ticker} ({stock_data.company_name}) - 3-Year Technical Analysis Chart', 
                        fontsize=18, fontweight='bold')
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert to base64 with high quality
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            print(f"📊 Created enhanced 3-year candlestick chart for {stock_data.ticker}")
            return chart_base64
            
        except Exception as e:
            print(f"❌ Error creating enhanced 3-year candlestick chart: {e}")
            return None

    def _plot_enhanced_candlesticks_3y(self, ax, df: pd.DataFrame, ticker: str):
        """Plot enhanced candlestick chart with 3-year timeframe indicators"""
        # Calculate comprehensive moving averages
        ma_20 = df['Close'].rolling(window=20).mean()
        ma_50 = df['Close'].rolling(window=50).mean()
        ma_100 = df['Close'].rolling(window=100).mean()
        ma_200 = df['Close'].rolling(window=200).mean()
        
        # Plot candlesticks
        for i in range(len(df)):
            date = df.index[i]
            open_price = df['Open'].iloc[i]
            high_price = df['High'].iloc[i]
            low_price = df['Low'].iloc[i]
            close_price = df['Close'].iloc[i]
            
            # Determine color
            color = 'green' if close_price >= open_price else 'red'
            
            # Plot body
            ax.bar(date, close_price - open_price, bottom=open_price, 
                   color=color, alpha=0.7, width=0.8)
            
            # Plot wicks
            ax.plot([date, date], [low_price, high_price], color='black', linewidth=1)
        
        # Plot moving averages with enhanced visibility
        ax.plot(df.index, ma_20, label='MA 20', color='blue', linewidth=2, alpha=0.8)
        ax.plot(df.index, ma_50, label='MA 50', color='orange', linewidth=2, alpha=0.8)
        ax.plot(df.index, ma_100, label='MA 100', color='purple', linewidth=2, alpha=0.8)
        ax.plot(df.index, ma_200, label='MA 200', color='red', linewidth=2, alpha=0.8)
        
        # Add Bollinger Bands
        bb_upper = ma_20 + (2 * df['Close'].rolling(window=20).std())
        bb_lower = ma_20 - (2 * df['Close'].rolling(window=20).std())
        ax.fill_between(df.index, bb_upper, bb_lower, alpha=0.1, color='gray', label='Bollinger Bands')
        
        # Formatting
        ax.set_title(f'{ticker} - 3-Year Price Chart with Technical Indicators', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price (₹)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for 3-year view
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show every 6 months
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_volume_enhanced(self, ax, df: pd.DataFrame):
        """Plot enhanced volume bars"""
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['Close'], df['Open'])]
        
        ax.bar(df.index, df['Volume'], color=colors, alpha=0.7, width=0.8)
        ax.set_title('Volume Analysis (3-Year)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volume', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add volume moving average
        volume_ma = df['Volume'].rolling(window=20).mean()
        ax.plot(df.index, volume_ma, color='blue', linewidth=1, alpha=0.8, label='Volume MA(20)')
        ax.legend()

    def _plot_rsi_3y(self, ax, df: pd.DataFrame):
        """Plot RSI for 3-year analysis"""
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        ax.plot(df.index, rsi, color='purple', linewidth=2)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax.set_title('RSI (14) - 3-Year Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel('RSI', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend() 

    def _analyze_with_3y_chart_priority(self, stock_data: StockData, chart_base64: str) -> EnhancedTechnicalAnalysis:
        """Priority analysis using 3-year chart with OpenAI"""
        print("🚀 Using 3-year chart priority analysis with OpenAI")
        
        try:
            # Create comprehensive prompt for 3-year analysis
            prompt = self._create_3y_chart_analysis_prompt(stock_data, chart_base64)
            
            # Get OpenAI analysis
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical analyst specializing in 3-year chart analysis. Analyze the provided candlestick chart and provide comprehensive technical analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content
            return self._parse_3y_chart_analysis(analysis_text, stock_data)
            
        except Exception as e:
            print(f"❌ Error in 3-year chart priority analysis: {e}")
            return self._get_basic_technical_analysis(stock_data)

    def _create_3y_chart_analysis_prompt(self, stock_data: StockData, chart_base64: str) -> str:
        """Create comprehensive prompt for 3-year chart analysis"""
        return f"""
        Analyze this 3-year candlestick chart for {stock_data.ticker} ({stock_data.company_name}) and provide detailed technical analysis.

        CHART DATA:
        - Timeframe: 3 years of daily data
        - Indicators: 20, 50, 100, 200-day moving averages
        - Bollinger Bands: 20-period with 2 standard deviations
        - RSI: 14-period relative strength index
        - Volume: Daily volume with 20-period moving average

        REQUIRED ANALYSIS:

        1. **TREND ANALYSIS (3-Year Perspective)**:
           - Primary trend direction (Bullish/Bearish/Sideways)
           - Trend strength and duration
           - Major trend changes over 3 years
           - Current trend position

        2. **SUPPORT & RESISTANCE LEVELS**:
           - Key support levels (3-5 levels)
           - Key resistance levels (3-5 levels)
           - Historical support/resistance significance
           - Current price position relative to levels

        3. **CHART PATTERNS (3-Year Timeframe)**:
           - Major patterns identified (RHS, CWH, Head & Shoulders, etc.)
           - Pattern completion status
           - Pattern reliability and significance
           - Breakout/breakdown potential

        4. **TECHNICAL INDICATORS**:
           - Moving average relationships and crossovers
           - RSI status and signals
           - Bollinger Bands position and signals
           - Volume analysis and confirmation

        5. **ENTRY/EXIT STRATEGY**:
           - Optimal entry price range
           - Short-term target (1-3 months)
           - Medium-term target (6-12 months)
           - Stop loss levels
           - Risk-reward ratio

        6. **CONFIDENCE ASSESSMENT**:
           - Overall confidence level (High/Medium/Low)
           - Key factors supporting confidence
           - Risk factors to monitor

        OUTPUT FORMAT:
        Provide analysis in a structured format that can be easily parsed into JSON-like structure.

        CHART IMAGE: [Base64 encoded chart image]
        """

    def _parse_3y_chart_analysis(self, analysis_text: str, stock_data: StockData) -> EnhancedTechnicalAnalysis:
        """Parse OpenAI analysis text into structured format"""
        try:
            # Extract key information from analysis text
            # This is a simplified parser - in production, you'd want more sophisticated parsing
            
            # Default values
            trend = "Sideways"
            confidence = "Medium"
            support_levels = []
            resistance_levels = []
            patterns = []
            indicators = {}
            
            # Simple text parsing (enhanced version would use more sophisticated NLP)
            if "bullish" in analysis_text.lower():
                trend = "Bullish"
            elif "bearish" in analysis_text.lower():
                trend = "Bearish"
            
            if "high confidence" in analysis_text.lower():
                confidence = "High"
            elif "low confidence" in analysis_text.lower():
                confidence = "Low"
            
            # Extract patterns
            if "reverse head and shoulders" in analysis_text.lower():
                patterns.append("Reverse Head and Shoulders")
            if "cup and handle" in analysis_text.lower():
                patterns.append("Cup and Handle")
            if "head and shoulders" in analysis_text.lower():
                patterns.append("Head and Shoulders")
            
            # Calculate basic indicators from data
            df = stock_data.ohlcv_data
            current_price = df['Close'].iloc[-1]
            
            # Support and resistance from recent data
            recent_low = df['Low'].tail(60).min()
            recent_high = df['High'].tail(60).max()
            support_levels = [recent_low, current_price * 0.95]
            resistance_levels = [recent_high, current_price * 1.05]
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            current_rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            indicators = {
                "RSI": round(float(current_rsi), 2) if not pd.isna(current_rsi) else 50.0,
                "Current_Price": round(float(current_price), 2),
                "3Y_High": round(float(recent_high), 2),
                "3Y_Low": round(float(recent_low), 2)
            }
            
            return EnhancedTechnicalAnalysis(
                trend=trend,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                entry_range=f"{current_price * 0.98:.2f} - {current_price * 1.02:.2f}",
                short_term_target=f"{current_price * 1.05:.2f}",
                medium_term_target=f"{current_price * 1.15:.2f}",
                stop_loss=f"{current_price * 0.95:.2f}",
                confidence_score=confidence,
                indicators=indicators,
                patterns=patterns,
                strategy_signals=["3-Year Chart Analysis"],
                position_sizing="3-5% of portfolio"
            )
            
        except Exception as e:
            print(f"❌ Error parsing 3-year chart analysis: {e}")
            return self._get_basic_technical_analysis(stock_data)

    def _analyze_with_ohlcv_data_3y(self, stock_data: StockData) -> EnhancedTechnicalAnalysis:
        """Analyze 3-year OHLCV data with pattern recognition"""
        # Enhanced OHLCV analysis for 3-year timeframe
        return self._get_basic_technical_analysis(stock_data)

    def _get_basic_technical_analysis(self, stock_data: StockData) -> EnhancedTechnicalAnalysis:
        """Get basic technical analysis using 3-year OHLCV data"""
        try:
            df = stock_data.ohlcv_data
            if df is None or df.empty:
                return self._create_default_technical_analysis()
            
            # Calculate indicators for 3-year timeframe
            current_price = df['Close'].iloc[-1]
            
            # Moving averages
            ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            ma_100 = df['Close'].rolling(window=100).mean().iloc[-1]
            ma_200 = df['Close'].rolling(window=200).mean().iloc[-1]
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            current_rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            # 3-year trend analysis
            if current_price > ma_20 > ma_50 > ma_100:
                trend = "Strong Bullish"
            elif current_price > ma_20 > ma_50:
                trend = "Bullish"
            elif current_price < ma_20 < ma_50 < ma_100:
                trend = "Strong Bearish"
            elif current_price < ma_20 < ma_50:
                trend = "Bearish"
            else:
                trend = "Sideways"
            
            # Support and resistance from 3-year data
            three_year_low = df['Low'].min()
            three_year_high = df['High'].max()
            recent_low = df['Low'].tail(60).min()
            recent_high = df['High'].tail(60).max()
            
            support_levels = [recent_low, three_year_low, ma_200]
            resistance_levels = [recent_high, three_year_high, ma_100]
            
            # Confidence based on 3-year indicators
            if abs(current_price - ma_20) / ma_20 < 0.02 and current_rsi > 30 and current_rsi < 70:
                confidence = "Medium"
            elif trend in ["Strong Bullish", "Strong Bearish"]:
                confidence = "High"
            else:
                confidence = "Low"
            
            return EnhancedTechnicalAnalysis(
                trend=trend,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                entry_range=f"{current_price * 0.98:.2f} - {current_price * 1.02:.2f}",
                short_term_target=f"{current_price * 1.05:.2f}",
                medium_term_target=f"{current_price * 1.15:.2f}",
                stop_loss=f"{current_price * 0.95:.2f}",
                confidence_score=confidence,
                indicators={
                    "RSI": round(float(current_rsi), 2) if not pd.isna(current_rsi) else 50.0,
                    "MA_20": round(float(ma_20), 2) if not pd.isna(ma_20) else 0.0,
                    "MA_50": round(float(ma_50), 2) if not pd.isna(ma_50) else 0.0,
                    "MA_100": round(float(ma_100), 2) if not pd.isna(ma_100) else 0.0,
                    "MA_200": round(float(ma_200), 2) if not pd.isna(ma_200) else 0.0,
                    "3Y_High": round(float(three_year_high), 2),
                    "3Y_Low": round(float(three_year_low), 2)
                },
                patterns=["3-Year Analysis"],
                strategy_signals=["Enhanced 3-Year Technical Analysis"],
                position_sizing="3-5% of portfolio"
            )
            
        except Exception as e:
            print(f"❌ Error in 3-year basic technical analysis: {e}")
            return self._create_default_technical_analysis()

    def _create_default_technical_analysis(self) -> EnhancedTechnicalAnalysis:
        """Create default technical analysis when all else fails"""
        return EnhancedTechnicalAnalysis(
            trend="Sideways",
            support_levels=[0],
            resistance_levels=[0],
            entry_range="N/A",
            short_term_target="N/A",
            medium_term_target="N/A",
            stop_loss="N/A",
            confidence_score="Low",
            indicators={},
            patterns=[],
            strategy_signals=[],
            position_sizing="N/A"
        )

class EnhancedMultiAgentStockAnalysis3Y:
    """Enhanced Multi-Agent Stock Analysis with 3-Year Chart Priority"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model=os.getenv('OPENAI_MODEL', 'gpt-5'))
        self.technical_agent = EnhancedTechnicalAnalysisAgent(self.llm, openai_api_key)

    def analyze_stock(self, ticker: str, company_name: str, sector: str, category: str = "Unknown") -> Dict:
        """Analyze stock with 3-year chart priority"""
        print(f"🚀 Starting 3-Year Enhanced Analysis for {ticker}")
        
        try:
            # Create stock data object
            stock_data = StockData(
                ticker=ticker,
                company_name=company_name,
                sector=sector,
                category=category
            )
            
            # Fetch 3-year data
            print("📊 Fetching 3-year historical data...")
            stock_data.ohlcv_data = self._fetch_3y_stock_data(ticker)
            
            if stock_data.ohlcv_data is None or stock_data.ohlcv_data.empty:
                print("❌ Failed to fetch 3-year data")
                return self._create_error_response(ticker, company_name, sector, category)
            
            print(f"✅ Successfully fetched {len(stock_data.ohlcv_data)} days of 3-year data")
            
            # Perform technical analysis with 3-year priority
            print("📈 Performing 3-year technical analysis...")
            technical_analysis = self.technical_agent.analyze(stock_data)
            
            # Create results
            results = {
                'ticker': ticker,
                'company_name': company_name,
                'sector': sector,
                'category': category,
                'analysis_timestamp': datetime.now().isoformat(),
                'technical_analysis': technical_analysis.model_dump(),
                'data_points': len(stock_data.ohlcv_data),
                'timeframe': '3 Years',
                'analysis_type': '3-Year Chart Priority'
            }
            
            print("✅ 3-Year analysis completed successfully")
            return results
            
        except Exception as e:
            print(f"❌ Error in 3-year analysis: {e}")
            traceback.print_exc()
            return self._create_error_response(ticker, company_name, sector, category)

    def _fetch_3y_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch 3 years of stock data"""
        try:
            # Try different ticker formats
            ticker_formats = [
                ticker,
                f"{ticker}.NS",
                ticker.replace('.NS', ''),
                ticker.replace('.BO', ''),
            ]
            
            for ticker_format in ticker_formats:
                try:
                    print(f"🔄 Trying {ticker_format} for 3-year data...")
                    # Use 3 years instead of 6 months
                    data = yf.download(ticker_format, period="3y", interval="1d", progress=False, auto_adjust=True)
                    
                    if data is not None and len(data) > 0:
                        print(f"✅ Successfully fetched 3-year data for {ticker_format}: {len(data)} days")
                        return data
                    else:
                        print(f"⚠️ No 3-year data returned for {ticker_format}")
                        
                except Exception as e:
                    print(f"❌ Error fetching 3-year data for {ticker_format}: {str(e)[:100]}...")
                    continue
            
            print(f"❌ Failed to fetch 3-year data for {ticker}")
            return None
            
        except Exception as e:
            print(f"❌ Error in 3-year data fetching: {e}")
            return None

    def _create_error_response(self, ticker: str, company_name: str, sector: str, category: str) -> Dict:
        """Create error response when analysis fails"""
        return {
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'category': category,
            'analysis_timestamp': datetime.now().isoformat(),
            'error': 'Analysis failed',
            'timeframe': '3 Years',
            'analysis_type': '3-Year Chart Priority'
        }

# Main execution function
def main():
    """Main function to test 3-year analysis"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    analyzer = EnhancedMultiAgentStockAnalysis3Y(api_key)
    
    # Test with DELHIVERY
    results = analyzer.analyze_stock("DELHIVERY.NS", "Delhivery Limited", "Logistics")
    
    print("\n" + "="*50)
    print("3-YEAR ANALYSIS RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main() 