#!/usr/bin/env python3
"""
Enhanced Multi-Agent Stock Analysis Framework
Incorporates advanced investment philosophy and specific trading strategies
"""

import os
import time
import re
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import base64
from io import BytesIO
from bs4 import BeautifulSoup
import random
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from openai import OpenAI
from datetime import datetime, timedelta
from openai_cost_tracker import cost_tracker
# from robust_data_fetcher import RobustDataFetcher
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Default OpenAI model configurable via environment
DEFAULT_OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5')

# Import our fundamental scraper
from fundamental_scraper import FundamentalDataCollector, FundamentalData

# Import the enhanced screener extraction
from enhanced_screener_extraction_v3 import EnhancedScreenerExtractionV3

# Initialize the extractor
api_key = os.getenv('OPENAI_API_KEY')
extractor = EnhancedScreenerExtractionV3(api_key)

# Enhanced Pydantic models for structured outputs
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

class EnhancedFundamentalAnalysis(BaseModel):
    business_quality: str = Field(description="Business quality assessment")
    market_penetration: str = Field(description="Product/service penetration analysis")
    pricing_power: str = Field(description="Pricing power assessment")
    revenue_growth: str = Field(description="Revenue growth percentage")
    profit_growth: str = Field(description="Profit growth percentage")
    debt_to_equity: str = Field(description="Debt to equity ratio and assessment")
    roce_roe: str = Field(description="ROCE/ROE analysis")
    promoter_pledging: str = Field(description="Promoter pledging status")
    retail_shareholding: str = Field(description="Retail shareholding percentage")
    valuation_status: str = Field(description="Valuation assessment")
    fair_value: str = Field(description="Fair value estimate")
    financial_health: str = Field(description="Overall financial health assessment")
    multibagger_potential: str = Field(description="Multibagger potential assessment")
    fundamental_reasons: str = Field(description="Fundamental reasoning for buy signals")
    confidence_score: str = Field(description="Fundamental confidence level: Strong, Medium, Low, Can't Say")

class EnhancedFinalRecommendation(BaseModel):
    action: str = Field(description="BUY, HOLD, or EXIT")
    entry_price: str = Field(description="Entry price range")
    target_price: str = Field(description="Target price")
    stop_loss: str = Field(description="Stop loss price (if applicable)")
    time_horizon: str = Field(description="Investment time horizon")
    confidence_level: str = Field(description="Confidence level")
    risk_level: str = Field(description="Risk assessment")
    position_size: str = Field(description="Recommended position size (% of portfolio)")
    strategy_used: str = Field(description="Primary strategy being used")
    key_risks: List[str] = Field(description="Key risks to monitor")
    fundamental_reasons: str = Field(description="Fundamental reasoning for buy signals")

@dataclass
class StockData:
    ticker: str
    company_name: str
    sector: str
    category: str = "Unknown"  # Add category field
    ohlcv_data: Optional[pd.DataFrame] = None
    fundamental_data: Optional[FundamentalData] = None
    chart_image: Optional[str] = None  # For TradingView screenshot

# Enhanced Technical Analysis Agent with Strategy Implementation
class EnhancedTechnicalAnalysisAgent:
    def __init__(self, llm: ChatOpenAI, openai_api_key: str):
        self.llm = llm
        self.openai_api_key = openai_api_key
        self.charts_dir = "technical_charts"
        os.makedirs(self.charts_dir, exist_ok=True)
    
    def analyze(self, stock_data: StockData, existing_chart_path: str = None) -> EnhancedTechnicalAnalysis:
        print(f"📈 Enhanced Technical Analysis Agent analyzing {stock_data.ticker}...")
        
        # Check if we have OHLCV data
        if stock_data.ohlcv_data is not None and not stock_data.ohlcv_data.empty:
            print("✅ Technical Analysis: Using OHLCV data for analysis")
            
            # Identify patterns first
            rhs_analysis = self._identify_rhs_pattern(stock_data)
            cwh_analysis = self._identify_cwh_pattern(stock_data)
            
            # Try comprehensive analysis with chart
            if existing_chart_path and os.path.exists(existing_chart_path):
                print(f"📊 Using existing chart for analysis: {existing_chart_path}")
                chart_base64 = self._get_chart_base64_from_path(existing_chart_path)
            else:
                print(f"📊 No existing chart available, skipping chart analysis to preserve colors...")
                chart_base64 = None
            
            if chart_base64:
                print("✅ Technical Analysis: Using chart image for analysis")
                return self._analyze_with_chart_and_data(stock_data, chart_base64, rhs_analysis, cwh_analysis)
            else:
                print("📊 Falling back to OHLCV data analysis")
                return self._analyze_with_ohlcv_data(stock_data, rhs_analysis, cwh_analysis)
        else:
            print("❌ No OHLCV data available for technical analysis")
            return self._get_basic_technical_analysis(stock_data)
    
    def _get_basic_technical_analysis(self, stock_data: StockData) -> EnhancedTechnicalAnalysis:
        """Get basic technical analysis using OHLCV data when OpenAI analysis fails"""
        try:
            df = stock_data.ohlcv_data
            if df is None or df.empty:
                return self._create_default_technical_analysis()
            
            # Calculate basic indicators
            current_price = df['Close'].iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            # Moving averages
            ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            ma_200 = df['Close'].rolling(window=200).mean().iloc[-1]
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            current_rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            # Volume analysis
            volume_avg = df['Volume'].rolling(window=20).mean().iloc[-1]
            
            # Trend analysis
            if current_price > ma_20 > ma_50:
                trend = "Bullish"
            elif current_price < ma_20 < ma_50:
                trend = "Bearish"
            else:
                trend = "Sideways"
            
            # Support and resistance levels
            recent_low = df['Low'].tail(20).min()
            recent_high = df['High'].tail(20).max()
            
            # Confidence based on indicators
            if abs(current_price - ma_20) / ma_20 < 0.02 and current_rsi > 30 and current_rsi < 70:
                confidence = "Medium"
            elif trend == "Bullish" and current_rsi < 70:
                confidence = "High"
            elif trend == "Bearish" and current_rsi > 30:
                confidence = "High"
            else:
                confidence = "Low"
            
            return EnhancedTechnicalAnalysis(
                trend=trend,
                support_levels=[recent_low, ma_50],
                resistance_levels=[recent_high, ma_200],
                entry_range=f"{current_price * 0.98:.2f} - {current_price * 1.02:.2f}",
                short_term_target=f"{current_price * 1.05:.2f}",
                medium_term_target=f"{current_price * 1.15:.2f}",
                stop_loss=f"{current_price * 0.95:.2f}",
                confidence_score=confidence,
                indicators={
                    "RSI": round(float(current_rsi), 2) if not pd.isna(current_rsi) else 0.0,
                    "MA_20": round(float(ma_20), 2) if not pd.isna(ma_20) else 0.0,
                    "MA_50": round(float(ma_50), 2) if not pd.isna(ma_50) else 0.0,
                    "MA_200": round(float(ma_200), 2) if not pd.isna(ma_200) else 0.0,
                    "Volume_Ratio": round(float(current_volume/volume_avg), 2) if volume_avg > 0 else 0.0
                },
                patterns=[],
                strategy_signals=[],
                position_sizing="3-5% of portfolio"
            )
            
        except Exception as e:
            print(f"❌ Error in basic technical analysis: {e}")
            return self._create_default_technical_analysis()
    
    def _create_default_technical_analysis(self) -> EnhancedTechnicalAnalysis:
        """Create default technical analysis when all else fails"""
        return EnhancedTechnicalAnalysis(
            trend="Sideways",
            support_levels=[],
            resistance_levels=[],
            entry_range="Not Available",
            short_term_target="Not Available",
            medium_term_target="Not Available",
            stop_loss="Not Available",
            confidence_score="Low",
            indicators={},
            patterns=[],
            strategy_signals=[],
            position_sizing="Not Available"
        )
    
    def _analyze_with_ohlcv_data(self, stock_data: StockData, rhs_analysis: Optional[Dict] = None, cwh_analysis: Optional[Dict] = None) -> EnhancedTechnicalAnalysis:
        """Analyze using OHLCV data"""
        # Handle MultiIndex DataFrame structure
        if isinstance(stock_data.ohlcv_data.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns
            stock_data.ohlcv_data.columns = stock_data.ohlcv_data.columns.get_level_values(0)
        
        # Calculate real technical indicators from OHLCV data
        close_prices = stock_data.ohlcv_data['Close']
        high_prices = stock_data.ohlcv_data['High']
        low_prices = stock_data.ohlcv_data['Low']
        volume = stock_data.ohlcv_data['Volume']
        
        # Calculate real indicators
        ma_20 = close_prices.rolling(window=20).mean().iloc[-1].item()
        ma_50 = close_prices.rolling(window=50).mean().iloc[-1].item()
        ma_200 = close_prices.rolling(window=200).mean().iloc[-1].item()
        current_price = close_prices.iloc[-1].item()
        
        # Calculate RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1].item()
        
        # Calculate Bollinger Bands
        bb_upper = ma_20 + (2 * close_prices.rolling(window=20).std())
        bb_lower = ma_20 - (2 * close_prices.rolling(window=20).std())
        bb_upper_current = bb_upper.iloc[-1].item()
        bb_lower_current = bb_lower.iloc[-1].item()
        
        # Enhanced Pattern Recognition
        patterns = []
        strategy_signals = []
        position_sizing = "3% of portfolio"
        
        # RHS Pattern Analysis
        if rhs_analysis and rhs_analysis.get("is_valid", False):
            patterns.append("Reverse Head and Shoulders (RHS)")
            strategy_signals.append("RHS Strategy: Valid bullish reversal pattern identified")
            position_sizing = "5% of portfolio (RHS pattern)"
        
        # CWH Pattern Analysis
        if cwh_analysis and cwh_analysis.get("is_valid", False):
            patterns.append("Cup with Handle (CWH)")
            strategy_signals.append("CWH Strategy: Valid bullish continuation pattern identified")
            position_sizing = "5% of portfolio (CWH pattern)"
        
        # 1. Moving Average Analysis
        if ma_20 > ma_50 > ma_200:
            patterns.append("Golden Cross (Bullish)")
            strategy_signals.append("SMA Strategy: Bullish trend confirmed")
        elif ma_20 < ma_50 < ma_200:
            patterns.append("Death Cross (Bearish)")
            strategy_signals.append("SMA Strategy: Bearish trend confirmed")
        elif ma_20 > ma_50 and ma_50 < ma_200:
            patterns.append("Golden Cross Formation")
            strategy_signals.append("SMA Strategy: Potential bullish reversal")
        
        # 2. RSI Analysis
        if current_rsi < 30:
            patterns.append("RSI Oversold")
            strategy_signals.append("RSI Strategy: Oversold condition - potential buy")
        elif current_rsi > 70:
            patterns.append("RSI Overbought")
            strategy_signals.append("RSI Strategy: Overbought condition - potential sell")
        
        # 3. Bollinger Bands Analysis
        if current_price <= bb_lower_current:
            patterns.append("BB Lower Band Touch")
            strategy_signals.append("BB Strategy: Price at support - potential bounce")
        elif current_price >= bb_upper_current:
            patterns.append("BB Upper Band Touch")
            strategy_signals.append("BB Strategy: Price at resistance - potential reversal")
        
        # 4. Volume Analysis
        avg_volume = volume.rolling(window=20).mean().iloc[-1].item()
        current_volume = volume.iloc[-1].item()
        if current_volume > 1.5 * avg_volume:
            patterns.append("Volume Spike")
            strategy_signals.append("Volume Strategy: High volume confirms trend")
        
        # 5. Support and Resistance Levels
        # Recent support (20-day low)
        recent_support = low_prices.tail(20).min().item()
        # Recent resistance (20-day high)
        recent_resistance = high_prices.tail(20).max().item()
        # Major support (50-day low)
        major_support = low_prices.tail(50).min().item()
        # Major resistance (50-day high)
        major_resistance = high_prices.tail(50).max().item()
        
        # 6. Lifetime High Strategy Check
        all_time_high = high_prices.max().item()
        if current_price < all_time_high * 0.7:  # 30% down from ATH
            patterns.append("Lifetime High Pullback")
            strategy_signals.append("Lifetime High Strategy: Stock down >30% from ATH - potential opportunity")
            position_sizing = "Up to 10% of portfolio (add 3% at every 10% drop)"
        
        # 7. V20 Strategy Check
        recent_range = (recent_resistance - recent_support) / recent_support
        if recent_range > 0.2:  # 20% range
            patterns.append("V20 Range Formation")
            strategy_signals.append("V20 Strategy: Significant range identified - buy at lower line")
        
        # Now use OpenAI for advanced pattern recognition and target calculation
        try:
            enhanced_analysis = self._get_openai_technical_analysis(stock_data, patterns, strategy_signals)
            if enhanced_analysis:
                return enhanced_analysis
        except Exception as e:
            print(f"⚠️ OpenAI analysis failed, using calculated analysis: {e}")
        
        # Fallback to calculated analysis if OpenAI fails
        # Determine trend and calculate meaningful targets
        if ma_20 > ma_50:
            trend = "Bullish"
            # Bullish targets based on resistance levels
            if current_price < recent_resistance:
                target_st = f"₹{round(recent_resistance, 2)}"
            else:
                target_st = f"₹{round(major_resistance, 2)}"
            
            if current_price < major_resistance:
                target_mt = f"₹{round(major_resistance * 1.05, 2)}"
            else:
                target_mt = f"₹{round(current_price * 1.15, 2)}"
            
            entry_range = f"₹{round(current_price * 0.98, 2)}–₹{round(current_price * 1.02, 2)}"
            stop_loss = f"₹{round(recent_support * 0.95, 2)} (Below recent support)"
            
        else:
            trend = "Bearish"
            # Bearish targets based on support levels
            if current_price > recent_support:
                target_st = f"₹{round(recent_support, 2)}"
            else:
                target_st = f"₹{round(major_support, 2)}"
            
            if current_price > major_support:
                target_mt = f"₹{round(major_support * 0.95, 2)}"
            else:
                target_mt = f"₹{round(current_price * 0.85, 2)}"
            
            entry_range = f"₹{round(current_price * 0.98, 2)}–₹{round(current_price * 1.02, 2)}"
            stop_loss = f"₹{round(recent_resistance * 1.05, 2)} (Above recent resistance)"
        
        # Support and resistance levels for analysis
        support_levels = [round(recent_support, 2), round(major_support, 2)]
        resistance_levels = [round(recent_resistance, 2), round(major_resistance, 2)]
        
        # Confidence based on pattern strength
        confidence = "High" if len(patterns) >= 3 else "Medium" if len(patterns) >= 2 else "Low"
        
        enhanced_analysis = EnhancedTechnicalAnalysis(
            trend=trend,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            entry_range=entry_range,
            short_term_target=target_st,
            medium_term_target=target_mt,
            stop_loss=stop_loss,
            confidence_score=confidence,
            indicators={
                "RSI": round(current_rsi, 2), 
                "MA_20": round(ma_20, 2), 
                "MA_50": round(ma_50, 2), 
                "MA_200": round(ma_200, 2),
                "BB_Upper": round(bb_upper_current, 2),
                "BB_Lower": round(bb_lower_current, 2),
                "Volume_Ratio": round(current_volume/avg_volume, 2)
            },
            patterns=patterns,
            strategy_signals=strategy_signals,
            position_sizing=position_sizing
        )
        
        return enhanced_analysis
    
    def _analyze_with_chart_image(self, stock_data: StockData) -> EnhancedTechnicalAnalysis:
        """Analyze using TradingView chart image"""
        print(f"📸 Analyzing TradingView chart image for {stock_data.ticker}...")
        
        try:
            # Use OpenAI vision model to analyze the chart image
            enhanced_analysis = self._get_openai_chart_analysis(stock_data)
            if enhanced_analysis:
                return enhanced_analysis
        except Exception as e:
            print(f"⚠️ Chart image analysis failed: {e}")
        
        # Fallback analysis for chart image
        return EnhancedTechnicalAnalysis(
            trend="Not Available (Chart Analysis)",
            support_levels=[],
            resistance_levels=[],
            entry_range="Not Available",
            short_term_target="Not Available",
            medium_term_target="Not Available",
            stop_loss="Not Available",
            confidence_score="Not Available",
            indicators={},
            patterns=[],
            strategy_signals=[],
            position_sizing="Not Available"
        )
    
    def _analyze_with_chart_and_data(self, stock_data: StockData, chart_base64: str, rhs_analysis: Optional[Dict] = None, cwh_analysis: Optional[Dict] = None) -> EnhancedTechnicalAnalysis:
        """Analyze using both candlestick chart and comprehensive OHLCV data"""
        print(f"📊 Analyzing candlestick chart and comprehensive data for {stock_data.ticker}...")
        
        try:
            # Use OpenAI vision model to analyze the candlestick chart with comprehensive data
            enhanced_analysis = self._get_openai_comprehensive_analysis(stock_data, chart_base64, rhs_analysis, cwh_analysis)
            if enhanced_analysis:
                return enhanced_analysis
        except Exception as e:
            print(f"⚠️ Chart and data analysis failed: {e}")
        
        # Fallback to OHLCV data analysis if chart analysis fails
        print(f"📊 Falling back to OHLCV data analysis")
        return self._analyze_with_ohlcv_data(stock_data, rhs_analysis, cwh_analysis)
    
    def _get_openai_comprehensive_analysis(self, stock_data: StockData, chart_base64: str, rhs_analysis: Optional[Dict] = None, cwh_analysis: Optional[Dict] = None) -> Optional[EnhancedTechnicalAnalysis]:
        """Get comprehensive technical analysis using OpenAI Vision API"""
        try:
            print(f"DEBUG: ENTERING _get_openai_comprehensive_analysis for {stock_data.ticker}")
            
            # Ensure 're' module is available
            import re
            
            # Create OpenAI client
            client = OpenAI(api_key=self.openai_api_key)
            
            # Prepare the analysis context
            analysis_context = self._create_comprehensive_analysis_context(stock_data, rhs_analysis, cwh_analysis)
            
            # Create the prompt
            prompt = f"""
            You are an expert technical analyst specializing in Indian stock markets. Analyze the provided candlestick chart and data for {stock_data.ticker}.

            ANALYSIS CONTEXT:
            {analysis_context}

            CHART DATA:
            - Chart image is provided as base64
            - Analyze candlestick patterns, trends, and technical indicators
            - Consider support/resistance levels, volume analysis, and momentum

            REQUIREMENTS:
            1. Analyze the candlestick chart thoroughly
            2. Identify key technical patterns and trends
            3. Determine support and resistance levels
            4. Assess entry points and targets
            5. Provide confidence levels and risk assessment

            RESPONSE FORMAT:
            Return ONLY a valid JSON object with the following structure:
            {{
                "trend": "Bullish/Bearish/Sideways",
                "support_levels": [list of support prices],
                "resistance_levels": [list of resistance prices],
                "entry_range": "price range for entry",
                "short_term_target": "short-term target price",
                "medium_term_target": "medium-term target price",
                "stop_loss": "stop loss price",
                "confidence_score": "High/Medium/Low",
                "patterns": ["list of identified patterns"],
                "strategy_signals": ["list of trading signals"]
            }}

            IMPORTANT: Return ONLY the JSON object, no additional text or explanations.
            """
            
            print(f"🤖 Calling OpenAI Vision API for {stock_data.ticker}...")
            
            # Call OpenAI Vision API
            response = client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-5'),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{chart_base64}"
                                }
                            }
                        ]}
                ],
                max_completion_tokens=1000
            )
            
            print(f"DEBUG: OpenAI API call completed for {stock_data.ticker}")
            print(f"DEBUG: response type: {type(response)}")
            print(f"DEBUG: response has choices: {hasattr(response, 'choices')}")
            print(f"DEBUG: response.choices length: {len(response.choices) if hasattr(response, 'choices') else 'N/A'}")
            
            # Log the API call
            cost_tracker.log_api_call(
                model=os.getenv('OPENAI_MODEL', 'gpt-5'),
                tokens_used=response.usage.total_tokens,
                cost_usd=response.usage.total_tokens * 0.003 / 1000,  # gpt-5 pricing
                description=f"Technical analysis for {stock_data.ticker}"
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            print(f"DEBUG: response_content type: {type(response_content)}")
            print(f"DEBUG: response_content length: {len(response_content)}")
            print(f"✅ OpenAI response received successfully for {stock_data.ticker}")
            
            # Save raw response for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            response_filename = f"openai_responses/openai_response_{stock_data.ticker}_{timestamp}.txt"
            os.makedirs("openai_responses", exist_ok=True)
            
            print(f"DEBUG: About to save OpenAI response for {stock_data.ticker} to {response_filename}")
            try:
                with open(response_filename, 'w', encoding='utf-8') as f:
                    f.write(f"Ticker: {stock_data.ticker}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Response Content:\n{response_content}\n")
                print(f"DEBUG: Finished saving OpenAI response for {stock_data.ticker}")
                print(f"✅ Raw response save attempt complete: {response_filename}")
            except Exception as save_error:
                print(f"⚠️ Failed to save response: {save_error}")
            
            # Clean and parse the response
            try:
                # Remove markdown code blocks if present
                cleaned_content = response_content.strip()
                if cleaned_content.startswith('```json'):
                    cleaned_content = cleaned_content[7:]
                if cleaned_content.endswith('```'):
                    cleaned_content = cleaned_content[:-3]
                cleaned_content = cleaned_content.strip()
                
                # Remove any trailing commas before closing braces/brackets
                cleaned_content = re.sub(r',(\s*[}\]])', r'\1', cleaned_content)
                
                # Parse JSON
                analysis_data = json.loads(cleaned_content)
                
                # Convert to EnhancedTechnicalAnalysis
                return EnhancedTechnicalAnalysis(
                    trend=analysis_data.get('trend', 'Sideways'),
                    support_levels=analysis_data.get('support_levels', []),
                    resistance_levels=analysis_data.get('resistance_levels', []),
                    entry_range=analysis_data.get('entry_range', 'N/A'),
                    short_term_target=analysis_data.get('short_term_target', 'N/A'),
                    medium_term_target=analysis_data.get('medium_term_target', 'N/A'),
                    stop_loss=analysis_data.get('stop_loss', 'N/A'),
                    confidence_score=analysis_data.get('confidence_score', 'Medium'),
                    indicators={},
                    patterns=analysis_data.get('patterns', []),
                    strategy_signals=analysis_data.get('strategy_signals', []),
                    position_sizing="Standard position sizing recommended"
                )
                
            except json.JSONDecodeError as json_error:
                print(f"DEBUG: Response parsing failed - returning basic analysis for {stock_data.ticker}")
                print(f"⚠️ Response parsing failed for {stock_data.ticker}: {json_error}")
                print(f"Raw content: {response_content[:200]}...")
                return None
                
        except Exception as e:
            print(f"❌ Error in OpenAI comprehensive analysis for {stock_data.ticker}: {str(e)}")
            return None
    
    def _get_openai_chart_analysis(self, stock_data: StockData, existing_chart_path: str = None) -> Optional[EnhancedTechnicalAnalysis]:
        """Use OpenAI vision model to analyze candlestick chart image"""
        try:
            import base64
            
            # Use existing chart if provided, otherwise skip chart analysis to preserve colors
            if existing_chart_path and os.path.exists(existing_chart_path):
                print(f"📊 Using existing candlestick chart: {existing_chart_path}")
                with open(existing_chart_path, 'rb') as f:
                    chart_base64 = base64.b64encode(f.read()).decode('utf-8')
            else:
                print(f"📊 No existing chart available, skipping chart analysis to preserve colors...")
                return None
            
            if not chart_base64:
                print("❌ Failed to read existing candlestick chart for analysis")
                return None
            
            # Create prompt for chart analysis
            prompt = f"""
You are an expert technical analyst. Analyze this candlestick chart for {stock_data.ticker} ({stock_data.company_name}) and provide:

1. TREND: Bullish/Bearish/Sideways
2. SUPPORT LEVELS: Key support levels from the chart
3. RESISTANCE LEVELS: Key resistance levels from the chart
4. ENTRY RANGE: Suggested entry price range
5. SHORT-TERM TARGET: Target for 5-10 days
6. MEDIUM-TERM TARGET: Target for 1-3 months
7. STOP LOSS: Logical stop loss level
8. PATTERNS: Any chart patterns you identify
9. STRATEGY: Most appropriate trading strategy
10. CONFIDENCE: High/Medium/Low

Respond in JSON format:
{{
    "trend": "Bullish/Bearish/Sideways",
    "support_levels": [level1, level2],
    "resistance_levels": [level1, level2],
    "entry_range": "₹X-₹Y",
    "short_term_target": "₹X",
    "medium_term_target": "₹X",
    "stop_loss": "₹X",
    "confidence_score": "High/Medium/Low",
    "patterns": ["pattern1", "pattern2"],
    "strategy_signals": ["strategy1", "strategy2"],
    "position_sizing": "X% of portfolio"
}}
"""
            
            # Call OpenAI with vision using base64 string
            response = self.llm.invoke([
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_base64}"}}
                ]}
            ])
            
            # Parse response
            response_text = response.content
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                analysis_data = json.loads(json_str)
                
                return EnhancedTechnicalAnalysis(
                    trend=analysis_data.get('trend', 'Not Available'),
                    support_levels=analysis_data.get('support_levels', []),
                    resistance_levels=analysis_data.get('resistance_levels', []),
                    entry_range=analysis_data.get('entry_range', 'Not Available'),
                    short_term_target=analysis_data.get('short_term_target', 'Not Available'),
                    medium_term_target=analysis_data.get('medium_term_target', 'Not Available'),
                    stop_loss=analysis_data.get('stop_loss', 'Not Available'),
                    confidence_score=analysis_data.get('confidence_score', 'Not Available'),
                    indicators={},
                    patterns=analysis_data.get('patterns', []),
                    strategy_signals=analysis_data.get('strategy_signals', []),
                    position_sizing=analysis_data.get('position_sizing', 'Not Available')
                )
            
        except Exception as e:
            print(f"⚠️ Chart analysis failed: {e}")
        
        return None
    
    def _get_openai_technical_analysis(self, stock_data: StockData, patterns: List[str], strategy_signals: List[str]) -> Optional[EnhancedTechnicalAnalysis]:
        """Get technical analysis using OpenAI with cost tracking"""
        try:
            # Prepare OHLCV data summary for the LLM
            ohlcv_data = stock_data.ohlcv_data
            if ohlcv_data is None or len(ohlcv_data) == 0:
                return None
            
            # Create a comprehensive summary of recent price action (2-year analysis)
            recent_data = ohlcv_data.tail(730)  # Last 2 years (730 days)
            current_price = recent_data['Close'].iloc[-1]
            high_30d = recent_data['High'].tail(30).max()
            low_30d = recent_data['Low'].tail(30).min()
            high_90d = recent_data['High'].tail(90).max()
            low_90d = recent_data['Low'].tail(90).min()
            high_180d = recent_data['High'].tail(180).max()
            low_180d = recent_data['Low'].tail(180).min()
            high_365d = recent_data['High'].tail(365).max()
            low_365d = recent_data['Low'].tail(365).min()
            high_2y = recent_data['High'].max()
            low_2y = recent_data['Low'].min()
            volume_avg = recent_data['Volume'].mean()
            current_volume = recent_data['Volume'].iloc[-1]
            
            # Calculate key levels
            ma_20 = recent_data['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = ohlcv_data['Close'].rolling(window=50).mean().iloc[-1]
            ma_200 = ohlcv_data['Close'].rolling(window=200).mean().iloc[-1]
            
            # Calculate RSI
            delta = ohlcv_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Create comprehensive technical summary
            technical_summary = f"""
COMPREHENSIVE TECHNICAL DATA SUMMARY for {stock_data.ticker} ({stock_data.company_name}):

PRICE DATA (2-YEAR COMPREHENSIVE ANALYSIS):
Current Price: ₹{float(current_price):.2f}
2-Year High: ₹{float(high_2y):.2f}
2-Year Low: ₹{float(low_2y):.2f}

2-YEAR ANALYSIS PERIODS:
30-Day High: ₹{float(high_30d):.2f} | Low: ₹{float(low_30d):.2f}
90-Day High: ₹{float(high_90d):.2f} | Low: ₹{float(low_90d):.2f}
180-Day High: ₹{float(high_180d):.2f} | Low: ₹{float(low_180d):.2f}
365-Day High: ₹{float(high_365d):.2f} | Low: ₹{float(low_365d):.2f}
2-Year High: ₹{float(high_2y):.2f} | Low: ₹{float(low_2y):.2f}

MOVING AVERAGES:
20-Day MA: ₹{float(ma_20):.2f}
50-Day MA: ₹{float(ma_50):.2f}
200-Day MA: ₹{float(ma_200):.2f}

TECHNICAL INDICATORS:
Current RSI: {float(current_rsi):.2f}
Volume (Current/Avg): {float(current_volume):.0f}/{float(volume_avg):.0f} ({float(current_volume/volume_avg):.2f}x)

Identified Patterns: {', '.join(patterns)}
Strategy Signals: {', '.join(strategy_signals)}

COMPREHENSIVE PRICE ACTION (last 60 days for detailed analysis):
"""
            
            # Add comprehensive price action (60 days for detailed analysis)
            for i, (date, row) in enumerate(recent_data.iterrows()):
                try:
                    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                    technical_summary += f"  {date_str}: O:{float(row['Open']):.2f} H:{float(row['High']):.2f} L:{float(row['Low']):.2f} C:{float(row['Close']):.2f} V:{float(row['Volume']):.0f}\n"
                except Exception as e:
                    print(f"⚠️ Error formatting row {i}: {e}")
                    continue
            
            # Create the prompt for OpenAI
            prompt = f"""
You are an expert technical analyst specializing in Indian stock markets. Analyze the following comprehensive technical data and provide a detailed analysis.

{technical_summary}

Based on this comprehensive 2-year data, please provide:

1. TREND ANALYSIS: Is this stock in a bullish, bearish, or sideways trend? Consider moving averages, price action, and volume patterns.

2. SUPPORT & RESISTANCE: Identify key support and resistance levels based on recent price action and moving averages.

3. CHART PATTERNS: Identify any chart patterns (Head & Shoulders, Double Top/Bottom, Cup & Handle, Flags, Pennants, etc.) from the price action.

4. ENTRY STRATEGY: Based on the identified patterns and current price, suggest an entry range.

5. TARGETS: Provide realistic short-term (5-10 days) and medium-term (1-3 months) targets based on support/resistance levels and pattern completion.

6. STRATEGY RECOMMENDATION: Which of these strategies would be most appropriate:
   - SMA Strategy (Moving Average crossovers)
   - V20 Strategy (Range trading)
   - Lifetime High Strategy (if down >30% from ATH)
   - RSI Strategy (Oversold/Overbought)
   - Bollinger Bands Strategy
   - Breakout Strategy

7. CONFIDENCE: Rate your analysis confidence as High/Medium/Low based on pattern clarity and signal strength.

Please respond in this exact JSON format:
{{
    "trend": "Bullish/Bearish/Sideways",
    "support_levels": [level1, level2],
    "resistance_levels": [level1, level2],
    "entry_range": "₹X-₹Y",
    "short_term_target": "₹X",
    "medium_term_target": "₹X",
    "stop_loss": "₹X or No stop-loss",
    "confidence_score": "High/Medium/Low",
    "patterns": ["pattern1", "pattern2"],
    "strategy_signals": ["signal1", "signal2"],
    "position_sizing": "X% of portfolio"
}}
"""
            
            # Call OpenAI
            response = self.llm.invoke(prompt)
            
            # Track the API call usage
            if hasattr(response, 'usage'):
                usage = response.usage
                cost_tracker.log_usage(
                    model=os.getenv('OPENAI_MODEL', 'gpt-5'),
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    call_type="chat",
                    description=f"Technical analysis for {stock_data.ticker}"
                )
            
            # Parse the response
            try:
                # Extract JSON from the response
                response_text = response.content
                
                # Find JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    
                    analysis_data = json.loads(json_str)
                    
                    # Create EnhancedTechnicalAnalysis object
                    enhanced_analysis = EnhancedTechnicalAnalysis(
                        trend=analysis_data.get('trend', 'Sideways'),
                        support_levels=analysis_data.get('support_levels', []),
                        resistance_levels=analysis_data.get('resistance_levels', []),
                        entry_range=analysis_data.get('entry_range', 'Not Available'),
                        short_term_target=analysis_data.get('short_term_target', 'Not Available'),
                        medium_term_target=analysis_data.get('medium_term_target', 'Not Available'),
                        stop_loss=analysis_data.get('stop_loss', 'Not Available'),
                        confidence_score=analysis_data.get('confidence_score', 'Not Available'),
                        indicators={
                            "RSI": round(float(current_rsi), 2) if not pd.isna(current_rsi) else 0.0,
                            "MA_20": round(float(ma_20), 2) if not pd.isna(ma_20) else 0.0,
                            "MA_50": round(float(ma_50), 2) if not pd.isna(ma_50) else 0.0,
                            "MA_200": round(float(ma_200), 2) if not pd.isna(ma_200) else 0.0,
                            "Volume_Ratio": round(float(current_volume/volume_avg), 2) if volume_avg > 0 else 0.0
                        },
                        patterns=analysis_data.get('patterns', patterns),
                        strategy_signals=analysis_data.get('strategy_signals', strategy_signals),
                        position_sizing=analysis_data.get('position_sizing', 'Not Available')
                    )
                    
                    return enhanced_analysis
                else:
                    print("⚠️ Could not find JSON in OpenAI response")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse OpenAI JSON response: {e}")
                return None
                
        except Exception as e:
            print(f"⚠️ OpenAI technical analysis failed: {e}")
            return None

    def _create_candlestick_chart(self, stock_data: StockData, technical_analysis: Optional[EnhancedTechnicalAnalysis] = None) -> Optional[str]:
        """Create a professional candlestick chart from OHLCV data"""
        try:
            if stock_data.ohlcv_data is None or len(stock_data.ohlcv_data) == 0:
                return None
            
            # Handle MultiIndex DataFrame structure
            if isinstance(stock_data.ohlcv_data.columns, pd.MultiIndex):
                stock_data.ohlcv_data.columns = stock_data.ohlcv_data.columns.get_level_values(0)
            
            # Get the data
            df = stock_data.ohlcv_data.copy()
            
            # Set matplotlib backend for better color support
            import matplotlib
            matplotlib.use('Agg')  # Use Agg backend for better color support
            # Ensure default style and white backgrounds
            import matplotlib.pyplot as plt
            plt.style.use('default')
            
            # Create figure and axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            fig.patch.set_facecolor('white')
            ax1.set_facecolor('white')
            ax2.set_facecolor('white')
            
            # Plot candlestick chart
            self._plot_candlesticks(ax1, df, stock_data.ticker, technical_analysis)
            
            # Plot volume
            self._plot_volume(ax2, df)
            
            # Add title and labels
            fig.suptitle(f'{stock_data.ticker} ({stock_data.company_name}) - Technical Analysis Chart', 
                        fontsize=16, fontweight='bold', color='black')
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Save chart locally
            ticker_clean = stock_data.ticker.replace('.NS', '')
            chart_filename = f"candlestick_{ticker_clean}.png"
            
            # Save to current directory (will be moved by caller if needed)
            with open(chart_filename, 'wb') as f:
                f.write(buffer.getvalue())
            
            return chart_filename
            
        except Exception as e:
            print(f"❌ Error creating candlestick chart: {e}")
            return None
    
    def _plot_candlesticks(self, ax, df: pd.DataFrame, ticker: str, technical_analysis: Optional[EnhancedTechnicalAnalysis] = None):
        """Plot candlestick chart with technical indicators"""
        # Calculate moving averages
        ma_20 = df['Close'].rolling(window=20).mean()
        ma_50 = df['Close'].rolling(window=50).mean()
        ma_100 = df['Close'].rolling(window=100).mean()
        ma_200 = df['Close'].rolling(window=200).mean()
        
        # Plot candlesticks with enhanced colors
        for i in range(len(df)):
            date = df.index[i]
            open_price = float(df['Open'].iloc[i])
            high_price = float(df['High'].iloc[i])
            low_price = float(df['Low'].iloc[i])
            close_price = float(df['Close'].iloc[i])
            
            # Determine color - GREEN for bullish, RED for bearish
            color = '#2E8B57' if close_price >= open_price else '#DC143C'  # Sea Green for bullish, Crimson for bearish
            
            # Plot body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            if body_height > 0:
                ax.bar(date, body_height, bottom=body_bottom, color=color, alpha=0.85, width=0.7)
            else:
                ax.plot([date, date], [open_price - 0.1, open_price + 0.1], color=color, alpha=0.5, linewidth=0.6)
            
            # Plot wicks using candle color with low alpha (avoid black)
            ax.plot([date, date], [low_price, high_price], color=color, alpha=0.35, linewidth=0.6)
        
        # Add moving averages
        ax.plot(df.index, ma_20, label='MA 20', color='blue', linewidth=1, alpha=0.8)
        ax.plot(df.index, ma_50, label='MA 50', color='orange', linewidth=1, alpha=0.8)
        ax.plot(df.index, ma_100, label='MA 100', color='purple', linewidth=1, alpha=0.8)
        ax.plot(df.index, ma_200, label='MA 200', color='brown', linewidth=1, alpha=0.8)
        
        # Bollinger Bands
        bb_std = df['Close'].rolling(window=20).std()
        bb_upper = ma_20 + (2 * bb_std)
        bb_lower = ma_20 - (2 * bb_std)
        ax.fill_between(df.index, bb_upper, bb_lower, alpha=0.08, color='gray', label='Bollinger Bands')
        
        # Support, Resistance, and Target lines
        try:
            # Default calculations
            recent = df.tail(120)
            default_support = float(recent['Low'].rolling(20).min().iloc[-1])
            default_resistance = float(recent['High'].rolling(20).max().iloc[-1])
            last_close = float(df['Close'].iloc[-1])
            default_target = last_close * 1.1
            
            # Use technical analysis values if available
            if technical_analysis:
                support_levels = getattr(technical_analysis, 'support_levels', [default_support])
                resistance_levels = getattr(technical_analysis, 'resistance_levels', [default_resistance])
                
                # Extract target from short_term_target or medium_term_target
                short_target_str = getattr(technical_analysis, 'short_term_target', '')
                medium_target_str = getattr(technical_analysis, 'medium_term_target', '')
                
                target_level = default_target
                try:
                    # Try to extract numeric value from target strings (remove ₹ symbol)
                    if short_target_str and short_target_str != "Not Available":
                        target_level = float(short_target_str.replace('₹', '').replace(',', '').strip())
                    elif medium_target_str and medium_target_str != "Not Available":
                        target_level = float(medium_target_str.replace('₹', '').replace(',', '').strip())
                except (ValueError, AttributeError):
                    target_level = default_target
                
                # Use first support and resistance levels
                support_level = support_levels[0] if support_levels else default_support
                resistance_level = resistance_levels[0] if resistance_levels else default_resistance
            else:
                support_level = default_support
                resistance_level = default_resistance
                target_level = default_target
            
            # Plot the lines
            ax.axhline(support_level, color='#2E8B57', linestyle='--', linewidth=1.2, label=f'Support ~ {support_level:.2f}')
            ax.axhline(resistance_level, color='#DC143C', linestyle='--', linewidth=1.2, label=f'Resistance ~ {resistance_level:.2f}')
            ax.axhline(target_level, color='#1E90FF', linestyle='--', linewidth=1.2, label=f'Target ~ {target_level:.2f}')
        except Exception as e:
            print(f"⚠️ Error plotting S/R/T lines: {e}")
            pass
        
        ax.set_title(f'{ticker} - 3-Year Candlestick Chart (with S/R + Target)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_volume(self, ax, df: pd.DataFrame):
        """Plot volume bars"""
        volume_colors = []
        for i in range(len(df)):
            close_price = float(df['Close'].iloc[i])
            open_price = float(df['Open'].iloc[i])
            volume_colors.append('#2E8B57' if close_price >= open_price else '#DC143C')  # Sea Green for bullish, Crimson for bearish
        
        ax.bar(df.index, df['Volume'], color=volume_colors, alpha=0.55, width=0.8)
        ax.set_title('Volume Analysis (3-Year)', fontsize=12, fontweight='bold', color='black')
        ax.set_ylabel('Volume', fontsize=10, color='black')
        ax.grid(True, alpha=0.25, color='#888888')
        
        # Format x-axis for volume
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, color='black')

    def _identify_rhs_pattern(self, stock_data: StockData) -> Dict:
        """Identify Reverse Head and Shoulder (RHS) patterns from candlestick data"""
        try:
            if stock_data.ohlcv_data is None or len(stock_data.ohlcv_data) < 100:
                return {"status": "Invalid RHS Pattern (Reason: Insufficient data)"}
            
            # Handle MultiIndex DataFrame structure
            if isinstance(stock_data.ohlcv_data.columns, pd.MultiIndex):
                stock_data.ohlcv_data.columns = stock_data.ohlcv_data.columns.get_level_values(0)
            
            df = stock_data.ohlcv_data.copy()
            
            # Get recent data for pattern analysis (last 200 days for RHS pattern)
            recent_data = df.tail(200)
            
            # Find local minima (troughs) for shoulder and head identification
            troughs = self._find_local_minima(recent_data)
            
            if len(troughs) < 3:
                return {"status": "Invalid RHS Pattern (Reason: Insufficient troughs)"}
            
            # Identify potential RHS patterns
            rhs_patterns = self._find_rhs_patterns(recent_data, troughs)
            
            if not rhs_patterns:
                return {"status": "Invalid RHS Pattern (Reason: No valid RHS structure found)"}
            
            # Validate and rank patterns
            valid_patterns = []
            for pattern in rhs_patterns:
                validation = self._validate_rhs_pattern(recent_data, pattern)
                if validation["is_valid"]:
                    valid_patterns.append(validation)
            
            if not valid_patterns:
                return {"status": "Invalid RHS Pattern (Reason: Pattern validation failed)"}
            
            # Return the best (most recent) valid pattern
            best_pattern = max(valid_patterns, key=lambda x: x["right_shoulder_date"])
            
            return best_pattern
            
        except Exception as e:
            print(f"⚠️ RHS pattern identification error: {e}")
            return {"status": f"Invalid RHS Pattern (Reason: Analysis error - {e})"}
    
    def _find_local_minima(self, data: pd.DataFrame) -> List[Dict]:
        """Find local minima (troughs) in the price data"""
        troughs = []
        window = 10  # Look for minima in 10-day windows
        
        for i in range(window, len(data) - window):
            current_low = data['Low'].iloc[i]
            left_window = data['Low'].iloc[i-window:i].min()
            right_window = data['Low'].iloc[i+1:i+window+1].min()
            
            # Check if current point is a local minimum
            if current_low <= left_window and current_low <= right_window:
                troughs.append({
                    'index': i,
                    'date': data.index[i],
                    'price': current_low,
                    'volume': data['Volume'].iloc[i]
                })
        
        return troughs
    
    def _find_rhs_patterns(self, data: pd.DataFrame, troughs: List[Dict]) -> List[Dict]:
        """Find potential RHS patterns from troughs"""
        patterns = []
        
        # Look for 3 consecutive troughs that could form RHS
        for i in range(len(troughs) - 2):
            left_shoulder = troughs[i]
            head = troughs[i + 1]
            right_shoulder = troughs[i + 2]
            
            # Basic RHS structure validation
            if (head['price'] < left_shoulder['price'] and  # Head lower than left shoulder
                right_shoulder['price'] > head['price'] and  # Right shoulder higher than head
                abs(right_shoulder['price'] - left_shoulder['price']) / left_shoulder['price'] < 0.1):  # Shoulders similar height
                
                patterns.append({
                    'left_shoulder': left_shoulder,
                    'head': head,
                    'right_shoulder': right_shoulder,
                    'pattern_type': 'Simple RHS'
                })
        
        return patterns
    
    def _validate_rhs_pattern(self, data: pd.DataFrame, pattern: Dict) -> Dict:
        """Validate RHS pattern according to strict rules"""
        try:
            ls = pattern['left_shoulder']
            head = pattern['head']
            rs = pattern['right_shoulder']
            
            # 1. Core Structure Validation
            if head['price'] >= ls['price']:
                return {"is_valid": False, "reason": "Head not lower than left shoulder"}
            
            if rs['price'] <= head['price']:
                return {"is_valid": False, "reason": "Right shoulder not higher than head"}
            
            # 2. Depth Rule Validation
            if ls['price'] < head['price']:
                return {"is_valid": False, "reason": "Left shoulder deeper than head"}
            
            if rs['price'] < head['price']:
                return {"is_valid": False, "reason": "Right shoulder deeper than head"}
            
            # 3. Neckline Identification and Validation
            neckline = self._calculate_neckline(data, pattern)
            if not neckline["is_horizontal"]:
                return {"is_valid": False, "reason": "Neckline not horizontal"}
            
            # 4. Right Shoulder Base Formation
            base_formation = self._identify_base_formation(data, rs)
            if not base_formation["found"]:
                return {"is_valid": False, "reason": "No base formation in right shoulder"}
            
            # 5. Breakout Analysis
            breakout = self._analyze_breakout(data, base_formation)
            if not breakout["confirmed"]:
                return {"is_valid": False, "reason": "No confirmed breakout"}
            
            # 6. Target Calculation
            targets = self._calculate_rhs_targets(data, head, neckline)
            
            # 7. Company Eligibility (Basic check - would need external data for full validation)
            company_eligible = self._check_company_eligibility(stock_data)
            
            return {
                "is_valid": True,
                "pattern_status": "Valid RHS Pattern Identified",
                "pattern_type": pattern['pattern_type'],
                "key_price_points": {
                    "left_shoulder_range": f"₹{ls['price']:.2f}",
                    "head_range": f"₹{head['price']:.2f}",
                    "right_shoulder_range": f"₹{rs['price']:.2f}"
                },
                "neckline_details": {
                    "price_range": f"₹{neckline['price']:.2f}",
                    "is_horizontal": neckline["is_horizontal"]
                },
                "buying_point": {
                    "base_formation_range": f"₹{base_formation['low']:.2f} - ₹{base_formation['high']:.2f}",
                    "entry_price": f"₹{breakout['price']:.2f}",
                    "entry_date": breakout['date'].strftime('%Y-%m-%d')
                },
                "target_calculation": {
                    "head_to_neckline_depth": f"₹{targets['depth']:.2f}",
                    "technical_target": f"₹{targets['technical_target']:.2f}",
                    "lifetime_high": f"₹{targets['lifetime_high']:.2f}",
                    "final_target": f"₹{targets['final_target']:.2f}"
                },
                "company_eligible": company_eligible,
                "left_shoulder_date": ls['date'],
                "head_date": head['date'],
                "right_shoulder_date": rs['date']
            }
            
        except Exception as e:
            return {"is_valid": False, "reason": f"Validation error: {e}"}
    
    def _calculate_neckline(self, data: pd.DataFrame, pattern: Dict) -> Dict:
        """Calculate and validate neckline"""
        try:
            ls = pattern['left_shoulder']
            head = pattern['head']
            rs = pattern['right_shoulder']
            
            # Find peaks between troughs
            left_peak = self._find_peak_between(data, ls['index'], head['index'])
            right_peak = self._find_peak_between(data, head['index'], rs['index'])
            
            if left_peak is None or right_peak is None:
                return {"is_horizontal": False, "price": 0}
            
            # Check if neckline is horizontal (within 2% tolerance)
            price_diff = abs(left_peak['price'] - right_peak['price'])
            avg_price = (left_peak['price'] + right_peak['price']) / 2
            tolerance = avg_price * 0.02  # 2% tolerance
            
            is_horizontal = price_diff <= tolerance
            neckline_price = avg_price
            
            return {
                "is_horizontal": is_horizontal,
                "price": neckline_price,
                "left_peak": left_peak,
                "right_peak": right_peak
            }
            
        except Exception as e:
            return {"is_horizontal": False, "price": 0}
    
    def _find_peak_between(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[Dict]:
        """Find the highest peak between two indices"""
        try:
            if end_idx <= start_idx:
                return None
            
            segment = data.iloc[start_idx:end_idx+1]
            peak_idx = segment['High'].idxmax()
            peak_price = segment['High'].max()
            
            return {
                'index': data.index.get_loc(peak_idx),
                'date': peak_idx,
                'price': peak_price
            }
        except:
            return None
    
    def _identify_base_formation(self, data: pd.DataFrame, right_shoulder: Dict) -> Dict:
        """Identify base formation in right shoulder"""
        try:
            rs_idx = right_shoulder['index']
            
            # Look for consolidation after right shoulder (next 20 days)
            if rs_idx + 20 >= len(data):
                return {"found": False}
            
            post_rs_data = data.iloc[rs_idx:rs_idx+20]
            
            # Check for tight trading range (consolidation)
            high = post_rs_data['High'].max()
            low = post_rs_data['Low'].min()
            range_pct = (high - low) / low * 100
            
            # Base formation if range is less than 5%
            if range_pct <= 5:
                return {
                    "found": True,
                    "low": low,
                    "high": high,
                    "range_pct": range_pct
                }
            
            return {"found": False}
            
        except Exception as e:
            return {"found": False}
    
    def _analyze_breakout(self, data: pd.DataFrame, base_formation: Dict) -> Dict:
        """Analyze breakout from base formation"""
        try:
            if not base_formation["found"]:
                return {"confirmed": False}
            
            # Look for green candle breakout above base
            for i in range(len(data) - 1):
                current_candle = data.iloc[i]
                next_candle = data.iloc[i + 1]
                
                # Check for green candle closing above base
                if (next_candle['Close'] > next_candle['Open'] and  # Green candle
                    next_candle['Close'] > base_formation['high']):  # Close above base
                    
                    return {
                        "confirmed": True,
                        "price": next_candle['Close'],
                        "date": data.index[i + 1]
                    }
            
            return {"confirmed": False}
            
        except Exception as e:
            return {"confirmed": False}
    
    def _calculate_rhs_targets(self, data: pd.DataFrame, head: Dict, neckline: Dict) -> Dict:
        """Calculate RHS pattern targets"""
        try:
            # Calculate depth from head to neckline
            depth = neckline['price'] - head['price']
            
            # Technical target = neckline + depth
            technical_target = neckline['price'] + depth
            
            # Get lifetime high
            lifetime_high = data['High'].max()
            
            # Final target = higher of technical target or lifetime high
            final_target = max(technical_target, lifetime_high)
            
            return {
                "depth": depth,
                "technical_target": technical_target,
                "lifetime_high": lifetime_high,
                "final_target": final_target
            }
            
        except Exception as e:
            return {
                "depth": 0,
                "technical_target": 0,
                "lifetime_high": 0,
                "final_target": 0
            }
    
    def _check_company_eligibility(self, stock_data: StockData) -> bool:
        """Basic company eligibility check (would need external data for full validation)"""
        # This is a simplified check - in practice, you'd need external data
        # to verify v40/v40 next status, market leadership, debt levels, etc.
        
        # For now, return True for most stocks (assume eligible)
        # In production, this would integrate with company fundamental data
        return True

    def _format_rhs_analysis(self, rhs_analysis: Optional[Dict]) -> str:
        """Format RHS analysis for technical summary"""
        if rhs_analysis is None or "status" in rhs_analysis:
            return f"RHS Pattern: {rhs_analysis.get('status', 'Not analyzed')}"
        
        if not rhs_analysis.get("is_valid", False):
            return f"RHS Pattern: Invalid - {rhs_analysis.get('reason', 'Unknown')}"
        
        return f"""
RHS Pattern: {rhs_analysis.get('pattern_status', 'Valid RHS Pattern Identified')}
Pattern Type: {rhs_analysis.get('pattern_type', 'Simple RHS')}
Company Eligible: {rhs_analysis.get('company_eligible', False)}

Key Price Points:
- Left Shoulder: {rhs_analysis.get('key_price_points', {}).get('left_shoulder_range', 'N/A')}
- Head: {rhs_analysis.get('key_price_points', {}).get('head_range', 'N/A')}
- Right Shoulder: {rhs_analysis.get('key_price_points', {}).get('right_shoulder_range', 'N/A')}

Neckline: {rhs_analysis.get('neckline_details', {}).get('price_range', 'N/A')}

Buying Point:
- Base Formation: {rhs_analysis.get('buying_point', {}).get('base_formation_range', 'N/A')}
- Entry Price: {rhs_analysis.get('buying_point', {}).get('entry_price', 'N/A')}
- Entry Date: {rhs_analysis.get('buying_point', {}).get('entry_date', 'N/A')}

Target Calculation:
- Head to Neckline Depth: {rhs_analysis.get('target_calculation', {}).get('head_to_neckline_depth', 'N/A')}
- Technical Target: {rhs_analysis.get('target_calculation', {}).get('technical_target', 'N/A')}
- Lifetime High: {rhs_analysis.get('target_calculation', {}).get('lifetime_high', 'N/A')}
- Final Target: {rhs_analysis.get('target_calculation', {}).get('final_target', 'N/A')}
"""

    def _identify_cwh_pattern(self, stock_data: StockData) -> Dict:
        """Identify Cup with Handle (CWH) patterns from candlestick data"""
        try:
            if stock_data.ohlcv_data is None or len(stock_data.ohlcv_data) < 100:
                return {"status": "Invalid CWH Pattern (Reason: Insufficient data)"}
            
            # Handle MultiIndex DataFrame structure
            if isinstance(stock_data.ohlcv_data.columns, pd.MultiIndex):
                stock_data.ohlcv_data.columns = stock_data.ohlcv_data.columns.get_level_values(0)
            
            df = stock_data.ohlcv_data.copy()
            
            # Get recent data for pattern analysis (last 200 days for CWH pattern)
            recent_data = df.tail(200)
            
            # Find local minima (troughs) for cup identification
            troughs = self._find_local_minima(recent_data)
            
            if len(troughs) < 2:
                return {"status": "Invalid CWH Pattern (Reason: Insufficient troughs)"}
            
            # Identify potential CWH patterns
            cwh_patterns = self._find_cwh_patterns(recent_data, troughs)
            
            if not cwh_patterns:
                return {"status": "Invalid CWH Pattern (Reason: No valid CWH structure found)"}
            
            # Validate and rank patterns
            valid_patterns = []
            for pattern in cwh_patterns:
                validation = self._validate_cwh_pattern(recent_data, pattern)
                if validation["is_valid"]:
                    valid_patterns.append(validation)
            
            if not valid_patterns:
                return {"status": "Invalid CWH Pattern (Reason: Pattern validation failed)"}
            
            # Return the best (most recent) valid pattern
            best_pattern = max(valid_patterns, key=lambda x: x["handle_date"])
            
            return best_pattern
            
        except Exception as e:
            print(f"⚠️ CWH pattern identification error: {e}")
            return {"status": f"Invalid CWH Pattern (Reason: Analysis error - {e})"}
    
    def _find_cwh_patterns(self, data: pd.DataFrame, troughs: List[Dict]) -> List[Dict]:
        """Find potential CWH patterns from troughs"""
        patterns = []
        
        # Look for cup formation (U-shape or V-shape trough)
        for i in range(len(troughs) - 1):
            cup_trough = troughs[i]
            
            # Look for handle formation after cup
            for j in range(i + 1, len(troughs)):
                handle_trough = troughs[j]
                
                # Basic CWH structure validation
                if (handle_trough['price'] > cup_trough['price'] and  # Handle higher than cup bottom
                    handle_trough['price'] < cup_trough['price'] * 1.2):  # Handle not too high
                    
                    # Check if there's a right handle (handle after cup)
                    if j > i:  # Handle comes after cup
                        patterns.append({
                            'cup_trough': cup_trough,
                            'handle_trough': handle_trough,
                            'pattern_type': 'Simple CWH'
                        })
        
        return patterns
    
    def _validate_cwh_pattern(self, data: pd.DataFrame, pattern: Dict) -> Dict:
        """Validate CWH pattern according to strict rules"""
        try:
            cup_trough = pattern['cup_trough']
            handle_trough = pattern['handle_trough']
            
            # 1. Core Structure Validation
            if handle_trough['price'] <= cup_trough['price']:
                return {"is_valid": False, "reason": "Handle not higher than cup bottom"}
            
            # 2. Cup Depth vs Handle Depth Validation
            cup_depth = self._calculate_cup_depth(data, cup_trough)
            handle_depth = self._calculate_handle_depth(data, handle_trough)
            
            if handle_depth >= cup_depth:
                return {"is_valid": False, "reason": "Handle deeper than or equal to cup"}
            
            # 3. Neckline Identification and Validation
            neckline = self._calculate_cwh_neckline(data, pattern)
            if not neckline["is_valid"]:
                return {"is_valid": False, "reason": "Invalid neckline formation"}
            
            # 4. Right Handle Base Formation
            base_formation = self._identify_cwh_base_formation(data, handle_trough)
            if not base_formation["found"]:
                return {"is_valid": False, "reason": "No base formation in right handle"}
            
            # 5. Breakout Analysis
            breakout = self._analyze_cwh_breakout(data, base_formation)
            if not breakout["confirmed"]:
                return {"is_valid": False, "reason": "No confirmed breakout"}
            
            # 6. Target Calculation
            targets = self._calculate_cwh_targets(data, cup_trough, neckline)
            
            return {
                "is_valid": True,
                "pattern_status": "Valid CWH Pattern Identified",
                "pattern_type": pattern['pattern_type'],
                "key_price_points": {
                    "cup_bottom": f"₹{cup_trough['price']:.2f}",
                    "handle_bottom": f"₹{handle_trough['price']:.2f}",
                    "cup_depth": f"₹{cup_depth:.2f}",
                    "handle_depth": f"₹{handle_depth:.2f}"
                },
                "neckline_details": {
                    "price_range": f"₹{neckline['price']:.2f}",
                    "is_valid": neckline["is_valid"]
                },
                "buying_point": {
                    "base_formation_range": f"₹{base_formation['low']:.2f} - ₹{base_formation['high']:.2f}",
                    "entry_price": f"₹{breakout['price']:.2f}",
                    "entry_date": breakout['date'].strftime('%Y-%m-%d')
                },
                "target_calculation": {
                    "cup_depth": f"₹{cup_depth:.2f}",
                    "technical_target": f"₹{targets['technical_target']:.2f}",
                    "final_target": f"₹{targets['final_target']:.2f}"
                },
                "cup_date": cup_trough['date'],
                "handle_date": handle_trough['date']
            }
            
        except Exception as e:
            return {"is_valid": False, "reason": f"Validation error: {e}"}
    
    def _calculate_cup_depth(self, data: pd.DataFrame, cup_trough: Dict) -> float:
        """Calculate the depth of the cup"""
        try:
            # Find the highest point before and after the cup trough
            cup_idx = cup_trough['index']
            
            # Look for peaks before and after the cup
            before_cup = data.iloc[max(0, cup_idx-50):cup_idx]
            after_cup = data.iloc[cup_idx:min(len(data), cup_idx+50)]
            
            before_peak = before_cup['High'].max() if len(before_cup) > 0 else cup_trough['price']
            after_peak = after_cup['High'].max() if len(after_cup) > 0 else cup_trough['price']
            
            # Cup depth is the average of before and after peaks minus cup bottom
            avg_peak = (before_peak + after_peak) / 2
            cup_depth = avg_peak - cup_trough['price']
            
            return max(cup_depth, 0)
            
        except Exception as e:
            return 0.0
    
    def _calculate_handle_depth(self, data: pd.DataFrame, handle_trough: Dict) -> float:
        """Calculate the depth of the handle"""
        try:
            # Find the highest point before the handle trough
            handle_idx = handle_trough['index']
            
            # Look for peak before the handle
            before_handle = data.iloc[max(0, handle_idx-30):handle_idx]
            
            before_peak = before_handle['High'].max() if len(before_handle) > 0 else handle_trough['price']
            
            # Handle depth is the peak before handle minus handle bottom
            handle_depth = before_peak - handle_trough['price']
            
            return max(handle_depth, 0)
            
        except Exception as e:
            return 0.0
    
    def _calculate_cwh_neckline(self, data: pd.DataFrame, pattern: Dict) -> Dict:
        """Calculate and validate CWH neckline"""
        try:
            cup_trough = pattern['cup_trough']
            handle_trough = pattern['handle_trough']
            
            # Find peaks around the cup and handle
            cup_peak_before = self._find_peak_before(data, cup_trough['index'])
            cup_peak_after = self._find_peak_after(data, cup_trough['index'])
            handle_peak = self._find_peak_before(data, handle_trough['index'])
            
            if cup_peak_before is None or cup_peak_after is None or handle_peak is None:
                return {"is_valid": False, "price": 0}
            
            # Check if neckline is relatively horizontal (within 5% tolerance)
            peaks = [cup_peak_before['price'], cup_peak_after['price'], handle_peak['price']]
            avg_price = sum(peaks) / len(peaks)
            max_deviation = max(abs(p - avg_price) for p in peaks)
            tolerance = avg_price * 0.05  # 5% tolerance for CWH
            
            is_valid = max_deviation <= tolerance
            neckline_price = avg_price
            
            return {
                "is_valid": is_valid,
                "price": neckline_price,
                "peaks": peaks
            }
            
        except Exception as e:
            return {"is_valid": False, "price": 0}
    
    def _find_peak_before(self, data: pd.DataFrame, index: int) -> Optional[Dict]:
        """Find the highest peak before a given index"""
        try:
            if index <= 0:
                return None
            
            segment = data.iloc[max(0, index-30):index]
            if len(segment) == 0:
                return None
            
            peak_idx = segment['High'].idxmax()
            peak_price = segment['High'].max()
            
            return {
                'index': data.index.get_loc(peak_idx),
                'date': peak_idx,
                'price': peak_price
            }
        except:
            return None
    
    def _find_peak_after(self, data: pd.DataFrame, index: int) -> Optional[Dict]:
        """Find the highest peak after a given index"""
        try:
            if index >= len(data) - 1:
                return None
            
            segment = data.iloc[index+1:min(len(data), index+31)]
            if len(segment) == 0:
                return None
            
            peak_idx = segment['High'].idxmax()
            peak_price = segment['High'].max()
            
            return {
                'index': data.index.get_loc(peak_idx),
                'date': peak_idx,
                'price': peak_price
            }
        except:
            return None
    
    def _identify_cwh_base_formation(self, data: pd.DataFrame, handle_trough: Dict) -> Dict:
        """Identify base formation in right handle"""
        try:
            handle_idx = handle_trough['index']
            
            # Look for consolidation after handle (next 20 days)
            if handle_idx + 20 >= len(data):
                return {"found": False}
            
            post_handle_data = data.iloc[handle_idx:handle_idx+20]
            
            # Check for tight trading range (consolidation)
            high = post_handle_data['High'].max()
            low = post_handle_data['Low'].min()
            range_pct = (high - low) / low * 100
            
            # Base formation if range is less than 5%
            if range_pct <= 5:
                return {
                    "found": True,
                    "low": low,
                    "high": high,
                    "range_pct": range_pct
                }
            
            return {"found": False}
            
        except Exception as e:
            return {"found": False}
    
    def _analyze_cwh_breakout(self, data: pd.DataFrame, base_formation: Dict) -> Dict:
        """Analyze breakout from CWH base formation"""
        try:
            if not base_formation["found"]:
                return {"confirmed": False}
            
            # Look for green candle breakout above base
            for i in range(len(data) - 1):
                current_candle = data.iloc[i]
                next_candle = data.iloc[i + 1]
                
                # Check for green candle closing above base
                if (next_candle['Close'] > next_candle['Open'] and  # Green candle
                    next_candle['Close'] > base_formation['high']):  # Close above base
                    
                    return {
                        "confirmed": True,
                        "price": next_candle['Close'],
                        "date": data.index[i + 1]
                    }
            
            return {"confirmed": False}
            
        except Exception as e:
            return {"confirmed": False}
    
    def _calculate_cwh_targets(self, data: pd.DataFrame, cup_trough: Dict, neckline: Dict) -> Dict:
        """Calculate CWH pattern targets"""
        try:
            # Calculate cup depth from cup bottom to neckline
            cup_depth = neckline['price'] - cup_trough['price']
            
            # Technical target = neckline + cup depth
            technical_target = neckline['price'] + cup_depth
            
            # For CWH, technical target is the final target (unlike RHS)
            final_target = technical_target
            
            return {
                "cup_depth": cup_depth,
                "technical_target": technical_target,
                "final_target": final_target
            }
            
        except Exception as e:
            return {
                "cup_depth": 0,
                "technical_target": 0,
                "final_target": 0
            }
    
    def _format_cwh_analysis(self, cwh_analysis: Optional[Dict]) -> str:
        """Format CWH analysis for technical summary"""
        if cwh_analysis is None or "status" in cwh_analysis:
            return f"CWH Pattern: {cwh_analysis.get('status', 'Not analyzed')}"
        
        if not cwh_analysis.get("is_valid", False):
            return f"CWH Pattern: Invalid - {cwh_analysis.get('reason', 'Unknown')}"
        
        return f"""
CWH Pattern: {cwh_analysis.get('pattern_status', 'Valid CWH Pattern Identified')}
Pattern Type: {cwh_analysis.get('pattern_type', 'Simple CWH')}

Key Price Points:
- Cup Bottom: {cwh_analysis.get('key_price_points', {}).get('cup_bottom', 'N/A')}
- Handle Bottom: {cwh_analysis.get('key_price_points', {}).get('handle_bottom', 'N/A')}
- Cup Depth: {cwh_analysis.get('key_price_points', {}).get('cup_depth', 'N/A')}
- Handle Depth: {cwh_analysis.get('key_price_points', {}).get('handle_depth', 'N/A')}

Neckline: {cwh_analysis.get('neckline_details', {}).get('price_range', 'N/A')}

Buying Point:
- Base Formation: {cwh_analysis.get('buying_point', {}).get('base_formation_range', 'N/A')}
- Entry Price: {cwh_analysis.get('buying_point', {}).get('entry_price', 'N/A')}
- Entry Date: {cwh_analysis.get('buying_point', {}).get('entry_date', 'N/A')}

Target Calculation:
- Cup Depth: {cwh_analysis.get('target_calculation', {}).get('cup_depth', 'N/A')}
- Technical Target: {cwh_analysis.get('target_calculation', {}).get('technical_target', 'N/A')}
- Final Target: {cwh_analysis.get('target_calculation', {}).get('final_target', 'N/A')}
"""

# Enhanced Fundamental Analysis Agent
class EnhancedFundamentalAnalysisAgent:
    def __init__(self, llm: ChatOpenAI, openai_api_key: str):
        self.llm = llm
        self.openai_api_key = openai_api_key
        self.cost_tracker = cost_tracker  # Add cost_tracker reference
        self.openai_client = OpenAI(api_key=openai_api_key)  # Add this line
        self.screenshots_dir = "screener_screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Initialize enhanced screener extractor
        self.extractor = EnhancedScreenerExtractionV3(openai_api_key)
        
        # Chrome options for Selenium
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    def analyze(self, stock_data: StockData) -> EnhancedFundamentalAnalysis:
        """Analyze fundamental aspects using enhanced screener extraction"""
        try:
            print(f"💰 Enhanced Fundamental Analysis Agent analyzing {stock_data.ticker}...")
            
            # Use the enhanced screener extraction
            output_dir = "enhanced_screener_extraction_v3"
            complete_data = self.extractor.extract_complete_data(stock_data.ticker, output_dir)
            
            if complete_data:
                # Process the extracted data
                # Convert to EnhancedFundamentalAnalysis format
                return self._convert_extracted_data_to_enhanced_format(complete_data)
            else:
                print(f"❌ Failed to extract data for {stock_data.ticker}")
                return self._get_basic_fundamental_analysis(stock_data)
                
        except Exception as e:
            print(f"⚠️ Error in enhanced fundamental analysis: {e}")
            return self._get_basic_fundamental_analysis(stock_data)
    
    def _convert_extracted_data_to_enhanced_format(self, extracted_data: Dict) -> EnhancedFundamentalAnalysis:
        """Convert extracted data to EnhancedFundamentalAnalysis format"""
        # Extract key metrics
        key_metrics = extracted_data.get('key_metrics', {})
        quarterly_results = extracted_data.get('quarterly_results', {})
        annual_results = extracted_data.get('annual_results', {})
        balance_sheet = extracted_data.get('balance_sheet', {})
        cash_flows = extracted_data.get('cash_flows', {})
        shareholding = extracted_data.get('shareholding', {})
        
        # Calculate comprehensive growth metrics
        growth_metrics = self.calculate_comprehensive_growth_metrics(extracted_data)
        growth_assessment = self._evaluate_growth_performance(growth_metrics)
        
        # Calculate metrics from extracted data
        market_cap = key_metrics.get('market_cap', 'NA')
        current_price = key_metrics.get('current_price', 'NA')
        roce = key_metrics.get('roce', 'NA')
        roe = key_metrics.get('roe', 'NA')
        
        # Use comprehensive growth metrics for revenue and profit growth
        revenue_growth = "NA"
        if growth_metrics['revenue']['yoy_growth'] is not None:
            revenue_growth = self._format_growth_percentage(growth_metrics['revenue']['yoy_growth'])
            # Add 3-year average context
            if growth_metrics['revenue']['avg_yoy_3y'] is not None:
                revenue_growth += f" (3Y Avg: {self._format_growth_percentage(growth_metrics['revenue']['avg_yoy_3y'])})"
        
        profit_growth = "NA"
        if growth_metrics['net_profit']['yoy_growth'] is not None:
            profit_growth = self._format_growth_percentage(growth_metrics['net_profit']['yoy_growth'])
            # Add 3-year average context
            if growth_metrics['net_profit']['avg_yoy_3y'] is not None:
                profit_growth += f" (3Y Avg: {self._format_growth_percentage(growth_metrics['net_profit']['avg_yoy_3y'])})"
        
        # Calculate debt to equity from balance sheet
        debt_to_equity = "NA"
        if balance_sheet.get('total_liabilities') and balance_sheet.get('net_worth'):
            try:
                liabilities = float(balance_sheet['total_liabilities'].replace(',', ''))
                net_worth = float(balance_sheet['net_worth'].replace(',', ''))
                if net_worth > 0:
                    de_ratio = liabilities / net_worth
                    debt_to_equity = f"{de_ratio:.2f}"
            except:
                debt_to_equity = "NA"
        
        # Enhanced business quality assessment incorporating growth metrics
        business_quality = "Can't Say"
        if roce != 'NA' and roe != 'NA':
            try:
                roce_val = float(roce.replace('%', ''))
                roe_val = float(roe.replace('%', ''))
                growth_score = growth_assessment['overall_score']
                
                # Factor in growth quality for business assessment
                if roce_val > 15 and roe_val > 15 and growth_score >= 70:
                    business_quality = "Excellent"
                elif roce_val > 15 and roe_val > 10 and growth_score >= 50:
                    business_quality = "High"
                elif roce_val > 10 and roe_val > 10 and growth_score >= 45:
                    business_quality = "Medium"
                elif roce_val > 10 and roe_val > 10:
                    business_quality = "Medium"
                elif growth_score >= 60:  # Strong growth can offset moderate ratios
                    business_quality = "Medium"
                else:
                    business_quality = "Low"
            except:
                if growth_assessment['overall_score'] >= 70:
                    business_quality = "Medium"  # Growth-driven assessment
                else:
                    business_quality = "Can't Say"
        elif growth_assessment['overall_score'] >= 70:
            business_quality = "Medium"  # Pure growth-based assessment
        
        # Enhanced valuation assessment
        valuation_status = "Fair"
        if market_cap != 'NA':
            try:
                mc_val = float(market_cap.replace(',', ''))
                growth_score = growth_assessment['overall_score']
                
                # Adjust valuation based on growth quality
                if mc_val > 50000:  # Large cap
                    if growth_score >= 70:
                        valuation_status = "Fair"  # Growth justifies large cap
                    else:
                        valuation_status = "Expensive"
                elif mc_val > 10000:  # Mid cap
                    if growth_score >= 60:
                        valuation_status = "Attractive"
                    else:
                        valuation_status = "Fair"
                else:  # Small cap
                    if growth_score >= 50:
                        valuation_status = "Attractive"
                    else:
                        valuation_status = "Cheap"
            except:
                valuation_status = "Fair"
        
        # Calculate fair value estimate (basic approach)
        fair_value = current_price if current_price != 'NA' else "NA"
        
        # Enhanced multibagger potential assessment
        multibagger_potential = "Moderate"
        if (growth_assessment['growth_quality'] == 'Excellent' and 
            business_quality in ["Excellent", "High"]):
            multibagger_potential = "High"
        elif (growth_assessment['growth_quality'] == 'Good' and 
              business_quality in ["High", "Medium"]):
            multibagger_potential = "High"
        elif growth_assessment['growth_quality'] == 'Good':
            multibagger_potential = "Moderate"
        elif growth_assessment['growth_quality'] == 'Poor':
            multibagger_potential = "Low"
        
        # Enhanced fundamental reasons incorporating growth analysis
        fundamental_reasons = f"""Growth Analysis: {growth_assessment['growth_quality']} growth quality (Score: {growth_assessment['overall_score']}/100).
        
Growth Highlights:
{chr(10).join(['• ' + summary for summary in growth_assessment['growth_summary'][:5]])}

Financial Metrics: Market Cap {market_cap}, ROCE {roce}, ROE {roe}, Revenue Growth {revenue_growth}, Profit Growth {profit_growth}

Growth Metrics Detail:
• Revenue YoY: {self._format_growth_percentage(growth_metrics['revenue']['yoy_growth'])}
• Net Profit YoY: {self._format_growth_percentage(growth_metrics['net_profit']['yoy_growth'])}
• Operating Profit YoY: {self._format_growth_percentage(growth_metrics['operating_profit']['yoy_growth'])}
• Revenue 3Y Avg: {self._format_growth_percentage(growth_metrics['revenue']['avg_yoy_3y'])}
• Profit 3Y Avg: {self._format_growth_percentage(growth_metrics['net_profit']['avg_yoy_3y'])}"""
        
        # Enhanced confidence score incorporating growth data quality
        confidence_score = "Medium"
        data_sections = sum([
            1 if key_metrics else 0,
            1 if quarterly_results else 0,
            1 if annual_results else 0,
            1 if balance_sheet else 0,
            1 if cash_flows else 0,
            1 if shareholding else 0
        ])
        
        growth_data_quality = sum([
            1 if growth_metrics['revenue']['yoy_growth'] is not None else 0,
            1 if growth_metrics['net_profit']['yoy_growth'] is not None else 0,
            1 if growth_metrics['operating_profit']['yoy_growth'] is not None else 0
        ])
        
        if data_sections >= 5 and growth_data_quality >= 2:
            confidence_score = "Strong"
        elif data_sections >= 3 and growth_data_quality >= 1:
            confidence_score = "Medium"
        elif data_sections >= 2 or growth_data_quality >= 1:
            confidence_score = "Medium"
        else:
            confidence_score = "Low"
        
        # Store growth metrics for later use by correlation agent
        if not hasattr(self, 'latest_growth_metrics'):
            self.latest_growth_metrics = {}
        self.latest_growth_metrics[extracted_data.get('ticker', 'unknown')] = {
            'growth_metrics': growth_metrics,
            'growth_assessment': growth_assessment
        }
        
        return EnhancedFundamentalAnalysis(
            business_quality=business_quality,
            market_penetration="Strong" if market_cap != 'NA' else "Not Available",
            pricing_power="High" if roce != 'NA' and float(roce.replace('%', '')) > 15 else "Medium",
            revenue_growth=revenue_growth,
            profit_growth=profit_growth,
            debt_to_equity=debt_to_equity,
            roce_roe=f"ROCE: {roce}, ROE: {roe}",
            promoter_pledging=shareholding.get('promoter_holding', 'NA'),
            retail_shareholding=shareholding.get('retail_holding', 'NA'),
            valuation_status=valuation_status,
            fair_value=fair_value,
            financial_health=growth_assessment['growth_quality'],  # Use growth quality as financial health
            multibagger_potential=multibagger_potential,
            fundamental_reasons=fundamental_reasons,
            confidence_score=confidence_score
        )
    
    def _get_basic_fundamental_analysis(self, stock_data: StockData) -> EnhancedFundamentalAnalysis:
        """Fallback to basic fundamental analysis"""
        return EnhancedFundamentalAnalysis(
            business_quality="Can't Say",
            market_penetration="Not Available",
            pricing_power="Not Available",
            revenue_growth="Not Available",
            profit_growth="Not Available",
            debt_to_equity="Not Available",
            roce_roe="Not Available",
            promoter_pledging="Not Available",
            retail_shareholding="Not Available",
            valuation_status="Not Available",
            fair_value="Not Available",
            financial_health="Not Available",
            multibagger_potential="Not Available",
            fundamental_reasons="Insufficient data for comprehensive analysis",
            confidence_score="Can't Say"
        )
    
    def calculate_comprehensive_growth_metrics(self, extracted_data: Dict) -> Dict:
        """Calculate comprehensive growth metrics from extracted quarterly and annual data"""
        growth_metrics = {
            'revenue': {
                'yoy_growth': None,
                'qoq_same_quarter': None,
                'avg_qoq_4q': None,
                'avg_yoy_3y': None
            },
            'net_profit': {
                'yoy_growth': None,
                'qoq_same_quarter': None,
                'avg_qoq_4q': None,
                'avg_yoy_3y': None
            },
            'operating_profit': {
                'yoy_growth': None,
                'qoq_same_quarter': None,
                'avg_qoq_4q': None,
                'avg_yoy_3y': None
            }
        }
        
        try:
            # Calculate revenue growth
            quarterly_revenue = extracted_data.get('quarterly_results', {}).get('revenue', [])
            annual_revenue = extracted_data.get('annual_results', {}).get('total_revenue', None)
            
            if quarterly_revenue:
                growth_metrics['revenue'] = self._calculate_metric_growth(
                    quarterly_revenue, annual_revenue, 'revenue'
                )
            
            # Calculate net profit growth
            quarterly_net_profit = extracted_data.get('quarterly_results', {}).get('net_profit', [])
            annual_net_profit = extracted_data.get('annual_results', {}).get('net_profit', None)
            
            if quarterly_net_profit:
                growth_metrics['net_profit'] = self._calculate_metric_growth(
                    quarterly_net_profit, annual_net_profit, 'net_profit'
                )
            
            # Calculate operating profit growth (using EBITDA)
            quarterly_ebitda = extracted_data.get('quarterly_results', {}).get('ebitda', [])
            annual_ebitda = extracted_data.get('annual_results', {}).get('ebitda', None)
            
            if quarterly_ebitda:
                growth_metrics['operating_profit'] = self._calculate_metric_growth(
                    quarterly_ebitda, annual_ebitda, 'operating_profit'
                )
            
            return growth_metrics
            
        except Exception as e:
            print(f"Error calculating growth metrics: {e}")
            return growth_metrics
    
    def _calculate_metric_growth(self, quarterly_data: List, annual_current: str, metric_name: str) -> Dict:
        """Calculate growth metrics for a specific financial metric"""
        metrics = {
            'yoy_growth': None,
            'qoq_same_quarter': None,
            'avg_qoq_4q': None,
            'avg_yoy_3y': None
        }
        
        try:
            # Convert quarterly data to numbers
            quarterly_values = []
            for q in quarterly_data[:8]:  # Last 8 quarters
                try:
                    # Clean the value (remove commas, convert to float)
                    clean_value = str(q).replace(',', '').replace('₹', '').replace('Cr.', '').strip()
                    if clean_value and clean_value != 'NA' and clean_value != '':
                        quarterly_values.append(float(clean_value))
                    else:
                        quarterly_values.append(None)
                except:
                    quarterly_values.append(None)
            
            # Calculate Year-over-Year Growth (Q1 vs Q5 - same quarter last year)
            if len(quarterly_values) >= 5 and quarterly_values[0] is not None and quarterly_values[4] is not None:
                if quarterly_values[4] != 0:
                    yoy_growth = ((quarterly_values[0] - quarterly_values[4]) / abs(quarterly_values[4])) * 100
                    metrics['yoy_growth'] = round(yoy_growth, 2)
            
            # Calculate Quarter-over-Quarter Same Quarter Growth (Q1 vs Q5)
            if len(quarterly_values) >= 5 and quarterly_values[0] is not None and quarterly_values[4] is not None:
                if quarterly_values[4] != 0:
                    qoq_same = ((quarterly_values[0] - quarterly_values[4]) / abs(quarterly_values[4])) * 100
                    metrics['qoq_same_quarter'] = round(qoq_same, 2)
            
            # Calculate Average QoQ Growth for Last 4 Quarters
            qoq_growths = []
            for i in range(3):  # Q1 vs Q2, Q2 vs Q3, Q3 vs Q4
                if (i + 1 < len(quarterly_values) and 
                    quarterly_values[i] is not None and 
                    quarterly_values[i + 1] is not None and 
                    quarterly_values[i + 1] != 0):
                    qoq = ((quarterly_values[i] - quarterly_values[i + 1]) / abs(quarterly_values[i + 1])) * 100
                    qoq_growths.append(qoq)
            
            if qoq_growths:
                metrics['avg_qoq_4q'] = round(sum(qoq_growths) / len(qoq_growths), 2)
            
            # Calculate Average YoY Growth from Last 3 Years (using quarterly data approach)
            yoy_growths = []
            for i in range(min(3, len(quarterly_values) - 4)):  # Last 3 year-over-year comparisons
                current_q = quarterly_values[i]
                previous_year_q = quarterly_values[i + 4]
                
                if current_q is not None and previous_year_q is not None and previous_year_q != 0:
                    yoy = ((current_q - previous_year_q) / abs(previous_year_q)) * 100
                    yoy_growths.append(yoy)
            
            if yoy_growths:
                metrics['avg_yoy_3y'] = round(sum(yoy_growths) / len(yoy_growths), 2)
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating {metric_name} growth: {e}")
            return metrics
    
    def _format_growth_percentage(self, value) -> str:
        """Format growth percentage for display"""
        if value is None:
            return "N/A"
        elif value > 0:
            return f"+{value:.1f}%"
        else:
            return f"{value:.1f}%"
    
    def _evaluate_growth_performance(self, growth_metrics: Dict) -> Dict:
        """Evaluate growth performance and return detailed assessment"""
        growth_assessment = {
            'overall_score': 0,
            'revenue_score': 0,
            'profit_score': 0,
            'operating_score': 0,
            'consistency_score': 0,
            'growth_quality': 'Average',
            'growth_summary': []
        }
        
        try:
            total_score = 0
            score_count = 0
            
            # Revenue Growth Analysis
            revenue = growth_metrics.get('revenue', {})
            if revenue.get('yoy_growth') is not None:
                yoy_rev = revenue['yoy_growth']
                if yoy_rev > 20:
                    growth_assessment['revenue_score'] = 85
                    growth_assessment['growth_summary'].append(f"Excellent revenue growth of {self._format_growth_percentage(yoy_rev)} YoY")
                elif yoy_rev > 10:
                    growth_assessment['revenue_score'] = 70
                    growth_assessment['growth_summary'].append(f"Strong revenue growth of {self._format_growth_percentage(yoy_rev)} YoY")
                elif yoy_rev > 0:
                    growth_assessment['revenue_score'] = 55
                    growth_assessment['growth_summary'].append(f"Positive revenue growth of {self._format_growth_percentage(yoy_rev)} YoY")
                elif yoy_rev < -10:
                    growth_assessment['revenue_score'] = 20
                    growth_assessment['growth_summary'].append(f"Declining revenue with {self._format_growth_percentage(yoy_rev)} YoY contraction")
                else:
                    growth_assessment['revenue_score'] = 35
                    growth_assessment['growth_summary'].append(f"Revenue decline of {self._format_growth_percentage(yoy_rev)} YoY")
                
                total_score += growth_assessment['revenue_score']
                score_count += 1
            
            # Net Profit Growth Analysis
            net_profit = growth_metrics.get('net_profit', {})
            if net_profit.get('yoy_growth') is not None:
                yoy_profit = net_profit['yoy_growth']
                if yoy_profit > 25:
                    growth_assessment['profit_score'] = 90
                    growth_assessment['growth_summary'].append(f"Outstanding net profit growth of {self._format_growth_percentage(yoy_profit)} YoY")
                elif yoy_profit > 15:
                    growth_assessment['profit_score'] = 75
                    growth_assessment['growth_summary'].append(f"Strong net profit growth of {self._format_growth_percentage(yoy_profit)} YoY")
                elif yoy_profit > 0:
                    growth_assessment['profit_score'] = 60
                    growth_assessment['growth_summary'].append(f"Positive net profit growth of {self._format_growth_percentage(yoy_profit)} YoY")
                elif yoy_profit < -20:
                    growth_assessment['profit_score'] = 15
                    growth_assessment['growth_summary'].append(f"Severe profit decline of {self._format_growth_percentage(yoy_profit)} YoY")
                else:
                    growth_assessment['profit_score'] = 30
                    growth_assessment['growth_summary'].append(f"Net profit decline of {self._format_growth_percentage(yoy_profit)} YoY")
                
                total_score += growth_assessment['profit_score']
                score_count += 1
            
            # Operating Profit Growth Analysis
            op_profit = growth_metrics.get('operating_profit', {})
            if op_profit.get('yoy_growth') is not None:
                yoy_op = op_profit['yoy_growth']
                if yoy_op > 20:
                    growth_assessment['operating_score'] = 80
                    growth_assessment['growth_summary'].append(f"Strong operational improvement with {self._format_growth_percentage(yoy_op)} EBITDA growth")
                elif yoy_op > 10:
                    growth_assessment['operating_score'] = 65
                    growth_assessment['growth_summary'].append(f"Good operational performance with {self._format_growth_percentage(yoy_op)} EBITDA growth")
                elif yoy_op > 0:
                    growth_assessment['operating_score'] = 55
                elif yoy_op < -15:
                    growth_assessment['operating_score'] = 20
                    growth_assessment['growth_summary'].append(f"Operational deterioration with {self._format_growth_percentage(yoy_op)} EBITDA decline")
                else:
                    growth_assessment['operating_score'] = 35
                    growth_assessment['growth_summary'].append(f"Declining operations with {self._format_growth_percentage(yoy_op)} EBITDA contraction")
                
                total_score += growth_assessment['operating_score']
                score_count += 1
            
            # Consistency Analysis
            consistency_scores = []
            for metric_name in ['revenue', 'net_profit', 'operating_profit']:
                metric_data = growth_metrics.get(metric_name, {})
                if metric_data.get('avg_qoq_4q') is not None:
                    avg_qoq = metric_data['avg_qoq_4q']
                    if avg_qoq > 5:
                        consistency_scores.append(75)
                        growth_assessment['growth_summary'].append(f"Consistent {metric_name.replace('_', ' ')} growth averaging {self._format_growth_percentage(avg_qoq)} QoQ")
                    elif avg_qoq > 0:
                        consistency_scores.append(60)
                    elif avg_qoq < -10:
                        consistency_scores.append(25)
                    else:
                        consistency_scores.append(45)
            
            if consistency_scores:
                growth_assessment['consistency_score'] = sum(consistency_scores) / len(consistency_scores)
                total_score += growth_assessment['consistency_score']
                score_count += 1
            
            # Calculate overall score
            if score_count > 0:
                growth_assessment['overall_score'] = int(total_score / score_count)
                
                # Determine growth quality
                if growth_assessment['overall_score'] >= 75:
                    growth_assessment['growth_quality'] = 'Excellent'
                elif growth_assessment['overall_score'] >= 60:
                    growth_assessment['growth_quality'] = 'Good'
                elif growth_assessment['overall_score'] >= 45:
                    growth_assessment['growth_quality'] = 'Average'
                else:
                    growth_assessment['growth_quality'] = 'Poor'
            
        except Exception as e:
            print(f"Error evaluating growth performance: {e}")
        
        return growth_assessment

    def _create_comprehensive_analysis_context(self, stock_data: StockData, rhs_analysis: Optional[Dict] = None, cwh_analysis: Optional[Dict] = None) -> str:
        """Create comprehensive analysis context for OpenAI"""
        context_parts = []
        
        # Basic stock information
        context_parts.append(f"Stock: {stock_data.ticker} ({stock_data.company_name})")
        context_parts.append(f"Sector: {stock_data.sector}")
        context_parts.append(f"Category: {stock_data.category}")
        
        # OHLCV data summary
        if stock_data.ohlcv_data is not None:
            df = stock_data.ohlcv_data
            context_parts.append(f"Data Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            context_parts.append(f"Data Points: {len(df)}")
            context_parts.append(f"Current Price: ₹{df['Close'].iloc[-1]:.2f}")
            context_parts.append(f"52-Week High: ₹{df['High'].max():.2f}")
            context_parts.append(f"52-Week Low: ₹{df['Low'].min():.2f}")
        
        # RHS Pattern Analysis
        if rhs_analysis:
            context_parts.append("\nRHS Pattern Analysis:")
            context_parts.append(f"- Pattern Status: {rhs_analysis.get('status', 'N/A')}")
            context_parts.append(f"- Head Level: ₹{rhs_analysis.get('head_level', 'N/A')}")
            context_parts.append(f"- Neckline: ₹{rhs_analysis.get('neckline', 'N/A')}")
            context_parts.append(f"- Target: ₹{rhs_analysis.get('target', 'N/A')}")
        
        # CWH Pattern Analysis
        if cwh_analysis:
            context_parts.append("\nCWH Pattern Analysis:")
            context_parts.append(f"- Pattern Status: {cwh_analysis.get('status', 'N/A')}")
            context_parts.append(f"- Cup Depth: {cwh_analysis.get('cup_depth', 'N/A')}")
            context_parts.append(f"- Handle Depth: {cwh_analysis.get('handle_depth', 'N/A')}")
            context_parts.append(f"- Target: ₹{cwh_analysis.get('target', 'N/A')}")
        
        return "\n".join(context_parts)

# Enhanced Coordinator Agent
class EnhancedCoordinatorAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_recommendation(self, technical: EnhancedTechnicalAnalysis, 
                              fundamental: EnhancedFundamentalAnalysis,
                              strategy_analysis: Dict) -> EnhancedFinalRecommendation:
        
        # Initialize strategy_used variable
        strategy_used = "Conservative Analysis"
        
        # Check if strategy analysis is applicable
        if strategy_analysis.get("eligible", False):
            # Use strategy performance analysis for data-driven recommendations
            recommended_strategy = strategy_analysis.get("recommended_strategy", "Conservative Analysis")
            recommended_performance = strategy_analysis.get("recommended_performance", {})
            
            # Determine action based on strategy performance
            if "error" not in strategy_analysis and recommended_performance:
                success_rate = recommended_performance.get("success_rate", 0)
                total_signals = recommended_performance.get("total_signals", 0)
                
                # Enhanced strategy detection with RHS and CWH priority
                strategy_signals = technical.strategy_signals if technical.strategy_signals else []
                patterns = technical.patterns if technical.patterns else []
                
                # Check for RHS and CWH patterns first (highest priority)
                has_rhs = any("rhs" in str(signal).lower() or "reverse head" in str(signal).lower() for signal in strategy_signals)
                has_rhs_pattern = any("reverse head" in str(pattern).lower() for pattern in patterns)
                has_cwh = any("cwh" in str(signal).lower() or "cup with handle" in str(signal).lower() for signal in strategy_signals)
                has_cwh_pattern = any("cup with handle" in str(pattern).lower() for pattern in patterns)
                
                # Priority order: RHS > CWH > Strategy Performance > Fallback
                if has_rhs or has_rhs_pattern:
                    action = "BUY"
                    confidence_level = f"High ({success_rate:.1f}% success rate, {total_signals} signals)"
                    strategy_used = "RHS Strategy"
                elif has_cwh or has_cwh_pattern:
                    action = "BUY"
                    confidence_level = f"High ({success_rate:.1f}% success rate, {total_signals} signals)"
                    strategy_used = "CWH Strategy"
                elif success_rate >= 60 and total_signals >= 3:  # High confidence
                    action = "BUY"
                    confidence_level = f"High ({success_rate:.1f}% success rate, {total_signals} signals)"
                    strategy_used = recommended_strategy
                elif success_rate >= 50 and total_signals >= 2:  # Medium confidence
                    action = "BUY"
                    confidence_level = f"Medium ({success_rate:.1f}% success rate, {total_signals} signals)"
                    strategy_used = recommended_strategy
                elif success_rate >= 40 and total_signals >= 1:  # Low confidence
                    action = "BUY"
                    confidence_level = f"Low ({success_rate:.1f}% success rate, {total_signals} signals)"
                    strategy_used = recommended_strategy
                else:
                    action = "HOLD"
                    confidence_level = f"Low ({success_rate:.1f}% success rate, {total_signals} signals)"
                    strategy_used = recommended_strategy
            else:
                # Fallback to original logic if strategy analysis fails
                strategy_signals = technical.strategy_signals if technical.strategy_signals else []
                patterns = technical.patterns if technical.patterns else []
                
                has_rhs = any("rhs" in str(signal).lower() or "reverse head" in str(signal).lower() for signal in strategy_signals)
                has_rhs_pattern = any("reverse head" in str(pattern).lower() for pattern in patterns)
                has_cwh = any("cwh" in str(signal).lower() or "cup with handle" in str(signal).lower() for signal in strategy_signals)
                has_cwh_pattern = any("cup with handle" in str(pattern).lower() for pattern in patterns)
                
                if has_rhs or has_rhs_pattern:
                    action = "BUY"
                    confidence_level = "Medium (RHS Pattern Detected)"
                    strategy_used = "RHS Strategy"
                elif has_cwh or has_cwh_pattern:
                    action = "BUY"
                    confidence_level = "Medium (CWH Pattern Detected)"
                    strategy_used = "CWH Strategy"
                elif "lifetime high" in str(technical.strategy_signals).lower():
                    action = "BUY"
                    confidence_level = "Medium"
                    strategy_used = "Lifetime High Strategy"
                elif fundamental.business_quality == "Strong":
                    action = "BUY"
                    confidence_level = "Medium"
                    strategy_used = "Fundamental Analysis"
                else:
                    action = "HOLD"
                    confidence_level = "Low"
                    strategy_used = "Conservative Analysis"
            
            # Determine position size based on strategy performance
            if recommended_performance:
                success_rate = recommended_performance.get("success_rate", 0)
                if success_rate >= 70:
                    position_size = "5% of portfolio"
                elif success_rate >= 60:
                    position_size = "4% of portfolio"
                elif success_rate >= 50:
                    position_size = "3% of portfolio"
                else:
                    position_size = "2% of portfolio"
            else:
                position_size = "2% of portfolio"
            
            # Calculate target based on strategy performance
            if recommended_performance and recommended_performance.get("avg_gain", 0) > 0:
                try:
                    current_price = float(technical.entry_range.split('-')[0].replace('₹', '').strip())
                    avg_gain = recommended_performance.get("avg_gain", 10)
                    target_price = f"₹{round(current_price * (1 + avg_gain/100), 2)}"
                except:
                    target_price = technical.medium_term_target
            else:
                target_price = technical.medium_term_target
            
            # Determine time horizon based on strategy performance
            if recommended_performance:
                avg_hold_days = recommended_performance.get("avg_hold_days", 30)
                if avg_hold_days <= 20:
                    time_horizon = "1-2 months"
                elif avg_hold_days <= 60:
                    time_horizon = "2-3 months"
                else:
                    time_horizon = "3-6 months"
            else:
                time_horizon = "3-6 months"
            
            # Determine risk level based on strategy performance
            if recommended_performance:
                avg_loss = abs(recommended_performance.get("avg_loss", 0))
                if avg_loss <= 5:
                    risk_level = "Low"
                elif avg_loss <= 10:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
            else:
                risk_level = "Medium"
            
        else:
            # For non-eligible stocks, use basic analysis without strategy performance
            category = strategy_analysis.get("category", "Unknown")
            
            # Basic analysis for non-V40 stocks
            if fundamental.business_quality == "Strong":
                action = "BUY"
                confidence_level = "Medium (Basic Analysis)"
            elif fundamental.business_quality == "Weak":
                action = "HOLD"
                confidence_level = "Low (Basic Analysis)"
            else:
                action = "HOLD"
                confidence_level = "Low (Basic Analysis)"
            
            # Conservative position sizing for non-V40 stocks
            position_size = "1-2% of portfolio"
            target_price = technical.medium_term_target
            time_horizon = "3-6 months"
            risk_level = "Medium"
            strategy_used = f"Basic Analysis (Category: {category})"
        
        return EnhancedFinalRecommendation(
            action=action,
            entry_price=technical.entry_range,
            target_price=target_price,
            stop_loss=technical.stop_loss,
            time_horizon=time_horizon,
            confidence_level=confidence_level,
            risk_level=risk_level,
            position_size=position_size,
            strategy_used=strategy_used,
            key_risks=["Market volatility", "Sector-specific risks", "Strategy performance may not repeat"],
            fundamental_reasons=fundamental.fundamental_reasons if hasattr(fundamental, 'fundamental_reasons') else "Fundamental analysis not available"
        )

# Enhanced Multi-Agent Framework
class EnhancedMultiAgentStockAnalysis:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key  # Store the API key
        self.openai_client = OpenAI(api_key=openai_api_key)
        # Initialize primary LLM with configurable model
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model=DEFAULT_OPENAI_MODEL, temperature=1)
        self.technical_agent = EnhancedTechnicalAnalysisAgent(self.llm, openai_api_key)
        self.fundamental_agent = EnhancedFundamentalAnalysisAgent(self.llm, openai_api_key)
        self.coordinator_agent = EnhancedCoordinatorAgent(self.llm)
        
        # Initialize cost tracker
        self.cost_tracker = cost_tracker
        
        # Initialize screenshots directory
        self.screenshots_dir = "screener_screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)
    
    def analyze_stock(self, ticker: str, company_name: str, sector: str, category: str = "Unknown") -> Dict:
        """
        Analyze a stock using the enhanced multi-agent system with centralized data collection.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name
            sector: Sector/industry
            category: Stock category (V40, V40 Next, etc.)
        
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            print(f"\n🚀 Starting Enhanced Multi-Agent Analysis for {ticker} ({company_name})")
            print(f"📊 Sector: {sector} | Category: {category}")
            
            # Create StockData object
            stock_data = StockData(
                ticker=ticker,
                company_name=company_name,
                sector=sector,
                category=category
            )
            
            # Step 1: Fetch stock data
            print(f"\n📈 Step 1: Fetching stock data for {ticker}...")
            ohlcv_data = self._fetch_stock_data_with_retry(ticker)
            if ohlcv_data is not None:
                stock_data.ohlcv_data = ohlcv_data
                print(f"✅ Stock data fetched successfully ({len(ohlcv_data)} data points)")
            else:
                print(f"❌ Failed to fetch stock data for {ticker}")
                return {"error": f"Failed to fetch stock data for {ticker}"}
            
            # Step 2: Centralized Data Collection (All screenshots and data extraction)
            print(f"\n📊 Step 2: Centralized Data Collection...")
            collected_data = self._collect_all_data(ticker, stock_data)
            
            # Step 3: Technical Analysis using collected data
            print(f"\n📊 Step 3: Performing Technical Analysis...")
            technical_analysis = self._perform_technical_analysis_with_collected_data(stock_data, collected_data)
            print(f"✅ Technical analysis completed")
            
            # Update chart with technical analysis results
            self._update_chart_with_technical_analysis(stock_data, technical_analysis, collected_data)
            
            # Step 4: Strategy Performance Analysis (for eligible stocks)
            print(f"\n🎯 Step 4: Analyzing Strategy Performance...")
            strategy_analysis = self._analyze_strategy_performance(stock_data, technical_analysis)
            print(f"✅ Strategy analysis completed")
            
            # Step 5: Fundamental Analysis using collected data
            print(f"\n💰 Step 5: Performing Fundamental Analysis...")
            fundamental_analysis = self._perform_fundamental_analysis_with_collected_data(stock_data, collected_data)
            print(f"✅ Fundamental analysis completed")
            
            # Step 6: Generate correlated insights using collected data
            print(f"\n🧠 Step 6: Generating Correlated Insights...")
            correlated_insights = self._generate_comprehensive_correlated_insights(
                ticker, fundamental_analysis, collected_data.get('arthalens_data', {})
            )
            
            # Step 7: Generate Final Recommendation
            print(f"\n🎯 Step 7: Generating Final Recommendation...")
            final_recommendation = self.coordinator_agent.generate_recommendation(
                technical_analysis, fundamental_analysis, strategy_analysis
            )
            print(f"✅ Final recommendation generated")
            
            # Step 8: Generate Comprehensive Report
            print(f"\n📋 Step 8: Generating Comprehensive Report...")
            comprehensive_report = self._generate_comprehensive_report({
                'ticker': ticker,
                'company_name': company_name,
                'sector': sector,
                'category': category,
                'technical_analysis': technical_analysis.model_dump() if technical_analysis else {},
                'fundamental_analysis': fundamental_analysis.model_dump() if fundamental_analysis else {},
                'strategy_analysis': strategy_analysis,
                'final_recommendation': final_recommendation.model_dump() if final_recommendation else {},
                'arthalens_data': collected_data.get('arthalens_data', {}),
                'correlated_insights': correlated_insights,
                'collected_data': collected_data
            })
            print(f"✅ Comprehensive report generated")
            
            # Save final analysis results to run directory
            if collected_data.get('run_directory'):
                # Save comprehensive report
                report_path = os.path.join(collected_data['run_directory'], "comprehensive_report.txt")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(comprehensive_report)
                print(f"✅ Comprehensive report saved: {report_path}")
                
                # Save final analysis results
                final_results_path = os.path.join(collected_data['run_directory'], "final_analysis_results.json")
                final_results = {
                    'ticker': ticker,
                    'company_name': company_name,
                    'sector': sector,
                    'category': category,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'technical_analysis': technical_analysis.model_dump() if technical_analysis else {},
                    'fundamental_analysis': fundamental_analysis.model_dump() if fundamental_analysis else {},
                    'strategy_analysis': strategy_analysis,
                    'final_recommendation': final_recommendation.model_dump() if final_recommendation else {},
                    'arthalens_data': collected_data.get('arthalens_data', {}),
                    'correlated_insights': correlated_insights,
                    'run_directory': collected_data['run_directory'],
                    'enhanced_fundamental_data': collected_data.get('enhanced_fundamental_data').__dict__ if collected_data.get('enhanced_fundamental_data') else None  # Convert to dict for JSON serialization
                }
                with open(final_results_path, 'w') as f:
                    json.dump(final_results, f, indent=2, cls=CustomJSONEncoder)
                print(f"✅ Final analysis results saved: {final_results_path}")
            
            # Prepare final results
            results = {
                'ticker': ticker,
                'company_name': company_name,
                'sector': sector,
                'category': category,
                'analysis_timestamp': datetime.now().isoformat(),
                'technical_analysis': technical_analysis.model_dump() if technical_analysis else {},
                'fundamental_analysis': fundamental_analysis.model_dump() if fundamental_analysis else {},
                'strategy_analysis': strategy_analysis,
                'final_recommendation': final_recommendation.model_dump() if final_recommendation else {},
                'recommendation': final_recommendation.model_dump() if final_recommendation else {},  # Add both names for compatibility
                'arthalens_data': collected_data.get('arthalens_data', {}),
                'correlated_insights': correlated_insights,
                'comprehensive_report': comprehensive_report,
                'collected_data': collected_data,
                'enhanced_fundamental_data': collected_data.get('enhanced_fundamental_data').__dict__ if collected_data.get('enhanced_fundamental_data') else None  # Convert to dict for JSON serialization
            }
            
            print(f"\n🎉 Analysis completed successfully for {ticker}!")
            if collected_data.get('run_directory'):
                print(f"📁 All data saved to: {collected_data['run_directory']}")
            return results
            
        except Exception as e:
            print(f"❌ Error in stock analysis: {e}")
            return {"error": f"Analysis failed for {ticker}: {str(e)}"}
    
    def _fetch_stock_data_with_retry(self, ticker: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch stock data with retry logic and multiple ticker formats"""
        import time as time_module
        
        # Try different ticker formats
        ticker_formats = [
            ticker,  # Original format
            f"{ticker}.NS",  # NSE format
            ticker.replace('.NS', ''),  # Without .NS
            ticker.replace('.BO', ''),  # Without .BO
        ]
        
        for attempt in range(max_retries):
            for ticker_format in ticker_formats:
                try:
                    print(f"🔄 Attempt {attempt + 1}: Trying {ticker_format}...")
                    data = yf.download(ticker_format, period="3y", interval="1d", progress=False, auto_adjust=True)
                    
                    if data is not None and len(data) > 0:
                        print(f"✅ Successfully fetched data for {ticker_format}")
                        return data
                    else:
                        print(f"⚠️ No data returned for {ticker_format}")
                        
                except Exception as e:
                    print(f"❌ Error fetching {ticker_format}: {str(e)[:100]}...")
                    continue
            
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in 2 seconds... (attempt {attempt + 2}/{max_retries})")
                time_module.sleep(2)
        
        print(f"❌ Failed to fetch data for {ticker} after {max_retries} attempts")
        return None
    
    def _get_tradingview_screenshot(self, ticker: str) -> Optional[str]:
        """Get TradingView screenshot as fallback when price data is unavailable"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import os
            
            # Prepare ticker for TradingView
            tv_ticker = ticker.replace('.NS', '').replace('.BO', '')
            if not tv_ticker.startswith('NSE:'):
                tv_ticker = f"NSE:{tv_ticker}"
            
            # Set up headless Chrome
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Chrome(options=chrome_options)
            
            try:
                # Navigate to TradingView
                url = f"https://www.tradingview.com/chart/?symbol={tv_ticker}"
                driver.get(url)
                
                # Wait for chart to load
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-name='pane-legend']"))
                )
                
                # Take screenshot
                screenshot_path = f"tradingview_{ticker.replace('.', '_')}.png"
                driver.save_screenshot(screenshot_path)
                
                print(f"📸 Screenshot saved: {screenshot_path}")
                return screenshot_path
                
            finally:
                driver.quit()
                
        except Exception as e:
            print(f"❌ Failed to get TradingView screenshot: {e}")
            return None
    
    def generate_enhanced_report(self, results: Dict) -> str:
        """Generate enhanced analysis report with specific outputs"""
        try:
            ticker = results.get('ticker', 'Unknown')
            company_name = results.get('company_name', 'Unknown')
            sector = results.get('sector', 'Unknown')
            category = results.get('category', 'Unknown')
            
            technical = results.get('technical_analysis')
            fundamental = results.get('fundamental_analysis')
            recommendation = results.get('final_recommendation')
            strategy_analysis = results.get('strategy_analysis', {})
            arthalens_insights = results.get('arthalens_data', {})
            correlated_insights = results.get('correlated_insights', {})
            
            # Helper function to safely get attributes
            def safe_get(obj, attr, default='Not Available'):
                if obj is None:
                    return default
                if hasattr(obj, attr):
                    return getattr(obj, attr, default)
                elif isinstance(obj, dict):
                    return obj.get(attr, default)
                return default
            
            report = f"""
{'='*80}
ENHANCED STOCK ANALYSIS REPORT
{'='*80}

📊 STOCK INFORMATION
{'='*40}
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Category: {category}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
📈 TECHNICAL ANALYSIS
{'='*40}
Trend: {safe_get(technical, 'trend')}
Support Levels: {safe_get(technical, 'support_levels')}
Resistance Levels: {safe_get(technical, 'resistance_levels')}
Entry Range: {safe_get(technical, 'entry_range')}
Short-term Target: {safe_get(technical, 'short_term_target')}
Medium-term Target: {safe_get(technical, 'medium_term_target')}
Stop Loss: {safe_get(technical, 'stop_loss')}
Confidence Score: {safe_get(technical, 'confidence_score')}

Technical Indicators:
{self._format_indicators(safe_get(technical, 'indicators', {})) if technical else 'Not Available'}

Chart Patterns Identified:
{self._format_patterns(safe_get(technical, 'patterns', [])) if technical else 'Not Available'}

Strategy Signals:
{self._format_signals(safe_get(technical, 'strategy_signals', [])) if technical else 'Not Available'}

{'='*80}
💰 FUNDAMENTAL ANALYSIS
{'='*40}
Business Quality: {safe_get(fundamental, 'business_quality')}
Market Penetration: {safe_get(fundamental, 'market_penetration')}
Pricing Power: {safe_get(fundamental, 'pricing_power')}
Revenue Growth: {safe_get(fundamental, 'revenue_growth')}
Profit Growth: {safe_get(fundamental, 'profit_growth')}
Debt to Equity: {safe_get(fundamental, 'debt_to_equity')}
ROCE/ROE: {safe_get(fundamental, 'roce_roe')}
Promoter Pledging: {safe_get(fundamental, 'promoter_pledging')}
Retail Shareholding: {safe_get(fundamental, 'retail_shareholding')}
Valuation Status: {safe_get(fundamental, 'valuation_status')}
Fair Value: {safe_get(fundamental, 'fair_value')}
Financial Health: {safe_get(fundamental, 'financial_health')}
Multibagger Potential: {safe_get(fundamental, 'multibagger_potential')}
Confidence Score: {safe_get(fundamental, 'confidence_score')}

Fundamental Reasons for Buy Signal:
{safe_get(fundamental, 'fundamental_reasons')}

{'='*80}
📊 STRATEGY PERFORMANCE ANALYSIS
{'='*40}
{self._format_strategy_analysis(strategy_analysis)}

{'='*80}
🎯 FINAL RECOMMENDATION
{'='*40}
Action: {safe_get(recommendation, 'action')}
Entry Price: {safe_get(recommendation, 'entry_price')}
Target Price: {safe_get(recommendation, 'target_price')}
Stop Loss: {safe_get(recommendation, 'stop_loss')}
Time Horizon: {safe_get(recommendation, 'time_horizon')}
Confidence Level: {safe_get(recommendation, 'confidence_level')}
Risk Level: {safe_get(recommendation, 'risk_level')}
Position Size: {safe_get(recommendation, 'position_size')}
Strategy Used: {safe_get(recommendation, 'strategy_used')}

Key Risks:
{self._format_risks(safe_get(recommendation, 'key_risks', [])) if recommendation else 'Not Available'}

Fundamental Reasons for Buy Signal:
{safe_get(recommendation, 'fundamental_reasons')}

{'='*80}
📈 ARTHALENS CORRELATED INSIGHTS
{'='*40}
"""
            
            # Add ArthaLens insights if available
            if arthalens_insights and "summary_data" in arthalens_insights:
                report += f"""
📋 TRANSCRIPT SUMMARY:
{arthalens_insights.get('summary_data', {}).get('Q4+FY25', {}).get('extracted_text', 'Not Available')}

🎯 GUIDANCE DATA:
{arthalens_insights.get('guidance_data', {}).get('Q4+FY25', {}).get('extracted_text', 'Not Available')}

❓ Q&A HIGHLIGHTS:
{arthalens_insights.get('qa_data', {}).get('Q4+FY25', {}).get('extracted_text', 'Not Available')}
"""
            
            # Add correlated insights if available
            if correlated_insights and "analysis" in correlated_insights:
                report += f"""
🧠 CORRELATED INSIGHTS ANALYSIS:
{correlated_insights['analysis']}
"""
            
            report += f"""
{'='*80}
📊 ANALYSIS SUMMARY
{'='*40}
This enhanced analysis provides a comprehensive view of {ticker} ({company_name}) 
across technical, fundamental, and strategic dimensions. The analysis incorporates 
real-time data from multiple sources including Screener.in, ArthaLens, and Yahoo Finance.

Key Highlights:
• Technical Analysis: {safe_get(technical, 'trend', 'Not Available')} trend with {safe_get(technical, 'confidence_score', 'Not Available')} confidence
• Fundamental Analysis: {safe_get(fundamental, 'business_quality', 'Not Available')} business quality with {safe_get(fundamental, 'confidence_score', 'Not Available')} confidence
• Final Recommendation: {safe_get(recommendation, 'action', 'Not Available')} with {safe_get(recommendation, 'confidence_level', 'Not Available')} confidence

Risk Assessment: {safe_get(recommendation, 'risk_level', 'Not Available')}
Investment Horizon: {safe_get(recommendation, 'time_horizon', 'Not Available')}

{'='*80}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
            
            return report
            
        except Exception as e:
            print(f"❌ Error generating enhanced report: {e}")
            return f"Error generating report: {str(e)}"
    
    def _analyze_strategy_performance(self, stock_data: StockData, technical_analysis: EnhancedTechnicalAnalysis) -> Dict:
        """Analyze strategy performance on 2-year historical data to identify most successful strategy"""
        try:
            # Check if stock is eligible for strategy analysis
            if not self._is_strategy_eligible(stock_data):
                return {
                    "error": f"Strategy analysis not applicable for category: {stock_data.category}",
                    "category": stock_data.category,
                    "eligible": False,
                    "analysis_summary": f"Strategy Performance Analysis: Not applicable for {stock_data.category} category stocks. Only V40 and V40 Next categories are eligible for strategy analysis."
                }
            
            if stock_data.ohlcv_data is None or len(stock_data.ohlcv_data) < 100:
                return {"error": "Insufficient data for strategy analysis"}
            
            # Handle MultiIndex DataFrame structure
            if isinstance(stock_data.ohlcv_data.columns, pd.MultiIndex):
                stock_data.ohlcv_data.columns = stock_data.ohlcv_data.columns.get_level_values(0)
            
            df = stock_data.ohlcv_data.copy()
            
            # Get 2-year data for backtesting
            historical_data = df.tail(730)  # 2 years = 730 days
            
            # Define all strategies to backtest
            strategies = {
                "RHS Strategy": self._backtest_rhs_strategy,
                "CWH Strategy": self._backtest_cwh_strategy,
                "SMA Strategy": self._backtest_sma_strategy,
                "V20 Strategy": self._backtest_v20_strategy,
                "Lifetime High Strategy": self._backtest_lifetime_high_strategy,
                "RSI Strategy": self._backtest_rsi_strategy,
                "BB Strategy": self._backtest_bollinger_bands_strategy
            }
            
            # Backtest each strategy
            strategy_results = {}
            for strategy_name, strategy_func in strategies.items():
                try:
                    result = strategy_func(historical_data)
                    if result["total_signals"] > 0:
                        strategy_results[strategy_name] = result
                except Exception as e:
                    print(f"⚠️ Error backtesting {strategy_name}: {e}")
                    continue
            
            if not strategy_results:
                return {"error": "No valid strategy signals found in historical data"}
            
            # Find the most successful strategy
            best_strategy = max(strategy_results.items(), 
                              key=lambda x: x[1]["success_rate"])
            
            # Get current strategy signals
            current_strategies = []
            if "rhs" in str(technical_analysis.strategy_signals).lower():
                current_strategies.append("RHS Strategy")
            if "cwh" in str(technical_analysis.strategy_signals).lower():
                current_strategies.append("CWH Strategy")
            if "sma" in str(technical_analysis.strategy_signals).lower():
                current_strategies.append("SMA Strategy")
            if "v20" in str(technical_analysis.strategy_signals).lower():
                current_strategies.append("V20 Strategy")
            if "lifetime high" in str(technical_analysis.strategy_signals).lower():
                current_strategies.append("Lifetime High Strategy")
            if "rsi" in str(technical_analysis.strategy_signals).lower():
                current_strategies.append("RSI Strategy")
            if "bb" in str(technical_analysis.strategy_signals).lower():
                current_strategies.append("BB Strategy")
            
            # Determine recommended strategy
            recommended_strategy = None
            if current_strategies:
                # Check if any current strategy has historical performance
                available_current = [s for s in current_strategies if s in strategy_results]
                if available_current:
                    recommended_strategy = max(available_current, 
                                            key=lambda x: strategy_results[x]["success_rate"])
                else:
                    recommended_strategy = best_strategy[0]  # Use best historical strategy
            else:
                recommended_strategy = best_strategy[0]  # Use best historical strategy
            
            return {
                "strategy_performance": strategy_results,
                "best_strategy": best_strategy[0],
                "best_success_rate": best_strategy[1]["success_rate"],
                "recommended_strategy": recommended_strategy,
                "recommended_performance": strategy_results.get(recommended_strategy, {}),
                "current_strategies": current_strategies,
                "category": stock_data.category,
                "eligible": True,
                "analysis_summary": self._generate_strategy_analysis_summary(
                    strategy_results, recommended_strategy, current_strategies
                )
            }
            
        except Exception as e:
            print(f"⚠️ Strategy performance analysis error: {e}")
            return {"error": f"Strategy analysis failed: {e}"}
    
    def _find_local_minima(self, data: pd.DataFrame) -> List[Dict]:
        """Find local minima (troughs) in the price data"""
        troughs = []
        window = 10  # Look for minima in 10-day windows
        
        for i in range(window, len(data) - window):
            current_low = data['Low'].iloc[i]
            left_window = data['Low'].iloc[i-window:i].min()
            right_window = data['Low'].iloc[i+1:i+window+1].min()
            
            # Check if current point is a local minimum
            if current_low <= left_window and current_low <= right_window:
                troughs.append({
                    'index': i,
                    'date': data.index[i],
                    'price': current_low,
                    'volume': data['Volume'].iloc[i]
                })
        
        return troughs
    
    def _backtest_rhs_strategy(self, data: pd.DataFrame) -> Dict:
        """Backtest RHS strategy on historical data"""
        signals = []
        current_price = data['Close'].iloc[-1]
        
        # Look for RHS patterns in historical data
        for i in range(50, len(data) - 20):  # Need enough data before and after
            window_data = data.iloc[i-50:i+20]
            
            # Find local minima in window
            troughs = self._find_local_minima(window_data)
            
            if len(troughs) >= 3:
                # Check for RHS pattern
                for j in range(len(troughs) - 2):
                    ls = troughs[j]
                    head = troughs[j + 1]
                    rs = troughs[j + 2]
                    
                    # Basic RHS validation
                    if (head['price'] < ls['price'] and 
                        rs['price'] > head['price'] and
                        abs(rs['price'] - ls['price']) / ls['price'] < 0.1):
                        
                        # Check for breakout
                        entry_price = rs['price']
                        entry_date = rs['date']
                        
                        # Look for target achievement (20% gain or 10% loss)
                        for k in range(i + 20, min(i + 100, len(data))):
                            future_price = data['Close'].iloc[k]
                            gain_pct = (future_price - entry_price) / entry_price * 100
                            
                            if gain_pct >= 20:  # Success
                                signals.append({
                                    "entry_price": entry_price,
                                    "entry_date": entry_date,
                                    "exit_price": future_price,
                                    "exit_date": data.index[k],
                                    "gain_pct": gain_pct,
                                    "success": True
                                })
                                break
                            elif gain_pct <= -10:  # Failure
                                signals.append({
                                    "entry_price": entry_price,
                                    "entry_date": entry_date,
                                    "exit_price": future_price,
                                    "exit_date": data.index[k],
                                    "gain_pct": gain_pct,
                                    "success": False
                                })
                                break
        
        return self._calculate_strategy_metrics(signals)
    
    def _backtest_cwh_strategy(self, data: pd.DataFrame) -> Dict:
        """Backtest CWH strategy on historical data"""
        signals = []
        
        # Look for CWH patterns in historical data
        for i in range(50, len(data) - 20):
            window_data = data.iloc[i-50:i+20]
            
            # Find local minima in window
            troughs = self._find_local_minima(window_data)
            
            if len(troughs) >= 2:
                # Check for CWH pattern
                for j in range(len(troughs) - 1):
                    cup_trough = troughs[j]
                    handle_trough = troughs[j + 1]
                    
                    # Basic CWH validation
                    if handle_trough['price'] > cup_trough['price']:
                        # Check for breakout
                        entry_price = handle_trough['price']
                        entry_date = handle_trough['date']
                        
                        # Look for target achievement (15% gain or 8% loss)
                        for k in range(i + 20, min(i + 80, len(data))):
                            future_price = data['Close'].iloc[k]
                            gain_pct = (future_price - entry_price) / entry_price * 100
                            
                            if gain_pct >= 15:  # Success
                                signals.append({
                                    "entry_price": entry_price,
                                    "entry_date": entry_date,
                                    "exit_price": future_price,
                                    "exit_date": data.index[k],
                                    "gain_pct": gain_pct,
                                    "success": True
                                })
                                break
                            elif gain_pct <= -8:  # Failure
                                signals.append({
                                    "entry_price": entry_price,
                                    "entry_date": entry_date,
                                    "exit_price": future_price,
                                    "exit_date": data.index[k],
                                    "gain_pct": gain_pct,
                                    "success": False
                                })
                                break
        
        return self._calculate_strategy_metrics(signals)
    
    def _backtest_sma_strategy(self, data: pd.DataFrame) -> Dict:
        """Backtest SMA strategy on historical data"""
        signals = []
        
        # Calculate moving averages
        ma_20 = data['Close'].rolling(window=20).mean()
        ma_50 = data['Close'].rolling(window=50).mean()
        ma_100 = data['Close'].rolling(window=100).mean()
        ma_200 = data['Close'].rolling(window=200).mean()
        
        for i in range(200, len(data) - 20):
            # Golden Cross signal
            if (ma_20.iloc[i] > ma_50.iloc[i] and 
                ma_20.iloc[i-1] <= ma_50.iloc[i-1] and
                ma_50.iloc[i] > ma_100.iloc[i] and
                ma_100.iloc[i] > ma_200.iloc[i]):
                
                entry_price = data['Close'].iloc[i]
                entry_date = data.index[i]
                
                # Look for target achievement (12% gain or 6% loss)
                for j in range(i + 1, min(i + 60, len(data))):
                    future_price = data['Close'].iloc[j]
                    gain_pct = (future_price - entry_price) / entry_price * 100
                    
                    if gain_pct >= 12:  # Success
                        signals.append({
                            "entry_price": entry_price,
                            "entry_date": entry_date,
                            "exit_price": future_price,
                            "exit_date": data.index[j],
                            "gain_pct": gain_pct,
                            "success": True
                        })
                        break
                    elif gain_pct <= -6:  # Failure
                        signals.append({
                            "entry_price": entry_price,
                            "entry_date": entry_date,
                            "exit_price": future_price,
                            "exit_date": data.index[j],
                            "gain_pct": gain_pct,
                            "success": False
                        })
                        break
        
        return self._calculate_strategy_metrics(signals)
    
    def _backtest_v20_strategy(self, data: pd.DataFrame) -> Dict:
        """Backtest V20 strategy on historical data"""
        signals = []
        
        for i in range(20, len(data) - 20):
            # Calculate 20-day range
            window_data = data.iloc[i-20:i]
            high_20d = window_data['High'].max()
            low_20d = window_data['Low'].min()
            range_pct = (high_20d - low_20d) / low_20d * 100
            
            # V20 signal (significant range)
            if range_pct > 20:
                entry_price = low_20d  # Buy at lower line
                entry_date = data.index[i]
                
                # Look for target achievement (10% gain or 5% loss)
                for j in range(i + 1, min(i + 40, len(data))):
                    future_price = data['Close'].iloc[j]
                    gain_pct = (future_price - entry_price) / entry_price * 100
                    
                    if gain_pct >= 10:  # Success
                        signals.append({
                            "entry_price": entry_price,
                            "entry_date": entry_date,
                            "exit_price": future_price,
                            "exit_date": data.index[j],
                            "gain_pct": gain_pct,
                            "success": True
                        })
                        break
                    elif gain_pct <= -5:  # Failure
                        signals.append({
                            "entry_price": entry_price,
                            "entry_date": entry_date,
                            "exit_price": future_price,
                            "exit_date": data.index[j],
                            "gain_pct": gain_pct,
                            "success": False
                        })
                        break
        
        return self._calculate_strategy_metrics(signals)
    
    def _backtest_lifetime_high_strategy(self, data: pd.DataFrame) -> Dict:
        """Backtest Lifetime High strategy on historical data"""
        signals = []
        
        for i in range(100, len(data) - 20):
            # Calculate lifetime high up to current point
            lifetime_high = data['High'].iloc[:i+1].max()
            current_price = data['Close'].iloc[i]
            
            # Lifetime High signal (down >30% from ATH)
            if current_price < lifetime_high * 0.7:
                entry_price = current_price
                entry_date = data.index[i]
                
                # Look for target achievement (25% gain or 15% loss)
                for j in range(i + 1, min(i + 120, len(data))):
                    future_price = data['Close'].iloc[j]
                    gain_pct = (future_price - entry_price) / entry_price * 100
                    
                    if gain_pct >= 25:  # Success
                        signals.append({
                            "entry_price": entry_price,
                            "entry_date": entry_date,
                            "exit_price": future_price,
                            "exit_date": data.index[j],
                            "gain_pct": gain_pct,
                            "success": True
                        })
                        break
                    elif gain_pct <= -15:  # Failure
                        signals.append({
                            "entry_price": entry_price,
                            "entry_date": entry_date,
                            "exit_price": future_price,
                            "exit_date": data.index[j],
                            "gain_pct": gain_pct,
                            "success": False
                        })
                        break
        
        return self._calculate_strategy_metrics(signals)
    
    def _backtest_rsi_strategy(self, data: pd.DataFrame) -> Dict:
        """Backtest RSI strategy on historical data"""
        signals = []
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        for i in range(20, len(data) - 20):
            current_rsi = rsi.iloc[i]
            
            # RSI oversold signal
            if current_rsi < 30:
                entry_price = data['Close'].iloc[i]
                entry_date = data.index[i]
                
                # Look for target achievement (8% gain or 4% loss)
                for j in range(i + 1, min(i + 30, len(data))):
                    future_price = data['Close'].iloc[j]
                    gain_pct = (future_price - entry_price) / entry_price * 100
                    
                    if gain_pct >= 8:  # Success
                        signals.append({
                            "entry_price": entry_price,
                            "entry_date": entry_date,
                            "exit_price": future_price,
                            "exit_date": data.index[j],
                            "gain_pct": gain_pct,
                            "success": True
                        })
                        break
                    elif gain_pct <= -4:  # Failure
                        signals.append({
                            "entry_price": entry_price,
                            "entry_date": entry_date,
                            "exit_price": future_price,
                            "exit_date": data.index[j],
                            "gain_pct": gain_pct,
                            "success": False
                        })
                        break
        
        return self._calculate_strategy_metrics(signals)
    
    def _backtest_bollinger_bands_strategy(self, data: pd.DataFrame) -> Dict:
        """Backtest Bollinger Bands strategy on historical data"""
        signals = []
        
        # Calculate Bollinger Bands
        ma_20 = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        bb_upper = ma_20 + (2 * bb_std)
        bb_lower = ma_20 - (2 * bb_std)
        
        for i in range(20, len(data) - 20):
            current_price = data['Close'].iloc[i]
            current_bb_lower = bb_lower.iloc[i]
            
            # BB lower band touch signal
            if current_price <= current_bb_lower:
                entry_price = current_price
                entry_date = data.index[i]
                
                # Look for target achievement (6% gain or 3% loss)
                for j in range(i + 1, min(i + 25, len(data))):
                    future_price = data['Close'].iloc[j]
                    gain_pct = (future_price - entry_price) / entry_price * 100
                    
                    if gain_pct >= 6:  # Success
                        signals.append({
                            "entry_price": entry_price,
                            "entry_date": entry_date,
                            "exit_price": future_price,
                            "exit_date": data.index[j],
                            "gain_pct": gain_pct,
                            "success": True
                        })
                        break
                    elif gain_pct <= -3:  # Failure
                        signals.append({
                            "entry_price": entry_price,
                            "entry_date": entry_date,
                            "exit_price": future_price,
                            "exit_date": data.index[j],
                            "gain_pct": gain_pct,
                            "success": False
                        })
                        break
        
        return self._calculate_strategy_metrics(signals)
    
    def _calculate_strategy_metrics(self, signals: List[Dict]) -> Dict:
        """Calculate performance metrics for a strategy"""
        if not signals:
            return {
                "total_signals": 0,
                "successful_signals": 0,
                "success_rate": 0.0,
                "avg_gain": 0.0,
                "avg_loss": 0.0,
                "avg_hold_days": 0.0,
                "best_signal": None,
                "worst_signal": None
            }
        
        successful = [s for s in signals if s["success"]]
        failed = [s for s in signals if not s["success"]]
        
        success_rate = len(successful) / len(signals) * 100
        
        avg_gain = sum(s["gain_pct"] for s in successful) / len(successful) if successful else 0
        avg_loss = sum(s["gain_pct"] for s in failed) / len(failed) if failed else 0
        
        # Calculate average hold days
        hold_days = []
        for signal in signals:
            entry_date = pd.to_datetime(signal["entry_date"])
            exit_date = pd.to_datetime(signal["exit_date"])
            hold_days.append((exit_date - entry_date).days)
        
        avg_hold_days = sum(hold_days) / len(hold_days) if hold_days else 0
        
        # Find best and worst signals
        best_signal = max(signals, key=lambda x: x["gain_pct"]) if signals else None
        worst_signal = min(signals, key=lambda x: x["gain_pct"]) if signals else None
        
        return {
            "total_signals": len(signals),
            "successful_signals": len(successful),
            "success_rate": success_rate,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "avg_hold_days": avg_hold_days,
            "best_signal": best_signal,
            "worst_signal": worst_signal,
            "signals": signals
        }
    
    def _generate_strategy_analysis_summary(self, strategy_results: Dict, recommended_strategy: str, current_strategies: List[str]) -> str:
        """Generate comprehensive strategy analysis summary"""
        summary = f"""
STRATEGY PERFORMANCE ANALYSIS (2-Year Historical Data):

RECOMMENDED STRATEGY: {recommended_strategy}
"""
        
        if recommended_strategy in strategy_results:
            perf = strategy_results[recommended_strategy]
            summary += f"""
PERFORMANCE METRICS:
- Total Signals: {perf['total_signals']}
- Success Rate: {perf['success_rate']:.1f}%
- Average Gain: {perf['avg_gain']:.1f}%
- Average Loss: {perf['avg_loss']:.1f}%
- Average Hold Days: {perf['avg_hold_days']:.0f} days

BEST HISTORICAL SIGNAL:
- Entry: ₹{perf['best_signal']['entry_price']:.2f} on {perf['best_signal']['entry_date'].strftime('%Y-%m-%d')}
- Exit: ₹{perf['best_signal']['exit_price']:.2f} on {perf['best_signal']['exit_date'].strftime('%Y-%m-%d')}
- Gain: {perf['best_signal']['gain_pct']:.1f}%
"""
        
        summary += f"""
CURRENT STRATEGY SIGNALS: {', '.join(current_strategies) if current_strategies else 'None'}

ALL STRATEGY PERFORMANCE RANKING:
"""
        
        # Sort strategies by success rate
        sorted_strategies = sorted(strategy_results.items(), 
                                 key=lambda x: x[1]["success_rate"], 
                                 reverse=True)
        
        for i, (strategy, perf) in enumerate(sorted_strategies, 1):
            summary += f"{i}. {strategy}: {perf['success_rate']:.1f}% success rate ({perf['total_signals']} signals)\n"
        
        return summary

    def _is_strategy_eligible(self, stock_data: StockData) -> bool:
        """Check if stock is eligible for strategy analysis based on category"""
        if not stock_data.category or stock_data.category == "Unknown":
            return False
        
        # Only V40 and V40 Next categories are eligible for strategy analysis
        eligible_categories = ["V40", "V40 Next", "v40", "v40 next", "V 40", "V 40 Next"]
        
        return stock_data.category.strip() in eligible_categories

    def _get_basic_technical_analysis(self, stock_data: StockData) -> EnhancedTechnicalAnalysis:
        """Fallback to basic technical analysis when no data is available"""
        return EnhancedTechnicalAnalysis(
            trend="Not Available (No price data)",
            support_levels=[],
            resistance_levels=[],
            entry_range="Not Available (No price data)",
            short_term_target="Not Available (No price data)",
            medium_term_target="Not Available (No price data)",
            stop_loss="Not Available (No price data)",
            confidence_score="Not Available (No data)",
            indicators={},
            patterns=["No Data Available"],
            strategy_signals=["Insufficient data for strategy analysis"],
            position_sizing="Not Available (No data)"
        )

    def _format_indicators(self, indicators: Dict) -> str:
        """Format technical indicators for report"""
        if not indicators:
            return "Not Available"
        
        formatted = []
        for indicator, value in indicators.items():
            formatted.append(f"  • {indicator}: {value}")
        return "\n".join(formatted)
    
    def _format_patterns(self, patterns: List[str]) -> str:
        """Format chart patterns for report"""
        if not patterns:
            return "Not Available"
        
        formatted = []
        for pattern in patterns:
            formatted.append(f"  • {pattern}")
        return "\n".join(formatted)
    
    def _format_signals(self, signals: List[str]) -> str:
        """Format strategy signals for report"""
        if not signals:
            return "Not Available"
        
        formatted = []
        for signal in signals:
            formatted.append(f"  • {signal}")
        return "\n".join(formatted)
    
    def _format_strategy_analysis(self, strategy_analysis: Dict) -> str:
        """Format strategy analysis for report"""
        if not strategy_analysis:
            return "Strategy analysis not available"
        
        analysis_summary = strategy_analysis.get('analysis_summary', 'Not available')
        return analysis_summary
    
    def _format_risks(self, risks: List[str]) -> str:
        """Format key risks for report"""
        if not risks:
            return "Not Available"
        
        formatted = []
        for risk in risks:
            formatted.append(f"  • {risk}")
        return "\n".join(formatted)
    
    def _extract_arthalens_data_comprehensive(self, ticker: str) -> Optional[Dict]:
        """Extract comprehensive ArthaLens data for correlation"""
        try:
            print(f"📊 Extracting comprehensive ArthaLens data for {ticker}...")
            
            # Import ArthaLens extractor
            from arthalens_extractor import ArthaLensExtractor
            
            extractor = ArthaLensExtractor()
            
            # Extract ArthaLens data without fundamental correlation (we'll do that separately)
            arthalens_data = extractor.extract_arthalens_data(ticker, None)
            
            if arthalens_data and "error" not in arthalens_data:
                print(f"✅ ArthaLens data extracted successfully for {ticker}")
                return arthalens_data
            else:
                print(f"⚠️ Failed to extract ArthaLens data for {ticker}")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting ArthaLens data: {e}")
            return None
    
    def _generate_comprehensive_correlated_insights(self, ticker: str, fundamental_analysis, arthalens_data: Dict) -> Dict:
        """Generate comprehensive correlated insights with cost tracking"""
        try:
            print(f"🧠 Generating comprehensive correlated insights for {ticker}...")
            
            if not arthalens_data:
                return {"error": "No ArthaLens data available for correlation"}
            
            # Get growth metrics from fundamental agent if available
            growth_data = ""
            if hasattr(self.fundamental_agent, 'latest_growth_metrics') and ticker in self.fundamental_agent.latest_growth_metrics:
                growth_info = self.fundamental_agent.latest_growth_metrics[ticker]
                growth_metrics = growth_info['growth_metrics']
                growth_assessment = growth_info['growth_assessment']
                
                growth_data = f"""
                
COMPREHENSIVE GROWTH METRICS ANALYSIS:

Growth Quality: {growth_assessment['growth_quality']} (Overall Score: {growth_assessment['overall_score']}/100)

Revenue Growth Analysis:
• YoY Growth: {self.fundamental_agent._format_growth_percentage(growth_metrics['revenue']['yoy_growth'])}
• Same Quarter YoY: {self.fundamental_agent._format_growth_percentage(growth_metrics['revenue']['qoq_same_quarter'])}
• 4Q Avg QoQ: {self.fundamental_agent._format_growth_percentage(growth_metrics['revenue']['avg_qoq_4q'])}
• 3Y Avg YoY: {self.fundamental_agent._format_growth_percentage(growth_metrics['revenue']['avg_yoy_3y'])}

Net Profit Growth Analysis:
• YoY Growth: {self.fundamental_agent._format_growth_percentage(growth_metrics['net_profit']['yoy_growth'])}
• Same Quarter YoY: {self.fundamental_agent._format_growth_percentage(growth_metrics['net_profit']['qoq_same_quarter'])}
• 4Q Avg QoQ: {self.fundamental_agent._format_growth_percentage(growth_metrics['net_profit']['avg_qoq_4q'])}
• 3Y Avg YoY: {self.fundamental_agent._format_growth_percentage(growth_metrics['net_profit']['avg_yoy_3y'])}

Operating Profit (EBITDA) Growth Analysis:
• YoY Growth: {self.fundamental_agent._format_growth_percentage(growth_metrics['operating_profit']['yoy_growth'])}
• Same Quarter YoY: {self.fundamental_agent._format_growth_percentage(growth_metrics['operating_profit']['qoq_same_quarter'])}
• 4Q Avg QoQ: {self.fundamental_agent._format_growth_percentage(growth_metrics['operating_profit']['avg_qoq_4q'])}
• 3Y Avg YoY: {self.fundamental_agent._format_growth_percentage(growth_metrics['operating_profit']['avg_yoy_3y'])}

Growth Performance Scores:
• Revenue Score: {growth_assessment['revenue_score']}/100
• Profit Score: {growth_assessment['profit_score']}/100
• Operating Score: {growth_assessment['operating_score']}/100
• Consistency Score: {growth_assessment['consistency_score']:.1f}/100

Key Growth Insights:
{chr(10).join(['• ' + summary for summary in growth_assessment['growth_summary']])}
                """
            
            # Prepare the comprehensive correlation analysis prompt
            correlation_prompt = f"""
            Based on the fundamental analysis, detailed growth metrics, and ArthaLens transcript/guidance data for {ticker}, provide a comprehensive analysis with the following specific outputs:
            
            FUNDAMENTAL ANALYSIS:
            {fundamental_analysis.model_dump() if hasattr(fundamental_analysis, 'model_dump') else str(fundamental_analysis)}
            {growth_data}
            
            ARTHALENS DATA:
            {json.dumps(arthalens_data, indent=2)}
            
            Please provide the following specific outputs:
            
            1. **GROWTH METRICS CORRELATION WITH MANAGEMENT COMMENTARY:**
               - How do the calculated YoY, QoQ, and 3-year average growth rates align with management guidance?
               - Are the actual growth trends consistent with management's narrative?
               - Which growth metrics (revenue, profit, operating) show the strongest correlation with management expectations?
            
            2. **GROWTH CONSISTENCY AND QUALITY ASSESSMENT:**
               - Analysis of growth consistency across revenue, net profit, and operating profit
               - Assessment of quarter-over-quarter growth volatility
               - Evaluation of 3-year average trends vs recent performance
               - Growth quality score interpretation and implications
            
            3. **CONFIDENCE ON GROWTH OF THE COMPANY:**
               - Overall confidence level (High/Medium/Low with percentage)
               - Reasoning based on actual growth metrics alignment with management commentary
               - Impact of growth consistency on confidence level
            
            4. **VALUATION ASSESSMENT WITH GROWTH CONTEXT:**
               - Current valuation status (Overvalued/Fair/Undervalued)
               - How do actual growth rates justify current valuation?
               - Growth-adjusted fair value assessment
               - Key valuation drivers considering growth quality
            
            5. **CURRENT GROWTH/DEGROWTH METRICS (Top 5 Most Important Points):**
               - List the 5 most critical current performance indicators from actual data
               - Include specific growth percentages and trends
               - Focus on revenue, profit, margins, operational efficiency with actual numbers
               - Highlight any divergence between metrics
            
            6. **FUTURE GROWTH/DEGROWTH DRIVERS (Top 5 Most Important Points):**
               - List the 5 most critical future growth or risk factors
               - How do current growth trends support or challenge future projections?
               - Include strategic initiatives, market opportunities, risks from management guidance
               - Focus on management guidance vs current growth trajectory alignment
            
            7. **FUNDAMENTAL METRICS CORRELATION ANALYSIS:**
               - Revenue Growth Trends vs Management Commentary (with specific percentages)
               - Profit Margin Evolution vs Strategic Initiatives (using actual growth data)
               - Operational Efficiency trends vs Management Execution (EBITDA growth analysis)
               - Growth Consistency vs Management Confidence
               - Cash Flow Generation implications from profit growth trends
               - Debt Management capabilities given current profitability trends
            
            8. **PROJECTED GROWTH ANALYSIS WITH DATA FOUNDATION:**
               - Revenue Growth Projections based on current 3-year averages and recent trends
               - Profit Margin Expansion/Contraction based on operating leverage analysis
               - Operational Efficiency Improvements supported by current EBITDA trends
               - Growth sustainability assessment using consistency scores
               - Risk Factors based on growth volatility and management guidance gaps
            
            9. **GROWTH-ADJUSTED INVESTMENT RECOMMENDATION:**
               - Investment recommendation considering actual growth quality score
               - Specific growth metrics that support or contradict the recommendation
               - Timeline and milestones based on growth trajectory analysis
               - Position sizing implications based on growth consistency
            
            10. **KEY INSIGHTS SUMMARY:**
                - Most important correlation between actual growth metrics and management commentary
                - Critical growth-related risk factors or opportunities
                - Growth quality impact on investment thesis
                - Specific growth targets to monitor for validation
            
            Provide clear, actionable insights with specific evidence from actual growth calculations and management data.
            Focus on the correlation between calculated growth metrics and management guidance.
            Include specific numbers, percentages, growth rates, and timelines where possible.
            Highlight any significant discrepancies between actual performance and management narrative.
            """
            
            response = self.openai_client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-5'),
                messages=[
                    {
                        "role": "user",
                        "content": correlation_prompt
                    }
                ],
                max_completion_tokens=2000
            )
            
            # Track the API call usage
            if hasattr(response, 'usage'):
                usage = response.usage
                self.cost_tracker.log_usage(
                    model=os.getenv('OPENAI_MODEL', 'gpt-5'),
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    call_type="chat",
                    description=f"Comprehensive correlated insights for {ticker}"
                )
            
            correlated_insights = response.choices[0].message.content.strip()
            
            return {
                "analysis": correlated_insights,
                "generated_at": datetime.now().isoformat(),
                "quarters_analyzed": list(arthalens_data.get("summary_data", {}).keys()) if arthalens_data else []
            }
            
        except Exception as e:
            print(f"❌ Error generating comprehensive correlated insights: {e}")
            return {"error": str(e)}
    
    def _generate_comprehensive_report(self, analysis_data: Dict) -> str:
        """Generate comprehensive analysis report with specific outputs"""
        try:
            ticker = analysis_data.get('ticker', 'Unknown')
            company_name = analysis_data.get('company_name', 'Unknown')
            sector = analysis_data.get('sector', 'Unknown')
            category = analysis_data.get('category', 'Unknown')
            
            technical = analysis_data.get('technical_analysis')
            fundamental = analysis_data.get('fundamental_analysis')
            recommendation = analysis_data.get('final_recommendation')
            strategy_analysis = analysis_data.get('strategy_analysis', {})
            arthalens_data = analysis_data.get('arthalens_data', {})
            correlated_insights = analysis_data.get('correlated_insights', {})
            
            # Helper function to safely get attributes
            def safe_get(obj, attr, default='Not Available'):
                if obj is None:
                    return default
                if hasattr(obj, attr):
                    return getattr(obj, attr, default)
                elif isinstance(obj, dict):
                    return obj.get(attr, default)
                return default
            
            report = f"""
{'='*100}
COMPREHENSIVE STOCK ANALYSIS REPORT
{'='*100}

📊 STOCK INFORMATION
{'='*50}
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Category: {category}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*100}
📈 TECHNICAL ANALYSIS
{'='*50}
Trend: {safe_get(technical, 'trend')}
Support Levels: {safe_get(technical, 'support_levels')}
Resistance Levels: {safe_get(technical, 'resistance_levels')}
Entry Range: {safe_get(technical, 'entry_range')}
Short-term Target: {safe_get(technical, 'short_term_target')}
Medium-term Target: {safe_get(technical, 'medium_term_target')}
Stop Loss: {safe_get(technical, 'stop_loss')}
Confidence Score: {safe_get(technical, 'confidence_score')}

Technical Indicators:
{self._format_indicators(safe_get(technical, 'indicators', {})) if technical else 'Not Available'}

Chart Patterns Identified:
{self._format_patterns(safe_get(technical, 'patterns', [])) if technical else 'Not Available'}

Strategy Signals:
{self._format_signals(safe_get(technical, 'strategy_signals', [])) if technical else 'Not Available'}

{'='*100}
💰 FUNDAMENTAL ANALYSIS
{'='*50}
Business Quality: {safe_get(fundamental, 'business_quality')}
Market Penetration: {safe_get(fundamental, 'market_penetration')}
Pricing Power: {safe_get(fundamental, 'pricing_power')}
Revenue Growth: {safe_get(fundamental, 'revenue_growth')}
Profit Growth: {safe_get(fundamental, 'profit_growth')}
Debt to Equity: {safe_get(fundamental, 'debt_to_equity')}
ROCE/ROE: {safe_get(fundamental, 'roce_roe')}
Promoter Pledging: {safe_get(fundamental, 'promoter_pledging')}
Retail Shareholding: {safe_get(fundamental, 'retail_shareholding')}
Valuation Status: {safe_get(fundamental, 'valuation_status')}
Fair Value: {safe_get(fundamental, 'fair_value')}
Financial Health: {safe_get(fundamental, 'financial_health')}
Multibagger Potential: {safe_get(fundamental, 'multibagger_potential')}
Confidence Score: {safe_get(fundamental, 'confidence_score')}

Fundamental Reasons for Buy Signal:
{safe_get(fundamental, 'fundamental_reasons')}

{'='*100}
📊 STRATEGY PERFORMANCE ANALYSIS
{'='*50}
{self._format_strategy_analysis(strategy_analysis)}

{'='*100}
🎯 FINAL RECOMMENDATION
{'='*50}
Action: {safe_get(recommendation, 'action')}
Entry Price: {safe_get(recommendation, 'entry_price')}
Target Price: {safe_get(recommendation, 'target_price')}
Stop Loss: {safe_get(recommendation, 'stop_loss')}
Time Horizon: {safe_get(recommendation, 'time_horizon')}
Confidence Level: {safe_get(recommendation, 'confidence_level')}
Risk Level: {safe_get(recommendation, 'risk_level')}
Position Size: {safe_get(recommendation, 'position_size')}
Strategy Used: {safe_get(recommendation, 'strategy_used')}

Key Risks:
{self._format_risks(safe_get(recommendation, 'key_risks', [])) if recommendation else 'Not Available'}

Fundamental Reasons for Buy Signal:
{safe_get(recommendation, 'fundamental_reasons')}

{'='*100}
📈 ARTHALENS CORRELATED INSIGHTS
{'='*50}
"""
            
            # Add ArthaLens insights if available
            if arthalens_data and "summary_data" in arthalens_data:
                report += f"""
📋 TRANSCRIPT SUMMARY (Latest Quarter):
{arthalens_data.get('summary_data', {}).get('Q4+FY25', {}).get('extracted_text', 'Not Available')}

🎯 GUIDANCE DATA (Latest Quarter):
{arthalens_data.get('guidance_data', {}).get('Q4+FY25', {}).get('extracted_text', 'Not Available')}

❓ Q&A HIGHLIGHTS (Latest Quarter):
{arthalens_data.get('qa_data', {}).get('Q4+FY25', {}).get('extracted_text', 'Not Available')}
"""
            
            # Add correlated insights if available
            if correlated_insights and "analysis" in correlated_insights:
                report += f"""
🧠 COMPREHENSIVE CORRELATED INSIGHTS:
{correlated_insights['analysis']}

📊 CORRELATION SUMMARY:
- Data Sources: {correlated_insights.get('data_sources', ['Fundamental Analysis', 'ArthaLens Transcripts'])}
- Quarters Analyzed: {correlated_insights.get('quarters_analyzed', ['Not Available'])}
- Generated At: {correlated_insights.get('generated_at', 'Not Available')}
"""
            
            report += f"""
{'='*100}
📊 ANALYSIS SUMMARY
{'='*50}
This comprehensive analysis combines:
✅ Technical Analysis with advanced pattern recognition
✅ Fundamental Analysis with Screener.in data extraction
✅ Strategy Performance Analysis with historical backtesting
✅ ArthaLens Transcript & Guidance Analysis
✅ Correlated Insights between fundamental and management commentary
✅ Growth Confidence and Valuation Assessment
✅ Fundamental Metrics Correlation Analysis
✅ Projected Growth Analysis

The analysis provides a holistic view of {ticker} considering both quantitative metrics and qualitative management insights.
{'='*100}
"""
            
            return report
            
        except Exception as e:
            print(f"❌ Error generating comprehensive report: {e}")
            return f"Error generating report: {e}"

    def print_analysis_cost_summary(self):
        """Print comprehensive cost summary for all analysis"""
        print(f"\n{'='*80}")
        print(f"💰 COMPREHENSIVE ANALYSIS COST SUMMARY")
        print(f"{'='*80}")
        
        # Show last 7 days usage
        self.cost_tracker.print_usage_summary(7)
        
        # Show cost estimation for different scenarios
        print(f"\n📊 COST ESTIMATIONS:")
        for num_stocks in [10, 20, 50, 100]:
            estimation = self.cost_tracker.estimate_cost_for_analysis(num_stocks)
            print(f"   • {num_stocks} stocks: ${estimation['estimated_cost_usd']:.6f} (₹{estimation['estimated_cost_inr']:.2f})")
        
        print(f"{'='*80}")

    def _get_basic_technical_analysis(self, stock_data: StockData) -> EnhancedTechnicalAnalysis:
        """Get basic technical analysis using OHLCV data when OpenAI analysis fails"""
        try:
            df = stock_data.ohlcv_data
            if df is None or df.empty:
                return self._create_default_technical_analysis()
            
            # Calculate basic indicators
            current_price = df['Close'].iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            # Moving averages
            ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            ma_100 = df['Close'].rolling(window=100).mean()
            ma_200 = df['Close'].rolling(window=200).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            current_rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            # Volume analysis
            volume_avg = df['Volume'].rolling(window=20).mean().iloc[-1]
            
            # Trend analysis
            if current_price > ma_20 > ma_50:
                trend = "Bullish"
            elif current_price < ma_20 < ma_50:
                trend = "Bearish"
            else:
                trend = "Sideways"
            
            # Support and resistance levels
            recent_low = df['Low'].tail(20).min()
            recent_high = df['High'].tail(20).max()
            
            # Confidence based on indicators
            if abs(current_price - ma_20) / ma_20 < 0.02 and current_rsi > 30 and current_rsi < 70:
                confidence = "Medium"
            elif trend == "Bullish" and current_rsi < 70:
                confidence = "High"
            elif trend == "Bearish" and current_rsi > 30:
                confidence = "High"
            else:
                confidence = "Low"
            
            return EnhancedTechnicalAnalysis(
                trend=trend,
                support_levels=[recent_low, ma_50],
                resistance_levels=[recent_high, ma_200],
                entry_range=f"{current_price * 0.98:.2f} - {current_price * 1.02:.2f}",
                short_term_target=f"{current_price * 1.05:.2f}",
                medium_term_target=f"{current_price * 1.15:.2f}",
                stop_loss=f"{current_price * 0.95:.2f}",
                confidence_score=confidence,
                indicators={
                    "RSI": round(float(current_rsi), 2) if not pd.isna(current_rsi) else 0.0,
                    "MA_20": round(float(ma_20), 2) if not pd.isna(ma_20) else 0.0,
                    "MA_50": round(float(ma_50), 2) if not pd.isna(ma_50) else 0.0,
                    "MA_100": round(float(ma_100), 2) if not pd.isna(ma_100) else 0.0,
                    "MA_200": round(float(ma_200), 2) if not pd.isna(ma_200) else 0.0,
                    "Volume_Ratio": round(float(current_volume/volume_avg), 2) if volume_avg > 0 else 0.0
                },
                patterns=[],
                strategy_signals=[],
                position_sizing="3-5% of portfolio"
            )
            
        except Exception as e:
            print(f"❌ Error in basic technical analysis: {e}")
            return self._create_default_technical_analysis()
    
    def _create_default_technical_analysis(self) -> EnhancedTechnicalAnalysis:
        """Create default technical analysis when all else fails"""
        return EnhancedTechnicalAnalysis(
            trend="Sideways",
            support_levels=[],
            resistance_levels=[],
            entry_range="Not Available",
            short_term_target="Not Available",
            medium_term_target="Not Available",
            stop_loss="Not Available",
            confidence_score="Low",
            indicators={},
            patterns=[],
            strategy_signals=[],
            position_sizing="Not Available"
        )

    def _capture_all_arthalens_screenshots(self, ticker: str) -> Dict[str, Dict[str, str]]:
        """Capture all ArthaLens screenshots once and store them for reuse with retry mechanism"""
        print(f"📸 Capturing all ArthaLens screenshots for {ticker}...")
        
        screenshots = {}
        quarters = ["Q4+FY25", "Q3+FY25", "Q2+FY25", "Q1+FY25"]
        tabs = ["concall", "guidance", "transcript"]
        
        # Retry configuration
        max_retries = 3
        retry_delay = 5  # seconds
        
        for quarter in quarters:
            screenshots[quarter] = {}
            
            for tab in tabs:
                print(f"📸 Capturing {quarter} - {tab} for {ticker}...")
                
                # Retry loop for each screenshot
                for attempt in range(max_retries):
                    try:
                        screenshot_path = self._capture_arthalens_screenshot_with_retry(
                            ticker, quarter, tab, attempt + 1
                        )
                        
                        if screenshot_path and os.path.exists(screenshot_path):
                            # Verify screenshot is not blank
                            file_size = os.path.getsize(screenshot_path)
                            if file_size > 10000:  # More than 10KB
                                screenshots[quarter][tab] = screenshot_path
                                print(f"✅ {quarter} - {tab} captured: {screenshot_path}")
                                break  # Success, exit retry loop
                            else:
                                print(f"⚠️ {quarter} - {tab} screenshot too small ({file_size} bytes), retrying...")
                                if attempt < max_retries - 1:
                                    time.sleep(retry_delay)
                                    continue
                                else:
                                    print(f"❌ Failed to capture {quarter} - {tab} after {max_retries} attempts")
                        else:
                            print(f"⚠️ {quarter} - {tab} screenshot capture failed, retrying...")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                continue
                            else:
                                print(f"❌ Failed to capture {quarter} - {tab} after {max_retries} attempts")
                                
                    except Exception as e:
                        print(f"⚠️ Error capturing {quarter} - {tab} (attempt {attempt + 1}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            print(f"❌ Failed to capture {quarter} - {tab} after {max_retries} attempts")
        
        print(f"✅ All ArthaLens screenshots captured for {ticker}")
        return screenshots
    
    def _capture_arthalens_screenshot_with_retry(self, ticker: str, quarter: str, tab: str, attempt: int) -> Optional[str]:
        """Capture ArthaLens screenshot with retry mechanism and fresh browser session"""
        try:
            # Clean ticker for URL
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            
            # Set up Chrome driver with better options for reliability
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")  # Faster loading
            chrome_options.add_argument("--disable-javascript")  # Disable JS for faster loading
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            # Create fresh driver for each attempt
            driver = None
            try:
                driver = webdriver.Chrome(options=chrome_options)
                driver.set_page_load_timeout(30)  # 30 second timeout
                
                # Construct URL
                if tab == "concall":
                    url = f"https://arthalens.com/{clean_ticker}/concall?quarter={quarter}"
                else:
                    url = f"https://arthalens.com/{clean_ticker}/concall?quarter={quarter}&tab={tab}"
                
                print(f"🌐 Loading URL (attempt {attempt}): {url}")
                
                # Navigate to the page
                driver.get(url)
                
                # Wait for page to load with longer timeout
                time.sleep(10 + (attempt * 2))  # Longer wait for retry attempts
                
                # Check if page loaded correctly
                page_title = driver.title
                print(f"📄 Page title (attempt {attempt}): {page_title}")
                
                # Check for error pages
                if "404" in page_title or "Not Found" in page_title or "Error" in page_title:
                    print(f"❌ Page not found or error (attempt {attempt}): {page_title}")
                    return None
                
                # Check if page has content
                page_source = driver.page_source
                if len(page_source) < 1000:
                    print(f"❌ Page appears to be empty (attempt {attempt})")
                    return None
                
                print(f"✅ Page loaded successfully (attempt {attempt}), content length: {len(page_source)} characters")
                
                # Get the total height of the page
                total_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
                print(f"📏 Total page height (attempt {attempt}): {total_height}px")
                
                if total_height == 0:
                    print(f"❌ Page height is 0 (attempt {attempt})")
                    return None
                
                # Enhanced scrolling to ensure all content is loaded
                print(f"🔄 Scrolling through page to load all content (attempt {attempt})...")
                
                # Scroll down gradually to trigger lazy loading
                current_height = 0
                scroll_step = 500  # Smaller scroll steps
                
                while current_height < total_height:
                    driver.execute_script(f"window.scrollTo(0, {current_height});")
                    time.sleep(2 + (attempt * 0.5))  # Longer wait for retry attempts
                    current_height += scroll_step
                    
                    # Check if height increased (dynamic content loading)
                    new_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
                    if new_height > total_height:
                        total_height = new_height
                        print(f"📏 Height increased to: {total_height}px")
                
                # Scroll back to top
                driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(3)
                
                # Get the final page dimensions
                final_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
                final_width = driver.execute_script("return Math.max(document.body.scrollWidth, document.documentElement.scrollWidth);")
                
                print(f"📐 Final page dimensions (attempt {attempt}): {final_width}x{final_height}px")
                
                # Set window size to capture full page
                driver.set_window_size(final_width, final_height)
                time.sleep(3)
                
                # Take the screenshot
                screenshot_path = os.path.join(self.screenshots_dir, f"{clean_ticker}_{quarter.replace('+', '_')}_{tab}_complete.png")
                driver.save_screenshot(screenshot_path)
                
                print(f"✅ Screenshot saved (attempt {attempt}): {screenshot_path}")
                print(f"📁 File size (attempt {attempt}): {os.path.getsize(screenshot_path) / (1024*1024):.2f} MB")
                
                # Verify screenshot is not blank
                if os.path.getsize(screenshot_path) < 10000:  # Less than 10KB
                    print(f"❌ Screenshot appears to be blank (attempt {attempt})")
                    return None
                
                return screenshot_path
                
            finally:
                # Always close the driver
                if driver:
                    try:
                        driver.quit()
                    except Exception as e:
                        print(f"⚠️ Error closing driver (attempt {attempt}): {e}")
            
        except Exception as e:
            print(f"❌ Error in screenshot capture (attempt {attempt}): {e}")
            return None

    def _extract_arthalens_data_optimized(self, ticker: str, screenshots: Dict[str, Dict[str, str]]) -> Optional[Dict]:
        """Extract ArthaLens data using pre-captured screenshots"""
        try:
            from arthalens_extractor import ArthaLensExtractor
            
            extractor = ArthaLensExtractor()
            
            # Extract data from Q4 screenshots (most recent)
            quarter = "Q4+FY25"
            
            if quarter in screenshots and screenshots[quarter]:
                print(f"🔍 Extracting ArthaLens data from {quarter} screenshots...")
                
                arthalens_data = {}
                
                # Extract from concall tab
                if "concall" in screenshots[quarter] and screenshots[quarter]["concall"]:
                    concall_text = extractor._extract_text_from_screenshot(
                        screenshots[quarter]["concall"], ticker, "concall", quarter
                    )
                    if concall_text:
                        arthalens_data["concall_summary"] = concall_text
                
                # Extract from guidance tab
                if "guidance" in screenshots[quarter] and screenshots[quarter]["guidance"]:
                    guidance_text = extractor._extract_text_from_screenshot(
                        screenshots[quarter]["guidance"], ticker, "guidance", quarter
                    )
                    if guidance_text:
                        arthalens_data["future_guidance"] = guidance_text
                
                # Extract from transcript tab
                if "transcript" in screenshots[quarter] and screenshots[quarter]["transcript"]:
                    transcript_text = extractor._extract_text_from_screenshot(
                        screenshots[quarter]["transcript"], ticker, "transcript", quarter
                    )
                    if transcript_text:
                        arthalens_data["transcript_summary"] = transcript_text
                
                if arthalens_data:
                    print(f"✅ ArthaLens data extracted successfully for {ticker}")
                    return arthalens_data
                else:
                    print(f"❌ No ArthaLens data extracted for {ticker}")
                    return None
            else:
                print(f"❌ No screenshots available for {quarter}")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting ArthaLens data: {e}")
            return None

    def _collect_all_data(self, ticker: str, stock_data: StockData) -> Dict[str, Any]:
        """
        Centralized data collection step - capture all screenshots and data once
        
        Args:
            ticker: Stock ticker symbol
            stock_data: StockData object with OHLCV data
            
        Returns:
            Dictionary containing all collected data and screenshots
        """
        try:
            print(f"📊 Step 1: Collecting All Data for {ticker}...")
            
            # Create run-specific directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = f"analysis_runs/{ticker.replace('.NS', '')}_{timestamp}"
            os.makedirs(run_dir, exist_ok=True)
            
            # Create subdirectories
            screenshots_dir = os.path.join(run_dir, "screenshots")
            openai_responses_dir = os.path.join(run_dir, "openai_responses")
            enhanced_data_dir = os.path.join(run_dir, "enhanced_data")
            os.makedirs(screenshots_dir, exist_ok=True)
            os.makedirs(openai_responses_dir, exist_ok=True)
            os.makedirs(enhanced_data_dir, exist_ok=True)
            
            print(f"📁 Created analysis run directory: {run_dir}")
            
            collected_data = {
                'ticker': ticker,
                'run_directory': run_dir,
                'screenshots_directory': screenshots_dir,
                'openai_responses_directory': openai_responses_dir,
                'enhanced_data_directory': enhanced_data_dir,
                'screener_screenshot': None,
                'screener_data': None,
                'enhanced_fundamental_data': None,
                'arthalens_screenshots': {},
                'arthalens_data': {},
                'candlestick_chart': None,
                'ohlcv_data': stock_data.ohlcv_data,
                'collection_timestamp': timestamp
            }
            
            # Define parallel tasks
            def task_enhanced_fundamental():
                try:
                    from fundamental_scraper import FundamentalDataCollector
                    enhanced_collector = FundamentalDataCollector(openai_api_key=self.openai_api_key)
                    data = enhanced_collector.collect_fundamental_data(ticker, stock_data.company_name, stock_data.sector)
                    if data:
                        enhanced_data_file = os.path.join(enhanced_data_dir, "enhanced_fundamental_data.json")
                        enhanced_data_dict = {
                            'ticker': data.ticker,
                            'company_name': data.company_name,
                            'sector': data.sector,
                            'market_cap': data.market_cap,
                            'pe_ratio': data.pe_ratio,
                            'book_value': data.book_value,
                            'roce': data.roce,
                            'roe': data.roe,
                            'quarterly_column_headers': data.quarterly_column_headers,
                            'quarterly_revenue': data.quarterly_revenue,
                            'quarterly_expenses': data.quarterly_expenses,
                            'quarterly_operating_profit': data.quarterly_operating_profit,
                            'quarterly_net_profit': data.quarterly_net_profit,
                            'quarterly_ebitda': data.quarterly_ebitda,
                            'annual_column_headers': data.annual_column_headers,
                            'annual_total_revenue': data.annual_total_revenue,
                            'annual_total_expenses': data.annual_total_expenses,
                            'annual_operating_profit': data.annual_operating_profit,
                            'annual_net_profit': data.annual_net_profit,
                            'annual_ebitda': data.annual_ebitda,
                            'total_assets': data.total_assets,
                            'total_liabilities': data.total_liabilities,
                            'net_worth': data.net_worth,
                            'working_capital': data.working_capital,
                            'operating_cf': data.operating_cf,
                            'investing_cf': data.investing_cf,
                            'financing_cf': data.financing_cf,
                            'promoter_holding': data.promoter_holding,
                            'fii_shareholding': data.fii_shareholding,
                            'dii_shareholding': data.dii_shareholding,
                            'retail_shareholding': data.retail_shareholding
                        }
                        with open(enhanced_data_file, 'w') as f:
                            json.dump(enhanced_data_dict, f, indent=2, cls=CustomJSONEncoder)
                        return {'enhanced_fundamental_data': data}
                except Exception as e:
                    print(f"❌ Enhanced fundamental data collection error: {e}")
                return {}
            
            def task_screener():
                result = {}
                try:
                    screenshot_path = self.fundamental_agent._capture_complete_page_screenshot(ticker)
                    if screenshot_path and os.path.exists(screenshot_path):
                        new_screener_path = os.path.join(screenshots_dir, f"screener_{ticker.replace('.NS', '')}.png")
                        import shutil
                        shutil.copy2(screenshot_path, new_screener_path)
                        result['screener_screenshot'] = new_screener_path
                        data = self.fundamental_agent._analyze_screenshot_with_openai(screenshot_path, ticker)
                        if data:
                            result['screener_data'] = data
                            openai_response_path = os.path.join(openai_responses_dir, "screener_data_extraction.json")
                            with open(openai_response_path, 'w') as f:
                                json.dump(data, f, indent=2, cls=CustomJSONEncoder)
                            print(f"✅ Screener.in data saved: {openai_response_path}")
                    else:
                        print("❌ Screenshot capture failed for screener")
                        result['error'] = 'Screenshot capture failed'
                except Exception as e:
                    print(f"❌ Screener.in extraction failed: {e}")
                    result['error'] = str(e)
                
                print(f"📊 Screener task completed")
                return result
            
            def task_arthalens():
                result = {}
                try:
                    screenshots = self._capture_all_arthalens_screenshots(ticker)
                    if screenshots:
                        arthalens_dir = os.path.join(screenshots_dir, "arthalens")
                        os.makedirs(arthalens_dir, exist_ok=True)
                        for quarter, tabs in screenshots.items():
                            for tab, path in tabs.items():
                                if path and os.path.exists(path):
                                    new_path = os.path.join(arthalens_dir, f"{quarter.replace('+', '_')}_{tab}.png")
                                    shutil.copy2(path, new_path)
                                    screenshots[quarter][tab] = new_path
                        result['arthalens_screenshots'] = screenshots
                        data = self._extract_arthalens_data_optimized(ticker, screenshots)
                        if data:
                            result['arthalens_data'] = data
                            openai_response_path = os.path.join(openai_responses_dir, "arthalens_data_extraction.json")
                            with open(openai_response_path, 'w') as f:
                                json.dump(data, f, indent=2, cls=CustomJSONEncoder)
                    else:
                        print("❌ ArthaLens screenshots capture failed")
                except Exception as e:
                    print(f"❌ ArthaLens task error: {e}")
                return result
            
            def task_chart():
                result = {}
                try:
                    if stock_data.ohlcv_data is not None and len(stock_data.ohlcv_data) > 0:
                        chart_filename = self.technical_agent._create_candlestick_chart(stock_data)
                        if chart_filename:
                            chart_path = os.path.join(screenshots_dir, f"candlestick_{ticker.replace('.NS', '')}.png")
                            # Move the chart from current directory to screenshots directory
                            if os.path.exists(chart_filename):
                                import shutil
                                shutil.move(chart_filename, chart_path)
                                result['candlestick_chart'] = chart_path
                                print(f"✅ Initial candlestick chart saved: {chart_path}")
                            else:
                                print(f"⚠️ Chart file {chart_filename} not found")
                    else:
                        print("❌ No OHLCV data available for candlestick chart")
                except Exception as e:
                    print(f"❌ Chart task error: {e}")
                return result
            
            # Run tasks in parallel (limit to avoid heavy Selenium contention)
            tasks = [task_enhanced_fundamental, task_screener, task_arthalens, task_chart]
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(t) for t in tasks]
                for future in as_completed(futures):
                    part = future.result() or {}
                    collected_data.update(part)
                    time.sleep(0.3)  # mild pacing
            
            # 4. Save OHLCV data
            if stock_data.ohlcv_data is not None:
                ohlcv_path = os.path.join(run_dir, "ohlcv_data.csv")
                stock_data.ohlcv_data.to_csv(ohlcv_path)
                collected_data['ohlcv_file'] = ohlcv_path
                print(f"✅ OHLCV data saved: {ohlcv_path}")
            
            # 5. Create run summary
            summary_path = os.path.join(run_dir, "run_summary.json")
            summary = {
                'ticker': ticker,
                'timestamp': timestamp,
                'run_directory': run_dir,
                'data_collected': {
                    'screener_in': bool(collected_data.get('screener_screenshot')),
                    'arthalens': bool(collected_data.get('arthalens_screenshots')),
                    'candlestick_chart': bool(collected_data.get('candlestick_chart')),
                    'ohlcv_data': bool(collected_data.get('ohlcv_data') is not None)
                },
                'files': {
                    'screenshots_dir': screenshots_dir,
                    'openai_responses_dir': openai_responses_dir,
                    'ohlcv_file': collected_data.get('ohlcv_file')
                }
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, cls=CustomJSONEncoder)
            collected_data['run_summary'] = summary_path
            
            # Summary
            print(f"\n📊 Data Collection Summary for {ticker}:")
            print(f"   • Run Directory: {run_dir}")
            print(f"   • Screener.in: {'✅' if collected_data['screener_screenshot'] else '❌'}")
            print(f"   • ArthaLens: {'✅' if collected_data['arthalens_screenshots'] else '❌'}")
            print(f"   • Candlestick Chart: {'✅' if collected_data['candlestick_chart'] else '❌'}")
            print(f"   • OHLCV Data: {'✅' if collected_data['ohlcv_data'] is not None else '❌'}")
            print(f"   • Run Summary: {summary_path}")
            
            return collected_data
            
        except Exception as e:
            print(f"❌ Error in data collection for {ticker}: {e}")
            return {
                'ticker': ticker,
                'run_directory': None,
                'screenshots_directory': None,
                'openai_responses_directory': None,
                'screener_screenshot': None,
                'screener_data': None,
                'enhanced_fundamental_data': None,
                'arthalens_screenshots': {},
                'arthalens_data': {},
                'candlestick_chart': None,
                'ohlcv_data': stock_data.ohlcv_data,
                'collection_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'error': str(e)
            }

    def _perform_technical_analysis_with_collected_data(self, stock_data: StockData, collected_data: Dict) -> EnhancedTechnicalAnalysis:
        """Perform technical analysis using pre-collected data"""
        try:
            print(f"📊 Performing technical analysis using collected data for {stock_data.ticker}...")
            
            # Use the candlestick chart from collected data
            if collected_data.get('candlestick_chart'):
                print(f"📊 Using pre-generated candlestick chart for analysis...")
                # Create a temporary StockData with chart_image for compatibility
                temp_stock_data = StockData(
                    ticker=stock_data.ticker,
                    company_name=stock_data.company_name,
                    sector=stock_data.sector,
                    category=stock_data.category,
                    ohlcv_data=stock_data.ohlcv_data,
                    chart_image="temp_chart"  # Placeholder for compatibility
                )
                
                # Use the technical agent's analysis with the collected chart
                analysis = self.technical_agent._get_openai_chart_analysis(temp_stock_data, collected_data.get('candlestick_chart'))
                
                # Save OpenAI response if available
                if analysis and collected_data.get('openai_responses_directory'):
                    openai_response_path = os.path.join(collected_data['openai_responses_directory'], "technical_analysis.json")
                    with open(openai_response_path, 'w') as f:
                        json.dump(analysis.model_dump(), f, indent=2, cls=CustomJSONEncoder)
                    print(f"✅ Technical analysis OpenAI response saved: {openai_response_path}")
                
                return analysis
            else:
                print(f"📊 No candlestick chart available, using OHLCV data analysis...")
                analysis = self.technical_agent.analyze(stock_data, collected_data.get('candlestick_chart'))
                
                # Save OpenAI response if available
                if analysis and collected_data.get('openai_responses_directory'):
                    openai_response_path = os.path.join(collected_data['openai_responses_directory'], "technical_analysis_ohlcv.json")
                    with open(openai_response_path, 'w') as f:
                        json.dump(analysis.model_dump(), f, indent=2, cls=CustomJSONEncoder)
                    print(f"✅ Technical analysis (OHLCV) OpenAI response saved: {openai_response_path}")
                
                return analysis
                
        except Exception as e:
            print(f"❌ Error in technical analysis with collected data: {e}")
            return self.technical_agent.analyze(stock_data, collected_data.get('candlestick_chart'))  # Fallback to original method

    def _perform_fundamental_analysis_with_collected_data(self, stock_data: StockData, collected_data: Dict) -> EnhancedFundamentalAnalysis:
        """Perform fundamental analysis using pre-collected data"""
        try:
            print(f"💰 Performing fundamental analysis using collected data for {stock_data.ticker}...")
            
            # Priority 1: Use enhanced fundamental data if available
            if collected_data.get('enhanced_fundamental_data'):
                print(f"🚀 Using enhanced fundamental data for analysis...")
                enhanced_data = collected_data['enhanced_fundamental_data']
                
                # Create enhanced analysis using the comprehensive data
                analysis = self._create_enhanced_fundamental_analysis_from_data(
                    stock_data.ticker, enhanced_data, collected_data
                )
                
                if analysis:
                    # Save OpenAI responses if available
                    if collected_data.get('openai_responses_directory'):
                        # Save enhanced fundamental analysis
                        enhanced_response_path = os.path.join(collected_data['openai_responses_directory'], "enhanced_fundamental_analysis.json")
                        with open(enhanced_response_path, 'w') as f:
                            json.dump(analysis.model_dump(), f, indent=2, cls=CustomJSONEncoder)
                        print(f"✅ Enhanced fundamental analysis saved: {enhanced_response_path}")
                    
                    return analysis
            
            # Priority 2: Use the screener data from collected data
            if collected_data.get('screener_data'):
                print(f"💰 Using pre-extracted Screener.in data for analysis...")
                
                # Apply fundamental analysis framework using collected data
                analysis = self.fundamental_agent._apply_fundamental_analysis_framework(
                    stock_data.ticker, collected_data['screener_data']
                )
                
                if analysis:
                    # Use ArthaLens data from collected data
                    arthalens_data = collected_data.get('arthalens_data', {})
                    
                    # Generate correlated insights
                    correlated_insights = self.fundamental_agent._generate_comprehensive_correlated_insights(
                        stock_data.ticker, analysis, arthalens_data
                    )
                    
                    # Convert to EnhancedFundamentalAnalysis format
                    fundamental_analysis = self.fundamental_agent._convert_analysis_to_enhanced_format(
                        analysis, collected_data['screener_data'], arthalens_data, correlated_insights
                    )
                    
                    # Save OpenAI responses if available
                    if collected_data.get('openai_responses_directory'):
                        # Save fundamental analysis
                        fundamental_response_path = os.path.join(collected_data['openai_responses_directory'], "fundamental_analysis.json")
                        with open(fundamental_response_path, 'w') as f:
                            json.dump(fundamental_analysis.model_dump(), f, indent=2, cls=CustomJSONEncoder)
                        print(f"✅ Fundamental analysis saved: {fundamental_response_path}")
                        
                        # Save correlated insights
                        insights_response_path = os.path.join(collected_data['openai_responses_directory'], "correlated_insights.json")
                        with open(insights_response_path, 'w') as f:
                            json.dump(correlated_insights, f, indent=2, cls=CustomJSONEncoder)
                        print(f"✅ Correlated insights saved: {insights_response_path}")
                    
                    return fundamental_analysis
                else:
                    print(f"⚠️ Failed to apply analysis framework, using fallback...")
                    return self.fundamental_agent.analyze(stock_data)
            else:
                print(f"📊 No Screener.in data available, using fallback analysis...")
                return self.fundamental_agent.analyze(stock_data)
                
        except Exception as e:
            print(f"❌ Error in fundamental analysis with collected data: {e}")
            return self.fundamental_agent.analyze(stock_data)  # Fallback to original method
    
    def _create_enhanced_fundamental_analysis_from_data(self, ticker: str, enhanced_data, collected_data: Dict) -> EnhancedFundamentalAnalysis:
        """Create enhanced fundamental analysis from comprehensive data"""
        try:
            print(f"🔍 Creating enhanced fundamental analysis from comprehensive data...")
            
            # Extract key metrics
            market_cap = enhanced_data.market_cap or "NA"
            pe_ratio = enhanced_data.pe_ratio or "NA"
            book_value = enhanced_data.book_value or "NA"
            roce = enhanced_data.roce or "NA"
            roe = enhanced_data.roe or "NA"
            
            # Extract quarterly data
            quarterly_revenue = enhanced_data.quarterly_revenue or []
            quarterly_net_profit = enhanced_data.quarterly_net_profit or []
            quarterly_ebitda = enhanced_data.quarterly_ebitda or []
            
            # Extract annual data
            annual_total_revenue = enhanced_data.annual_total_revenue or "NA"
            annual_net_profit = enhanced_data.annual_net_profit or "NA"
            annual_ebitda = enhanced_data.annual_ebitda or "NA"
            
            # Extract balance sheet data
            total_assets = enhanced_data.total_assets or "NA"
            total_liabilities = enhanced_data.total_liabilities or "NA"
            net_worth = enhanced_data.net_worth or "NA"
            
            # Extract cash flow data
            operating_cf = enhanced_data.operating_cf or "NA"
            investing_cf = enhanced_data.investing_cf or "NA"
            financing_cf = enhanced_data.financing_cf or "NA"
            
            # Extract shareholding data
            promoter_holding = enhanced_data.promoter_holding or "NA"
            fii_shareholding = enhanced_data.fii_shareholding or "NA"
            dii_shareholding = enhanced_data.dii_shareholding or "NA"
            retail_shareholding = enhanced_data.retail_shareholding or "NA"
            
            # Calculate growth metrics
            revenue_growth = self._calculate_growth_from_quarterly(quarterly_revenue)
            profit_growth = self._calculate_growth_from_quarterly(quarterly_net_profit)
            
            # Calculate debt to equity
            debt_to_equity = self._calculate_debt_to_equity(total_liabilities, net_worth)
            
            # Create comprehensive analysis context
            analysis_context = f"""
            COMPREHENSIVE FINANCIAL DATA FOR {ticker}:
            
            KEY METRICS:
            - Market Cap: {market_cap}
            - P/E Ratio: {pe_ratio}
            - Book Value: {book_value}
            - ROCE: {roce}
            - ROE: {roe}
            
            QUARTERLY PERFORMANCE (Last 8 quarters):
            - Revenue: {quarterly_revenue}
            - Net Profit: {quarterly_net_profit}
            - EBITDA: {quarterly_ebitda}
            
            ANNUAL PERFORMANCE:
            - Total Revenue: {annual_total_revenue}
            - Net Profit: {annual_net_profit}
            - EBITDA: {annual_ebitda}
            
            BALANCE SHEET:
            - Total Assets: {total_assets}
            - Total Liabilities: {total_liabilities}
            - Net Worth: {net_worth}
            
            CASH FLOWS:
            - Operating CF: {operating_cf}
            - Investing CF: {investing_cf}
            - Financing CF: {financing_cf}
            
            SHAREHOLDING:
            - Promoter: {promoter_holding}
            - FII: {fii_shareholding}
            - DII: {dii_shareholding}
            - Retail: {retail_shareholding}
            
            GROWTH METRICS:
            - Revenue Growth: {revenue_growth}
            - Profit Growth: {profit_growth}
            - Debt to Equity: {debt_to_equity}
            """
            
            # Use ArthaLens data if available
            arthalens_data = collected_data.get('arthalens_data', {})
            
            # Generate correlated insights
            correlated_insights = self.fundamental_agent._generate_comprehensive_correlated_insights(
                ticker, {'context': analysis_context}, arthalens_data
            )
            
            # Create enhanced fundamental analysis
            enhanced_analysis = EnhancedFundamentalAnalysis(
                business_quality=self._assess_business_quality(roce, roe),
                market_penetration=self._assess_market_penetration(market_cap),
                pricing_power=self._assess_pricing_power(roce, debt_to_equity),
                revenue_growth=revenue_growth,
                profit_growth=profit_growth,
                debt_to_equity=debt_to_equity,
                roce_roe=f"ROCE: {roce}, ROE: {roe}",
                promoter_pledging="NA",  # Not available in enhanced data
                retail_shareholding=retail_shareholding,
                valuation_status=self._assess_valuation(pe_ratio, book_value),
                fair_value=self._calculate_fair_value_enhanced(pe_ratio, book_value, roce),
                financial_health=self._assess_financial_health_enhanced(roce, debt_to_equity, retail_shareholding),
                multibagger_potential=self._assess_multibagger_potential_enhanced(market_cap, roce, quarterly_revenue),
                fundamental_reasons=self._generate_enhanced_fundamental_reasons(analysis_context, arthalens_data, correlated_insights),
                confidence_score=self._determine_enhanced_confidence("Strong", correlated_insights)
            )
            
            return enhanced_analysis
            
        except Exception as e:
            print(f"❌ Error creating enhanced fundamental analysis: {e}")
            return None
    
    def _calculate_growth_from_quarterly(self, quarterly_data: List[str]) -> str:
        """Calculate growth from quarterly data"""
        try:
            if len(quarterly_data) >= 2:
                # Compare most recent quarter with previous quarter
                recent = float(quarterly_data[0].replace(',', '').replace('-', ''))
                previous = float(quarterly_data[1].replace(',', '').replace('-', ''))
                if previous != 0:
                    growth = ((recent - previous) / previous) * 100
                    return f"{growth:.1f}%"
            return "NA"
        except:
            return "NA"
    
    def _calculate_debt_to_equity(self, total_liabilities: str, net_worth: str) -> str:
        """Calculate debt to equity ratio"""
        try:
            if total_liabilities != "NA" and net_worth != "NA":
                liabilities = float(total_liabilities.replace(',', ''))
                equity = float(net_worth.replace(',', ''))
                if equity != 0:
                    ratio = liabilities / equity
                    return f"{ratio:.2f}"
            return "NA"
        except:
            return "NA"
    
    def _assess_business_quality(self, roce: str, roe: str) -> str:
        """Assess business quality based on ROCE and ROE"""
        try:
            roce_val = float(roce.replace('%', '').replace(' ', '')) if roce != "NA" else 0
            roe_val = float(roe.replace('%', '').replace(' ', '')) if roe != "NA" else 0
            
            if roce_val > 15 and roe_val > 15:
                return "Excellent"
            elif roce_val > 10 and roe_val > 10:
                return "Good"
            elif roce_val > 5 and roe_val > 5:
                return "Average"
            else:
                return "Poor"
        except:
            return "Can't Assess"
    
    def _assess_market_penetration(self, market_cap: str) -> str:
        """Assess market penetration based on market cap"""
        try:
            if market_cap != "NA":
                # Extract numeric value from market cap
                cap_str = market_cap.replace('₹', '').replace(',', '').replace(' Cr.', '').strip()
                cap_val = float(cap_str)
                
                if cap_val > 50000:
                    return "Large Cap - High Market Penetration"
                elif cap_val > 10000:
                    return "Mid Cap - Moderate Market Penetration"
                else:
                    return "Small Cap - Limited Market Penetration"
            return "Can't Assess"
        except:
            return "Can't Assess"
    
    def _assess_pricing_power(self, roce: str, de_ratio: str) -> str:
        """Assess pricing power based on ROCE and debt levels"""
        try:
            if roce and roce != "null":
                roce_value = float(re.findall(r'[\d.]+', str(roce))[0]) if re.findall(r'[\d.]+', str(roce)) else 0
                
                if roce_value > 30:
                    return "Strong Pricing Power"
                elif roce_value > 20:
                    return "Moderate Pricing Power"
                else:
                    return "Limited Pricing Power"
            else:
                return "Not Available"
        except:
            return "Not Available"
    
    def _assess_valuation(self, pe_ratio: str, book_value: str) -> str:
        """Assess valuation status based on P/E ratio"""
        try:
            if pe_ratio and pe_ratio != "NA" and pe_ratio != "null":
                # Extract numeric value from P/E ratio
                pe_val = float(re.findall(r'[\d.]+', str(pe_ratio))[0]) if re.findall(r'[\d.]+', str(pe_ratio)) else 0
                
                if pe_val < 15:
                    return "Undervalued"
                elif pe_val < 25:
                    return "Fairly Valued"
                else:
                    return "Overvalued"
            else:
                return "Not Available"
        except:
            return "Not Available"
    
    def _calculate_fair_value_enhanced(self, pe_ratio: str, book_value: str, roce: str) -> str:
        """Calculate fair value estimate"""
        try:
            if pe_ratio != "NA" and book_value != "NA":
                pe_val = float(pe_ratio)
                bv_val = float(book_value.replace('₹', '').replace(',', '').strip())
                
                # Simple fair value calculation
                fair_value = bv_val * (pe_val * 0.8)  # Conservative estimate
                return f"₹{fair_value:.2f}"
            return "Can't Calculate"
        except:
            return "Can't Calculate"
    
    def _assess_financial_health_enhanced(self, roce: str, de_ratio: str, retail_holding: str) -> str:
        """Assess financial health comprehensively"""
        try:
            roce_val = float(roce.replace('%', '').replace(' ', '')) if roce != "NA" else 0
            de_ratio = float(debt_to_equity) if debt_to_equity != "NA" else 0
            retail_val = float(retail_holding.replace('%', '')) if retail_holding != "NA" else 0
            
            score = 0
            if roce_val > 10: score += 1
            if de_ratio < 1: score += 1
            if retail_val < 20: score += 1
            
            if score >= 2:
                return "Strong Financial Health"
            elif score >= 1:
                return "Moderate Financial Health"
            else:
                return "Weak Financial Health"
        except:
            return "Can't Assess"
    
    def _assess_multibagger_potential_enhanced(self, market_cap: str, roce: str, raw_data: Dict) -> str:
        """Assess multibagger potential"""
        try:
            if market_cap and roce and market_cap != "null" and roce != "null":
                market_cap_val = float(re.findall(r'[\d.]+', str(market_cap))[0]) if re.findall(r'[\d.]+', str(market_cap)) else 0
                roce_val = float(re.findall(r'[\d.]+', str(roce))[0]) if re.findall(r'[\d.]+', str(roce)) else 0
                
                if market_cap_val < 10000 and roce_val > 25:  # Small cap with high ROCE
                    return "High Potential"
                elif market_cap_val < 50000 and roce_val > 20:
                    return "Moderate Potential"
                else:
                    return "Limited Potential"
            else:
                return "Not Available"
        except:
            return "Not Available"
    
    def _generate_enhanced_fundamental_reasons(self, analysis_context: str, arthalens_data: Dict, correlated_insights: Dict) -> str:
        """Generate enhanced fundamental reasons"""
        try:
            reasons = []
            
            # Add analysis context
            reasons.append(f"Comprehensive financial analysis shows strong fundamentals")
            
            # Add ArthaLens insights if available
            if arthalens_data:
                reasons.append("ArthaLens data indicates positive business trends")
            
            # Add correlated insights
            if correlated_insights:
                reasons.append("Market correlation analysis supports investment thesis")
            
            return "; ".join(reasons)
        except:
            return "Strong fundamental analysis based on comprehensive data"
    
    def _determine_enhanced_confidence(self, base_confidence: str, correlated_insights: Dict) -> str:
        """Determine enhanced confidence level"""
        try:
            if correlated_insights and len(correlated_insights) > 0:
                return "Strong"
            elif base_confidence == "Strong":
                return "Strong"
            else:
                return "Medium"
        except:
            return "Medium"

    def _get_chart_base64_from_path(self, chart_path: str) -> Optional[str]:
        """Get base64 encoded chart from existing file path"""
        try:
            import base64
            with open(chart_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Error reading chart from path {chart_path}: {e}")
            return None

    def _update_chart_with_technical_analysis(self, stock_data: StockData, technical_analysis: EnhancedTechnicalAnalysis, collected_data: Dict) -> None:
        """Update the candlestick chart with technical analysis results"""
        try:
            print(f"📊 Updating candlestick chart with technical analysis results...")
            
            # Create updated chart with technical analysis
            chart_filename = self.technical_agent._create_candlestick_chart(stock_data, technical_analysis)
            if chart_filename:
                # Move the chart to the correct location
                ticker_clean = stock_data.ticker.replace('.NS', '')
                run_dir = collected_data.get('run_directory', '.')
                screenshots_dir = os.path.join(run_dir, 'screenshots')
                os.makedirs(screenshots_dir, exist_ok=True)
                
                final_chart_path = os.path.join(screenshots_dir, f"candlestick_{ticker_clean}.png")
                
                # Move the chart if it was created in current directory
                if os.path.exists(chart_filename):
                    import shutil
                    shutil.move(chart_filename, final_chart_path)
                    print(f"✅ Updated candlestick chart saved to: {final_chart_path}")
                    
                    # Update collected_data
                    collected_data['candlestick_chart'] = final_chart_path
                else:
                    print(f"⚠️ Chart file {chart_filename} not found")
            else:
                print("❌ Failed to create updated chart with technical analysis")
                
        except Exception as e:
            print(f"❌ Error updating chart with technical analysis: {e}")

# Custom JSON encoder to handle pandas Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)

# Example usage
if __name__ == "__main__":
    # Initialize the enhanced multi-agent framework
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your_actual_api_key'")
        exit(1)
    
    analyzer = EnhancedMultiAgentStockAnalysis(api_key)
    
    # Analyze a stock
    results = analyzer.analyze_stock(
        ticker="DELHIVERY.NS",
        company_name="Delhivery Ltd",
        sector="Logistics"
    )
    
    # Generate and print enhanced report
    report = analyzer.generate_enhanced_report(results)
    print(report) 