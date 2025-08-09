#!/usr/bin/env python3
"""
Test Main System Chart Generation
Tests the main system's candlestick chart generation with proper colors
"""

import os
import sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Add the current directory to the path to import EnhancedMultiAgent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EnhancedMultiAgent import EnhancedMultiAgentStockAnalysis, StockData

def test_main_chart_generation():
    """Test the main system's chart generation"""
    print("🎯 Testing Main System Chart Generation")
    print("=" * 50)
    
    # Test with DELHIVERY.NS
    ticker = "DELHIVERY.NS"
    print(f"📊 Testing with {ticker}")
    
    try:
        # Initialize the main system
        print("🚀 Initializing Enhanced Multi-Agent System...")
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("❌ OPENAI_API_KEY not found in environment variables")
            return False
        
        system = EnhancedMultiAgentStockAnalysis(openai_api_key)
        print("✅ System initialized successfully")
        
        # Fetch stock data
        print("📈 Fetching stock data...")
        data = yf.download(ticker, period="3y", interval="1d", progress=False, auto_adjust=True)
        
        if data is None or len(data) == 0:
            print("❌ No data received")
            return False
            
        print(f"✅ Data fetched: {len(data)} data points")
        print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Create StockData object
        stock_data = StockData(
            ticker=ticker,
            company_name="DELHIVERY LTD",
            sector="Logistics",
            category="V40 Next",
            ohlcv_data=data
        )
        
        # Test the technical agent's chart generation
        print("🎨 Testing technical agent's chart generation...")
        technical_agent = system.technical_agent
        
        # Generate candlestick chart
        chart_base64 = technical_agent._create_candlestick_chart(stock_data)
        
        if chart_base64:
            print("✅ Chart generated successfully")
            
            # Save the chart
            import base64
            output_dir = "test_output"
            os.makedirs(output_dir, exist_ok=True)
            chart_path = os.path.join(output_dir, f"main_system_test_{ticker.replace('.', '_')}.png")
            
            with open(chart_path, 'wb') as f:
                f.write(base64.b64decode(chart_base64))
            
            print(f"✅ Chart saved: {chart_path}")
            
            # Open the chart
            print("🖼️ Opening chart...")
            if sys.platform == "darwin":  # macOS
                os.system(f"open {chart_path}")
            elif sys.platform == "win32":  # Windows
                os.system(f"start {chart_path}")
            else:  # Linux
                os.system(f"xdg-open {chart_path}")
            
            print("\n🎉 Test completed!")
            print("📊 Check the chart - you should see:")
            print("   • Green candles for bullish days (close >= open)")
            print("   • Red candles for bearish days (close < open)")
            print("   • 3-year timeframe with 6-month intervals")
            print("   • 4 moving averages (20, 50, 100, 200)")
            print("   • Volume bars with matching colors")
            print("   • Bollinger Bands")
            
            return True
        else:
            print("❌ Chart generation failed")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_main_chart_generation() 