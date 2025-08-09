#!/usr/bin/env python3
"""
Simple Candlestick Test
Basic test for candlestick chart colors
"""

import os
import sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def simple_candlestick_test():
    """Simple candlestick chart test"""
    print("🎯 Simple Candlestick Test")
    print("=" * 40)
    
    # Test with DELHIVERY.NS
    ticker = "DELHIVERY.NS"
    print(f"📊 Testing with {ticker}")
    
    try:
        # Fetch 3-year data
        print("📈 Fetching 3-year data...")
        data = yf.download(ticker, period="3y", interval="1d", progress=False, auto_adjust=True)
        
        if data is None or len(data) == 0:
            print("❌ No data received")
            return False
            
        print(f"✅ Data fetched: {len(data)} data points")
        print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Create simple candlestick chart
        print("🎨 Creating candlestick chart...")
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot candlesticks with colors
        for i in range(len(data)):
            date = data.index[i]
            open_price = float(data['Open'].iloc[i])
            high_price = float(data['High'].iloc[i])
            low_price = float(data['Low'].iloc[i])
            close_price = float(data['Close'].iloc[i])
            
            # Determine color - GREEN for bullish, RED for bearish
            color = 'green' if close_price >= open_price else 'red'
            
            # Plot body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            if body_height > 0:
                ax.bar(date, body_height, bottom=body_bottom, color=color, width=0.8, alpha=0.8)
            
            # Plot wicks
            ax.plot([date, date], [low_price, high_price], color='black', linewidth=1)
        
        # Add moving averages
        ma_20 = data['Close'].rolling(window=20).mean()
        ma_50 = data['Close'].rolling(window=50).mean()
        
        ax.plot(data.index, ma_20, label='MA 20', color='blue', linewidth=1, alpha=0.8)
        ax.plot(data.index, ma_50, label='MA 50', color='orange', linewidth=1, alpha=0.8)
        
        ax.set_title(f'{ticker} - 3-Year Candlestick Chart\nGreen=Bullish, Red=Bearish', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save the chart
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, f"simple_test_{ticker.replace('.', '_')}.png")
        
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
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
        print("   • 3-year timeframe")
        print("   • Moving averages (20, 50)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_candlestick_test() 