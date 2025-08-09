#!/usr/bin/env python3
"""
Lightweight Test for Candlestick Chart Colors
Tests the candlestick chart generation with proper green/red colors
"""

import os
import sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def test_candlestick_colors():
    """Test candlestick chart generation with proper colors"""
    print("🎯 Testing Candlestick Chart Colors")
    print("=" * 50)
    
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
        
        # Create candlestick chart with proper colors
        print("🎨 Creating candlestick chart with colors...")
        
        # Set up the figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Calculate moving averages
        ma_20 = data['Close'].rolling(window=20).mean()
        ma_50 = data['Close'].rolling(window=50).mean()
        ma_100 = data['Close'].rolling(window=100).mean()
        ma_200 = data['Close'].rolling(window=200).mean()
        
        # Plot candlesticks with colors
        for i in range(len(data)):
            date = data.index[i]
            open_price = data['Open'].iloc[i]
            high_price = data['High'].iloc[i]
            low_price = data['Low'].iloc[i]
            close_price = data['Close'].iloc[i]
            
            # Determine color - GREEN for bullish, RED for bearish
            color = 'green' if close_price >= open_price else 'red'
            
            # Plot body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            if body_height > 0:
                ax1.bar(date, body_height, bottom=body_bottom, color=color, width=0.8, alpha=0.8)
            
            # Plot wicks
            ax1.plot([date, date], [low_price, high_price], color='black', linewidth=1)
        
        # Plot moving averages
        ax1.plot(data.index, ma_20, label='MA 20', color='blue', linewidth=1, alpha=0.8)
        ax1.plot(data.index, ma_50, label='MA 50', color='orange', linewidth=1, alpha=0.8)
        ax1.plot(data.index, ma_100, label='MA 100', color='purple', linewidth=1, alpha=0.8)
        ax1.plot(data.index, ma_200, label='MA 200', color='red', linewidth=1, alpha=0.8)
        
        ax1.set_title(f'{ticker} - 3-Year Candlestick Chart (Green=Bullish, Red=Bearish)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis for 3-year view
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show every 6 months
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot volume
        volume_colors = []
        for i in range(len(data)):
            close_price = data['Close'].iloc[i]
            open_price = data['Open'].iloc[i]
            volume_colors.append('green' if close_price >= open_price else 'red')
        
        ax2.bar(data.index, data['Volume'], color=volume_colors, alpha=0.7, width=0.8)
        ax2.set_title('Volume Analysis (3-Year)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volume', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis for volume
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Calculate and plot RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Remove NaN values for plotting
        rsi_clean = rsi.dropna()
        dates_clean = data.index[len(data) - len(rsi_clean):]
        
        ax3.plot(dates_clean, rsi_clean, color='purple', linewidth=1)
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.set_title('RSI (14)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('RSI', fontsize=10)
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis for RSI
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save the chart
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, f"test_candlestick_{ticker.replace('.', '_')}.png")
        
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
        print("   • 3-year timeframe with 6-month intervals")
        print("   • 4 moving averages (20, 50, 100, 200)")
        print("   • Volume bars with matching colors")
        print("   • RSI indicator")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_candlestick_colors() 