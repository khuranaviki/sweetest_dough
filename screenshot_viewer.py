#!/usr/bin/env python3
"""
Screenshot Viewer for Enhanced Stock Analysis System
Helps view and organize all saved screenshots from the analysis
"""

import os
import glob
from datetime import datetime
import subprocess
import platform

def list_screenshots():
    """List all available screenshots"""
    print("📸 SCREENSHOT INVENTORY")
    print("="*60)
    
    # 1. Candlestick Charts
    print("\n🎯 CANDLESTICK CHARTS:")
    candlestick_files = glob.glob("analysis_runs/*/screenshots/candlestick_*.png")
    for file in sorted(candlestick_files):
        size = os.path.getsize(file) / 1024  # KB
        date = datetime.fromtimestamp(os.path.getmtime(file))
        print(f"   📊 {file} ({size:.1f}KB) - {date.strftime('%Y-%m-%d %H:%M')}")
    
    # 2. Screener.in Screenshots
    print("\n📊 SCREENER.IN SCREENSHOTS:")
    screener_files = glob.glob("analysis_runs/*/screenshots/screener_*.png")
    for file in sorted(screener_files):
        size = os.path.getsize(file) / 1024  # KB
        date = datetime.fromtimestamp(os.path.getmtime(file))
        print(f"   📈 {file} ({size:.1f}KB) - {date.strftime('%Y-%m-%d %H:%M')}")
    
    # 3. ArthaLens Screenshots
    print("\n🎙️ ARTHALENS SCREENSHOTS:")
    arthalens_files = glob.glob("screener_screenshots/*.png")
    
    # Group by stock
    stocks = {}
    for file in arthalens_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 3:
            stock = parts[0]
            quarter = f"{parts[1]}_{parts[2]}"
            tab = parts[3] if len(parts) > 3 else "unknown"
            
            if stock not in stocks:
                stocks[stock] = {}
            if quarter not in stocks[stock]:
                stocks[stock][quarter] = []
            
            size = os.path.getsize(file) / 1024  # KB
            stocks[stock][quarter].append((tab, file, size))
    
    for stock in sorted(stocks.keys()):
        print(f"   🏢 {stock}:")
        for quarter in sorted(stocks[stock].keys()):
            print(f"      📅 {quarter}:")
            for tab, file, size in stocks[stock][quarter]:
                print(f"         • {tab}: {os.path.basename(file)} ({size:.1f}KB)")

def open_screenshot(file_path):
    """Open screenshot with default system viewer"""
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", file_path])
        elif platform.system() == "Windows":
            subprocess.run(["start", file_path], shell=True)
        else:  # Linux
            subprocess.run(["xdg-open", file_path])
        print(f"✅ Opened: {file_path}")
    except Exception as e:
        print(f"❌ Error opening {file_path}: {e}")

def view_latest_candlestick():
    """View the most recent candlestick chart"""
    candlestick_files = glob.glob("analysis_runs/*/screenshots/candlestick_*.png")
    if candlestick_files:
        latest = max(candlestick_files, key=os.path.getmtime)
        print(f"📊 Opening latest candlestick chart: {latest}")
        open_screenshot(latest)
    else:
        print("❌ No candlestick charts found")

def view_latest_screener():
    """View the most recent screener screenshot"""
    screener_files = glob.glob("analysis_runs/*/screenshots/screener_*.png")
    if screener_files:
        latest = max(screener_files, key=os.path.getmtime)
        print(f"📈 Opening latest screener screenshot: {latest}")
        open_screenshot(latest)
    else:
        print("❌ No screener screenshots found")

def view_arthalens_for_stock(stock_name):
    """View ArthaLens screenshots for a specific stock"""
    arthalens_files = glob.glob(f"screener_screenshots/{stock_name}_*.png")
    if arthalens_files:
        print(f"🎙️ Opening ArthaLens screenshots for {stock_name}:")
        for file in sorted(arthalens_files):
            print(f"   📄 {os.path.basename(file)}")
            open_screenshot(file)
    else:
        print(f"❌ No ArthaLens screenshots found for {stock_name}")

def get_screenshot_summary():
    """Get summary of all screenshots"""
    print("\n📊 SCREENSHOT SUMMARY")
    print("="*60)
    
    # Count files
    candlestick_count = len(glob.glob("analysis_runs/*/screenshots/candlestick_*.png"))
    screener_count = len(glob.glob("analysis_runs/*/screenshots/screener_*.png"))
    arthalens_count = len(glob.glob("screener_screenshots/*.png"))
    
    # Calculate total size
    total_size = 0
    for file in glob.glob("analysis_runs/*/screenshots/*.png") + glob.glob("screener_screenshots/*.png"):
        total_size += os.path.getsize(file)
    
    print(f"📊 Candlestick Charts: {candlestick_count}")
    print(f"📈 Screener.in Screenshots: {screener_count}")
    print(f"🎙️ ArthaLens Screenshots: {arthalens_count}")
    print(f"💾 Total Size: {total_size / (1024*1024):.1f} MB")
    
    # Latest analysis
    analysis_dirs = glob.glob("analysis_runs/*")
    if analysis_dirs:
        latest_analysis = max(analysis_dirs, key=os.path.getmtime)
        print(f"🕒 Latest Analysis: {os.path.basename(latest_analysis)}")

def main():
    """Main function"""
    print("📸 Enhanced Stock Analysis - Screenshot Viewer")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. List all screenshots")
        print("2. View latest candlestick chart")
        print("3. View latest screener screenshot")
        print("4. View ArthaLens screenshots for DELHIVERY")
        print("5. View ArthaLens screenshots for EDELWEISS")
        print("6. Get screenshot summary")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            list_screenshots()
        elif choice == "2":
            view_latest_candlestick()
        elif choice == "3":
            view_latest_screener()
        elif choice == "4":
            view_arthalens_for_stock("DELHIVERY")
        elif choice == "5":
            view_arthalens_for_stock("EDELWEISS")
        elif choice == "6":
            get_screenshot_summary()
        elif choice == "7":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 