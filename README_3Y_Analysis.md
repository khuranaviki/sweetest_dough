# 🚀 Enhanced Multi-Agent Stock Analysis System - 3 Year Chart Version

## 📊 Overview

This is a **forked and enhanced version** of the original Enhanced Multi-Agent Stock Analysis System, specifically designed to provide **superior technical analysis** using **3 years of historical data** and **prioritized chart-based OpenAI analysis**.

## 🎯 Key Enhancements

### 1. **Extended Timeframe Analysis**
- **Original**: 6 months of data (~120-150 days)
- **Enhanced**: 3 years of data (~750-800 days)
- **Benefit**: Better trend identification and pattern recognition

### 2. **Chart-Based Analysis Priority**
- **Original**: OHLCV data analysis first, chart analysis secondary
- **Enhanced**: **Chart-based OpenAI analysis is PRIORITY #1**
- **Benefit**: More accurate technical analysis using AI vision capabilities

### 3. **Enhanced Technical Indicators**
- **Original**: MA 20, 50, 200-day
- **Enhanced**: MA 20, 50, 100, 200-day + RSI + Bollinger Bands + Volume MA
- **Benefit**: Comprehensive technical analysis

### 4. **Improved Pattern Recognition**
- **Original**: Short-term patterns
- **Enhanced**: Long-term patterns (RHS, CWH, Head & Shoulders)
- **Benefit**: Better suited for medium-term investments

## 🏗️ Architecture

```
EnhancedMultiAgent_3Y.py
├── EnhancedTechnicalAnalysisAgent
│   ├── _create_enhanced_3y_candlestick_chart()
│   ├── _analyze_with_3y_chart_priority()  ← PRIORITY #1
│   ├── _create_3y_chart_analysis_prompt()
│   └── _parse_3y_chart_analysis()
├── EnhancedMultiAgentStockAnalysis3Y
│   ├── _fetch_3y_stock_data()  ← 3 years instead of 6 months
│   └── analyze_stock()
└── test_3y_analysis.py
```

## 🔧 Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd <repository-directory>
```

2. **Install dependencies**:
```bash
pip install -r requirements_multi_agent.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

## 🚀 Usage

### Quick Start

```python
from EnhancedMultiAgent_3Y import EnhancedMultiAgentStockAnalysis3Y
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize 3-year analyzer
analyzer = EnhancedMultiAgentStockAnalysis3Y(os.getenv('OPENAI_API_KEY'))

# Analyze a stock with 3-year priority
results = analyzer.analyze_stock(
    ticker="DELHIVERY.NS",
    company_name="Delhivery Limited", 
    sector="Logistics"
)

print(f"Trend: {results['technical_analysis']['trend']}")
print(f"Confidence: {results['technical_analysis']['confidence_score']}")
print(f"Data Points: {results['data_points']}")
```

### Run Test Script

```bash
python test_3y_analysis.py
```

This will:
- Test the 3-year analysis on multiple stocks
- Generate comprehensive results
- Save results to JSON file
- Show comparison with original system

## 📈 Analysis Flow

### 1. **Data Collection** (3 Years)
```python
# Fetch 3 years of historical data
data = yf.download(ticker, period="3y", interval="1d")
```

### 2. **Chart Generation** (Enhanced)
```python
# Create comprehensive 3-year chart with:
# - 4 moving averages (20, 50, 100, 200-day)
# - Bollinger Bands
# - RSI indicator
# - Volume analysis
chart_base64 = _create_enhanced_3y_candlestick_chart()
```

### 3. **OpenAI Analysis** (Priority #1)
```python
# Send chart to OpenAI for comprehensive analysis
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Expert technical analyst..."},
        {"role": "user", "content": comprehensive_3y_prompt}
    ]
)
```

### 4. **Pattern Recognition** (Long-term)
- Reverse Head and Shoulders
- Cup and Handle
- Head and Shoulders
- Support/Resistance levels

## 📊 Technical Indicators

### Enhanced Moving Averages
- **MA 20**: Short-term trend
- **MA 50**: Medium-term trend  
- **MA 100**: Long-term trend
- **MA 200**: Major trend

### Additional Indicators
- **RSI (14)**: Overbought/Oversold levels
- **Bollinger Bands**: Volatility and price channels
- **Volume MA (20)**: Volume trend analysis

## 🎯 Analysis Priority

### Priority 1: Chart-Based OpenAI Analysis
```python
if chart_base64:
    return _analyze_with_3y_chart_priority(stock_data, chart_base64)
```

### Priority 2: OHLCV Data Analysis
```python
elif ohlcv_data:
    return _analyze_with_ohlcv_data_3y(stock_data)
```

### Priority 3: Basic Analysis
```python
else:
    return _get_basic_technical_analysis(stock_data)
```

## 📋 Output Format

```json
{
  "ticker": "DELHIVERY.NS",
  "company_name": "Delhivery Limited",
  "sector": "Logistics",
  "analysis_timestamp": "2024-01-15T10:30:00",
  "technical_analysis": {
    "trend": "Bullish",
    "confidence_score": "High",
    "support_levels": [450, 480, 520],
    "resistance_levels": [580, 620, 650],
    "entry_range": "540-560",
    "short_term_target": "580",
    "medium_term_target": "620",
    "stop_loss": "520",
    "indicators": {
      "RSI": 65.5,
      "MA_20": 545.2,
      "MA_50": 532.8,
      "MA_100": 518.4,
      "MA_200": 495.6,
      "3Y_High": 650.0,
      "3Y_Low": 420.0
    },
    "patterns": ["3-Year Analysis"],
    "strategy_signals": ["Enhanced 3-Year Technical Analysis"]
  },
  "data_points": 750,
  "timeframe": "3 Years",
  "analysis_type": "3-Year Chart Priority"
}
```

## 🔍 Comparison with Original

| Feature | Original (6M) | Enhanced (3Y) |
|---------|---------------|---------------|
| **Timeframe** | 6 months | 3 years |
| **Data Points** | ~120-150 | ~750-800 |
| **Chart Priority** | Secondary | **Primary** |
| **Moving Averages** | 20, 50, 200 | 20, 50, 100, 200 |
| **Pattern Detection** | Short-term | Long-term |
| **Confidence** | Medium | High |
| **Investment Horizon** | Short-term | Medium-term |

## 🎯 Benefits

### 1. **Better Trend Identification**
- 3 years of data provides clearer trend patterns
- Reduced noise from short-term fluctuations
- More reliable trend direction

### 2. **Improved Pattern Recognition**
- Long-term patterns are more reliable
- Better identification of major support/resistance
- Higher probability of successful breakouts

### 3. **Enhanced Confidence**
- More data points = higher statistical significance
- AI vision analysis of comprehensive charts
- Multiple timeframe confirmation

### 4. **Medium-Term Focus**
- Better suited for 6-12 month investments
- Reduced false signals
- More stable technical analysis

## 🚨 Important Notes

### API Usage
- Uses GPT-4o for chart analysis (higher cost)
- Requires OpenAI API key with sufficient credits
- Chart analysis is prioritized but may fail gracefully

### Data Requirements
- Requires 3 years of historical data
- May not work for newly listed stocks
- Falls back to shorter timeframes if needed

### Performance
- Slower than 6-month analysis due to more data
- Higher memory usage for chart generation
- More comprehensive results justify the trade-off

## 🔧 Customization

### Modify Timeframe
```python
# In _fetch_3y_stock_data()
data = yf.download(ticker_format, period="5y", interval="1d")  # 5 years
```

### Adjust Chart Indicators
```python
# In _plot_enhanced_candlesticks_3y()
ma_50 = df['Close'].rolling(window=50).mean()  # Change window
```

### Customize OpenAI Prompt
```python
# In _create_3y_chart_analysis_prompt()
# Modify the prompt for specific analysis requirements
```

## 📞 Support

For issues or questions:
1. Check the test script output
2. Verify OpenAI API key and credits
3. Ensure sufficient historical data exists
4. Review the generated JSON results

## 🎉 Conclusion

The 3-Year Enhanced Analysis System provides **superior technical analysis** by:
- Using **3 years of historical data**
- **Prioritizing chart-based OpenAI analysis**
- Providing **comprehensive technical indicators**
- Focusing on **medium-term investment horizons**

This makes it ideal for investors looking for **more reliable technical signals** and **better trend identification**. 