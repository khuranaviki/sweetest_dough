# Enhanced Multi-Agent Stock Analysis System

A comprehensive stock analysis system that combines technical analysis, fundamental analysis, and management insights to provide detailed investment recommendations.

## 🚀 Core Components

### Main Analysis Engine
- **`EnhancedMultiAgent.py`** - The core multi-agent analysis system
- **`streamlit_app_enhanced.py`** - Enhanced Streamlit web application
- **`delhivery_analysis.py`** - Main analysis script for running comprehensive analysis

### Data Collection Modules
- **`fundamental_scraper.py`** - Enhanced fundamental data collection from Screener.in
- **`enhanced_screener_extraction_v3.py`** - Latest enhanced Screener.in data extraction
- **`arthalens_extractor.py`** - ArthaLens management transcript and guidance extraction

### Integration & Utilities
- **`notion_integration.py`** - Notion database integration for storing analysis results
- **`openai_cost_tracker.py`** - OpenAI API usage and cost tracking

### Configuration Files
- **`.env`** - Environment variables (API keys, etc.)
- **`requirements_streamlit.txt`** - Python dependencies for Streamlit app
- **`requirements_multi_agent.txt`** - Python dependencies for multi-agent system
- **`credentials.json`** & **`token.json`** - Notion API credentials

## 🔒 Security

**⚠️ IMPORTANT**: This system handles sensitive API keys. Please read the [SECURITY.md](SECURITY.md) file for comprehensive security guidelines.

### Quick Security Setup
1. **Never commit API keys** to version control
2. **Use environment variables** for all credentials
3. **Copy `.env.example` to `.env`** and fill in your actual keys
4. **Keep `.env` file secure** and never share it

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_streamlit.txt
pip install -r requirements_multi_agent.txt
```

### 2. Set Up Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual API keys
# Required: OPENAI_API_KEY
# Optional: NOTION_TOKEN, NOTION_DATABASE_ID
```

**Required Environment Variables:**
```bash
OPENAI_API_KEY="your-openai-api-key-here"
```

**Optional Environment Variables:**
```bash
NOTION_TOKEN="your-notion-token-here"
NOTION_STOCK_LIST_PAGE_ID="your-notion-page-id"
NOTION_REPORTS_PAGE_ID="your-notion-page-id"
```

### 3. Run Streamlit App
```bash
streamlit run streamlit_app_enhanced.py
```

### 4. Run Direct Analysis
```bash
python delhivery_analysis.py
```

## 🔧 Key Features

### Enhanced Data Collection
- **Dynamic coordinate detection** for accurate Screener.in data extraction
- **Column headers extraction** for quarterly and annual data
- **Comprehensive financial metrics** including expenses, EBITDA, balance sheet, cash flows
- **Management insights** from ArthaLens earnings calls and guidance

### Multi-Agent Analysis
- **Technical Analysis Agent** - Chart patterns, indicators, support/resistance
- **Fundamental Analysis Agent** - Business quality, valuation, financial health
- **Strategy Analysis Agent** - Backtesting, performance metrics
- **Correlation Agent** - Cross-analysis insights

### Advanced Features
- **Real-time cost tracking** for OpenAI API usage
- **Notion integration** for result storage and sharing
- **Enhanced Streamlit UI** with progress tracking and detailed displays
- **Comprehensive reporting** with actionable insights

## 📊 Analysis Output

The system generates:
- Technical analysis with entry/exit points
- Fundamental analysis with business quality assessment
- Strategy performance metrics
- Management insights and guidance
- Comprehensive final recommendations
- Detailed reports in multiple formats

## 🛠️ System Architecture

```
EnhancedMultiAgent.py (Core Engine)
├── Technical Analysis Agent
├── Fundamental Analysis Agent  
├── Strategy Analysis Agent
└── Correlation Agent

Data Collection Layer
├── fundamental_scraper.py (Screener.in)
├── enhanced_screener_extraction_v3.py (Enhanced extraction)
└── arthalens_extractor.py (Management insights)

Presentation Layer
├── streamlit_app_enhanced.py (Web UI)
└── notion_integration.py (Database storage)
```

## 📈 Usage Examples

### Streamlit Web Interface
Access the enhanced web interface at `http://localhost:8501` for interactive analysis.

### Direct Analysis
Run comprehensive analysis for any stock:
```python
from EnhancedMultiAgent import EnhancedMultiAgentStockAnalysis

analyzer = EnhancedMultiAgentStockAnalysis(openai_api_key=os.getenv('OPENAI_API_KEY'))
results = analyzer.analyze_stock("DELHIVERY.NS", "Delhivery Limited", "Logistics")
```

## 🔒 Security Best Practices

- **API keys stored in `.env` file** (not committed to version control)
- **Secure credential management** for Notion integration
- **Cost tracking** to monitor API usage
- **Input validation** and sanitization
- **Secure error handling** without exposing sensitive data

## 📝 Notes

- The system uses the latest enhanced data extraction methods
- All analysis results are saved to `analysis_runs/` directory
- Screenshots and data are cached in `screener_screenshots/` directory
- OpenAI usage is logged in `openai_usage_log.json`
- **Always use environment variables** for API keys
- **Never commit sensitive credentials** to version control

## 📚 Documentation

- [SECURITY.md](SECURITY.md) - Comprehensive security guidelines
- [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - Code cleanup documentation 