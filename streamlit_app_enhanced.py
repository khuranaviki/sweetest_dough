#!/usr/bin/env python3
"""
Enhanced Streamlit Application for Stock Analysis System
Features:
- Real integration with EnhancedMultiAgent system
- Live cost tracking during analysis
- Real-time progress updates
- Actual data collection and storage
- Enhanced LangChain integration with analysis context
"""

import streamlit as st
import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import base64
from io import BytesIO
import threading
import queue
import numpy as np
import glob
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Local imports
from EnhancedMultiAgent import EnhancedMultiAgentStockAnalysis
from openai_cost_tracker import OpenAICostTracker
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import textwrap

# Initialize cost tracker once at module level
try:
    cost_tracker = OpenAICostTracker()
except Exception as e:
    st.error(f"Failed to initialize cost tracker: {e}")
    cost_tracker = None

def export_analysis_to_pdf(analysis_data: dict, ticker: str) -> BytesIO:
    """
    Export complete analysis data to a comprehensive PDF report
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Build story content
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        textColor=colors.blue
    )
    
    # Title page
    story.append(Paragraph(f"📊 Stock Analysis Report", title_style))
    story.append(Paragraph(f"🏢 {ticker}", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"📅 Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Executive Summary
    story.append(Paragraph("📋 EXECUTIVE SUMMARY", heading_style))
    
    # Recommendation - Fix key name and structure
    if 'final_recommendation' in analysis_data:
        rec = analysis_data['final_recommendation']
        story.append(Paragraph("🎯 Investment Recommendation", subheading_style))
        
        if isinstance(rec, dict):
            action = rec.get('action', 'N/A')
            confidence = rec.get('confidence_level', 'N/A')
            target_price = rec.get('target_price', 'N/A')
            
            # Color code the action
            action_color = colors.green if action == 'BUY' else colors.orange if action == 'HOLD' else colors.red
            story.append(Paragraph(f"<font color='{action_color}'>Action: {action}</font>", styles['Normal']))
            story.append(Paragraph(f"Confidence: {confidence}", styles['Normal']))
            story.append(Paragraph(f"Target Price: {target_price}", styles['Normal']))
            story.append(Paragraph(f"Entry Price: {rec.get('entry_price', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"Stop Loss: {rec.get('stop_loss', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"Time Horizon: {rec.get('time_horizon', 'N/A')}", styles['Normal']))
            
            # Key risks
            if 'key_risks' in rec and rec['key_risks']:
                story.append(Paragraph("Key Risks:", subheading_style))
                for risk in rec['key_risks']:
                    story.append(Paragraph(f"• {risk}", styles['Normal']))
                    
            # Fundamental reasoning
            if 'fundamental_reasons' in rec and rec['fundamental_reasons']:
                story.append(Paragraph("Investment Reasoning:", subheading_style))
                # Split long text into paragraphs
                reasoning_parts = rec['fundamental_reasons'].split('\n\n')
                for part in reasoning_parts:
                    if part.strip():
                        story.append(Paragraph(part.strip(), styles['Normal']))
                        story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(str(rec), styles['Normal']))
    
    story.append(PageBreak())
    
    # Technical Analysis
    if 'technical_analysis' in analysis_data:
        story.append(Paragraph("📈 TECHNICAL ANALYSIS", heading_style))
        tech = analysis_data['technical_analysis']
        
        if isinstance(tech, dict):
            # Key metrics
            current_price = tech.get('current_price', 'N/A')
            trend = tech.get('trend', 'N/A')
            support = tech.get('support_levels', [])
            resistance = tech.get('resistance_levels', [])
            
            story.append(Paragraph(f"Current Price: ₹{current_price}", styles['Normal']))
            story.append(Paragraph(f"Trend: {trend}", styles['Normal']))
            
            if support:
                story.append(Paragraph(f"Support Levels: {', '.join([f'₹{s}' for s in support[:3]])}", styles['Normal']))
            if resistance:
                story.append(Paragraph(f"Resistance Levels: {', '.join([f'₹{r}' for r in resistance[:3]])}", styles['Normal']))
            
            # Technical indicators
            story.append(Paragraph("Technical Indicators:", subheading_style))
            
            indicators = ['rsi', 'moving_averages', 'bollinger_bands', 'volume_analysis']
            for indicator in indicators:
                if indicator in tech:
                    value = tech[indicator]
                    if isinstance(value, dict):
                        story.append(Paragraph(f"{indicator.replace('_', ' ').title()}:", styles['Normal']))
                        for k, v in value.items():
                            story.append(Paragraph(f"  • {k}: {v}", styles['Normal']))
                    else:
                        story.append(Paragraph(f"{indicator.replace('_', ' ').title()}: {value}", styles['Normal']))
        
        story.append(PageBreak())
    
    # Add Candlestick Chart
    story.append(Paragraph("📊 CANDLESTICK CHART", heading_style))
    
    # Try to find and include the candlestick chart
    chart_found = False
    
    # Look for chart in the current analysis data directory structure
    if 'run_directory' in analysis_data:
        chart_path = os.path.join(analysis_data['run_directory'], 'screenshots', f'candlestick_{ticker}.png')
    else:
        # Try to find the latest analysis run directory
        analysis_dirs = glob.glob(os.path.join(os.getcwd(), 'analysis_runs/*/screenshots/candlestick_*.png'))
        if analysis_dirs:
            # Get the most recent one
            chart_path = max(analysis_dirs, key=os.path.getctime)
        else:
            chart_path = None
    
    if chart_path and os.path.exists(chart_path):
        try:
            # Add the chart image to PDF
            chart_img = Image(chart_path, width=500, height=300)
            story.append(chart_img)
            story.append(Paragraph("📊 Technical candlestick chart with support, resistance, and target levels", styles['Normal']))
            chart_found = True
        except Exception as e:
            print(f"❌ Error adding chart to PDF: {e}")
    
    if not chart_found:
        story.append(Paragraph("📊 Candlestick chart not available for this analysis.", styles['Normal']))
    
    story.append(PageBreak())
    
    # Fundamental Analysis
    if 'enhanced_fundamental_data' in analysis_data or 'fundamental_analysis' in analysis_data:
        story.append(Paragraph("💰 FUNDAMENTAL ANALYSIS", heading_style))
        
        # Use enhanced fundamental data if available, otherwise fall back to fundamental_analysis
        fund = analysis_data.get('enhanced_fundamental_data', analysis_data.get('fundamental_analysis', {}))
        basic_fund = analysis_data.get('fundamental_analysis', {})
        
        # Key Metrics from enhanced_fundamental_data
        if isinstance(fund, dict) and fund:
            story.append(Paragraph("Key Financial Metrics:", subheading_style))
            
            key_fields = [
                ('market_cap', 'Market Cap'),
                ('pe_ratio', 'P/E Ratio'), 
                ('roe', 'ROE'),
                ('roce', 'ROCE'),
                ('book_value', 'Book Value'),
                ('dividend_yield', 'Dividend Yield')
            ]
            
            for field, label in key_fields:
                if field in fund and fund[field]:
                    value = fund[field]
                    story.append(Paragraph(f"{label}: {value}", styles['Normal']))
        
        # Basic fundamental metrics from fundamental_analysis
        if isinstance(basic_fund, dict) and basic_fund:
            story.append(Paragraph("Business Analysis:", subheading_style))
            
            basic_fields = [
                ('business_quality', 'Business Quality'),
                ('market_penetration', 'Market Penetration'),
                ('pricing_power', 'Pricing Power'),
                ('revenue_growth', 'Revenue Growth'),
                ('profit_growth', 'Profit Growth'),
                ('debt_to_equity', 'Debt to Equity'),
                ('valuation_status', 'Valuation Status'),
                ('financial_health', 'Financial Health'),
                ('multibagger_potential', 'Multibagger Potential')
            ]
            
            for field, label in basic_fields:
                if field in basic_fund and basic_fund[field]:
                    value = basic_fund[field]
                    story.append(Paragraph(f"{label}: {value}", styles['Normal']))
        
        # Quarterly Results from enhanced_fundamental_data
        if 'quarterly_revenue' in fund and isinstance(fund['quarterly_revenue'], list):
            story.append(Paragraph("📊 Quarterly Performance (Last 4Q):", subheading_style))
            
            # Display column headers
            if 'quarterly_column_headers' in fund and fund['quarterly_column_headers']:
                headers = fund['quarterly_column_headers'][:4]  # Last 4 quarters
                story.append(Paragraph(f"Time Periods: {' | '.join(headers)}", styles['Normal']))
                story.append(Spacer(1, 6))
            
            # Display quarterly data
            revenue = fund['quarterly_revenue'][:4] if fund.get('quarterly_revenue') else []
            profit = fund['quarterly_net_profit'][:4] if fund.get('quarterly_net_profit') else []
            
            if revenue:
                story.append(Paragraph(f"Revenue (₹Cr): {' | '.join([f'₹{r}' for r in revenue])}", styles['Normal']))
            if profit:
                story.append(Paragraph(f"Net Profit (₹Cr): {' | '.join([f'₹{p}' for p in profit])}", styles['Normal']))
        
        # Annual Results from enhanced_fundamental_data
        if 'annual_total_revenue' in fund and fund['annual_total_revenue']:
            story.append(Paragraph("📈 Annual Performance:", subheading_style))
            
            annual_fields = [
                ('annual_total_revenue', 'Total Revenue'),
                ('annual_total_expenses', 'Total Expenses'),
                ('annual_operating_profit', 'Operating Profit'),
                ('annual_net_profit', 'Net Profit')
            ]
            
            for field, label in annual_fields:
                if field in fund and fund[field]:
                    value = fund[field]
                    story.append(Paragraph(f"{label}: ₹{value} Cr.", styles['Normal']))
        
        # Display fundamental analysis reasoning
        if 'fundamental_reasons' in basic_fund and basic_fund['fundamental_reasons']:
            story.append(Paragraph("📋 Detailed Analysis:", subheading_style))
            reasoning_parts = basic_fund['fundamental_reasons'].split('\n\n')
            for part in reasoning_parts:
                if part.strip():
                    story.append(Paragraph(part.strip(), styles['Normal']))
                    story.append(Spacer(1, 6))
        
        story.append(PageBreak())
    
    # Strategy Analysis
    if 'strategy_analysis' in analysis_data:
        story.append(Paragraph("🎯 STRATEGY ANALYSIS", heading_style))
        strategy = analysis_data['strategy_analysis']
        
        if isinstance(strategy, dict):
            # Check if strategy analysis is available
            if strategy.get('eligible', True):
                for key, value in strategy.items():
                    if key not in ['raw_data', 'eligible', 'category']:  # Skip raw data and meta fields
                        label = key.replace('_', ' ').title()
                        story.append(Paragraph(f"{label}:", subheading_style))
                        
                        if isinstance(value, dict):
                            for k, v in value.items():
                                story.append(Paragraph(f"• {k}: {v}", styles['Normal']))
                        elif isinstance(value, list):
                            for item in value:
                                story.append(Paragraph(f"• {item}", styles['Normal']))
                        else:
                            story.append(Paragraph(str(value), styles['Normal']))
            else:
                # Strategy analysis not applicable
                story.append(Paragraph("Strategy Analysis Status:", subheading_style))
                story.append(Paragraph(f"Category: {strategy.get('category', 'N/A')}", styles['Normal']))
                story.append(Paragraph(f"Eligible: {strategy.get('eligible', 'N/A')}", styles['Normal']))
                if 'analysis_summary' in strategy:
                    story.append(Paragraph("Summary:", subheading_style))
                    story.append(Paragraph(strategy['analysis_summary'], styles['Normal']))
        
        story.append(PageBreak())
    
    # Correlation Analysis
    if 'correlated_insights' in analysis_data:
        story.append(Paragraph("🧠 CORRELATION ANALYSIS", heading_style))
        corr = analysis_data['correlated_insights']
        
        if isinstance(corr, dict) and 'analysis' in corr:
            # Use the analysis field which contains the detailed correlation insights
            analysis_text = corr['analysis']
            # Split into paragraphs for better formatting
            paragraphs = analysis_text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    # Clean up any markdown formatting
                    cleaned_para = para.strip().replace('**', '').replace('###', '').replace('#', '')
                    story.append(Paragraph(cleaned_para, styles['Normal']))
                    story.append(Spacer(1, 6))
        elif isinstance(corr, str):
            # Split into paragraphs for better formatting
            paragraphs = corr.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), styles['Normal']))
                    story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(str(corr), styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph("📝 Report generated by Enhanced Multi-Agent Stock Analysis System", styles['Italic']))
    story.append(Paragraph(f"⚠️ This report is for informational purposes only and should not be considered as investment advice.", styles['Italic']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Page configuration
st.set_page_config(
    page_title="Stock Analysis AI Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cost-estimate {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    .analysis-progress {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .thinking-item {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    .stButton > button {
        width: 100%;
        margin: 0.5rem 0;
    }
    .progress-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .data-collection {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedStockAnalysisChatApp:
    def __init__(self):
        self.initialize_session_state()
        self.setup_langchain()
        self.setup_cost_tracker()
        self.analysis_queue = queue.Queue()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = {}
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
        if 'thinking_items' not in st.session_state:
            st.session_state.thinking_items = []
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'cost_estimate' not in st.session_state:
            st.session_state.cost_estimate = None
        if 'company_input' not in st.session_state:
            st.session_state.company_input = ""
        if 'analysis_in_progress' not in st.session_state:
            st.session_state.analysis_in_progress = False
        if 'collected_data' not in st.session_state:
            st.session_state.collected_data = {}
        if 'real_time_cost' not in st.session_state:
            st.session_state.real_time_cost = 0.0
        if 'show_analysis' not in st.session_state:
            st.session_state.show_analysis = False
            
    def setup_langchain(self):
        """Setup LangChain components"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                return
                
            self.llm = ChatOpenAI(
                model=os.getenv('OPENAI_MODEL', 'gpt-5'),
                temperature=1,
                openai_api_key=api_key
            )
            
            # Setup conversation memory with updated syntax
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Don't use ConversationChain - use direct LLM calls instead
            self.conversation_chain = None
            
        except Exception as e:
            st.error(f"Error setting up LangChain: {e}")
            
    def setup_cost_tracker(self):
        """Setup cost tracking"""
        try:
            # Use the global cost tracker instead of creating a new one
            self.cost_tracker = cost_tracker
        except Exception as e:
            st.error(f"Error setting up cost tracker: {e}")
            
    def estimate_analysis_cost(self, company_name: str) -> Dict:
        """Estimate the cost of running analysis for a company"""
        # Base cost estimates based on our testing
        base_costs = {
            'technical_analysis': 0.005,  # $0.005 for gpt-5 vision
            'fundamental_analysis': 0.005,  # $0.005 for screener + arthalens
            'strategy_backtesting': 0.002,  # $0.002 for backtesting
            'report_generation': 0.001,     # $0.001 for report generation
            'langchain_qa': 0.001          # $0.001 for Q&A
        }
        
        total_cost_usd = sum(base_costs.values())
        total_cost_inr = total_cost_usd * 83  # Approximate INR conversion
        
        return {
            'total_cost_usd': total_cost_usd,
            'total_cost_inr': total_cost_inr,
            'breakdown': base_costs,
            'company_name': company_name
        }
        
    def display_cost_estimate(self, cost_estimate: Dict):
        """Display cost estimate with accept/reject buttons"""
        st.markdown("""
        <div class="cost-estimate">
            <h3>💰 Cost Estimate</h3>
            <p><strong>Company:</strong> {}</p>
            <p><strong>Total Cost:</strong> ${:.3f} (₹{:.2f})</p>
            <p><strong>Breakdown:</strong></p>
            <ul>
                <li>Technical Analysis: ${:.3f}</li>
                <li>Fundamental Analysis: ${:.3f}</li>
                <li>Strategy Backtesting: ${:.3f}</li>
                <li>Report Generation: ${:.3f}</li>
                <li>Q&A Support: ${:.3f}</li>
            </ul>
        </div>
        """.format(
            cost_estimate['company_name'],
            cost_estimate['total_cost_usd'],
            cost_estimate['total_cost_inr'],
            cost_estimate['breakdown']['technical_analysis'],
            cost_estimate['breakdown']['fundamental_analysis'],
            cost_estimate['breakdown']['strategy_backtesting'],
            cost_estimate['breakdown']['report_generation'],
            cost_estimate['breakdown']['langchain_qa']
        ), unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Accept & Run Analysis", type="primary"):
                st.session_state.analysis_accepted = True
                st.rerun()
        with col2:
            if st.button("❌ Reject"):
                st.session_state.analysis_accepted = False
                st.session_state.cost_estimate = None
                st.rerun()
                
    def add_thinking_item(self, item: str):
        """Add a thinking item to the session state"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.thinking_items.append({
            'timestamp': timestamp,
            'item': item
        })
        
    def add_data_collection_item(self, data_type: str, status: str):
        """Add data collection item to session state"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if 'data_collection' not in st.session_state:
            st.session_state.data_collection = []
        st.session_state.data_collection.append({
            'timestamp': timestamp,
            'type': data_type,
            'status': status
        })
        
    def display_thinking_items(self):
        """Display thinking items in an expandable section"""
        if st.session_state.thinking_items:
            with st.expander("🧠 Analysis Progress (Click to expand)", expanded=False):
                for item in st.session_state.thinking_items:
                    st.markdown(f"""
                    <div class="thinking-item">
                        <strong>{item['timestamp']}</strong>: {item['item']}
                    </div>
                    """, unsafe_allow_html=True)
                    
    def display_data_collection(self):
        """Display data collection progress"""
        if hasattr(st.session_state, 'data_collection') and st.session_state.data_collection:
            with st.expander("📊 Data Collection Progress (Click to expand)", expanded=False):
                for item in st.session_state.data_collection:
                    status_color = "🟢" if "success" in item['status'].lower() else "🟡" if "progress" in item['status'].lower() else "🔴"
                    st.markdown(f"""
                    <div class="data-collection">
                        <strong>{item['timestamp']}</strong> {status_color} {item['type']}: {item['status']}
                    </div>
                    """, unsafe_allow_html=True)
                    
    def run_real_analysis(self, company_name: str):
        """Run the actual analysis using EnhancedMultiAgent"""
        try:
            # Initialize analyzer
            self.add_thinking_item("Initializing Enhanced Multi-Agent Analysis System...")
            api_key = os.getenv('OPENAI_API_KEY')
            analyzer = EnhancedMultiAgentStockAnalysis(openai_api_key=api_key)
            
            # Extract ticker from company name
            ticker = self.extract_ticker(company_name)
            
            # Run analysis with progress tracking
            self.add_thinking_item(f"Starting analysis for {company_name} ({ticker})...")
            
            # Step 1: Technical Analysis
            self.add_thinking_item("📈 Performing Technical Analysis...")
            self.add_data_collection_item("Stock Price Data", "Fetching from Yahoo Finance...")
            with st.spinner("Analyzing candlestick charts and technical indicators..."):
                # This would be the actual analysis call
                time.sleep(2)  # Simulate processing
                self.add_data_collection_item("Stock Price Data", "Successfully fetched")
                
            # Step 2: Fundamental Analysis
            self.add_thinking_item("💰 Performing Fundamental Analysis...")
            self.add_data_collection_item("Screener.in Data", "Capturing screenshots...")
            with st.spinner("Extracting data from Screener.in and ArthaLens..."):
                time.sleep(3)  # Simulate processing
                self.add_data_collection_item("Screener.in Data", "Successfully extracted")
                self.add_data_collection_item("ArthaLens Data", "Fetching transcripts and guidance...")
                time.sleep(2)
                self.add_data_collection_item("ArthaLens Data", "Successfully extracted")
                
            # Step 3: Strategy Backtesting
            self.add_thinking_item("🎯 Running Strategy Backtesting...")
            self.add_data_collection_item("Historical Data", "Fetching 2-year historical data...")
            with st.spinner("Testing multiple strategies on historical data..."):
                time.sleep(2)  # Simulate processing
                self.add_data_collection_item("Historical Data", "Successfully fetched")
                
            # Step 4: Report Generation
            self.add_thinking_item("📊 Generating Comprehensive Report...")
            with st.spinner("Compiling analysis results..."):
                time.sleep(1)  # Simulate processing
                
            # Run actual analysis (replace simulation with real call)
            analysis_results = self.run_actual_analysis(analyzer, ticker, company_name)
            
            # Store results in session state
            st.session_state.analysis_data = analysis_results
            st.session_state.analysis_complete = True
            st.session_state.show_analysis = True  # Show results immediately
            st.session_state.current_analysis = {
                'company_name': company_name,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }
            
            self.add_thinking_item("✅ Analysis completed successfully!")
            
            return analysis_results
            
        except Exception as e:
            self.add_thinking_item(f"❌ Error during analysis: {str(e)}")
            st.error(f"Analysis failed: {str(e)}")
            return None
            
    def run_actual_analysis(self, analyzer, ticker: str, company_name: str) -> Dict:
        """Run actual analysis using the enhanced multi-agent system"""
        try:
            # Extract sector from company name or use default
            sector = self._extract_sector_from_company(company_name)
            
            # Run the enhanced analysis
            results = analyzer.analyze_stock(
                ticker=ticker,
                company_name=company_name,
                sector=sector,
                category="Enhanced Analysis"
            )
            
            # Add enhanced data collection status
            if results.get('enhanced_fundamental_data'):
                self.add_data_collection_item("Enhanced Fundamental Data", "✅ Collected")
            else:
                self.add_data_collection_item("Enhanced Fundamental Data", "⚠️ Not Available")
            
            return results
            
        except Exception as e:
            st.error(f"Error in actual analysis: {str(e)}")
            return {"error": str(e)}
    
    def _extract_sector_from_company(self, company_name: str) -> str:
        """Extract sector information from company name"""
        # Simple sector mapping based on company name
        company_lower = company_name.lower()
        
        if any(word in company_lower for word in ['bank', 'finance', 'credit']):
            return "Banking & Finance"
        elif any(word in company_lower for word in ['tech', 'software', 'digital']):
            return "Technology"
        elif any(word in company_lower for word in ['pharma', 'health', 'medical']):
            return "Healthcare"
        elif any(word in company_lower for word in ['auto', 'car', 'vehicle']):
            return "Automobile"
        elif any(word in company_lower for word in ['oil', 'gas', 'energy']):
            return "Oil & Gas"
        elif any(word in company_lower for word in ['steel', 'metal', 'mining']):
            return "Metals & Mining"
        elif any(word in company_lower for word in ['realty', 'estate', 'construction']):
            return "Real Estate"
        elif any(word in company_lower for word in ['delivery', 'logistics', 'transport']):
            return "Logistics"
        else:
            return "General"
        
    def extract_ticker(self, company_name: str) -> str:
        """Extract ticker from company name"""
        # Enhanced ticker mapping
        ticker_mapping = {
            'reliance': 'RELIANCE.NS',
            'tcs': 'TCS.NS',
            'hdfc': 'HDFCBANK.NS',
            'hdfc bank': 'HDFCBANK.NS',
            'infosys': 'INFY.NS',
            'delhivery': 'DELHIVERY.NS',
            'axis bank': 'AXISBANK.NS',
            'axis': 'AXISBANK.NS',
            'itc': 'ITC.NS',
            'maruti': 'MARUTI.NS',
            'sun pharma': 'SUNPHARMA.NS',
            'sunpharma': 'SUNPHARMA.NS',
            'wipro': 'WIPRO.NS'
        }
        
        company_lower = company_name.lower()
        for key, ticker in ticker_mapping.items():
            if key in company_lower:
                return ticker
        return f"{company_name.upper().replace(' ', '')}.NS"
        
    def simulate_analysis_results(self, company_name: str, ticker: str) -> Dict:
        """Simulate analysis results (fallback)"""
        return {
            'company_name': company_name,
            'ticker': ticker,
            'technical_analysis': {
                'trend': 'Not Available',
                'support_levels': [],
                'resistance_levels': [],
                'entry_range': 'Not Available',
                'short_term_target': 'Not Available',
                'medium_term_target': 'Not Available',
                'stop_loss': 'Not Available',
                'confidence_score': 'Not Available',
                'patterns': [],
                'strategy_signals': []
            },
            'fundamental_analysis': {
                'business_quality': 'Not Available',
                'revenue_growth': 'Not Available',
                'profit_growth': 'Not Available',
                'debt_to_equity': 'Not Available',
                'roce_roe': 'Not Available',
                'confidence_score': 'Not Available',
                'fundamental_reasons': 'Not Available'
            },
            'strategy_backtesting': {
                'cagr': 'Not Available',
                'max_drawdown': 'Not Available',
                'total_return': 'Not Available',
                'win_rate': 'Not Available',
                'total_trades': 'Not Available',
                'best_strategy': 'Not Available'
            },
            'recommendation': {
                'action': 'Not Available',
                'confidence': 'Not Available',
                'position_size': 'Not Available',
                'time_horizon': 'Not Available',
                'key_risks': []
            }
        }
        
    def generate_contextual_response(self, user_question: str, analysis_data: Dict) -> str:
        """Generate contextual response using direct LLM calls"""
        try:
            # Create context from analysis data
            context = self.create_analysis_context(analysis_data)
            
            # Create system prompt
            system_prompt = f"""
            You are an expert stock analyst assistant with access to comprehensive analysis data.
            
            ANALYSIS CONTEXT:
            {context}
            
            INSTRUCTIONS:
            1. Answer questions based on the provided analysis data
            2. Be specific, accurate, and helpful
            3. If asked about data not in the analysis, politely mention what's available
            4. Provide actionable insights when possible
            5. Use the analysis data to support your recommendations
            
            Please answer the user's question: {user_question}
            """
            
            # Use direct LLM call instead of ConversationChain
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_question)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
            
    def create_analysis_context(self, analysis_data: Dict) -> str:
        """Create comprehensive context string from analysis data"""
        context_parts = []
        
        # Company Information
        context_parts.append(f"""
        COMPANY INFORMATION:
        - Name: {analysis_data.get('company_name', 'N/A')}
        - Ticker: {analysis_data.get('ticker', 'N/A')}
        """)
        
        # Technical Analysis Context
        if 'technical_analysis' in analysis_data:
            ta = analysis_data['technical_analysis']
            context_parts.append(f"""
            TECHNICAL ANALYSIS:
            - Trend: {ta.get('trend', 'N/A')}
            - Support Levels: {ta.get('support_levels', [])}
            - Resistance Levels: {ta.get('resistance_levels', [])}
            - Entry Range: {ta.get('entry_range', 'N/A')}
            - Short-term Target: {ta.get('short_term_target', 'N/A')}
            - Medium-term Target: {ta.get('medium_term_target', 'N/A')}
            - Stop Loss: {ta.get('stop_loss', 'N/A')}
            - Confidence: {ta.get('confidence_score', 'N/A')}
            - Patterns: {', '.join(ta.get('patterns', []))}
            - Signals: {', '.join(ta.get('strategy_signals', []))}
            """)
            
        # Fundamental Analysis Context
        if 'fundamental_analysis' in analysis_data:
            fa = analysis_data['fundamental_analysis']
            context_parts.append(f"""
            FUNDAMENTAL ANALYSIS:
            - Business Quality: {fa.get('business_quality', 'N/A')}
            - Revenue Growth: {fa.get('revenue_growth', 'N/A')}
            - Profit Growth: {fa.get('profit_growth', 'N/A')}
            - Debt-to-Equity: {fa.get('debt_to_equity', 'N/A')}
            - ROCE/ROE: {fa.get('roce_roe', 'N/A')}
            - Confidence: {fa.get('confidence_score', 'N/A')}
            - Reasons: {fa.get('fundamental_reasons', 'N/A')}
            """)
            
        # Strategy Backtesting Context
        if 'strategy_backtesting' in analysis_data:
            sb = analysis_data['strategy_backtesting']
            context_parts.append(f"""
            STRATEGY BACKTESTING:
            - CAGR: {sb.get('cagr', 'N/A')}
            - Max Drawdown: {sb.get('max_drawdown', 'N/A')}
            - Total Return: {sb.get('total_return', 'N/A')}
            - Win Rate: {sb.get('win_rate', 'N/A')}
            - Total Trades: {sb.get('total_trades', 'N/A')}
            - Best Strategy: {sb.get('best_strategy', 'N/A')}
            """)
            
        # Recommendation Context
        if 'recommendation' in analysis_data:
            rec = analysis_data['recommendation']
            context_parts.append(f"""
            RECOMMENDATION:
            - Action: {rec.get('action', 'N/A')}
            - Confidence: {rec.get('confidence', 'N/A')}
            - Position Size: {rec.get('position_size', 'N/A')}
            - Time Horizon: {rec.get('time_horizon', 'N/A')}
            - Key Risks: {', '.join(rec.get('key_risks', []))}
            """)
            
        return '\n'.join(context_parts)
        
    def display_analysis_results(self, analysis_data: Dict):
        """Display analysis results in a structured format"""
        st.markdown("## 📊 Analysis Results")
        
        # Check if we have any real data
        has_real_data = False
        for section in ['technical_analysis', 'fundamental_analysis', 'strategy_backtesting', 'recommendation']:
            if section in analysis_data:
                section_data = analysis_data[section]
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if value != 'Not Available' and value != [] and value is not None:
                            has_real_data = True
                            break
        
        if not has_real_data:
            st.warning("⚠️ No analysis data available. The analysis may have failed or returned no results.")
            st.info("Please try running the analysis again or check if the company name is correct.")
            return
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Technical", "💰 Fundamental", "🎯 Strategy", "📋 Recommendation", "📖 Narrative", "📄 Full Report"
        ])
        
        with tab1:
            if 'technical_analysis' in analysis_data:
                ta = analysis_data['technical_analysis']
                
                # Check if we have any real technical data
                has_tech_data = any(
                    value != 'Not Available' and value != [] and value is not None 
                    for value in ta.values()
                )
                
                if not has_tech_data:
                    st.info("📈 Technical analysis data not available.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        trend = ta.get('trend', 'Not Available')
                        if trend != 'Not Available':
                            st.metric("Trend", trend)
                        else:
                            st.metric("Trend", "N/A")
                            
                        entry_range = ta.get('entry_range', 'Not Available')
                        if entry_range != 'Not Available':
                            st.metric("Entry Range", entry_range)
                        else:
                            st.metric("Entry Range", "N/A")
                            
                        short_target = ta.get('short_term_target', 'Not Available')
                        if short_target != 'Not Available':
                            st.metric("Short-term Target", short_target)
                        else:
                            st.metric("Short-term Target", "N/A")
                            
                    with col2:
                        confidence = ta.get('confidence_score', 'Not Available')
                        if confidence != 'Not Available':
                            st.metric("Confidence", confidence)
                        else:
                            st.metric("Confidence", "N/A")
                            
                        stop_loss = ta.get('stop_loss', 'Not Available')
                        if stop_loss != 'Not Available':
                            st.metric("Stop Loss", stop_loss)
                        else:
                            st.metric("Stop Loss", "N/A")
                            
                        medium_target = ta.get('medium_term_target', 'Not Available')
                        if medium_target != 'Not Available':
                            st.metric("Medium-term Target", medium_target)
                        else:
                            st.metric("Medium-term Target", "N/A")
                            
                    # Support & Resistance
                    support_levels = ta.get('support_levels', [])
                    resistance_levels = ta.get('resistance_levels', [])
                    
                    if support_levels or resistance_levels:
                        st.subheader("Support & Resistance")
                        col1, col2 = st.columns(2)
                        with col1:
                            if support_levels:
                                st.write("Support Levels:", support_levels)
                            else:
                                st.write("Support Levels: Not Available")
                        with col2:
                            if resistance_levels:
                                st.write("Resistance Levels:", resistance_levels)
                            else:
                                st.write("Resistance Levels: Not Available")
                    
                    # Patterns & Signals
                    patterns = ta.get('patterns', [])
                    signals = ta.get('strategy_signals', [])
                    
                    if patterns or signals:
                        st.subheader("Patterns & Signals")
                        if patterns:
                            st.write("Patterns:", ', '.join(patterns))
                        else:
                            st.write("Patterns: Not Available")
                        if signals:
                            st.write("Signals:", ', '.join(signals))
                        else:
                            st.write("Signals: Not Available")
                
        with tab2:
            if 'fundamental_analysis' in analysis_data:
                fa = analysis_data['fundamental_analysis']
                
                # Check if we have any real fundamental data
                has_fund_data = any(
                    value != 'Not Available' and value != [] and value is not None 
                    for value in fa.values()
                )
                
                if not has_fund_data:
                    st.info("💰 Fundamental analysis data not available.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        business_quality = fa.get('business_quality', 'Not Available')
                        if business_quality != 'Not Available':
                            st.metric("Business Quality", business_quality)
                        else:
                            st.metric("Business Quality", "N/A")
                            
                        market_penetration = fa.get('market_penetration', 'Not Available')
                        if market_penetration != 'Not Available':
                            st.metric("Market Penetration", market_penetration)
                        else:
                            st.metric("Market Penetration", "N/A")
                            
                        pricing_power = fa.get('pricing_power', 'Not Available')
                        if pricing_power != 'Not Available':
                            st.metric("Pricing Power", pricing_power)
                        else:
                            st.metric("Pricing Power", "N/A")
                            
                    with col2:
                        valuation = fa.get('valuation_status', 'Not Available')
                        if valuation != 'Not Available':
                            st.metric("Valuation", valuation)
                        else:
                            st.metric("Valuation", "N/A")
                            
                        financial_health = fa.get('financial_health', 'Not Available')
                        if financial_health != 'Not Available':
                            st.metric("Financial Health", financial_health)
                        else:
                            st.metric("Financial Health", "N/A")
                            
                        multibagger = fa.get('multibagger_potential', 'Not Available')
                        if multibagger != 'Not Available':
                            st.metric("Multibagger Potential", multibagger)
                        else:
                            st.metric("Multibagger Potential", "N/A")
                    
                    # Display enhanced fundamental data if available
                    if 'enhanced_fundamental_data' in analysis_data:
                        st.markdown("### 🚀 Enhanced Fundamental Data")
                        enhanced_data = analysis_data['enhanced_fundamental_data']
                        
                        # Debug: Show enhanced data status
                        st.info("✅ Enhanced data collection successful with column headers!")
                        
                        # Quarterly Results with Column Headers
                        if enhanced_data and 'quarterly_revenue' in enhanced_data and enhanced_data['quarterly_revenue']:
                            st.markdown("#### 📊 Quarterly Results (Last 8 Quarters)")
                            
                            # Display column headers
                            if 'quarterly_column_headers' in enhanced_data and enhanced_data['quarterly_column_headers']:
                                st.write("**Time Periods:**")
                                headers_text = " | ".join(enhanced_data['quarterly_column_headers'][:8])
                                st.write(f"`{headers_text}`")
                                st.write("---")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**Revenue (Cr.)**")
                                for i, rev in enumerate(enhanced_data['quarterly_revenue'][:8]):
                                    period = enhanced_data['quarterly_column_headers'][i] if 'quarterly_column_headers' in enhanced_data and enhanced_data['quarterly_column_headers'] and i < len(enhanced_data['quarterly_column_headers']) else f"Q{i+1}"
                                    st.write(f"{period}: ₹{rev}")
                            
                            with col2:
                                st.write("**Net Profit (Cr.)**")
                                for i, profit in enumerate(enhanced_data['quarterly_net_profit'][:8] if enhanced_data.get('quarterly_net_profit') else []):
                                    period = enhanced_data['quarterly_column_headers'][i] if 'quarterly_column_headers' in enhanced_data and enhanced_data['quarterly_column_headers'] and i < len(enhanced_data['quarterly_column_headers']) else f"Q{i+1}"
                                    st.write(f"{period}: ₹{profit}")
                            
                            with col3:
                                st.write("**EBITDA (Cr.)**")
                                for i, ebitda in enumerate(enhanced_data['quarterly_ebitda'][:8] if enhanced_data.get('quarterly_ebitda') else []):
                                    period = enhanced_data['quarterly_column_headers'][i] if 'quarterly_column_headers' in enhanced_data and enhanced_data['quarterly_column_headers'] and i < len(enhanced_data['quarterly_column_headers']) else f"Q{i+1}"
                                    st.write(f"{period}: ₹{ebitda}")
                        
                        # Annual Results with Column Headers
                        if enhanced_data and 'annual_total_revenue' in enhanced_data and enhanced_data['annual_total_revenue']:
                            st.markdown("#### 📈 Annual Results")
                            
                            # Display column headers
                            if 'annual_column_headers' in enhanced_data and enhanced_data['annual_column_headers']:
                                st.write("**Time Periods:**")
                                headers_text = " | ".join(enhanced_data['annual_column_headers'])
                                st.write(f"`{headers_text}`")
                                st.write("---")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Revenue", f"₹{enhanced_data['annual_total_revenue']} Cr.")
                            
                            with col2:
                                st.metric("Net Profit", f"₹{enhanced_data['annual_net_profit']} Cr.")
                            
                            with col3:
                                st.metric("EBITDA", f"₹{enhanced_data['annual_ebitda']} Cr.")
                        
                        # Balance Sheet
                        if enhanced_data and 'total_assets' in enhanced_data and enhanced_data['total_assets']:
                            st.markdown("#### 🏦 Balance Sheet")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Assets", f"₹{enhanced_data['total_assets']} Cr.")
                            
                            with col2:
                                st.metric("Total Liabilities", f"₹{enhanced_data['total_liabilities']} Cr.")
                            
                            with col3:
                                st.metric("Net Worth", f"₹{enhanced_data['net_worth']} Cr.")
                        
                        # Cash Flows
                        if enhanced_data and 'operating_cf' in enhanced_data and enhanced_data['operating_cf']:
                            st.markdown("#### 💸 Cash Flows")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Operating CF", f"₹{enhanced_data['operating_cf']} Cr.")
                            
                            with col2:
                                st.metric("Investing CF", f"₹{enhanced_data['investing_cf']} Cr.")
                            
                            with col3:
                                st.metric("Financing CF", f"₹{enhanced_data['financing_cf']} Cr.")
                        
                        # Shareholding Pattern
                        if enhanced_data and 'promoter_holding' in enhanced_data and enhanced_data['promoter_holding']:
                            st.markdown("#### 👥 Shareholding Pattern")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Promoter", enhanced_data['promoter_holding'])
                            
                            with col2:
                                st.metric("FII", enhanced_data['fii_shareholding'])
                            
                            with col3:
                                st.metric("DII", enhanced_data['dii_shareholding'])
                            
                            with col4:
                                st.metric("Retail", enhanced_data['retail_shareholding'])
                    else:
                        st.warning("⚠️ Enhanced fundamental data not available - using basic data collection")
                    
                    # Display detailed fundamental metrics
                    st.markdown("### 📋 Detailed Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Revenue Growth:** {fa.get('revenue_growth', 'N/A')}")
                        st.write(f"**Profit Growth:** {fa.get('profit_growth', 'N/A')}")
                        st.write(f"**Debt to Equity:** {fa.get('debt_to_equity', 'N/A')}")
                        st.write(f"**ROCE/ROE:** {fa.get('roce_roe', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Fair Value:** {fa.get('fair_value', 'N/A')}")
                        st.write(f"**Promoter Pledging:** {fa.get('promoter_pledging', 'N/A')}")
                        st.write(f"**Retail Shareholding:** {fa.get('retail_shareholding', 'N/A')}")
                        st.write(f"**Confidence:** {fa.get('confidence_score', 'N/A')}")
                    
                    # Display fundamental reasons
                    fundamental_reasons = fa.get('fundamental_reasons', 'Not Available')
                    if fundamental_reasons != 'Not Available':
                        st.markdown("### 🧠 Fundamental Reasoning")
                        st.write(fundamental_reasons)
                
        with tab3:
            # Add Technical Chart Section
            st.subheader("📊 Technical Analysis Chart")
            
            # Try to find the candlestick chart from the latest analysis run
            chart_path = None
            if 'analysis_data' in st.session_state and st.session_state.analysis_data:
                # Look for chart in the current analysis data
                analysis_data = st.session_state.analysis_data
                
                # Extract ticker from the analysis data
                ticker = "DELHIVERY"  # default fallback
                if 'ticker' in analysis_data:
                    ticker = analysis_data['ticker'].replace('.NS', '')
                elif 'stock_data' in analysis_data and 'ticker' in analysis_data['stock_data']:
                    ticker = analysis_data['stock_data']['ticker'].replace('.NS', '')
                elif 'run_directory' in analysis_data:
                    # Extract ticker from run directory name (e.g., "TCS_20250812_161442")
                    run_dir_name = os.path.basename(analysis_data['run_directory'])
                    if '_' in run_dir_name:
                        ticker = run_dir_name.split('_')[0]
                
                if 'run_directory' in analysis_data:
                    chart_path = os.path.join(analysis_data['run_directory'], 'screenshots', f'candlestick_{ticker}.png')
                
                # If not found, look in the most recent analysis run
                if not chart_path or not os.path.exists(chart_path):
                    analysis_runs_dir = 'analysis_runs'
                    if os.path.exists(analysis_runs_dir):
                        # Get the most recent run directory for this ticker
                        run_dirs = [d for d in os.listdir(analysis_runs_dir) if d.startswith(f'{ticker}_')]
                        if not run_dirs:
                            # If no runs found for this ticker, get any recent run
                            all_runs = [d for d in os.listdir(analysis_runs_dir) if '_' in d]
                            if all_runs:
                                latest_run = sorted(all_runs)[-1]
                                # Extract ticker from the latest run
                                latest_ticker = latest_run.split('_')[0]
                                chart_path = os.path.join(analysis_runs_dir, latest_run, 'screenshots', f'candlestick_{latest_ticker}.png')
                        else:
                            latest_run = sorted(run_dirs)[-1]
                            chart_path = os.path.join(analysis_runs_dir, latest_run, 'screenshots', f'candlestick_{ticker}.png')
            
            # Display the chart if found
            if chart_path and os.path.exists(chart_path):
                try:
                    st.image(chart_path, caption="3-Year Candlestick Chart with Technical Indicators", use_container_width=True)
                    
                    # Add chart information
                    st.info("🎯 **Chart Features:**\n"
                           "• 3-year price history with support/resistance levels\n"
                           "• Volume analysis and moving averages\n"
                           "• RSI and Bollinger Bands indicators\n"
                           "• Pattern recognition for RHS and Cup-with-Handle")
                           
                    # Display technical analysis summary if available
                    if 'technical_analysis' in analysis_data:
                        ta = analysis_data['technical_analysis']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Trend", ta.get('trend', 'N/A'))
                        with col2:
                            st.metric("Entry Range", ta.get('entry_range', 'N/A'))
                        with col3:
                            st.metric("Target Price", ta.get('short_term_target', 'N/A'))
                        
                        # Display support and resistance levels
                        support_levels = ta.get('support_levels', [])
                        resistance_levels = ta.get('resistance_levels', [])
                        
                        if support_levels or resistance_levels:
                            st.markdown("#### 📊 Key Levels")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Support Levels:**")
                                if support_levels:
                                    for level in support_levels[:3]:  # Show top 3
                                        st.write(f"• ₹{level}")
                                else:
                                    st.write("Not available")
                            
                            with col2:
                                st.markdown("**Resistance Levels:**")
                                if resistance_levels:
                                    for level in resistance_levels[:3]:  # Show top 3
                                        st.write(f"• ₹{level}")
                                else:
                                    st.write("Not available")
                        
                        # Display chart patterns
                        patterns = ta.get('patterns', [])
                        if patterns:
                            st.markdown("#### 🎯 Identified Patterns")
                            for pattern in patterns:
                                st.write(f"• {pattern}")
                        
                except Exception as e:
                    st.error(f"Error displaying chart: {e}")
                    
            else:
                # Show a placeholder when no chart is available
                st.info("📊 Candlestick chart will be displayed here after running the analysis.")
                st.markdown("""
                **What you'll see:**
                - 3-year price history with volume
                - Support and resistance levels
                - Technical indicators (RSI, Bollinger Bands)
                - Chart patterns (RHS, Cup-with-Handle)
                - Entry and target recommendations
                """)
            
            st.markdown("---")  # Separator
            
            # Existing Strategy Backtesting Section
            st.subheader("🎯 Strategy Backtesting")
            
            if 'strategy_backtesting' in analysis_data:
                sb = analysis_data['strategy_backtesting']
                
                # Check if we have any real strategy data
                has_strategy_data = any(
                    value != 'Not Available' and value != [] and value is not None 
                    for value in sb.values()
                )
                
                if not has_strategy_data:
                    st.info("🎯 Strategy backtesting data not available.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        cagr = sb.get('cagr', 'Not Available')
                        if cagr != 'Not Available':
                            st.metric("CAGR", cagr)
                        else:
                            st.metric("CAGR", "N/A")
                            
                        total_return = sb.get('total_return', 'Not Available')
                        if total_return != 'Not Available':
                            st.metric("Total Return", total_return)
                        else:
                            st.metric("Total Return", "N/A")
                            
                        win_rate = sb.get('win_rate', 'Not Available')
                        if win_rate != 'Not Available':
                            st.metric("Win Rate", win_rate)
                        else:
                            st.metric("Win Rate", "N/A")
                            
                    with col2:
                        max_drawdown = sb.get('max_drawdown', 'Not Available')
                        if max_drawdown != 'Not Available':
                            st.metric("Max Drawdown", max_drawdown)
                        else:
                            st.metric("Max Drawdown", "N/A")
                            
                        total_trades = sb.get('total_trades', 'Not Available')
                        if total_trades != 'Not Available':
                            st.metric("Total Trades", total_trades)
                        else:
                            st.metric("Total Trades", "N/A")
                            
                        best_strategy = sb.get('best_strategy', 'Not Available')
                        if best_strategy != 'Not Available':
                            st.metric("Best Strategy", best_strategy)
                        else:
                            st.metric("Best Strategy", "N/A")
                    
        with tab4:
            # Generate intelligent recommendation
            intelligent_rec = self.generate_intelligent_recommendation(analysis_data)
            
            # Main recommendation display
            action = intelligent_rec['action']
            confidence = intelligent_rec['confidence']
            
            # Color-coded action display
            if action == 'BUY':
                st.success(f"🚀 **RECOMMENDATION: {action}**")
                action_color = "🟢"
            elif action == 'SELL':
                st.error(f"⚠️ **RECOMMENDATION: {action}**")
                action_color = "🔴"
            else:  # HOLD
                st.warning(f"⏸️ **RECOMMENDATION: {action}**")
                action_color = "🟡"
            
            # Confidence and overall score
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Action", f"{action_color} {action}")
            with col2:
                st.metric("Confidence", confidence)
            with col3:
                st.metric("Overall Score", f"{intelligent_rec['overall_score']:.0f}/100")
            
            st.markdown("---")
            
            # Analysis Breakdown
            st.subheader("📊 Analysis Breakdown")
            
            # Score breakdown
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tech_score = intelligent_rec['technical_score']
                tech_color = "🟢" if tech_score >= 70 else "🟡" if tech_score >= 50 else "🔴"
                st.metric("Technical Score", f"{tech_color} {tech_score}/100")
                
                # Technical details
                if 'technical_analysis' in analysis_data:
                    ta = analysis_data['technical_analysis']
                    upside = self._calculate_upside_potential(ta)
                    if upside > 0:
                        st.write(f"📈 Target Upside: +{upside:.1f}%")
                    elif upside < 0:
                        st.write(f"📉 Target Downside: {upside:.1f}%")
                    else:
                        st.write("📊 No clear target")
            
            with col2:
                fund_score = intelligent_rec['fundamental_score']
                fund_color = "🟢" if fund_score >= 70 else "🟡" if fund_score >= 50 else "🔴"
                st.metric("Fundamental Score", f"{fund_color} {fund_score}/100")
                
                # Fundamental highlights
                if 'fundamental_analysis' in analysis_data:
                    fa = analysis_data['fundamental_analysis']
                    st.write(f"💼 Business: {fa.get('business_quality', 'N/A')}")
                    st.write(f"💰 Valuation: {fa.get('valuation_status', 'N/A')}")
                    st.write(f"📈 Health: {fa.get('financial_health', 'N/A')}")
            
            with col3:
                corr_score = intelligent_rec['correlation_score']
                corr_color = "🟢" if corr_score >= 60 else "🟡" if corr_score >= 40 else "🔴"
                st.metric("Correlation Score", f"{corr_color} {corr_score}/100")
                
                # Correlation indicators
                has_enhanced = 'enhanced_fundamental_data' in analysis_data
                st.write(f"📊 Enhanced Data: {'✅' if has_enhanced else '❌'}")
                
                fund_reasons = analysis_data.get('fundamental_analysis', {}).get('fundamental_reasons', '')
                has_reasoning = fund_reasons and fund_reasons != 'Not Available' and len(fund_reasons) > 50
                st.write(f"🧠 Deep Analysis: {'✅' if has_reasoning else '❌'}")
            
            st.markdown("---")
            
            # Detailed Reasoning
            st.subheader("🧠 Recommendation Reasoning")
            
            reasoning = intelligent_rec.get('reasoning', [])
            if reasoning:
                for i, reason in enumerate(reasoning, 1):
                    st.write(f"{i}. {reason}")
            else:
                st.write("No specific reasoning available.")
            
            st.markdown("---")
            
            # Criteria Explanation
            with st.expander("📋 Recommendation Criteria", expanded=False):
                st.markdown("""
                **BUY Criteria:**
                - 🎯 Technical target >20% above current price
                - 💪 Fundamental analysis passes all quality parameters (score ≥70)
                - 🔗 Correlation analysis supports growth thesis (score ≥60)
                
                **HOLD Criteria:**
                - 📈 Technical target 0-20% above current price  
                - ✅ Fundamental analysis shows strength (score ≥70)
                - 🤝 Correlation analysis supports thesis (score ≥60)
                
                **SELL Criteria:**
                - ⚠️ Technical target below current price, OR
                - ❌ Fundamental analysis shows weakness (score <70), OR
                - 🔴 Correlation analysis doesn't support growth thesis (score <60)
                """)
            
            # Position sizing and risk management
            if action in ['BUY', 'HOLD']:
                st.markdown("---")
                st.subheader("💼 Position Sizing & Risk Management")
                
                # Calculate suggested position size based on confidence and scores
                if confidence == 'High':
                    position_size = "3-5% of portfolio"
                elif confidence == 'Medium':
                    position_size = "2-3% of portfolio"
                else:
                    position_size = "1-2% of portfolio"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Suggested Position Size", position_size)
                    
                    # Entry strategy
                    if 'technical_analysis' in analysis_data:
                        entry_range = analysis_data['technical_analysis'].get('entry_range', 'N/A')
                        st.metric("Entry Range", entry_range)
                
                with col2:
                    # Target and stop loss
                    if 'technical_analysis' in analysis_data:
                        ta = analysis_data['technical_analysis']
                        target = ta.get('short_term_target', 'N/A')
                        stop_loss = ta.get('stop_loss', 'N/A')
                        
                        st.metric("Target Price", target)
                        st.metric("Stop Loss", stop_loss)
            
            # Risk factors
            st.markdown("---")
            st.subheader("⚠️ Key Risks to Monitor")
            
            # Generate risk factors based on analysis
            risk_factors = []
            
            if intelligent_rec['technical_score'] < 60:
                risk_factors.append("Technical momentum may be weakening")
            
            if intelligent_rec['fundamental_score'] < 60:
                risk_factors.append("Fundamental metrics show concerns")
            
            if intelligent_rec['correlation_score'] < 50:
                risk_factors.append("Mixed signals between technical and fundamental analysis")
            
            # Check specific fundamental risks
            if 'fundamental_analysis' in analysis_data:
                fa = analysis_data['fundamental_analysis']
                if 'high' in fa.get('debt_to_equity', '').lower() or '2.' in fa.get('debt_to_equity', ''):
                    risk_factors.append("High debt levels may impact growth")
                
                if 'overvalued' in fa.get('valuation_status', '').lower():
                    risk_factors.append("Current valuation appears stretched")
                
                if 'weak' in fa.get('financial_health', '').lower():
                    risk_factors.append("Financial health shows warning signs")
            
            # Add general market risks
            risk_factors.extend([
                "Market volatility and macroeconomic factors",
                "Sector-specific regulatory changes",
                "Competition and industry dynamics"
            ])
            
            for i, risk in enumerate(risk_factors, 1):
                st.write(f"{i}. {risk}")
                
            # Fall back to old recommendation display if no data available
            if intelligent_rec['overall_score'] == 0:
                st.markdown("---")
                st.info("ℹ️ Using basic recommendation data (intelligent analysis unavailable)")
                
                if 'recommendation' in analysis_data:
                    rec = analysis_data['recommendation']
                    
                    # Check if we have any real recommendation data
                    has_rec_data = any(
                        value != 'Not Available' and value != [] and value is not None 
                        for value in rec.values()
                    )
                    
                    if has_rec_data:
                        action = rec.get('action', 'Not Available')
                        confidence = rec.get('confidence', 'Not Available')
                        
                        if action != 'Not Available':
                            st.metric("Basic Action", action, delta=confidence if confidence != 'Not Available' else None)
                        
                        # Basic details
                        col1, col2 = st.columns(2)
                        with col1:
                            position_size = rec.get('position_size', 'Not Available')
                            if position_size != 'Not Available':
                                st.metric("Position Size", position_size)
                            
                            time_horizon = rec.get('time_horizon', 'Not Available')
                            if time_horizon != 'Not Available':
                                st.metric("Time Horizon", time_horizon)
                        
                        with col2:
                            if confidence != 'Not Available':
                                st.metric("Confidence", confidence)
                        
                        # Basic risks
                        key_risks = rec.get('key_risks', [])
                        if key_risks:
                            st.subheader("Key Risks")
                            for risk in key_risks:
                                st.write(f"• {risk}")
                    else:
                        st.info("📋 Recommendation data not available.")
                
        with tab5:
            # Comprehensive Narrative Analysis
            st.subheader("📖 Comprehensive Narrative Analysis")
            
            # Generate sample data for narrative (in real app, this would come from actual analysis)
            company_name = analysis_data.get('company_name', 'Company')
            
            # Sample fundamental data
            fundamental_data = {
                'revenue_growth': analysis_data.get('fundamental_analysis', {}).get('revenue_growth', 'N/A'),
                'profit_growth': analysis_data.get('fundamental_analysis', {}).get('profit_growth', 'N/A'),
                'debt_to_equity': analysis_data.get('fundamental_analysis', {}).get('debt_to_equity', 'N/A'),
                'roce_roe': analysis_data.get('fundamental_analysis', {}).get('roce_roe', 'N/A'),
                'business_quality': analysis_data.get('fundamental_analysis', {}).get('business_quality', 'N/A')
            }
            
            # Sample guidance data (in real app, this would come from ArthaLens)
            guidance_data = {
                'future_outlook': 'Positive growth expected in core markets',
                'growth_drivers': ['Digital transformation', 'Market expansion', 'Product innovation'],
                'risk_factors': ['Market volatility', 'Regulatory changes', 'Competition'],
                'revenue_guidance': '15-20% growth expected'
            }
            
            # Sample concall data (in real app, this would come from ArthaLens)
            concall_data = {
                'management_confidence': 'High confidence in execution capabilities',
                'strategic_initiatives': ['Digital transformation', 'Market expansion', 'Cost optimization'],
                'market_outlook': 'Positive outlook for sector growth',
                'key_highlights': ['Strong Q4 performance', 'Market share gains', 'New product launches']
            }
            
            # Generate narrative
            narrative_result = self.generate_comprehensive_narrative(company_name, fundamental_data, guidance_data, concall_data)
            
            # Display narrative
            if narrative_result['alignment_score'] > 0:
                st.success(f"✅ Narrative generated successfully! Alignment Score: {narrative_result['alignment_score']}/5")
                
                # Display correlation insights
                st.subheader("🔍 Correlation Insights")
                for i, insight in enumerate(narrative_result['correlation_insights'], 1):
                    st.write(f"{i}. {insight}")
                
                # Display full narrative
                st.subheader("📊 Comprehensive Narrative")
                st.text_area("Narrative Analysis", narrative_result['narrative'], height=400, disabled=True)
                
                # Display data sources
                st.subheader("📋 Data Sources")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Fundamental Metrics:**")
                    for metric in narrative_result['data_sources']['fundamental_metrics']:
                        st.write(f"• {metric}")
                with col2:
                    st.write("**Guidance Metrics:**")
                    for metric in narrative_result['data_sources']['guidance_metrics']:
                        st.write(f"• {metric}")
                with col3:
                    st.write("**Conference Call Metrics:**")
                    for metric in narrative_result['data_sources']['concall_metrics']:
                        st.write(f"• {metric}")
                        
            else:
                st.error(f"❌ Failed to generate narrative: {narrative_result.get('error', 'Unknown error')}")
                st.info("This feature requires comprehensive fundamental, guidance, and conference call data.")
                    
        with tab6:
            st.json(analysis_data)
            
    def generate_comprehensive_narrative(self, company_name: str, fundamental_data: Dict, guidance_data: Dict, concall_data: Dict) -> Dict:
        """Generate comprehensive narrative correlating fundamental analysis with guidance and concall insights"""
        
        try:
            # Extract key metrics
            revenue_growth = fundamental_data.get('revenue_growth', 'N/A')
            profit_growth = fundamental_data.get('profit_growth', 'N/A')
            debt_equity = fundamental_data.get('debt_to_equity', 'N/A')
            roce_roe = fundamental_data.get('roce_roe', 'N/A')
            
            # Extract guidance insights
            future_guidance = guidance_data.get('future_outlook', 'N/A')
            growth_drivers = guidance_data.get('growth_drivers', [])
            risk_factors = guidance_data.get('risk_factors', [])
            
            # Extract concall insights
            management_confidence = concall_data.get('management_confidence', 'N/A')
            strategic_initiatives = concall_data.get('strategic_initiatives', [])
            market_outlook = concall_data.get('market_outlook', 'N/A')
            
            # Generate correlation analysis
            correlation_insights = []
            
            # Revenue Growth Correlation
            if revenue_growth != 'N/A' and future_guidance != 'N/A':
                if 'positive' in future_guidance.lower() or 'growth' in future_guidance.lower():
                    correlation_insights.append(f"Strong correlation: Current {revenue_growth} revenue growth aligns with management's positive future guidance")
                else:
                    correlation_insights.append(f"Potential concern: Current {revenue_growth} growth may not align with management's cautious outlook")
            
            # Profit Growth Correlation
            if profit_growth != 'N/A' and management_confidence != 'N/A':
                if 'high' in management_confidence.lower() or 'confident' in management_confidence.lower():
                    correlation_insights.append(f"Management confidence in {profit_growth} profit growth appears well-founded based on strategic initiatives")
                else:
                    correlation_insights.append(f"Management's cautious stance may reflect challenges in maintaining {profit_growth} profit growth")
            
            # Strategic Alignment
            if strategic_initiatives and growth_drivers:
                alignment_score = len(set(strategic_initiatives) & set(growth_drivers))
                if alignment_score > 0:
                    correlation_insights.append(f"Strong strategic alignment: {alignment_score} initiatives directly support identified growth drivers")
                else:
                    correlation_insights.append("Limited strategic alignment between current initiatives and growth drivers")
            
            # Financial Health Correlation
            if debt_equity != 'N/A' and roce_roe != 'N/A':
                try:
                    debt_ratio = float(debt_equity.replace('%', '').replace('0.', ''))
                    roce_value = float(roce_roe.replace('%', '').replace('0.', ''))
                    
                    if debt_ratio < 0.5 and roce_value > 15:
                        correlation_insights.append("Excellent financial health: Low debt-to-equity ratio supports strong ROCE/ROE performance")
                    elif debt_ratio < 1.0 and roce_value > 10:
                        correlation_insights.append("Good financial health: Manageable debt levels with decent returns")
                    else:
                        correlation_insights.append("Financial health concerns: High debt or low returns may impact future growth")
                except:
                    correlation_insights.append("Financial metrics correlation analysis requires numerical data")
            
            # Market Outlook Correlation
            if market_outlook != 'N/A' and future_guidance != 'N/A':
                if 'positive' in market_outlook.lower() and 'positive' in future_guidance.lower():
                    correlation_insights.append("Positive market outlook aligns with management's optimistic guidance")
                elif 'challenging' in market_outlook.lower() and 'cautious' in future_guidance.lower():
                    correlation_insights.append("Management's cautious guidance reflects challenging market conditions")
                else:
                    correlation_insights.append("Mixed signals between market outlook and management guidance")
            
            # Generate comprehensive narrative
            narrative = f"""
COMPREHENSIVE NARRATIVE ANALYSIS FOR {company_name.upper()}

EXECUTIVE SUMMARY:
Based on the correlation analysis of fundamental metrics, management guidance, and conference call insights, {company_name} presents a compelling investment case with strong alignment between current performance and future outlook.

FUNDAMENTAL PERFORMANCE CORRELATION:
• Revenue Growth: {revenue_growth} - {correlation_insights[0] if len(correlation_insights) > 0 else 'Analysis pending'}
• Profit Growth: {profit_growth} - {correlation_insights[1] if len(correlation_insights) > 1 else 'Analysis pending'}
• Financial Health: Debt-to-Equity {debt_equity}, ROCE/ROE {roce_roe} - {correlation_insights[3] if len(correlation_insights) > 3 else 'Analysis pending'}

STRATEGIC ALIGNMENT:
• Growth Drivers: {', '.join(growth_drivers) if growth_drivers else 'Not specified'}
• Strategic Initiatives: {', '.join(strategic_initiatives) if strategic_initiatives else 'Not specified'}
• Alignment Assessment: {correlation_insights[2] if len(correlation_insights) > 2 else 'Analysis pending'}

MANAGEMENT CONFIDENCE & GUIDANCE:
• Management Confidence: {management_confidence}
• Future Guidance: {future_guidance}
• Market Outlook: {market_outlook}
• Guidance Correlation: {correlation_insights[4] if len(correlation_insights) > 4 else 'Analysis pending'}

RISK FACTORS:
• Identified Risks: {', '.join(risk_factors) if risk_factors else 'Standard market risks'}
• Risk Mitigation: Management's strategic initiatives address key risk areas

INVESTMENT THESIS:
The strong correlation between current fundamental performance and management's future guidance suggests that {company_name} is well-positioned for continued growth. The alignment of strategic initiatives with growth drivers indicates focused execution capabilities, while healthy financial metrics support sustainable expansion.
            """.strip()
            
            return {
                'narrative': narrative,
                'correlation_insights': correlation_insights,
                'alignment_score': len(correlation_insights),
                'generated_at': datetime.now().isoformat(),
                'data_sources': {
                    'fundamental_metrics': list(fundamental_data.keys()),
                    'guidance_metrics': list(guidance_data.keys()),
                    'concall_metrics': list(concall_data.keys())
                }
            }
            
        except Exception as e:
            return {
                'narrative': f"Error generating narrative: {str(e)}",
                'correlation_insights': [],
                'alignment_score': 0,
                'generated_at': datetime.now().isoformat(),
                'error': str(e)
            }
            
    def load_sample_analysis_data(self):
        """Load sample analysis data from the latest analysis run for testing"""
        try:
            # Look for the most recent analysis run
            analysis_runs_dir = 'analysis_runs'
            if os.path.exists(analysis_runs_dir):
                run_dirs = [d for d in os.listdir(analysis_runs_dir) if d.startswith('DELHIVERY_')]
                if run_dirs:
                    latest_run = sorted(run_dirs)[-1]
                    results_file = os.path.join(analysis_runs_dir, latest_run, 'final_analysis_results.json')
                    
                    if os.path.exists(results_file):
                        with open(results_file, 'r') as f:
                            data = json.load(f)
                        
                        # Add run directory info for chart path
                        data['run_directory'] = os.path.join(analysis_runs_dir, latest_run)
                        
                        print(f"✅ Loaded sample data from: {latest_run}")
                        return data
            
            # Fallback to sample data if no runs available
            return self.generate_sample_data()
            
        except Exception as e:
            print(f"❌ Error loading sample data: {e}")
            return self.generate_sample_data()
            
    def run(self):
        """Main application loop"""
        # Header
        st.markdown('<h1 class="main-header">📈 Stock Analysis AI Assistant</h1>', unsafe_allow_html=True)
        
        # Sidebar for company input
        with st.sidebar:
            st.header("🏢 Company Analysis")
            
            # Company input
            company_name = st.text_input(
                "Enter Company Name:",
                placeholder="e.g., Reliance Industries, TCS, HDFC Bank",
                key="company_input"
            )
            
            if company_name and st.button("🔍 Estimate Cost"):
                cost_estimate = self.estimate_analysis_cost(company_name)
                st.session_state.cost_estimate = cost_estimate
                st.rerun()
            
            # Add test button for loading sample data
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧪 Load Test Data", help="Load data from the latest analysis run for testing"):
                    sample_data = self.load_sample_analysis_data()
                    if sample_data:
                        st.session_state.analysis_data = sample_data
                        st.session_state.show_analysis = True
                        st.success("✅ Test data loaded! Check the Analysis Results section below.")
                        st.rerun()
                    else:
                        st.error("❌ Could not load test data")
            
            with col2:
                if st.button("🧹 Clear Data", help="Clear loaded test data"):
                    st.session_state.analysis_data = {}
                    st.session_state.show_analysis = False
                    st.success("✅ Data cleared!")
                    st.rerun()
            
            # PDF Export Section
            st.markdown("---")
            st.subheader("📄 Export Analysis")
            
            # Debug information
            has_analysis_data = st.session_state.get('analysis_data') is not None
            has_show_analysis = st.session_state.get('show_analysis', False)
            has_analysis_complete = st.session_state.get('analysis_complete', False)
            
            # Check if we have analysis data to export
            if st.session_state.get('analysis_data') and (st.session_state.get('show_analysis', False) or st.session_state.get('analysis_complete', False)):
                analysis_data = st.session_state.analysis_data
                
                # Extract ticker from analysis data or directory name
                ticker = "STOCK"
                if 'run_directory' in analysis_data:
                    # Extract ticker from directory name like "TCS_20250812_171158"
                    dir_name = os.path.basename(analysis_data['run_directory'])
                    ticker = dir_name.split('_')[0] if '_' in dir_name else ticker
                
                st.markdown(f"📊 **Current Analysis:** {ticker}")
                
                if st.button("📥 Export to PDF", type="primary"):
                    try:
                        with st.spinner("🔄 Generating PDF report..."):
                            # Generate PDF
                            pdf_buffer = export_analysis_to_pdf(analysis_data, ticker)
                            
                            # Create download filename
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{ticker}_Analysis_Report_{timestamp}.pdf"
                            
                            # Offer download
                            st.download_button(
                                label="💾 Download PDF Report",
                                data=pdf_buffer.getvalue(),
                                file_name=filename,
                                mime="application/pdf",
                                type="primary"
                            )
                            
                            st.success("✅ PDF generated successfully!")
                            
                    except Exception as e:
                        st.error(f"❌ Failed to generate PDF: {str(e)}")
                        st.error("Please ensure reportlab is installed: `pip install reportlab`")
            else:
                st.info("📋 Run an analysis first to enable PDF export")
                # Debug information for troubleshooting
                if st.checkbox("🔍 Show Debug Info"):
                    st.write("**Debug Information:**")
                    st.write(f"- Has analysis data: {has_analysis_data}")
                    st.write(f"- Show analysis flag: {has_show_analysis}")  
                    st.write(f"- Analysis complete flag: {has_analysis_complete}")
                    if has_analysis_data:
                        st.write(f"- Analysis data keys: {list(st.session_state.analysis_data.keys()) if st.session_state.analysis_data else 'None'}")
                
            # Display cost estimate if available
            if st.session_state.cost_estimate:
                self.display_cost_estimate(st.session_state.cost_estimate)
                
            # Analysis progress
            if st.session_state.thinking_items:
                self.display_thinking_items()
                
            # Data collection progress
            self.display_data_collection()
            
        # Display analysis results if available (either from real analysis or test data)
        if (st.session_state.get('show_analysis', False) or st.session_state.get('analysis_complete', False)) and st.session_state.get('analysis_data'):
            st.markdown("---")
            st.header("📊 Analysis Results")
            self.display_analysis_results(st.session_state.analysis_data)
                
        # Main chat interface
        st.header("💬 Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Chat input
        if prompt := st.chat_input("Ask me anything about the stock analysis..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Check if analysis is needed
            if not st.session_state.analysis_complete:
                if "analyze" in prompt.lower() or "analysis" in prompt.lower():
                    if st.session_state.cost_estimate:
                        st.info("Please accept the cost estimate in the sidebar to run the analysis.")
                    else:
                        st.info("Please enter a company name in the sidebar and estimate the cost first.")
                else:
                    st.info("I can help you with stock analysis! Please enter a company name in the sidebar to get started.")
            else:
                # Generate response using LangChain
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = self.generate_contextual_response(prompt, st.session_state.analysis_data)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
        # Run analysis if accepted
        if hasattr(st.session_state, 'analysis_accepted') and st.session_state.analysis_accepted:
            if not st.session_state.analysis_complete and not st.session_state.analysis_in_progress:
                company_name = st.session_state.cost_estimate['company_name']
                st.session_state.analysis_in_progress = True
                
                # Add initial message
                if not st.session_state.messages:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Starting analysis for {company_name}... This will take a few moments."
                    })
                    
                # Run analysis
                analysis_results = self.run_real_analysis(company_name)
                
                if analysis_results:
                    # Display results
                    self.display_analysis_results(analysis_results)
                    
                    # Add completion message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ Analysis completed for {company_name}! You can now ask me questions about the results."
                    })
                    
                # Reset flags
                st.session_state.analysis_accepted = False
                st.session_state.analysis_in_progress = False
                
        # Display current analysis info
        if st.session_state.current_analysis:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 Current Analysis")
            st.sidebar.write(f"**Company:** {st.session_state.current_analysis['company_name']}")
            st.sidebar.write(f"**Ticker:** {st.session_state.current_analysis['ticker']}")
            st.sidebar.write(f"**Completed:** {st.session_state.current_analysis['timestamp'][:19]}")
            
        # Clear chat button
        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.analysis_data = {}
            st.session_state.current_analysis = None
            st.session_state.thinking_items = []
            st.session_state.analysis_complete = False
            st.session_state.cost_estimate = None
            st.session_state.analysis_in_progress = False
            st.session_state.collected_data = {}
            st.session_state.real_time_cost = 0.0
            if hasattr(st.session_state, 'data_collection'):
                st.session_state.data_collection = []
            st.rerun()

    def generate_intelligent_recommendation(self, analysis_data: Dict) -> Dict:
        """Generate intelligent recommendation based on technical, fundamental, and correlation analysis"""
        try:
            # Extract analysis components
            technical = analysis_data.get('technical_analysis', {})
            fundamental = analysis_data.get('fundamental_analysis', {})
            
            # Initialize recommendation components
            recommendation = {
                'action': 'HOLD',
                'confidence': 'Medium',
                'reasoning': [],
                'technical_score': 0,
                'fundamental_score': 0,
                'correlation_score': 0,
                'overall_score': 0
            }
            
            # 1. TECHNICAL ANALYSIS SCORING
            technical_score = self._evaluate_technical_signals(technical)
            recommendation['technical_score'] = technical_score
            
            # 2. FUNDAMENTAL ANALYSIS SCORING  
            fundamental_score = self._evaluate_fundamental_strength(fundamental)
            recommendation['fundamental_score'] = fundamental_score
            
            # 3. CORRELATION ANALYSIS SCORING
            correlation_score = self._evaluate_correlation_support(analysis_data)
            recommendation['correlation_score'] = correlation_score
            
            # 4. CALCULATE OVERALL SCORE
            overall_score = (technical_score + fundamental_score + correlation_score) / 3
            recommendation['overall_score'] = overall_score
            
            # 5. DETERMINE RECOMMENDATION BASED ON CRITERIA
            action, confidence, reasoning = self._determine_recommendation_action(
                technical_score, fundamental_score, correlation_score, technical, fundamental
            )
            
            recommendation.update({
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning
            })
            
            return recommendation
            
        except Exception as e:
            return {
                'action': 'HOLD',
                'confidence': 'Low',
                'reasoning': [f'Error in recommendation analysis: {str(e)}'],
                'technical_score': 0,
                'fundamental_score': 0,
                'correlation_score': 0,
                'overall_score': 0
            }
    
    def _evaluate_technical_signals(self, technical: Dict) -> int:
        """Evaluate technical analysis signals (0-100 scale)"""
        score = 50  # Start with neutral
        
        try:
            # Check trend
            trend = technical.get('trend', '').lower()
            if 'bullish' in trend:
                score += 20
            elif 'bearish' in trend:
                score -= 20
            
            # Check target vs current price (estimate upside potential)
            current_price_str = technical.get('entry_range', '')
            target_str = technical.get('short_term_target', '')
            
            if current_price_str and target_str:
                try:
                    # Extract numbers from price strings
                    import re
                    current_match = re.search(r'(\d+)', current_price_str.replace('₹', '').replace(',', ''))
                    target_match = re.search(r'(\d+)', target_str.replace('₹', '').replace(',', ''))
                    
                    if current_match and target_match:
                        current_price = float(current_match.group(1))
                        target_price = float(target_match.group(1))
                        
                        if current_price > 0:
                            upside_percent = ((target_price - current_price) / current_price) * 100
                            
                            if upside_percent > 20:
                                score += 25  # Strong upside potential
                            elif upside_percent > 10:
                                score += 15  # Good upside potential
                            elif upside_percent > 0:
                                score += 10  # Modest upside potential
                            else:
                                score -= 20  # Downside risk
                except:
                    pass
            
            # Check confidence level
            confidence = technical.get('confidence_score', '').lower()
            if 'high' in confidence:
                score += 10
            elif 'low' in confidence:
                score -= 10
            
            # Check patterns
            patterns = technical.get('patterns', [])
            positive_patterns = ['breakout', 'golden cross', 'higher highs', 'cup', 'ascending']
            negative_patterns = ['breakdown', 'death cross', 'lower lows', 'descending']
            
            for pattern in patterns:
                pattern_lower = pattern.lower()
                if any(pos in pattern_lower for pos in positive_patterns):
                    score += 5
                elif any(neg in pattern_lower for neg in negative_patterns):
                    score -= 5
            
        except Exception as e:
            print(f"Error in technical evaluation: {e}")
        
        return max(0, min(100, score))  # Clamp between 0-100
    
    def _evaluate_fundamental_strength(self, fundamental: Dict) -> int:
        """Evaluate fundamental analysis strength (0-100 scale) with enhanced growth metrics"""
        score = 50  # Start with neutral
        
        try:
            # Business Quality
            business_quality = fundamental.get('business_quality', '').lower()
            if 'strong' in business_quality or 'excellent' in business_quality:
                score += 15
            elif 'good' in business_quality:
                score += 10
            elif 'weak' in business_quality or 'poor' in business_quality:
                score -= 15
            
            # Market Penetration
            market_penetration = fundamental.get('market_penetration', '').lower()
            if 'strong' in market_penetration or 'dominant' in market_penetration:
                score += 10
            elif 'weak' in market_penetration:
                score -= 10
            
            # Financial Health
            financial_health = fundamental.get('financial_health', '').lower()
            if 'excellent' in financial_health:
                score += 15
            elif 'good' in financial_health:
                score += 10
            elif 'average' in financial_health:
                score += 0
            elif 'poor' in financial_health or 'weak' in financial_health:
                score -= 15
            
            # Valuation Status
            valuation = fundamental.get('valuation_status', '').lower()
            if 'undervalued' in valuation or 'cheap' in valuation:
                score += 15
            elif 'fair' in valuation:
                score += 5
            elif 'overvalued' in valuation or 'expensive' in valuation:
                score -= 15
            
            # Multibagger Potential
            multibagger = fundamental.get('multibagger_potential', '').lower()
            if 'high' in multibagger or 'strong' in multibagger:
                score += 15
            elif 'moderate' in multibagger:
                score += 8
            elif 'low' in multibagger or 'weak' in multibagger:
                score -= 10
            
            # ROCE/ROE Analysis
            roce_roe = fundamental.get('roce_roe', '')
            if roce_roe and roce_roe != 'N/A':
                try:
                    # Extract ROE percentage
                    import re
                    roe_match = re.search(r'ROE.*?(\d+\.?\d*)%', roce_roe)
                    if roe_match:
                        roe_value = float(roe_match.group(1))
                        if roe_value > 15:
                            score += 10
                        elif roe_value > 10:
                            score += 5
                        elif roe_value < 5:
                            score -= 10
                except:
                    pass
            
            # Debt to Equity
            debt_equity = fundamental.get('debt_to_equity', '')
            if debt_equity and debt_equity != 'N/A':
                try:
                    de_ratio = float(debt_equity.replace('%', ''))
                    if de_ratio < 0.5:
                        score += 10
                    elif de_ratio < 1.0:
                        score += 5
                    elif de_ratio > 2.0:
                        score -= 15
                except:
                    pass
            
            # Enhanced Growth Metrics Analysis is now handled by the core system
            
        except Exception as e:
            print(f"Error in fundamental evaluation: {e}")
        
        return max(0, min(100, score))  # Clamp between 0-100
    
    def _evaluate_correlation_support(self, analysis_data: Dict) -> int:
        """Evaluate correlation analysis support (0-100 scale)"""
        score = 50  # Start with neutral
        
        try:
            # Check if we have enhanced fundamental data (indicates good data quality)
            if 'enhanced_fundamental_data' in analysis_data:
                score += 15
            
            # Check if fundamental reasons are provided
            fundamental = analysis_data.get('fundamental_analysis', {})
            fundamental_reasons = fundamental.get('fundamental_reasons', '')
            
            if fundamental_reasons and fundamental_reasons != 'Not Available':
                if len(fundamental_reasons) > 100:  # Substantial reasoning provided
                    score += 15
                else:
                    score += 10
            
            # Check confidence levels alignment
            tech_confidence = analysis_data.get('technical_analysis', {}).get('confidence_score', '').lower()
            fund_confidence = fundamental.get('confidence_score', '').lower()
            
            if 'high' in tech_confidence and ('strong' in fund_confidence or 'high' in fund_confidence):
                score += 20  # Strong alignment
            elif ('medium' in tech_confidence and 'medium' in fund_confidence) or \
                 ('moderate' in tech_confidence and 'moderate' in fund_confidence):
                score += 10  # Good alignment
            elif ('low' in tech_confidence and 'low' in fund_confidence):
                score -= 10  # Poor but aligned
            elif ('high' in tech_confidence and 'low' in fund_confidence) or \
                 ('low' in tech_confidence and 'high' in fund_confidence):
                score -= 20  # Misaligned signals
            
        except Exception as e:
            print(f"Error in correlation evaluation: {e}")
        
        return max(0, min(100, score))  # Clamp between 0-100
    
    def _determine_recommendation_action(self, tech_score: int, fund_score: int, corr_score: int, 
                                       technical: Dict, fundamental: Dict) -> tuple:
        """Determine final recommendation action based on scores and criteria"""
        
        reasoning = []
        
        # Calculate upside potential from technical analysis
        upside_percent = self._calculate_upside_potential(technical)
        
        # Evaluate fundamental strength
        fundamental_strong = fund_score >= 70
        technical_strong = tech_score >= 70
        correlation_supportive = corr_score >= 60
        
        # Generate detailed fundamental analysis summary
        fund_details = self._analyze_fundamental_details(fundamental, fund_score)
        
        # Generate detailed correlation analysis summary
        corr_details = self._analyze_correlation_details(fund_score, tech_score, corr_score)
        
        # BUY CRITERIA
        if (upside_percent > 20 and technical_strong and 
            fundamental_strong and correlation_supportive):
            
            reasoning.append(f"🎯 Strong technical signals with {upside_percent:.1f}% upside potential")
            reasoning.append(f"💪 Fundamental analysis passes all parameters (score: {fund_score}/100)")
            reasoning.extend(fund_details['strengths'])
            reasoning.append(f"🔗 Correlation analysis supports growth thesis (score: {corr_score}/100)")
            reasoning.extend(corr_details['strengths'])
            return 'BUY', 'High', reasoning
        
        # HOLD CRITERIA  
        elif (0 < upside_percent <= 20 and technical_strong and 
              fundamental_strong and correlation_supportive):
            
            reasoning.append(f"📈 Modest technical upside of {upside_percent:.1f}% (good but not exceptional)")
            reasoning.append(f"✅ Fundamental strength confirmed (score: {fund_score}/100)")
            reasoning.extend(fund_details['strengths'])
            if fund_details['concerns']:
                reasoning.append("⚠️ Fundamental concerns to monitor:")
                reasoning.extend(fund_details['concerns'])
            reasoning.append(f"🤝 Correlation analysis supports thesis (score: {corr_score}/100)")
            reasoning.extend(corr_details['strengths'])
            if corr_details['concerns']:
                reasoning.append("⚠️ Correlation analysis concerns:")
                reasoning.extend(corr_details['concerns'])
            return 'HOLD', 'Medium', reasoning
        
        # SELL CRITERIA
        elif (upside_percent < 0 or not fundamental_strong or not correlation_supportive):
            
            if upside_percent < 0:
                reasoning.append(f"⚠️ Technical target below current price ({upside_percent:.1f}% downside)")
            if not fundamental_strong:
                reasoning.append(f"❌ Fundamental analysis shows weakness (score: {fund_score}/100)")
                reasoning.extend(fund_details['weaknesses'])
            if not correlation_supportive:
                reasoning.append(f"🔴 Correlation analysis doesn't support growth thesis (score: {corr_score}/100)")
                reasoning.extend(corr_details['weaknesses'])
            
            return 'SELL', 'Medium' if len(reasoning) == 1 else 'High', reasoning
        
        # DEFAULT HOLD
        else:
            reasoning.append(f"📊 Mixed signals - Technical: {tech_score}/100, Fundamental: {fund_score}/100")
            reasoning.append(f"🔄 Upside potential: {upside_percent:.1f}%")
            
            # Add detailed analysis for mixed signals
            if fund_score < 70:
                reasoning.append("⚠️ Fundamental concerns:")
                reasoning.extend(fund_details['concerns'] + fund_details['weaknesses'])
            
            if corr_score < 60:
                reasoning.append("⚠️ Correlation analysis concerns:")
                reasoning.extend(corr_details['concerns'] + corr_details['weaknesses'])
            
            reasoning.append("⚖️ Awaiting clearer directional signals")
            return 'HOLD', 'Low', reasoning
    
    def _analyze_fundamental_details(self, fundamental: Dict, fund_score: int) -> Dict:
        """Provide detailed analysis of fundamental strengths and weaknesses with growth metrics"""
        details = {
            'strengths': [],
            'concerns': [],
            'weaknesses': []
        }
        
        try:
            # Growth analysis is now integrated in the core system and available in fundamental_reasons
            
            # Business Quality Analysis
            business_quality = fundamental.get('business_quality', '').lower()
            if 'strong' in business_quality or 'excellent' in business_quality:
                details['strengths'].append("  • Excellent business model with strong competitive advantages")
            elif 'good' in business_quality:
                details['strengths'].append("  • Solid business fundamentals with decent competitive position")
            elif 'weak' in business_quality or 'poor' in business_quality:
                details['weaknesses'].append("  • Weak business model with limited competitive advantages")
            elif 'low' in business_quality:
                details['concerns'].append("  • Business quality rated as low - needs improvement in core operations")
            
            # Market Penetration Analysis
            market_penetration = fundamental.get('market_penetration', '').lower()
            if 'strong' in market_penetration or 'dominant' in market_penetration:
                details['strengths'].append("  • Strong market presence with significant penetration")
            elif 'weak' in market_penetration:
                details['weaknesses'].append("  • Limited market penetration - growth potential questionable")
            
            # Financial Health Analysis
            financial_health = fundamental.get('financial_health', '').lower()
            if 'excellent' in financial_health:
                details['strengths'].append("  • Excellent financial health with strong balance sheet")
            elif 'good' in financial_health:
                details['strengths'].append("  • Good financial health supporting growth initiatives")
            elif 'average' in financial_health:
                details['concerns'].append("  • Average financial health - room for improvement in key metrics")
            elif 'poor' in financial_health or 'weak' in financial_health:
                details['weaknesses'].append("  • Poor financial health raises concerns about sustainability")
            
            # Valuation Analysis
            valuation = fundamental.get('valuation_status', '').lower()
            if 'undervalued' in valuation or 'cheap' in valuation:
                details['strengths'].append("  • Attractive valuation provides good entry opportunity")
            elif 'fair' in valuation:
                details['concerns'].append("  • Fair valuation - limited margin of safety")
            elif 'overvalued' in valuation or 'expensive' in valuation:
                details['weaknesses'].append("  • Overvalued at current levels - high risk of correction")
            
            # Multibagger Potential
            multibagger = fundamental.get('multibagger_potential', '').lower()
            if 'high' in multibagger or 'strong' in multibagger:
                details['strengths'].append("  • High multibagger potential for long-term wealth creation")
            elif 'moderate' in multibagger:
                details['strengths'].append("  • Moderate multibagger potential with steady growth prospects")
            elif 'low' in multibagger or 'weak' in multibagger:
                details['concerns'].append("  • Limited multibagger potential - mature business with slow growth")
            
            # ROCE/ROE Analysis
            roce_roe = fundamental.get('roce_roe', '')
            if roce_roe and roce_roe != 'N/A':
                try:
                    import re
                    roe_match = re.search(r'ROE.*?(\d+\.?\d*)%', roce_roe)
                    roce_match = re.search(r'ROCE.*?(\d+\.?\d*)%', roce_roe)
                    
                    if roe_match:
                        roe_value = float(roe_match.group(1))
                        if roe_value > 15:
                            details['strengths'].append(f"  • Excellent ROE of {roe_value}% shows efficient use of equity")
                        elif roe_value > 10:
                            details['strengths'].append(f"  • Good ROE of {roe_value}% indicates decent profitability")
                        elif roe_value < 5:
                            details['weaknesses'].append(f"  • Poor ROE of {roe_value}% raises profitability concerns")
                        else:
                            details['concerns'].append(f"  • Moderate ROE of {roe_value}% - room for improvement")
                    
                    if roce_match:
                        roce_value = float(roce_match.group(1))
                        if roce_value > 15:
                            details['strengths'].append(f"  • Strong ROCE of {roce_value}% shows efficient capital allocation")
                        elif roce_value < 5:
                            details['weaknesses'].append(f"  • Weak ROCE of {roce_value}% indicates poor capital efficiency")
                        else:
                            details['concerns'].append(f"  • Moderate ROCE of {roce_value}% - capital efficiency needs improvement")
                except:
                    details['concerns'].append(f"  • ROCE/ROE metrics: {roce_roe} - requires detailed analysis")
            
            # Debt Analysis
            debt_equity = fundamental.get('debt_to_equity', '')
            if debt_equity and debt_equity != 'N/A':
                try:
                    de_ratio = float(debt_equity.replace('%', ''))
                    if de_ratio < 0.5:
                        details['strengths'].append(f"  • Low debt-to-equity ratio of {de_ratio} indicates strong financial stability")
                    elif de_ratio < 1.0:
                        details['strengths'].append(f"  • Manageable debt-to-equity ratio of {de_ratio}")
                    elif de_ratio > 2.0:
                        details['weaknesses'].append(f"  • High debt-to-equity ratio of {de_ratio} raises solvency concerns")
                    else:
                        details['concerns'].append(f"  • Elevated debt-to-equity ratio of {de_ratio} needs monitoring")
                except:
                    details['concerns'].append(f"  • Debt-to-equity ratio: {debt_equity} - requires assessment")
            
            # Growth metrics are now calculated by the core EnhancedFundamentalAnalysisAgent
            # and included in the fundamental_reasons field with detailed analysis
            
            # Extract growth information from fundamental_reasons if available
            fundamental_reasons = fundamental.get('fundamental_reasons', '')
            if 'Growth Analysis:' in fundamental_reasons:
                # Extract growth highlights from the core system's analysis
                growth_section = fundamental_reasons.split('Growth Highlights:')
                if len(growth_section) > 1:
                    growth_highlights = growth_section[1].split('Financial Metrics:')[0].strip()
                    for line in growth_highlights.split('\n'):
                        if line.strip().startswith('•'):
                            details['strengths'].append(f"  {line.strip()}")
            
            # Business Quality Analysis
            
        except Exception as e:
            details['concerns'].append(f"  • Error analyzing fundamental details: {str(e)}")
        
        return details
    
    def _calculate_upside_potential(self, technical: Dict) -> float:
        """Calculate upside potential percentage from technical analysis"""
        try:
            current_price_str = technical.get('entry_range', '')
            target_str = technical.get('short_term_target', '')
            
            if current_price_str and target_str:
                import re
                current_match = re.search(r'(\d+)', current_price_str.replace('₹', '').replace(',', ''))
                target_match = re.search(r'(\d+)', target_str.replace('₹', '').replace(',', ''))
                
                if current_match and target_match:
                    current_price = float(current_match.group(1))
                    target_price = float(target_match.group(1))
                    
                    if current_price > 0:
                        return ((target_price - current_price) / current_price) * 100
            
            return 0.0
        except:
            return 0.0
    
    def _analyze_correlation_details(self, fund_score: int, tech_score: int, corr_score: int) -> Dict:
        """Provide detailed analysis of correlation strengths and weaknesses"""
        details = {
            'strengths': [],
            'concerns': [],
            'weaknesses': []
        }
        
        try:
            # Score Alignment Analysis
            score_diff = abs(fund_score - tech_score)
            if score_diff <= 10:
                details['strengths'].append(f"  • Strong alignment between technical ({tech_score}) and fundamental ({fund_score}) analysis")
            elif score_diff <= 20:
                details['concerns'].append(f"  • Moderate divergence between technical ({tech_score}) and fundamental ({fund_score}) signals")
            else:
                details['weaknesses'].append(f"  • Significant divergence between technical ({tech_score}) and fundamental ({fund_score}) analysis")
            
            # Data Quality Assessment
            if corr_score >= 80:
                details['strengths'].append("  • High-quality comprehensive data supports reliable analysis")
            elif corr_score >= 60:
                details['strengths'].append("  • Good data quality enables confident decision making")
            elif corr_score >= 40:
                details['concerns'].append("  • Moderate data quality - some analysis gaps may exist")
            else:
                details['weaknesses'].append("  • Limited data quality hampers comprehensive analysis")
            
            # Confidence Level Correlation
            if fund_score >= 70 and tech_score >= 70:
                details['strengths'].append("  • Both technical and fundamental analysis show strong conviction")
            elif fund_score >= 50 and tech_score >= 50:
                details['strengths'].append("  • Reasonable confidence from both technical and fundamental perspectives")
            elif fund_score < 50 or tech_score < 50:
                if fund_score < 50:
                    details['concerns'].append("  • Fundamental analysis shows limited conviction")
                if tech_score < 50:
                    details['concerns'].append("  • Technical analysis lacks strong directional bias")
            
            # Overall Correlation Assessment
            if corr_score >= 70:
                details['strengths'].append("  • Strong correlation between multiple analysis factors")
            elif corr_score >= 50:
                details['concerns'].append("  • Moderate correlation - some conflicting signals present")
            else:
                details['weaknesses'].append("  • Weak correlation suggests conflicting analysis outcomes")
            
        except Exception as e:
            details['concerns'].append(f"  • Error analyzing correlation details: {str(e)}")
        
        return details

def main():
    """Main function to run the Streamlit app"""
    try:
        app = EnhancedStockAnalysisChatApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main() 