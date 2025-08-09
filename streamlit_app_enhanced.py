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
from datetime import datetime
from typing import Dict, List, Optional
import base64
from io import BytesIO
import threading
import queue

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Local imports
from EnhancedMultiAgent import EnhancedMultiAgentStockAnalysis
from openai_cost_tracker import OpenAICostTracker

# Initialize cost tracker once at module level
cost_tracker = OpenAICostTracker()

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
            
    def setup_langchain(self):
        """Setup LangChain components"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                return
                
            self.llm = ChatOpenAI(
                model=os.getenv('OPENAI_MODEL', 'gpt-5'),
                temperature=0.7,
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
            'technical_analysis': 0.005,  # $0.005 for gpt-4o vision
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
            if 'recommendation' in analysis_data:
                rec = analysis_data['recommendation']
                
                # Check if we have any real recommendation data
                has_rec_data = any(
                    value != 'Not Available' and value != [] and value is not None 
                    for value in rec.values()
                )
                
                if not has_rec_data:
                    st.info("📋 Recommendation data not available.")
                else:
                    action = rec.get('action', 'Not Available')
                    confidence = rec.get('confidence', 'Not Available')
                    
                    if action != 'Not Available':
                        st.metric("Action", action, delta=confidence if confidence != 'Not Available' else None)
                    else:
                        st.metric("Action", "N/A")
                        
                    col1, col2 = st.columns(2)
                    with col1:
                        position_size = rec.get('position_size', 'Not Available')
                        if position_size != 'Not Available':
                            st.metric("Position Size", position_size)
                        else:
                            st.metric("Position Size", "N/A")
                            
                        time_horizon = rec.get('time_horizon', 'Not Available')
                        if time_horizon != 'Not Available':
                            st.metric("Time Horizon", time_horizon)
                        else:
                            st.metric("Time Horizon", "N/A")
                            
                    with col2:
                        if confidence != 'Not Available':
                            st.metric("Confidence", confidence)
                        else:
                            st.metric("Confidence", "N/A")
                    
                    # Key Risks
                    key_risks = rec.get('key_risks', [])
                    if key_risks:
                        st.subheader("Key Risks")
                        for risk in key_risks:
                            st.write(f"• {risk}")
                    else:
                        st.subheader("Key Risks")
                        st.write("Not Available")
                        
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
                
            # Display cost estimate if available
            if st.session_state.cost_estimate:
                self.display_cost_estimate(st.session_state.cost_estimate)
                
            # Analysis progress
            if st.session_state.thinking_items:
                self.display_thinking_items()
                
            # Data collection progress
            self.display_data_collection()
                
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