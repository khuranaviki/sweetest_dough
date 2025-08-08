#!/usr/bin/env python3
"""
Delhivery Comprehensive Stock Analysis
Runs complete analysis with cost tracking and generates detailed reports
"""

import os
import sys
from datetime import datetime
from EnhancedMultiAgent import EnhancedMultiAgentStockAnalysis
from openai_cost_tracker import cost_tracker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def analyze_delhivery():
    """Run comprehensive analysis for Delhivery"""
    
    # Initialize the enhanced multi-agent system
    print("🚀 Initializing Enhanced Multi-Agent Stock Analysis System...")
    
    try:
        analyzer = EnhancedMultiAgentStockAnalysis(openai_api_key=os.getenv('OPENAI_API_KEY'))
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    # Delhivery stock details
    delhivery_stock = {
        'ticker': 'DELHIVERY',
        'company_name': 'DELHIVERY LTD',
        'sector': 'Logistics',
        'category': 'V40 Next'
    }
    
    print(f"\n📊 Starting Comprehensive Analysis for {delhivery_stock['ticker']}")
    print(f"🏢 Company: {delhivery_stock['company_name']}")
    print(f"📈 Sector: {delhivery_stock['sector']}")
    print(f"🏷️ Category: {delhivery_stock['category']}")
    
    # Get initial cost estimation
    print(f"\n💰 Cost Estimation for Analysis:")
    estimation = cost_tracker.estimate_cost_for_analysis(1)
    print(f"Estimated cost for 1 stock: ${estimation['estimated_cost_usd']:.6f} (₹{estimation['estimated_cost_inr']:.2f})")
    print(f"Estimated tokens: {estimation['total_tokens']:,}")
    
    # Run comprehensive analysis
    try:
        results = analyzer.analyze_stock(
            ticker=delhivery_stock['ticker'],
            company_name=delhivery_stock['company_name'],
            sector=delhivery_stock['sector'],
            category=delhivery_stock['category']
        )
        
        if "error" in results:
            print(f"❌ Analysis failed: {results['error']}")
            return
        
        print(f"\n🎉 Comprehensive Analysis Completed Successfully!")
        
        # Generate comprehensive report
        generate_comprehensive_report(results, delhivery_stock)
        
        # Generate cost analysis report
        generate_cost_analysis_report(delhivery_stock)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def generate_comprehensive_report(results: dict, stock_info: dict):
    """Generate comprehensive analysis report"""
    
    try:
        # Create reports directory if it doesn't exist
        reports_dir = "delhivery_analysis_reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stock_info['ticker']}_comprehensive_analysis_{timestamp}.txt"
        filepath = os.path.join(reports_dir, filename)
        
        # Get the comprehensive report
        comprehensive_report = results.get('comprehensive_report', 'Report not available')
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(comprehensive_report)
        
        print(f"\n📄 Comprehensive Report Generated Successfully!")
        print(f"📁 File: {filepath}")
        print(f"📊 File Size: {os.path.getsize(filepath) / 1024:.2f} KB")
        
        # Print key summary to console
        print_summary_to_console(results, stock_info)
        
    except Exception as e:
        print(f"❌ Error generating comprehensive report: {e}")
        import traceback
        traceback.print_exc()

def generate_cost_analysis_report(stock_info: dict):
    """Generate detailed cost analysis report"""
    
    try:
        # Create reports directory if it doesn't exist
        reports_dir = "delhivery_analysis_reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stock_info['ticker']}_cost_analysis_{timestamp}.txt"
        filepath = os.path.join(reports_dir, filename)
        
        # Get today's usage
        daily_usage = cost_tracker.get_daily_usage()
        
        # Get weekly summary
        weekly_summary = cost_tracker.get_usage_summary(7)
        
        # Create cost report
        cost_report = f"""
{'='*80}
💰 DELHIVERY ANALYSIS COST REPORT
{'='*80}

📊 ANALYSIS DETAILS:
Ticker: {stock_info['ticker']}
Company: {stock_info['company_name']}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
📈 TODAY'S API USAGE SUMMARY
{'='*50}
Total API Calls: {daily_usage.total_calls:,}
Total Tokens: {daily_usage.total_tokens:,}
  • Input Tokens: {daily_usage.total_prompt_tokens:,}
  • Output Tokens: {daily_usage.total_completion_tokens:,}
Total Cost: ${daily_usage.total_cost_usd:.6f} (₹{daily_usage.total_cost_inr:.2f})

BREAKDOWN BY MODEL:
"""
        
        for model, calls in daily_usage.calls_by_model.items():
            cost = daily_usage.costs_by_model.get(model, 0)
            cost_report += f"  • {model}: {calls} calls, ${cost:.6f}\n"
        
        cost_report += f"""
{'='*80}
📊 WEEKLY USAGE SUMMARY (Last 7 Days)
{'='*50}
Total Calls: {weekly_summary['total_calls']:,}
Total Tokens: {weekly_summary['total_tokens']:,}
Total Cost: ${weekly_summary['total_cost_usd']:.6f} (₹{weekly_summary['total_cost_inr']:.2f})

BREAKDOWN BY MODEL:
"""
        
        for model, data in weekly_summary['model_breakdown'].items():
            cost_report += f"  • {model}:\n"
            cost_report += f"    - Calls: {data['calls']:,}\n"
            cost_report += f"    - Tokens: {data['tokens']:,}\n"
            cost_report += f"    - Cost: ${data['cost_usd']:.6f} (₹{data['cost_inr']:.2f})\n"
        
        cost_report += f"""
{'='*80}
📅 DAILY BREAKDOWN (Last 5 Days)
{'='*50}
"""
        
        for day_data in weekly_summary['daily_breakdown'][-5:]:
            if day_data['total_calls'] > 0:
                cost_report += f"  {day_data['date']}: {day_data['total_calls']} calls, "
                cost_report += f"{day_data['total_tokens']:,} tokens, "
                cost_report += f"${day_data['total_cost_usd']:.6f} (₹{day_data['total_cost_inr']:.2f})\n"
        
        cost_report += f"""
{'='*80}
💰 COST ESTIMATIONS FOR DIFFERENT SCENARIOS
{'='*50}
"""
        
        for num_stocks in [1, 5, 10, 20, 50, 100]:
            estimation = cost_tracker.estimate_cost_for_analysis(num_stocks)
            cost_report += f"  • {num_stocks} stocks: ${estimation['estimated_cost_usd']:.6f} (₹{estimation['estimated_cost_inr']:.2f})\n"
            cost_report += f"    - Total tokens: {estimation['total_tokens']:,}\n"
            cost_report += f"    - Tokens per stock: {estimation['estimated_tokens_per_stock']:,}\n"
        
        cost_report += f"""
{'='*80}
📋 COST OPTIMIZATION RECOMMENDATIONS
{'='*50}
1. Use GPT-4o-mini for basic analysis to reduce costs
2. Batch process multiple stocks to optimize API calls
3. Cache fundamental data to avoid repeated extractions
4. Use local analysis for simple technical indicators
5. Monitor usage regularly to stay within budget

{'='*80}
"""
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cost_report)
        
        print(f"\n💰 Cost Analysis Report Generated Successfully!")
        print(f"📁 File: {filepath}")
        print(f"📊 File Size: {os.path.getsize(filepath) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ Error generating cost analysis report: {e}")
        import traceback
        traceback.print_exc()

def print_summary_to_console(results: dict, stock_info: dict):
    """Print key analysis summary to console"""
    
    print(f"\n{'='*80}")
    print(f"📋 DELHIVERY ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    # Extract key information
    technical = results.get('technical_analysis')
    fundamental = results.get('fundamental_analysis')
    recommendation = results.get('final_recommendation')
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

    print(f"📈 Technical Analysis:")
    print(f"   • Trend: {safe_get(technical, 'trend', 'N/A')}")
    print(f"   • Confidence: {safe_get(technical, 'confidence_score', 'N/A')}")
    print(f"   • Entry Range: {safe_get(technical, 'entry_range', 'N/A')}")
    print(f"   • Target: {safe_get(technical, 'medium_term_target', 'N/A')}")
    
    print(f"\n💰 Fundamental Analysis:")
    print(f"   • Business Quality: {safe_get(fundamental, 'business_quality', 'N/A')}")
    print(f"   • Revenue Growth: {safe_get(fundamental, 'revenue_growth', 'N/A')}")
    print(f"   • ROCE/ROE: {safe_get(fundamental, 'roce_roe', 'N/A')}")
    print(f"   • Confidence: {safe_get(fundamental, 'confidence_score', 'N/A')}")
    
    print(f"\n🎯 Final Recommendation:")
    print(f"   • Action: {safe_get(recommendation, 'action', 'N/A')}")
    print(f"   • Confidence: {safe_get(recommendation, 'confidence_level', 'N/A')}")
    print(f"   • Strategy: {safe_get(recommendation, 'strategy_used', 'N/A')}")
    print(f"   • Position Size: {safe_get(recommendation, 'position_size', 'N/A')}")
    
    print(f"\n🧠 Correlated Insights:")
    if correlated_insights and "analysis" in correlated_insights:
        analysis = correlated_insights['analysis']
        
        # Extract key sections
        sections = {
            "CONFIDENCE ON GROWTH": "Growth Confidence",
            "VALUATION ASSESSMENT": "Valuation",
            "CURRENT GROWTH": "Current Metrics",
            "FUTURE GROWTH": "Future Drivers",
            "FUNDAMENTAL METRICS CORRELATION": "Metrics Correlation",
            "PROJECTED GROWTH": "Projected Growth"
        }
        
        for section_key, section_name in sections.items():
            if section_key in analysis:
                section_content = analysis.split(section_key)[1]
                if "FUTURE GROWTH" in section_content:
                    section_content = section_content.split("FUTURE GROWTH")[0]
                elif "FUNDAMENTAL METRICS" in section_content:
                    section_content = section_content.split("FUNDAMENTAL METRICS")[0]
                elif "PROJECTED GROWTH" in section_content:
                    section_content = section_content.split("PROJECTED GROWTH")[0]
                elif "KEY INSIGHTS" in section_content:
                    section_content = section_content.split("KEY INSIGHTS")[0]
                
                print(f"   • {section_name}: {section_content.strip()[:100]}...")
    else:
        print(f"   • Not Available")
    
    print(f"\n📄 Reports saved in: delhivery_analysis_reports/")
    print(f"{'='*80}")

def main():
    """Main function"""
    print("🚚 DELHIVERY COMPREHENSIVE STOCK ANALYSIS")
    print("=" * 60)
    print("This script will run complete analysis including:")
    print("✅ Technical Analysis with pattern recognition")
    print("✅ Fundamental Analysis with Screener.in data")
    print("✅ ArthaLens Transcript & Guidance Analysis")
    print("✅ Correlated Insights and Projected Growth")
    print("✅ Cost Tracking and Optimization")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Run the analysis
    analyze_delhivery()
    
    print(f"\n🎉 Delhivery analysis completed!")
    print(f"📁 Check the 'delhivery_analysis_reports' directory for detailed reports.")

if __name__ == "__main__":
    main() 