#!/usr/bin/env python3
"""
Test script for 3-Year Enhanced Stock Analysis System
Demonstrates the forked version with 3-year chart data and prioritized chart-based OpenAI analysis
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from EnhancedMultiAgent_3Y import EnhancedMultiAgentStockAnalysis3Y

# Load environment variables
load_dotenv()

def test_3y_analysis():
    """Test the 3-year analysis system"""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    print("🚀 Testing 3-Year Enhanced Stock Analysis System")
    print("="*60)
    
    # Initialize the 3-year analyzer
    analyzer = EnhancedMultiAgentStockAnalysis3Y(api_key)
    
    # Test stocks
    test_stocks = [
        {
            "ticker": "DELHIVERY.NS",
            "company_name": "Delhivery Limited",
            "sector": "Logistics",
            "category": "3-Year Analysis Test"
        },
        {
            "ticker": "RELIANCE.NS",
            "company_name": "Reliance Industries Limited",
            "sector": "Oil & Gas",
            "category": "3-Year Analysis Test"
        }
    ]
    
    results = []
    
    for stock in test_stocks:
        print(f"\n📊 Analyzing {stock['ticker']} ({stock['company_name']})")
        print("-" * 50)
        
        try:
            # Perform 3-year analysis
            result = analyzer.analyze_stock(
                ticker=stock['ticker'],
                company_name=stock['company_name'],
                sector=stock['sector'],
                category=stock['category']
            )
            
            results.append(result)
            
            # Display key results
            if 'technical_analysis' in result:
                tech = result['technical_analysis']
                print(f"✅ Analysis completed successfully!")
                print(f"📈 Trend: {tech.get('trend', 'N/A')}")
                print(f"🎯 Confidence: {tech.get('confidence_score', 'N/A')}")
                print(f"📊 Data Points: {result.get('data_points', 'N/A')}")
                print(f"⏰ Timeframe: {result.get('timeframe', 'N/A')}")
                print(f"🔍 Analysis Type: {result.get('analysis_type', 'N/A')}")
                
                # Show indicators
                indicators = tech.get('indicators', {})
                if indicators:
                    print(f"📊 Key Indicators:")
                    for key, value in indicators.items():
                        print(f"   • {key}: {value}")
                
                # Show patterns
                patterns = tech.get('patterns', [])
                if patterns:
                    print(f"📈 Patterns Detected: {', '.join(patterns)}")
                
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error analyzing {stock['ticker']}: {e}")
            results.append({
                'ticker': stock['ticker'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"3y_analysis_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Summary
    print("\n" + "="*60)
    print("📋 3-YEAR ANALYSIS SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if 'technical_analysis' in r)
    failed = len(results) - successful
    
    print(f"✅ Successful analyses: {successful}")
    print(f"❌ Failed analyses: {failed}")
    print(f"📊 Total stocks tested: {len(results)}")
    
    if successful > 0:
        print(f"\n🎯 Key Improvements in 3-Year Analysis:")
        print(f"   • Uses 3 years of historical data (vs 6 months)")
        print(f"   • Prioritizes chart-based OpenAI analysis")
        print(f"   • Enhanced technical indicators (MA 20, 50, 100, 200)")
        print(f"   • Comprehensive RSI and volume analysis")
        print(f"   • Better trend detection for longer timeframes")
        print(f"   • More accurate support/resistance levels")

def compare_with_original():
    """Compare 3-year analysis with original 6-month analysis"""
    print("\n🔄 Comparison with Original 6-Month Analysis")
    print("="*60)
    
    print("📊 Original System (6 months):")
    print("   • Timeframe: 6 months")
    print("   • Data points: ~120-150 days")
    print("   • Moving averages: 20, 50, 200-day")
    print("   • Chart priority: Secondary to OHLCV analysis")
    print("   • Pattern detection: Short-term patterns")
    
    print("\n🚀 Enhanced 3-Year System:")
    print("   • Timeframe: 3 years")
    print("   • Data points: ~750-800 days")
    print("   • Moving averages: 20, 50, 100, 200-day")
    print("   • Chart priority: PRIMARY - OpenAI chart analysis first")
    print("   • Pattern detection: Long-term patterns (RHS, CWH, etc.)")
    print("   • Enhanced indicators: RSI, Bollinger Bands, Volume MA")
    
    print("\n✅ Benefits of 3-Year Analysis:")
    print("   • Better trend identification")
    print("   • More reliable support/resistance levels")
    print("   • Improved pattern recognition")
    print("   • Higher confidence in technical signals")
    print("   • Better suited for medium-term investments")

if __name__ == "__main__":
    print("🎯 3-Year Enhanced Stock Analysis System Test")
    print("="*60)
    
    # Run the test
    test_3y_analysis()
    
    # Show comparison
    compare_with_original()
    
    print("\n🎉 Test completed!")
    print("Check the generated JSON file for detailed results.") 