#!/usr/bin/env python3
"""
Comprehensive ArthaLens Extraction System
Extracts transcript and guidance data with dynamic content handling and fundamental correlation
"""

import os
import time
import json
import re
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ArthaLensExtractor:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize Chrome options for headless browsing
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Create directories for screenshots and results
        self.screenshots_dir = "arthalens_screenshots"
        self.results_dir = "arthalens_results"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def extract_arthalens_data(self, ticker: str, fundamental_data: Dict = None) -> Dict:
        """
        Extract comprehensive ArthaLens data with fundamental correlation
        
        Args:
            ticker: Stock ticker symbol
            fundamental_data: Optional fundamental analysis data for correlation
        
        Returns:
            Dictionary containing extracted data and correlated insights
        """
        print(f"🔍 Extracting ArthaLens data for {ticker}...")
        
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            
            # Extract data from different tabs
            extraction_results = {
                "ticker": ticker,
                "extraction_timestamp": datetime.now().isoformat(),
                "summary_data": {},
                "guidance_data": {},
                "qa_data": {},
                "correlated_insights": {},
                "confidence_analysis": {}
            }
            
            # Test all quarters to get comprehensive data
            quarters = ["Q4+FY25", "Q3+FY25", "Q2+FY25", "Q1+FY25"]
            
            for quarter in quarters:
                print(f"📊 Processing quarter: {quarter}")
                
                # Extract Summary tab data for this quarter
                summary_data = self._extract_tab_data(driver, ticker, quarter, "summary")
                if summary_data:
                    extraction_results["summary_data"][quarter] = summary_data
                
                # Extract Guidance tab data for this quarter
                guidance_data = self._extract_tab_data(driver, ticker, quarter, "guidance")
                if guidance_data:
                    extraction_results["guidance_data"][quarter] = guidance_data
                
                # Extract Q&A tab data for this quarter
                qa_data = self._extract_tab_data(driver, ticker, quarter, "qa")
                if qa_data:
                    extraction_results["qa_data"][quarter] = qa_data
                
                # Add delay between quarters to be respectful
                if quarter != quarters[-1]:  # Not the last quarter
                    print(f"⏳ Waiting 5 seconds before next quarter...")
                    time.sleep(5)
            
            driver.quit()
            
            # Generate correlated insights if fundamental data is provided
            if fundamental_data:
                extraction_results["correlated_insights"] = self._generate_correlated_insights(
                    ticker, extraction_results, fundamental_data
                )
                extraction_results["confidence_analysis"] = self._analyze_future_confidence(
                    ticker, extraction_results, fundamental_data
                )
            
            # Save results
            result_file = os.path.join(self.results_dir, f"{ticker}_arthalens_extraction.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ ArthaLens extraction completed for {ticker}")
            print(f"📁 Results saved: {result_file}")
            
            return extraction_results
            
        except Exception as e:
            print(f"❌ Error in ArthaLens extraction for {ticker}: {e}")
            if 'driver' in locals():
                driver.quit()
            return {"error": str(e)}
    
    def _extract_tab_data(self, driver: webdriver.Chrome, ticker: str, quarter: str, tab: str) -> Optional[Dict]:
        """Extract data from a specific tab (summary, guidance, qa) with complete page screenshot"""
        try:
            # Clean ticker for URL (remove .NS suffix for ArthaLens compatibility)
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            
            # Construct URL
            if tab == "summary":
                url = f"https://arthalens.com/{clean_ticker}/concall?quarter={quarter}"
            else:
                url = f"https://arthalens.com/{clean_ticker}/concall?quarter={quarter}&tab={tab}"
            
            print(f"🌐 Loading URL: {url}")
            driver.get(url)
            
            # Wait for page to load and content to appear
            wait = WebDriverWait(driver, 20)
            
            # Wait for loading to complete
            try:
                wait.until_not(EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Loading transcript data')]")))
            except TimeoutException:
                print(f"⚠️ Loading timeout for {tab} tab")
            
            # Additional wait for dynamic content
            time.sleep(8)
            
            # Capture complete page screenshot with scrolling
            screenshot_path = self._capture_complete_page_screenshot(driver, ticker, quarter, tab)
            
            if not screenshot_path:
                print(f"❌ Failed to capture screenshot for {ticker} {quarter} {tab}")
                return None
            
            # Extract text using OpenAI Vision
            extracted_text = self._extract_text_from_screenshot(screenshot_path, ticker, tab, quarter)
            
            if extracted_text:
                return {
                    "quarter": quarter,
                    "tab": tab,
                    "screenshot_path": screenshot_path,
                    "extracted_text": extracted_text,
                    "extraction_timestamp": datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error extracting {tab} tab data: {e}")
            return None
    
    def _capture_complete_page_screenshot(self, driver: webdriver.Chrome, ticker: str, quarter: str, tab: str) -> Optional[str]:
        """Capture complete webpage screenshot including all scrollable sections with enhanced loading"""
        try:
            print(f"📸 Capturing complete page screenshot for {ticker} {quarter} {tab}...")
            
            # Clean ticker for URL
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            
            # Construct URL
            if tab == "concall":
                url = f"https://arthalens.com/{clean_ticker}/concall?quarter={quarter}"
            else:
                url = f"https://arthalens.com/{clean_ticker}/concall?quarter={quarter}&tab={tab}"
            
            print(f"🌐 Loading URL: {url}")
            
            # Navigate to the page
            driver.get(url)
            
            # Wait for page to load with longer timeout
            import time
            time.sleep(10)  # Increased wait time
            
            # Check if page loaded correctly
            page_title = driver.title
            print(f"📄 Page title: {page_title}")
            
            # Check for error pages
            if "404" in page_title or "Not Found" in page_title or "Error" in page_title:
                print(f"❌ Page not found or error: {page_title}")
                return None
            
            # Wait for content to load
            try:
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                from selenium.webdriver.common.by import By
                
                wait = WebDriverWait(driver, 20)
                
                # Wait for any content to appear
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                
                # Check if page has content
                page_source = driver.page_source
                if len(page_source) < 1000:
                    print(f"❌ Page appears to be empty or not loaded properly")
                    return None
                
                print(f"✅ Page loaded successfully, content length: {len(page_source)} characters")
                
            except Exception as e:
                print(f"⚠️ Wait timeout, but continuing: {e}")
            
            # Get the total height of the page
            total_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
            print(f"📏 Total page height: {total_height}px")
            
            if total_height == 0:
                print(f"❌ Page height is 0 - page may not have loaded properly")
                return None
            
            # Enhanced scrolling to ensure all content is loaded
            print(f"🔄 Scrolling through page to load all content...")
            
            # Scroll down gradually to trigger lazy loading
            current_height = 0
            scroll_step = 500  # Smaller scroll steps
            
            while current_height < total_height:
                driver.execute_script(f"window.scrollTo(0, {current_height});")
                time.sleep(2)  # Wait longer for content to load
                current_height += scroll_step
                
                # Check if height increased (dynamic content loading)
                new_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
                if new_height > total_height:
                    total_height = new_height
                    print(f"📏 Height increased to: {total_height}px")
            
            # Scroll back to top
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(3)
            
            # Get the final page dimensions
            final_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
            final_width = driver.execute_script("return Math.max(document.body.scrollWidth, document.documentElement.scrollWidth);")
            
            print(f"📐 Final page dimensions: {final_width}x{final_height}px")
            
            # Set window size to capture full page
            driver.set_window_size(final_width, final_height)
            time.sleep(3)
            
            # Take the screenshot
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            screenshot_path = os.path.join(self.screenshots_dir, f"{clean_ticker}_{quarter}_{tab}_complete.png")
            driver.save_screenshot(screenshot_path)
            
            print(f"✅ Complete page screenshot saved: {screenshot_path}")
            print(f"📁 File size: {os.path.getsize(screenshot_path) / (1024*1024):.2f} MB")
            
            # Verify screenshot is not blank
            if os.path.getsize(screenshot_path) < 10000:  # Less than 10KB
                print(f"❌ Screenshot appears to be blank or too small")
                return None
            
            return screenshot_path
            
        except Exception as e:
            print(f"❌ Error capturing complete page screenshot: {e}")
            return None
    
    def _extract_text_from_screenshot(self, screenshot_path: str, ticker: str, tab: str, quarter: str) -> Optional[str]:
        """Extract text from screenshot using OpenAI Vision with enhanced prompts for complete content"""
        try:
            with open(screenshot_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create specific prompts for each tab type with enhanced instructions
            if tab == "summary":
                prompt = f"""
Extract comprehensive transcript summary data from this COMPLETE ArthaLens page for {ticker} ({quarter}).

This is a FULL PAGE screenshot, so extract ALL visible content including:

1. **Financial Highlights and Key Metrics:**
   - Revenue, profit, margins, growth rates
   - Key performance indicators
   - Segment-wise performance
   - Year-over-year comparisons

2. **Segment Performance Analysis:**
   - Business segment breakdowns
   - Geographic performance
   - Product-wise analysis
   - Customer segment performance

3. **Management Commentary and Strategic Insights:**
   - CEO/CFO statements
   - Strategic initiatives
   - Market outlook
   - Business model insights
   - Competitive positioning

4. **Key Operational Metrics:**
   - Operational efficiency metrics
   - Cost structure analysis
   - Productivity measures
   - Quality metrics

5. **Forward-Looking Statements:**
   - Future guidance
   - Growth projections
   - Strategic plans
   - Investment plans

6. **Revenue and Profit Trends:**
   - Historical trends
   - Growth drivers
   - Margin analysis
   - Revenue mix

7. **Market Position and Competitive Analysis:**
   - Market share
   - Competitive advantages
   - Industry positioning
   - Market opportunities

8. **Risk Factors and Challenges:**
   - Operational risks
   - Market risks
   - Regulatory risks
   - Competitive threats

9. **Capital Allocation and Investment:**
   - Capex plans
   - Investment priorities
   - Dividend policy
   - Share buybacks

10. **ESG and Sustainability:**
    - Environmental initiatives
    - Social responsibility
    - Governance practices

Provide the extracted information in a structured format with clear sections and subsections.
Include ALL visible text and data from the complete page.
"""
            elif tab == "guidance":
                prompt = f"""
Extract comprehensive future guidance and outlook data from this COMPLETE ArthaLens page for {ticker} ({quarter}).

This is a FULL PAGE screenshot, so extract ALL visible content including:

1. **Revenue Guidance:**
   - Revenue growth projections
   - Segment-wise revenue outlook
   - Geographic revenue expectations
   - Product-wise revenue forecasts

2. **Profitability Guidance:**
   - Margin improvement targets
   - Cost optimization plans
   - Profit growth projections
   - EBITDA margin expectations

3. **Operational Guidance:**
   - Capacity expansion plans
   - Operational efficiency targets
   - Market expansion strategies
   - Customer acquisition goals

4. **Investment and Capex Plans:**
   - Capital expenditure guidance
   - Investment priorities
   - R&D spending plans
   - Technology investments

5. **Market Outlook:**
   - Industry growth expectations
   - Market share targets
   - Competitive positioning
   - Market opportunity assessment

6. **Risk Factors:**
   - Key risks to guidance
   - External factors affecting outlook
   - Regulatory challenges
   - Market uncertainties

7. **Strategic Initiatives:**
   - New product launches
   - Market entry plans
   - Partnership strategies
   - Digital transformation

8. **Timeline and Milestones:**
   - Quarterly targets
   - Annual goals
   - Long-term objectives
   - Key milestones

Provide the extracted information in a structured format with clear sections and subsections.
Include ALL visible text and data from the complete page.
"""
            elif tab == "concall":
                prompt = f"""
Extract comprehensive concall transcript data from this COMPLETE ArthaLens page for {ticker} ({quarter}).

This is a FULL PAGE screenshot, so extract ALL visible content including:

1. **Management Discussion:**
   - CEO/CFO opening remarks
   - Key highlights and achievements
   - Performance overview
   - Strategic updates

2. **Financial Performance:**
   - Revenue analysis
   - Profitability discussion
   - Margin trends
   - Segment performance

3. **Operational Updates:**
   - Business operations
   - Market performance
   - Customer metrics
   - Operational efficiency

4. **Strategic Initiatives:**
   - New projects
   - Market expansion
   - Product development
   - Technology investments

5. **Market Outlook:**
   - Industry trends
   - Market opportunities
   - Competitive landscape
   - Growth drivers

6. **Q&A Session:**
   - Analyst questions
   - Management responses
   - Key insights
   - Future guidance

7. **Risk Factors:**
   - Operational risks
   - Market risks
   - Regulatory challenges
   - Competitive threats

8. **Forward-Looking Statements:**
   - Future guidance
   - Growth projections
   - Strategic plans
   - Investment priorities

Provide the extracted information in a structured format with clear sections and subsections.
Include ALL visible text and data from the complete page.
"""
            else:
                prompt = f"""
Extract comprehensive data from this COMPLETE ArthaLens page for {ticker} ({quarter}) - {tab} tab.

This is a FULL PAGE screenshot, so extract ALL visible content including:

1. **Financial Data:**
   - Revenue, profit, margins
   - Growth rates and trends
   - Key performance indicators
   - Segment-wise performance

2. **Operational Data:**
   - Business metrics
   - Operational efficiency
   - Market performance
   - Customer data

3. **Strategic Information:**
   - Management commentary
   - Strategic initiatives
   - Market outlook
   - Future plans

4. **Risk and Challenges:**
   - Risk factors
   - Operational challenges
   - Market risks
   - Competitive threats

Provide the extracted information in a structured format with clear sections and subsections.
Include ALL visible text and data from the complete page.
"""
            
            # Call OpenAI with the appropriate prompt
            response = self.openai_client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-5'),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=3000
            )
            
            extracted_text = response.choices[0].message.content.strip()
            print(f"✅ Text extracted from {tab} tab ({quarter})")
            return extracted_text
            
        except Exception as e:
            print(f"❌ Error extracting text from screenshot: {e}")
            return None
    
    def _generate_correlated_insights(self, ticker: str, arthalens_data: Dict, fundamental_data: Dict) -> Dict:
        """Generate correlated insights between ArthaLens data and fundamental analysis"""
        try:
            print(f"🧠 Generating correlated insights for {ticker}...")
            
            # Prepare the correlation analysis prompt with multi-quarter data
            correlation_prompt = f"""
            Analyze the correlation between fundamental data and ArthaLens transcript/guidance data for {ticker} across multiple quarters.
            
            FUNDAMENTAL DATA:
            {json.dumps(fundamental_data, indent=2)}
            
            ARTHALENS DATA (Multiple Quarters):
            {json.dumps(arthalens_data, indent=2)}
            
            Provide a comprehensive correlation analysis that includes:
            
            1. **Multi-Quarter Trend Analysis:**
               - Compare fundamental trends across quarters with management commentary
               - Identify if management guidance aligns with fundamental performance over time
               - Analyze if strategic initiatives are reflected in financial metrics consistently
               - Track changes in management outlook and guidance across quarters
            
            2. **Growth Driver Evolution:**
               - Identify key growth drivers mentioned in transcripts across quarters
               - Correlate with fundamental growth metrics over time
               - Assess if growth drivers are sustainable based on financial health trends
               - Track how growth drivers have evolved or remained consistent
            
            3. **Risk Assessment Evolution:**
               - Compare fundamental risks with management's risk discussion across quarters
               - Identify any changes in risk perception or mitigation strategies
               - Assess risk mitigation strategies mentioned and their effectiveness over time
               - Track emerging or diminishing risks across quarters
            
            4. **Strategic Alignment Consistency:**
               - Evaluate if management strategy aligns with financial performance across quarters
               - Assess if capital allocation matches strategic priorities consistently
               - Identify strategic initiatives that could impact future fundamentals
               - Track strategic execution consistency over time
            
            5. **Management Credibility Assessment:**
               - Identify positive/negative signals from correlation across quarters
               - Assess management credibility based on guidance accuracy over time
               - Evaluate consistency between what management says and financial reality
               - Track management's ability to deliver on promises across quarters
            
            6. **Quarter-over-Quarter Insights:**
               - Compare guidance accuracy between quarters
               - Identify patterns in management communication
               - Assess consistency in strategic messaging
               - Track performance vs. guidance alignment
            
            Provide specific insights with supporting evidence from both datasets across all available quarters.
            Focus on trends, patterns, and consistency over time.
            """
            
            response = self.openai_client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-5'),
                messages=[
                    {
                        "role": "user",
                        "content": correlation_prompt
                    }
                ],
                max_completion_tokens=3000
            )
            
            correlated_insights = response.choices[0].message.content.strip()
            
            return {
                "analysis": correlated_insights,
                "generated_at": datetime.now().isoformat(),
                "quarters_analyzed": list(arthalens_data.get("summary_data", {}).keys())
            }
            
        except Exception as e:
            print(f"❌ Error generating correlated insights: {e}")
            return {"error": str(e)}
    
    def _analyze_future_confidence(self, ticker: str, arthalens_data: Dict, fundamental_data: Dict) -> Dict:
        """Analyze confidence in future growth based on correlated data across multiple quarters"""
        try:
            print(f"🎯 Analyzing future confidence for {ticker}...")
            
            confidence_prompt = f"""
            Based on the fundamental data and ArthaLens transcript/guidance analysis for {ticker} across multiple quarters, 
            provide a comprehensive confidence assessment for future growth.
            
            FUNDAMENTAL DATA:
            {json.dumps(fundamental_data, indent=2)}
            
            ARTHALENS DATA (Multiple Quarters):
            {json.dumps(arthalens_data, indent=2)}
            
            Provide a structured confidence analysis including:
            
            1. **Overall Confidence Level:** (High/Medium/Low with percentage)
               - Consider consistency across quarters
            
            2. **Growth Confidence Factors:**
               - Revenue growth sustainability (based on multi-quarter trends)
               - Profit margin expansion potential (based on consistent performance)
               - Market share growth opportunities (based on strategic consistency)
               - Strategic initiative success probability (based on execution track record)
            
            3. **Risk Factors:**
               - Key risks that could impact growth (based on evolving risk landscape)
               - Management's risk mitigation effectiveness (based on historical performance)
               - External factors affecting growth (based on quarter-over-quarter analysis)
            
            4. **Management Credibility:**
               - Past guidance accuracy (across multiple quarters)
               - Strategic execution track record (consistency over time)
               - Communication transparency (based on quarter-over-quarter analysis)
            
            5. **Financial Health Assessment:**
               - Balance sheet strength for growth (based on consistent metrics)
               - Cash flow sustainability (based on multi-quarter trends)
               - Capital allocation efficiency (based on strategic consistency)
            
            6. **Competitive Position:**
               - Market position strength (based on consistent performance)
               - Competitive advantages (based on strategic execution)
               - Industry tailwinds/headwinds (based on evolving market conditions)
            
            7. **Investment Recommendation:**
               - Buy/Hold/Sell with reasoning (based on multi-quarter analysis)
               - Time horizon for investment (based on strategic consistency)
               - Key catalysts to watch (based on quarter-over-quarter trends)
            
            8. **Quarter-over-Quarter Analysis:**
               - Guidance accuracy trends
               - Strategic execution consistency
               - Performance predictability
               - Management reliability over time
            
            Provide specific evidence and reasoning for each assessment based on multi-quarter data.
            Consider trends, patterns, and consistency across all available quarters.
            """
            
            response = self.openai_client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-5'),
                messages=[
                    {
                        "role": "user",
                        "content": confidence_prompt
                    }
                ],
                max_completion_tokens=2500
            )
            
            confidence_analysis = response.choices[0].message.content.strip()
            
            return {
                "analysis": confidence_analysis,
                "generated_at": datetime.now().isoformat(),
                "quarters_analyzed": list(arthalens_data.get("summary_data", {}).keys())
            }
            
        except Exception as e:
            print(f"❌ Error analyzing future confidence: {e}")
            return {"error": str(e)}
    
    def extract_multiple_stocks(self, tickers: List[str], fundamental_data_dict: Dict = None) -> Dict:
        """Extract ArthaLens data for multiple stocks"""
        print(f"🚀 Starting ArthaLens extraction for {len(tickers)} stocks...")
        
        results = {}
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n📊 Processing {i}/{len(tickers)}: {ticker}")
            
            # Get fundamental data for this ticker if available
            fundamental_data = fundamental_data_dict.get(ticker) if fundamental_data_dict else None
            
            # Extract ArthaLens data
            result = self.extract_arthalens_data(ticker, fundamental_data)
            results[ticker] = result
            
            # Add delay between requests
            if i < len(tickers):
                print("⏳ Waiting 10 seconds before next stock...")
                time.sleep(10)
        
        # Save combined results
        combined_file = os.path.join(self.results_dir, "combined_arthalens_extraction.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 Extraction completed for all stocks!")
        print(f"📁 Combined results saved: {combined_file}")
        
        return results

def main():
    """Main function to test ArthaLens extraction"""
    extractor = ArthaLensExtractor()
    
    # Test stocks
    test_tickers = ['HDFCBANK', 'RELIANCE', 'TCS']
    
    # Mock fundamental data for testing (replace with actual data)
    mock_fundamental_data = {
        'HDFCBANK': {
            'revenue_growth': '+15.2%',
            'profit_growth': '+12.8%',
            'roce': '18.5%',
            'debt_to_equity': '8.2%',
            'market_cap': '₹15,28,675.9 Cr',
            'current_price': '₹1,978.4'
        },
        'RELIANCE': {
            'revenue_growth': '+22.1%',
            'profit_growth': '+18.7%',
            'roce': '12.3%',
            'debt_to_equity': '15.8%',
            'market_cap': '₹18,45,321.2 Cr',
            'current_price': '₹2,456.8'
        },
        'TCS': {
            'revenue_growth': '+8.9%',
            'profit_growth': '+11.2%',
            'roce': '45.2%',
            'debt_to_equity': '2.1%',
            'market_cap': '₹12,34,567.8 Cr',
            'current_price': '₹3,456.7'
        }
    }
    
    # Extract data for all stocks
    results = extractor.extract_multiple_stocks(test_tickers, mock_fundamental_data)
    
    # Print summary
    print(f"\n📊 EXTRACTION SUMMARY:")
    for ticker, result in results.items():
        if "error" not in result:
            print(f"   ✅ {ticker}: Successfully extracted")
            if "correlated_insights" in result:
                print(f"      📈 Correlated insights generated")
            if "confidence_analysis" in result:
                print(f"      🎯 Confidence analysis completed")
        else:
            print(f"   ❌ {ticker}: Failed - {result['error']}")

if __name__ == "__main__":
    main() 