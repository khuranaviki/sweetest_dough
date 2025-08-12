# Import the extraction function
from enhanced_screener_extraction_v3 import EnhancedScreenerExtractionV3
import os

# Use the extraction function in the main workflow

def main():
    # Initialize the extractor
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
        
    extractor = EnhancedScreenerExtractionV3(api_key)
    
    # Extract data for the specified company
    company_data = extractor.extract_complete_data('DELHIVERY.NS', 'output_directory')
    
    # Process and display the data
    # ... existing code ...
    
    # Render data in Streamlit app
    render_data_in_streamlit(company_data)

# Function to render data in Streamlit

def render_data_in_streamlit(data):
    import streamlit as st
    
    st.title('Company Financial Data')
    
    # Display Key Metrics
    st.header('Key Metrics')
    st.write(data['key_metrics'])
    
    # Display Quarterly Results
    st.header('Quarterly Results')
    st.write(data['quarterly_results'])
    
    # Display Annual Results
    st.header('Annual Results')
    st.write(data['annual_results'])
    
    # Display Balance Sheet
    st.header('Balance Sheet')
    st.write(data['balance_sheet'])
    
    # Display Cash Flows
    st.header('Cash Flows')
    st.write(data['cash_flows'])
    
    # Display Shareholding Pattern
    st.header('Shareholding Pattern')
    st.write(data['shareholding'])

if __name__ == '__main__':
    main() 