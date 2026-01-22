import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Indian Stock Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Session state for debugging
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'error_logs' not in st.session_state:
    st.session_state.error_logs = []

# Indian stock symbols (NSE)
DEFAULT_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'ICICIBANK.NS': 'ICICI Bank',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ITC.NS': 'ITC Limited',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro',
    'ASIANPAINT.NS': 'Asian Paints',
    'AXISBANK.NS': 'Axis Bank',
    'MARUTI.NS': 'Maruti Suzuki',
    'TITAN.NS': 'Titan Company',
    'WIPRO.NS': 'Wipro',
    'ADANIENT.NS': 'Adani Enterprises',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'HCLTECH.NS': 'HCL Technologies',
    'SUNPHARMA.NS': 'Sun Pharma'
}

def log_error(message):
    """Log errors for debugging"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    st.session_state.error_logs.append(f"[{timestamp}] {message}")
    if st.session_state.debug_mode:
        st.warning(f"Debug: {message}")

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_stock_data(symbol, max_retries=3):
    """Fetch current stock data from Yahoo Finance with retry logic"""
    for attempt in range(max_retries):
        try:
            # Add delay between retries
            if attempt > 0:
                time.sleep(2)
            
            # Create ticker with timeout
            stock = yf.Ticker(symbol)
            
            # Try to get data with multiple methods
            hist = stock.history(period='5d', timeout=10)
            
            if hist.empty:
                # Try alternative period
                hist = stock.history(period='1mo', timeout=10)
            
            if hist.empty:
                log_error(f"{symbol}: No historical data available")
                continue
                
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = ((current_price - prev_close) / prev_close) * 100
            volume = hist['Volume'].iloc[-1]
            
            # Try to get info, but don't fail if it doesn't work
            try:
                info = stock.info
                market_cap = info.get('marketCap', 'N/A')
            except:
                market_cap = 'N/A'
            
            return {
                'symbol': symbol,
                'price': round(current_price, 2),
                'change': round(change, 2),
                'volume': int(volume),
                'prev_close': round(prev_close, 2),
                'market_cap': market_cap
            }
            
        except Exception as e:
            log_error(f"{symbol} attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            continue
    
    return None

@st.cache_data(ttl=1800)
def scrape_moneycontrol_news(company_name, limit=5):
    """Scrape news from MoneyControl with improved error handling"""
    news_items = []
    try:
        # Format company name for search
        search_query = company_name.replace(' ', '-').lower()
        
        # Try multiple URL patterns
        urls = [
            f"https://www.moneycontrol.com/news/business/stocks/{search_query}-{limit}.html",
            f"https://www.moneycontrol.com/news/tags/{search_query}.html",
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Try multiple selectors
                    articles = soup.find_all('li', class_='clearfix', limit=limit)
                    if not articles:
                        articles = soup.find_all('h2', limit=limit)
                    
                    for article in articles:
                        try:
                            title_tag = article.find('a')
                            if title_tag:
                                title = title_tag.text.strip()
                                if len(title) > 20:  # Valid title
                                    news_items.append(title)
                        except:
                            continue
                    
                    if len(news_items) >= 2:
                        break
            except:
                continue
                
    except Exception as e:
        log_error(f"MoneyControl scraping error for {company_name}: {str(e)}")
    
    # If MoneyControl fails, try Economic Times
    if len(news_items) < 2:
        news_items.extend(scrape_economic_times_news(company_name, limit))
    
    # If still not enough, try generic news
    if len(news_items) < 2:
        news_items.extend(scrape_generic_news(company_name, limit))
    
    return news_items[:limit]

def scrape_economic_times_news(company_name, limit=5):
    """Scrape news from Economic Times"""
    news_items = []
    try:
        search_query = company_name.replace(' ', '%20')
        url = f"https://economictimes.indiatimes.com/topic/{search_query}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('h3', limit=limit)
            
            for article in articles:
                try:
                    title = article.text.strip()
                    if len(title) > 20:
                        news_items.append(title)
                except:
                    continue
                    
    except Exception as e:
        log_error(f"Economic Times scraping error for {company_name}: {str(e)}")
    
    return news_items

def scrape_generic_news(company_name, limit=5):
    """Scrape news from Google News with improved headers"""
    news_items = []
    try:
        search_query = company_name.replace(' ', '+') + '+stock+NSE+news'
        url = f"https://www.google.com/search?q={search_query}&tbm=nws"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try multiple selectors for Google News
            articles = soup.find_all(['h3', 'div'], class_=re.compile('BNeawe|n0jPhd'), limit=limit*2)
            
            for article in articles:
                try:
                    title = article.text.strip()
                    if len(title) > 20 and len(title) < 200:
                        news_items.append(title)
                except:
                    continue
                    
    except Exception as e:
        log_error(f"Generic news scraping error for {company_name}: {str(e)}")
    
    return news_items[:limit]

def analyze_sentiment(news_items):
    """Analyze sentiment of news items using VADER"""
    if not news_items or len(news_items) == 0:
        return 0, "No recent news found. Sentiment based on market data only."
    
    sentiments = []
    positive_reasons = []
    negative_reasons = []
    
    for news in news_items:
        if not news or len(news) < 10:
            continue
            
        sentiment_score = analyzer.polarity_scores(news)
        compound = sentiment_score['compound']
        sentiments.append(compound)
        
        # Extract key phrases for reasons with better formatting
        if compound > 0.05:
            positive_reasons.append(f"📈 {news[:120]}...")
        elif compound < -0.05:
            negative_reasons.append(f"📉 {news[:120]}...")
    
    if not sentiments:
        return 0, "No significant news sentiment detected."
    
    avg_sentiment = sum(sentiments) / len(sentiments)
    
    # Build reason text
    reason_parts = []
    if positive_reasons:
        reason_parts.append("POSITIVE SIGNALS:\n" + "\n".join(positive_reasons[:3]))
    if negative_reasons:
        reason_parts.append("NEGATIVE SIGNALS:\n" + "\n".join(negative_reasons[:3]))
    
    if not reason_parts:
        reason_text = f"Analyzed {len(news_items)} news items - Overall sentiment: {'Positive' if avg_sentiment > 0 else 'Negative' if avg_sentiment < 0 else 'Neutral'}"
    else:
        reason_text = "\n\n".join(reason_parts)
    
    return avg_sentiment, reason_text

def process_stock(symbol, name):
    """Process a single stock - fetch data and analyze sentiment with robust error handling"""
    try:
        # Get stock data with retry
        stock_data = get_stock_data(symbol)
        
        if not stock_data:
            log_error(f"{name}: Failed to fetch stock data")
            # Return minimal data to not completely fail
            return {
                'Company': name,
                'Symbol': symbol.replace('.NS', ''),
                'Price (₹)': 'N/A',
                'Change (%)': 0.0,
                'Volume': 'N/A',
                'Sentiment Score': 0.0,
                'Sentiment': 'Data Unavailable',
                'Reasons': 'Unable to fetch stock data. Please try again.',
                'Market Cap': 'N/A',
                'Status': 'Failed'
            }
        
        # Get news and analyze sentiment
        try:
            news_items = scrape_moneycontrol_news(name, limit=5)
            sentiment_score, reasons = analyze_sentiment(news_items)
        except Exception as e:
            log_error(f"{name}: Sentiment analysis failed - {str(e)}")
            sentiment_score = 0.0
            reasons = "Sentiment analysis unavailable"
        
        # Determine sentiment category
        if sentiment_score > 0.05:
            sentiment = 'Positive'
        elif sentiment_score < -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'Company': name,
            'Symbol': symbol.replace('.NS', ''),
            'Price (₹)': stock_data['price'],
            'Change (%)': stock_data['change'],
            'Volume': f"{stock_data['volume']:,}",
            'Sentiment Score': round(sentiment_score, 3),
            'Sentiment': sentiment,
            'Reasons': reasons,
            'Market Cap': stock_data['market_cap'],
            'Status': 'Success'
        }
        
    except Exception as e:
        log_error(f"{name}: Complete processing failed - {str(e)}")
        return {
            'Company': name,
            'Symbol': symbol.replace('.NS', ''),
            'Price (₹)': 'N/A',
            'Change (%)': 0.0,
            'Volume': 'N/A',
            'Sentiment Score': 0.0,
            'Sentiment': 'Error',
            'Reasons': f'Processing error: {str(e)[:100]}',
            'Market Cap': 'N/A',
            'Status': 'Failed'
        }

def main():
    st.title("📊 Indian Stock Sentiment Dashboard")
    st.markdown("*Real-time sentiment analysis based on news and market data*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=False)
        
        # Stock selection
        selected_stocks = st.multiselect(
            "Select Stocks to Analyze",
            options=list(DEFAULT_STOCKS.keys()),
            default=list(DEFAULT_STOCKS.keys())[:10],
            format_func=lambda x: f"{DEFAULT_STOCKS[x]} ({x.replace('.NS', '')})"
        )
        
        # Sentiment threshold
        sentiment_threshold = st.slider(
            "Sentiment Threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Minimum sentiment score to classify as positive/negative"
        )
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.error_logs = []
            st.rerun()
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
        
        st.markdown("---")
        st.markdown("**Data Sources:**")
        st.markdown("- Stock Data: Yahoo Finance")
        st.markdown("- News: MoneyControl, ET, Google")
        st.markdown("- Sentiment: VADER Analysis")
        
        if st.session_state.error_logs and st.session_state.debug_mode:
            st.markdown("---")
            st.markdown("**Error Logs:**")
            with st.expander("View Logs"):
                for log in st.session_state.error_logs[-10:]:
                    st.text(log)
    
    if not selected_stocks:
        st.warning("Please select at least one stock to analyze.")
        return
    
    # Process stocks with progress bar
    st.subheader("📈 Processing Stock Data...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    failed_stocks = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced workers to avoid rate limiting
        futures = {
            executor.submit(process_stock, symbol, DEFAULT_STOCKS[symbol]): symbol 
            for symbol in selected_stocks
        }
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                if result.get('Status') == 'Success':
                    results.append(result)
                else:
                    failed_stocks.append(result['Company'])
                    results.append(result)  # Include failed ones too
            completed += 1
            progress = completed / len(selected_stocks)
            progress_bar.progress(progress)
            status_text.text(f"Processed {completed}/{len(selected_stocks)} stocks")
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("❌ No data could be retrieved. Please try again later.")
        st.markdown("**Troubleshooting:**")
        st.markdown("- Check your internet connection")
        st.markdown("- Try selecting different stocks")
        st.markdown("- Enable Debug Mode in sidebar to see detailed errors")
        st.markdown("- Click 'Clear Cache' and try again")
        return
    
    # Show warning for failed stocks
    if failed_stocks:
        st.warning(f"⚠️ Failed to retrieve complete data for: {', '.join(failed_stocks[:5])}")
    
    # Filter successful results
    successful_results = [r for r in results if r.get('Status') == 'Success' and r['Price (₹)'] != 'N/A']
    
    if not successful_results:
        st.error("❌ No complete stock data available. Please try again.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(successful_results)
    
    # Separate positive and negative sentiments
    positive_df = df[df['Sentiment Score'] > sentiment_threshold].sort_values('Sentiment Score', ascending=False)
    negative_df = df[df['Sentiment Score'] < -sentiment_threshold].sort_values('Sentiment Score', ascending=True)
    neutral_df = df[(df['Sentiment Score'] >= -sentiment_threshold) & (df['Sentiment Score'] <= sentiment_threshold)]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", len(df))
    with col2:
        st.metric("Positive Sentiment", len(positive_df), delta="Bullish", delta_color="normal")
    with col3:
        st.metric("Negative Sentiment", len(negative_df), delta="Bearish", delta_color="inverse")
    with col4:
        st.metric("Neutral", len(neutral_df))
    
    # Positive Sentiment Section
    st.markdown("---")
    st.subheader("🟢 Positive Sentiment Stocks")
    
    if not positive_df.empty:
        for idx, row in positive_df.iterrows():
            with st.expander(f"**{row['Company']}** ({row['Symbol']}) - Sentiment: {row['Sentiment Score']:.3f}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Price", f"₹{row['Price (₹)']}", f"{row['Change (%)']}%")
                with col2:
                    st.metric("Volume", row['Volume'])
                with col3:
                    sentiment_emoji = "🚀" if row['Sentiment Score'] > 0.2 else "📈"
                    st.metric("Sentiment", f"{sentiment_emoji} Strong" if row['Sentiment Score'] > 0.2 else "📈 Positive")
                
                st.markdown("**News Analysis:**")
                st.text_area("Reasons for Positive Sentiment", row['Reasons'], height=150, key=f"pos_{idx}", disabled=True)
    else:
        st.info("No stocks with positive sentiment found based on current threshold.")
    
    # Negative Sentiment Section
    st.markdown("---")
    st.subheader("🔴 Negative Sentiment Stocks")
    
    if not negative_df.empty:
        for idx, row in negative_df.iterrows():
            with st.expander(f"**{row['Company']}** ({row['Symbol']}) - Sentiment: {row['Sentiment Score']:.3f}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Price", f"₹{row['Price (₹)']}", f"{row['Change (%)']}%")
                with col2:
                    st.metric("Volume", row['Volume'])
                with col3:
                    sentiment_emoji = "⚠️" if row['Sentiment Score'] < -0.2 else "📉"
                    st.metric("Sentiment", f"{sentiment_emoji} Strong" if row['Sentiment Score'] < -0.2 else "📉 Negative")
                
                st.markdown("**News Analysis:**")
                st.text_area("Reasons for Negative Sentiment", row['Reasons'], height=150, key=f"neg_{idx}", disabled=True)
    else:
        st.info("No stocks with negative sentiment found based on current threshold.")
    
    # Neutral Sentiment Section
    if not neutral_df.empty:
        st.markdown("---")
        st.subheader("⚪ Neutral Sentiment Stocks")
        st.dataframe(
            neutral_df[['Company', 'Symbol', 'Price (₹)', 'Change (%)', 'Volume', 'Sentiment Score']],
            use_container_width=True,
            hide_index=True
        )
    
    # Full Data Table
    st.markdown("---")
    st.subheader("📋 Complete Stock Screener")
    
    # Format the dataframe for display
    display_df = df[['Company', 'Symbol', 'Price (₹)', 'Change (%)', 'Volume', 'Sentiment Score', 'Sentiment']].copy()
    
    # Color coding function
    def color_sentiment(val):
        if isinstance(val, (int, float)):
            if val > sentiment_threshold:
                return 'background-color: #90EE90'
            elif val < -sentiment_threshold:
                return 'background-color: #FFB6C6'
            return 'background-color: #FFFACD'
        return ''
    
    styled_df = display_df.style.applymap(color_sentiment, subset=['Sentiment Score'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Download options
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv,
            file_name=f"stock_sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()
