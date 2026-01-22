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

# Page configuration
st.set_page_config(
    page_title="Indian Stock Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

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

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_stock_data(symbol):
    """Fetch current stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period='5d')
        
        if hist.empty:
            return None
            
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = ((current_price - prev_close) / prev_close) * 100
        volume = hist['Volume'].iloc[-1]
        
        return {
            'symbol': symbol,
            'price': round(current_price, 2),
            'change': round(change, 2),
            'volume': int(volume),
            'prev_close': round(prev_close, 2),
            'market_cap': info.get('marketCap', 'N/A')
        }
    except Exception as e:
        return None

@st.cache_data(ttl=1800)
def scrape_moneycontrol_news(company_name, limit=5):
    """Scrape news from MoneyControl"""
    news_items = []
    try:
        # Format company name for search
        search_query = company_name.replace(' ', '+')
        url = f"https://www.moneycontrol.com/news/tags/{search_query.lower()}.html"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('li', class_='clearfix', limit=limit)
            
            for article in articles:
                try:
                    title_tag = article.find('h2')
                    if title_tag and title_tag.find('a'):
                        title = title_tag.find('a').text.strip()
                        news_items.append(title)
                except:
                    continue
                    
        # If MoneyControl fails, try generic news search
        if len(news_items) < 2:
            news_items.extend(scrape_generic_news(company_name, limit))
            
    except Exception as e:
        news_items.extend(scrape_generic_news(company_name, limit))
    
    return news_items[:limit]

def scrape_generic_news(company_name, limit=5):
    """Scrape news from Google News"""
    news_items = []
    try:
        search_query = company_name.replace(' ', '+') + '+stock+india'
        url = f"https://news.google.com/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('article', limit=limit)
            
            for article in articles:
                try:
                    title_tag = article.find('a', class_='gPFEn')
                    if title_tag:
                        title = title_tag.text.strip()
                        news_items.append(title)
                except:
                    continue
                    
    except Exception as e:
        pass
    
    return news_items

def analyze_sentiment(news_items):
    """Analyze sentiment of news items using VADER"""
    if not news_items:
        return 0, "No recent news available"
    
    sentiments = []
    reasons = []
    
    for news in news_items:
        sentiment_score = analyzer.polarity_scores(news)
        sentiments.append(sentiment_score['compound'])
        
        # Extract key phrases for reasons
        if sentiment_score['compound'] > 0.05:
            reasons.append(f"✓ {news[:100]}...")
        elif sentiment_score['compound'] < -0.05:
            reasons.append(f"✗ {news[:100]}...")
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    reason_text = "\n".join(reasons[:3]) if reasons else "Neutral news sentiment"
    
    return avg_sentiment, reason_text

def process_stock(symbol, name):
    """Process a single stock - fetch data and analyze sentiment"""
    try:
        # Get stock data
        stock_data = get_stock_data(symbol)
        if not stock_data:
            return None
        
        # Get news and analyze sentiment
        news_items = scrape_moneycontrol_news(name, limit=5)
        sentiment_score, reasons = analyze_sentiment(news_items)
        
        return {
            'Company': name,
            'Symbol': symbol.replace('.NS', ''),
            'Price (₹)': stock_data['price'],
            'Change (%)': stock_data['change'],
            'Volume': f"{stock_data['volume']:,}",
            'Sentiment Score': round(sentiment_score, 3),
            'Sentiment': 'Positive' if sentiment_score > 0.05 else 'Negative' if sentiment_score < -0.05 else 'Neutral',
            'Reasons': reasons,
            'Market Cap': stock_data['market_cap']
        }
    except Exception as e:
        return None

def main():
    st.title("📊 Indian Stock Sentiment Dashboard")
    st.markdown("*Real-time sentiment analysis based on news and market data*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
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
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Data Sources:**")
        st.markdown("- Stock Data: Yahoo Finance")
        st.markdown("- News: MoneyControl, Google News")
        st.markdown("- Sentiment: VADER Analysis")
    
    if not selected_stocks:
        st.warning("Please select at least one stock to analyze.")
        return
    
    # Process stocks with progress bar
    st.subheader("📈 Processing Stock Data...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_stock, symbol, DEFAULT_STOCKS[symbol]): symbol 
            for symbol in selected_stocks
        }
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
            completed += 1
            progress = completed / len(selected_stocks)
            progress_bar.progress(progress)
            status_text.text(f"Processed {completed}/{len(selected_stocks)} stocks")
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("No data could be retrieved. Please try again later.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
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
        st.info("No stocks with positive sentiment found.")
    
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
        st.info("No stocks with negative sentiment found.")
    
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
        if val > sentiment_threshold:
            return 'background-color: #90EE90'
        elif val < -sentiment_threshold:
            return 'background-color: #FFB6C6'
        return 'background-color: #FFFACD'
    
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
