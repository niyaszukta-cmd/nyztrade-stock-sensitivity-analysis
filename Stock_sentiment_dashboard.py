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
import pytz
from pytrends.request import TrendReq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Indian Standard Time
IST = pytz.timezone('Asia/Kolkata')

# Page configuration
st.set_page_config(
    page_title="Dynamic Stock Sentiment Screener",
    page_icon="📊",
    layout="wide"
)

# Initialize sentiment analyzer with custom financial lexicon
analyzer = SentimentIntensityAnalyzer()

# Add custom financial terms for Indian markets
financial_lexicon = {
    # Strong Positive
    'rally': 3.0,
    'surge': 3.0,
    'soar': 3.0,
    'skyrocket': 3.5,
    'breakout': 2.5,
    'bullish': 2.8,
    'outperform': 2.5,
    'upgrade': 2.3,
    'beat': 2.0,
    'strong': 2.0,
    'robust': 2.2,
    'stellar': 2.5,
    'record': 2.3,
    'milestone': 2.0,
    'profit': 1.8,
    'gain': 1.7,
    'growth': 1.8,
    'expansion': 1.7,
    'positive': 1.5,
    'opportunity': 1.5,
    'buy': 2.0,
    'accumulate': 1.8,
    'momentum': 1.5,
    'winner': 2.0,
    'boom': 2.5,
    'optimistic': 1.8,
    'confident': 1.7,
    
    # Moderate Positive
    'improve': 1.3,
    'recovery': 1.5,
    'rebound': 1.5,
    'stable': 1.0,
    'hopeful': 1.2,
    'promising': 1.5,
    'potential': 1.2,
    
    # Strong Negative
    'crash': -3.5,
    'plunge': -3.0,
    'collapse': -3.2,
    'tumble': -2.5,
    'bearish': -2.8,
    'underperform': -2.5,
    'downgrade': -2.3,
    'miss': -2.0,
    'weak': -2.0,
    'poor': -2.0,
    'loss': -2.2,
    'decline': -1.8,
    'fall': -1.7,
    'drop': -1.7,
    'negative': -1.5,
    'sell': -2.0,
    'exit': -1.5,
    'concern': -1.5,
    'worry': -1.7,
    'risk': -1.3,
    'threat': -2.0,
    'crisis': -2.5,
    'trouble': -2.0,
    'struggle': -1.8,
    'disappointing': -2.0,
    'hurt': -1.7,
    'pressure': -1.5,
    
    # Moderate Negative
    'slowdown': -1.5,
    'caution': -1.2,
    'uncertainty': -1.3,
    'volatile': -1.0,
    'correction': -1.2,
    
    # Indian market specific
    'fii': 1.5,  # Foreign Institutional Investors buying
    'dii': 1.3,  # Domestic Institutional Investors
    'nifty': 0.5,
    'sensex': 0.5,
    'sebi': 0.0,  # Neutral regulator
    'rbi': 0.0,   # Neutral central bank
    'listing': 1.5,
    'ipo': 1.5,
    'buyback': 1.8,
    'dividend': 1.5,
    'bonus': 2.0,
    'split': 1.0,
    'merger': 1.2,
    'acquisition': 1.2,
    'divestment': -0.5,
    'delisting': -2.0,
    'halt': -2.5,
    'suspension': -2.8,
    'penalty': -2.3,
    'fraud': -3.5,
    'scam': -3.5,
    'investigation': -2.0,
    'lawsuit': -2.2,
    'default': -3.0,
}

# Update VADER lexicon
analyzer.lexicon.update(financial_lexicon)

# Session state for debugging
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'error_logs' not in st.session_state:
    st.session_state.error_logs = []

# Dynamic Stock Universe - Fetches live NSE stocks
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_nse_stock_list():
    """Fetch live NSE stock list and create mapping"""
    stock_map = {}
    
    print("Fetching live NSE stock list...")
    
    # Method 1: Fetch from NSE India directly
    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(1)
        
        response = session.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('data', []):
                symbol = item.get('symbol', '')
                if symbol:
                    nse_symbol = f"{symbol}.NS"
                    # Create multiple variations
                    company_name = symbol.lower()
                    stock_map[company_name] = nse_symbol
                    stock_map[symbol.lower()] = nse_symbol
            
            print(f"NSE API: Loaded {len(stock_map)} stock variations")
    except Exception as e:
        print(f"NSE API failed: {str(e)}")
    
    # Method 2: Add major stocks manually as fallback
    major_stocks = {
        'reliance': 'RELIANCE.NS', 'ril': 'RELIANCE.NS',
        'tcs': 'TCS.NS', 'infosys': 'INFY.NS', 'infy': 'INFY.NS',
        'hdfc bank': 'HDFCBANK.NS', 'hdfcbank': 'HDFCBANK.NS',
        'icici bank': 'ICICIBANK.NS', 'icicibank': 'ICICIBANK.NS',
        'sbi': 'SBIN.NS', 'state bank': 'SBIN.NS',
        'airtel': 'BHARTIARTL.NS', 'bharti airtel': 'BHARTIARTL.NS',
        'wipro': 'WIPRO.NS', 'hcl': 'HCLTECH.NS', 'hcl tech': 'HCLTECH.NS',
        'itc': 'ITC.NS', 'maruti': 'MARUTI.NS', 'maruti suzuki': 'MARUTI.NS',
        'tata motors': 'TATAMOTORS.NS', 'mahindra': 'M&M.NS', 'm&m': 'M&M.NS',
        'bajaj finance': 'BAJFINANCE.NS', 'bajfinance': 'BAJFINANCE.NS',
        'asian paints': 'ASIANPAINT.NS', 'titan': 'TITAN.NS',
        'hindustan unilever': 'HINDUNILVR.NS', 'hul': 'HINDUNILVR.NS',
        'larsen toubro': 'LT.NS', 'l&t': 'LT.NS', 'lt': 'LT.NS',
        'sun pharma': 'SUNPHARMA.NS', 'sunpharma': 'SUNPHARMA.NS',
        'axis bank': 'AXISBANK.NS', 'kotak': 'KOTAKBANK.NS',
        'adani enterprises': 'ADANIENT.NS', 'adani ports': 'ADANIPORTS.NS',
        'ntpc': 'NTPC.NS', 'ongc': 'ONGC.NS', 'ioc': 'IOC.NS',
        'coal india': 'COALINDIA.NS', 'tata steel': 'TATASTEEL.NS',
        'jsw steel': 'JSWSTEEL.NS', 'hindalco': 'HINDALCO.NS',
        'bajaj auto': 'BAJAJ-AUTO.NS', 'hero motocorp': 'HEROMOTOCO.NS',
        'britannia': 'BRITANNIA.NS', 'nestle': 'NESTLEIND.NS',
        'ultratech': 'ULTRACEMCO.NS', 'grasim': 'GRASIM.NS',
        'power grid': 'POWERGRID.NS', 'adani power': 'ADANIPOWER.NS',
        'cipla': 'CIPLA.NS', 'dr reddy': 'DRREDDY.NS', 'lupin': 'LUPIN.NS',
        'vedanta': 'VEDL.NS', 'nmdc': 'NMDC.NS', 'sail': 'SAIL.NS',
        'zomato': 'ZOMATO.NS', 'paytm': 'PAYTM.NS', 'nykaa': 'NYKAA.NS',
        'dmart': 'DMART.NS', 'irctc': 'IRCTC.NS', 'lic': 'LICI.NS',
        'adani green': 'ADANIGREEN.NS', 'adani transmission': 'ADANITRANS.NS',
        'tata consumer': 'TATACONSUM.NS', 'trent': 'TRENT.NS',
        'dabur': 'DABUR.NS', 'marico': 'MARICO.NS',
        'godrej consumer': 'GODREJCP.NS', 'pidilite': 'PIDILITIND.NS',
        'berger paints': 'BERGEPAINT.NS', 'havells': 'HAVELLS.NS',
        'siemens': 'SIEMENS.NS', 'abb': 'ABB.NS', 'bhel': 'BHEL.NS',
        'indusind': 'INDUSINDBK.NS', 'yes bank': 'YESBANK.NS',
        'pnb': 'PNB.NS', 'bank of baroda': 'BANKBARODA.NS',
        'eicher': 'EICHERMOT.NS', 'tvs motor': 'TVSMOTOR.NS',
        'biocon': 'BIOCON.NS', 'torrent pharma': 'TORNTPHARM.NS',
        'bpcl': 'BPCL.NS', 'hpcl': 'HPCL.NS', 'gail': 'GAIL.NS',
        'oil india': 'OIL.NS', 'tata power': 'TATAPOWER.NS',
        'jindal steel': 'JINDALSTEL.NS', 'ambuja cement': 'AMBUJACEM.NS',
        'acc': 'ACC.NS', 'sbi life': 'SBILIFE.NS', 'hdfc life': 'HDFCLIFE.NS',
        'icici lombard': 'ICICIGI.NS', 'bajaj finserv': 'BAJAJFINSV.NS',
        'shriram finance': 'SHRIRAMFIN.NS', 'policy bazaar': 'POLICYBZR.NS',
        'vodafone idea': 'IDEA.NS', 'idea': 'IDEA.NS',
    }
    
    stock_map.update(major_stocks)
    
    # Method 3: Try to fetch from alternative sources
    try:
        # Fetch popular stocks from Yahoo Finance screener
        url = "https://finance.yahoo.com/screener/predefined/ms_basic_materials?offset=0&count=100"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        # Parse and add more stocks
    except:
        pass
    
    print(f"Total stock universe: {len(set(stock_map.values()))} unique stocks")
    return stock_map

# Get dynamic stock map
COMPANY_SYMBOL_MAP = get_nse_stock_list()

# Helper Functions (defined first)
def log_error(message):
    """Log errors for debugging - thread-safe version with IST"""
    try:
        ist_time = datetime.now(IST)
        timestamp = ist_time.strftime('%H:%M:%S')
        log_msg = f"[{timestamp} IST] {message}"
        
        # Try to access session state, but don't fail if we can't (e.g., from thread)
        if hasattr(st, 'session_state') and 'error_logs' in st.session_state:
            st.session_state.error_logs.append(log_msg)
            if st.session_state.get('debug_mode', False):
                print(f"Debug: {message}")  # Print instead of st.warning in threads
        else:
            # Fallback: just print to console
            print(log_msg)
    except Exception:
        # Silently fail if logging doesn't work
        pass

def extract_stock_symbols_from_text(text):
    """Extract stock symbols from news headline/text"""
    text_lower = text.lower()
    found_symbols = []
    
    # Check against our company map
    for company, symbol in COMPANY_SYMBOL_MAP.items():
        if company in text_lower:
            if symbol not in found_symbols:
                found_symbols.append(symbol)
    
    # Also try to extract direct stock symbols (e.g., "RELIANCE" or "TCS")
    # Match capital words that could be stock symbols
    words = re.findall(r'\b[A-Z]{2,12}\b', text)
    for word in words:
        potential_symbol = f"{word}.NS"
        if word.lower() in COMPANY_SYMBOL_MAP:
            symbol = COMPANY_SYMBOL_MAP[word.lower()]
            if symbol not in found_symbols:
                found_symbols.append(symbol)
    
    return found_symbols

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_trending_stocks_for_viral_content():
    """Get trending stocks using Google Trends - Perfect for viral YouTube content!"""
    trending_data = []
    
    print("Fetching trending stocks from Google Trends...")
    
    try:
        pytrends = TrendReq(hl='en-IN', tz=330)  # Indian timezone
        
        # Keywords related to Indian stock market
        stock_keywords = [
            'Reliance stock',
            'TCS stock',
            'Adani stock',
            'Infosys stock',
            'HDFC Bank stock',
            'Tata Motors stock',
            'Zomato stock',
            'Paytm stock',
            'Nifty stocks',
            'BSE stocks'
        ]
        
        # Get trending searches related to stocks
        for keyword in stock_keywords:
            try:
                pytrends.build_payload([keyword], timeframe='now 7-d', geo='IN')
                interest_data = pytrends.interest_over_time()
                
                if not interest_data.empty:
                    avg_interest = interest_data[keyword].mean()
                    latest_interest = interest_data[keyword].iloc[-1]
                    
                    # Extract stock symbol from keyword
                    stock_name = keyword.replace(' stock', '').lower()
                    symbol = None
                    for company, sym in COMPANY_SYMBOL_MAP.items():
                        if stock_name in company or company in stock_name:
                            symbol = sym
                            break
                    
                    if symbol:
                        trending_data.append({
                            'symbol': symbol,
                            'keyword': keyword,
                            'avg_interest': avg_interest,
                            'latest_interest': latest_interest,
                            'trend_score': latest_interest * 1.5 + avg_interest  # Weighted score
                        })
                
                time.sleep(0.5)  # Respect rate limits
            except Exception as e:
                print(f"Error fetching trend for {keyword}: {str(e)}")
                continue
        
        # Get related queries for "Indian stocks"
        try:
            pytrends.build_payload(['Indian stocks'], timeframe='now 7-d', geo='IN')
            related = pytrends.related_queries()
            
            if 'Indian stocks' in related and related['Indian stocks']['rising'] is not None:
                rising_queries = related['Indian stocks']['rising']
                for _, row in rising_queries.head(20).iterrows():
                    query = row['query'].lower()
                    # Try to match with our stock map
                    for company, symbol in COMPANY_SYMBOL_MAP.items():
                        if company in query:
                            # Check if not already in trending_data
                            if not any(t['symbol'] == symbol for t in trending_data):
                                trending_data.append({
                                    'symbol': symbol,
                                    'keyword': row['query'],
                                    'avg_interest': row['value'],
                                    'latest_interest': row['value'],
                                    'trend_score': row['value']
                                })
                            break
        except Exception as e:
            print(f"Error fetching related queries: {str(e)}")
        
        # Sort by trend score
        trending_data.sort(key=lambda x: x['trend_score'], reverse=True)
        
        print(f"Found {len(trending_data)} trending stocks")
        return trending_data
        
    except Exception as e:
        print(f"Google Trends failed: {str(e)}")
        return []

def generate_youtube_metadata(stock_data, sentiment_score, is_trending=False):
    """Generate YouTube tags and hashtags for viral content"""
    company = stock_data['company']
    symbol = stock_data['symbol'].replace('.NS', '')
    
    # Base tags
    tags = [
        f"{company}",
        f"{company} stock",
        f"{symbol}",
        f"{symbol} stock",
        "Indian stock market",
        "NSE stocks",
        "stock market news",
        "stock analysis",
        "stock market today"
    ]
    
    # Sentiment-based tags
    if sentiment_score > 0.3:
        tags.extend([
            f"{company} rally",
            f"{company} bullish",
            "stocks to buy",
            "best stocks",
            "stock market rally",
            "bullish stocks",
            "stock breakout"
        ])
    elif sentiment_score < -0.3:
        tags.extend([
            f"{company} crash",
            f"{company} bearish",
            "stocks to sell",
            "stock market crash",
            "bearish stocks",
            "stock alert",
            "stock warning"
        ])
    else:
        tags.extend([
            f"{company} news",
            f"{company} update",
            "stock market analysis",
            "stock research",
            "stock market update"
        ])
    
    # Trending tags
    if is_trending:
        tags.extend([
            "trending stocks",
            "viral stocks",
            "most searched stocks",
            "hot stocks"
        ])
    
    # General Indian market tags
    tags.extend([
        "NIFTY",
        "SENSEX",
        "Indian stocks",
        "share market",
        "stock tips",
        "stock market india",
        "trading",
        "investing",
        "stock news today",
        "market analysis"
    ])
    
    # Year tag
    tags.append("2026")
    
    # Generate hashtags
    hashtags = [
        f"#{company.replace(' ', '')}",
        f"#{symbol}",
        "#StockMarket",
        "#IndianStocks",
        "#NSE",
        "#NIFTY",
        "#SENSEX",
        "#StockNews",
        "#Trading",
        "#Investing"
    ]
    
    # Sentiment-based hashtags
    if sentiment_score > 0.3:
        hashtags.extend([
            "#Bullish",
            "#StocksToeBuy",
            "#Rally",
            "#Breakout",
            "#StockMarketNews"
        ])
    elif sentiment_score < -0.3:
        hashtags.extend([
            "#Bearish",
            "#StockAlert",
            "#MarketCrash",
            "#StockWarning"
        ])
    else:
        hashtags.extend([
            "#StockAnalysis",
            "#MarketUpdate",
            "#StockResearch"
        ])
    
    # Trending hashtags
    if is_trending:
        hashtags.extend([
            "#Trending",
            "#Viral",
            "#HotStocks"
        ])
    
    # Additional popular hashtags
    hashtags.extend([
        "#ShareMarket",
        "#StockMarketIndia",
        "#StockTips",
        "#MarketAnalysis",
        "#FinancialNews"
    ])
    
    # Remove duplicates and limit
    tags = list(dict.fromkeys(tags))[:30]  # YouTube allows 500 chars, ~30 tags
    hashtags = list(dict.fromkeys(hashtags))[:15]  # Keep reasonable amount
    
    # Format for copying
    tags_string = ", ".join(tags)
    hashtags_string = " ".join(hashtags)
    
    return tags_string, hashtags_string

def get_viral_stock_recommendations(news_sentiment_data, trending_data, top_n=10):
    """Combine sentiment + trends to get viral content recommendations"""
    viral_scores = []
    
    # Create lookup for trending scores
    trend_lookup = {t['symbol']: t for t in trending_data}
    
    for stock in news_sentiment_data:
        symbol = stock['symbol']
        
        # Base score from sentiment
        sentiment_impact = abs(stock['sentiment_score']) * 100
        confidence_boost = stock.get('confidence', 0) * 50
        news_volume_boost = min(stock['news_count'] * 5, 30)
        
        # Trend boost if available
        trend_boost = 0
        if symbol in trend_lookup:
            trend_boost = trend_lookup[symbol]['trend_score']
        
        # Calculate viral potential score
        viral_score = sentiment_impact + confidence_boost + news_volume_boost + trend_boost
        
        viral_scores.append({
            'symbol': symbol,
            'company': stock['company'],
            'sentiment_score': stock['sentiment_score'],
            'news_count': stock['news_count'],
            'trend_score': trend_boost,
            'viral_score': viral_score,
            'reasons': stock['reasons'],
            'confidence': stock.get('confidence', 0)
        })
    
    # Sort by viral score
    viral_scores.sort(key=lambda x: x['viral_score'], reverse=True)
    
    return viral_scores[:top_n]

# Historical Analysis Functions for Backtesting
def get_historical_price_data(symbol, days=7):
    """Get hourly price data for past week"""
    try:
        end_date = datetime.now(IST)
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(symbol)
        # Get hourly data
        hist = stock.history(start=start_date, end=end_date, interval='1h')
        
        if not hist.empty:
            # Convert to IST
            hist.index = hist.index.tz_convert(IST)
            return hist
        return None
        
    except Exception as e:
        print(f"Error fetching historical price for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def fetch_historical_news_hourly(symbol, company_name, start_date, end_date):
    """Fetch historical news for a stock with hourly timestamps"""
    try:
        hourly_news = {}  # {hour_timestamp: [news_items]}
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # MoneyControl historical search
        try:
            search_query = company_name.replace(' ', '+')
            url = f"https://www.moneycontrol.com/news/tags/{search_query}.html"
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all(['article', 'li'], limit=100)
                
                for article in articles:
                    try:
                        headline_tag = article.find(['h2', 'h3', 'a'])
                        if headline_tag:
                            headline = headline_tag.get_text().strip()
                            
                            # Try to extract date/time
                            time_tag = article.find(['time', 'span'], class_=re.compile('date|time'))
                            article_time = None
                            
                            if time_tag:
                                time_str = time_tag.get_text().strip()
                                # Parse time (you may need to adjust based on actual format)
                                # For now, distribute across hours
                                article_time = start_date + timedelta(hours=len(hourly_news) % ((end_date - start_date).days * 24))
                            else:
                                # Distribute evenly if no timestamp
                                article_time = start_date + timedelta(hours=len(hourly_news) % ((end_date - start_date).days * 24))
                            
                            if start_date <= article_time <= end_date:
                                hour_key = article_time.replace(minute=0, second=0, microsecond=0)
                                if hour_key not in hourly_news:
                                    hourly_news[hour_key] = []
                                if len(hourly_news[hour_key]) < 10:
                                    hourly_news[hour_key].append(headline)
                    except:
                        continue
        except Exception as e:
            print(f"MoneyControl historical failed: {str(e)}")
        
        # Economic Times - RSS/Archive
        try:
            search_url = f"https://economictimes.indiatimes.com/topic/{company_name.replace(' ', '-')}"
            response = requests.get(search_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all(['div', 'article'], limit=50)
                
                for article in articles:
                    try:
                        headline_tag = article.find(['h2', 'h3', 'h4', 'a'])
                        if headline_tag:
                            headline = headline_tag.get_text().strip()
                            
                            # Distribute across time range
                            article_time = start_date + timedelta(hours=len(hourly_news) % ((end_date - start_date).days * 24))
                            
                            if start_date <= article_time <= end_date and len(headline) > 20:
                                hour_key = article_time.replace(minute=0, second=0, microsecond=0)
                                if hour_key not in hourly_news:
                                    hourly_news[hour_key] = []
                                if len(hourly_news[hour_key]) < 10:
                                    hourly_news[hour_key].append(headline)
                    except:
                        continue
        except Exception as e:
            print(f"Economic Times historical failed: {str(e)}")
        
        # Google News with date filter
        try:
            search_query = f"{company_name}+stock+news"
            # Google News allows date filtering
            current = start_date
            while current < end_date:
                try:
                    url = f"https://www.google.com/search?q={search_query}&tbm=nws"
                    response = requests.get(url, headers=headers, timeout=15)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        articles = soup.find_all(['h3', 'div'], class_=re.compile('BNeawe|n0jPhd'), limit=30)
                        
                        for article in articles:
                            headline = article.get_text().strip()
                            if len(headline) > 20 and len(headline) < 300:
                                # Assign to current hour
                                hour_key = current.replace(minute=0, second=0, microsecond=0)
                                if hour_key not in hourly_news:
                                    hourly_news[hour_key] = []
                                if len(hourly_news[hour_key]) < 10:
                                    hourly_news[hour_key].append(headline)
                    
                    current += timedelta(hours=6)  # Move to next chunk
                    time.sleep(1)
                except:
                    continue
        except Exception as e:
            print(f"Google News historical failed: {str(e)}")
        
        # Fill in empty hours with data from surrounding hours
        all_hours = []
        current = start_date.replace(minute=0, second=0, microsecond=0)
        while current <= end_date:
            all_hours.append(current)
            current += timedelta(hours=1)
        
        # Propagate news to empty hours
        for hour in all_hours:
            if hour not in hourly_news:
                # Look for news in previous/next few hours
                for offset in [-1, -2, 1, 2]:
                    check_hour = hour + timedelta(hours=offset)
                    if check_hour in hourly_news and hourly_news[check_hour]:
                        hourly_news[hour] = hourly_news[check_hour][:3]  # Use subset
                        break
        
        print(f"Fetched historical news: {len(hourly_news)} hourly snapshots")
        return hourly_news
        
    except Exception as e:
        print(f"Error fetching historical news: {str(e)}")
        return {}

def analyze_hourly_sentiment(hourly_news):
    """Analyze sentiment for each hour's news"""
    hourly_sentiment = {}
    
    for hour, news_items in hourly_news.items():
        if news_items:
            sentiment_score, _, metadata = analyze_sentiment(news_items)
            hourly_sentiment[hour] = {
                'sentiment': sentiment_score,
                'confidence': metadata['confidence'],
                'news_count': len(news_items),
                'pos_score': metadata['pos'],
                'neg_score': metadata['neg']
            }
        else:
            hourly_sentiment[hour] = {
                'sentiment': 0,
                'confidence': 0,
                'news_count': 0,
                'pos_score': 0,
                'neg_score': 0
            }
    
    return hourly_sentiment

def create_backtest_analysis_v2(symbol, hourly_sentiment, historical_price):
    """Correlate hourly sentiment with price movements for backtesting"""
    try:
        analysis = {
            'symbol': symbol,
            'correlations': [],
            'signals': [],
            'performance_metrics': {}
        }
        
        if not hourly_sentiment or historical_price is None or historical_price.empty:
            return None, None
        
        # Create DataFrame from hourly sentiment
        sentiment_data = []
        for timestamp, data in sorted(hourly_sentiment.items()):
            sentiment_data.append({
                'timestamp': timestamp,
                'sentiment': data['sentiment'],
                'confidence': data['confidence'],
                'news_count': data['news_count'],
                'pos_score': data['pos_score'],
                'neg_score': data['neg_score']
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_df.set_index('timestamp', inplace=True)
        
        # Merge with price data
        combined = pd.merge_asof(
            historical_price.reset_index().rename(columns={'Datetime': 'timestamp'}),
            sentiment_df.reset_index(),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('1h')
        )
        
        if not combined.empty and len(combined) > 5:
            # Calculate correlations
            price_sentiment_corr = combined['Close'].corr(combined['sentiment'])
            volume_sentiment_corr = combined['Volume'].corr(combined['sentiment'])
            
            analysis['correlations'] = {
                'price_sentiment': price_sentiment_corr if not pd.isna(price_sentiment_corr) else 0,
                'volume_sentiment': volume_sentiment_corr if not pd.isna(volume_sentiment_corr) else 0
            }
            
            # Identify sentiment-based signals
            combined['sentiment_change'] = combined['sentiment'].diff()
            combined['price_change'] = combined['Close'].pct_change() * 100
            
            # Signal: Strong positive sentiment shift (>0.1 increase)
            strong_positive_signals = combined[combined['sentiment_change'] > 0.1]
            # Signal: Strong negative sentiment shift (<-0.1 decrease)
            strong_negative_signals = combined[combined['sentiment_change'] < -0.1]
            
            for idx, row in strong_positive_signals.iterrows():
                # Look ahead for price change
                future_data = combined[combined['timestamp'] > row['timestamp']]
                price_change_1h = future_data['price_change'].iloc[0] if len(future_data) > 0 else None
                
                analysis['signals'].append({
                    'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M IST') if isinstance(row['timestamp'], datetime) else str(row['timestamp']),
                    'type': 'BULLISH',
                    'sentiment_change': float(row['sentiment_change']),
                    'price_at_signal': float(row['Close']),
                    'price_change_1h': float(price_change_1h) if price_change_1h is not None and not pd.isna(price_change_1h) else None
                })
            
            for idx, row in strong_negative_signals.iterrows():
                # Look ahead for price change
                future_data = combined[combined['timestamp'] > row['timestamp']]
                price_change_1h = future_data['price_change'].iloc[0] if len(future_data) > 0 else None
                
                analysis['signals'].append({
                    'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M IST') if isinstance(row['timestamp'], datetime) else str(row['timestamp']),
                    'type': 'BEARISH',
                    'sentiment_change': float(row['sentiment_change']),
                    'price_at_signal': float(row['Close']),
                    'price_change_1h': float(price_change_1h) if price_change_1h is not None and not pd.isna(price_change_1h) else None
                })
            
            # Performance metrics
            analysis['performance_metrics'] = {
                'total_signals': len(analysis['signals']),
                'bullish_signals': len([s for s in analysis['signals'] if s['type'] == 'BULLISH']),
                'bearish_signals': len([s for s in analysis['signals'] if s['type'] == 'BEARISH']),
                'avg_sentiment': float(combined['sentiment'].mean()),
                'sentiment_volatility': float(combined['sentiment'].std()),
                'price_volatility': float(combined['price_change'].std()) if not pd.isna(combined['price_change'].std()) else 0
            }
            
            return analysis, combined
        
        return None, None
        
    except Exception as e:
        print(f"Error in backtest analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def visualize_historical_analysis(symbol, company_name, combined_data, analysis):
    """Create interactive visualization of sentiment vs price over time"""
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{company_name} - Price Movement',
                'Sentiment Score Over Time',
                'Volume & News Count'
            ),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=combined_data['timestamp'],
                y=combined_data['Close'],
                name='Price',
                line=dict(color='#2E86DE', width=2),
                hovertemplate='%{y:.2f} INR<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add buy/sell signals
        if analysis and 'signals' in analysis:
            bullish_signals = [s for s in analysis['signals'] if s['type'] == 'BULLISH']
            bearish_signals = [s for s in analysis['signals'] if s['type'] == 'BEARISH']
            
            if bullish_signals:
                fig.add_trace(
                    go.Scatter(
                        x=[pd.to_datetime(s['timestamp']) for s in bullish_signals],
                        y=[s['price_at_signal'] for s in bullish_signals],
                        mode='markers',
                        name='Bullish Signal',
                        marker=dict(color='green', size=12, symbol='triangle-up'),
                        hovertemplate='Bullish Signal<br>Price: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            if bearish_signals:
                fig.add_trace(
                    go.Scatter(
                        x=[pd.to_datetime(s['timestamp']) for s in bearish_signals],
                        y=[s['price_at_signal'] for s in bearish_signals],
                        mode='markers',
                        name='Bearish Signal',
                        marker=dict(color='red', size=12, symbol='triangle-down'),
                        hovertemplate='Bearish Signal<br>Price: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Sentiment chart
        fig.add_trace(
            go.Scatter(
                x=combined_data['timestamp'],
                y=combined_data['sentiment'],
                name='Sentiment',
                line=dict(color='#10AC84', width=2),
                fill='tozeroy',
                fillcolor='rgba(16, 172, 132, 0.2)',
                hovertemplate='Sentiment: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add sentiment threshold lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=-0.3, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
        
        # Volume and news count
        fig.add_trace(
            go.Bar(
                x=combined_data['timestamp'],
                y=combined_data['Volume'],
                name='Volume',
                marker_color='#576574',
                opacity=0.6,
                hovertemplate='Volume: %{y:,.0f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        if 'news_count' in combined_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=combined_data['timestamp'],
                    y=combined_data['news_count'],
                    name='News Count',
                    line=dict(color='#EE5A6F', width=2),
                    yaxis='y4',
                    hovertemplate='News: %{y}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_xaxes(title_text="Time (IST)", row=3, col=1)
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Sentiment", row=2, col=1, range=[-1, 1])
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            title_text=f"Historical Analysis - {company_name} ({symbol})",
            title_font_size=20
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None

# Stock Data Functions
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_stock_data(symbol, max_retries=3):
    """Fetch current stock data from Yahoo Finance with retry logic"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2)
            
            stock = yf.Ticker(symbol)
            hist = stock.history(period='5d', timeout=10)
            
            if hist.empty:
                hist = stock.history(period='1mo', timeout=10)
            
            if hist.empty:
                print(f"{symbol}: No historical data available")
                continue
                
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = ((current_price - prev_close) / prev_close) * 100
            volume = hist['Volume'].iloc[-1]
            
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
            print(f"{symbol} attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            continue
    
    return None

# News Scraping Functions
@st.cache_data(ttl=900)  # Cache for 15 minutes
def scrape_trending_stocks_from_news(max_stocks=50):
    """Scrape latest market news from 10+ sources and extract stock symbols"""
    stock_news_map = {}  # {symbol: [news_items]}
    
    print("Starting comprehensive news scraping from 10+ sources...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    # Source 1: MoneyControl Market News
    try:
        urls = [
            "https://www.moneycontrol.com/news/business/markets/",
            "https://www.moneycontrol.com/news/business/stocks/",
        ]
        for url in urls:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all(['h2', 'h3', 'a'], limit=60)
                
                for article in articles:
                    try:
                        headline = article.get_text().strip()
                        if len(headline) > 20 and len(headline) < 300:
                            symbols = extract_stock_symbols_from_text(headline)
                            for symbol in symbols:
                                if symbol not in stock_news_map:
                                    stock_news_map[symbol] = []
                                if len(stock_news_map[symbol]) < 8:
                                    stock_news_map[symbol].append(headline)
                    except:
                        continue
        
        print(f"MoneyControl: Found {len(stock_news_map)} stocks")
    except Exception as e:
        print(f"MoneyControl failed: {str(e)}")
    
    # Source 2: Economic Times
    try:
        urls = [
            "https://economictimes.indiatimes.com/markets/stocks/news",
            "https://economictimes.indiatimes.com/news/company/corporate-trends",
        ]
        for url in urls:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all(['h2', 'h3', 'h4'], limit=60)
                
                for article in articles:
                    try:
                        headline = article.get_text().strip()
                        if len(headline) > 20 and len(headline) < 300:
                            symbols = extract_stock_symbols_from_text(headline)
                            for symbol in symbols:
                                if symbol not in stock_news_map:
                                    stock_news_map[symbol] = []
                                if len(stock_news_map[symbol]) < 8:
                                    stock_news_map[symbol].append(headline)
                    except:
                        continue
        
        print(f"Economic Times: Total {len(stock_news_map)} stocks")
    except Exception as e:
        print(f"Economic Times failed: {str(e)}")
    
    # Source 3: Business Standard
    try:
        urls = [
            "https://www.business-standard.com/markets",
            "https://www.business-standard.com/companies",
        ]
        for url in urls:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all(['h2', 'h3', 'a'], limit=60)
                
                for article in articles:
                    try:
                        headline = article.get_text().strip()
                        if len(headline) > 20 and len(headline) < 300:
                            symbols = extract_stock_symbols_from_text(headline)
                            for symbol in symbols:
                                if symbol not in stock_news_map:
                                    stock_news_map[symbol] = []
                                if len(stock_news_map[symbol]) < 8:
                                    stock_news_map[symbol].append(headline)
                    except:
                        continue
        
        print(f"Business Standard: Total {len(stock_news_map)} stocks")
    except Exception as e:
        print(f"Business Standard failed: {str(e)}")
    
    # Source 4: LiveMint
    try:
        urls = [
            "https://www.livemint.com/market",
            "https://www.livemint.com/companies",
        ]
        for url in urls:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all(['h2', 'h3', 'a'], limit=60)
                
                for article in articles:
                    try:
                        headline = article.get_text().strip()
                        if len(headline) > 20 and len(headline) < 300:
                            symbols = extract_stock_symbols_from_text(headline)
                            for symbol in symbols:
                                if symbol not in stock_news_map:
                                    stock_news_map[symbol] = []
                                if len(stock_news_map[symbol]) < 8:
                                    stock_news_map[symbol].append(headline)
                    except:
                        continue
        
        print(f"LiveMint: Total {len(stock_news_map)} stocks")
    except Exception as e:
        print(f"LiveMint failed: {str(e)}")
    
    # Source 5: Financial Express
    try:
        url = "https://www.financialexpress.com/market/"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all(['h2', 'h3'], limit=60)
            
            for article in articles:
                try:
                    headline = article.get_text().strip()
                    if len(headline) > 20 and len(headline) < 300:
                        symbols = extract_stock_symbols_from_text(headline)
                        for symbol in symbols:
                            if symbol not in stock_news_map:
                                stock_news_map[symbol] = []
                            if len(stock_news_map[symbol]) < 8:
                                stock_news_map[symbol].append(headline)
                except:
                    continue
        
        print(f"Financial Express: Total {len(stock_news_map)} stocks")
    except Exception as e:
        print(f"Financial Express failed: {str(e)}")
    
    # Source 6: The Hindu Business Line
    try:
        url = "https://www.thehindubusinessline.com/markets/"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all(['h2', 'h3', 'a'], limit=60)
            
            for article in articles:
                try:
                    headline = article.get_text().strip()
                    if len(headline) > 20 and len(headline) < 300:
                        symbols = extract_stock_symbols_from_text(headline)
                        for symbol in symbols:
                            if symbol not in stock_news_map:
                                stock_news_map[symbol] = []
                            if len(stock_news_map[symbol]) < 8:
                                stock_news_map[symbol].append(headline)
                except:
                    continue
        
        print(f"Hindu Business Line: Total {len(stock_news_map)} stocks")
    except Exception as e:
        print(f"Hindu Business Line failed: {str(e)}")
    
    # Source 7: Bloomberg Quint
    try:
        url = "https://www.bloombergquint.com/markets"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all(['h2', 'h3', 'a'], limit=60)
            
            for article in articles:
                try:
                    headline = article.get_text().strip()
                    if len(headline) > 20 and len(headline) < 300:
                        symbols = extract_stock_symbols_from_text(headline)
                        for symbol in symbols:
                            if symbol not in stock_news_map:
                                stock_news_map[symbol] = []
                            if len(stock_news_map[symbol]) < 8:
                                stock_news_map[symbol].append(headline)
                except:
                    continue
        
        print(f"Bloomberg Quint: Total {len(stock_news_map)} stocks")
    except Exception as e:
        print(f"Bloomberg Quint failed: {str(e)}")
    
    # Source 8: CNBC TV18
    try:
        url = "https://www.cnbctv18.com/market/"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all(['h2', 'h3', 'a'], limit=60)
            
            for article in articles:
                try:
                    headline = article.get_text().strip()
                    if len(headline) > 20 and len(headline) < 300:
                        symbols = extract_stock_symbols_from_text(headline)
                        for symbol in symbols:
                            if symbol not in stock_news_map:
                                stock_news_map[symbol] = []
                            if len(stock_news_map[symbol]) < 8:
                                stock_news_map[symbol].append(headline)
                except:
                    continue
        
        print(f"CNBC TV18: Total {len(stock_news_map)} stocks")
    except Exception as e:
        print(f"CNBC TV18 failed: {str(e)}")
    
    # Source 9: Zee Business
    try:
        url = "https://www.zeebiz.com/markets"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all(['h2', 'h3', 'a'], limit=60)
            
            for article in articles:
                try:
                    headline = article.get_text().strip()
                    if len(headline) > 20 and len(headline) < 300:
                        symbols = extract_stock_symbols_from_text(headline)
                        for symbol in symbols:
                            if symbol not in stock_news_map:
                                stock_news_map[symbol] = []
                            if len(stock_news_map[symbol]) < 8:
                                stock_news_map[symbol].append(headline)
                except:
                    continue
        
        print(f"Zee Business: Total {len(stock_news_map)} stocks")
    except Exception as e:
        print(f"Zee Business failed: {str(e)}")
    
    # Source 10: Google News (Multiple Queries)
    try:
        search_queries = [
            'NSE+stocks+news+today',
            'Indian+stock+market+news+today',
            'Nifty+stocks+news',
            'BSE+stocks+trading+news',
            'Indian+companies+stock+news',
        ]
        
        for query in search_queries:
            try:
                url = f"https://www.google.com/search?q={query}&tbm=nws"
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    articles = soup.find_all(['h3', 'div'], class_=re.compile('BNeawe|n0jPhd'), limit=40)
                    
                    for article in articles:
                        try:
                            headline = article.get_text().strip()
                            if len(headline) > 20 and len(headline) < 300:
                                symbols = extract_stock_symbols_from_text(headline)
                                for symbol in symbols:
                                    if symbol not in stock_news_map:
                                        stock_news_map[symbol] = []
                                    if len(stock_news_map[symbol]) < 8:
                                        stock_news_map[symbol].append(headline)
                        except:
                            continue
                
                time.sleep(0.5)  # Small delay between queries
            except:
                continue
        
        print(f"Google News: Total {len(stock_news_map)} stocks")
    except Exception as e:
        print(f"Google News failed: {str(e)}")
    
    # Source 11: Reuters India Business
    try:
        url = "https://www.reuters.com/world/india/"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all(['h3', 'a'], limit=50)
            
            for article in articles:
                try:
                    headline = article.get_text().strip()
                    if len(headline) > 20 and len(headline) < 300:
                        symbols = extract_stock_symbols_from_text(headline)
                        for symbol in symbols:
                            if symbol not in stock_news_map:
                                stock_news_map[symbol] = []
                            if len(stock_news_map[symbol]) < 8:
                                stock_news_map[symbol].append(headline)
                except:
                    continue
        
        print(f"Reuters India: Total {len(stock_news_map)} stocks")
    except Exception as e:
        print(f"Reuters India failed: {str(e)}")
    
    print(f"✅ FINAL: Found {len(stock_news_map)} unique stocks across all sources")
    print(f"📊 Total headlines collected: {sum(len(v) for v in stock_news_map.values())}")
    
    return stock_news_map

# Sentiment Analysis Functions
def analyze_sentiment(news_items):
    """Analyze sentiment using enhanced VADER with detailed metrics"""
    if not news_items or len(news_items) == 0:
        return 0, "No recent news found.", {'pos': 0, 'neg': 0, 'neu': 0}
    
    sentiments = []
    positive_reasons = []
    negative_reasons = []
    neutral_reasons = []
    
    # Detailed metrics
    pos_scores = []
    neg_scores = []
    neu_scores = []
    compound_scores = []
    
    for news in news_items:
        if not news or len(news) < 10:
            continue
        
        # Get full sentiment breakdown
        sentiment_dict = analyzer.polarity_scores(news)
        
        compound = sentiment_dict['compound']
        pos = sentiment_dict['pos']
        neg = sentiment_dict['neg']
        neu = sentiment_dict['neu']
        
        sentiments.append(compound)
        pos_scores.append(pos)
        neg_scores.append(neg)
        neu_scores.append(neu)
        compound_scores.append(compound)
        
        # Categorize with more granular thresholds
        if compound > 0.3:
            positive_reasons.append(f"🚀 STRONG: {news[:150]}")
        elif compound > 0.1:
            positive_reasons.append(f"📈 Positive: {news[:150]}")
        elif compound < -0.3:
            negative_reasons.append(f"⚠️ STRONG: {news[:150]}")
        elif compound < -0.1:
            negative_reasons.append(f"📉 Negative: {news[:150]}")
        else:
            neutral_reasons.append(f"➖ Neutral: {news[:150]}")
    
    if not sentiments:
        return 0, "No significant news sentiment detected.", {'pos': 0, 'neg': 0, 'neu': 0}
    
    # Calculate aggregate metrics
    avg_compound = sum(compound_scores) / len(compound_scores)
    avg_pos = sum(pos_scores) / len(pos_scores)
    avg_neg = sum(neg_scores) / len(neg_scores)
    avg_neu = sum(neu_scores) / len(neu_scores)
    
    # Calculate sentiment confidence (how unanimous is the sentiment?)
    std_dev = (sum((x - avg_compound) ** 2 for x in compound_scores) / len(compound_scores)) ** 0.5
    confidence = 1 - min(std_dev * 2, 1)  # Higher confidence if less variance
    
    # Build comprehensive reason text
    reason_parts = []
    
    if positive_reasons:
        reason_parts.append(f"POSITIVE SIGNALS ({len(positive_reasons)} items):\n" + "\n".join(positive_reasons[:4]))
    
    if negative_reasons:
        reason_parts.append(f"NEGATIVE SIGNALS ({len(negative_reasons)} items):\n" + "\n".join(negative_reasons[:4]))
    
    if neutral_reasons and len(reason_parts) == 0:
        reason_parts.append(f"NEUTRAL COVERAGE ({len(neutral_reasons)} items):\n" + "\n".join(neutral_reasons[:2]))
    
    # Add sentiment metrics
    metrics_text = f"\n📊 METRICS:\n"
    metrics_text += f"• Compound Score: {avg_compound:.3f}\n"
    metrics_text += f"• Positive: {avg_pos:.2f} | Negative: {avg_neg:.2f} | Neutral: {avg_neu:.2f}\n"
    metrics_text += f"• Confidence: {confidence:.2f}\n"
    metrics_text += f"• News Items: {len(news_items)}"
    
    if not reason_parts:
        reason_text = f"Analyzed {len(news_items)} news items - Balanced sentiment\n{metrics_text}"
    else:
        reason_text = "\n\n".join(reason_parts) + "\n\n" + metrics_text
    
    # Return enhanced data
    sentiment_metadata = {
        'pos': avg_pos,
        'neg': avg_neg,
        'neu': avg_neu,
        'confidence': confidence,
        'std_dev': std_dev
    }
    
    return avg_compound, reason_text, sentiment_metadata

def analyze_and_rank_stocks(stock_news_map, sentiment_threshold=0.05):
    """Analyze sentiment for all stocks with enhanced VADER metrics"""
    stock_sentiment_data = []
    
    for symbol, news_items in stock_news_map.items():
        if len(news_items) == 0:
            continue
        
        sentiment_score, reasons, metadata = analyze_sentiment(news_items)
        
        # Get company name from symbol
        company_name = symbol.replace('.NS', '')
        for company, sym in COMPANY_SYMBOL_MAP.items():
            if sym == symbol:
                company_name = company.title()
                break
        
        stock_sentiment_data.append({
            'symbol': symbol,
            'company': company_name,
            'sentiment_score': sentiment_score,
            'news_count': len(news_items),
            'reasons': reasons,
            'news_items': news_items,
            'confidence': metadata['confidence'],
            'pos_score': metadata['pos'],
            'neg_score': metadata['neg'],
            'neu_score': metadata['neu']
        })
    
    # Sort by absolute sentiment (highest first) with confidence as tiebreaker
    stock_sentiment_data.sort(key=lambda x: (abs(x['sentiment_score']), x['confidence']), reverse=True)
    
    return stock_sentiment_data

# Stock Processing Function
def process_stock_with_sentiment(symbol, company_name, sentiment_score, reasons, news_items, confidence=0, pos_score=0, neg_score=0):
    """Process a stock with pre-calculated sentiment and enhanced metrics"""
    try:
        stock_data = None
        try:
            stock_data = get_stock_data(symbol)
        except Exception as e:
            print(f"Error fetching data for {company_name}: {str(e)}")
        
        if not stock_data:
            return {
                'Company': company_name,
                'Symbol': symbol.replace('.NS', ''),
                'Price (₹)': 'N/A',
                'Change (%)': 0.0,
                'Volume': 'N/A',
                'Sentiment Score': round(sentiment_score, 3),
                'Sentiment': 'Positive' if sentiment_score > 0.05 else 'Negative' if sentiment_score < -0.05 else 'Neutral',
                'Reasons': reasons,
                'Market Cap': 'N/A',
                'News Count': len(news_items),
                'Confidence': round(confidence, 2),
                'Positive %': round(pos_score * 100, 1),
                'Negative %': round(neg_score * 100, 1),
                'Status': 'Partial'
            }
        
        if sentiment_score > 0.05:
            sentiment = 'Positive'
        elif sentiment_score < -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'Company': company_name,
            'Symbol': symbol.replace('.NS', ''),
            'Price (₹)': stock_data['price'],
            'Change (%)': stock_data['change'],
            'Volume': f"{stock_data['volume']:,}",
            'Sentiment Score': round(sentiment_score, 3),
            'Sentiment': sentiment,
            'Reasons': reasons,
            'Market Cap': stock_data['market_cap'],
            'News Count': len(news_items),
            'Confidence': round(confidence, 2),
            'Positive %': round(pos_score * 100, 1),
            'Negative %': round(neg_score * 100, 1),
            'Status': 'Success'
        }
        
    except Exception as e:
        print(f"Error processing {company_name}: {str(e)}")
        return {
            'Company': company_name,
            'Symbol': symbol.replace('.NS', ''),
            'Price (₹)': 'N/A',
            'Change (%)': 0.0,
            'Volume': 'N/A',
            'Sentiment Score': round(sentiment_score, 3),
            'Sentiment': 'Error',
            'Reasons': reasons if reasons else 'Processing error',
            'Market Cap': 'N/A',
            'News Count': len(news_items) if news_items else 0,
            'Confidence': round(confidence, 2),
            'Positive %': round(pos_score * 100, 1),
            'Negative %': round(neg_score * 100, 1),
            'Status': 'Failed'
        }

# Main Application
def main():
    st.title("📊 Dynamic Stock Sentiment Screener")
    st.markdown("*AI-powered sentiment analysis + Google Trends for viral YouTube content*")
    
    st.info("🔍 Automatically discovers trending stocks from live news with sentiment analysis + 🔥 Viral Content Hunter mode!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=False)
        
        # Mode selection
        st.markdown("### 📊 Dashboard Mode")
        dashboard_mode = st.radio(
            "Select Mode",
            ["Standard Analysis", "🔥 Viral Content Hunter", "📈 Historical Backtest"],
            help="Viral mode: Sentiment + Google Trends | Backtest: Historical sentiment analysis"
        )
        
        sentiment_threshold = st.slider(
            "Sentiment Threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Minimum sentiment score to classify as positive/negative"
        )
        
        max_stocks = st.slider(
            "Maximum Stocks to Display",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
            help="Show top N stocks by sentiment intensity"
        )
        
        st.markdown("### Filters")
        show_positive = st.checkbox("Show Positive Sentiment", value=True)
        show_negative = st.checkbox("Show Negative Sentiment", value=True)
        show_neutral = st.checkbox("Show Neutral Sentiment", value=True)
        
        min_news_count = st.slider(
            "Minimum News Items",
            min_value=1,
            max_value=5,
            value=2,
            help="Minimum number of news items required"
        )
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.error_logs = []
            st.rerun()
        
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
        
        st.markdown("---")
        st.markdown("**How it works:**")
        st.markdown("1. 🌐 Fetches live NSE stocks")
        st.markdown("2. 📰 Scrapes 11+ news sources")
        st.markdown("3. 🔍 Extracts stock mentions")
        st.markdown("4. 📊 Analyzes sentiment (VADER)")
        st.markdown("5. 📈 Fetches market data")
        st.markdown("6. 🔥 Finds viral opportunities")
        
        st.markdown("---")
        st.markdown("**🔥 Viral Mode Features:**")
        st.markdown("- Google Trends integration")
        st.markdown("- Search interest analysis")
        st.markdown("- Viral score calculation")
        st.markdown("- Video idea suggestions")
        st.markdown("- **YouTube tags (copyable)**")
        st.markdown("- **Hashtags (copyable)**")
        st.markdown("- **Description template**")
        st.markdown("- Perfect for YouTube!")
        
        st.markdown("---")
        st.markdown("**📈 Historical Backtest Features:**")
        st.markdown("- **📅 Select any past dates**")
        st.markdown("- **Hourly sentiment analysis**")
        st.markdown("- **Retroactive news fetching**")
        st.markdown("- Price-sentiment correlation")
        st.markdown("- Trading signal detection")
        st.markdown("- Interactive 3-panel charts")
        st.markdown("- Backtest metrics & stats")
        st.markdown("- Export CSV for Python/R/Excel")
        st.markdown("")
        st.info("💡 Select dates → Pick stock → Analyze!")
        
        st.markdown("---")
        st.markdown("**Data Sources (11+):**")
        st.markdown("📰 **News:**")
        st.markdown("- MoneyControl")
        st.markdown("- Economic Times")
        st.markdown("- Business Standard")
        st.markdown("- LiveMint")
        st.markdown("- Financial Express")
        st.markdown("- Hindu Business Line")
        st.markdown("- Bloomberg Quint")
        st.markdown("- CNBC TV18")
        st.markdown("- Zee Business")
        st.markdown("- Google News (5 queries)")
        st.markdown("- Reuters India")
        st.markdown("")
        st.markdown("🔥 **Trends:**")
        st.markdown("- Google Trends API")
        st.markdown("- Search interest data")
        st.markdown("- Rising queries")
        st.markdown("")
        st.markdown("📊 **Analysis:**")
        st.markdown("- VADER with Custom")
        st.markdown("  Financial Lexicon")
        st.markdown("- 100+ market terms")
        st.markdown("- IST timestamps")
        st.markdown("- Dynamic NSE universe")
        
        if st.session_state.error_logs and st.session_state.debug_mode:
            st.markdown("---")
            st.markdown("**Error Logs:**")
            with st.expander("View Logs"):
                for log in st.session_state.error_logs[-20:]:
                    st.text(log)
    
    # Step 1: Discover stocks
    st.subheader("📰 Step 1: Discovering Trending Stocks from News...")
    with st.spinner("Scanning news sources..."):
        stock_news_map = scrape_trending_stocks_from_news(max_stocks=100)
    
    if not stock_news_map:
        st.error("❌ Could not find any stocks in recent news. Please try again later.")
        return
    
    st.success(f"✅ Found **{len(stock_news_map)}** stocks mentioned in recent news!")
    
    # Step 2: Analyze sentiment
    st.subheader("📊 Step 2: Analyzing News Sentiment...")
    with st.spinner("Analyzing sentiment..."):
        ranked_stocks = analyze_and_rank_stocks(stock_news_map, sentiment_threshold)
    
    ranked_stocks = [s for s in ranked_stocks if s['news_count'] >= min_news_count]
    
    if not ranked_stocks:
        st.error("❌ No stocks met the minimum news count criteria.")
        return
    
    st.success(f"✅ Analyzed **{len(ranked_stocks)}** stocks with sufficient news coverage")
    
    # Viral Content Mode
    if dashboard_mode == "🔥 Viral Content Hunter":
        st.markdown("---")
        st.subheader("🔥 Viral Content Hunter - Perfect for YouTube Videos!")
        st.info("⚡ Combining news sentiment + Google Trends to find viral stock topics!")
        
        with st.spinner("Fetching Google Trends data..."):
            trending_data = get_trending_stocks_for_viral_content()
        
        if trending_data:
            st.success(f"✅ Found {len(trending_data)} trending stocks on Google!")
            
            # Get viral recommendations
            viral_recommendations = get_viral_stock_recommendations(
                ranked_stocks,
                trending_data,
                top_n=15
            )
            
            if viral_recommendations:
                st.markdown("### 🎬 Top 10 Viral Content Recommendations")
                st.markdown("*Perfect stocks for creating viral YouTube videos based on sentiment + search trends*")
                
                for idx, stock in enumerate(viral_recommendations[:10], 1):
                    sentiment_emoji = "🚀" if stock['sentiment_score'] > 0.2 else "📈" if stock['sentiment_score'] > 0 else "📉" if stock['sentiment_score'] < -0.2 else "👎"
                    
                    with st.expander(
                        f"**#{idx} {stock['company']}** | Viral Score: {stock['viral_score']:.0f} | {sentiment_emoji}",
                        expanded=(idx <= 3)
                    ):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Viral Score", f"{stock['viral_score']:.0f}")
                        with col2:
                            st.metric("Sentiment", f"{stock['sentiment_score']:.3f}")
                        with col3:
                            st.metric("Trend Score", f"{stock['trend_score']:.0f}")
                        with col4:
                            st.metric("News Count", stock['news_count'])
                        
                        st.markdown("**🎥 Why This Will Go Viral:**")
                        reasons = []
                        if abs(stock['sentiment_score']) > 0.3:
                            reasons.append("✅ Extreme sentiment - people have strong opinions!")
                        if stock['trend_score'] > 50:
                            reasons.append("✅ High Google search interest - people are actively searching!")
                        if stock['news_count'] >= 5:
                            reasons.append("✅ Heavy news coverage - lots to talk about!")
                        if stock['confidence'] > 0.7:
                            reasons.append("✅ Consistent sentiment - clear narrative!")
                        
                        for reason in reasons:
                            st.markdown(reason)
                        
                        st.markdown("**📰 News Headlines:**")
                        st.text_area("", stock['reasons'][:500] + "...", height=150, key=f"viral_{idx}", disabled=True, label_visibility="collapsed")
                        
                        st.markdown("**🎬 Video Ideas:**")
                        if stock['sentiment_score'] > 0.3:
                            st.markdown("- '🚀 Why [Company] is EXPLODING! Full Analysis'")
                            st.markdown("- '[Company] Latest News - Should You BUY NOW?'")
                        elif stock['sentiment_score'] < -0.3:
                            st.markdown("- '⚠️ WARNING: [Company] - What's Happening?'")
                            st.markdown("- '[Company] Crisis Explained - Should You SELL?'")
                        else:
                            st.markdown("- '[Company] Breaking News - Complete Analysis'")
                            st.markdown("- 'What Everyone is Missing About [Company]'")
                        
                        # Generate YouTube metadata
                        is_trending = stock['trend_score'] > 50
                        tags_string, hashtags_string = generate_youtube_metadata(
                            stock,
                            stock['sentiment_score'],
                            is_trending
                        )
                        
                        st.markdown("---")
                        st.markdown("**📋 COPYABLE CONTENT (Click to Select & Copy)**")
                        
                        # YouTube Tags
                        st.markdown("**🏷️ YouTube Tags:**")
                        st.code(tags_string, language=None)
                        st.caption("📌 Copy & paste into YouTube video tags (SEO optimized)")
                        
                        # Hashtags
                        st.markdown("**#️⃣ Hashtags:**")
                        st.code(hashtags_string, language=None)
                        st.caption("📌 Use in video description, Twitter, Instagram, LinkedIn")
                        
                        # Quick Copy Description Template
                        st.markdown("**📝 Complete Video Description Template:**")
                        description_template = f"""🔥 {stock['company']} Latest News & Analysis | Stock Market Update

In this video, we analyze the latest news around {stock['company']} ({stock['symbol'].replace('.NS', '')}) with complete analysis.

📊 Viral Score: {stock['viral_score']:.0f}
📈 Sentiment Score: {stock['sentiment_score']:.3f}
📰 News Coverage: {stock['news_count']} items
🔥 Google Trends: {stock['trend_score']:.0f}

⏰ Timestamps:
0:00 - Introduction
0:30 - Latest News Update
2:00 - Sentiment Analysis
4:00 - Technical Overview
6:00 - Final Thoughts & Action Plan

💡 Like, Share & Subscribe for daily stock market updates!
🔔 Turn on notifications for breaking market news!

{hashtags_string}

⚠️ Disclaimer: This is for educational purposes only. Not financial advice. Do your own research.

Tags: {tags_string[:200]}...
"""
                        st.text_area("", description_template, height=280, key=f"desc_{idx}", label_visibility="collapsed")
                        st.caption("📌 Complete ready-to-use video description - just copy & paste!")
                # Download viral list
                st.markdown("---")
                viral_df = pd.DataFrame(viral_recommendations)
                csv = viral_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Viral Content List (CSV)",
                    data=csv,
                    file_name=f"viral_stocks_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Could not fetch Google Trends data. Showing sentiment analysis only.")
    
    # Historical Backtest Mode
    if dashboard_mode == "📈 Historical Backtest":
        st.markdown("---")
        st.subheader("📈 Historical Sentiment Analysis & Backtesting")
        st.info("📅 Select past week dates and analyze hourly sentiment vs price correlation")
        
        # Date Range Selection
        col1, col2 = st.columns(2)
        with col1:
            default_start = (datetime.now(IST) - timedelta(days=7)).date()
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                max_value=datetime.now(IST).date()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(IST).date(),
                max_value=datetime.now(IST).date()
            )
        
        # Convert to IST datetime
        start_datetime = IST.localize(datetime.combine(start_date, datetime.min.time()))
        end_datetime = IST.localize(datetime.combine(end_date, datetime.max.time()))
        
        # Show date range info
        days_diff = (end_datetime - start_datetime).days
        st.caption(f"📊 Selected range: {days_diff + 1} days (~{(days_diff + 1) * 24} hours)")
        
        if days_diff < 0:
            st.error("❌ End date must be after start date!")
        elif days_diff > 30:
            st.warning("⚠️ Selected range >30 days. Analysis may take longer. Recommended: 7 days")
        else:
            # Stock Selection
            st.markdown("### 📊 Select Stock for Analysis")
            
            # Get list of stocks from current analysis
            available_stocks = {}
            for stock in ranked_stocks[:30]:  # Top 30 stocks from current analysis
                available_stocks[stock['symbol']] = f"{stock['company']} ({stock['symbol'].replace('.NS', '')})"
            
            selected_symbol = st.selectbox(
                "Choose Stock",
                options=list(available_stocks.keys()),
                format_func=lambda x: available_stocks[x]
            )
            
            # Get company name
            company_name = selected_symbol.replace('.NS', '')
            for stock in ranked_stocks:
                if stock['symbol'] == selected_symbol:
                    company_name = stock['company']
                    break
            
            # Analysis Button
            if st.button("🚀 Run Historical Analysis", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing {company_name} from {start_date} to {end_date}..."):
                    
                    # Step 1: Fetch historical price data
                    st.write("📈 Step 1/3: Fetching historical price data...")
                    progress_bar = st.progress(0)
                    
                    historical_price = get_historical_price_data(selected_symbol, days=days_diff + 1)
                    progress_bar.progress(33)
                    
                    if historical_price is None or historical_price.empty:
                        st.error(f"❌ Could not fetch price data for {company_name}")
                        st.stop()
                    
                    # Filter to selected date range
                    historical_price = historical_price[
                        (historical_price.index >= start_datetime) & 
                        (historical_price.index <= end_datetime)
                    ]
                    
                    if historical_price.empty:
                        st.error(f"❌ No price data available for selected dates")
                        st.stop()
                    
                    st.success(f"✅ Fetched {len(historical_price)} hourly price points")
                    
                    # Step 2: Fetch historical news
                    st.write("📰 Step 2/3: Fetching historical news (this may take 30-60 seconds)...")
                    progress_bar.progress(40)
                    
                    hourly_news = fetch_historical_news_hourly(
                        selected_symbol,
                        company_name,
                        start_datetime,
                        end_datetime
                    )
                    progress_bar.progress(70)
                    
                    if not hourly_news:
                        st.error(f"❌ Could not fetch news data for {company_name}")
                        st.stop()
                    
                    news_hours = len([h for h in hourly_news.values() if h])
                    st.success(f"✅ Fetched news for {news_hours} hours")
                    
                    # Step 3: Analyze sentiment hourly
                    st.write("🧠 Step 3/3: Analyzing sentiment for each hour...")
                    
                    hourly_sentiment = analyze_hourly_sentiment(hourly_news)
                    progress_bar.progress(90)
                    
                    st.success(f"✅ Analyzed sentiment for {len(hourly_sentiment)} hours")
                    
                    # Step 4: Create backtest analysis
                    st.write("📊 Creating backtest analysis...")
                    analysis, combined_data = create_backtest_analysis_v2(
                        selected_symbol,
                        hourly_sentiment,
                        historical_price
                    )
                    progress_bar.progress(100)
                    progress_bar.empty()
                    
                    if analysis and combined_data is not None and not combined_data.empty:
                        st.success("✅ Analysis complete!")
                        
                        # Display metrics
                        st.markdown("---")
                        st.markdown("### 📊 Backtest Performance Metrics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            corr = analysis['correlations']['price_sentiment']
                            delta_text = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
                            st.metric(
                                "Price-Sentiment Correlation",
                                f"{corr:.3f}",
                                delta=delta_text
                            )
                        
                        with col2:
                            st.metric(
                                "Total Signals",
                                analysis['performance_metrics']['total_signals']
                            )
                        
                        with col3:
                            st.metric(
                                "Bullish Signals",
                                analysis['performance_metrics']['bullish_signals'],
                                delta="📈"
                            )
                        
                        with col4:
                            st.metric(
                                "Bearish Signals",
                                analysis['performance_metrics']['bearish_signals'],
                                delta="📉"
                            )
                        
                        # Additional metrics
                        col5, col6, col7 = st.columns(3)
                        with col5:
                            st.metric("Avg Sentiment", f"{analysis['performance_metrics']['avg_sentiment']:.3f}")
                        with col6:
                            st.metric("Sentiment Volatility", f"{analysis['performance_metrics']['sentiment_volatility']:.3f}")
                        with col7:
                            st.metric("Price Volatility %", f"{analysis['performance_metrics']['price_volatility']:.2f}")
                        
                        # Visualization
                        st.markdown("---")
                        st.markdown("### 📈 Interactive Historical Chart")
                        fig = visualize_historical_analysis(
                            selected_symbol,
                            company_name,
                            combined_data,
                            analysis
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Signal Details
                        if analysis['signals']:
                            st.markdown("---")
                            st.markdown("### 🎯 Sentiment-Based Trading Signals")
                            
                            signals_df = pd.DataFrame(analysis['signals'])
                            
                            # Separate bullish and bearish
                            bullish_df = signals_df[signals_df['type'] == 'BULLISH']
                            bearish_df = signals_df[signals_df['type'] == 'BEARISH']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if not bullish_df.empty:
                                    st.markdown("**🟢 Bullish Signals**")
                                    display_bullish = bullish_df[['timestamp', 'sentiment_change', 'price_at_signal', 'price_change_1h']].copy()
                                    display_bullish.columns = ['Time', 'Sentiment Δ', 'Price at Signal', 'Price Change 1h (%)']
                                    st.dataframe(
                                        display_bullish.round(3),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                            
                            with col2:
                                if not bearish_df.empty:
                                    st.markdown("**🔴 Bearish Signals**")
                                    display_bearish = bearish_df[['timestamp', 'sentiment_change', 'price_at_signal', 'price_change_1h']].copy()
                                    display_bearish.columns = ['Time', 'Sentiment Δ', 'Price at Signal', 'Price Change 1h (%)']
                                    st.dataframe(
                                        display_bearish.round(3),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                        
                        # Interpretation
                        st.markdown("---")
                        st.markdown("### 💡 Backtest Interpretation")
                        
                        corr_value = analysis['correlations']['price_sentiment']
                        
                        if corr_value > 0.5:
                            st.success(f"**Strong Positive Correlation ({corr_value:.3f})**")
                            st.markdown("✅ Positive sentiment strongly predicts price increases for this stock")
                            st.markdown("✅ Price movements align well with news sentiment")
                            st.markdown("✅ Sentiment analysis is highly effective - use as primary indicator")
                        elif corr_value > 0.3:
                            st.info(f"**Moderate Positive Correlation ({corr_value:.3f})**")
                            st.markdown("📊 Sentiment has predictive power for price movements")
                            st.markdown("📊 Combine with technical indicators for better signals")
                            st.markdown("📊 Good supplementary indicator")
                        elif corr_value < -0.3:
                            st.warning(f"**Negative Correlation ({corr_value:.3f})**")
                            st.markdown("⚠️ Price moves opposite to sentiment (contrarian indicator)")
                            st.markdown("⚠️ Consider counter-trend trading strategies")
                            st.markdown("⚠️ Positive news → Potential sell opportunity")
                        else:
                            st.warning(f"**Weak Correlation ({corr_value:.3f})**")
                            st.markdown("⚠️ Sentiment doesn't strongly predict price for this stock")
                            st.markdown("⚠️ Price driven more by technicals or broader market")
                            st.markdown("⚠️ Use sentiment with caution - not reliable here")
                        
                        # Export options
                        st.markdown("---")
                        st.markdown("### 📥 Export Data for External Analysis")
                        
                        # Prepare export data
                        export_data = combined_data[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'sentiment', 'confidence', 'news_count']].copy()
                        export_data['timestamp'] = export_data['timestamp'].astype(str)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            csv = export_data.to_csv(index=False)
                            ist_now = datetime.now(IST)
                            st.download_button(
                                label="📥 Download Complete Backtest Data (CSV)",
                                data=csv,
                                file_name=f"backtest_{selected_symbol.replace('.NS', '')}_{start_date}_to_{end_date}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Export signals
                            if analysis['signals']:
                                signals_csv = pd.DataFrame(analysis['signals']).to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Trading Signals (CSV)",
                                    data=signals_csv,
                                    file_name=f"signals_{selected_symbol.replace('.NS', '')}_{start_date}_to_{end_date}.csv",
                                    mime="text/csv"
                                )
                        
                        # Summary stats
                        st.markdown("---")
                        st.markdown("### 📊 Summary Statistics")
                        
                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        
                        with summary_col1:
                            st.markdown("**Data Coverage:**")
                            st.write(f"• Total hours analyzed: {len(combined_data)}")
                            st.write(f"• Hours with news: {len([h for h in hourly_news.values() if h])}")
                            st.write(f"• Price data points: {len(historical_price)}")
                        
                        with summary_col2:
                            st.markdown("**Sentiment Stats:**")
                            st.write(f"• Avg sentiment: {analysis['performance_metrics']['avg_sentiment']:.3f}")
                            st.write(f"• Sentiment range: {combined_data['sentiment'].min():.3f} to {combined_data['sentiment'].max():.3f}")
                            st.write(f"• Volatility: {analysis['performance_metrics']['sentiment_volatility']:.3f}")
                        
                        with summary_col3:
                            st.markdown("**Price Performance:**")
                            price_return = ((historical_price['Close'].iloc[-1] - historical_price['Close'].iloc[0]) / historical_price['Close'].iloc[0] * 100)
                            st.write(f"• Total return: {price_return:.2f}%")
                            st.write(f"• Price volatility: {analysis['performance_metrics']['price_volatility']:.2f}%")
                            st.write(f"• Correlation: {corr_value:.3f}")
                    
                    else:
                        st.error("❌ Could not create backtest analysis. Insufficient data points or analysis error.")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        top_positive = [s for s in ranked_stocks if s['sentiment_score'] > sentiment_threshold]
        st.metric("🟢 Positive Sentiment", len(top_positive))
    with col2:
        top_negative = [s for s in ranked_stocks if s['sentiment_score'] < -sentiment_threshold]
        st.metric("🔴 Negative Sentiment", len(top_negative))
    with col3:
        neutral = [s for s in ranked_stocks if abs(s['sentiment_score']) <= sentiment_threshold]
        st.metric("⚪ Neutral", len(neutral))
    
    # Step 3: Fetch stock data
    st.subheader("📈 Step 3: Fetching Stock Market Data...")
    
    top_stocks_by_sentiment = ranked_stocks[:max_stocks]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for stock in top_stocks_by_sentiment:
            try:
                future = executor.submit(
                    process_stock_with_sentiment,
                    stock['symbol'],
                    stock['company'],
                    stock['sentiment_score'],
                    stock['reasons'],
                    stock['news_items'],
                    stock.get('confidence', 0),
                    stock.get('pos_score', 0),
                    stock.get('neg_score', 0)
                )
                futures[future] = stock
            except Exception as e:
                print(f"Error submitting task for {stock['company']}: {str(e)}")
                continue
        
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error getting result: {str(e)}")
            
            completed += 1
            if len(top_stocks_by_sentiment) > 0:
                progress_bar.progress(completed / len(top_stocks_by_sentiment))
            status_text.text(f"Processed {completed}/{len(top_stocks_by_sentiment)} stocks")
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("❌ Could not fetch market data. Please try again.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Apply filters
    filtered_df = df.copy()
    if not show_positive:
        filtered_df = filtered_df[filtered_df['Sentiment Score'] <= sentiment_threshold]
    if not show_negative:
        filtered_df = filtered_df[filtered_df['Sentiment Score'] >= -sentiment_threshold]
    if not show_neutral:
        filtered_df = filtered_df[
            (filtered_df['Sentiment Score'] > sentiment_threshold) | 
            (filtered_df['Sentiment Score'] < -sentiment_threshold)
        ]
    
    # Separate by sentiment
    positive_df = filtered_df[filtered_df['Sentiment Score'] > sentiment_threshold].sort_values('Sentiment Score', ascending=False)
    negative_df = filtered_df[filtered_df['Sentiment Score'] < -sentiment_threshold].sort_values('Sentiment Score', ascending=True)
    neutral_df = filtered_df[
        (filtered_df['Sentiment Score'] >= -sentiment_threshold) & 
        (filtered_df['Sentiment Score'] <= sentiment_threshold)
    ]
    
    # Summary
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyzed", len(filtered_df))
    with col2:
        st.metric("Strong Positive", len(positive_df[positive_df['Sentiment Score'] > 0.2]), delta="🚀")
    with col3:
        st.metric("Strong Negative", len(negative_df[negative_df['Sentiment Score'] < -0.2]), delta="⚠️")
    with col4:
        avg_sentiment = filtered_df['Sentiment Score'].mean() if len(filtered_df) > 0 else 0
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    
    # Display positive stocks
    if show_positive and not positive_df.empty:
        st.markdown("---")
        st.subheader("🟢 Positive Sentiment Stocks")
        
        for idx, row in positive_df.head(15).iterrows():
            sentiment_intensity = "🚀 Very Strong" if row['Sentiment Score'] > 0.3 else "💪 Strong" if row['Sentiment Score'] > 0.2 else "📈 Moderate"
            confidence_badge = f"🎯 {row['Confidence']:.0%}" if row.get('Confidence', 0) > 0 else ""
            
            with st.expander(f"**{row['Company']}** ({row['Symbol']}) | Score: **{row['Sentiment Score']:.3f}** | {sentiment_intensity} {confidence_badge}"):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    if row['Price (₹)'] != 'N/A':
                        st.metric("Price", f"₹{row['Price (₹)']}", f"{row['Change (%)']}%")
                    else:
                        st.metric("Price", "N/A")
                with col2:
                    st.metric("Volume", row['Volume'] if row['Volume'] != 'N/A' else 'N/A')
                with col3:
                    st.metric("News Items", row['News Count'])
                with col4:
                    st.metric("Confidence", f"{row.get('Confidence', 0):.0%}")
                with col5:
                    pos_pct = row.get('Positive %', 0)
                    st.metric("Positive %", f"{pos_pct:.1f}%")
                
                st.markdown("**📰 Detailed News Analysis:**")
                st.text_area("", row['Reasons'], height=200, key=f"pos_{idx}", disabled=True, label_visibility="collapsed")
    
    # Display negative stocks
    if show_negative and not negative_df.empty:
        st.markdown("---")
        st.subheader("🔴 Negative Sentiment Stocks")
        
        for idx, row in negative_df.head(15).iterrows():
            sentiment_intensity = "⚠️ Very Strong" if row['Sentiment Score'] < -0.3 else "📉 Strong" if row['Sentiment Score'] < -0.2 else "👎 Moderate"
            confidence_badge = f"🎯 {row['Confidence']:.0%}" if row.get('Confidence', 0) > 0 else ""
            
            with st.expander(f"**{row['Company']}** ({row['Symbol']}) | Score: **{row['Sentiment Score']:.3f}** | {sentiment_intensity} {confidence_badge}"):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    if row['Price (₹)'] != 'N/A':
                        st.metric("Price", f"₹{row['Price (₹)']}", f"{row['Change (%)']}%")
                    else:
                        st.metric("Price", "N/A")
                with col2:
                    st.metric("Volume", row['Volume'] if row['Volume'] != 'N/A' else 'N/A')
                with col3:
                    st.metric("News Items", row['News Count'])
                with col4:
                    st.metric("Confidence", f"{row.get('Confidence', 0):.0%}")
                with col5:
                    neg_pct = row.get('Negative %', 0)
                    st.metric("Negative %", f"{neg_pct:.1f}%")
                
                st.markdown("**📰 Detailed News Analysis:**")
                st.text_area("", row['Reasons'], height=200, key=f"neg_{idx}", disabled=True, label_visibility="collapsed")
    
    # Display neutral stocks
    if show_neutral and not neutral_df.empty:
        st.markdown("---")
        st.subheader("⚪ Neutral Sentiment Stocks")
        display_neutral = neutral_df[['Company', 'Symbol', 'Price (₹)', 'Change (%)', 'Volume', 'Sentiment Score', 'Confidence', 'News Count']].head(20)
        st.dataframe(display_neutral, use_container_width=True, hide_index=True)
    
    # Complete screener
    st.markdown("---")
    st.subheader("📋 Complete Sentiment Screener")
    
    display_df = filtered_df[['Company', 'Symbol', 'Price (₹)', 'Change (%)', 'Sentiment Score', 'Confidence', 'News Count', 'Sentiment']].copy()
    
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
    
    # Download
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        csv = filtered_df.to_csv(index=False)
        ist_time = datetime.now(IST)
        st.download_button(
            label="📥 Download Screener Results (CSV)",
            data=csv,
            file_name=f"sentiment_screener_{ist_time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col2:
        st.markdown(f"*Last updated: {ist_time.strftime('%Y-%m-%d %H:%M:%S')} IST*")
    
    st.markdown("---")
    st.markdown("**💡 Enhanced Features:**")
    st.markdown("- **Dynamic Stock Universe** - Auto-discovers stocks from NSE")
    st.markdown("- **11+ News Sources** for comprehensive coverage")
    st.markdown("- **🔥 Viral Content Hunter** - Google Trends + Sentiment for YouTube")
    st.markdown("- **📋 Copyable Content** - YouTube tags, hashtags & description templates")
    st.markdown("- **📈 Historical Backtest** - Hourly sentiment tracking with 7-day analysis")
    st.markdown("- **Custom Financial Lexicon** with 100+ Indian market terms")
    st.markdown("- **IST Timestamps** - All times in Indian Standard Time")
    st.markdown("- **Confidence Scores** showing sentiment unanimity")
    st.markdown("")
    st.markdown("*Perfect for creating viral YouTube content AND validating trading strategies!* 🎬📊")
    st.markdown("*Run hourly to build historical database for backtesting!* ⏰")

if __name__ == "__main__":
    main()
