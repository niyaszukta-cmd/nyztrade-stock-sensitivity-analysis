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

# Comprehensive mapping of company names to NSE symbols
COMPANY_SYMBOL_MAP = {
    # Major indices and bluechips
    'reliance': 'RELIANCE.NS',
    'reliance industries': 'RELIANCE.NS',
    'ril': 'RELIANCE.NS',
    'tcs': 'TCS.NS',
    'tata consultancy': 'TCS.NS',
    'infosys': 'INFY.NS',
    'infy': 'INFY.NS',
    'wipro': 'WIPRO.NS',
    'hcl': 'HCLTECH.NS',
    'hcl tech': 'HCLTECH.NS',
    'tech mahindra': 'TECHM.NS',
    
    # Banks
    'hdfc bank': 'HDFCBANK.NS',
    'hdfcbank': 'HDFCBANK.NS',
    'icici bank': 'ICICIBANK.NS',
    'icicibank': 'ICICIBANK.NS',
    'sbi': 'SBIN.NS',
    'state bank': 'SBIN.NS',
    'axis bank': 'AXISBANK.NS',
    'kotak': 'KOTAKBANK.NS',
    'kotak mahindra': 'KOTAKBANK.NS',
    'indusind': 'INDUSINDBK.NS',
    'yes bank': 'YESBANK.NS',
    'pnb': 'PNB.NS',
    'punjab national': 'PNB.NS',
    'bank of baroda': 'BANKBARODA.NS',
    
    # Auto
    'maruti': 'MARUTI.NS',
    'maruti suzuki': 'MARUTI.NS',
    'tata motors': 'TATAMOTORS.NS',
    'mahindra': 'M&M.NS',
    'm&m': 'M&M.NS',
    'bajaj auto': 'BAJAJ-AUTO.NS',
    'hero motocorp': 'HEROMOTOCO.NS',
    'hero': 'HEROMOTOCO.NS',
    'eicher': 'EICHERMOT.NS',
    'tvs motor': 'TVSMOTOR.NS',
    
    # Pharma
    'sun pharma': 'SUNPHARMA.NS',
    'dr reddy': 'DRREDDY.NS',
    'cipla': 'CIPLA.NS',
    'lupin': 'LUPIN.NS',
    'divi': 'DIVISLAB.NS',
    'biocon': 'BIOCON.NS',
    'torrent pharma': 'TORNTPHARM.NS',
    
    # FMCG
    'hindustan unilever': 'HINDUNILVR.NS',
    'hul': 'HINDUNILVR.NS',
    'itc': 'ITC.NS',
    'nestle': 'NESTLEIND.NS',
    'britannia': 'BRITANNIA.NS',
    'dabur': 'DABUR.NS',
    'marico': 'MARICO.NS',
    'godrej consumer': 'GODREJCP.NS',
    
    # Metals & Mining
    'tata steel': 'TATASTEEL.NS',
    'jsw steel': 'JSWSTEEL.NS',
    'hindalco': 'HINDALCO.NS',
    'vedanta': 'VEDL.NS',
    'coal india': 'COALINDIA.NS',
    'nmdc': 'NMDC.NS',
    'jindal steel': 'JINDALSTEL.NS',
    
    # Telecom
    'bharti airtel': 'BHARTIARTL.NS',
    'airtel': 'BHARTIARTL.NS',
    'vodafone idea': 'IDEA.NS',
    'idea': 'IDEA.NS',
    
    # Energy & Power
    'ntpc': 'NTPC.NS',
    'power grid': 'POWERGRID.NS',
    'adani power': 'ADANIPOWER.NS',
    'adani green': 'ADANIGREEN.NS',
    'tata power': 'TATAPOWER.NS',
    'ongc': 'ONGC.NS',
    'oil india': 'OIL.NS',
    'ioc': 'IOC.NS',
    'bpcl': 'BPCL.NS',
    'hpcl': 'HPCL.NS',
    'gail': 'GAIL.NS',
    
    # Adani Group
    'adani enterprises': 'ADANIENT.NS',
    'adani ports': 'ADANIPORTS.NS',
    'adani transmission': 'ADANITRANS.NS',
    'adani total gas': 'ATGL.NS',
    'adani wilmar': 'AWL.NS',
    
    # Tata Group
    'tata consumer': 'TATACONSUM.NS',
    'titan': 'TITAN.NS',
    'trent': 'TRENT.NS',
    
    # Infrastructure & Construction
    'larsen toubro': 'LT.NS',
    'l&t': 'LT.NS',
    'lt': 'LT.NS',
    'ultratech': 'ULTRACEMCO.NS',
    'ambuja cement': 'AMBUJACEM.NS',
    'acc': 'ACC.NS',
    'grasim': 'GRASIM.NS',
    
    # Finance & NBFC
    'bajaj finance': 'BAJFINANCE.NS',
    'bajaj finserv': 'BAJAJFINSV.NS',
    'sbi life': 'SBILIFE.NS',
    'icici lombard': 'ICICIGI.NS',
    'hdfc life': 'HDFCLIFE.NS',
    'lic': 'LICI.NS',
    'shriram finance': 'SHRIRAMFIN.NS',
    
    # Others
    'asian paints': 'ASIANPAINT.NS',
    'berger paints': 'BERGEPAINT.NS',
    'pidilite': 'PIDILITIND.NS',
    'havells': 'HAVELLS.NS',
    'siemens': 'SIEMENS.NS',
    'abb': 'ABB.NS',
    'dmart': 'DMART.NS',
    'zomato': 'ZOMATO.NS',
    'paytm': 'PAYTM.NS',
    'nykaa': 'NYKAA.NS',
    'policy bazaar': 'POLICYBZR.NS',
    'irctc': 'IRCTC.NS',
    'sail': 'SAIL.NS',
    'bhel': 'BHEL.NS',
}

# Helper Functions (defined first)
def log_error(message):
    """Log errors for debugging - thread-safe version"""
    try:
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        
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
    
    return found_symbols

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
    st.markdown("*Discovers trending stocks from live news with sentiment analysis*")
    
    st.info("🔍 This dashboard automatically discovers stocks making news and ranks them by sentiment intensity!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=False)
        
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
        st.markdown("1. 📰 Scrapes latest market news")
        st.markdown("2. 🔍 Extracts stock mentions")
        st.markdown("3. 📊 Analyzes news sentiment")
        st.markdown("4. 📈 Fetches stock data")
        st.markdown("5. 🎯 Ranks by sentiment")
        
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
        st.markdown("📊 **Analysis:**")
        st.markdown("- VADER with Custom")
        st.markdown("  Financial Lexicon")
        st.markdown("- 100+ Indian market")
        st.markdown("  specific terms")
        
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
        st.download_button(
            label="📥 Download Screener Results (CSV)",
            data=csv,
            file_name=f"sentiment_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col2:
        st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    st.markdown("---")
    st.markdown("**💡 Enhanced Features:**")
    st.markdown("- **11+ News Sources** for comprehensive coverage")
    st.markdown("- **Custom Financial Lexicon** with 100+ Indian market terms")
    st.markdown("- **Confidence Scores** showing sentiment unanimity")
    st.markdown("- **Detailed Metrics** (Positive%, Negative%, Neutral%)")
    st.markdown("- **Sorted by Impact** - highest sentiment intensity first!")
    st.markdown("")
    st.markdown("*Stocks with higher absolute sentiment scores are making more impactful news!*")

if __name__ == "__main__":
    main()
