import json
import ollama
import pandas as pd
import yfinance as yf


def calculate_rsi(data: pd.Series, window: int = 14) -> float:
    """Calculates the 14-day Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Exponential smoothing approximation for RSI
    for i in range(window, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (window - 1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (window - 1) + loss.iloc[i]) / window

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def analyze_stock_with_news(ticker_symbol: str, model_name: str = "qwen3.5:latest"):
    """Fetches stock data, calculates RSI, grabs recent news, and queries Ollama."""
    print(f"Fetching data and headlines for {ticker_symbol}...")

    # 1. Fetch data from Yahoo Finance
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period="3mo")  # 3 months needed to calculate stable 14-day RSI
    info = ticker.info
    news_items = ticker.news

    if history.empty:
        return print("No data found for this ticker symbol.")

    # 2. Extract key metrics and compute RSI
    latest_close = history["Close"].iloc[-1]
    prev_close = history["Close"].iloc[-2]
    price_change_pct = ((latest_close - prev_close) / prev_close) * 100

    history["MA5"] = history["Close"].rolling(window=5).mean()
    history["MA20"] = history["Close"].rolling(window=20).mean()
    current_rsi = calculate_rsi(history["Close"], window=14)

    # 3. Clean and parse recent news headlines
    recent_headlines = []
    if news_items:
        for item in news_items[:5]:  # Limit to top 5 news stories
            title = item.get("content").get("title")
            url = item.get("content").get('canonicalUrl')['url']
            recent_headlines.append(f"- title: {title} [Read more]({url})")
    else:
        print("No recent news headlines available.")
        recent_headlines.append("No recent news headlines available.")

    # Format a compact dataset for the LLM prompt
    market_summary = {
        "ticker": ticker_symbol,
        "company_name": info.get("longName", ticker_symbol),
        "current_price": round(latest_close, 2),
        "daily_change_percent": round(price_change_pct, 2),
        "volume": int(history["Volume"].iloc[-1]),
        "14_day_rsi": round(current_rsi, 2) if not pd.isna(current_rsi) else "N/A",
        "5_day_moving_avg": round(history["MA5"].iloc[-1], 2) if not history["MA5"].empty else None,
        "20_day_moving_avg": round(history["MA20"].iloc[-1], 2) if not history["MA20"].empty else None,
    }
    print(f"Market Summary: {json.dumps(recent_headlines, indent=2)}")

    # 4. Formulate the comprehensive prompt
    prompt = f"""
    You are an expert financial analyst. Analyze the following local stock data and news headlines to provide a concise breakdown.
    
    Technical Data:
    {json.dumps(market_summary, indent=2)}
    
    Recent News Headlines:
    {"\n".join(recent_headlines)}
    Check provided url to verify the news headlines.
    Provide:
    1. A technical summary based on the moving averages and RSI (note if overbought >70 or oversold <30).
    2. A sentiment assessment blending the technical indicators with the recent news headlines.
    3. A clear disclaimer stating this is AI analysis and not financial advice.
    """

    print(f"Sending data to Ollama using model '{model_name}'...")

    # 5. Stream response from Ollama
    try:
        response = ollama.generate(model=model_name, prompt=prompt, stream=True)

        print("\n--- AI Technical & News Analysis ---")
        for chunk in response:
            print(chunk["response"], end="", flush=True)
        print("\n-------------------------------------")

    except Exception as e:
        print(f"\nError communicating with Ollama: {e}")


if __name__ == "__main__":
    # Ensure the model name exactly matches your downloaded Ollama model
    analyze_stock_with_news(ticker_symbol="TSLA", model_name="qwen3.5:latest")
    # analyze_stock_with_news(ticker_symbol="AAPL", model_name="gemma4:e2b")
