import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

# Twitter API 认证信息
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAITI2wEAAAAA2LljYwgcTLpwwatxzyXzYK%2F9Qos%3DsfnzN8gkDp91qqOyCEFQqCNSvuuS0RdXHUfTBcRxb6HeKcYKfe"

def get_tweets(keyword, count=100):
    """
    使用 Twitter API 抓取包含关键词的最新推文
    """
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    query = f"{keyword} lang:en -is:retweet"
    tweets = client.search_recent_tweets(query=query, max_results=min(count,100), tweet_fields=["text"])
    tweet_texts = [tweet.text for tweet in tweets.data] if tweets.data else []
    return tweet_texts

def analyze_sentiment(texts):
    """
    对文本列表进行情感分析，返回分数和高频情感词
    """
    analyzer = SentimentIntensityAnalyzer()
    results = []
    all_words = []
    for text in texts:
        score = analyzer.polarity_scores(text)
        results.append(score)
        # 简单分词统计
        words = [w.lower() for w in text.split() if w.isalpha()]
        all_words.extend(words)
    # 统计高频词
    word_freq = Counter(all_words).most_common(30)
    return results, word_freq

if __name__ == "__main__":
    keyword = "Tesla"  # 可替换为任意关键词
    tweets = get_tweets(keyword, count=50)
    print(f"抓取到{len(tweets)}条推文")
    results, word_freq = analyze_sentiment(tweets)
    avg_compound = sum(r['compound'] for r in results) / len(results) if results else 0
    print(f"平均情感分数: {avg_compound:.3f}")
    print("高频词:", word_freq)
