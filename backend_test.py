from sentiment_hf_mvp_api import SentimentAnalysisAPI

base_url = "https://jozoe-sentiment-analysis-space2.hf.space"
api = SentimentAnalysisAPI(base_url)

def analyze_sentiment(text):
    try:
        event_id = api.submit_request(text)
        result = api.get_result(event_id)
        return result
    except Exception as e:
        return {"error": str(e)}

# 示例
if __name__ == "__main__":
    text = "I love using Hugging Face!"
    sentiment = analyze_sentiment(text)
    print("Sentiment Analysis Result:", sentiment)
