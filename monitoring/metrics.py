from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "sentiment_requests_total",
    "Total requests"
)

REQUEST_LATENCY = Histogram(
    "sentiment_request_latency_seconds",
    "Latency"
)

POSITIVE_COUNT = Counter("sentiment_positive", "positive tweets")
NEGATIVE_COUNT = Counter("sentiment_negative", "negative tweets")
NEUTRAL_COUNT = Counter("sentiment_neutral", "neutral tweets")