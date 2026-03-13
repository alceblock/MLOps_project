from prometheus_client import Counter, Histogram
#from prometheus_client import Gauge

# # model's PERFORMANCE
# MODEL_ACCURACY = Gauge("model_accuracy", "Accuracy of the model")
# MODEL_F1 = Gauge("model_f1", "F1 score of the model")
# MODEL_PRECISION = Gauge("model_precision", "Precision of the model")
# MODEL_RECALL = Gauge("model_recall", "Recall of the model")

# model's SENTIMENT
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