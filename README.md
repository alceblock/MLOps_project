# Just a README file

dockerfile to creatse a container (.dockerignore same as .gitignore)

This values have been set in order to have a smooth github's runner due to computational limitations of the machine:
- ```raw_datasets = load_dataset(DATASET, "sentiment", split={"train": "train[:100]", "test": "test[:100]", "validation": "validation[:100]"})```; increase the values to have a bigger dataset.
- same reason for ```per_device_train_batch_size=1,``` and ```per_device_eval_batch_size=1,``` inside ```TraningArguments```; ideally bring it at least up to value 8.

# How to use it:
## Useful commands:
docker system prune -af && docker builder prune -af
docker compose up --build -d
docker compose logs <service_name>

### Grafana dashboard:
Go on ports > open port 3000 in your browser, setup a new connection (http://prometheus:9090) and link it to a new dashboard.
- example of queries that can be called on Grafana: sentiment_positive_total, sentiment_negative_total, sentiment_requests_total

### if you want to test it using a tool like Postman, this CURL can be used:
curl -X POST \
"https://{codespace_name_or_localhost}-8000.app.github.dev/predict" \
-H "Content-Type: application/json" \
-d '{"text":"Ciao questo è un test"}'