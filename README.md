# Just a README file
MLOps project:
Dettagli del Progetto
Fase 1: Implementazione del Modello di Analisi del sentiment con FastText
Modello: Utilizzare un modello pre-addestrato FastText per un’analisi del sentiment in grado di classificare testi dai social media in sentiment positivo, neutro o negativo. Servirsi di questo modello: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
Dataset: Utilizzare dataset pubblici contenenti testi e le rispettive etichette di sentiment.
Fase 2: Creazione della Pipeline CI/CD
Pipeline CI/CD: Sviluppare una pipeline automatizzata per il training del modello, i test di integrazione e il deploy dell'applicazione su HuggingFace.
Fase 3: Deploy e Monitoraggio Continuo
Deploy su HuggingFace (facoltativo): Implementare il modello di analisi del sentiment, inclusi dati e applicazione, su HuggingFace per facilitare l'integrazione e la scalabilità.
Sistema di Monitoraggio: Configurare un sistema di monitoraggio per valutare continuamente le performance del modello e il sentiment rilevato.

# PREVIEW:
## Structure
MLops_project/
├── .github/workflows/
│   └── CI_integration.yml    # Continuous integration pipeline
├── model_app/
│   ├── model_inference.py    # Inference and relative apis
│   └── model_training.py     # Model training
│   └── model_utility.py      # Utility functions for model_app
├── monitoring/
│   ├── metrics.py            # Base structure of metrics
│   └── prometheus.yml        # Prometheus config
├── tests/
│   └── test_model_inference.py # Inference integration tests
│   └── test_model_training.py # Training integration tests
├── docker-compose.yml        # Docker compose config
├── Dockerfile                # Docker image config
└── requirements.txt
└── pytest.ini                # File needed by pytest


## Use suggestions:
Tested on a macOS (UNIX).


# OVERVIEW
The project is designed to run at first with model ```a``` and dataset ```b```. 


# Technical choices:
- In order to avoid disk space troubles (for example on github codespaces) these values have been set as small as reasonably possible:
-- ```raw_datasets = load_dataset(DATASET, "sentiment", split={"train": "train[:100]", "test": "test[:100]", "validation": "validation[:100]"})```; increase the values to have a bigger dataset.
-- same reason for ```per_device_train_batch_size=1,``` and ```per_device_eval_batch_size=1,``` inside ```TraningArguments``` in ```model_app/model_training.py```; ideally bring it at least up to value 8.


# HOW TO USE IT:
## Useful commands:

- Create an environment:
```python -m venv <name_your_venv>```
and activate it:
```source <name_your_venv>/bin/activate```
to deactivate:
```deactivate```

- Running the app:
```docker compose up --build -d```
everything will start properly.

- In case of any issue with docker run:
```docker compose logs <service_name>```
to debug the service

- In case of space limitations (example github: codespaces), run these two commands to clean docker:
WARNING: this can delete data, use it carefully.
```docker system prune -af && docker builder prune -af```

## Run Tests:
Just run ```pytest``` in the terminal to execute all the tests inside tests folder.

### Grafana dashboard:
Go on ports > open port 3000 in your browser, setup a new connection (http://prometheus:9090) and link it to a new dashboard.
- example of queries that can be called on Grafana: sentiment_positive_total, sentiment_negative_total, sentiment_requests_total

### if you want to test it using a tool like Postman, this CURL can be used:
curl -X POST \
"https://{codespace_name_or_localhost}-8000.app.github.dev/predict" \
-H "Content-Type: application/json" \
-d '{"text":"Ciao questo è un test"}'