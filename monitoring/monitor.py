# # Configurare un sistema di monitoraggio per valutare continuamente le performance del modello e il sentiment rilevato.
# from prometheus_client import start_http_server, Counter
# import time
# import random

# # Definizione della metrica: un contatore chiamato 'richieste_totali'
# REQUEST_COUNT = Counter('richieste_totali', 'Numero totale di richieste ricevute')

# if __name__ == '__main__':
#     # Avvia il server per esporre le metriche su http://localhost:8000/metrics
#     start_http_server(8001)
#     print("Server metriche avviato sulla porta 8001...")
    
#     while True:
#         # Simula l'arrivo di richieste casuali
#         REQUEST_COUNT.inc() 
#         time.sleep(random.random())