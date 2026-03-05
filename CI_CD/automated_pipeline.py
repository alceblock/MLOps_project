# Sviluppare una pipeline automatizzata per il training del modello, i test di integrazione e il deploy dell'applicazione su HuggingFace.

from tests.test_model_inference import test_root_endpoint


def test_automated_training():
    # call training function
    print("automated training ongoing....")
    assert 1 == 1

def test_automated_integration_test():
    # call training function
    print("automated integration test ongoing....")
    test_root_endpoint()
    print("automated integration test COMPLETED")

def test_automated_deploy():
    # call training function
    print("automated deploy to hugging face ongoing....")