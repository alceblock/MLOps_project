# Just a README file

dockerfile to creatse a container (.dockerignore same as .gitignore)

This values have been set in order to have a smooth github's runner due to computational limitations of the machine:
- ```raw_datasets = load_dataset(DATASET, "sentiment", split={"train": "train[:100]", "test": "test[:100]", "validation": "validation[:100]"})```; increase the values to have a bigger dataset.
- same reason for ```per_device_train_batch_size=1,``` and ```per_device_eval_batch_size=1,``` inside ```TraningArguments```; ideally bring it at least up to value 8.