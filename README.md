# Just a README file

dockerfile to creat e a container (.dockerignore same as .gitignore)

```raw_datasets = load_dataset(DATASET, "sentiment", split={"train": "train[:100]", "test": "test[:100]", "validation": "validation[:100]"})``` per consentire runner di github di girare senza problemi