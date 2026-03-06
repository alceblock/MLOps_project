from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from model.model_utility import DATASET, MODEL_PATH, new_version_path_builder

# reproducibility training
from transformers import set_seed
set_seed(42)

# raw_datasets = load_dataset(DATASET, "sentiment")
raw_datasets = load_dataset(DATASET, "sentiment", 
    split={"train": "train[:100]", "test": "test[:100]", "validation": "validation[:100]"})

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# NOTE: this seed is to reproduce dataset, not training, do not remove
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)#.select(range(100))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)#.select(range(100))

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=3
)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    # per_device_train_batch_size=8,
    # per_device_eval_batch_size=8,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    #
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# metric = evaluate.load("accuracy")
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels, average="macro")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="macro")["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average="macro")["f1"],
    }

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()
results = trainer.evaluate()
print(f"\nResults:\n{results}")

# Save new model version
# trainer.save_model("my_model_versions")
# tokenizer.save_pretrained("my_model_versions")
final_path = new_version_path_builder()
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)