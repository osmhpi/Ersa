#!/usr/bin/env python
import time
start = time.time()

from transformers import AutoModel
from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from typing import Dict, Any
from datasets import Dataset
import evaluate
import numpy as np
from transformers import EvalPrediction
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer

SEED = 42

dataset = load_dataset("ag_news", cache_dir="./data")
dataset = dataset.shuffle(SEED)

dataset["train"] = dataset["train"].select(range(400))

subset = dataset["train"].shuffle(SEED).train_test_split(0.1, seed=SEED)
dataset["train"] = subset["train"]
dataset["val"] = subset["test"]

num_labels = len(torch.unique(torch.as_tensor([k["label"] for k in dataset["train"]])))

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")


def preprocess_function(sample: Dict[str, Any], seq_len: int):
    """
    Function applied to all the examples in the Dataset (individually or in batches).
    It accepts as input a sample as a dictionary and return a new dictionary with the BERT tokens for that sample
    """
    t = tokenizer(
        sample["text"], padding="max_length", truncation=True, max_length=seq_len
    )
    return t


encoded_ds = dataset.map(preprocess_function, fn_kwargs={"seq_len": 256})

BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 3#5

args = TrainingArguments(
    output_dir="nlp_course_from_pretrained",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
)

config = AutoConfig.from_pretrained("distilroberta-base", num_labels=num_labels)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base", config=config
)

def compute_metrics(eval_pred: EvalPrediction):
    """Compute metrics at evaluation, return a dictionary string to metric values."""
    import evaluate

    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

result = None

try: 
    device = "cuda:0"
    model = model.to(device)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_ds["train"],
        eval_dataset=encoded_ds["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    result = trainer.evaluate()

except torch.cuda.OutOfMemoryError:
    result = "out of memory"

except RuntimeError as error:
    error_message = str(error)
    if error_message.startswith("CUDA error: out of memory"):
        result = "out of memory"

else:
    result = result['eval_accuracy']

end = time.time()
duration = end - start