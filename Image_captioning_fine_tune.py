from huggingface_hub import notebook_login
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
from evaluate import load
import torch


def plot_images(images, captions):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        caption = captions[i]
        caption = "\n".join(wrap(caption, 12))
        plt.title(caption)
        plt.imshow(images[i])
        plt.axis("off")

def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}

model_name = "microsoft/git-base"
wer = load("wer")


pokemon_dataset = load_dataset("diffusers/pokemon-gpt4-captions")
pokemon_dataset = pokemon_dataset["train"].train_test_split(test_size=0.1)

train_dataset = pokemon_dataset["train"]
test_dataset = pokemon_dataset["test"]

sample_images_to_visualize = [np.array(train_dataset[i]["image"]) for i in range(5)]
sample_captions = [train_dataset[i]["text"] for i in range(5)]
plot_images(sample_images_to_visualize, sample_captions)

processor = AutoProcessor.from_pretrained(model_name)

train_dataset.set_transform(transforms)
test_dataset.set_transform(transforms)
model = AutoModelForCausalLM.from_pretrained(model_name)

model_name = model_name.split("/")[1]

training_args = TrainingArguments(
    output_dir=f"models/{model_name}-pokemon-finetuned",
    overwrite_output_dir=True,
    learning_rate=5e-5,
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=50,
    remove_unused_columns=False,
    push_to_hub=True,
    label_names=["labels"],
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()


