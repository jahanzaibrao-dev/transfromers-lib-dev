from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np

model_name = "t5-small"
prefix = "summarize: "


def preprocess_data(example):
    inputs = [prefix + doc for doc in example["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=example["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

notebook_login()

samsum_dataset = load_dataset("samsum", split="test")
samsum_dataset = samsum_dataset.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_samsum_dataset = samsum_dataset.map(preprocess_data, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
rouge = evaluate.load("rouge")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

training_args = Seq2SeqTrainingArguments(
    output_dir="models/Finetuned_summarization_model",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_samsum_dataset["train"],
    eval_dataset=tokenized_samsum_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()


print("Fine Tuned t5-small on dialogue dataset")

