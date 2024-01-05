from datasets import load_dataset, Audio
from huggingface_hub import notebook_login
from transformers import AutoProcessor, AutoModelForCTC, TrainingArguments, Trainer
from data_collator_ctc import DataCollatorCTCWithPadding
import evaluate
import numpy as np

def convert_text_to_uppercase(example):
    return {"text": example["text"].upper()}

def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["text"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


notebook_login()

arabic_dataset = load_dataset("arabic_speech_corpus",split="train[:100]")

arabic_dataset = arabic_dataset.train_test_split(test_size=0.2)

arabic_dataset = arabic_dataset.remove_columns(["phonetic", "orthographic"])

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

arabic_dataset = arabic_dataset.cast_column("audio", Audio(sampling_rate=16_000))

arabic_dataset = arabic_dataset.map(convert_text_to_uppercase)

encoded_dataset = arabic_dataset.map(prepare_dataset, remove_columns=arabic_dataset.column_names["train"], num_proc=4)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

wer = evaluate.load("wer")

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

training_args = TrainingArguments(
    output_dir="models/Arabic_fine_tuned_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()

print(arabic_dataset["train"][0])