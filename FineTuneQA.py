from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator
from datasets import load_dataset_builder, load_dataset

def preprocess_data(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        # print("answer", i, ": ", answer)
        start_char = answer["answer_start"][0] if len(answer["answer_start"]) > 0 else 0
        end_char = answer["answer_start"][0] + len(answer["text"][0]) if len(answer["answer_start"]) > 0 else 0
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

notebook_login()

squad_dataset = load_dataset("squad_v2", split="train[:5000]")

squad_dataset = squad_dataset.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

tokenized_data = squad_dataset.map(preprocess_data, batched=True, remove_columns=squad_dataset["train"].column_names)

training_args = TrainingArguments(output_dir="my_fine_tuned_qa_model", evaluation_strategy="epoch", overwrite_output_dir=True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True)

data_collator = DefaultDataCollator()

model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_data["train"], eval_dataset=tokenized_data["test"], tokenizer=tokenizer,data_collator=data_collator )

trainer.train()

trainer.push_to_hub()


print("Model Fine Tuned and pushed to hub")