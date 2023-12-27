from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate


##### Function to get and tokenize the text from dataset dictionary #####
def tokenize_text(text_dict):
    return tokenizer(text_dict["text"], truncation=True, padding="max_length")


##### Trainer Function to compute metrics to evaluate model performance #######
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_evaluation.compute(predictions=predictions, references=labels)

####### Data Preprocessing ###############

data_set = load_dataset("yelp_review_full")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenized_dataset = data_set.map(tokenize_text, batched=True)

small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))


####### Training Hyper Parameters #########
training_arguments = TrainingArguments(output_dir="trainer_output_dir", evaluation_strategy="epoch")

###### Accuracy function for model evaluation ####
metric_evaluation = evaluate.load("accuracy")


###### Model instance of bert-base-cased pretrained model for Sequence classification ######
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5) ### number of labels depend on the dataset. In this dataset "yelp_review_full" number of labels are 5.

###### Training a pretrained model on the dataset of our choice in order to get more appropriate responses ########
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
fine_tuned_model = trainer.train()

print("Done.")
print(fine_tuned_model)



