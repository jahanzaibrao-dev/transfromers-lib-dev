import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

modelName = "distilbert-base-uncased-finetuned-sst-2-english"


XTrainData = ["This is sample text for tokenizer.", "This is second sentence"]

model = AutoModelForSequenceClassification.from_pretrained(modelName)
tokenizer = AutoTokenizer.from_pretrained(modelName)

classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

response = classifier(XTrainData)
print(response)

batch = tokenizer(XTrainData, padding=True, truncation=True, return_tensors="pt", max_length=512)
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)
