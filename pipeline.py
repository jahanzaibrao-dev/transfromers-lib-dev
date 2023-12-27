from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

modelName = "distilbert-base-uncased-finetuned-sst-2-english"


classifier = pipeline("sentiment-analysis", model = modelName)


response = classifier("This is sample text for classifier")

print(response)


model = AutoModelForSequenceClassification.from_pretrained(modelName)
tokenizer = AutoTokenizer.from_pretrained(modelName)

classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

response = classifier("This is sample text for classifier")

print(response)

text = "This is sample text for classifier. This is second sentence"

response = tokenizer(text)
print(response)

tokens = tokenizer.tokenize(text)
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decoded_string = tokenizer.decode(ids)
print(decoded_string)

# generator = pipeline("text-generation", "distilgpt2")

# response = generator("Today is the battle day. we'll accumulate the cavalry on right wing and ", max_length=100, num_return_sequences=3)

# print(response)

