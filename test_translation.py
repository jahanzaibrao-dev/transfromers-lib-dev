from transformers import pipeline

text = "translate English to Urdu: So, Are You In The Market?"

model_name = "Jahanzaibrao/urdu-translation-fine-tuned-model"

translator = pipeline("translation", model=model_name)
translation = translator(text)

print(translation)