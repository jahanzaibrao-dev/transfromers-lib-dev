from transformers import AutoModelForCausalLM, AutoTokenizer

input_text = "The quick brown"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")

model_inputs = tokenizer([input_text], return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
)

masked_input_ids = model.generate(**model_inputs)

decoded_response = tokenizer.batch_decode(masked_input_ids, skip_special_tokens=True)[0]


print(decoded_response) 