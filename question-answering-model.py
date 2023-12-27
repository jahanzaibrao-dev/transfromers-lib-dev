from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = [{
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}, {
    'question': 'Who won ICC champions trophy 2017',
    'context': 'In ICC champions trophy 2017 8 top teams in ICC rankings participated. The final match of the was played between Pakistan and India, resulting in Pakistan becoming the champions'
}]
res = nlp(QA_input)

print(res)

