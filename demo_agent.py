from transformers import HfAgent
from huggingface_hub import login

login("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")

agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")

picture = agent.run("Draw me a picture of river.", wait_for_model=True)


