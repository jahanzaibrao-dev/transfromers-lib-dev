from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor


dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
audio_input = [dataset[0]["audio"]["array"]]

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
response = feature_extractor(audio_input, sampling_rate=16000)

print(response)
