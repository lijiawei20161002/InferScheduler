from modelscope.hub.api import HubApi
import os
import datetime
import pandas as pd
import nltk
from nltk.corpus import brown
import time

access_token = os.getenv('ACCESS_TOKEN')
if access_token is None:
    raise ValueError("ACCESS_TOKEN environment variable not set.")

api = HubApi()
api.login(access_token)
from vllm import LLM, SamplingParams

csv_file_path = '../data/AzureLLMInferenceTrace_conv.csv'  
df = pd.read_csv(csv_file_path)
nltk.download('brown')
brown_words = brown.words()

def generate_meaningful_prompt(token_length):
    # Create a meaningful sentence with the exact number of tokens
    sentence = " ".join(brown_words[:token_length])
    return sentence

prompts = [
    f"{i+1}.{generate_meaningful_prompt(100)}"
    for i in range(256*5)
]

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10, min_tokens=10)
llm = LLM(model="gpt2")

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()
goodput = 0
total_inference_time = end_time - start_time
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    #print('\n======================\n')
    #print(prompt, generated_text)
    #print(output.metrics.finished_time <= output.metrics.deadline,"arrival:", output.metrics.arrival_time, "finished_time:", output.metrics.finished_time, "deadline:", output.metrics.deadline)
    #print('\n======================\n')
    if output.metrics.finished_time <= output.metrics.deadline:
        goodput += 1
total_tokens_generated = sum(len(output.outputs[0].text.split()) for output in outputs)
average_time_per_token = total_inference_time / total_tokens_generated
batch_size = 256
print(f"Inference time for batch size {batch_size}: {total_inference_time:.2f} seconds")
print(f"Total tokens generated: {total_tokens_generated}")
print(f"Average time per token: {average_time_per_token:.4f} seconds")
print(f"Goodput: {goodput}")
