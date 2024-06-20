from modelscope.hub.api import HubApi
from huggingface_hub import HfApi, HfFolder, login
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
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

hf_token = os.getenv('hf_token')
if hf_token is None:
    raise ValueError("hf_token environment variable not set.")
login(token=hf_token)
print("Successfully logged in to Hugging Face")


model_name = 'PartAI/Dorna-Llama3-8B-Instruct'

from vllm import LLM, SamplingParams

csv_file_path = '../data/AzureLLMInferenceTrace_conv.csv'  
df = pd.read_csv(csv_file_path)
nltk.download('brown')
brown_words = brown.words()

def generate_meaningful_prompt(token_length):
    # Create a meaningful sentence with the exact number of tokens
    sentence = " ".join(brown_words[:token_length])
    return sentence

# Parameters
burst_size = 10
burst_interval = 5  # seconds between bursts
regular_interval = 0.1  # seconds between prompts within a burst

# Generate prompts with bursty arrival patterns
#prompts_with_arrivals = simulate_bursty_arrivals(df, burst_size, burst_interval, regular_interval)

prompts = [
    "Output 100 tokens and end with '.'"
] * 100

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1, min_tokens=1)
llm = LLM(model=model_name, trust_remote_code=True)

max_iterations = 1000
goodput = 0

current_time = time.time()
outputs = llm.generate(prompts, sampling_params)
for output in output:
    print("arrival time:", output.arrival_time)
    print("context text:", output.prompt)
    print("generated text:", output.output_text)
    print("finish time:", output.finished_time)

# Profile Inference Time
'''        
total_tokens_generated = sum(len(output.outputs[0].text.split()) for output in outputs)
average_time_per_token = total_inference_time / total_tokens_generated
batch_size = 256
print(f"Inference time for batch size {batch_size}: {total_inference_time:.2f} seconds")
print(f"Total tokens generated: {total_tokens_generated}")
print(f"Average time per token: {average_time_per_token:.4f} seconds")
print(f"Goodput: {goodput}")'''
