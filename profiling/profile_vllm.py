from modelscope.hub.api import HubApi
import os
import datetime

access_token = os.getenv('ACCESS_TOKEN')
if access_token is None:
    raise ValueError("ACCESS_TOKEN environment variable not set.")

api = HubApi()
api.login(access_token)

from vllm import LLM, SamplingParams

prompts = [
    f"{i+1}. {prompt}" for i, prompt in enumerate([
        "Output 10 tokens Output 10 tokens Output 10 tokens Output 10 tokens",
    ] * 100)
]

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10, min_tokens=10)
llm = LLM(model="gpt2")

outputs = llm.generate(prompts, sampling_params)
goodput = 0
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    #print(generated_text)
    print(output.metrics.finished_time <= output.metrics.deadline, output.metrics.arrival_time, output.metrics.finished_time, output.metrics.deadline)
    if output.metrics.finished_time <= output.metrics.deadline:
        goodput += 1
print(f"Goodput: {goodput}")