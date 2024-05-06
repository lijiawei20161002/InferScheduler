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
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is"
    ] * 25)
]

deadlines = {}
for prompt in prompts:
    deadlines[prompt] = 2
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(output.metrics.finished_time)
    #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")