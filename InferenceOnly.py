import tinker
from tinker import types
from tinker_cookbook import model_info, renderers, tokenizer_utils
from config import tinker_api_key
import os
os.environ["TINKER_API_KEY"] = tinker_api_key

model_name ="meta-llama/Llama-3.1-8B"
checkpoint = ""

tokenizer = tokenizer_utils.get_tokenizer(model_name)
renderer_name = model_info.get_recommended_renderer_name(model_name)
renderer = renderers.get_renderer(renderer_name, tokenizer)

service_client = tinker.ServiceClient()

sampling_client = service_client.create_sampling_client(model_path="tinker://8dbf6113-9676-4592-b7ac-abfcd7810d37/sampler_weights/rafi-test05")

message = [{"role": "user", "content": "What are Newton’s three laws of motion?"
}]
prompt = renderer.build_generation_prompt(message) # hopefully this will return the prompt

sampling_params = types.SamplingParams(
    max_tokens=200,
    temperature=0.7,
    top_p=0.95,

)

future = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1)

result = future.result()

print("Query: ", message[0]["content"])
for seq in result.sequences:
    text = tokenizer.decode(seq.tokens)
    print("Text: ", text)