import tinker
from tinker import types
from tinker_cookbook import model_info, renderers, tokenizer_utils
from config import tinker_api_key
import os
os.environ["TINKER_API_KEY"] = tinker_api_key

model_name ="meta-llama/Llama-3.2-1B"
checkpoint = ""

tokenizer = tokenizer_utils.get_tokenizer(model_name)
renderer_name = model_info.get_recommended_renderer_name(model_name)
renderer = renderers.get_renderer(renderer_name, tokenizer)

service_client = tinker.ServiceClient()

sampling_client = service_client.create_sampling_client(model_path="tinker://3a7a8e97-4358-4d3f-af94-6388a866b909/sampler_weights/rafi-dpo-test4-working")

message = [{"role": "user", "content": "How to live a meaningful life?"
}]
prompt = renderer.build_generation_prompt(message)

sampling_params = types.SamplingParams(
    max_tokens=50,
    temperature=0.8,
    top_p=0.95,

)

future = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1)

result = future.result()

print("Query: ", message[0]["content"])
for seq in result.sequences:
    text = tokenizer.decode(seq.tokens)
    print("Text: ", text)