from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import time

start_time = time.time()
model_id = "st_model" # Model Directory (Safetensors Format)
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

load_time = time.time() - start_time

# Define a list of prompts for benchmarking
prompts = [
    "Generate a short poem about a skeleton",
    "Generate a description of the company Qualtech Systems Inc",
    "Generate python code for finding the nth fibonacci number"
]

# Benchmark the model by generating text for each prompt
start_time = time.time()
num_tokens = 0
for prompt in prompts:
    print("\n\n***              New Prompt              *** \n")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    num_tokens += len(outputs[0])
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
eval_time = time.time() - start_time

print(f"device type: {args.device}")

print(f"total duration: {eval_time + load_time:.2f} seconds")

print(f"load duration: {load_time:.2f} seconds")

print(f"eval duration: {eval_time:.2f} seconds")

print(f"eval count: {num_tokens} token(s)")

# Calculate Tokens per Second
tps = num_tokens / eval_time
print(f"eval rate (output tokens per second): {tps:.2f} tokens/second")
