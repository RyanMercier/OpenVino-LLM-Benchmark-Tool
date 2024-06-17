from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import time
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Text generation benchmark script")
parser.add_argument("--device", type=str, default="GPU", choices=["CPU", "GPU"], help="Specify the device (CPU or GPU)")
args = parser.parse_args()

start_time = time.time()
model_id = "ov_model" # Model Directory (OpenVino IR Format)
model = OVModelForCausalLM.from_pretrained(model_id, device=args.device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

load_time = time.time() - start_time

# Define a list of prompts for benchmarking
prompts = [
    "Generate a short poem about a skeleton",
    "Generate a description of the company Microsoft",
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