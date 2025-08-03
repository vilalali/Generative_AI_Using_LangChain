import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Model ID
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use float16 to reduce memory
    device_map="auto"  # Automatically assigns the model to GPU
)

# Initialize pipeline (Remove 'device' argument)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Input prompt
prompt = "What is the capital of India?"

# Generate response
result = pipe(prompt, max_new_tokens=200, temperature=0.7, do_sample=True)

# Print output
print(result[0]["generated_text"])
