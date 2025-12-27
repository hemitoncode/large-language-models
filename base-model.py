from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


'''
Base models are "token predictors" which determine the next token after a window of context tokens based on configured parameters which were updated from training. 
'''

model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()  # no training
prompt = "Here is a sentence followed by next token:"

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# Get the next token probabilities
next_token_logits = logits[:, -1, :]
next_token_id = next_token_logits.argmax(dim=-1).item()
next_token_str = tokenizer.decode([next_token_id])

print("Predicted next token:", next_token_str)
