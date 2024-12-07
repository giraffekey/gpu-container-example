import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=50, model_name='gpt2', top_k=50, top_p=0.9, temperature=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    model.to(device)
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(inputs.shape, device=inputs.device)

    start_time = time.time()
    outputs = model.generate(
        inputs,
        max_length=max_length, 
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    )
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "Once upon a time in a land far, far away"
    print(generate_text(prompt))
