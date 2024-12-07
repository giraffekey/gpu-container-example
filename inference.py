import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=50):
    # Check if GPU is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load downloaded GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("./gpt2_model")
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_tokenizer", clean_up_tokenization_spaces=True)
    
    # Move model to the GPU
    model.to(device)

    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Create attention mask
    attention_mask = torch.ones(inputs.shape, device=inputs.device)

    # Generate text
    outputs = model.generate(
        inputs,
        max_length=max_length, 
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "Once upon a time in a land far, far away"
    generated_text = generate_text(prompt)
    print(generated_text)
