# src/model_wrapper.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import time
import random

def load_model_and_tokenizer(model_id: str, device: str):
    """
    Loads a model and tokenizer from Hugging Face.

    Args:
        model_id (str): The Hugging Face model identifier.
        device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Set pad token if it doesn't exist (for GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model.to(device)
    model.eval() # Set to evaluation mode
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def reconstruct_sentence(model, tokenizer, sentence: str, prompt_len: int, device: str) -> tuple[str, float]:
    """
    Prompts the model to reconstruct a sentence and measures latency.

    Args:
        model: The pre-trained transformer model.
        tokenizer: The corresponding tokenizer.
        sentence (str): The full sentence to reconstruct.
        prompt_len (int): The number of tokens to use as a prompt.
        device (str): The device the model is on.

    Returns:
        tuple[str, float]: A tuple containing the reconstructed text and inference time in milliseconds.
    """
    # Tokenize the full sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    full_sequence_ids = inputs.input_ids[0] # Get the 1D tensor of IDs
    full_sequence_len = len(full_sequence_ids)

    # Create the prompt
    if prompt_len >= full_sequence_len:
        return sentence, 0.0

    # --- CHANGE 1: Get prompt_ids and attention_mask separately ---
    prompt_ids = full_sequence_ids[:prompt_len].unsqueeze(0).to(device) # unsqueeze to make it a batch of 1
    # Create the attention mask for the prompt (it's all 1s)
    attention_mask = torch.ones_like(prompt_ids).to(device)

    # Generate the completion
    start_time = time.time()
    with torch.no_grad():
        # --- CHANGE 2: Pass attention_mask and remove early_stopping ---
        output_ids = model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_length=full_sequence_len,
            num_beams=1 # Greedy search
            # early_stopping=True -> REMOVED
        )
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000

    # Decode the full generated sequence
    reconstructed_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return reconstructed_text, inference_time_ms

def load_mlm_model_and_tokenizer(model_id: str, device: str):
    """
    Loads a Masked Language Model (like BERT) and its tokenizer.

    Args:
        model_id (str): The Hugging Face model identifier.
        device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    print(f"Loading MLM model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    print("MLM Model and tokenizer loaded successfully.")
    return model, tokenizer

def restore_masked_sentence(model, tokenizer, sentence: str, unmasked_ratio: float, device: str) -> tuple[str, float]:
    """
    Masks a portion of a sentence and uses the MLM to restore it.
    ... (docstring is the same) ...
    """
    # Tokenize the original sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"][0].clone() # This tensor lives on the CPU

    # Exclude special tokens [CLS] and [SEP] from being masked
    special_tokens_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
    maskable_indices = [i for i, m in enumerate(special_tokens_mask) if m == 0]

    # Determine how many tokens to mask
    num_to_mask = round(len(maskable_indices) * (1 - unmasked_ratio))
    if num_to_mask == 0:
        if len(maskable_indices) > 0 and unmasked_ratio < 1.0:
            num_to_mask = 1
        else:
            return sentence, 0.0

    # Randomly select indices to mask
    indices_to_mask = random.sample(maskable_indices, num_to_mask)

    # Create the masked input
    masked_input_ids = input_ids.clone()
    masked_input_ids[indices_to_mask] = tokenizer.mask_token_id

    # Move the input to the correct device for the model
    masked_input_ids = masked_input_ids.unsqueeze(0).to(device)

    # Perform inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(masked_input_ids)
        predictions = outputs.logits.argmax(dim=-1) # This tensor lives on the GPU
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000

    # Fill in the masked tokens with predictions to create the restored sentence
    restored_ids = input_ids.clone() # This tensor lives on the CPU

    # --- THIS IS THE FIX ---
    # Move the predictions from GPU to CPU before assigning them to the CPU tensor
    restored_ids[indices_to_mask] = predictions[0, indices_to_mask].cpu()

    # Decode the restored sentence
    restored_text = tokenizer.decode(restored_ids, skip_special_tokens=True)

    return restored_text, inference_time_ms

def restore_masked_sentence1(model, tokenizer, sentence: str, unmasked_ratio: float, device: str) -> tuple[str, float]:
    """
    Masks a portion of a sentence and uses the MLM to restore it.

    Args:
        model: The pre-trained MLM model (e.g., BERT).
        tokenizer: The corresponding tokenizer.
        sentence (str): The full sentence to restore.
        unmasked_ratio (float): The ratio of tokens to leave unmasked (our θ budget).
        device (str): The device the model is on.

    Returns:
        tuple[str, float]: A tuple containing the restored text and inference time in milliseconds.
    """
    # Tokenize the original sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"][0].clone() # Get a mutable copy

    # Exclude special tokens [CLS] and [SEP] from being masked
    special_tokens_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
    maskable_indices = [i for i, m in enumerate(special_tokens_mask) if m == 0]

    # Determine how many tokens to mask
    num_to_mask = round(len(maskable_indices) * (1 - unmasked_ratio))
    if num_to_mask == 0: # Ensure at least one token is masked if possible
        if len(maskable_indices) > 0 and unmasked_ratio < 1.0:
            num_to_mask = 1
        else: # Cannot mask anything
            return sentence, 0.0

    # Randomly select indices to mask
    indices_to_mask = random.sample(maskable_indices, num_to_mask)

    # Create the masked input
    masked_input_ids = input_ids.clone()
    masked_input_ids[indices_to_mask] = tokenizer.mask_token_id

    masked_input_ids = masked_input_ids.unsqueeze(0).to(device) # Add batch dimension

    # Perform inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(masked_input_ids)
        predictions = outputs.logits.argmax(dim=-1)
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000

    # Fill in the masked tokens with predictions to create the restored sentence
    restored_ids = input_ids.clone()
    restored_ids[indices_to_mask] = predictions[0, indices_to_mask]

    # Decode the restored sentence
    restored_text = tokenizer.decode(restored_ids, skip_special_tokens=True)

    return restored_text, inference_time_ms

# --- Self-Testing Block ---
if __name__ == "__main__":
    print("--- Running model_wrapper.py self-test ---")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "gpt2"
    
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, DEVICE)
    
    test_sentence = "The quick brown fox jumps over the lazy dog."
    prompt_length = 5
    
    reconstruction, latency = reconstruct_sentence(model, tokenizer, test_sentence, prompt_length, DEVICE)
    
    print(f"\nOriginal Sentence: '{test_sentence}'")
    print(f"Prompt Length: {prompt_length} tokens")
    print(f"Reconstruction: '{reconstruction}'")
    print(f"Latency: {latency:.2f} ms")
    
    assert isinstance(reconstruction, str)
    assert latency > 0
    print("\nSelf-test completed successfully.")

    print("\n" + "="*50 + "\n")
    print("--- Running model_wrapper.py MLM self-test ---")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MLM_MODEL_ID = "bert-base-uncased"

    mlm_model, mlm_tokenizer = load_mlm_model_and_tokenizer(MLM_MODEL_ID, DEVICE)

    test_sentence_mlm = "The capital of France is Paris and it is a beautiful city."
    unmasked_ratio = 0.5 # Keep 50% of tokens as context

    restored_version, mlm_latency = restore_masked_sentence(
        mlm_model, mlm_tokenizer, test_sentence_mlm, unmasked_ratio, DEVICE
    )

    print(f"\nOriginal Sentence: '{test_sentence_mlm}'")
    print(f"Unmasked Ratio (θ): {unmasked_ratio*100}%")
    print(f"Restored Sentence: '{restored_version}'")
    print(f"Latency: {mlm_latency:.2f} ms")

    assert isinstance(restored_version, str)
    assert mlm_latency > 0
    print("\nMLM self-test completed successfully.")
