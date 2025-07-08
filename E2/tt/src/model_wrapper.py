# E2/tt/src/model_wrapper.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import time
import random

def load_model_and_tokenizer(model_id: str, device: str):
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model.to(device)
    model.eval() 
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def reconstruct_sentence(model, tokenizer, sentence: str, prompt_len: int, device: str) -> tuple[str, float]:
    inputs = tokenizer(sentence, return_tensors="pt")
    full_sequence_ids = inputs.input_ids[0] 
    full_sequence_len = len(full_sequence_ids)

    if prompt_len >= full_sequence_len:
        return sentence, 0.0

    prompt_ids = full_sequence_ids[:prompt_len].unsqueeze(0).to(device) 
    attention_mask = torch.ones_like(prompt_ids).to(device)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_length=full_sequence_len,
            num_beams=1 
        )
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000

    reconstructed_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return reconstructed_text, inference_time_ms

def load_mlm_model_and_tokenizer(model_id: str, device: str):
    print(f"Loading MLM model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    print("MLM Model and tokenizer loaded successfully.")
    return model, tokenizer

def restore_masked_sentence(model, tokenizer, sentence: str, unmasked_ratio: float, device: str) -> tuple[str, float]:
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"][0].clone() 

    special_tokens_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
    maskable_indices = [i for i, m in enumerate(special_tokens_mask) if m == 0]

    num_to_mask = round(len(maskable_indices) * (1 - unmasked_ratio))
    if num_to_mask == 0:
        if len(maskable_indices) > 0 and unmasked_ratio < 1.0:
            num_to_mask = 1
        else:
            return sentence, 0.0

    indices_to_mask = random.sample(maskable_indices, num_to_mask)

    masked_input_ids = input_ids.clone()
    masked_input_ids[indices_to_mask] = tokenizer.mask_token_id

    masked_input_ids = masked_input_ids.unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(masked_input_ids)
        predictions = outputs.logits.argmax(dim=-1) 
    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000
    restored_ids = input_ids.clone() 
    restored_ids[indices_to_mask] = predictions[0, indices_to_mask].cpu()
    restored_text = tokenizer.decode(restored_ids, skip_special_tokens=True)

    return restored_text, inference_time_ms
