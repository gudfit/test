# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]["text"]
            if not text.strip():  # Skip empty documents
                continue

            enc = tokenizer(text, return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                # Take the first seqlen tokens
                inp = enc.input_ids[:, :seqlen]
                tar = inp.clone()
                tar[:, :-1] = -100
                trainloader.append((inp, tar))
                break

    # Encode the test set for perplexity evaluation
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    # Use streaming to avoid downloading the whole dataset
    traindata = load_dataset("allenai/c4", "en", split="train", streaming=True)
    valdata = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    # Generate samples from training set
    random.seed(seed)
    trainloader = []

    # Iterate through the streaming dataset
    for sample in traindata:
        if len(trainloader) == nsamples:
            break
        # Process only non-empty texts
        if sample["text"]:
            enc = tokenizer(sample["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                # Take a random slice of seqlen from the document
                i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = enc.input_ids[:, i:j]
                tar = inp.clone()
                tar[:, :-1] = -100
                trainloader.append((inp, tar))

    # Prepare validation dataset using streaming
    val_texts = []
    # Take a fixed number of samples for consistent validation
    for i, sample in enumerate(valdata):
        if i == 1024:
            break
        if sample["text"]:
            val_texts.append(sample["text"])

    valenc = tokenizer(" ".join(val_texts), return_tensors="pt")
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "wikitext" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)

    raise ValueError(f"Dataset '{name}' is not supported in vendor/wanda/lib/data.py")
