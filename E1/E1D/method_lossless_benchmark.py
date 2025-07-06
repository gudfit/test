#!/usr/bin/env python3
# E1/E1D/method_lossless_benchmark.py
from __future__ import annotations
import argparse, gzip, bz2, math, subprocess, sys
from pathlib import Path
from typing import Dict, Tuple, List, Union
import torch, os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

sys.path.append(str(Path(__file__).resolve().parents[2]))
from E1.E1A.bakeoff_utils import (
    predictive_masking_compress,
    decompress_text,
    LSQDecoder,
)

BITS = 8
STRIDE = 512


def hb(x):
    u = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while x >= 1024 and i < len(u) - 1:
        x /= 1024
        i += 1
    return f"{x:.2f} {u[i]}"


def dir_size(p):
    p = Path(p)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def get_dir_size_bytes(path="."):
    return dir_size(path)


def token_acc(orig: str, rec: str, tok) -> float:
    o = tok(orig).input_ids
    r = tok(rec).input_ids
    L = min(len(o), len(r))
    if L == 0:
        return 0.0
    return sum(1 for i in range(L) if o[i] == r[i]) / L


def hf_dir(name, cache):
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(name, cache_dir=str(cache), local_files_only=True))
    except Exception:
        m = list(cache.glob(f"models--{name.replace('/', '--')}*"))
        return m[0] if m else Path("/dev/null")


def raw_stats(fp):
    t = fp.read_text(encoding="utf-8")
    return len(t), t


def alg(fp, n):
    r = {}
    raw = fp.read_bytes()
    g = gzip.compress(raw)
    b = bz2.compress(raw)
    r["gzip"] = ((len(g) * BITS) / n, len(g))
    r["bzip2"] = ((len(b) * BITS) / n, len(b))
    zst = fp.with_suffix(fp.suffix + ".zst")
    try:
        subprocess.run(["zstd", "-q", "-f", str(fp), "-o", str(zst)], check=True, capture_output=True)
        s = zst.stat().st_size
        r["zstd"] = ((s * BITS) / n, s)
    except (FileNotFoundError, subprocess.CalledProcessError):
        r["zstd"] = (float("nan"), 0)
    finally:
        if zst.exists():
            zst.unlink(missing_ok=True)
    return r


def ar(model_name, text, n_chars, cache):
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if d == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache))
    m = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache), torch_dtype=dtype).to(d)
    m.eval()

    m_sz = dir_size(hf_dir(model_name, cache))
    enc = tok(text, return_tensors="pt")
    seq = enc.input_ids.size(1)
    max_len = m.config.max_position_embeddings
    nll = 0.0
    toks = 0
    for beg in tqdm(range(0, seq, STRIDE), desc="AR"):
        end = min(beg + max_len, seq)
        trg = end - beg if beg == 0 else min(STRIDE, seq - beg)
        ids = enc.input_ids[:, beg:end].to(d)
        lbl = ids.clone()
        lbl[:, :-trg] = -100
        with torch.no_grad():
            nll += m(ids, labels=lbl).loss.item() * trg
        toks += trg
    bits = nll / math.log(2)

    recon = tok.decode(enc.input_ids[0], skip_special_tokens=True)
    acc = token_acc(text, recon, tok)

    return bits / toks, bits / n_chars, m_sz, bits / BITS, acc


def chunks(t, n):
    return [t[i : i + n] for i in range(0, len(t), n)]


def pm(model_name, text, n_chars, cache, mask=0.5):
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache))
    m = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=str(cache))
    d = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(d).eval()
    m_sz = dir_size(hf_dir(model_name, cache))
    comps, toks = [], 0
    for blk in chunks(text, 4096):
        comps.append(predictive_masking_compress(blk, tok, mask))
        toks += len(tok(blk).input_ids)
    payload = "".join(comps).encode()
    rec = decompress_text("".join(comps), m, tok, d)
    acc = token_acc(text, rec, tok)
    bits = len(payload) * BITS
    return bits / toks, bits / n_chars, m_sz, len(payload), acc


def lsq(model_name, text, n_chars, cache, max_tok=512):
    d = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache))
    enc_m = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=str(cache)).to(d)
    enc_m.eval()
    dec_p = Path(__file__).resolve().parents[2] / "E1/E1A/models" / f"{model_name.split('/')[-1]}_lsq_decoder.pth"
    if not dec_p.exists():
        return "miss", "miss", 0, 0, 0
    dec = LSQDecoder(enc_m.config.hidden_size, enc_m.config.vocab_size).to(d)
    dec.load_state_dict(torch.load(dec_p, map_location=d))
    dec.eval()
    m_sz = dir_size(hf_dir(model_name, cache)) + dec_p.stat().st_size
    total_b, toks = 0, 0
    rec_out = []
    for blk in chunks(text, max_tok * 4):
        inp = tok(blk, return_tensors="pt", truncation=True, max_length=max_tok).to(d)
        if inp.input_ids.numel() == 0:
            continue
        toks += inp.input_ids.numel()
        with torch.no_grad():
            out = enc_m(**inp, output_hidden_states=True, return_dict=True)
            lat = out.hidden_states[-1]
            scale = float((lat.max() - lat.min()) / 255) or 1.0
            q = torch.quantize_per_tensor(lat, scale, 0, torch.qint8)
            total_b += q.int_repr().numel()
            logits = dec(q.dequantize())
            preds = torch.argmax(logits, -1)
        rec_out.append(tok.batch_decode(preds, skip_special_tokens=True)[0])
    rec_text = "".join(rec_out)
    acc = token_acc(text, rec_text, tok)
    bits = total_b * BITS
    return bits / toks, bits / n_chars, m_sz, total_b, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--dataset-file", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=Path("./cache"))
    p.add_argument("--base-model-size", type=int, default=None)
    a = p.parse_args()

    n_chars, text = raw_stats(a.dataset_file)
    alg_res = alg(a.dataset_file, n_chars)

    llm = []
    if "bert" in a.model_name.lower() or "roberta" in a.model_name.lower():
        llm.append(("PM",) + pm(a.model_name, text, n_chars, a.cache_dir))
        llm.append(("LSQ",) + lsq(a.model_name, text, n_chars, a.cache_dir))
    else:
        llm.append(("AR",) + ar(a.model_name, text, n_chars, a.cache_dir))

    print("\nLLM compressors")
    for label, bpt, bpc, msz, psz, acc in llm:
        if isinstance(bpt, str):
            print(f"{label:<6}: {bpt}")
            continue
        print(f"{label:<6}: {bpt:8.4f} bits/tok | {bpc:8.4f} bits/char | acc {acc:.3f} | payload {hb(psz):>9} | model {hb(msz):>9}")

    print("\nTraditional compressors")
    for n, (bpc, sz) in alg_res.items():
        if math.isnan(bpc):
            print(f"{n.upper():<6}: missing")
        else:
            print(f"{n.upper():<6}: {bpc:8.4f} bits/char | {hb(sz):>9}")


if __name__ == "__main__":
    main()

