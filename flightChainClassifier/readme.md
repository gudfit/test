# ✈️ Flight-Chain Delay Classifier

A lightweight pipeline that

1. **builds chains** of three consecutive flights operated by the **same aircraft (tail number)**  
2. **optionally augments** them with small noise (“jitter”)  
3. **trains** a CNN + LSTM (with attention) to predict the **delay class** of the 3ʳᵈ flight.

<p align="center">
\[
\underbrace{\bigl[\;x^{(1)},\,x^{(2)},\,x^{(3)}\bigr]}_{\text{chain}}\;
\;\xrightarrow{\;f_\theta\;}\;
\hat y\in\{0,1,2,3,4\}
\]
</p>

---

## Installation

```bash
git clone https://github.com/your-handle/flightChainClassifier.git
cd flightChainClassifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

*Dataset* - drop your merged **`0_merged_raw_flights.csv`** into `data/`.

---

## Quick start

Run the full 3-stage pipeline (process ➞ train ➞ evaluate):

```bash
python -m src.main                # default = SimAM model, 50 epochs
```

Skip stages if you already have the artefacts:

```bash
# Just retrain & re-evaluate
python -m src.main --skip-data --model qtsimam --epochs n
```

---

### CLI flags (most common)

| flag | description | default |
|------|-------------|---------|
| `--model {cbam,simam,qtsimam}` | choose architecture | `simam` |
| `--epochs N`        | training epochs | `50` |
| `--batch-size N`    | mini-batch size | `32` |
| `--sim-factor k`    | create **k** jittered copies per chain (`k=1` → **no** jitter) | `3` |
| `--no-aug`          | shortcut for `--sim-factor 1` | – |
| `--lstm-layers n`   | 1 → hidden = 128, ≥2 → hidden = 256 (auto) | `auto` |
| `--balanced`        | oversample minority delay classes | off |
| `--skip-data / --skip-train / --skip-eval` | skip pipeline stages | off |

---

##  Maths

### 1.  Label mapping  

Let raw delay of flight 3 be \(d=\text{ArrDelayMinutes}_{(3)}\).

\[
y=\begin{cases}
0, & d\le 15 \\
1, & 15<d\le 60 \\
2, & 60<d\le 120\\
3, & 120<d\le 240\\
4, & d>240
\end{cases}
\]

### 2.  Synthetic jitter (optional)

For each numeric feature \(x\) we sample

\[
\tilde x = x + \varepsilon,
\qquad
\varepsilon \sim \mathcal N\!\bigl(0,\;0.05^{2}\bigr)
\]

producing *k* extra chains per original.

### 3.  Loss

Cross-entropy with optional class weights \(w_c\):

\[
\mathcal L = -\sum_{c=0}^{4} w_c\,\mathbf 1_{[y=c]}\,\log p_{\theta}(y=c\mid x)
\]

---

## Layout

```
src/
 ├─ data_processing/
 │   └─ chain_constructor.py     # build & split chains
 ├─ training/
 │   ├─ dataset.py               # npy → PyTorch Dataset
 │   ├─ trainer.py               # training loop
 │   └─ *                        # CNN/LSTM code
 ├─ evaluation/
 │   └─ evaluate.py              # metrics + confusion-matrix
 ├─ modeling/                    # attention modules & models
 └─ main.py                      # orchestrates the three stages
data/                             # put raw CSV here
results/
 ├─ models/                      # best .pt + meta.json
 ├─ evaluation/                  # metrics .json
 └─ plots/                       # confusion matrix, Optuna plots
```
