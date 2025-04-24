import re
import matplotlib.pyplot as plt
from collections import defaultdict

# --- CONFIG ---
LOG_FILE = 'results.txt'

# The sections in the log in order of appearance
SECTIONS = [
    'simam (no bidrecction)',
    'simam (bidirection)',
    'qtsimam (nobidirection)',
    'qtsimam (with bidirection)'
]
# Friendly names for plotting
MODEL_NAMES = [
    'SimAM (no bidir)',
    'SimAM (bidir)',
    'QTSimAM (no bidir)',
    'QTSimAM (bidir)'
]

# Regex to capture epoch lines:
#   Epoch 01/50 | Dur: 50.08s | Train Loss: 1.5339, Acc: 0.5885 | Val Loss: 1.5286, Acc: 0.5080
epoch_re = re.compile(
    r'Epoch\s+\d+/\d+\s+\|\s+Dur: [\d\.]+s\s+\|\s+'
    r'Train Loss:\s*([\d\.]+),\s*Acc:\s*([\d\.]+)\s+\|\s+'
    r'Val Loss:\s*([\d\.]+),\s*Acc:\s*([\d\.]+)'
)

def parse_log(filename):
    data = {
        name: {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        for name in MODEL_NAMES
    }
    current_section = None
    section_idx = -1

    with open(filename, 'r') as f:
        for line in f:
            # Check if this line marks the start of a section
            lower = line.strip().lower()
            for idx, sec in enumerate(SECTIONS):
                if sec in lower:
                    current_section = MODEL_NAMES[idx]
                    section_idx = idx
                    break

            # If inside a known section, try to parse an epoch line
            if current_section:
                m = epoch_re.search(line)
                if m:
                    t_loss, t_acc, v_loss, v_acc = map(float, m.groups())
                    rec = data[current_section]
                    rec['train_loss'].append(t_loss)
                    rec['train_acc']. append(t_acc)
                    rec['val_loss'].append(v_loss)
                    rec['val_acc'].append(v_acc)

    return data

def plot_metrics(data):
    # --- Plot Loss ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    for ax, model in zip(axes, MODEL_NAMES):
        rec = data[model]
        epochs = range(1, len(rec['train_loss']) + 1)
        ax.plot(epochs, rec['train_loss'], label='Train Loss')
        ax.plot(epochs, rec['val_loss'],   label='Val Loss')
        ax.set_title(model)
        ax.set_xlabel('Epoch')
        if ax is axes[0]:
            ax.set_ylabel('Loss')
        ax.legend()
    fig.suptitle('Training vs Validation Loss')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Plot Accuracy ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    for ax, model in zip(axes, MODEL_NAMES):
        rec = data[model]
        epochs = range(1, len(rec['train_acc']) + 1)
        ax.plot(epochs, rec['train_acc'], label='Train Acc')
        ax.plot(epochs, rec['val_acc'],   label='Val Acc')
        ax.set_title(model)
        ax.set_xlabel('Epoch')
        if ax is axes[0]:
            ax.set_ylabel('Accuracy')
        ax.legend()
    fig.suptitle('Training vs Validation Accuracy')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

if __name__ == '__main__':
    metrics = parse_log(LOG_FILE)
    plot_metrics(metrics)

