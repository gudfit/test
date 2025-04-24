import re
import matplotlib.pyplot as plt

# --- CONFIG ---
LOG_FILE = 'results.txt'

SECTIONS = [
    'simam (no bidrecction)',
    'simam (bidirection)',
    'qtsimam (nobidirection)',
    'qtsimam (with bidirection)'
]
MODEL_NAMES = [
    'SimAM (no bidir)',
    'SimAM (bidir)',
    'QTSimAM (no bidir)',
    'QTSimAM (bidir)'
]

epoch_re = re.compile(
    r'Epoch\s+\d+/\d+\s+\|\s+Dur: [\d\.]+s\s+\|\s+'
    r'Train Loss:\s*([\d\.]+),\s*Acc:\s*([\d\.]+)\s+\|\s+'
    r'Val Loss:\s*([\d\.]+),\s*Acc:\s*([\d\.]+)'
)

def parse_log(filename):
    data = {name: {'train_loss': [], 'val_loss': [],
                   'train_acc': [], 'val_acc': []}
            for name in MODEL_NAMES}
    current = None
    with open(filename) as f:
        for line in f:
            low = line.lower()
            for idx, sec in enumerate(SECTIONS):
                if sec in low:
                    current = MODEL_NAMES[idx]
                    break
            if current:
                m = epoch_re.search(line)
                if m:
                    t_loss, t_acc, v_loss, v_acc = map(float, m.groups())
                    rec = data[current]
                    rec['train_loss'].append(t_loss)
                    rec['val_loss'].append(v_loss)
                    rec['train_acc'].append(t_acc)
                    rec['val_acc'].append(v_acc)
    return data

def plot_and_save(data):
    # Combined Loss Plot
    plt.figure(figsize=(10, 6))
    for model in MODEL_NAMES:
        rec = data[model]
        epochs = range(1, len(rec['train_loss']) + 1)
        plt.plot(epochs, rec['train_loss'], label=f'{model} Train')
        plt.plot(epochs, rec['val_loss'],   label=f'{model} Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss (All Models)')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig('loss_comparison.png')
    plt.close()
    print("Saved combined loss plot to loss_comparison.png")

    # Accuracy Subplots (unchanged)
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
    fig.savefig('accuracy_curves.png')
    plt.close(fig)
    print("Saved accuracy curves to accuracy_curves.png")

if __name__ == '__main__':
    metrics = parse_log(LOG_FILE)
    plot_and_save(metrics)

