import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from cycler import cycler

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

# Professional color palette
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
MARKERS = ['o', 's', 'D', '^']

# Set the style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14

# Regular expression for parsing log files
epoch_re = re.compile(
    r'Epoch\s+\d+/\d+\s+\|\s+Dur: [\d\.]+s\s+\|\s+'
    r'Train Loss:\s*([\d\.]+),\s*Acc:\s*([\d\.]+)\s+\|\s+'
    r'Val Loss:\s*([\d\.]+),\s*Acc:\s*([\d\.]+)'
)

def parse_log(filename):
    """Parse log file and extract training metrics."""
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

def plot_loss_comparison(data):
    """Create a professional loss comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Set custom color cycles for train and validation
    ax.set_prop_cycle(cycler('color', COLORS))
    
    for i, model in enumerate(MODEL_NAMES):
        rec = data[model]
        epochs = range(1, len(rec['train_loss']) + 1)
        
        # Plot with markers at data points
        ax.plot(epochs, rec['train_loss'], label=f'{model} (Train)', 
                linewidth=2, marker=MARKERS[i], markersize=5, markevery=5)
        ax.plot(epochs, rec['val_loss'], label=f'{model} (Val)', 
                linewidth=1.5, linestyle='--', marker=MARKERS[i], 
                markersize=4, markevery=5, alpha=0.8)
    
    # Enhance the plot
    ax.set_xlabel('Epoch', weight='bold')
    ax.set_ylabel('Loss', weight='bold')
    ax.set_title('Training and Validation Loss Comparison', fontsize=14, weight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # X-axis as integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Create a nice legend
    legend = ax.legend(loc='upper right', frameon=True, framealpha=0.95, 
                       fancybox=True, shadow=True, ncol=2)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('#E0E0E0')
    
    # Add a light box around the plot area
    ax.patch.set_edgecolor('#E0E0E0')
    ax.patch.set_linewidth(1)
    
    plt.tight_layout()
    return fig

def plot_accuracy_subplots(data):
    """Create professional accuracy subplots."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150, sharey=True)
    
    for i, (ax, model) in enumerate(zip(axes, MODEL_NAMES)):
        rec = data[model]
        epochs = range(1, len(rec['train_acc']) + 1)
        
        # Plot with markers
        ax.plot(epochs, rec['train_acc'], color=COLORS[i], linewidth=2, 
                label='Training', marker='o', markersize=5, markevery=5)
        ax.plot(epochs, rec['val_acc'], color=COLORS[i], linewidth=1.5, 
                linestyle='--', label='Validation', marker='s', 
                markersize=4, markevery=5, alpha=0.8)
        
        # Set limits and formatting
        ax.set_ylim(0.5, 1.0)  # Assuming accuracy is between 0.5 and 1.0
        ax.set_title(model, fontsize=12, weight='bold')
        ax.set_xlabel('Epoch', weight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # X-axis as integers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add a legend
        legend = ax.legend(loc='lower right', frameon=True)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('#E0E0E0')
    
    # Add y-label only to the first subplot
    axes[0].set_ylabel('Accuracy', weight='bold')
    
    # Add a main title
    fig.suptitle('Training and Validation Accuracy by Model', 
                 fontsize=16, weight='bold', y=1.05)
    
    plt.tight_layout()
    return fig

def create_summary_plot(data):
    """Create a summary plot showing final validation metrics."""
    # Extract final validation metrics
    val_acc_final = [data[model]['val_acc'][-1] for model in MODEL_NAMES]
    val_loss_final = [data[model]['val_loss'][-1] for model in MODEL_NAMES]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # Bar chart for final validation accuracy
    bars1 = ax1.bar(MODEL_NAMES, val_acc_final, color=COLORS, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    ax1.set_title('Final Validation Accuracy', fontsize=14, weight='bold')
    ax1.set_ylabel('Accuracy', weight='bold')
    ax1.set_ylim(min(val_acc_final) - 0.05, 1.0)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Bar chart for final validation loss
    bars2 = ax2.bar(MODEL_NAMES, val_loss_final, color=COLORS, alpha=0.8,
                   edgecolor='black', linewidth=1)
    ax2.set_title('Final Validation Loss', fontsize=14, weight='bold')
    ax2.set_ylabel('Loss', weight='bold')
    ax2.set_ylim(0, max(val_loss_final) + 0.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Rotate x-tick labels for better readability
    for ax in [ax1, ax2]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    plt.tight_layout()
    return fig

def plot_and_save(data):
    """Create and save all visualizations."""
    # Loss comparison plot
    loss_fig = plot_loss_comparison(data)
    loss_fig.savefig('loss_comparison.png', bbox_inches='tight', dpi=300)
    plt.close(loss_fig)
    print("Saved enhanced loss plot to loss_comparison.png")
    
    # Accuracy subplots
    acc_fig = plot_accuracy_subplots(data)
    acc_fig.savefig('accuracy_curves.png', bbox_inches='tight', dpi=300)
    plt.close(acc_fig)
    print("Saved enhanced accuracy curves to accuracy_curves.png")
    
    # Summary plot (optional but useful)
    summary_fig = create_summary_plot(data)
    summary_fig.savefig('model_summary.png', bbox_inches='tight', dpi=300)
    plt.close(summary_fig)
    print("Saved model summary comparison to model_summary.png")

if __name__ == '__main__':
    metrics = parse_log(LOG_FILE)
    plot_and_save(metrics)
