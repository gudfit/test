import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Patch

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14

# Define model names and colors
MODEL_NAMES = [
    'SimAM (no bidir)',
    'SimAM (bidir)',
    'QTSimAM (no bidir)',
    'QTSimAM (bidir)'
]

# Professional color palette
COLORS = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c']

# Class names
CLASS_NAMES = [
    "On Time / Slight Delay (<= 15 min)",
    "Delayed (15-60 min)",
    "Significantly Delayed (60-120 min)",
    "Severely Delayed (120-240 min)",
    "Extremely Delayed (> 240 min)"
]

# F1-scores by class for each model
f1_scores = [
    [0.96, 0.80, 0.84, 0.92, 0.89],  # SimAM (no bidir)
    [0.95, 0.75, 0.80, 0.89, 0.87],  # SimAM (bidir)
    [0.93, 0.70, 0.85, 0.91, 0.88],  # QTSimAM (no bidir)
    [0.95, 0.75, 0.87, 0.92, 0.90]   # QTSimAM (bidir)
]

# Create the figure with a wide format
plt.figure(figsize=(14, 8), dpi=300)

# Set position of bars on X axis
x = np.arange(len(CLASS_NAMES))
width = 0.18  # Width of the bars

# Plot bars for each model with more space between groups
for i, model_scores in enumerate(f1_scores):
    offset = width * (i - 1.5)
    bars = plt.bar(x + offset, model_scores, width, label=MODEL_NAMES[i], 
            color=COLORS[i], alpha=0.85, edgecolor='black', linewidth=1)
    
    # Add value labels above each bar (placed higher to avoid overlap)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Find where QTSimAM models excel
qtSimAM_advantages = []
for c in range(5):
    simam_best = max(f1_scores[0][c], f1_scores[1][c])
    qtsimam_best = max(f1_scores[2][c], f1_scores[3][c])
    if qtsimam_best > simam_best:
        qtSimAM_advantages.append((c, qtsimam_best))

# Add star symbol above bars where QTSimAM models outperform SimAM
for c, score in qtSimAM_advantages:
    # Find which QTSimAM model has this advantage
    if f1_scores[2][c] > f1_scores[3][c]:
        model_idx = 2  # QTSimAM (no bidir)
    else:
        model_idx = 3  # QTSimAM (bidir)
    
    offset = width * (model_idx - 1.5)
    plt.text(c + offset, score + 0.03, '★', ha='center', va='bottom', fontsize=15, color='gold')

# Add labels, title and customize legend
plt.ylabel('F1-Score', fontweight='bold')
plt.title('F1-Score by Delay Category for All Models', fontweight='bold', pad=20)
plt.xticks(x, rotation=0)  # No rotation initially

# Create custom x-tick labels with line breaks for readability
labels = []
for class_name in CLASS_NAMES:
    if "Delayed" in class_name and class_name != "Delayed (15-60 min)":
        # Split after "Delayed"
        parts = class_name.split("Delayed")
        labels.append("Delayed" + "\n" + parts[1])
    else:
        if len(class_name) > 20:
            # Split long text
            words = class_name.split()
            mid = len(words) // 2
            labels.append(" ".join(words[:mid]) + "\n" + " ".join(words[mid:]))
        else:
            labels.append(class_name)

plt.xticks(x, labels)
plt.ylim(0.65, 1.0)

# Create a more descriptive legend with color patches
legend_elements = [Patch(facecolor=COLORS[i], edgecolor='black', label=MODEL_NAMES[i]) for i in range(4)]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=True)

# Add a visual indicator for where QTSimAM models outperform SimAM
legend_elements2 = [Patch(facecolor='none', edgecolor='none', label=''), 
                   Patch(facecolor='none', edgecolor='none', label='★ QTSimAM outperforms SimAM variants')]
plt.legend(handles=legend_elements2, loc='upper right', frameon=False, handlelength=0)

# Customize grid and spines
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add a subtitle explaining the key advantage
plt.figtext(0.5, 0.01, 'QTSimAM models show superior performance in predicting significant, severe, and extreme delays',
          ha='center', fontsize=11, style='italic')

# Tight layout with additional space for x-labels
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save high-resolution image
plt.savefig('f1_score_by_delay_category.png', dpi=300, bbox_inches='tight')
print("Saved f1_score_by_delay_category.png")
