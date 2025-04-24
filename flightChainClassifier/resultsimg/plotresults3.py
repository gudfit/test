import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
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

# Data from the classification reports
accuracies = [0.9294, 0.9062, 0.8823, 0.9064]

# F1-scores by class for each model
f1_scores = [
    [0.96, 0.80, 0.84, 0.92, 0.89],  # SimAM (no bidir)
    [0.95, 0.75, 0.80, 0.89, 0.87],  # SimAM (bidir)
    [0.93, 0.70, 0.85, 0.91, 0.88],  # QTSimAM (no bidir)
    [0.95, 0.75, 0.87, 0.92, 0.90]   # QTSimAM (bidir)
]

# Recall values by class for each model
recall_values = [
    [0.93, 0.92, 0.92, 0.97, 0.91],  # SimAM (no bidir)
    [0.91, 0.89, 0.93, 0.94, 0.91],  # SimAM (bidir)
    [0.87, 0.92, 0.96, 0.91, 0.86],  # QTSimAM (no bidir)
    [0.90, 0.94, 0.94, 0.94, 0.82]   # QTSimAM (bidir)
]

# Precision values by class for each model
precision_values = [
    [1.00, 0.71, 0.77, 0.88, 0.87],  # SimAM (no bidir)
    [1.00, 0.64, 0.70, 0.84, 0.83],  # SimAM (bidir)
    [1.00, 0.57, 0.77, 0.90, 0.90],  # QTSimAM (no bidir)
    [1.00, 0.63, 0.80, 0.90, 1.00]   # QTSimAM (bidir)
]

# Create the figure with custom layout
fig = plt.figure(figsize=(15, 12), dpi=300)
gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])

# Plot 1: Overall Accuracy Bar Chart
ax1 = fig.add_subplot(gs[0, :])

# Create bars with custom appearance
bars = ax1.bar(
    MODEL_NAMES, 
    accuracies, 
    color=COLORS, 
    edgecolor='black', 
    linewidth=1,
    alpha=0.85,
    width=0.6
)

# Customize the appearance
ax1.set_title('Overall Model Accuracy', fontweight='bold', pad=15)
ax1.set_ylabel('Accuracy', fontweight='bold')
ax1.set_ylim(0.87, 0.94)  # Focused range to highlight differences
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2, 
        height + 0.001,
        f'{height:.4f}', 
        ha='center', 
        va='bottom', 
        fontsize=10,
        fontweight='bold'
    )

# Highlight best model with a star
best_model_idx = np.argmax(accuracies)
ax1.text(
    best_model_idx,
    accuracies[best_model_idx] + 0.004,
    '★',
    ha='center',
    va='bottom',
    fontsize=20,
    color='gold'
)

# Plot 2: Class 2 F1-scores (where QTSimAM has an advantage)
ax2 = fig.add_subplot(gs[1, 0])

# Extract Class 2 F1-scores and create a focused bar chart
class_2_f1 = [model_scores[2] for model_scores in f1_scores]

bars2 = ax2.bar(
    MODEL_NAMES, 
    class_2_f1, 
    color=COLORS, 
    edgecolor='black', 
    linewidth=1,
    alpha=0.85,
    width=0.6
)

# Highlight QTSimAM advantage
ax2.set_title('Class 2 F1-Score', fontweight='bold')
ax2.set_ylabel('F1-Score', fontweight='bold')
ax2.set_ylim(0.79, 0.88)  # Focused range
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Rotate x-labels for better fit
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width()/2, 
        height + 0.001,
        f'{height:.2f}', 
        ha='center', 
        va='bottom', 
        fontsize=9,
        fontweight='bold'
    )

# Plot 3: Class 4 F1-scores (where QTSimAM has an advantage)
ax3 = fig.add_subplot(gs[1, 1])

# Extract Class 4 F1-scores
class_4_f1 = [model_scores[4] for model_scores in f1_scores]

bars3 = ax3.bar(
    MODEL_NAMES, 
    class_4_f1, 
    color=COLORS, 
    edgecolor='black', 
    linewidth=1,
    alpha=0.85,
    width=0.6
)

ax3.set_title('Class 4 F1-Score', fontweight='bold')
ax3.set_ylabel('F1-Score', fontweight='bold')
ax3.set_ylim(0.86, 0.91)  # Focused range
ax3.grid(axis='y', linestyle='--', alpha=0.7)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Rotate x-labels for better fit
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width()/2, 
        height + 0.001,
        f'{height:.2f}', 
        ha='center', 
        va='bottom', 
        fontsize=9,
        fontweight='bold'
    )

# Plot 4: Class 2 Recall (where QTSimAM no bidir has an advantage)
ax4 = fig.add_subplot(gs[1, 2])

# Extract Class 2 Recall values
class_2_recall = [model_recall[2] for model_recall in recall_values]

bars4 = ax4.bar(
    MODEL_NAMES, 
    class_2_recall, 
    color=COLORS, 
    edgecolor='black', 
    linewidth=1,
    alpha=0.85,
    width=0.6
)

ax4.set_title('Class 2 Recall', fontweight='bold')
ax4.set_ylabel('Recall', fontweight='bold')
ax4.set_ylim(0.91, 0.97)  # Focused range
ax4.grid(axis='y', linestyle='--', alpha=0.7)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Rotate x-labels for better fit
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Add value labels
for bar in bars4:
    height = bar.get_height()
    ax4.text(
        bar.get_x() + bar.get_width()/2, 
        height + 0.001,
        f'{height:.2f}', 
        ha='center', 
        va='bottom', 
        fontsize=9,
        fontweight='bold'
    )

# Plot 5: Comparative Advantages Grid - Highlight where QTSimAM excels
ax5 = fig.add_subplot(gs[2, :])

# Create a table-like visualization showing advantages
# We'll highlight metrics where QTSimAM models outperform SimAM models

# Categories where QTSimAM has advantages
categories = [
    "Class 2 F1", 
    "Class 4 F1", 
    "Class 2 Recall", 
    "Class 1 Recall",
    "Class 3 Precision",
    "Class 4 Precision"
]

# Create advantage data as percentage improvements
advantages = [
    # QTSimAM (bidir) vs Best SimAM for Class 2 F1
    ((f1_scores[3][2] / max(f1_scores[0][2], f1_scores[1][2])) - 1) * 100,
    
    # QTSimAM (bidir) vs Best SimAM for Class 4 F1
    ((f1_scores[3][4] / max(f1_scores[0][4], f1_scores[1][4])) - 1) * 100,
    
    # QTSimAM (no bidir) vs Best SimAM for Class 2 Recall
    ((recall_values[2][2] / max(recall_values[0][2], recall_values[1][2])) - 1) * 100,
    
    # QTSimAM (bidir) vs Best SimAM for Class 1 Recall
    ((recall_values[3][1] / max(recall_values[0][1], recall_values[1][1])) - 1) * 100,
    
    # QTSimAM (no bidir) vs Best SimAM for Class 3 Precision
    ((precision_values[2][3] / max(precision_values[0][3], precision_values[1][3])) - 1) * 100,
    
    # QTSimAM (bidir) vs Best SimAM for Class 4 Precision
    ((precision_values[3][4] / max(precision_values[0][4], precision_values[1][4])) - 1) * 100
]

# QTSimAM models that have these advantages
models = [
    "QTSimAM (bidir)",
    "QTSimAM (bidir)",
    "QTSimAM (no bidir)",
    "QTSimAM (bidir)",
    "QTSimAM (no bidir)",
    "QTSimAM (bidir)"
]

# Create horizontal bar chart with percentage improvements
y_pos = np.arange(len(categories))
bars5 = ax5.barh(
    y_pos,
    advantages,
    color='#d62728',  # Red color for emphasis
    alpha=0.7,
    edgecolor='black',
    linewidth=1,
    height=0.5
)

# Add labels for each bar showing which QTSimAM model has this advantage
for i, (model, adv) in enumerate(zip(models, advantages)):
    ax5.text(
        adv + 0.2, 
        i, 
        f"{model}", 
        va='center',
        fontsize=9
    )
    
    # Add percentage at the end of each bar
    ax5.text(
        0.1,  # Place inside bar
        i,
        f"+{adv:.1f}%", 
        va='center',
        ha='left',
        fontsize=9,
        fontweight='bold',
        color='white'
    )

ax5.set_yticks(y_pos)
ax5.set_yticklabels(categories)
ax5.set_xlabel('Percentage Improvement Over Best SimAM Model', fontweight='bold')
ax5.set_title('QTSimAM Advantages', fontweight='bold', pad=15)
ax5.set_xlim(0, max(advantages) + 7)  # Add some padding
ax5.grid(axis='x', linestyle='--', alpha=0.7)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# Add a custom legend to explain the visualization
legend_elements = [
    Patch(facecolor=COLORS[0], edgecolor='black', label='SimAM (no bidir)'),
    Patch(facecolor=COLORS[1], edgecolor='black', label='SimAM (bidir)'),
    Patch(facecolor=COLORS[2], edgecolor='black', label='QTSimAM (no bidir)'),
    Patch(facecolor=COLORS[3], edgecolor='black', label='QTSimAM (bidir)'),
]

fig.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.05),
    ncol=4,
    frameon=True
)

# Title for the entire figure
fig.suptitle(
    'Model Comparison: QTSimAM vs SimAM Performance Analysis',
    fontsize=16,
    fontweight='bold',
    y=0.98
)

# Add a subtitle explaining the key takeaway
plt.figtext(
    0.5, 0.955, 
    'While SimAM (no bidir) has the highest overall accuracy, QTSimAM models excel in specific classes',
    ha='center',
    fontsize=12,
    style='italic'
)

# Adjust spacing
plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.subplots_adjust(hspace=0.4)

# Save the figure
plt.savefig('qtsimam_advantages.png', dpi=300, bbox_inches='tight')

print("Visualization saved as 'qtsimam_advantages.png'")
