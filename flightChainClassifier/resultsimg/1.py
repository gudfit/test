import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

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

# Plot 1: Overall Accuracy Bar Chart
def plot_overall_accuracy():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Create bars with custom appearance
    bars = ax.bar(
        MODEL_NAMES, 
        accuracies, 
        color=COLORS, 
        edgecolor='black', 
        linewidth=1,
        alpha=0.85,
        width=0.6
    )
    
    # Customize the appearance
    ax.set_title('Overall Model Accuracy', fontweight='bold', pad=15)
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_ylim(0.87, 0.94)  # Focused range to highlight differences
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
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
    ax.text(
        best_model_idx,
        accuracies[best_model_idx] + 0.004,
        '★',
        ha='center',
        va='bottom',
        fontsize=20,
        color='gold'
    )
    
    plt.tight_layout()
    plt.savefig('plot1_overall_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot1_overall_accuracy.png")

# Plot 2: Class 2 F1-scores (where QTSimAM has an advantage)
def plot_class2_f1():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Extract Class 2 F1-scores and create a focused bar chart
    class_2_f1 = [model_scores[2] for model_scores in f1_scores]
    
    bars = ax.bar(
        MODEL_NAMES, 
        class_2_f1, 
        color=COLORS, 
        edgecolor='black', 
        linewidth=1,
        alpha=0.85,
        width=0.6
    )
    
    # Highlight QTSimAM advantage
    ax.set_title('Class 2 F1-Score', fontweight='bold')
    ax.set_ylabel('F1-Score', fontweight='bold')
    ax.set_ylim(0.79, 0.88)  # Focused range
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height + 0.001,
            f'{height:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=9,
            fontweight='bold'
        )
    
    # Highlight the QTSimAM models with gold edge
    bars[2].set_edgecolor('gold')  # QTSimAM (no bidir)
    bars[2].set_linewidth(2)
    bars[3].set_edgecolor('gold')  # QTSimAM (bidir)
    bars[3].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('plot2_class2_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot2_class2_f1.png")

# Plot 3: Class 4 F1-scores (where QTSimAM has an advantage)
def plot_class4_f1():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Extract Class 4 F1-scores
    class_4_f1 = [model_scores[4] for model_scores in f1_scores]
    
    bars = ax.bar(
        MODEL_NAMES, 
        class_4_f1, 
        color=COLORS, 
        edgecolor='black', 
        linewidth=1,
        alpha=0.85,
        width=0.6
    )
    
    ax.set_title('Class 4 F1-Score', fontweight='bold')
    ax.set_ylabel('F1-Score', fontweight='bold')
    ax.set_ylim(0.86, 0.91)  # Focused range
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height + 0.001,
            f'{height:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=9,
            fontweight='bold'
        )
    
    # Highlight the QTSimAM (bidir) model with gold edge
    bars[3].set_edgecolor('gold')  # QTSimAM (bidir)
    bars[3].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('plot3_class4_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot3_class4_f1.png")

# Plot 4: Class 2 Recall (where QTSimAM no bidir has an advantage)
def plot_class2_recall():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Extract Class 2 Recall values
    class_2_recall = [model_recall[2] for model_recall in recall_values]
    
    bars = ax.bar(
        MODEL_NAMES, 
        class_2_recall, 
        color=COLORS, 
        edgecolor='black', 
        linewidth=1,
        alpha=0.85,
        width=0.6
    )
    
    ax.set_title('Class 2 Recall', fontweight='bold')
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_ylim(0.91, 0.97)  # Focused range
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height + 0.001,
            f'{height:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=9,
            fontweight='bold'
        )
    
    # Highlight the QTSimAM (no bidir) model with gold edge
    bars[2].set_edgecolor('gold')  # QTSimAM (no bidir)
    bars[2].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('plot4_class2_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot4_class2_recall.png")

# Plot 5: Class 1 Recall (where QTSimAM bidir has an advantage)
def plot_class1_recall():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Extract Class 1 Recall values
    class_1_recall = [model_recall[1] for model_recall in recall_values]
    
    bars = ax.bar(
        MODEL_NAMES, 
        class_1_recall, 
        color=COLORS, 
        edgecolor='black', 
        linewidth=1,
        alpha=0.85,
        width=0.6
    )
    
    ax.set_title('Class 1 Recall', fontweight='bold')
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_ylim(0.88, 0.95)  # Focused range
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height + 0.001,
            f'{height:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=9,
            fontweight='bold'
        )
    
    # Highlight the QTSimAM (bidir) model with gold edge
    bars[3].set_edgecolor('gold')  # QTSimAM (bidir)
    bars[3].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('plot5_class1_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot5_class1_recall.png")

# Plot 6: Comparative Advantages Summary
def plot_advantages_summary():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # Categories where QTSimAM has advantages
    categories = [
        "Class 2 F1", 
        "Class 4 F1", 
        "Class 2 Recall", 
        "Class 1 Recall",
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
        
        # QTSimAM (bidir) vs Best SimAM for Class 4 Precision
        ((precision_values[3][4] / max(precision_values[0][4], precision_values[1][4])) - 1) * 100
    ]
    
    # QTSimAM models that have these advantages
    models = [
        "QTSimAM (bidir)",
        "QTSimAM (bidir)",
        "QTSimAM (no bidir)",
        "QTSimAM (bidir)",
        "QTSimAM (bidir)"
    ]
    
    # Create horizontal bar chart with percentage improvements
    y_pos = np.arange(len(categories))
    bars = ax.barh(
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
        ax.text(
            adv + 0.2, 
            i, 
            f"{model}", 
            va='center',
            fontsize=10
        )
        
        # Add percentage at the end of each bar
        ax.text(
            0.3,  # Place inside bar
            i,
            f"+{adv:.1f}%", 
            va='center',
            ha='left',
            fontsize=11,
            fontweight='bold',
            color='white'
        )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Percentage Improvement Over Best SimAM Model', fontweight='bold')
    ax.set_title('QTSimAM Advantages', fontweight='bold', pad=15)
    ax.set_xlim(0, max(advantages) + 7)  # Add some padding
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('plot6_advantages_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot6_advantages_summary.png")

# Plot 7: All F1 scores across classes
def plot_all_f1_scores():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # Set position of bars on X axis
    class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    x = np.arange(len(class_labels))
    width = 0.2  # Width of the bars
    
    # Plot bars for each model
    for i, model_scores in enumerate(f1_scores):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, model_scores, width, label=MODEL_NAMES[i], 
                color=COLORS[i], alpha=0.85, edgecolor='black', linewidth=1)
        
        # Add value labels above each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Add labels, title and legend
    ax.set_ylabel('F1-Score', fontweight='bold')
    ax.set_title('F1-Score by Class for All Models', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    ax.set_ylim(0.65, 1.0)
    
    # Customize grid and spines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotations for QTSimAM advantages
    ax.annotate('QTSimAM (bidir)\nadvantage', xy=(2 + width * 1.5, f1_scores[3][2]), 
                xytext=(2 + width * 1.5, f1_scores[3][2] + 0.07),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                ha='center', fontsize=9)
    
    ax.annotate('QTSimAM (bidir)\nadvantage', xy=(4 + width * 1.5, f1_scores[3][4]), 
                xytext=(4 + width * 1.5, f1_scores[3][4] + 0.07),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plot7_all_f1_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot7_all_f1_scores.png")
    
# Execute all plot functions
plot_overall_accuracy()
plot_class2_f1()
plot_class4_f1()
plot_class2_recall()
plot_class1_recall()
plot_advantages_summary()
plot_all_f1_scores()

print("All plots successfully saved as individual image files!")
