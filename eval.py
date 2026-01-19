import json
import os
from pathlib import Path
from collections import namedtuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Prompt names for labeling
PROMPT_NAMES = {
    0: "Baseline",
    1: "Strict JSON",
    2: "Structural Rigidity",
    3: "Python",
    4: "Oulipo",
    5: "Banned Words"
}

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('default')


def load_results_jsonl(filepath):
    """Load JSONL results into a DataFrame."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_all_results(data_dir="data"):
    """Load all JSONL files and combine into a single DataFrame."""
    data_path = Path(data_dir)
    jsonl_files = list(data_path.glob("results_*.jsonl"))
    
    if not jsonl_files:
        raise FileNotFoundError(f"No results files found in {data_dir}")
    
    print(f"Loading {len(jsonl_files)} result file(s)...")
    dfs = [load_results_jsonl(f) for f in jsonl_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} total results")
    return df


def create_summary_table(df):
    """Create summary statistics table by prompt."""
    summary = df.groupby('prompt').agg({
        'score': ['mean', 'sum', 'count'],
        'input_tokens': 'mean',
        'output_tokens': 'mean',
    }).round(2)
    
    summary.columns = ['accuracy', 'correct', 'total', 'avg_input_tokens', 'avg_output_tokens']
    summary['prompt_name'] = summary.index.map(PROMPT_NAMES)
    summary = summary[['prompt_name', 'accuracy', 'correct', 'total', 'avg_input_tokens', 'avg_output_tokens']]
    
    print("\n" + "="*80)
    print("Summary Statistics by Prompt")
    print("="*80)
    print(summary.to_string())
    print("="*80 + "\n")
    
    return summary


def plot_accuracy_by_prompt(df, output_dir="figures"):
    """Create bar plot of accuracy by prompt."""
    os.makedirs(output_dir, exist_ok=True)
    
    summary = df.groupby('prompt')['score'].mean().sort_index()
    prompt_labels = [PROMPT_NAMES.get(i, f"Prompt {i}") for i in summary.index]
    
    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(summary)), summary.values, 
                  color=plt.cm.viridis(np.linspace(0.2, 0.8, len(summary))))
    
    ax.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by Prompt Type on GPQA Diamond', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(prompt_labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, summary.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "accuracy_by_prompt.pdf")
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_with_error_bars(df, output_dir="figures"):
    """Create bar plot with error bars showing variance across repetitions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate mean and std across repetitions for each prompt
    grouped = df.groupby(['prompt', 'repetition'])['score'].mean().reset_index()
    summary_stats = grouped.groupby('prompt')['score'].agg(['mean', 'std', 'count']).sort_index()
    
    prompt_labels = [PROMPT_NAMES.get(i, f"Prompt {i}") for i in summary_stats.index]
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(summary_stats))
    bars = ax.bar(x_pos, summary_stats['mean'], 
                  yerr=summary_stats['std'],
                  capsize=5,
                  color=plt.cm.viridis(np.linspace(0.2, 0.8, len(summary_stats))),
                  alpha=0.8,
                  edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (Mean ± Std)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by Prompt Type with Standard Deviation Across Repetitions', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(prompt_labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, summary_stats['mean'], summary_stats['std'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.02,
                f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "accuracy_with_error_bars.pdf")
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_token_usage(df, output_dir="figures"):
    """Create comparison of token usage by prompt."""
    os.makedirs(output_dir, exist_ok=True)
    
    token_summary = df.groupby('prompt').agg({
        'input_tokens': 'mean',
        'output_tokens': 'mean'
    }).sort_index()
    
    prompt_labels = [PROMPT_NAMES.get(i, f"Prompt {i}") for i in token_summary.index]
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(token_summary))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, token_summary['input_tokens'], width,
                   label='Input Tokens', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, token_summary['output_tokens'], width,
                   label='Output Tokens', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Tokens', fontsize=12, fontweight='bold')
    ax.set_title('Token Usage by Prompt Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(prompt_labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "token_usage.pdf")
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_distribution(df, output_dir="figures"):
    """Create violin plot showing distribution of accuracy across questions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate accuracy per question for each prompt
    question_accuracy = df.groupby(['prompt', 'question_id'])['score'].mean().reset_index()
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    prompts = sorted(question_accuracy['prompt'].unique())
    data_to_plot = [question_accuracy[question_accuracy['prompt'] == p]['score'].values 
                    for p in prompts]
    prompt_labels = [PROMPT_NAMES.get(p, f"Prompt {p}") for p in prompts]
    
    parts = ax.violinplot(data_to_plot, positions=range(len(prompts)), 
                          showmeans=True, showmedians=True)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plt.cm.viridis(i / len(prompts)))
        pc.set_alpha(0.7)
    
    ax.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Question-Level Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Question-Level Accuracy by Prompt Type', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels(prompt_labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "accuracy_distribution.pdf")
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_repetition_consistency(df, output_dir="figures"):
    """Create heatmap showing accuracy across repetitions for each prompt."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate mean accuracy per prompt and repetition
    heatmap_data = df.groupby(['prompt', 'repetition'])['score'].mean().unstack(level=0)
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels([PROMPT_NAMES.get(p, f"Prompt {p}") for p in heatmap_data.columns],
                       rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels([f"Rep {r}" for r in heatmap_data.index])
    
    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            text = ax.text(j, i, f'{heatmap_data.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Accuracy Heatmap: Prompt × Repetition', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "repetition_consistency.pdf")
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def save_summary_table_latex(summary, output_dir="figures"):
    """Save summary table as LaTeX format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Format for LaTeX
    latex_table = summary.to_latex(index=False, float_format="%.3f", 
                                   caption="Accuracy and Token Usage by Prompt Type",
                                   label="tab:prompt_summary")
    
    output_path = os.path.join(output_dir, "summary_table.tex")
    with open(output_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved: {output_path}")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("GPQA Evaluation Analysis")
    print("="*80)
    
    # Load data
    df = load_all_results()
    
    # Create summary table
    summary = create_summary_table(df)
    
    # Generate plots
    print("\nGenerating figures...")
    plot_accuracy_by_prompt(df)
    plot_accuracy_with_error_bars(df)
    plot_token_usage(df)
    plot_accuracy_distribution(df)
    plot_repetition_consistency(df)
    
    # Save LaTeX table
    save_summary_table_latex(summary)
    
    print("\n" + "="*80)
    print("Analysis complete! All outputs saved to 'figures/' directory")
    print("="*80)


if __name__ == "__main__":
    main()
