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
    """Create summary statistics table by prompt with error bars."""
    # Calculate accuracy with std across repetitions
    grouped = df.groupby(['prompt', 'repetition'])['score'].mean().reset_index()
    accuracy_stats = grouped.groupby('prompt')['score'].agg(['mean', 'std']).sort_index()
    
    # Calculate token usage
    token_stats = df.groupby('prompt').agg({
        'input_tokens': 'mean',
        'output_tokens': 'mean',
    }).round(0)
    
    # Combine
    summary = pd.DataFrame({
        'prompt_name': [PROMPT_NAMES.get(i, f"Prompt {i}") for i in accuracy_stats.index],
        'accuracy_mean': accuracy_stats['mean'].round(3),
        'accuracy_std': accuracy_stats['std'].round(3),
        'avg_input_tokens': token_stats['input_tokens'].astype(int),
        'avg_output_tokens': token_stats['output_tokens'].astype(int),
    })
    
    print("\n" + "="*80)
    print("Summary Statistics by Prompt")
    print("="*80)
    print(summary.to_string())
    print("="*80 + "\n")
    
    return summary


def plot_accuracy_with_error_bars(df, output_dir="output"):
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


def plot_token_usage(df, output_dir="output"):
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


def analyze_errors_by_subdomain_and_prompt(df, output_dir="output"):
    """Accuracy and error counts by subdomain x prompt. Requires df with 'subdomain'."""
    os.makedirs(output_dir, exist_ok=True)
    g = df.groupby(["subdomain", "prompt"])["score"].agg(["mean", "sum", "count"])
    g = g.rename(columns={"mean": "accuracy", "sum": "correct", "count": "n"})
    g["errors"] = g["n"] - g["correct"]
    # Pivot: rows=subdomain, cols=prompt, values=accuracy
    acc = df.pivot_table(index="subdomain", columns="prompt", values="score", aggfunc="mean")
    acc.columns = [PROMPT_NAMES.get(c, f"P{c}") for c in acc.columns]
    print("\n" + "=" * 80)
    print("Accuracy by Subdomain x Prompt")
    print("=" * 80)
    print(acc.round(3).to_string())
    print("=" * 80 + "\n")
    # LaTeX: subdomain (index) x prompt columns, values as .3f
    acc_tex = acc.round(3).map(lambda x: f"{x:.3f}").reset_index()
    acc_tex = acc_tex.rename(columns={acc_tex.columns[0]: "Subdomain"})
    latex_str = acc_tex.to_latex(
        caption="Accuracy by subdomain and prompt type.",
        label="tab:subdomain_prompt",
        index=False,
        escape=False,
        position="H",
    )
    latex_str = latex_str.replace("\\begin{table}[H]\n", "\\begin{table}[H]\n\\centering\n\\small\n")
    out_tex = os.path.join(output_dir, "accuracy_subdomain_prompt.tex")
    with open(out_tex, "w") as f:
        f.write(latex_str)
    print(f"Saved: {out_tex}")
    # Error counts for (subdomain, prompt) with most errors
    err = g.reset_index().nlargest(20, "errors")[["subdomain", "prompt", "errors", "n", "accuracy"]]
    err["prompt_name"] = err["prompt"].map(PROMPT_NAMES)
    print("\nTop 20 (subdomain, prompt) by error count:")
    print(err.to_string(index=False))
    print()


def save_summary_table_latex(summary, output_dir="output"):
    """Save summary table as LaTeX format with error bars."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Format accuracy with error bars
    summary['accuracy'] = summary.apply(
        lambda row: f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f}", 
        axis=1
    )
    
    # Select and rename columns for LaTeX
    latex_df = summary[['prompt_name', 'accuracy', 'avg_input_tokens', 'avg_output_tokens']].copy()
    latex_df.columns = ['Prompt Type', 'Accuracy', 'Avg Input Tokens', 'Avg Output Tokens']
    
    # Only tabular; main.tex wraps with \begin{table}[h]\centering\caption\label\input{...}\end{table}
    full = latex_df.to_latex(index=False, escape=False)
    start = full.find(r"\begin{tabular}")
    end = full.find(r"\end{tabular}") + len(r"\end{tabular}")
    tabular_str = full[start:end] if start >= 0 and end > start else full
    output_path = os.path.join(output_dir, "summary_table.tex")
    with open(output_path, 'w') as f:
        f.write(tabular_str)
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
    plot_accuracy_with_error_bars(df)
    plot_token_usage(df)
    
    # Save LaTeX table
    save_summary_table_latex(summary)

    # Subdomain x prompt analysis
    analyze_errors_by_subdomain_and_prompt(load_results_jsonl("data/gpqa_with_subdomain.jsonl"))
    
    print("\n" + "="*80)
    print("Analysis complete! All outputs saved to 'output/' directory")
    print("="*80)


if __name__ == "__main__":
    main()
