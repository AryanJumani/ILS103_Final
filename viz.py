import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def generate_presentation_visualization(input_csv: str, output_png: str):
    """
    Bar chart
    """
    df = pd.read_csv(input_csv)
    soc_df = df.groupby('Normalized_SoC', as_index=False).agg({
        'Theoretical_Max_AI_Ops_Per_Frame': 'max'
        })
    top_socs = soc_df.sort_values(by='Theoretical_Max_AI_Ops_Per_Frame', ascending=False).head(15)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(12, 8))

    bars = sns.barplot(
        data=top_socs,
        x='Theoretical_Max_AI_Ops_Per_Frame',
        y='Normalized_SoC',
        hue='Normalized_SoC',
        palette='viridis',
        legend=False,
        ax=ax
    )
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f'{int(width)}',
                (width, p.get_y() + p.get_height() / 2.),
                ha='left', va='center',
                xytext=(5, 0),
                textcoords='offset points',
                fontweight='bold')
        ax.set_title('Theoretical Max AI Operations Per Frame (60 FPS Target)',
            fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Max AI Operations (Micro-Kernels / Frame)', fontsize=14, labelpad=10)
        ax.set_ylabel('System on Chip (SoC)', fontsize=14, labelpad=10)
        sns.despine(left=True, bottom=True)

        plt.tight_layout()
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"Visualization successfully generated: {output_png}")
if __name__ == "__main__":
    INPUT_FILE = 'merged_metrics.csv'
    OUTPUT_FILE = 'viz.png'
    generate_presentation_visualization(INPUT_FILE, OUTPUT_FILE)
