#!/usr/bin/env python3
"""
Plot transfer latency as a percentage of step latency.
Compares latent transfer times with inference step times.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File paths
TRANSFER_FILE = Path("log/benchmark/latent_transfer/flux/latent_transfer_benchmark_full.json")
STEP_FILE = Path("log/benchmark/scaling_efficiency/flux/H100_HBM3/e2e_scaling_efficiency.json")

def load_transfer_data(file_path):
    """Load transfer latency data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    transfer_data = {}
    for config in data['configurations']:
        key = (config['batch_size'], config['height'], config['width'])
        # Calculate average transfer time (skip first iteration as it's often a warmup)
        transfer_times = config['transfer_times']
        if len(transfer_times) > 1:
            # Use all but the first (warmup) iteration
            # avg_transfer = sum(transfer_times[1:]) / len(transfer_times[1:])
            avg_transfer = config["total_time"] / config["iterations"]
        else:
            avg_transfer = transfer_times[0] if transfer_times else 0.0
        
        transfer_data[key] = {
            'avg_transfer_time': avg_transfer,
            'total_time': config['total_time'],
            'iterations': config['iterations']
        }
    
    return transfer_data

def load_step_data(file_path):
    """Load inference step latency data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    step_data = {}
    for entry in data:
        key = (entry['bs'], entry['height'], entry['width'])
        step_data[key] = {
            'avg_step_time': entry['avg_denoising_time'],
            'steps': entry['steps']
        }
    
    return step_data

def match_and_calculate_percentages(transfer_data, step_data):
    """Match configurations and calculate transfer/step percentage."""
    results = []
    
    for key in transfer_data:
        if key in step_data:
            transfer_time = transfer_data[key]['avg_transfer_time']
            step_time = step_data[key]['avg_step_time']
            
            if step_time > 0:
                percentage = (transfer_time / step_time) * 100
                results.append({
                    'batch_size': key[0],
                    'height': key[1],
                    'width': key[2],
                    'transfer_time': transfer_time,
                    'step_time': step_time,
                    'percentage': percentage
                })
    
    # Sort by resolution (height*width) and batch size
    results.sort(key=lambda x: (x['height'] * x['width'], x['batch_size']))
    
    return results

def plot_column_graph(results, output_file='transfer_overhead_percentage.png'):
    """Create a column graph showing transfer latency as percentage of step latency."""
    # Prepare data for plotting
    labels = []
    percentages = []
    colors = []
    hatches = []
    
    # Academic color palette
    academic_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    # Hatch patterns (all batch sizes get a hatch pattern)
    hatch_patterns = ["///", "///", "///", "///"]
    
    # Color scheme based on batch size
    batch_sizes = sorted(set(result['batch_size'] for result in results))
    batch_colors = {bs: academic_colors[i % len(academic_colors)] for i, bs in enumerate(batch_sizes)}
    batch_hatches = {bs: hatch_patterns[i % len(hatch_patterns)] for i, bs in enumerate(batch_sizes)}
    
    for result in results:
        label = f"{result['height']}×{result['width']}\nBS={result['batch_size']}"
        labels.append(label)
        percentages.append(result['percentage'])
        colors.append(batch_colors.get(result['batch_size'], '#9467bd'))
        hatches.append(batch_hatches.get(result['batch_size'], ''))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Put grid behind bars
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    
    # Create bars with academic styling
    bars = ax.bar(range(len(labels)), percentages, color=colors, 
                  edgecolor='black', linewidth=0.8, hatch=hatches, zorder=3)
    
    # Customize the plot
    ax.set_xlabel('Configuration (Resolution × Batch Size)', fontsize=28)
    ax.set_ylabel('Transfer Lat / Step Lat (%)', fontsize=28)
    ax.set_title('Latent Transfer Overhead as Percentage of Inference Step Latency', 
                 fontsize=24, pad=20)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
    # Add value labels on top of bars
    max_pct = max(percentages) if percentages else 0
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_pct * 0.02,
                f'{pct:.2f}%',
                ha='center', va='bottom', fontsize=16)
    
    # Add legend for batch sizes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=batch_colors[bs], hatch=batch_hatches[bs], 
                           edgecolor='black', linewidth=0.8, label=f'Batch Size {bs}') 
                      for bs in sorted(batch_colors.keys())]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    # Save both PNG and PDF
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    print(f"Plot saved to {pdf_file}")
    
    # Also print summary statistics
    print("\nSummary Statistics:")
    print(f"  Total configurations: {len(results)}")
    print(f"  Average percentage: {np.mean(percentages):.2f}%")
    print(f"  Median percentage: {np.median(percentages):.2f}%")
    print(f"  Min percentage: {np.min(percentages):.2f}%")
    print(f"  Max percentage: {np.max(percentages):.2f}%")
    
    return fig

def main():
    """Main function."""
    # Load data
    print("Loading transfer data...")
    transfer_data = load_transfer_data(TRANSFER_FILE)
    print(f"  Found {len(transfer_data)} transfer configurations")
    
    print("Loading step data...")
    step_data = load_step_data(STEP_FILE)
    print(f"  Found {len(step_data)} step configurations")
    
    # Match and calculate
    print("Matching configurations and calculating percentages...")
    results = match_and_calculate_percentages(transfer_data, step_data)
    print(f"  Matched {len(results)} configurations")
    
    if not results:
        print("ERROR: No matching configurations found!")
        return
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 80)
    print(f"{'Resolution':<15} {'BS':<5} {'Transfer (ms)':<15} {'Step (ms)':<15} {'Percentage':<12}")
    print("-" * 80)
    for result in results:
        print(f"{result['height']}×{result['width']:<10} {result['batch_size']:<5} "
              f"{result['transfer_time']*1000:>12.3f}    {result['step_time']*1000:>12.3f}    "
              f"{result['percentage']:>10.2f}%")
    
    # Create plot
    print("\nGenerating plot...")
    plot_column_graph(results)
    
    print("\nDone!")

"""
python benchmark/latent_transfer/plot_transfer_overhead.py
"""

if __name__ == "__main__":
    main()

