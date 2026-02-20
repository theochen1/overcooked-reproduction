"""
Replicate Figure 4 from the paper:
"On the Utility of Learning about Humans for Human-AI Coordination"

Figure 4 shows:
- (a) Comparison with agents trained in self-play
- (b) Comparison with agents trained via PBT

Key elements:
- White bars: Self-play baselines (SP+SP, PBT+PBT)
- Teal bars: SP/PBT paired with Human Proxy
- Orange bars: PPO_BC paired with Human Proxy
- Gray bars: BC paired with Human Proxy
- Red dotted line: Gold standard (PPO trained with actual H_Proxy)
- Hatched bars: Switched starting positions
- Error bars: Standard error over 5 seeds
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple

# Paper-style colors
COLORS = {
    'self_play': '#4A90A4',      # Teal for SP+HP, PBT+HP
    'ppo_bc': '#E8944A',         # Orange for PPO_BC+HP
    'bc': '#808080',             # Gray for BC+HP
    'baseline_white': 'white',   # White for SP+SP, PBT+PBT
    'gold_standard': '#C44E52',  # Red for gold standard line
}

LAYOUT_NAMES = {
    'cramped_room': 'Cramped Rm.',
    'asymmetric_advantages': 'Asymm. Adv.',
    'coordination_ring': 'Coord. Ring',
    'forced_coordination': 'Forced Coord.',
    'counter_circuit': 'Counter Circ.',
}

LAYOUTS = ['cramped_room', 'asymmetric_advantages', 'coordination_ring', 
           'forced_coordination', 'counter_circuit']


def load_evaluation_results(results_dir: str) -> Dict:
    """
    Load evaluation results from JSON files.
    
    Expected structure:
    results_dir/
        cramped_room/
            sp_sp.json
            sp_hp.json
            ppo_bc_hp.json
            bc_hp.json
            ppo_hp_hp.json  (gold standard)
    """
    import json
    
    results = {}
    for layout in LAYOUTS:
        layout_dir = os.path.join(results_dir, layout)
        if not os.path.exists(layout_dir):
            continue
            
        results[layout] = {}
        for agent_pair in ['sp_sp', 'sp_hp', 'ppo_bc_hp', 'bc_hp', 'ppo_hp_hp',
                           'pbt_pbt', 'pbt_hp']:
            filepath = os.path.join(layout_dir, f'{agent_pair}.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    results[layout][agent_pair] = json.load(f)
    
    return results


def create_sample_data() -> Dict:
    """
    Create sample data structure matching paper Figure 4.
    Replace with actual evaluation results.
    """
    # Sample data based on paper Figure 4 (approximate values)
    data = {
        'cramped_room': {
            'sp_sp': {'mean': 200, 'std': 15, 'se': 7},
            'sp_hp': {'mean': 100, 'std': 20, 'se': 9},
            'sp_hp_swapped': {'mean': 105, 'std': 18, 'se': 8},
            'ppo_bc_hp': {'mean': 155, 'std': 15, 'se': 7},
            'ppo_bc_hp_swapped': {'mean': 160, 'std': 12, 'se': 5},
            'bc_hp': {'mean': 110, 'std': 25, 'se': 11},
            'bc_hp_swapped': {'mean': 108, 'std': 22, 'se': 10},
            'ppo_hp_hp': {'mean': 175, 'std': 10, 'se': 4},  # Gold standard
            # PBT versions
            'pbt_pbt': {'mean': 210, 'std': 20, 'se': 9},
            'pbt_hp': {'mean': 120, 'std': 25, 'se': 11},
            'pbt_hp_swapped': {'mean': 115, 'std': 20, 'se': 9},
        },
        'asymmetric_advantages': {
            'sp_sp': {'mean': 230, 'std': 18, 'se': 8},
            'sp_hp': {'mean': 40, 'std': 15, 'se': 7},
            'sp_hp_swapped': {'mean': 85, 'std': 20, 'se': 9},
            'ppo_bc_hp': {'mean': 180, 'std': 20, 'se': 9},
            'ppo_bc_hp_swapped': {'mean': 185, 'std': 18, 'se': 8},
            'bc_hp': {'mean': 130, 'std': 25, 'se': 11},
            'bc_hp_swapped': {'mean': 135, 'std': 22, 'se': 10},
            'ppo_hp_hp': {'mean': 195, 'std': 12, 'se': 5},
            'pbt_pbt': {'mean': 180, 'std': 22, 'se': 10},
            'pbt_hp': {'mean': 140, 'std': 18, 'se': 8},
            'pbt_hp_swapped': {'mean': 175, 'std': 20, 'se': 9},
        },
        'coordination_ring': {
            'sp_sp': {'mean': 155, 'std': 20, 'se': 9},
            'sp_hp': {'mean': 70, 'std': 18, 'se': 8},
            'sp_hp_swapped': {'mean': 75, 'std': 15, 'se': 7},
            'ppo_bc_hp': {'mean': 125, 'std': 18, 'se': 8},
            'ppo_bc_hp_swapped': {'mean': 130, 'std': 15, 'se': 7},
            'bc_hp': {'mean': 80, 'std': 20, 'se': 9},
            'bc_hp_swapped': {'mean': 85, 'std': 18, 'se': 8},
            'ppo_hp_hp': {'mean': 145, 'std': 12, 'se': 5},
            'pbt_pbt': {'mean': 145, 'std': 18, 'se': 8},
            'pbt_hp': {'mean': 75, 'std': 15, 'se': 7},
            'pbt_hp_swapped': {'mean': 80, 'std': 18, 'se': 8},
        },
        'forced_coordination': {
            'sp_sp': {'mean': 100, 'std': 22, 'se': 10},
            'sp_hp': {'mean': 25, 'std': 12, 'se': 5},
            'sp_hp_swapped': {'mean': 60, 'std': 18, 'se': 8},
            'ppo_bc_hp': {'mean': 55, 'std': 15, 'se': 7},
            'ppo_bc_hp_swapped': {'mean': 70, 'std': 18, 'se': 8},
            'bc_hp': {'mean': 40, 'std': 15, 'se': 7},
            'bc_hp_swapped': {'mean': 55, 'std': 12, 'se': 5},
            'ppo_hp_hp': {'mean': 80, 'std': 10, 'se': 4},
            'pbt_pbt': {'mean': 55, 'std': 15, 'se': 7},
            'pbt_hp': {'mean': 15, 'std': 10, 'se': 4},
            'pbt_hp_swapped': {'mean': 25, 'std': 12, 'se': 5},
        },
        'counter_circuit': {
            'sp_sp': {'mean': 115, 'std': 18, 'se': 8},
            'sp_hp': {'mean': 50, 'std': 15, 'se': 7},
            'sp_hp_swapped': {'mean': 55, 'std': 12, 'se': 5},
            'ppo_bc_hp': {'mean': 90, 'std': 18, 'se': 8},
            'ppo_bc_hp_swapped': {'mean': 95, 'std': 15, 'se': 7},
            'bc_hp': {'mean': 60, 'std': 18, 'se': 8},
            'bc_hp_swapped': {'mean': 65, 'std': 15, 'se': 7},
            'ppo_hp_hp': {'mean': 100, 'std': 10, 'se': 4},
            'pbt_pbt': {'mean': 75, 'std': 15, 'se': 7},
            'pbt_hp': {'mean': 55, 'std': 12, 'se': 5},
            'pbt_hp_swapped': {'mean': 60, 'std': 15, 'se': 7},
        },
    }
    return data


def plot_figure_4a(ax, data: Dict, show_legend: bool = True):
    """
    Plot Figure 4(a): Comparison with agents trained in self-play
    
    Bars (left to right per layout):
    1. SP+SP (white, baseline)
    2. SP+H_Proxy (teal)
    3. SP+H_Proxy swapped (teal, hatched)
    4. PPO_BC+H_Proxy (orange)
    5. PPO_BC+H_Proxy swapped (orange, hatched)
    6. BC+H_Proxy (gray)
    7. BC+H_Proxy swapped (gray, hatched)
    
    Plus red dotted line for gold standard
    """
    x = np.arange(len(LAYOUTS))
    width = 0.12
    
    # Extract data
    sp_sp = [data[l].get('sp_sp', {}).get('mean', 0) for l in LAYOUTS]
    sp_sp_se = [data[l].get('sp_sp', {}).get('se', 0) for l in LAYOUTS]
    
    sp_hp = [data[l].get('sp_hp', {}).get('mean', 0) for l in LAYOUTS]
    sp_hp_se = [data[l].get('sp_hp', {}).get('se', 0) for l in LAYOUTS]
    sp_hp_sw = [data[l].get('sp_hp_swapped', {}).get('mean', 0) for l in LAYOUTS]
    sp_hp_sw_se = [data[l].get('sp_hp_swapped', {}).get('se', 0) for l in LAYOUTS]
    
    ppo_bc = [data[l].get('ppo_bc_hp', {}).get('mean', 0) for l in LAYOUTS]
    ppo_bc_se = [data[l].get('ppo_bc_hp', {}).get('se', 0) for l in LAYOUTS]
    ppo_bc_sw = [data[l].get('ppo_bc_hp_swapped', {}).get('mean', 0) for l in LAYOUTS]
    ppo_bc_sw_se = [data[l].get('ppo_bc_hp_swapped', {}).get('se', 0) for l in LAYOUTS]
    
    bc_hp = [data[l].get('bc_hp', {}).get('mean', 0) for l in LAYOUTS]
    bc_hp_se = [data[l].get('bc_hp', {}).get('se', 0) for l in LAYOUTS]
    bc_hp_sw = [data[l].get('bc_hp_swapped', {}).get('mean', 0) for l in LAYOUTS]
    bc_hp_sw_se = [data[l].get('bc_hp_swapped', {}).get('se', 0) for l in LAYOUTS]
    
    gold = [data[l].get('ppo_hp_hp', {}).get('mean', 0) for l in LAYOUTS]
    
    # Plot bars
    bars1 = ax.bar(x - 3*width, sp_sp, width, yerr=sp_sp_se, 
                   color='white', edgecolor='black', linewidth=1.5,
                   capsize=2, label='SP+SP')
    
    bars2 = ax.bar(x - 2*width, sp_hp, width, yerr=sp_hp_se,
                   color=COLORS['self_play'], edgecolor='black', linewidth=0.5,
                   capsize=2, label='SP+H$_{Proxy}$')
    bars3 = ax.bar(x - width, sp_hp_sw, width, yerr=sp_hp_sw_se,
                   color=COLORS['self_play'], edgecolor='black', linewidth=0.5,
                   hatch='///', capsize=2)
    
    bars4 = ax.bar(x, ppo_bc, width, yerr=ppo_bc_se,
                   color=COLORS['ppo_bc'], edgecolor='black', linewidth=0.5,
                   capsize=2, label='PPO$_{BC}$+H$_{Proxy}$')
    bars5 = ax.bar(x + width, ppo_bc_sw, width, yerr=ppo_bc_sw_se,
                   color=COLORS['ppo_bc'], edgecolor='black', linewidth=0.5,
                   hatch='///', capsize=2)
    
    bars6 = ax.bar(x + 2*width, bc_hp, width, yerr=bc_hp_se,
                   color=COLORS['bc'], edgecolor='black', linewidth=0.5,
                   capsize=2, label='BC+H$_{Proxy}$')
    bars7 = ax.bar(x + 3*width, bc_hp_sw, width, yerr=bc_hp_sw_se,
                   color=COLORS['bc'], edgecolor='black', linewidth=0.5,
                   hatch='///', capsize=2)
    
    # Gold standard line
    for i, g in enumerate(gold):
        ax.hlines(g, x[i] - 3.5*width, x[i] + 3.5*width, 
                  colors=COLORS['gold_standard'], linestyles='dotted', linewidth=2)
    
    # Formatting
    ax.set_ylabel('Average reward per episode', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([LAYOUT_NAMES[l] for l in LAYOUTS], fontsize=10)
    ax.set_ylim(0, 260)
    ax.set_title('Performance with human proxy model', fontsize=12, fontweight='bold')
    
    if show_legend:
        # Create custom legend
        legend_elements = [
            plt.Line2D([0], [0], color=COLORS['gold_standard'], linestyle='dotted', 
                       linewidth=2, label='PPO$_{H_{Proxy}}$+H$_{Proxy}$'),
            mpatches.Patch(facecolor='white', edgecolor='black', linewidth=1.5, 
                          label='SP+SP'),
            mpatches.Patch(facecolor=COLORS['self_play'], edgecolor='black', 
                          label='SP+H$_{Proxy}$'),
            mpatches.Patch(facecolor=COLORS['ppo_bc'], edgecolor='black', 
                          label='PPO$_{BC}$+H$_{Proxy}$'),
            mpatches.Patch(facecolor=COLORS['bc'], edgecolor='black', 
                          label='BC+H$_{Proxy}$'),
            mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', 
                          label='Switched indices'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                  framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_figure_4b(ax, data: Dict, show_legend: bool = True):
    """
    Plot Figure 4(b): Comparison with agents trained via PBT
    
    Similar to 4(a) but with PBT instead of SP
    """
    x = np.arange(len(LAYOUTS))
    width = 0.12
    
    # Extract data
    pbt_pbt = [data[l].get('pbt_pbt', {}).get('mean', 0) for l in LAYOUTS]
    pbt_pbt_se = [data[l].get('pbt_pbt', {}).get('se', 0) for l in LAYOUTS]
    
    pbt_hp = [data[l].get('pbt_hp', {}).get('mean', 0) for l in LAYOUTS]
    pbt_hp_se = [data[l].get('pbt_hp', {}).get('se', 0) for l in LAYOUTS]
    pbt_hp_sw = [data[l].get('pbt_hp_swapped', {}).get('mean', 0) for l in LAYOUTS]
    pbt_hp_sw_se = [data[l].get('pbt_hp_swapped', {}).get('se', 0) for l in LAYOUTS]
    
    ppo_bc = [data[l].get('ppo_bc_hp', {}).get('mean', 0) for l in LAYOUTS]
    ppo_bc_se = [data[l].get('ppo_bc_hp', {}).get('se', 0) for l in LAYOUTS]
    ppo_bc_sw = [data[l].get('ppo_bc_hp_swapped', {}).get('mean', 0) for l in LAYOUTS]
    ppo_bc_sw_se = [data[l].get('ppo_bc_hp_swapped', {}).get('se', 0) for l in LAYOUTS]
    
    bc_hp = [data[l].get('bc_hp', {}).get('mean', 0) for l in LAYOUTS]
    bc_hp_se = [data[l].get('bc_hp', {}).get('se', 0) for l in LAYOUTS]
    bc_hp_sw = [data[l].get('bc_hp_swapped', {}).get('mean', 0) for l in LAYOUTS]
    bc_hp_sw_se = [data[l].get('bc_hp_swapped', {}).get('se', 0) for l in LAYOUTS]
    
    gold = [data[l].get('ppo_hp_hp', {}).get('mean', 0) for l in LAYOUTS]
    
    # Plot bars
    bars1 = ax.bar(x - 3*width, pbt_pbt, width, yerr=pbt_pbt_se,
                   color='white', edgecolor='black', linewidth=1.5,
                   capsize=2, label='PBT+PBT')
    
    bars2 = ax.bar(x - 2*width, pbt_hp, width, yerr=pbt_hp_se,
                   color=COLORS['self_play'], edgecolor='black', linewidth=0.5,
                   capsize=2, label='PBT+H$_{Proxy}$')
    bars3 = ax.bar(x - width, pbt_hp_sw, width, yerr=pbt_hp_sw_se,
                   color=COLORS['self_play'], edgecolor='black', linewidth=0.5,
                   hatch='///', capsize=2)
    
    bars4 = ax.bar(x, ppo_bc, width, yerr=ppo_bc_se,
                   color=COLORS['ppo_bc'], edgecolor='black', linewidth=0.5,
                   capsize=2, label='PPO$_{BC}$+H$_{Proxy}$')
    bars5 = ax.bar(x + width, ppo_bc_sw, width, yerr=ppo_bc_sw_se,
                   color=COLORS['ppo_bc'], edgecolor='black', linewidth=0.5,
                   hatch='///', capsize=2)
    
    bars6 = ax.bar(x + 2*width, bc_hp, width, yerr=bc_hp_se,
                   color=COLORS['bc'], edgecolor='black', linewidth=0.5,
                   capsize=2, label='BC+H$_{Proxy}$')
    bars7 = ax.bar(x + 3*width, bc_hp_sw, width, yerr=bc_hp_sw_se,
                   color=COLORS['bc'], edgecolor='black', linewidth=0.5,
                   hatch='///', capsize=2)
    
    # Gold standard line
    for i, g in enumerate(gold):
        ax.hlines(g, x[i] - 3.5*width, x[i] + 3.5*width,
                  colors=COLORS['gold_standard'], linestyles='dotted', linewidth=2)
    
    # Formatting
    ax.set_ylabel('Average reward per episode', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([LAYOUT_NAMES[l] for l in LAYOUTS], fontsize=10)
    ax.set_ylim(0, 260)
    ax.set_title('Performance with human proxy model', fontsize=12, fontweight='bold')
    
    if show_legend:
        legend_elements = [
            plt.Line2D([0], [0], color=COLORS['gold_standard'], linestyle='dotted',
                       linewidth=2, label='PPO$_{H_{Proxy}}$+H$_{Proxy}$'),
            mpatches.Patch(facecolor='white', edgecolor='black', linewidth=1.5,
                          label='PBT+PBT'),
            mpatches.Patch(facecolor=COLORS['self_play'], edgecolor='black',
                          label='PBT+H$_{Proxy}$'),
            mpatches.Patch(facecolor=COLORS['ppo_bc'], edgecolor='black',
                          label='PPO$_{BC}$+H$_{Proxy}$'),
            mpatches.Patch(facecolor=COLORS['bc'], edgecolor='black',
                          label='BC+H$_{Proxy}$'),
            mpatches.Patch(facecolor='white', edgecolor='black', hatch='///',
                          label='Switched indices'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                  framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_figure_4(data: Dict, save_path: Optional[str] = None):
    """
    Create the complete Figure 4 with both subplots.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    plot_figure_4a(ax1, data)
    ax1.text(-0.5, -30, '(a) Comparison with agents trained in self-play',
             fontsize=11, fontweight='bold', ha='left')
    
    plot_figure_4b(ax2, data)
    ax2.text(-0.5, -30, '(b) Comparison with agents trained via PBT',
             fontsize=11, fontweight='bold', ha='left')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_simple_comparison(data: Dict, save_path: Optional[str] = None):
    """
    Create a simplified version showing just SP comparison (4a).
    Useful if PBT is not implemented.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_figure_4a(ax, data)
    ax.set_title('Performance with human proxy model\n(Comparison with agents trained in self-play)',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot Figure 4 from the paper')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory containing evaluation results')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save the figure')
    parser.add_argument('--simple', action='store_true',
                        help='Plot only Figure 4(a) without PBT')
    parser.add_argument('--demo', action='store_true',
                        help='Use sample data for demonstration')
    
    args = parser.parse_args()
    
    if args.demo or args.results_dir is None:
        print("Using sample data for demonstration...")
        print("To use actual results, provide --results_dir")
        data = create_sample_data()
    else:
        data = load_evaluation_results(args.results_dir)
    
    if args.simple:
        plot_simple_comparison(data, args.save)
    else:
        plot_figure_4(data, args.save)


if __name__ == '__main__':
    main()

