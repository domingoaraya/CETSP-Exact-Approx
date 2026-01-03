import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import colorsys
import matplotlib.colors as mcolors

def plot_time_gap_core(ax, dfs, labels=None, colors=None, split_point=300, gap_range=300):
    """
    Core logic for plotting
    """
    def color_variants(hex_color, n=3):
        rgb = mcolors.to_rgb(hex_color)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        return [colorsys.hls_to_rgb(h, min(l * factor, 1.0), s)
                for factor in np.linspace(0.7, 1.5, n)]

    n_dfs = len(dfs)
    
    # Colors defaulting
    if colors is None:
        if n_dfs == 6:
            base_colors = ['#003f5c', '#f6ae2d', '#d7263d', 
                           '#1b9aaa', '#5c3c92', '#708090']
            colors = color_variants(base_colors[2])[::-1] + \
                     color_variants(base_colors[3])[::-1]
        elif n_dfs == 3:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        else:
            cmap = plt.cm.get_cmap('tab10', n_dfs)
            colors = [cmap(i) for i in range(n_dfs)]

    # Labels defaulting
    if labels is None:
        if n_dfs == 6:
            labels = ['ABF', 'ABF-D', 'ABF-A-D', 'SBF', 'SBF-D', 'SBF-A-D']
        else:
            labels = [f'Method {i+1}' for i in range(n_dfs)]

    instances_amount = len(dfs[0])
    second_split = split_point + 50
    max_time_global = 600
    max_gap_global = 1.0

    for df, color, label in zip(dfs, colors, labels):
        # --- Process Optimal (Left) ---
        optimal = df[df['Status'] == 'Optimal'].sort_values('Time')
        x_time, y_time = [0], [0]
        if not optimal.empty:
            st = split_point * (optimal['Time'] / max_time_global)
            x_time += list(st)
            y_time += list(range(1, len(optimal) + 1))

        # --- Process Gaps (Right) ---
        time_limit = df[df['Status'].isin(['Time_limit', 'Suboptimal'])].sort_values('Gap')
        
        x_gap, y_gap = [], []
        if not time_limit.empty:
            sg = second_split + gap_range * (time_limit['Gap'] / max_gap_global)
            x_gap += list(sg)
            y_gap += list(range(len(optimal) + 1, len(optimal) + len(time_limit) + 1))

        # --- Concatenate Coordinates ---
        if x_time and x_gap:
            x = x_time + [second_split] + x_gap
            y = y_time + [y_time[-1]] + y_gap
        elif x_time:
            x, y = x_time, y_time
        else:
            x = [0, second_split] + x_gap
            y = [0, 0] + y_gap

        ax.step(x, y, color=color, label=label, where='post')

    # Formatting Ticks
    time_ticks_real = np.linspace(0, max_time_global, 5)
    time_ticks = split_point * (time_ticks_real / max_time_global)
    time_labels = [f"{int(t)}" for t in time_ticks_real]

    gap_ticks_real = np.linspace(0, max_gap_global, 5)
    gap_ticks = second_split + gap_range * (gap_ticks_real / max_gap_global)
    gap_labels = [f"{round(g*100, 1)}" for g in gap_ticks_real]

    ax.set_xticks(list(time_ticks) + [split_point, second_split] + list(gap_ticks))
    ax.set_xticklabels(time_labels + [f"{int(max_time_global)}", "0"] + gap_labels)

    ax.set_yticks(range(0, instances_amount + 10, 15))
    ax.axvline(split_point, color='black', lw=0.5, ls='--')
    ax.axvline(second_split, color='black', lw=0.5, ls='--')
    ax.axhline(instances_amount, color='black', lw=0.5, ls='--')

    ax.set_xlim(-5, second_split + gap_range + 1)
    ax.set_ylim(bottom=-1)
    ax.set_ylabel("Number of instances solved")
    ax.legend(loc='lower right')

def plot_single_time_gap(dfs, labels, colors=None, title=None, figsize=(10, 6), filename=None):
    """
    Generates a single time-gap plot with the standard xlabel.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_time_gap_core(ax, dfs, labels, colors)
    
    if title:
        ax.set_title(title)
        
    ax.set_xlabel("Time (seconds) / Gap (%)")
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_triple_time_gap(
    list_of_dfs_groups,           
    labels,
    colors,
    titles=None,
    figsize=(18, 10),
    filename=None,
    save_res=300
):
    """
    Creates a 3-plot layout (2 top, 1 bottom-center) with consistent labeling.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 4, figure=fig)

    # Define the 3 axes (Top Left, Top Right, Bottom Center)
    ax1 = fig.add_subplot(gs[0, 0:2]) 
    ax2 = fig.add_subplot(gs[0, 2:4]) 
    ax3 = fig.add_subplot(gs[1, 1:3]) 
    
    all_axes = [ax1, ax2, ax3]

    for i, ax in enumerate(all_axes):
        if i < len(list_of_dfs_groups):
            dfs = list_of_dfs_groups[i]
            plot_time_gap_core(ax, dfs, labels, colors)

            # Assign titles
            if titles:
                ax.set_title(titles[i])
            else:
                ax.set_title(f"Plot {i+1}")
            
            # Apply the standard xlabel to every subplot
            ax.set_xlabel("Time (seconds) / Gap (%)")

    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=save_res, bbox_inches='tight', transparent=True)
    plt.show()

def build_summary_table(dfs, labels):
    """
    Builds a summary table comparing multiple DataFrames.
    """
    summaries = []
    
    for df, label in zip(dfs, labels):
        # 1. Calculate raw numeric values first
        optimal_mask = df['Status'] == 'Optimal'
        optimal_count = optimal_mask.sum()
        
        avg_time_opt = df.loc[optimal_mask, 'Time'].mean()
        avg_gap_all = df['Gap'].mean()
        avg_gap_noopt = df.loc[~optimal_mask, 'Gap'].mean()
        avg_cuts = df['Cuts'].mean()
        
        # 2. Helper to format numbers or return '-' if NaN/Zero
        def format_val(val, multiplier=1, is_gap=False):
            if pd.isna(val) or val is None:
                return '-'
            if not is_gap and val == 0: # Specific check for Cuts
                return '-'
            return f"{val * multiplier:.2f}"

        # 3. Build the dictionary using the helper
        summaries.append({
            'Formulation': label,
            'Opt': optimal_count,
            'TimeOpt': format_val(avg_time_opt),
            'Gap (%)': format_val(avg_gap_all, multiplier=100, is_gap=True),
            'Gap (NoOpt) (%)': format_val(avg_gap_noopt, multiplier=100, is_gap=True),
            'Cuts': format_val(avg_cuts)
        })
    
    return pd.DataFrame(summaries)