import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors_mod
import seaborn as sns
import os
import pickle

def pickle_obj(obj, savefolder, filename):
    # Ensure the savefolder exists, create it if it doesn't
    os.makedirs(savefolder, exist_ok=True)
    
    # Combine the savefolder and filename to create the full path
    filepath = os.path.join(savefolder, filename)
    
    # Open the file for writing
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def open_pickle(filename):  # Open a pickle of preprocessed imaging data for a cohort of flies
    return pd.read_pickle(filename)
    # This will return a dictionary with keys for each fly that was imaged; the corresponding values themselves are
    # dictionaries with keys 'a1', 'd', 'di', 'do', 'ft', all with values storing dataframes


def plot_FF_trajectory(figure_folder, filename, lobes, colors, strip_width, strip_length, xlim, ylim, hlines=[], save=False, keyword=None):
    data = open_pickle(filename)
    flylist_n = np.array(list(data.keys()))

    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

    # Determine flies to plot
    if keyword is not None and 1 <= keyword <= len(flylist_n):
        # Plotting for a specific fly
        flies_to_plot = [(keyword, data[flylist_n[keyword - 1]])]
    else:
        # Plotting for all flies
        flies_to_plot = [(fly_key, this_fly) for fly_key, this_fly in enumerate(data.values(), start=1)]
    
    # Plot data
    for fly_number, this_fly in flies_to_plot:
        # Assign the correct dataframe ('a1' is all the data collected)
        this_fly_all_data = this_fly['a1']
        
        # Create a single figure
        fig, axs = plt.subplots(1, len(lobes), figsize=(8 * len(lobes), 10))
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        fig.patch.set_facecolor('black')  # Set background to black

        # Set font color to white
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'

        # Iterate through the lobes
        for i, (lobe, color) in enumerate(zip(lobes, colors)):
            # Filter df to exp start
            first_on_index = this_fly_all_data[this_fly_all_data['instrip']].index[0]
            exp_df = this_fly_all_data.loc[first_on_index:] # This filters the dataframe
            # Establish the plume origin, at first odor onset
            xo = exp_df.iloc[0]['ft_posx']
            yo = exp_df.iloc[0]['ft_posy']
            
            # Assign FF (fluorescence) to the correct df column
            FF = exp_df[lobe]
            # Smooth fluorescence data
            smoothed_FF = FF.rolling(window=10, min_periods=1).mean()

            # Define color map
            cmap = plt.get_cmap('coolwarm')

            # Normalize FF to [0, 1] for colormap
            min_FF = smoothed_FF.min()
            max_FF = smoothed_FF.max()
            range_FF = max_FF - min_FF
            norm = colors_mod.Normalize(vmin=min_FF - 0.1 * range_FF, vmax=max_FF + 0.1 * range_FF)
            axs[i].add_patch(patches.Rectangle((-strip_width / 2, 0), strip_width, strip_length, facecolor='black', edgecolor='white'))
            # Plot the trajectory on the corresponding subplot
            axs[i].set_facecolor('black')  # Set background of plotting area to black
            # Plot each segment individually
            for j in range(len(exp_df) - 1):
                # Coordinates of the current segment's start and end points
                x1 = exp_df['ft_posx'].iloc[j] - xo
                y1 = exp_df['ft_posy'].iloc[j] - yo
                x2 = exp_df['ft_posx'].iloc[j + 1] - xo
                y2 = exp_df['ft_posy'].iloc[j + 1] - yo

                # Average fluorescence for the segment
                avg_FF = (smoothed_FF.iloc[j] + smoothed_FF.iloc[j + 1]) / 2

                # Color for the segment based on the average fluorescence
                color = cmap(norm(avg_FF))

                # Plot the segment
                axs[i].plot([x1, x2], [y1, y2], color=color, linewidth=3)

            if hlines is not None:
                for j in range(len(hlines)):
                    axs[i].hlines(y=hlines[j], xmin=-100, xmax=100, colors='k', linestyles='--', linewidth=1)
            
            title = f'fly {fly_number}'

            # Set axes, labels, and title
            axs[i].set_xlim(xlim)
            axs[i].set_ylim(ylim)
            axs[i].set_xlabel('x position', fontsize=14)
            axs[i].set_ylabel('y position', fontsize=14)
            axs[i].set_title(f'{title} {lobe} lobe', fontsize=14)

            # Further customization
            axs[i].tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
            for pos in ['right', 'top']:
                axs[i].spines[pos].set_visible(False)

            for _, spine in axs[i].spines.items():
                spine.set_linewidth(2)
            for spine in axs[i].spines.values():
                spine.set_edgecolor('black')

        # Apply tight layout to the entire figure
        fig.tight_layout()

        # Save and show the plot
        if save:
            plt.suptitle(f'dF/F traj fly {fly_number}')
            plt.savefig(os.path.join(figure_folder, f'FF_traj_{fly_number}_bw.pdf'))
        else:
            plt.show()

def plot_speed_trajectory(figure_folder, filename, strip_width, strip_length, xlim, ylim, hlines=[], save=False, keyword=None):
    data = open_pickle(filename)
    flylist_n = np.array(list(data.keys()))

    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

    # Determine flies to plot
    if keyword is not None and 1 <= keyword <= len(flylist_n):
        # Plotting for a specific fly
        flies_to_plot = [(keyword, data[flylist_n[keyword - 1]])]
    else:
        # Plotting for all flies
        flies_to_plot = [(fly_key, this_fly) for fly_key, this_fly in enumerate(data.values(), start=1)]
    
    # Plot data
    for fly_number, this_fly in flies_to_plot:
        # Assign the correct dataframe ('a1' is all the data collected)
        this_fly_all_data = this_fly['a1']
        
        # Create a single figure
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'

        # Filter df to exp start
        first_on_index = this_fly_all_data[this_fly_all_data['instrip']].index[0]
        exp_df = this_fly_all_data.loc[first_on_index:] # This filters the dataframe
        # Establish the plume origin, at first odor onset
        xo = exp_df.iloc[0]['ft_posx']
        yo = exp_df.iloc[0]['ft_posy']
            
        # Assign FF (fluorescence) to the correct df column
        FF = exp_df['speed']
        # Smooth fluorescence data
        smoothed_FF = FF.rolling(window=10, min_periods=1).mean()

        # Define color map
        cmap = plt.get_cmap('coolwarm')

        # Normalize FF to [0, 1] for colormap
        min_FF = smoothed_FF.min()
        max_FF = smoothed_FF.max()
        range_FF = max_FF - min_FF
        norm = colors_mod.Normalize(vmin=min_FF - 0.1 * range_FF, vmax=max_FF + 0.1 * range_FF)

        # Plot the trajectory on the corresponding subplot
        axs.scatter(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, c=smoothed_FF, cmap=cmap, norm=norm, s=5)

        axs.add_patch(patches.Rectangle((-strip_width / 2, 0), strip_width, strip_length, facecolor='white', edgecolor='lightgrey', alpha=0.3))

        if hlines is not None:
            for j in range(len(hlines)):
                axs[i].hlines(y=hlines[j], xmin=-100, xmax=100, colors='k', linestyles='--', linewidth=1)
            
        title = f'fly {fly_number}'

        # Set axes, labels, and title
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        axs.set_xlabel('x position', fontsize=14)
        axs.set_ylabel('y position', fontsize=14)
        axs.set_title('speed', fontsize=14)

        # Further customization
        axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            axs.spines[pos].set_visible(False)

        for _, spine in axs.spines.items():
            spine.set_linewidth(2)
        for spine in axs.spines.values():
            spine.set_edgecolor('black')

        # Apply tight layout to the entire figure
        fig.tight_layout()

        # Save and show the plot
        if save:
            plt.savefig(os.path.join(figure_folder, f'speed_traj_{fly_number}'))
        else:
            plt.show()



def plot_triggered_norm_FF(figure_folder, filename, lobes, colors, window_size=5, event_type='entry'):
    data = open_pickle(filename)
    flylist_n = np.array(list(data.keys()))

    # Create a figure
    fig, axs = plt.subplots(1, len(lobes), figsize=(4 * len(lobes), 5), sharex=True, sharey=True)

    # Set the font to Arial for all text
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Iterate through the lobes, assigning variables like in the previous function
    for i, (lobe, color) in enumerate(zip(lobes, colors)):
        for fly_n, fly_key in enumerate(flylist_n, start=1):
            this_fly = data[fly_key]
            this_fly_all_data = this_fly['a1']

            FF = this_fly_all_data[lobe]
            time = this_fly_all_data['relative_time']

            # Choose the dataset based on event type
            if event_type == 'entry':
                d_event = this_fly['di']
            else:
                d_event = this_fly['do']

            for key, df in d_event.items():
                time_on = df['relative_time'].iloc[0]

                # Extract a window around the event
                window_start = time_on - window_size / 2
                window_end = time_on + window_size / 2
                window_mask = (time >= window_start) & (time <= window_end)

                # Check if data points fall within the window
                if any(window_mask):
                    # Z-score fluorecence
                    normalized_FF = stats.zscore(FF[window_mask])
                    time_aligned = time[window_mask] - time_on
                    # Plot fluorescence aligned to the event time
                    axs[i].plot(time_aligned, normalized_FF, color=color, alpha=0.1, linewidth=0.2)
                    
        # Customize the plot
        axs[i].set_title(lobe)
        axs[i].set_ylim(-3, 8)
        axs[i].set_ylabel('dF/F')
        axs[i].set_xlabel('time (sec)')
        axs[i].grid(False)
        axs[i].vlines(x=0, ymin=-5, ymax=10, color='grey', alpha=0.5, linestyles='--')

    plt.suptitle(f'normalized dF/F at {event_type}')
    plt.savefig(os.path.join(figure_folder, f'norm_FF_{event_type}'))
