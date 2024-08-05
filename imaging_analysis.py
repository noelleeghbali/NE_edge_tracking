import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors_mod
from matplotlib import cm
import seaborn as sns
import os
import pickle
from scipy.stats import gaussian_kde
from behavior_analysis import *


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
    # For Andy's data this will return a dictionary with keys for each fly that was imaged; the corresponding values themselves are
    # dictionaries with keys 'a1', 'd', 'di', 'do', 'ft', all with values storing dataframes

def process_pickle(img_dict, neuron):
    pv2 = img_dict['pv2']
    ft2 = img_dict['ft2']
    ft2['instrip'] = ft2['instrip'].replace({1: True, 0: False})
    ft2[neuron] = pv2[f'0_{neuron}']
    ft2['relative_time'] = pv2['relative_time']
    return pv2, ft2

def fictrac_repair(x,y):
        dx = np.abs(np.diff(x))
        dy = np.abs(np.diff(y))
        lrgx = dx > 5 
        lrgy = dy > 5 
        bth = np.logical_or(lrgx, lrgy)
        
        fixdx = [i+1 for i, b in enumerate(bth) if b]
        for i, f in enumerate(fixdx):
            x[f:] =x[f:] - (x[f]-x[f-1])
            y[f:] = y[f:] - (y[f]-y[f-1])
        return x, y

def trace_FF(figure_folder, neuron):
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            print('found a pickle')
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = 'Arial'
            axs.plot(time, FF, color='k', linewidth=.75)
            d, di, do = inside_outside(ft2)
            for key, df in di.items():
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                timestamp = time_off - time_on
                rectangle = patches.Rectangle((time_on, FF.min()), timestamp, FF.max() + 0.5, facecolor='orange', alpha=0.5)
                axs.add_patch(rectangle)
            #plt.xlim(100, 400)
            plt.title(filename, size=16)
            plt.ylabel('dF/F', size=16)
            plt.xlabel('time (s)', size=16)
            plt.show()

def trace_FF_bw(figure_folder, neuron):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            print('found a pickle')
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = 'Arial'
            fig.patch.set_facecolor('black')  # Set background to black
            axs.set_facecolor('black')
            axs.xaxis.label.set_color('white')
            axs.yaxis.label.set_color('white')
            axs.tick_params(axis='x', colors='white')
            axs.tick_params(axis='y', colors='white')
            axs.spines['bottom'].set_color('white')
            axs.spines['left'].set_color('white')
            axs.plot(time, FF, color=color, linewidth=.75)
            d, di, do = inside_outside(ft2)
            for key, df in di.items():
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                timestamp = time_off - time_on
                rectangle = patches.Rectangle((time_on, FF.min()), timestamp, FF.max() + 0.5, facecolor='#fbceb1', alpha=0.3)
                axs.add_patch(rectangle)
            #plt.xlim(100, 400)
            plt.title(filename, color='white', size=16)
            plt.ylabel('dF/F', color='white', size=16)
            plt.xlabel('time (s)', color='white', size=16)
            plt.show()

def triggered_FF(figure_folder, neuron, tbef=30, taf=30, event_type='entry', first = True):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    all_mn_mat = []
    max_len = 0
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            print(f"Processing file: {filename}")
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            if pv2 is None or ft2 is None:
                print(f"Skipping file {filename} due to missing data.")
                continue
            FF = ft2[neuron]
            time = ft2['relative_time']
            d, di, do = inside_outside(ft2)
            td = ft2['instrip'].to_numpy()
            tdiff = np.diff(td)
            if event_type == 'entry':
                son = np.where((td[:-1] == False) & (td[1:] == True))[0]
            elif event_type == 'exit':
                son = np.where((td[:-1] == True) & (td[1:] == False))[0]
            if len(son) == 0:
                print(f"No {event_type} events found in file: {filename}")
                continue
            print(f"Found {len(son)} {event_type} events in file: {filename}")
            if first:
                son = son[:1]
            tinc = np.mean(np.diff(pv2['relative_time']))
            idx_bef = int(np.round(float(tbef) / tinc))
            ca = FF
            idx_af = int(np.round(float(taf) / tinc))
            mn_mat = np.full((len(son), idx_bef + idx_af + 1), np.nan)  # Initialize with NaNs
            for i, s in enumerate(son):
                idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
                idx_array = idx_array[(idx_array >= 0) & (idx_array < len(ca))]  # Ensure idx_array is within bounds
                segment = ca[idx_array]
                mn_mat[i, :len(segment)] = segment  # Pad with NaNs if segment is shorter
            all_mn_mat.append(mn_mat)
            max_len = max(max_len, mn_mat.shape[1])
    if len(all_mn_mat) == 0:
        print(f"No {event_type} events found in any file.")
        return
    # Pad all matrices to the same length
    padded_mn_mat = []
    for mat in all_mn_mat:
        if mat.shape[1] < max_len:
            padding = np.full((mat.shape[0], max_len - mat.shape[1]), np.nan)
            mat = np.hstack((mat, padding))
        padded_mn_mat.append(mat)
    # Combine results from all files
    combined_mn_mat = np.concatenate(padded_mn_mat, axis=0)
    # Mask NaN values
    masked_combined_mn_mat = np.ma.masked_array(combined_mn_mat, np.isnan(combined_mn_mat))
    # Calculate mean and standard deviation, ignoring NaNs
    plt_mn = np.ma.mean(masked_combined_mn_mat, axis=0)
    std = np.ma.std(masked_combined_mn_mat, axis=0)
    t = np.linspace(-tbef, taf, max_len)
    # Plotting
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    fig.patch.set_facecolor('black')  
    axs.set_facecolor('black')
    axs.xaxis.label.set_color('white')
    axs.yaxis.label.set_color('white')
    axs.tick_params(axis='x', colors='white')
    axs.tick_params(axis='y', colors='white')
    axs.spines['bottom'].set_color('white')
    axs.spines['left'].set_color('white')
    plt.fill_between(t, plt_mn + std, plt_mn - std, color=color, alpha=0.3)
    plt.plot(t, plt_mn, color=color)
    mn = np.min(plt_mn - std)
    mx = np.max(plt_mn + std)
    plt.plot([0, 0], [mn, mx], color='white', linestyle='--')
    plt.xlabel('time (s)', color='white', size=16)
    plt.ylabel('dF/F', color='white', size=16)
    if first:
        plt.title(f"first {event_type}", color='white', size=16)
    if not first:
        plt.title(f"every {event_type}", color='white', size=16)
    plt.show()


def heading_FF(figure_folder, neuron):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            print('found a pickle')
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            heading = ((ft2['ft_heading']+math.pi) % (2*math.pi))-math.pi
            ft2['transformed_heading'] = heading
            ft2 = ft2.replace([np.inf, -np.inf], np.nan).dropna()
            # Extract theta and r
            theta = ft2['transformed_heading']
            r = ft2[neuron]
            # Create a KDE plot
            kde = gaussian_kde([theta, r])
            theta_grid = np.linspace(0, 2 * np.pi, 100)
            r_grid = np.linspace(r.min(), r.max(), 100)
            theta_grid, r_grid = np.meshgrid(theta_grid, r_grid)
            kde_values = kde(np.vstack([theta_grid.ravel(), r_grid.ravel()])).reshape(theta_grid.shape)
            # Create the polar plot
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='polar')
            c = ax.contourf(theta_grid, r_grid, kde_values, cmap='coolwarm')
            # Add a color bar
            plt.colorbar(c, ax=ax, label='Density')
            ax.set_title('Polar Plot of Transformed Heading vs. FF')
            plt.show()
            
def traj_FF(figure_folder, neuron, cmin=0, cmax=1):
    plt.ion()  # Turn on interactive mode
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            print(f"Processing file: {filename}")
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            x = ft2['ft_posx']
            y = ft2['ft_posy']
            x,y = fictrac_repair(x,y)
            colour = ft2[neuron].to_numpy().flatten()
            xrange = np.max(x) - np.min(x)
            yrange = np.max(y) - np.min(y)
            mrange = np.max([xrange, yrange]) + 100
            y_med = np.median(y)
            x_med = np.median(x)
            ylims = [y_med - mrange / 2, y_med + mrange / 2]
            xlims = [x_med - mrange / 2, x_med + mrange / 2]
            acv = ft2['instrip']
            inplume = acv > 0
            c_map = plt.get_cmap('coolwarm')
            if cmin == cmax:
                cmax = np.round(np.percentile(colour[~np.isnan(colour)], 97.5), decimals=1)
            cnorm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
            scalarMap = cm.ScalarMappable(cnorm, c_map)
            c_map_rgb = scalarMap.to_rgba(colour)
            x = x - x[0]
            y = y - y[0]
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = 'Arial'
            fig = plt.figure(figsize=(8, 8))  # Adjust the size here
            ax = fig.add_subplot(111)
            ax.scatter(x[inplume], y[inplume], color=[0.5, 0.5, 0.5])
            for i in range(len(x) - 1):
                ax.plot(x[i:i+2], y[i:i+2], color=c_map_rgb[i+1, :3])
            plt.xlim(xlims)
            plt.ylim(ylims)
            plt.xlabel('x position (mm)')
            plt.ylabel('y position (mm)')
            plt.title(filename)
            ax.set_aspect('equal', adjustable='box')
            plt.show(block=True)  # Ensure the plot stays open


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
    fig, axs = plt.subplots(1, len(lobes), figsize=(4 * len(lobes), 5), sharex=True, sharey=True)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
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
    plt.show()
    plt.savefig(os.path.join(figure_folder, f'norm_FF_{event_type}'))


