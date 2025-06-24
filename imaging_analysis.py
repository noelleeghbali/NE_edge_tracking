import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors_mod
from matplotlib import cm
import statsmodels.api as sm
import seaborn as sns
import os
import pickle
from scipy.stats import gaussian_kde
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import itertools
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
    # print(ft2.columns)
    ft2['instrip'] = ft2['instrip'].replace({1: True, 0: False})
    pv2['instrip'] = ft2['instrip']
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

def rebaseline(pkl_folder, neuron, span=500):
    for filename in os.listdir(pkl_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{pkl_folder}/{filename}')
            pv2 = img_dict['pv2']
            ft2 = img_dict['ft2']
            y = pv2[f'0_{neuron}'].to_numpy()
            ts = pv2['relative_time']
            frac = float(span) / np.max(ts)
            lowess = sm.nonparametric.lowess
            yf = lowess(y, np.arange(0, len(y)), frac=frac)
            if len(y) != len(yf[:, 1]):
                print(f"Warning: Lengths differ: y ({len(y)}), yf[:, 1] ({len(yf[:, 1])}). Correcting lengths.")
                min_len = min(len(y), len(yf[:, 1]))
                y = y[:min_len]
                yf = yf[:min_len, :]
                pv2 = pv2.iloc[:min_len]
            y_og = y.copy()
            y = y - yf[:, 1]
            pv2[f'0_{neuron}'] = y
            img_dict = {'pv2': pv2, 'ft2': ft2}
            pickle_obj(img_dict, f'{pkl_folder}/rb', f'{savename}_rb.pkl')

def add_odor(axs, di, ymin, ymax):
    for key, df in di.items():
        time_on = df['relative_time'].iloc[0]
        time_off = df['relative_time'].iloc[-1]
        timestamp = time_off - time_on
        rectangle = patches.Rectangle((time_on, ymin), timestamp, ymax - ymin, facecolor='#fbceb1', alpha=0.5)
        axs.add_patch(rectangle)

def add_oct(axs, di, ymin, ymax):
    for key, df in di.items():
        time_on = df['relative_time'].iloc[0]
        time_off = df['relative_time'].iloc[-1]
        timestamp = time_off - time_on
        rectangle = patches.Rectangle((time_on, ymin), timestamp, ymax - ymin, facecolor='#89cff0', alpha=0.5)
        axs.add_patch(rectangle)

def trace_FF(figure_folder, neuron, window=(60,360)):
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = 'Arial'
            axs.plot(time, FF, color='k', linewidth=.75)
            d, di, do = inside_outside(ft2)
            add_odor(axs, di, FF.min(), FF.max())
            #plt.FF(100, 400)
            plt.title(filename, size=16)
            plt.ylabel(r'$\Delta$F/F', size=16)
            plt.xlabel('time (s)', size=16)
            if window is not None:
                plt.xlim(window)
            plt.show()
            if not os.path.exists(f'{figure_folder}/fig/trace_FF'):
                os.makedirs(f'{figure_folder}/fig/trace_FF')
            # fig.savefig(f'{figure_folder}/fig/trace_FF/{savename}.pdf', bbox_inches='tight')

def stacked_FF(figure_folder, neuron):
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            net_motion = ft2['net_motion'] 
            y_velocity = ft2['y_velocity']  
            x_velocity = ft2['x_velocity']
            d, di, do = inside_outside(ft2)
            fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
            plt.rcParams['text.color'] = 'black'
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = 'Arial'
            axs[0].plot(time, FF, color='k', linewidth=.75)
            add_odor(axs[0], di, FF.min(), FF.max() + 0.5)
            axs[0].set_title(filename, size=16)
            axs[0].set_ylabel(r'$\Delta$F/F', size=16)
            axs[0].set_xticks([])
            axs[0].set_xlabel('')
            axs[1].plot(time, net_motion, color='k', linewidth=.75)
            add_odor(axs[1], di, net_motion.min(), net_motion.max())
            axs[1].set_ylabel('Net Motion', size=16)
            axs[1].set_xticks([])
            axs[1].set_xlabel('')
            axs[2].plot(time, y_velocity, color='k', linewidth=.75)
            add_odor(axs[2], di, -20, 20)
            axs[2].set_ylim(-20, 20)
            axs[2].set_ylabel('Upwind Velocity', size=16)
            axs[2].set_xlabel('time (s)', size=16)
            axs[3].plot(time, y_velocity, color='k', linewidth=.75)
            add_odor(axs[3], di, -20, 20)
            axs[3].set_ylim(-20, 20)
            axs[3].set_ylabel('Crosswind Velocity', size=16)
            axs[3].set_xlabel('time (s)', size=16)
            #plt.FF(60, 360)
            plt.tight_layout()
            plt.title(filename)
            plt.show()
            if not os.path.exists(f'{figure_folder}/fig/stacked_FF'):
                os.makedirs(f'{figure_folder}/fig/stacked_FF')
            fig.savefig(f'{figure_folder}/fig/stacked_FF/{savename}.pdf', bbox_inches='tight')

def stacked_FF_oct(figure_folder, neuron):
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            net_motion = ft2['net_motion'] 
            y_velocity = ft2['y_velocity']  
            d, di, do = inside_outside_oct(ft2)
            fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = 'Arial'
            axs[0].plot(time, FF, color='k', linewidth=.75)
            add_oct(axs[0], di, FF.min(), FF.max() + 0.5)
            axs[0].set_title(filename, size=16)
            axs[0].set_ylabel(r'$\Delta$F/F', size=16)
            axs[0].set_xticks([])
            axs[0].set_xlabel('')
            axs[1].plot(time, net_motion, color='k', linewidth=.75)
            add_oct(axs[1], di, net_motion.min(), net_motion.max())
            axs[1].set_ylabel('Net Motion', size=16)
            axs[1].set_xticks([])
            axs[1].set_xlabel('')
            axs[2].plot(time, y_velocity, color='k', linewidth=.75)
            add_oct(axs[2], di, -20, 20)
            axs[2].set_ylim(-20, 20)
            axs[2].set_ylabel('Upwind Velocity', size=16)
            axs[2].set_xlabel('time (s)', size=16)
            #plt.FF(60, 360)
            plt.tight_layout()
            plt.show()
            if not os.path.exists(f'{figure_folder}/fig/stacked_FF'):
                os.makedirs(f'{figure_folder}/fig/stacked_FF')
            fig.savefig(f'{figure_folder}/fig/stacked_FF/{savename}.pdf', bbox_inches='tight')

def dF_entries_bw(figure_folder, neuron, thresh=20, sign='pos'):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            fig, axs = configure_bw_plot(size=(10,5), xaxis=True)
            d, di, do = inside_outside(ft2)
            differences = []
            for i, (key, df) in enumerate(di.items()):
                if i >= thresh:
                    break
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                before_mask = (time >= time_on - 0.5) & (time < time_on)
                avg_before = FF[before_mask].mean()
                if sign == 'pos':
                    after_mask = (time > time_on) & (time < time_off)
                    avg_after = FF[after_mask].max()
                    difference = avg_after - avg_before
                elif sign == 'neg':
                    after_mask = (time > time_off - 0.5) & (time <= time_off)
                    avg_after = FF[after_mask].mean()
                    difference = avg_after - avg_before
                avg_after = FF[after_mask].mean()
                difference = avg_after - avg_before
                differences.append(difference)

            entry_numbers = range(1, len(differences) + 1)
            axs.scatter(entry_numbers, differences, color=color)
            axs.plot(entry_numbers, differences, linestyle='-', marker='o', color=color)
            axs.axhline(0, color='white', linestyle='--', linewidth=1)
            axs.set_xticks(entry_numbers)
            axs.set_xticklabels(entry_numbers, color='white')
            plt.title(filename, size=16, color='white')
            plt.ylabel(r'change in $\Delta$F/F', size=16, color='white')
            plt.xlabel('entry #', size=16, color='white')
            plt.show()
            if not os.path.exists(f'{figure_folder}/fig/dF_entries_bw'):
                os.makedirs(f'{figure_folder}/fig/dF_entries_bw')
            fig.savefig(f'{figure_folder}/fig/dF_entries_bw/{savename}.pdf', bbox_inches='tight')

def entry_auc_bw(figure_folder, neuron, unit_time=False, thresh=20, oct=True):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    if unit_time:
        save_path = f'{figure_folder}/fig/entry_auc_time_bw'
    else:
        save_path = f'{figure_folder}/fig/entry_auc_bw'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Initialize dictionary to store AUCs per entry number
    entry_aucs_dict = {}
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            fig, axs, markc = configure_bw_plot(size=(9, 3), xaxis=True)
            if oct:
                d, di, do = inside_outside_oct(ft2)
            else:
                d, di, do = inside_outside(ft2)
            aucs = []
            i = 0
            valid_entries = 0
            entry_numbers = []
            while valid_entries < thresh and i < len(di):
                key, df = list(di.items())[i]
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                entry_numbers.append(i + 1)  # Entry numbers start from 1
                baseline_mask = (time >= time_on - 0.5) & (time < time_on)
                baseline = FF[baseline_mask].mean()
                interval_mask = (time >= time_on) & (time <= time_off)
                FF_adjusted = FF[interval_mask] - baseline
                auc = np.trapz(FF_adjusted, time[interval_mask])
                duration = time_off - time_on  
                if not np.isnan(auc / duration):
                    auc_value = auc / duration if unit_time else auc
                    aucs.append(auc_value)
                    valid_entries += 1
                else:
                    entry_numbers.pop()
                i += 1  # Move to the next entry
            # Collect AUCs across files
            for entry_num, auc_value in zip(entry_numbers, aucs):
                if entry_num not in entry_aucs_dict:
                    entry_aucs_dict[entry_num] = []
                entry_aucs_dict[entry_num].append(auc_value)
            # Plotting per file
            axs.scatter(entry_numbers, aucs, color=color)
            axs.plot(entry_numbers, aucs, linestyle='-', marker='o', color=color)
            axs.axhline(0, color='white', linestyle='--', linewidth=1)
            axs.set_xticks(entry_numbers)
            axs.set_xticklabels(entry_numbers, color='white')
            plt.title(filename, size=16, color='white')
            ylabel = r'$\int_{t_{on}}^{t_{off}} F(t)/F \, dt$' if unit_time else r'$\int_{t_{on}}^{t_{off}} F(t)/F$'
            plt.ylabel(ylabel, size=16, color='white')
            plt.xlabel('Entry #', size=16, color='white')
            plt.show()
            fig.savefig(f'{save_path}/{savename}.pdf', bbox_inches='tight')
    # Compute averages and standard errors
    sorted_entry_numbers = sorted(entry_aucs_dict.keys())
    mean_aucs = [np.mean(entry_aucs_dict[entry_num]) for entry_num in sorted_entry_numbers]
    std_aucs = [np.std(entry_aucs_dict[entry_num], ddof=1) for entry_num in sorted_entry_numbers]
    n_values = [len(entry_aucs_dict[entry_num]) for entry_num in sorted_entry_numbers]
    sem_aucs = [std / np.sqrt(n) if n > 1 else 0 for std, n in zip(std_aucs, n_values)]
    # Generate summary plot
    fig, axs, markc = configure_bw_plot(size=(9, 3), xaxis=True)
    axs.errorbar(
        sorted_entry_numbers,
        mean_aucs,
        yerr=sem_aucs,
        fmt='-o',
        color=color,
        ecolor='lightgray',
        elinewidth=3,
        capsize=0
    )
    axs.axhline(0, color='white', linestyle='--', linewidth=1)
    axs.set_xticks(sorted_entry_numbers)
    axs.set_xticklabels(sorted_entry_numbers, color='white')
    # plt.title('Average AUC across files', size=16, color='white')
    ylabel = r'Average $\int_{t_{on}}^{t_{off}} F(t)/F \, dt$' if unit_time else r'Average $\int_{t_{on}}^{t_{off}} F(t)/F$'
    plt.ylabel(ylabel, size=16, color='white')
    plt.xlabel('entry #', size=16, color='white')
    plt.show()
    fig.savefig(f'{save_path}/average_auc.pdf', bbox_inches='tight')

def entry_peaks_first_two(figure_folder, neuron, thresh=20, plotc='black',oct=True):
    if plotc=='white':
        fig, axs, markc = configure_white_plot(size=(3,5), xaxis=False)
    if plotc=='black':
        fig, axs, markc = configure_bw_plot(size=(3,5), xaxis=False)
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    save_path = f'{figure_folder}/fig/entry_peaks_first_two'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    entry_peaks_dict = {}
    file_peaks = {}  # To maintain pairing of data by file

    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            if oct:
                d, di, do = inside_outside_oct(ft2)
            else:
                d, di, do = inside_outside(ft2)
            peaks = []
            i = 0
            valid_entries = 0
            entry_numbers = []
            while valid_entries < thresh and i < len(di):
                key, df = list(di.items())[i]
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                entry_numbers.append(i + 1)  # Entry numbers start from 1
                baseline_mask = (time >= time_on - 0.5) & (time < time_on)
                baseline = FF[baseline_mask].mean()
                interval_mask = (time >= time_on) & (time <= time_off)
                FF_adjusted = FF[interval_mask] - baseline
                peak_ff = np.max(FF_adjusted)

                # Check for valid peak value
                if not np.isnan(peak_ff):
                    peaks.append(peak_ff)
                    valid_entries += 1
                else:
                    entry_numbers.pop()
                i += 1  # Move to the next entry

            # Store peaks for this file for pairing
            if 1 in entry_numbers and 2 in entry_numbers:
                if savename not in file_peaks:
                    file_peaks[savename] = {}
                file_peaks[savename][1] = peaks[0]
                file_peaks[savename][2] = peaks[1]

    # Extract paired data for entry 1 and 2
    entry1_peaks = []
    entry2_peaks = []
    differences = []
    for file, peaks in file_peaks.items():
        if 1 in peaks and 2 in peaks:
            entry1_peaks.append(peaks[1])
            entry2_peaks.append(peaks[2])
            differences.append((peaks[2] - peaks[1])/peaks[1])
    # Print the differences
    print("Fold changes frrom the first to second peaks:")
    for i, diff in enumerate(differences):
        print(f"File {i + 1}: {diff:.4f}")
    noise = 0.05 * np.random.randn(len(entry1_peaks))    
    # Scatter and connect paired points
    for i, (peak1, peak2) in enumerate(zip(entry1_peaks, entry2_peaks)):
        axs.scatter(1 + noise[i], peak1, color=color, alpha=0.7)
        axs.scatter(2 + noise[i], peak2, color='#0070C0', alpha=0.7)
        axs.plot([1 + noise[i], 2 + noise[i]], [peak1, peak2], color='grey', alpha=0.5)

    # Add space around the edges and set labels
    axs.axhline(0, color=markc, linestyle='--', linewidth=1)
    axs.set_xticks([1, 2])
    axs.set_xticklabels(['entry 1', 'entry 2'], fontsize=14)
    axs.set_xlim(0.5, 2.5)  # Adjust x-axis limits for better spacing
    plt.ylabel('peak dF/F', color=markc, size=16)
    plt.xlabel('entry #', color=markc, size=16)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{save_path}/paired_entry_peaks.pdf', bbox_inches='tight')

def calculate_auc(di, time, FF):
    auc_list = []
    for key, df in di.items():
        time_on = df['relative_time'].iloc[0]
        time_off = df['relative_time'].iloc[-1]
        baseline_mask = (time >= time_on - 0.5) & (time < time_on)
        interval_mask = (time >= time_on) & (time <= time_off)
        if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
            continue
        baseline = FF[baseline_mask].mean()
        FF_adjusted = FF[interval_mask] - baseline
        auc = np.trapz(FF_adjusted, time[interval_mask])
        duration = time_off - time_on
        if duration <= 0 or np.isnan(auc / duration):
            continue
        auc_list.append(auc/duration)
    return auc_list

def calculate_peak(di, time, FF):
    peak_list = []
    for key, df in di.items():
        time_on = df['relative_time'].iloc[0]
        time_off = df['relative_time'].iloc[-1]
        baseline_mask = (time >= time_on - 0.5) & (time < time_on)
        interval_mask = (time >= time_on) & (time <= time_off)
        if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
            continue
        baseline = FF[baseline_mask].mean()
        FF_adjusted = FF[interval_mask] - baseline
        peak_ff = np.max(FF_adjusted)
        peak_list.append(peak_ff)
    return peak_list

def entry_peaks_first_ten(figure_folder, neuron, thresh=10, plotc='black', oct=True):
    if plotc == 'white':
        fig, axs, markc = configure_white_plot(size=(6, 5), xaxis=False)
    if plotc == 'black':
        fig, axs, markc = configure_bw_plot(size=(6, 5), xaxis=False)
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    save_path = f'{figure_folder}/fig/entry_peaks_first_ten'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    entry_peaks_dict = {}
    file_peaks = {}  # To maintain pairing of data by file
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            if oct:
                d, di, do = inside_outside_oct(ft2)
            else:
                d, di, do = inside_outside(ft2)
            peaks = []
            i = 0
            valid_entries = 0
            entry_numbers = []
            while valid_entries < thresh and i < len(di):
                key, df = list(di.items())[i]
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                entry_numbers.append(i + 1)  # Entry numbers start from 1
                baseline_mask = (time >= time_on - 0.5) & (time < time_on)
                baseline = FF[baseline_mask].mean()
                interval_mask = (time >= time_on) & (time <= time_off)
                FF_adjusted = FF[interval_mask] - baseline
                peak_ff = np.max(FF_adjusted)

                # Check for valid peak value
                if not np.isnan(peak_ff):
                    peaks.append(peak_ff)
                    valid_entries += 1
                else:
                    entry_numbers.pop()
                i += 1  # Move to the next entry

            # Store peaks for this file for pairing
            for j in range(min(thresh, len(peaks))):
                if savename not in file_peaks:
                    file_peaks[savename] = {}
                file_peaks[savename][j + 1] = peaks[j]

    # Extract paired data for entries 1 through 10
    entry_peaks = {i: [] for i in range(1, thresh + 1)}
    for file, peaks in file_peaks.items():
        for entry in range(1, thresh + 1):
            if entry in peaks:
                entry_peaks[entry].append(peaks[entry])

    # Scatter plot for each entry
    noise = 0.05 * np.random.randn(len(entry_peaks[1]))
    for entry in range(1, thresh + 1):
        if entry in entry_peaks:
            for i, peak in enumerate(entry_peaks[entry]):
                axs.scatter(entry + noise[i], peak, color=color if entry == 1 else '#0070C0', alpha=0.7)

    # Plot connections between consecutive entries
    for i in range(len(entry_peaks[1])):
        x = []
        y = []
        for entry in range(1, thresh + 1):
            if len(entry_peaks[entry]) > i:
                x.append(entry + noise[i])
                y.append(entry_peaks[entry][i])
        axs.plot(x, y, color='grey', alpha=0.5)

    # Add space around the edges and set labels
    axs.axhline(0, color=markc, linestyle='--', linewidth=1)
    axs.set_xticks(range(1, thresh + 1))
    axs.set_xticklabels([f'entry {i}' for i in range(1, thresh + 1)], fontsize=10, rotation=45)
    axs.set_xlim(0.5, thresh + 0.5)  # Adjust x-axis limits for better spacing
    plt.ylabel('peak dF/F', color=markc, size=16)
    plt.xlabel('entry #', color=markc, size=16)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{save_path}/paired_entry_peaks_first_ten.pdf', bbox_inches='tight')

def dF_entries_time_bw(figure_folder, neuron, thresh, sign='pos'):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            fig, axs = configure_bw_plot(size=(10,5), xaxis=True)
            d, di, do = inside_outside(ft2)
            differences = []
            entry_times = []
            i = 0
            valid_entries = 0  # Counter for valid entries
            while valid_entries < thresh and i < len(di):
                key, df = list(di.items())[i]
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                entry_times.append(time_on)
                before_mask = (time >= time_on - 0.5) & (time < time_on)
                avg_before = FF[before_mask].mean() if before_mask.any() else np.nan
                if sign == 'pos':
                    after_mask = (time > time_on) & (time < time_off)
                    avg_after = FF[after_mask].max() if after_mask.any() else np.nan
                elif sign == 'neg':
                    after_mask = (time > time_off - 0.5) & (time <= time_off)
                    avg_after = FF[after_mask].mean() if after_mask.any() else np.nan
                difference = avg_after - avg_before
                if not np.isnan(difference):  # Only append non-NaN differences
                    differences.append(difference)
                    valid_entries += 1  # Increment valid entry count
                else:
                    entry_times.pop()  # Remove the last added invalid time_on value
                    print(f"Missing data for interval {i}, trying next entry.")
                i += 1  # Move to the next interval
            axs.scatter(entry_times, differences, color=color)
            axs.plot(entry_times, differences, linestyle='-', marker='o', color=color)
            axs.axhline(0, color='white', linestyle='--', linewidth=1)
            for i, txt in enumerate(range(1, len(differences) + 1)):
                axs.annotate(txt, (entry_times[i], differences[i]), textcoords="offset points", xytext=(0,5), ha='center', color='white')
            plt.title(filename, size=16, color='white')
            plt.ylabel(r'change in $\Delta$F/F', size=16, color='white')
            plt.xlabel('time (s)', size=16, color='white')
            plt.show()
            if not os.path.exists(f'{figure_folder}/fig/dF_entries_time_bw'):
                os.makedirs(f'{figure_folder}/fig/dF_entries_time_bw')
            fig.savefig(f'{figure_folder}/fig/dF_entries_time_bw/{savename}.pdf', bbox_inches='tight')


def trace_FF_bw(figure_folder, neuron, window=(30,360)): 
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            fig, axs, markc = configure_bw_plot((15,4), xaxis=True)
            d, di, do = inside_outside(ft2)
            print(di)
            for key, df in di.items():
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                timestamp = time_off - time_on
                rectangle = patches.Rectangle((time_on, (FF.min()-0.1)), timestamp, (FF.max() - FF.min() + 0.2), facecolor='#ffa700', alpha=0.6)
                axs.add_patch(rectangle)
            axs.plot(time, FF, color='white', linewidth=1)
            plt.title(filename, color='white', size=16)
            plt.ylabel(r'$\Delta$F/F', color='white', size=16)
            plt.xlabel('time (s)', color='white', size=16)
            if not os.path.exists(f'{figure_folder}/fig/trace_FF_bw'):
                os.makedirs(f'{figure_folder}/fig/trace_FF_bw')
            if window is not None:
                plt.xlim(window)
                fig.savefig(f'{figure_folder}/fig/trace_FF_bw/{savename}_{window[0]}{window[1]}.pdf', bbox_inches='tight')
            elif window is None:
                fig.savefig(f'{figure_folder}/fig/trace_FF_bw/{savename}.pdf', bbox_inches='tight')
            plt.show()

def et_replay_traces_bw(figure_folder, lobes, colors, size):
    paired_files = {}
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            if '_et_' in filename:
                prefix = filename.split('_et_')[0]
                paired_files.setdefault(prefix, {})['et'] = filename
            elif '_replay_' in filename:
                prefix = filename.split('_replay_')[0]
                paired_files.setdefault(prefix, {})['replay'] = filename
    for fly_id, files in paired_files.items():
        if 'et' not in files or 'replay' not in files:
            print(f"Skipping {fly_id}: missing pair")
            continue
    
        nrows = len(lobes) + 1
        ncols = 2
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True, nrows=nrows, ncols=ncols)
        axs = np.array(axs).reshape(nrows, ncols)
        fig.suptitle(fly_id)
        
        data_et = open_pickle(os.path.join(figure_folder, files['et']))
        data_replay = open_pickle(os.path.join(figure_folder, files['replay']))
        pv2_et = data_et['strip1']['pv2']
        pv2_replay = data_replay['strip1']['pv2']
        d_et, di_et, _ = inside_outside(pv2_et)
        d_replay, di_replay, _ = inside_outside(pv2_replay)
        for row_idx, lobe in enumerate(lobes):
            FF_et = pv2_et[lobe] / 100
            FF_replay = pv2_replay[lobe] / 100
            ymin = min(FF_et.min(), FF_replay.min())
            ymax = max(FF_et.max(), FF_replay.max())
            # ET plot
            ax = axs[row_idx][0]
            ax.plot(pv2_et['relative_time'], FF_et, linewidth=0.5, color=colors[row_idx % len(colors)])
            add_odor(ax, di_et, ymin, ymax)
            ax.set_ylim([ymin, ymax])  
            ax.set_ylabel(lobe, color=markc)
            # Replay plot
            ax = axs[row_idx][1]
            ax.plot(pv2_replay['relative_time'], FF_replay, linewidth=0.5, color=colors[row_idx % len(colors)])
            add_odor(ax, di_replay, ymin, ymax)
            ax.set_ylim([ymin, ymax])
        
        axs[0, 0].set_title('edge tracking', color=markc)
        axs[0, 1].set_title('replay', color=markc)    

        # add net motion
        row_idx = len(lobes)
        FF_et = pv2_et['net_motion']
        FF_replay = pv2_replay['net_motion']
        ymin = min(FF_et.min(), FF_replay.min())
        ymax = max(FF_et.max(), FF_replay.max())

        ax = axs[row_idx][0]
        ax.plot(pv2_et['relative_time'], FF_et, linewidth=0.5, color=markc)
        add_odor(ax, di_et, ymin, ymax)
        ax.set_ylim([ymin, ymax])
        ax.set_ylabel('net motion', color=markc)

        ax = axs[row_idx][1]
        ax.plot(pv2_replay['relative_time'], FF_replay, linewidth=0.5, color=markc)
        add_odor(ax, di_replay, ymin, ymax)
        ax.set_ylim([ymin, ymax])
        plt.tight_layout()
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/traces'):
            os.makedirs(f'{figure_folder}/fig/traces')
        fig.savefig(f'{figure_folder}/fig/traces/{fly_id}_et_vs_replay.pdf', bbox_inches='tight')


def et_replay_auc_comp(figure_folder, lobes, colors, size):
    paired_files = {}
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            if '_et_' in filename:
                prefix = filename.split('_et_')[0]
                paired_files.setdefault(prefix, {})['et'] = filename
            elif '_replay_' in filename:
                prefix = filename.split('_replay_')[0]
                paired_files.setdefault(prefix, {})['replay'] = filename
    for fly_id, files in paired_files.items():
        if 'et' not in files or 'replay' not in files:
            print(f"Skipping {fly_id}: missing pair")
            continue
    
        nrows = len(lobes)
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True, nrows=nrows)
        axs = np.ravel(axs)         
        fig.suptitle(fly_id)
        
        data_et = open_pickle(os.path.join(figure_folder, files['et']))
        data_replay = open_pickle(os.path.join(figure_folder, files['replay']))
        pv2_et = data_et['strip1']['pv2']
        pv2_replay = data_replay['strip1']['pv2']
        time_et = pv2_et['relative_time']
        time_replay = pv2_replay['relative_time']
        d_et, di_et, _ = inside_outside(pv2_et)
        d_replay, di_replay, _ = inside_outside(pv2_replay)

        aucs_et = {f'G{i+2}': [] for i in range(len(lobes))}
        aucs_replay = {f'G{i+2}': [] for i in range(len(lobes))}

        for i, lobe in enumerate(lobes):
            FF_et = pv2_et[lobe] / 100
            FF_replay = pv2_replay[lobe] / 100

            aucs_et[f'G{i+2}'] = calculate_auc(di_et, time_et, FF_et)
            aucs_replay[f'G{i+2}'] = calculate_auc(di_replay, time_replay, FF_replay)

            entry_number_et = range(1, len(aucs_et[f'G{i+2}']) + 1)
            entry_number_replay = range(1, len(aucs_replay[f'G{i+2}']) + 1)

            axs[i].plot(entry_number_et, aucs_et[f'G{i+2}'], color=colors[i % len(colors)], alpha=0.8)
            axs[i].plot(entry_number_replay, aucs_replay[f'G{i+2}'], color=colors[i % len(colors)], alpha=0.3)
            axs[i].set_ylabel(lobe, color=markc)
           

        plt.tight_layout()
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/auc_comp'):
            os.makedirs(f'{figure_folder}/fig/auc_comp')
        fig.savefig(f'{figure_folder}/fig/auc_comp/{fly_id}_et_vs_replay_entry_auc.pdf', bbox_inches='tight')

def et_replay_tuning(figure_folder, lobes, colors, size):
    paired_files = {}
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            if '_et_' in filename:
                prefix = filename.split('_et_')[0]
                paired_files.setdefault(prefix, {})['et'] = filename
            elif '_replay_' in filename:
                prefix = filename.split('_replay_')[0]
                paired_files.setdefault(prefix, {})['replay'] = filename

    for fly_id, files in paired_files.items():
        if 'et' not in files or 'replay' not in files:
            print(f"Skipping {fly_id}: missing pair")
            continue

        # Setup 2x4 plot: ET row 0, Replay row 1
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True, nrows=2, ncols=len(lobes))
        axs = np.array(axs).reshape(2, len(lobes))
        fig.suptitle(fly_id)

        # Load data
        data_et = open_pickle(os.path.join(figure_folder, files['et']))
        data_replay = open_pickle(os.path.join(figure_folder, files['replay']))
        df_et = data_et['strip1']['pv2']
        df_replay = data_replay['strip1']['pv2']

        # Calculate entry angles
        _, di_et, do_et = inside_outside(df_et)
        _, di_replay, do_replay = inside_outside(df_replay)
        entry_et = compute_mean_entry_heading(do_et)
        entry_replay = compute_mean_entry_heading(do_replay)

        bins = np.arange(-180, 181, 10)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        for row_idx, (df, entry_heading) in enumerate([(df_et, entry_et), (df_replay, entry_replay)]):
            df = df[df['net_motion'] > 0]  # Restrict to moving

            heading_rad = ((df['heading'] + np.pi) % (2 * np.pi)) - np.pi
            heading_deg = np.rad2deg(heading_rad)

            for col_idx, lobe in enumerate(lobes):
                FF = df[lobe] / 100
                binned_FF = [[] for _ in range(len(bin_centers))]

                for angle, ff_val in zip(heading_deg, FF):
                    bin_idx = np.digitize(angle, bins) - 1
                    if 0 <= bin_idx < len(binned_FF):
                        binned_FF[bin_idx].append(ff_val)

                mean_FF = [np.mean(vals) if vals else np.nan for vals in binned_FF]
                ax = axs[row_idx, col_idx]

                for j, vals in enumerate(binned_FF):
                    if vals:
                        jitter = np.random.uniform(-4, 4, size=len(vals))
                        ax.scatter(np.full(len(vals), bin_centers[j]), vals,
                                   color=colors[col_idx], alpha=0.6, s=10)

                ax.plot(bin_centers, mean_FF, color=markc, linestyle='-')
                ax.set_xlim(-180, 180)
                ax.set_xticks(np.arange(-180, 181, 60))
                # ax.axhline(0, color=markc, linestyle='--')
                ax.axvline(0, color=markc, linestyle='--')

                if entry_heading is not None:
                    ax.axvline(entry_heading, color='white', linestyle=':', linewidth=1)

                if row_idx == 0:
                    ax.set_title(f'{lobe} ET', color=markc)
                else:
                    ax.set_xlabel('heading (deg)', color=markc)
                    if col_idx == 0:
                        ax.set_ylabel('dF/F')
                    ax.set_title(f'{lobe} replay', color=markc)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/tuning'):
            os.makedirs(f'{figure_folder}/fig/tuning')
        fig.savefig(f'{figure_folder}/fig/tuning/{fly_id}_entry_heading_tuning.pdf', bbox_inches='tight')


def et_replay_peak_comp(figure_folder, lobes, colors, size):
    paired_files = {}
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            if '_et_' in filename:
                prefix = filename.split('_et_')[0]
                paired_files.setdefault(prefix, {})['et'] = filename
            elif '_replay_' in filename:
                prefix = filename.split('_replay_')[0]
                paired_files.setdefault(prefix, {})['replay'] = filename
    for fly_id, files in paired_files.items():
        if 'et' not in files or 'replay' not in files:
            print(f"Skipping {fly_id}: missing pair")
            continue
    
        nrows = len(lobes)
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True, nrows=nrows)
        axs = np.ravel(axs)         
        fig.suptitle(fly_id)
        
        data_et = open_pickle(os.path.join(figure_folder, files['et']))
        data_replay = open_pickle(os.path.join(figure_folder, files['replay']))
        pv2_et = data_et['strip1']['pv2']
        pv2_replay = data_replay['strip1']['pv2']
        time_et = pv2_et['relative_time']
        time_replay = pv2_replay['relative_time']
        d_et, di_et, _ = inside_outside(pv2_et)
        d_replay, di_replay, _ = inside_outside(pv2_replay)

        peaks_et = {f'G{i+2}': [] for i in range(len(lobes))}
        peaks_replay = {f'G{i+2}': [] for i in range(len(lobes))}

        for i, lobe in enumerate(lobes):
            FF_et = pv2_et[lobe] / 100
            FF_replay = pv2_replay[lobe] / 100

            peaks_et[f'G{i+2}'] = calculate_peak(di_et, time_et, FF_et)
            peaks_replay[f'G{i+2}'] = calculate_peak(di_replay, time_replay, FF_replay)

            entry_number_et = range(1, len(peaks_et[f'G{i+2}']) + 1)
            entry_number_replay = range(1, len(peaks_replay[f'G{i+2}']) + 1)

            axs[i].plot(entry_number_et, peaks_et[f'G{i+2}'], color=colors[i % len(colors)], alpha=0.8)
            axs[i].plot(entry_number_replay, peaks_replay[f'G{i+2}'], color=colors[i % len(colors)], alpha=0.3)
            axs[i].set_ylabel(lobe, color=markc)
           

        plt.tight_layout()
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/peaks'):
            os.makedirs(f'{figure_folder}/fig/peaks')
        fig.savefig(f'{figure_folder}/fig/peaks/{fly_id}_peaks_per_entry.pdf', bbox_inches='tight')

def oct_trace_FF_bw(figure_folder, neuron, window=(30,360)): 
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            fig, axs = configure_bw_plot((15,4), xaxis=True)
            d, di, do = inside_outside_oct(ft2)
            for key, df in di.items():
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                timestamp = time_off - time_on
                rectangle = patches.Rectangle((time_on, (FF.min()-0.1)), timestamp, (FF.max() - FF.min() + 0.2), facecolor='#89cff0', alpha=0.8)
                axs.add_patch(rectangle)
            axs.plot(time, FF, color='white', linewidth=1)
            plt.title(filename, color='white', size=16)
            plt.ylabel(r'$\Delta$F/F', color='white', size=16)
            plt.xlabel('time (s)', color='white', size=16)
            if not os.path.exists(f'{figure_folder}/fig/trace_FF_bw'):
                os.makedirs(f'{figure_folder}/fig/trace_FF_bw')
            if window is not None:
                plt.xlim(window)
                fig.savefig(f'{figure_folder}/fig/trace_FF_bw/{savename}_{window[0]}{window[1]}.pdf', bbox_inches='tight')
            elif window is None:
                fig.savefig(f'{figure_folder}/fig/trace_FF_bw/{savename}.pdf', bbox_inches='tight')
            plt.show()

def trace_FF_bw_bouts(figure_folder, neuron, pre_post_time=5, separation=1, thresh=None, oct=True):
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            fig, axs, markc = configure_bw_plot((15, 4), xaxis=True)
            if oct == True:
                d, di, do = inside_outside_oct(ft2)
            if oct == False:
                d, di, do = inside_outside(ft2)
            # Create lists to store concatenated traces and times
            concatenated_FF = []
            concatenated_time = []
            current_x = 0  # Initial position on the new x-axis
            # Determine the number of bouts to visualize based on thresh
            if thresh is None:
                thresh = len(di)  # Use all bouts if thresh is not specified
            # Process only the specified number of bouts
            for i, (key, df) in enumerate(di.items()):
                if i >= thresh:
                    break  # Stop after the specified number of bouts
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                # Get the indices for the odor period plus pre and post time
                start_idx = np.searchsorted(time, time_on - pre_post_time)
                end_idx = np.searchsorted(time, time_off + pre_post_time)
                # Extract the corresponding data
                epoch_FF = FF[start_idx:end_idx]
                epoch_time = time[start_idx:end_idx] - time[start_idx] + current_x
                # Concatenate to the lists
                concatenated_FF.extend(epoch_FF)
                concatenated_time.extend(epoch_time)
                # Add NaN to separate this bout from the next one
                concatenated_FF.append(np.nan)
                concatenated_time.append(np.nan)
                # Draw rectangle for the odor period
                odor_duration = time_off - time_on
                rectangle = patches.Rectangle((current_x + pre_post_time, (FF.min()-0.1)),
                                              odor_duration, (FF.max() - FF.min() + 0.2),
                                              facecolor='#ffa700', alpha=0.6)
                axs.add_patch(rectangle)
                axs.axvline(x=current_x, color='white', linestyle='--', linewidth=1)
                axs.axvline(x=current_x + pre_post_time + odor_duration + pre_post_time, color='white', linestyle='--', linewidth=1)
                # Update the current_x position for the next epoch
                current_x = concatenated_time[-2] + separation  # -2 to skip the NaN added at the end
            # Plot the concatenated epochs
            axs.plot(concatenated_time, concatenated_FF, color='white', linewidth=1)
            plt.title(filename, color='white', size=16)
            plt.ylabel(r'$\Delta$F/F', color='white', size=16)
            plt.xlabel('concatenated time (s)', color='white', size=16)
            output_folder = f'{figure_folder}/fig/trace_FF_bw'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            fig.savefig(f'{output_folder}/{savename}_epochs.pdf', bbox_inches='tight')
            plt.show()


def triggered_FF(figure_folder, neuron, tbef=30, taf=30, event_type='entry', first = True):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    all_mn_mat = []
    max_len = 0
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
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
    plt.ylabel(r'$\Delta$F/F', color='white', size=16)
    if first:
        plt.title(f"first {event_type}", color='white', size=16)
        fig.savefig(f'{figure_folder}/fig/first {event_type}.pdf', bbox_inches='tight')
    if not first:
        plt.title(f"every {event_type}", color='white', size=16)
        fig.savefig(f'{figure_folder}/fig/every {event_type}.pdf', bbox_inches='tight')
    plt.show()

def interp_return_FF(figure_folder, neuron):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    
    max_len = 100  # Choose the number of points for the normalized time axis
    
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            all_normalized_traces = []
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            if pv2 is None or ft2 is None:
                print(f"Skipping file {filename} due to missing data.")
                continue
            FF = ft2[neuron]
            time = ft2['relative_time']
            d, di, do = inside_outside(ft2)
            td = ft2['instrip'].to_numpy()
            # Identify exit events (transition from inside to outside)
            exit_onsets = np.where((td[:-1] == True) & (td[1:] == False))[0]
            if len(exit_onsets) == 0:
                print(f"No exit events found in file: {filename}")
                continue
            print(f"Found {len(exit_onsets)} exit events in file: {filename}")
            for onset in exit_onsets:
                # Find the end of the exit, which is when the 'outside' state ends
                end_idx_array = np.where(td[onset:] == True)[0]
                if len(end_idx_array) > 1:
                    end_idx = onset + end_idx_array[1]  # Use the second 'end' index
                elif len(end_idx_array) == 1:
                    end_idx = onset + end_idx_array[0]  # If only one end index exists
                else:
                    end_idx = len(td) - 1  # If exit continues until the end of recording
                # Calculate bout duration and slice FF segment
                bout_duration = end_idx - onset + 1
                # Debugging output
                print(f"onset: {onset}, end_idx: {end_idx}, bout_duration: {bout_duration}")
                if bout_duration <= 1:
                    print(f"Skipping invalid bout with duration {bout_duration}")
                    continue
                ff_segment = FF[onset:end_idx + 1]
                # Ensure ff_segment is not empty
                if ff_segment.size == 0:
                    print(f"Skipping empty ff_segment for file: {filename}, onset: {onset}, end_idx: {end_idx}")
                    continue
                # Normalize the time for this segment
                normalized_time = np.linspace(0, 1, max_len)
                interp_function = interp1d(np.linspace(0, 1, bout_duration), ff_segment, kind='linear')
                interpolated_trace = interp_function(normalized_time)   
                all_normalized_traces.append(interpolated_trace)
            if len(all_normalized_traces) == 0:
                print(f"No valid bouts found in file: {filename}")
                continue
            all_normalized_traces = np.array(all_normalized_traces)
            mean_trace = np.mean(all_normalized_traces, axis=0)
            std_trace = np.std(all_normalized_traces, axis=0)
            mean_trace_interp = np.interp(np.arange(len(mean_trace)), np.arange(len(mean_trace))[~np.isnan(mean_trace)], mean_trace[~np.isnan(mean_trace)])
            std_trace_interp = np.interp(np.arange(len(std_trace)), np.arange(len(std_trace))[~np.isnan(std_trace)], std_trace[~np.isnan(std_trace)])
            t_norm = np.linspace(0, 1, max_len)
            fig, axs, markc = configure_bw_plot(size=(6,5), xaxis=True)
            axs.fill_between(t_norm, mean_trace_interp + std_trace_interp, mean_trace_interp - std_trace_interp, color=color, alpha=0.3)
            axs.plot(t_norm, mean_trace_interp, color=color)
            plt.xlabel('Normalized time', color='white', size=16)
            plt.ylabel(r'$\Delta$F/F', color='white', size=16)
            plt.title(f'Average fluorescence trace for {savename}', color='white', size=16)
            plt.show()
            if not os.path.exists(f'{figure_folder}/fig/interp_return_FF'):
                os.makedirs(f'{figure_folder}/fig/interp_return_FF')
            fig.savefig(f'{figure_folder}/fig/interp_return_FF/{savename}.pdf', bbox_inches='tight')

def interp_jump_FF(figure_folder, neuron):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    max_len = 100
    for filename in os.listdir(figure_folder):
        all_normalized_traces = []
        if filename.endswith('.pkl'):
            all_normalized_traces = []
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            td = ft2['instrip'].to_numpy()

            #store jumped exit onsets
            jumped_exits = []
            exit_onsets = np.where((td[:-1] == True) & (td[1:] == False))[0]
            last_jump = None
            for onset in exit_onsets:
                current_jump = ft2.loc[onset, 'jump']
                if last_jump is None or current_jump != last_jump:
                    jumped_exits.append(onset)
                    last_jump = current_jump

            for onset in jumped_exits:
                # Find the end of the exit, which is when the 'outside' state ends
                end_idx_array = np.where(td[onset:] == True)[0]
                if len(end_idx_array) > 1:
                    end_idx = onset + end_idx_array[1]  # Use the second 'end' index
                elif len(end_idx_array) == 1:
                    end_idx = onset + end_idx_array[0]  # If only one end index exists
                else:
                    end_idx = len(td) - 1  # If exit continues until the end of recording
                # Calculate bout duration and slice FF segment
                bout_duration = end_idx - onset + 1
                # Debugging output
                print(f"onset: {onset}, end_idx: {end_idx}, bout_duration: {bout_duration}")
                if bout_duration <= 1:
                    print(f"Skipping invalid bout with duration {bout_duration}")
                    continue
                ff_segment = FF[onset:end_idx + 1]
                # Ensure ff_segment is not empty
                if ff_segment.size == 0:
                    print(f"Skipping empty ff_segment for file: {filename}, onset: {onset}, end_idx: {end_idx}")
                    continue
                # Normalize the time for this segment
                normalized_time = np.linspace(0, 1, max_len)
                interp_function = interp1d(np.linspace(0, 1, bout_duration), ff_segment, kind='linear')
                interpolated_trace = interp_function(normalized_time)   
                all_normalized_traces.append(interpolated_trace)
            all_normalized_traces = np.array(all_normalized_traces)
            mean_trace = np.mean(all_normalized_traces, axis=0)
            std_trace = np.std(all_normalized_traces, axis=0)
            mean_trace_interp = np.interp(np.arange(len(mean_trace)), np.arange(len(mean_trace))[~np.isnan(mean_trace)], mean_trace[~np.isnan(mean_trace)])
            std_trace_interp = np.interp(np.arange(len(std_trace)), np.arange(len(std_trace))[~np.isnan(std_trace)], std_trace[~np.isnan(std_trace)])
            t_norm = np.linspace(0, 1, max_len)
            fig, axs = configure_bw_plot(size=(6,5), xaxis=True)
            axs.fill_between(t_norm, mean_trace_interp + std_trace_interp, mean_trace_interp - std_trace_interp, color=color, alpha=0.3)
            axs.plot(t_norm, mean_trace_interp, color=color)
            plt.xlabel('Normalized time', color='white', size=16)
            plt.ylabel(r'$\Delta$F/F', color='white', size=16)
            plt.title(f'Average fluorescence trace for {savename}', color='white', size=16)
            plt.show()
            if not os.path.exists(f'{figure_folder}/fig/interp_jump_FF'):
                os.makedirs(f'{figure_folder}/fig/interp_jump_FF')
            fig.savefig(f'{figure_folder}/fig/interp_jump_FF/{savename}.pdf', bbox_inches='tight')

def triggered_zFF(figure_folder, neuron, tbef=30, taf=30, event_type='entry', first=True):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    all_mn_mat = []
    max_len = 0
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
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
            # Normalize the data (Z-score normalization)
            mn_mat = stats.zscore(mn_mat, axis=None, nan_policy='omit')
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
    combined_mn_mat = np.concatenate(padded_mn_mat, axis=0)
    masked_combined_mn_mat = np.ma.masked_array(combined_mn_mat, np.isnan(combined_mn_mat))
    plt_mn = np.ma.mean(masked_combined_mn_mat, axis=0)
    std = np.ma.std(masked_combined_mn_mat, axis=0)
    t = np.linspace(-tbef, taf, max_len)
    fig, axs, markc = configure_bw_plot(size=(4, 6), xaxis=True)
    plt.fill_between(t, plt_mn + std, plt_mn - std, color=color, alpha=0.3)
    plt.plot(t, plt_mn, color=color)
    mn = np.min(plt_mn - std)
    mx = np.max(plt_mn + std)
    plt.ylim(-2,8)
    plt.plot([0, 0], [-2, 8], color='white', linestyle='--')
    plt.xlabel('time (s)', color='white', size=16)
    plt.ylabel('Z-score', color='white', size=16)
    figure_folder = f'{figure_folder}/fig/triggered_zFF'
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    if first:
        plt.title(f"first {event_type}", color='white', size=16)
        fig.savefig(f'{figure_folder}/ffirst_{event_type}_Zscored.pdf', bbox_inches='tight')
    if not first:
        plt.title(f"every {event_type}", color='white', size=16)
        fig.savefig(f'{figure_folder}/every_{event_type}_Zscored.pdf', bbox_inches='tight')
    plt.show()

def triggered_zFF_first_two(figure_folder, neuron, tbef=30, taf=30, event_type='entry', plotc='white', oct=True):
    # Define colors for the first and second events
    if neuron == 'mbon09':
        color_first = '#bd33a4'
        color_second = '#0070C0'  # Example color for the second event
    elif neuron == 'mbon21':
        color_first = '#ff4f00'
        color_second = '#0070C0'
    elif neuron == 'mbon30':
        color_first = '#00cc99'
        color_second = '#0070C0'
    all_mn_mat_first = []
    all_mn_mat_second = []
    max_len_first = 0
    max_len_second = 0
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            if pv2 is None or ft2 is None:
                print(f"Skipping file {filename} due to missing data.")
                continue
            FF = ft2[neuron]
            time = ft2['relative_time']
            if oct:
                td = (ft2['mfc3_stpt'] > 0).to_numpy()
            else:
                td = ft2['instrip'].to_numpy()
            tdiff = np.diff(td)
            # Identify event indices
            if event_type == 'entry':
                son = np.where((td[:-1] == False) & (td[1:] == True))[0]
            elif event_type == 'exit':
                son = np.where((td[:-1] == True) & (td[1:] == False))[0]
            if len(son) == 0:
                print(f"No {event_type} events found in file: {filename}")
                continue
            print(f"Found {len(son)} {event_type} events in file: {filename}")
            tinc = np.mean(np.diff(pv2['relative_time']))
            idx_bef = int(np.round(float(tbef) / tinc))
            idx_af = int(np.round(float(taf) / tinc))
            total_len = idx_bef + idx_af + 1
            # Process first event
            if len(son) >= 1:
                s1 = son[0]
                idx_array = np.arange(s1 - idx_bef, s1 + idx_af + 1, dtype=int)
                idx_array_valid = idx_array[(idx_array >= 0) & (idx_array < len(FF))]
                segment = np.full(total_len, np.nan)
                valid_len = len(idx_array_valid)
                segment[:valid_len] = FF[idx_array_valid]
                # Baseline normalization
                baseline_indices = np.arange(idx_bef)[(idx_array[:idx_bef] >= 0)]
                baseline_mean = np.nanmean(segment[baseline_indices])
                segment -= baseline_mean
                all_mn_mat_first.append(segment)
                max_len_first = max(max_len_first, len(segment))
            # Process second event
            if len(son) >= 2:
                s2 = son[1]
                # Check for overlap between first event's post-event window and second event's pre-event window
                overlap = (s2 - idx_bef) < (s1 + idx_af)
                if overlap:
                    print(f"Second event in file {filename} overlaps with first event's post-event window. Adjusting time windows.")
                    # Adjust idx_bef for second event
                    max_idx_bef = s2 - (s1 + idx_af)
                    if max_idx_bef <= 0:
                        print(f"Not enough time before second event in file {filename} to prevent overlap. Skipping second event.")
                        continue
                    idx_bef_second = max_idx_bef
                else:
                    idx_bef_second = idx_bef
                # Adjust total_len for second event
                total_len_second = idx_bef_second + idx_af + 1
                idx_array = np.arange(s2 - idx_bef_second, s2 + idx_af + 1, dtype=int)
                idx_array_valid = idx_array[(idx_array >= 0) & (idx_array < len(FF))]
                segment = np.full(total_len_second, np.nan)
                valid_len = len(idx_array_valid)
                segment[:valid_len] = FF[idx_array_valid]
                # Baseline normalization
                baseline_indices = np.arange(idx_bef_second)[(idx_array[:idx_bef_second] >= 0)]
                baseline_mean = np.nanmean(segment[baseline_indices])
                segment -= baseline_mean
                all_mn_mat_second.append(segment)
                max_len_second = max(max_len_second, len(segment))
    if len(all_mn_mat_first) == 0 and len(all_mn_mat_second) == 0:
        print(f"No {event_type} events found in any file.")
        return
    # Prepare time vectors
    t_first = np.linspace(-tbef, taf, max_len_first)
    t_second = np.linspace(-tbef, taf, max_len_second)
    if plotc == 'white':
        fig, axs, markc = configure_white_plot(size=(4,5), xaxis=True)
    elif plotc == 'black':
        fig, axs, markc = configure_bw_plot(size=(4,5), xaxis=True)
    if len(all_mn_mat_first) > 0:
        padded_mn_mat_first = []
        for segment in all_mn_mat_first:
            if len(segment) < max_len_first:
                padding = np.full(max_len_first - len(segment), np.nan)
                segment = np.hstack((segment, padding))
            padded_mn_mat_first.append(segment)
        combined_mn_mat_first = np.vstack(padded_mn_mat_first)
        masked_combined_mn_mat_first = np.ma.masked_array(combined_mn_mat_first, np.isnan(combined_mn_mat_first))
        for trace in masked_combined_mn_mat_first:
            axs.plot(t_first, trace, color=color_first, alpha=0.5, linewidth=0.8)  # Set alpha and linewidth for better visibility
        plt_mn_first = np.ma.mean(masked_combined_mn_mat_first, axis=0)
        axs.plot(t_first, plt_mn_first, color=color_first, label='first entry', linewidth=2)
    if len(all_mn_mat_second) > 0:
        # Pad sequences for the second events
        padded_mn_mat_second = []
        for segment in all_mn_mat_second:
            if len(segment) < max_len_second:
                padding = np.full(max_len_second - len(segment), np.nan)
                segment = np.hstack((segment, padding))
            padded_mn_mat_second.append(segment)
        combined_mn_mat_second = np.vstack(padded_mn_mat_second)
        masked_combined_mn_mat_second = np.ma.masked_array(combined_mn_mat_second, np.isnan(combined_mn_mat_second))
        for trace in masked_combined_mn_mat_second:
            axs.plot(t_second, trace, color=color_second, alpha=0.5, linewidth=0.8)  # Set alpha and linewidth for better visibility
        plt_mn_second = np.ma.mean(masked_combined_mn_mat_second, axis=0)
        axs.plot(t_second, plt_mn_second, color=color_second, label='second entry', linewidth=2)
    # Adjust y-axis limits and plot vertical line at x=0
    ymin, ymax = axs.get_ylim()
    if plotc == 'white':
        axs.plot([0, 0], [ymin, ymax], color='black', linestyle='--')
    elif plotc == 'black':
        axs.plot([0, 0], [ymin, ymax], color='white', linestyle='--')
    # Set labels and legend
    axs.set_xlabel('time (s)', fontsize=16)
    if plotc == 'white':
        axs.set_ylabel(r'$\Delta$F/F', color='black', size=16)
    if plotc == 'black':
        axs.set_ylabel(r'$\Delta$F/F', color='white', size=16)
    # plt.legend()
    plt.tight_layout()

    # Save and show the figure
    figure_output_folder = f'{figure_folder}/fig/triggered_zFF'
    if not os.path.exists(figure_output_folder):
        os.makedirs(figure_output_folder)
    if plotc == 'white':
        fig.savefig(f'{figure_output_folder}/first_second_{event_type}_baseline_normalized.pdf', bbox_inches='tight')
    elif plotc == 'black':
        fig.savefig(f'{figure_output_folder}/first_second_{event_type}_baseline_normalized_bw.pdf', bbox_inches='tight')
    plt.show()


def FF_xpos_corr(figure_folder, neuron):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    fig, axs = configure_bw_plot(size=(5,5), xaxis=True)
    xpos_all = []
    FF_all = []
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            ft2 = ft2.replace([np.inf, -np.inf], np.nan).dropna()
            FF = ft2[neuron]
            time = ft2['relative_time']
            odor_start = ft2[ft2['instrip']].index[0]
            xo = ft2.loc[odor_start, 'ft_posx']
            xpos = np.abs(ft2['ft_posx'] - xo)
            xpos_all.extend(xpos.tolist())
            FF_all.extend(FF.tolist())
    
    plt.scatter(xpos_all, FF_all, color=color)
    plt.xlabel('X Position')
    plt.ylabel(r'$\Delta F/F$')
    plt.title(r'Correlation between X Position and $\Delta F/F$', color='white')

    X = np.array(xpos_all).reshape(-1, 1)
    y = np.array(FF_all)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    plt.plot(xpos_all, y_pred, color='white', linestyle='-')

    r_squared = reg.score(X, y)
    plt.text(max(xpos_all) * 0.7, max(FF_all) * 0.9, f'R² = {r_squared:.2f}', color='white')
    
    plt.show()
    #fig.savefig(f'{figure_folder}/fig/FF_xpos_corr.pdf', bbox_inches='tight')

def FF_avel_corr(figure_folder, neuron, sign='pos'): ##MESSYYYYYY
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            fig, axs = configure_bw_plot(size=(5,5), xaxis=True)
            avel_all = []
            FF_all = []
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            ft2 = ft2.replace([np.inf, -np.inf], np.nan).dropna()
            try:
                odor_start_idx = ft2[ft2['instrip']].index[0]
            except IndexError:
                print(f"No 'instrip' event found in {filename}, skipping file.")
                continue
            odor_start = ft2[ft2['instrip']].iloc[0]
            ft2 = ft2.loc[odor_start_idx:]
            ft2 = ft2[ft2['instrip']==False]
            FF = ft2[neuron]
            avel = (ft2['ang_velocity'])
            if sign == 'pos':
                avel[avel < 0] = 0
            elif sign == 'neg':
                avel[avel > 0] = 0
            avel_all.extend(avel.tolist())
            FF_all.extend(FF.tolist())
            plt.scatter(avel_all, FF_all, color=color)
            plt.xlabel('angular velocity (mm/s)')
            plt.ylabel(r'$\Delta F/F$')
            plt.title(r'correlation between angular velocity and $\Delta F/F$', color='white')
            X = np.array(avel_all).reshape(-1, 1)
            y = np.array(FF_all)
            reg = LinearRegression().fit(X, y)
            av_pred = reg.predict(X)
            plt.plot(avel_all, av_pred, color='white', linestyle='-')
            r_squared = reg.score(X, y)
            plt.title(savename, color='white', fontsize=16)
            plt.text(max(avel_all) * 0.7, max(FF_all) * 0.9, f'R² = {r_squared:.2f}', color='white')
            plt.show()
            #fig.savefig(f'{figure_folder}/fig/FF_xpos_corr.pdf', bbox_inches='tight')

def FF_avel_corr_both(figure_folder, neuron):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            ft2 = ft2.replace([np.inf, -np.inf], np.nan).dropna()
            
            try:
                odor_start_idx = ft2[ft2['instrip']].index[0]
            except IndexError:
                print(f"No 'instrip' event found in {filename}, skipping file.")
                continue
            
            odor_start = ft2[ft2['instrip']].iloc[0]
            ft2 = ft2.loc[odor_start_idx:]
            ft2 = ft2[ft2['instrip'] == False]
            FF = ft2[neuron]
            avel = ft2['ang_velocity']
            
            for i, sign in enumerate(['neg', 'pos']):
                avel_sign = avel.copy()                
                if sign == 'pos':
                    avel_sign[avel_sign < 0] = 0
                    
                elif sign == 'neg':
                    avel_sign[avel_sign > 0] = 0
                
                avel_all = avel_sign.tolist()
                FF_all = FF.tolist()
                
                # Scatter plot
                axs[i].scatter(avel_all, FF_all, color=color)
                axs[i].set_xlabel('Angular velocity (mm/s)')
                axs[i].set_ylabel(r'$\Delta F/F$')
                axs[i].set_title(f'{sign.capitalize()} angular velocity', color='white')
                
                # Perform linear regression
                X = np.array(avel_all).reshape(-1, 1)
                y = np.array(FF_all)
                reg = LinearRegression().fit(X, y)
                av_pred = reg.predict(X)
                axs[i].plot(avel_all, av_pred, color='white', linestyle='-')
                r_squared = reg.score(X, y)
                axs[i].text(max(avel_all) * 0.7, max(FF_all) * 0.9, f'R² = {r_squared:.2f}', color='white')
            
            plt.suptitle(savename, color='k', fontsize=16)
            plt.tight_layout()
            plt.show()

def FF_selected_avel_corr(figure_folder, neuron):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            fig, axs = configure_bw_plot(size=(5, 5), xaxis=True)
            avel_all = []
            FF_all = []
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            ft2 = ft2.replace([np.inf, -np.inf], np.nan).dropna()
            
            try:
                odor_start_idx = ft2[ft2['instrip']].index[0]
            except IndexError:
                print(f"No 'instrip' event found in {filename}, skipping file.")
                continue
            
            # Analyze position to determine turn direction
            x_pos = ft2['ft_posx']
            y_pos = ft2['ft_posy']
            ang_vel = ft2['ang_velocity']
            
            # Slice data before and after plume entry
            pre_odor = ft2.iloc[:odor_start_idx]
            post_odor = ft2.iloc[odor_start_idx:]

            # Calculate direction of entry into the plume
            x_change = np.mean(post_odor['ft_posx']) - np.mean(pre_odor['ft_posx'])
            y_change = np.mean(post_odor['ft_posy']) - np.mean(pre_odor['ft_posy'])

            # Determine the main direction of the turn
            if abs(x_change) > abs(y_change):
                if x_change < 0:
                    turn_direction = 'left'  # Negative x-direction
                    relevant_avel = ang_vel[ft2['ft_posx'].diff() < 0]
                else:
                    turn_direction = 'right'  # Positive x-direction
                    relevant_avel = ang_vel[ft2['ft_posx'].diff() > 0]
            else:
                if y_change < 0:
                    turn_direction = 'down'  # Negative y-direction
                    relevant_avel = ang_vel[ft2['ft_posy'].diff() < 0]
                else:
                    turn_direction = 'up'  # Positive y-direction
                    relevant_avel = ang_vel[ft2['ft_posy'].diff() > 0]
            
            # Select FF data corresponding to the relevant angular velocity
            FF = ft2.loc[relevant_avel.index, neuron]
            
            # Store data for correlation analysis
            avel_all.extend(relevant_avel.tolist())
            FF_all.extend(FF.tolist())
            
            # Plot correlation
            plt.scatter(avel_all, FF_all, color=color)
            plt.xlabel('Angular velocity')
            plt.ylabel(r'$\Delta F/F$')
            plt.title(r'Correlation between angular velocity and $\Delta F/F$', color='white')
            
            # Perform linear regression
            X = np.array(avel_all).reshape(-1, 1)
            y = np.array(FF_all)
            reg = LinearRegression().fit(X, y)
            av_pred = reg.predict(X)
            plt.plot(avel_all, av_pred, color='white', linestyle='-')
            r_squared = reg.score(X, y)
            plt.text(max(avel_all) * 0.7, max(FF_all) * 0.9, f'R² = {r_squared:.2f}', color='white')
            plt.show()

def FF_time_corr(figure_folder, neuron):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    differences = []
    corrs = []
    fig, axs = configure_bw_plot(size=(5,5), xaxis=True)
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            ft2 = ft2.replace([np.inf, -np.inf], np.nan).dropna()
            FF = ft2[neuron]
            time = ft2['relative_time']
            d, di, do = inside_outside(ft2)
            bouts = []
            for i, (key, df) in enumerate(do.items()):
                if i == 0:
                    continue  # Skip the first bout
                if df['relative_time'].iloc[-1] - df['relative_time'].iloc[0] > 0.5:
                    exit_bout = key
                    entry_bout = exit_bout + 1
                    if entry_bout in di:
                        bouts.append((exit_bout, entry_bout))
                        corr = df['relative_time'].iloc[-1] - df['relative_time'].iloc[0]
                        corrs.append(corr)
            
            for exit_bout, entry_bout in bouts:
                print(entry_bout)
                df = di[entry_bout]
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                before_mask = (time >= time_on - 0.5) & (time < time_on)
                avg_before = FF[before_mask].mean()
                after_mask = (time > time_off - 0.5) & (time <= time_off)
                avg_after = FF[after_mask].mean()
                difference = avg_after - avg_before
                differences.append(difference)
    plt.scatter(corrs, differences[:(len(corrs)+1)], color=color)
    plt.xlabel('time (s)')
    plt.ylabel(r'change in $\Delta$F/F')
    plt.title(r'reentry $\Delta$F/F vs. time since last encounter', color='white')
    # Fitting a line of best fit
    X = np.array(corrs).reshape(-1, 1)
    y = np.array(differences[:len(corrs)])
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    plt.plot(corrs, y_pred, color='white', linestyle='--')
    # Calculating R-squared value
    r_squared = reg.score(X, y)
    plt.text(max(corrs) * 0.7, max(differences) * 0.9, f'R² = {r_squared:.2f}', color='white')
    plt.show()
    fig.savefig(f'{figure_folder}/fig/FF_time_corr.pdf', bbox_inches='tight')

def peak_FF_heading_vs_entry_angle(figure_folder, neuron):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            d, di, do = inside_outside(ft2)
            FF = ft2[neuron]
            time = ft2['relative_time']
            bouts = []
            entry_angles = []
            peak_FF_headings = []
            fig, axs, markc = configure_bw_plot(size=(10,5), xaxis=True)
            for i, (key, df) in enumerate(do.items()):
                if i == 0:
                    continue  # Skip the first bout
                if df['relative_time'].iloc[-1] - df['relative_time'].iloc[0] > 0.5:
                    exit_bout = key
                    entry_bout = exit_bout - 1
                    if entry_bout in di:
                        bouts.append((exit_bout, entry_bout))
            # print(bouts)
            for exit_bout, entry_bout in bouts:
                entry_data = di[entry_bout]
                entry_window = entry_data[(entry_data['relative_time'] - entry_data['relative_time'].iloc[0]) <= 0.5]
                entry_angle = entry_window['ft_heading'].mean()
                entry_angles.append(entry_angle * (180 / np.pi))
 
                exit_data = do[exit_bout]
                peak_index = FF[exit_data.index].idxmax()
                peak_time = time.loc[peak_index]
                peak_window = exit_data[(exit_data['relative_time'] >= peak_time - 0.5) & (exit_data['relative_time'] <= peak_time + 0.5)]
                peak_heading = peak_window['ft_heading'].mean()
                peak_FF_headings.append(peak_heading * (180 / np.pi))
            print(entry_angles)
            print(peak_FF_headings)
            bouts_range = range(len(entry_angles))
            axs.plot(bouts_range, entry_angles, label="entry angle", linestyle='-', color='red')
            axs.plot(bouts_range, peak_FF_headings, label="heading at peak dF/F in subsequent exit", linestyle='--', color='blue')
            axs.set_xticks(bouts_range)
            axs.set_xticklabels([str(i) for i in bouts_range], rotation=0)
            axs.set_title(filename, color=markc)
            axs.set_xlabel('bout')
            axs.set_ylabel('angle (degrees)')
            axs.legend()    
            # Save and display the plot
            plt.tight_layout()            
            plt.show()
            if not os.path.exists(f'{figure_folder}/fig/heading_entry_corr'):
                os.makedirs(f'{figure_folder}/fig/heading_entry_corr')
            fig.savefig(f'{figure_folder}/fig/heading_entry_corr/heading_entry_corr.pdf', bbox_inches='tight')

def peak_FF_heading_vs_entry_angle_corr(figure_folder, neuron):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            FF = ft2[neuron]
            time = ft2['relative_time']
            d, di, do = inside_outside(ft2)
            FF = ft2[neuron]
            time = ft2['relative_time']
            bouts = []
            entry_angles = []
            peak_FF_headings = []
            fig, axs, markc = configure_bw_plot(size=(10,5), xaxis=True)
            for i, (key, df) in enumerate(do.items()):
                if i == 0:
                    continue  # Skip the first bout
                if df['relative_time'].iloc[-1] - df['relative_time'].iloc[0] > 0.5:
                    exit_bout = key
                    entry_bout = exit_bout - 1
                    if entry_bout in di:
                        bouts.append((exit_bout, entry_bout))
            # print(bouts)
            for exit_bout, entry_bout in bouts:
                entry_data = di[entry_bout]
                entry_window = entry_data[(entry_data['relative_time'] - entry_data['relative_time'].iloc[0]) <= 0.5]
                entry_angle = entry_window['ft_heading'].mean()
                entry_angles.append(entry_angle * (180 / np.pi))
 
                exit_data = do[exit_bout]
                peak_index = FF[exit_data.index].idxmax()
                peak_time = time.loc[peak_index]
                peak_window = exit_data[(exit_data['relative_time'] >= peak_time - 0.5) & (exit_data['relative_time'] <= peak_time + 0.5)]
                peak_heading = peak_window['ft_heading'].mean()
                peak_FF_headings.append(peak_heading * (180 / np.pi))
            
            axs.scatter(entry_angles, peak_FF_headings, color='blue', label='Bout Pairs', alpha=0.7)

            # Add labels to each scatter point
            for i, (x, y) in enumerate(zip(entry_angles, peak_FF_headings)):
                axs.annotate(f'{i}', (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)

            # Set axis labels and title
            axs.set_title(f"Entry Angle vs Peak FF Heading ({filename})")
            axs.set_xlabel('Entry Angle (degrees)')
            axs.set_ylabel('Peak FF Heading (degrees)')

            # Ensure the y-axis range matches degrees
            axs.set_xlim([-180, 180])  # Adjust range to match data
            axs.set_ylim([-180, 180])  # Adjust range to match data
            axs.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # Add horizontal line at y=0
            axs.axvline(0, color='gray', linestyle='--', linewidth=0.5)  # Add vertical line at x=0
            axs.legend()

            # Save and display the plot
            plt.tight_layout()
            plt.show()

            if not os.path.exists(f'{figure_folder}/fig/heading_entry_corr'):
                os.makedirs(f'{figure_folder}/fig/heading_entry_corr')
            fig.savefig(f'{figure_folder}/fig/heading_entry_corr/entry_vs_peak_scatter.pdf', bbox_inches='tight')

def FF_peaks(figure_folder, neuron):
    # Define neuron-specific colors
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
   
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            ft2 = ft2.replace([np.inf, -np.inf], np.nan).dropna()
            FF = ft2[neuron]
            time = ft2['relative_time']
            ft2_odor = ft2[ft2['instrip']]
            d, di, do = inside_outside(ft2)
            peak_windows = []
            odor_stamps = []
            
            # Variable to track the last 'jump' value
            last_jump_value = None

            for key, df in di.items():
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                timestamp = time_off - time_on
                
                # Get the current 'jump' value
                current_jump_value = df['jump'].iloc[0]
                current_jump_value = current_jump_value - (current_jump_value % 3)
                # Check if the 'jump' value has changed
                if last_jump_value is None or current_jump_value != last_jump_value:
                    color = np.random.rand(3,)  # Random color for each new jump value
                    last_jump_value = current_jump_value

                odor_stamps.append([time_on, time_off, color])
                
            for i, (key, df) in enumerate(do.items()):
                if i == 0:
                    continue  # Skip the first bout
                if df['relative_time'].iloc[-1] - df['relative_time'].iloc[0] > 0.5:
                    activity = FF.loc[df.index].to_numpy().flatten()
                    peak = np.round(np.percentile(activity[~np.isnan(activity)], 10), decimals=1)
                    
                    # Generate a 5-second time window around the peak
                    peak_time = df['relative_time'].iloc[np.argmax(activity)]  # Get the time corresponding to the peak
                    window_size = 5  # 5-second window
                    half_window = window_size / 2
                    
                    window_start = peak_time - half_window
                    window_end = peak_time + half_window
                    
                    # Store the peak window for highlighting later
                    peak_windows.append((window_start, window_end))

            # Plotting different variables
            fig, axs = plt.subplots(7, 1, figsize=(12, 8), sharex=True)

            y_velocity = ft2['y_velocity']
            y_velocity[np.abs(y_velocity) > 20] = 0
            axs[0].plot(time, y_velocity, color='k')
            axs[0].set_ylabel('Y Velocity')
            axs[0].set_title(f'{savename} - Y Velocity')
            
            x_velocity = ft2['x_velocity']
            x_velocity[np.abs(x_velocity) > 20] = 0
            axs[1].plot(time, x_velocity, color='k')
            axs[1].set_ylabel('X Velocity')
            axs[1].set_title(f'{savename} - X Velocity')

            # Plot Angular Velocity
            ang_velocity = ft2['ang_velocity']
            axs[2].plot(time, ang_velocity, color='k')
            axs[2].set_ylabel('Angular Velocity')
            axs[2].set_title(f'{savename} - Angular Velocity')

            # Plot X Position
            pos_x = ft2['ft_posx']
            axs[3].plot(time, pos_x, color='k')
            axs[3].set_ylabel('X Position')
            axs[3].set_title(f'{savename} - X Position')

            # Plot Y Position
            pos_y = ft2['ft_posy']
            axs[4].plot(time, pos_y, color='k')
            axs[4].set_ylabel('Y Position')
            axs[4].set_title(f'{savename} - Y Position')

            net_motion = ft2['net_motion']
            axs[5].plot(time, net_motion, color='k')
            axs[5].set_ylabel('Net Motion')
            axs[5].set_title(f'{savename} - Net Motion')

            FF = ft2[neuron]
            axs[6].plot(time, FF, color='k')
            axs[6].set_ylabel('dF/F')
            axs[6].set_title(f'{savename} - dF/F')

            # Highlight the odor blocks with changing colors based on jump values
            for timestamp in odor_stamps:
                for ax in axs:
                    ax.axvspan(timestamp[0], timestamp[1], color=timestamp[2], alpha=0.3)

            # Highlight the peak windows with red blocks
            # for window_start, window_end in peak_windows:
            #     for ax in axs:
            #         ax.axvspan(window_start, window_end, color='red', alpha=0.3)

            axs[-1].set_xlabel('Time (s)')
            plt.tight_layout()
            plt.show()       
                    

def inbound_outbound_FF(figure_folder, neuron, window=6):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            fig, axs = configure_bw_plot(size=(5,6), xaxis=True)
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            ft2 = ft2.replace([np.inf, -np.inf], np.nan).dropna()
            FF = ft2[neuron]
            time = ft2['relative_time']
            d, di, do = inside_outside(ft2)
            outies = []
            innies = []
            middles = []
            for key, df in do.items():
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                duration = time_off - time_on
                if duration > window:
                    outbound_mask = (time > time_on) & (time <= time_on + (window/6))
                    avg_outbound = FF[outbound_mask].mean()
                    inbound_mask = (time > time_off - (window/6)) & (time <= time_off)
                    avg_inbound = FF[inbound_mask].mean()
                    middle_start = time_on + (duration - (window/3)) / 2
                    middle_end = middle_start + 5
                    middle_mask = (time >= middle_start) & (time <= middle_end)
                    avg_middle = FF[middle_mask].mean()
                    outies.append(avg_outbound)
                    innies.append(avg_inbound)
                    middles.append(avg_middle)
            axs.set_xticks([1, 2, 3])
            axs.set_xticklabels(['outbound', 'middle', 'inbound'], color='white', fontsize=16)
            axs.set_ylabel(r'avg $\Delta F/F$', color='white', fontsize=16)
            ob_avg = sum(outies) / len(outies)
            ib_avg = sum(innies) / len(innies)
            mid_avg = sum(middles) / len(middles)
            noise = 0.05
            x_o = np.random.normal(1, noise, size=len(outies))
            x_m = np.random.normal(2, noise, size=len(middles))
            x_i = np.random.normal(3, noise, size=len(innies))
            plt.scatter(x_o, outies, color=color, alpha=0.5)
            plt.scatter(x_m, middles, color=color, alpha=0.5)
            plt.scatter(x_i, innies, color=color, alpha=0.5)
            plt.plot([1, 2, 3], [ob_avg, mid_avg, ib_avg], color='white')
            plt.title(savename, color='white', fontsize=16)
            plt.tight_layout()
            plt.show()
            if not os.path.exists(f'{figure_folder}/fig/inbound_outbound_FF'):
                os.makedirs(f'{figure_folder}/fig/inbound_outbound_FF')
            fig.savefig(f'{figure_folder}/fig/inbound_outbound_FF/{savename}_{window}.pdf', bbox_inches='tight')
            
# def FF_tuning(figure_folder, neuron):
#     if neuron == 'mbon09':
#         color = '#bd33a4'
#     elif neuron == 'mbon21':
#         color = '#ff4f00'
#     elif neuron == 'mbon30':
#         color = '#00cc99'
        
#     all_entry_angles = []
#     bins = np.arange(-180, 181, 10)  # 10-degree bins from 0 to 360
#     bin_centers = (bins[:-1] + bins[1:]) / 2
#     binned_FF = np.zeros(len(bin_centers))

#     for filename in os.listdir(figure_folder):
#         if filename.endswith('.pkl'):
#             fig, axs, markc = configure_bw_plot(size=(6,5), xaxis=True)
#             savename, extension = os.path.splitext(filename)
#             img_dict = open_pickle(f'{figure_folder}/{filename}')
#             pv2, ft2 = process_pickle(img_dict, neuron)
#             ft2 = calculate_trav_dir(ft2)
#             ft2['trav_dir'] = np.rad2deg(ft2['trav_dir'])
#             FF = ft2[neuron]
#             heading = ((ft2['ft_heading'] + math.pi) % (2 * math.pi)) - math.pi
#             ft2['transformed_heading'] = heading
#             # ft2['transformed_heading'] = np.rad2deg(heading)
#             ft2 = ft2.replace([np.inf, -np.inf], np.nan).dropna()
#             d, di, do = inside_outside(ft2)

#             entry_headings = []
#             for key, df in di.items():
#                 if df['relative_time'].iloc[-1] - df['relative_time'].iloc[0] >= 1:
#                     df = get_last_second(df)
#                     entry_angle = circmean_heading(df, entry_headings) # append entry angle to list
#             entry_angle = np.mean(np.rad2deg(np.array(entry_headings)))  
#             print(entry_angle)
#             all_entry_angles.append(entry_angle)  
                    
#             for i in range(len(bins) - 1):
#                 mask = (np.rad2deg(ft2['transformed_heading']) >= bins[i]) & (np.rad2deg(ft2['transformed_heading']) < bins[i + 1])
#                 mask = mask.reindex(FF.index, fill_value=False)  # Align mask with FF index
#                 FF_masked = FF[mask]
#                 if not FF_masked.empty:
#                     binned_FF[i] = FF_masked.mean()
#                 else:
#                     binned_FF[i] = np.nan  # Handle the case with no values
            
#             plt.plot(bin_centers, binned_FF, linestyle='-', color=color)
#             plt.axvline(x=entry_angle, color=markc, linestyle='--')
#             plt.xlabel('heading (degrees)')
#             plt.ylabel('dF/F')
#             plt.title(f'{neuron} - {savename}', color=markc)
#             plt.xlim(-180, 180)
#             plt.xticks(np.arange(-180, 181, 60))
#             plt.tight_layout()

#             plt.show()
#             if not os.path.exists(f'{figure_folder}/fig/tuning'):
#                 os.makedirs(f'{figure_folder}/fig/tuning')
#             fig.savefig(f'{figure_folder}/fig/tuning/{savename}.pdf', bbox_inches='tight', facecolor='black')

def FF_tuning(figure_folder, neuron):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
        
    bins = np.arange(-180, 181, 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            fig, ax, markc = configure_bw_plot(size=(4, 3), xaxis=True)
            savename, _ = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            ft2 = calculate_trav_dir(ft2)
            ft2['trav_dir'] = np.rad2deg(ft2['trav_dir'])

            FF = ft2[neuron]
            heading_rad = ((ft2['ft_heading'] + np.pi) % (2 * np.pi)) - np.pi
            heading_deg = np.rad2deg(heading_rad)
            ft2['transformed_heading'] = heading_deg

            ft2 = ft2.replace([np.inf, -np.inf], np.nan).dropna()
            d, di, do = inside_outside(ft2)

            entry_headings = []
            for key, df in di.items():
                if df['relative_time'].iloc[-1] - df['relative_time'].iloc[0] >= 1:
                    df = get_last_second(df)
                    angle = circmean_heading(df, entry_headings)
            entry_angle = np.mean(np.rad2deg(np.array(entry_headings)))
            print(entry_angle)

            # Bin FF values
            binned_FF = [[] for _ in range(len(bin_centers))]
            for angle, ff_val in zip(heading_deg, FF):
                if np.isfinite(ff_val):  # Only use valid (non-NaN, non-inf) values
                    bin_idx = np.digitize(angle, bins) - 1
                    if 0 <= bin_idx < len(binned_FF):
                        binned_FF[bin_idx].append(ff_val)
            mean_FF = [np.mean(vals) if vals else np.nan for vals in binned_FF]
            mean_FF = pd.Series(mean_FF).interpolate(limit_direction='both').tolist()

            # Scatter plot with jitter
            for j, vals in enumerate(binned_FF):
                if vals:
                    jitter = np.random.uniform(-4, 4, size=len(vals))
                    ax.scatter(np.full(len(vals), bin_centers[j]), vals, color=color, alpha=0.6, s=10)

            # Mean FF line
            ax.plot(bin_centers, mean_FF, color=markc, linestyle='-')

            # Axes and annotations
            ax.axvline(entry_angle, color=markc, linestyle=':', linewidth=1)
            ax.axvline(0, color=markc, linestyle='--', linewidth=1)
            # ax.axhline(0, color=markc, linestyle='--', linewidth=1)
            ax.set_xlabel('heading (deg)')
            ax.set_ylabel('dF/F')
            ax.set_xlim(-180, 180)
            ax.set_xticks(np.arange(-180, 181, 60))
            ax.set_title(f'{neuron} - {savename}', color=markc)

            plt.tight_layout()
            plt.show()
            if not os.path.exists(f'{figure_folder}/fig/tuning'):
                os.makedirs(f'{figure_folder}/fig/tuning')
            fig.savefig(f'{figure_folder}/fig/tuning/{savename}.pdf', bbox_inches='tight', facecolor='black')

            



def traj_FF(figure_folder, neuron):
    for filename in os.listdir(figure_folder):
        if filename.endswith('.pkl'):
            savename, extension = os.path.splitext(filename)
            img_dict = open_pickle(f'{figure_folder}/{filename}')
            pv2, ft2 = process_pickle(img_dict, neuron)
            x = ft2['ft_posx']
            y = ft2['ft_posy']
            x, y = fictrac_repair(x, y)
            colour = ft2[neuron].to_numpy().flatten()
            cmin = np.round(np.percentile(colour[~np.isnan(colour)], 1), decimals=1)  
            cmax = np.round(np.percentile(colour[~np.isnan(colour)], 99), decimals=1) 
            xrange = np.max(x) - np.min(x)
            yrange = np.max(y) - np.min(y)
            mrange = np.max([xrange, yrange]) + 100
            y_med = np.median(y)
            x_med = np.median(x)
            ylims = [y_med - mrange / 2, y_med + mrange / 2]
            xlims = [x_med - mrange / 2, x_med + mrange / 2]
            FFs = [x_med - mrange / 2, x_med + mrange / 2]
            if (ft2['mfc2_stpt'] == 0).all():
                inplume = ft2.mfc3_stpt>0
            if (ft2['mfc3_stpt'] == 0).all():
                inplume = ft2.mfc2_stpt>0
            c_map = plt.get_cmap('coolwarm')
            cnorm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
            scalarMap = cm.ScalarMappable(norm=cnorm, cmap=c_map)
            c_map_rgb = scalarMap.to_rgba(colour)
            x = x - x[0]
            y = y - y[0]
            
            # Set the background and text colors
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = 'Arial'
            plt.rcParams['figure.facecolor'] = 'black'  # Background color of the figure
            plt.rcParams['axes.facecolor'] = 'black'    # Background color of the axes
            plt.rcParams['axes.edgecolor'] = 'white'    # Edge color of the axes
            plt.rcParams['axes.labelcolor'] = 'white'   # Color of the labels
            plt.rcParams['xtick.color'] = 'white'       # Color of the x-tick labels
            plt.rcParams['ytick.color'] = 'white'       # Color of the y-tick labels
            plt.rcParams['text.color'] = 'white'        # Color of the text
            
            fig = plt.figure(figsize=(8, 8))  # Adjust the size here
            ax = fig.add_subplot(111)
            ax.scatter(x[inplume], y[inplume], color=[0.5, 0.5, 0.5])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')
            for i in range(len(x) - 1):
                ax.plot(x[i:i+2], y[i:i+2], color=c_map_rgb[i+1, :3])
            plt.xlabel('x position (mm)')
            plt.ylabel('y position (mm)')
            plt.title(filename, color='white')
            ax.set_aspect('equal', adjustable='box')
            if not os.path.exists(f'{figure_folder}/fig/traj_FF'):
                os.makedirs(f'{figure_folder}/fig/traj_FF')
            fig.savefig(f'{figure_folder}/fig/traj_FF/{savename}.pdf', bbox_inches='tight', facecolor='black')
            plt.show()

# def AS_DA_tone:
    

# def plot_FF_trajectory(figure_folder, filename, lobes, colors, strip_width, strip_length, FF, ylim, hlines=[], save=False, keyword=None):
#     data = open_pickle(filename)
#     flylist_n = np.array(list(data.keys()))

#     if not os.path.exists(figure_folder):
#         os.makedirs(figure_folder)

#     # Determine flies to plot
#     if keyword is not None and 1 <= keyword <= len(flylist_n):
#         # Plotting for a specific fly
#         flies_to_plot = [(keyword, data[flylist_n[keyword - 1]])]
#     else:
#         # Plotting for all flies
#         flies_to_plot = [(fly_key, this_fly) for fly_key, this_fly in enumerate(data.values(), start=1)]
    
#     # Plot data
#     for fly_number, this_fly in flies_to_plot:
#         # Assign the correct dataframe ('a1' is all the data collected)
#         this_fly_all_data = this_fly['a1']
        
#         # Create a single figure
#         fig, axs = plt.subplots(1, len(lobes), figsize=(8 * len(lobes), 10))
#         plt.rcParams['font.family'] = 'sans-serif'
#         plt.rcParams['font.sans-serif'] = 'Arial'
#         fig.patch.set_facecolor('black')  # Set background to black

#         # Set font color to white
#         plt.rcParams['text.color'] = 'white'
#         plt.rcParams['axes.labelcolor'] = 'white'
#         plt.rcParams['xtick.color'] = 'white'
#         plt.rcParams['ytick.color'] = 'white'

#         # Iterate through the lobes
#         for i, (lobe, color) in enumerate(zip(lobes, colors)):
#             # Filter df to exp start
#             first_on_index = this_fly_all_data[this_fly_all_data['instrip']].index[0]
#             exp_df = this_fly_all_data.loc[first_on_index:] # This filters the dataframe
#             # Establish the plume origin, at first odor onset
#             xo = exp_df.iloc[0]['ft_posx']
#             yo = exp_df.iloc[0]['ft_posy']
            
#             # Assign FF (fluorescence) to the correct df column
#             FF = exp_df[lobe]
#             # Smooth fluorescence data
#             smoothed_FF = FF.rolling(window=10, min_periods=1).mean()

#             # Define color map
#             cmap = plt.get_cmap('coolwarm')

#             # Normalize FF to [0, 1] for colormap
#             min_FF = smoothed_FF.min()
#             max_FF = smoothed_FF.max()
#             range_FF = max_FF - min_FF
#             norm = colors_mod.Normalize(vmin=min_FF - 0.1 * range_FF, vmax=max_FF + 0.1 * range_FF)
#             axs[i].add_patch(patches.Rectangle((-strip_width / 2, 0), strip_width, strip_length, facecolor='black', edgecolor='white'))
#             # Plot the trajectory on the corresponding subplot
#             axs[i].set_facecolor('black')  # Set background of plotting area to black
#             # Plot each segment individually
#             for j in range(len(exp_df) - 1):
#                 # Coordinates of the current segment's start and end points
#                 x1 = exp_df['ft_posx'].iloc[j] - xo
#                 y1 = exp_df['ft_posy'].iloc[j] - yo
#                 x2 = exp_df['ft_posx'].iloc[j + 1] - xo
#                 y2 = exp_df['ft_posy'].iloc[j + 1] - yo

#                 # Average fluorescence for the segment
#                 avg_FF = (smoothed_FF.iloc[j] + smoothed_FF.iloc[j + 1]) / 2

#                 # Color for the segment based on the average fluorescence
#                 color = cmap(norm(avg_FF))

#                 # Plot the segment
#                 axs[i].plot([x1, x2], [y1, y2], color=color, linewidth=3)

#             if hlines is not None:
#                 for j in range(len(hlines)):
#                     axs[i].hlines(y=hlines[j], xmin=-100, xmax=100, colors='k', linestyles='--', linewidth=1)
            
#             title = f'fly {fly_number}'

#             # Set axes, labels, and title
#             axs[i].set_FF(FF)
#             axs[i].set_ylim(ylim)
#             axs[i].set_xlabel('x position', fontsize=14)
#             axs[i].set_ylabel('y position', fontsize=14)
#             axs[i].set_title(f'{title} {lobe} lobe', fontsize=14)

#             # Further customization
#             axs[i].tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
#             for pos in ['right', 'top']:
#                 axs[i].spines[pos].set_visible(False)

#             for _, spine in axs[i].spines.items():
#                 spine.set_linewidth(2)
#             for spine in axs[i].spines.values():
#                 spine.set_edgecolor('black')

#         # Apply tight layout to the entire figure
#         fig.tight_layout()

#         # Save and show the plot
#         if save:
#             plt.suptitle(f'dF/F traj fly {fly_number}')
#             plt.savefig(os.path.join(figure_folder, f'FF_traj_{fly_number}_bw.pdf'))
#         else:
#             plt.show()

# def plot_speed_trajectory(figure_folder, filename, strip_width, strip_length, FF, ylim, hlines=[], save=False, keyword=None):
#     data = open_pickle(filename)
#     flylist_n = np.array(list(data.keys()))
#     if not os.path.exists(figure_folder):
#         os.makedirs(figure_folder)
#     # Determine flies to plot
#     if keyword is not None and 1 <= keyword <= len(flylist_n):
#         # Plotting for a specific fly
#         flies_to_plot = [(keyword, data[flylist_n[keyword - 1]])]
#     else:
#         # Plotting for all flies
#         flies_to_plot = [(fly_key, this_fly) for fly_key, this_fly in enumerate(data.values(), start=1)]
    
#     # Plot data
#     for fly_number, this_fly in flies_to_plot:
#         # Assign the correct dataframe ('a1' is all the data collected)
#         this_fly_all_data = this_fly['a1']
        
#         # Create a single figure
#         fig, axs = plt.subplots(1, 1, figsize=(8, 8))
#         plt.rcParams['font.family'] = 'sans-serif'
#         plt.rcParams['font.sans-serif'] = 'Arial'

#         # Filter df to exp start
#         first_on_index = this_fly_all_data[this_fly_all_data['instrip']].index[0]
#         exp_df = this_fly_all_data.loc[first_on_index:] # This filters the dataframe
#         # Establish the plume origin, at first odor onset
#         xo = exp_df.iloc[0]['ft_posx']
#         yo = exp_df.iloc[0]['ft_posy']
            
#         # Assign FF (fluorescence) to the correct df column
#         FF = exp_df['speed']
#         # Smooth fluorescence data
#         smoothed_FF = FF.rolling(window=10, min_periods=1).mean()

#         # Define color map
#         cmap = plt.get_cmap('coolwarm')

#         # Normalize FF to [0, 1] for colormap
#         min_FF = smoothed_FF.min()
#         max_FF = smoothed_FF.max()
#         range_FF = max_FF - min_FF
#         norm = colors_mod.Normalize(vmin=min_FF - 0.1 * range_FF, vmax=max_FF + 0.1 * range_FF)

#         # Plot the trajectory on the corresponding subplot
#         axs.scatter(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, c=smoothed_FF, cmap=cmap, norm=norm, s=5)

#         axs.add_patch(patches.Rectangle((-strip_width / 2, 0), strip_width, strip_length, facecolor='white', edgecolor='lightgrey', alpha=0.3))

#         if hlines is not None:
#             for j in range(len(hlines)):
#                 axs[i].hlines(y=hlines[j], xmin=-100, xmax=100, colors='k', linestyles='--', linewidth=1)
            
#         title = f'fly {fly_number}'

#         # Set axes, labels, and title
#         axs.set_FF(FF)
#         axs.set_ylim(ylim)
#         axs.set_xlabel('x position', fontsize=14)
#         axs.set_ylabel('y position', fontsize=14)
#         axs.set_title('speed', fontsize=14)

#         # Further customization
#         axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
#         for pos in ['right', 'top']:
#             axs.spines[pos].set_visible(False)

#         for _, spine in axs.spines.items():
#             spine.set_linewidth(2)
#         for spine in axs.spines.values():
#             spine.set_edgecolor('black')

#         # Apply tight layout to the entire figure
#         fig.tight_layout()

#         # Save and show the plot
#         if save:
#             plt.savefig(os.path.join(figure_folder, f'speed_traj_{fly_number}'))
#         else:
#             plt.show()


# def plot_triggered_norm_FF(figure_folder, filename, lobes, colors, window_size=5, event_type='entry'):
#     data = open_pickle(filename)
#     flylist_n = np.array(list(data.keys()))
#     fig, axs = plt.subplots(1, len(lobes), figsize=(4 * len(lobes), 5), sharex=True, sharey=True)
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = 'Arial'
#     for i, (lobe, color) in enumerate(zip(lobes, colors)):
#         for fly_n, fly_key in enumerate(flylist_n, start=1):
#             this_fly = data[fly_key]
#             this_fly_all_data = this_fly['a1']
#             FF = this_fly_all_data[lobe]
#             time = this_fly_all_data['relative_time']
#             # Choose the dataset based on event type
#             if event_type == 'entry':
#                 d_event = this_fly['di']
#             else:
#                 d_event = this_fly['do']
#             for key, df in d_event.items():
#                 time_on = df['relative_time'].iloc[0]
#                 # Extract a window around the event
#                 window_start = time_on - window_size / 2
#                 window_end = time_on + window_size / 2
#                 window_mask = (time >= window_start) & (time <= window_end)
#                 # Check if data points fall within the window
#                 if any(window_mask):
#                     # Z-score fluorecence
#                     normalized_FF = stats.zscore(FF[window_mask])
#                     time_aligned = time[window_mask] - time_on
#                     # Plot fluorescence aligned to the event time
#                     axs[i].plot(time_aligned, normalized_FF, color=color, alpha=0.1, linewidth=0.2)           
#         # Customize the plot
#         axs[i].set_title(lobe)
#         axs[i].set_ylim(-3, 8)
#         axs[i].set_ylabel('dF/F')
#         axs[i].set_xlabel('time (sec)')
#         axs[i].grid(False)
#         axs[i].vlines(x=0, ymin=-5, ymax=10, color='grey', alpha=0.5, linestyles='--')
#     plt.suptitle(f'normalized dF/F at {event_type}')
#     plt.show()
#     plt.savefig(os.path.join(figure_folder, f'norm_FF_{event_type}'))


