import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib import cm
import statsmodels.api as sm
import seaborn as sns
import os
import pickle
from scipy.stats import gaussian_kde
from scipy.stats import ttest_rel, zscore, pearsonr
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
    pv2 = img_dict['pv2'].copy()
    ft2 = img_dict['ft2'].copy()

    # consistent boolean
    ft2['instrip'] = ft2['instrip'].replace({1: True, 0: False})

    # OPTIONAL: if you really want instrip in pv2, align by position
    pv2 = pv2.reset_index(drop=True)
    ft2 = ft2.reset_index(drop=True)
    n = min(len(pv2), len(ft2))
    pv2['instrip'] = ft2.loc[:n-1, 'instrip'].to_numpy()

    # Add pv2-derived signals to ft2 under new names (DO NOT overwrite ft2 timebase)
    ft2[f'{neuron}_pv2'] = np.nan
    ft2['relative_time_pv2'] = np.nan
    ft2.loc[:n-1, f'{neuron}_pv2'] = pv2.loc[:n-1, f'0_{neuron}'].to_numpy()
    ft2.loc[:n-1, 'relative_time_pv2'] = pv2.loc[:n-1, 'relative_time'].to_numpy()

    first_time = ft2['timestamp'].iloc[0]
    ft2['seconds_processed'] = (ft2['timestamp']- first_time)
    del_t = np.mean(np.diff(ft2.seconds_processed))
    effective_rate = 1/del_t
    yv = np.gradient(ft2.ft_posy)*effective_rate
    ft2['y-vel'] = yv 
    xv = np.gradient(ft2.ft_posx)*effective_rate
    ft2['x-vel'] = xv
    xv = np.gradient(ft2.ft_posx, ft2.seconds)
    yv = np.gradient(ft2.ft_posy, ft2.seconds)
    speed = np.sqrt(xv**2+yv**2)
    ft2['speed'] = speed
    heading = ((ft2['ft_heading']+math.pi) % (2*math.pi))-math.pi
    ft2['transformed_heading'] = np.rad2deg(heading)

    return pv2, ft2

def process_pickle_old(img_dict, neuron):
    pv2 = img_dict['pv2']
    ft2 = img_dict['ft2']
    # print(ft2.columns)
    ft2['instrip'] = ft2['instrip'].replace({1: True, 0: False})
    pv2['instrip'] = ft2['instrip']
    ft2[neuron] = pv2[f'0_{neuron}']
    ft2['relative_time'] = pv2['relative_time']

    return pv2, ft2

def process_dLight_pickle(img_dict, rois):
    pv2 = img_dict['pv2']
    ft2 = img_dict['ft2']
    ft2['instrip'] = ft2['instrip'].replace({1: True, 0: False})
    pv2['instrip'] = ft2['instrip']
    for roi in rois:
        ft2[roi] = pv2[f'0_{roi}']
    ft2['relative_time'] = pv2['relative_time']
    return pv2, ft2

def process_CX_pickle(img_dict):
    pv2 = img_dict['pv2']
    ft2 = img_dict['ft2']
    ft2['instrip'] = ft2['instrip'].replace({1: True, 0: False})
    pv2['instrip'] = ft2['instrip']
    rois = [f'{i}_fsb_upper' for i in range(16)]
    for roi in rois:
        ft2[roi] = pv2[roi]
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
        x = np.array(x, copy=True)
        y = np.array(y, copy=True)
    return x, y

def rebaseline(pkl_folder, lobes, span=500):
    # Allow a single name or a list of names
    if isinstance(lobes, str):
        lobes = [lobes]

    lowess = sm.nonparametric.lowess

    for filename in os.listdir(pkl_folder):
        if not filename.endswith('.pkl'):
            continue

        savename, _ = os.path.splitext(filename)
        img_dict = open_pickle(f'{pkl_folder}/{filename}')
        pv2 = img_dict['pv2'].copy()
        ft2 = img_dict['ft2']

        # Same for every region in this file, so compute once
        ts = pv2['relative_time']
        frac = min(1.0, float(span) / np.max(ts))

        for lobe in lobes:
            col = f'0_{lobe}'
            if col not in pv2.columns:
                print(f"Warning: {col} not found in {filename}, skipping.")
                continue

            y = pv2[col].to_numpy()
            baseline = lowess(y, np.arange(len(y)), frac=frac)[:, 1]
            pv2[col] = y - baseline

        img_dict = {'pv2': pv2, 'ft2': ft2}
        pickle_obj(img_dict, f'{pkl_folder}/rb', f'{savename}_rb.pkl')

def shade_on_intervals(ax, time, odor_on, mode="dash", alpha=None, color="white",
                        linewidth=4, y=0.95, ymin=0, ymax=1):
    """
    mode="span": full-height shaded region (axvspan)
    mode="dash": short dash mark at the top of the axes (hlines)
    """
    odor_on = np.asarray(odor_on).astype(bool)
    time = np.asarray(time)
    edges = np.flatnonzero(np.diff(odor_on.astype(int)) != 0) + 1
    starts = np.r_[0, edges][odor_on[np.r_[0, edges]]]
    ends   = np.r_[edges, len(odor_on)][odor_on[np.r_[0, edges]]]

    if mode == "span":
        a = 0.2 if alpha is None else alpha
        for s, e in zip(starts, ends):
            ax.axvspan(time[s], time[e - 1], ymin=ymin, ymax=ymax,
                       alpha=a, color=color, lw=0)
    elif mode == "dash":
        a = 1.0 if alpha is None else alpha
        trans = ax.get_xaxis_transform()
        for s, e in zip(starts, ends):
            ax.hlines(y, time[s], time[e - 1], transform=trans,
                       color=color, linewidth=linewidth, alpha=a,
                       capstyle='butt', zorder=5, clip_on=False)
    else:
        raise ValueError("mode must be 'span' or 'dash'")

def bout_duration(td, time, idx, direction):
    """Duration (in time units) that td stays constant starting at idx,
    walking backward (direction=-1) or forward (direction=+1)."""
    state = td[idx]
    j = idx
    if direction == -1:
        while j - 1 >= 0 and td[j - 1] == state:
            j -= 1
        return time[idx] - time[j]
    else:
        while j + 1 < len(td) and td[j + 1] == state:
            j += 1
        return time[j] - time[idx]

def sliding_pearson(x, y, time, window_s, step_s):
    dt = np.median(np.diff(time))
    win = int(window_s / dt)
    step = int(step_s / dt)
    r_vals = []
    t_centers = []
    for start in range(0, len(x) - win, step):
        stop = start + win
        xx = x[start:stop]
        yy = y[start:stop]
        mask = np.isfinite(xx) & np.isfinite(yy)
        if mask.sum() < win * 0.8:
            r = np.nan
        else:
            r = pearsonr(xx[mask], yy[mask])[0]
        r_vals.append(r)
        t_centers.append(time[start + win // 2])
    return np.array(t_centers), np.array(r_vals)


def circ_mean(angles):
    angles = np.asarray(angles)
    angles = angles[~np.isnan(angles)]
    if len(angles) == 0:
        return np.nan
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

def angdiff(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def label_plume_entries(ft2,
                        plume_angles_deg=(25, -25),
                        tol_deg=45,
                        instrip_col='instrip',
                        heading_col='ft_heading'):
    """
    Adds boolean columns 'instrip_1', 'instrip_2', and 'instrip_ambiguous'
    to ft2, splitting `instrip` epochs based on the fly's mean heading
    during each entry.

    'instrip_1'         = the plume the fly tracked first
    'instrip_2'         = the other plume
    'instrip_ambiguous' = entries whose mean heading is far from both plumes
                          (e.g. searching/transition entries)
    """
    instrip = ft2[instrip_col].to_numpy().astype(float)
    heading = ft2[heading_col].to_numpy()
    n = len(instrip)

    sdiff = np.diff(instrip)
    son  = np.where(sdiff > 0)[0] + 1
    soff = np.where(sdiff < 0)[0] + 1
    if instrip[0] > 0:
        son = np.insert(son, 0, 0)
    if instrip[-1] > 0:
        soff = np.append(soff, n - 1)

    plume_angles = [np.deg2rad(a) for a in plume_angles_deg]
    tol = np.deg2rad(tol_deg)

    entry_labels = []
    for s, e in zip(son, soff):
        mh = circ_mean(heading[s:e])
        if np.isnan(mh):
            entry_labels.append(None)
            continue
        dists = [abs(angdiff(mh, p)) for p in plume_angles]
        nearest = int(np.argmin(dists))
        if dists[nearest] < tol:
            entry_labels.append(nearest)
        else:
            entry_labels.append(None)

    first_plume = next((lbl for lbl in entry_labels if lbl is not None), None)

    instrip_1 = np.zeros(n, dtype=bool)
    instrip_2 = np.zeros(n, dtype=bool)
    instrip_ambig = np.zeros(n, dtype=bool)

    for (s, e), lbl in zip(zip(son, soff), entry_labels):
        if lbl is None:
            instrip_ambig[s:e] = True
        elif first_plume is not None and lbl == first_plume:
            instrip_1[s:e] = True
        else:
            instrip_2[s:e] = True

    ft2 = ft2.copy()
    ft2['instrip_1'] = instrip_1
    ft2['instrip_2'] = instrip_2
    ft2['instrip_ambiguous'] = instrip_ambig
    return ft2

def trace_FF(figure_folder, rois, window=None, shade_mode="dash"):
    n_rois = len(rois)
    out_folder = os.path.join(figure_folder, "fig", "trace_FF")
    os.makedirs(out_folder, exist_ok=True)

    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue
        savename, _ = os.path.splitext(filename)
        fullpath = os.path.join(figure_folder, filename)
        img_dict = open_pickle(fullpath)
        pv2, ft2 = process_dLight_pickle(img_dict, rois)

        fig, axs, markc = configure_bw_plot(size=(15, 3 * n_rois), xaxis=True, nrows=n_rois)
        if n_rois == 1:
            axs = np.array([axs])

        time = pv2['relative_time'].to_numpy()
        odor1_on = ft2['mfc2_stpt'].to_numpy()
        odor2_on = ft2['mfc3_stpt'].to_numpy()

        for r_idx, roi in enumerate(rois):
            ax = axs[r_idx]
            FF = pv2[f'0_{roi}'].to_numpy()

            n = min(len(time), len(FF), len(odor1_on), len(odor2_on))
            t = time[:n]
            y = FF[:n]
            o1 = odor1_on[:n]
            o2 = odor2_on[:n]

            ax.plot(t, y, color=markc, linewidth=0.75)

            color1 = '#d473d4' if 'mch' in filename else '#fbceb1'
            shade_on_intervals(ax, t, o1, mode=shade_mode, alpha=0.4, color=color1)
            shade_on_intervals(ax, t, o2, mode=shade_mode, alpha=0.4, color='#7ed4e6')

            ax.set_title(roi, color=markc, size=14)
            ax.set_ylabel(r'$\Delta$F/F', color=markc, size=12)

            if window is not None:
                ax.set_xlim(window)

        axs[-1].set_xlabel('time (s)', color=markc, size=12)
        fig.suptitle(filename, color=markc, size=16)

        fname = f"{savename}_{'_'.join(rois)}_trace.pdf"
        fig.savefig(os.path.join(out_folder, fname), bbox_inches="tight")
        plt.show()
        plt.close(fig)

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
        baseline_mask = (time >= time_on - 1) & (time < time_on)
        interval_mask = (time >= time_on) & (time <= time_on + 2)
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
            FF_et = ft2_et[lobe].to_numpy()
            FF_replay = ft2_replay[lobe].to_numpy()
            ymin = min(FF_et.min(), FF_replay.min())
            ymax = max(FF_et.max(), FF_replay.max())
            # ET plot
            ax = axs[row_idx][0]
            ax.plot(pv2_et['relative_time'].to_numpy(), FF_et, linewidth=0.5, color=colors[row_idx % len(colors)])
            # add_odor(ax, di_et, ymin, ymax)
            ax.set_ylim([ymin, ymax])  
            ax.set_ylabel(lobe, color=markc)
            # Replay plot
            ax = axs[row_idx][1]
            ax.plot(pv2_replay['relative_time'].to_numpy(), FF_replay, linewidth=0.5, color=colors[row_idx % len(colors)])
            # add_odor(ax, di_replay, ymin, ymax)
            ax.set_ylim([ymin, ymax])
        
        axs[0, 0].set_title('edge tracking', color=markc)
        axs[0, 1].set_title('replay', color=markc)    

        # add net motion
        row_idx = len(lobes)
        FF_et = pv2_et['net_motion'].to_numpy()
        FF_replay = pv2_replay['net_motion'].to_numpy()
        ymin = min(FF_et.min(), FF_replay.min())
        ymax = max(FF_et.max(), FF_replay.max())

        ax = axs[row_idx][0]
        ax.plot(pv2_et['relative_time'].to_numpy(), FF_et, linewidth=0.5, color=markc)
        # add_odor(ax, di_et, ymin, ymax)
        ax.set_ylim([ymin, ymax])
        ax.set_ylabel('net motion', color=markc)

        ax = axs[row_idx][1]
        ax.plot(pv2_replay['relative_time'].to_numpy(), FF_replay, linewidth=0.5, color=markc)
        # add_odor(ax, di_replay, ymin, ymax)
        ax.set_ylim([ymin, ymax])
        plt.tight_layout()
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/traces'):
            os.makedirs(f'{figure_folder}/fig/traces')
        fig.savefig(f'{figure_folder}/fig/traces/{fly_id}_et_vs_replay.pdf', bbox_inches='tight')


def DAN_corr(figure_folder, rois):
    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue
        file_path = os.path.join(figure_folder, filename)
        img_dict = open_pickle(file_path)
        pv2, ft2 = process_dLight_pickle(img_dict, rois)

        relative_time = ft2["relative_time"].to_numpy()
        odor_mfc2 = ft2["mfc2_stpt"].to_numpy()
        odor_mfc3 = ft2["mfc3_stpt"].to_numpy()

        if 'acv' in filename:
            is_odor_on = odor_mfc2.astype(bool)
        elif 'oct' in filename:
            is_odor_on = odor_mfc3.astype(bool)

        in_odor_bouts = odor_bout_indices(is_odor_on)
        out_of_odor_bouts = exit_bout_indices(is_odor_on)

        out_bout_by_end_idx = {
            out_end_idx: (out_start_idx, out_end_idx)
            for out_start_idx, out_end_idx in out_of_odor_bouts
        }

        n_rois = len(rois)
        fig, axs, markc = configure_bw_plot(size=(4 * n_rois, 5), xaxis=True, ncols=n_rois)

        for r_idx, roi in enumerate(rois):
            ax = axs[r_idx]
            x_outside_sec = []
            y_peak_ff = []
            ff_trace = pv2[f'0_{roi}'].to_numpy()

            for in_start_idx, in_end_idx in in_odor_bouts:
                previous_out_bout = out_bout_by_end_idx.get(in_start_idx)
                if previous_out_bout is None:
                    continue
                out_start_idx, out_end_idx = previous_out_bout
                dur = relative_time[out_end_idx - 1] - relative_time[out_start_idx]
                in_bout_ff = ff_trace[in_start_idx:in_end_idx]
                peak_ff = float(np.nanmax(in_bout_ff))
                x_outside_sec.append(dur)
                y_peak_ff.append(peak_ff)

                out_start_idx, out_end_idx = previous_out_bout
                dur = relative_time[out_end_idx - 1] - relative_time[out_start_idx]
                in_bout_ff = ff_trace[in_start_idx:in_end_idx]
                peak_ff = float(np.nanmax(in_bout_ff))
                x_outside_sec.append(dur)
                y_peak_ff.append(peak_ff)

            ax.scatter(x_outside_sec, y_peak_ff, s=20, alpha=0.8)
            ax.set_title(str(roi), color=markc, fontsize=10)
            ax.set_xlabel("time since last entry (s)", fontsize=12)
            ax.set_ylabel("peak dF/F", fontsize=12)

            if len(x_outside_sec) >= 2:
                x = np.asarray(x_outside_sec, dtype=float)
                y = np.asarray(y_peak_ff, dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)

                if np.count_nonzero(valid) >= 2:
                    slope, intercept = np.polyfit(x[valid], y[valid], 1)
                    x_fit = np.linspace(x[valid].min(), x[valid].max(), 100)
                    y_fit = slope * x_fit + intercept
                    ax.plot(x_fit, y_fit, linewidth=2)
                    y_pred = slope * x[valid] + intercept
                    ss_res = np.sum((y[valid] - y_pred) ** 2)
                    ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    ax.text(0.05, 0.95,f"slope = {slope:.3g}\nR² = {r2:.3f}\n(n={np.count_nonzero(valid)})",transform=ax.transAxes,va="top")
            
        fig.suptitle(filename)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/corr'):
            os.makedirs(f'{figure_folder}/fig/corr')

        fig.savefig(f'{figure_folder}/fig/corr/{filename}_time.pdf', bbox_inches='tight')
        plt.close(fig)



def et_replay_traces_bw_new(figure_folder, lobes, colors, size, xlim):
    paired_files = {}
    for fn in os.listdir(figure_folder):
        if fn.endswith(".pkl"):
            if "_et_" in fn:
                paired_files.setdefault(fn.split("_et_")[0], {})["et"] = fn
            elif "_replay_" in fn:
                paired_files.setdefault(fn.split("_replay_")[0], {})["replay"] = fn

    outdir = os.path.join(figure_folder, "fig", "traces")
    os.makedirs(outdir, exist_ok=True)

    for fly_id, files in paired_files.items():
        if "et" not in files or "replay" not in files:
            print(f"Skipping {fly_id}: missing pair")
            continue

        data_et = open_pickle(os.path.join(figure_folder, files["et"]))
        data_replay = open_pickle(os.path.join(figure_folder, files["replay"]))
        pv2_et, ft2_et = process_dLight_pickle(data_et, lobes)
        pv2_replay, ft2_replay = process_dLight_pickle(data_replay, lobes)

        nrows, ncols = len(lobes) + 1, 2
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True, nrows=nrows, ncols=ncols)
        axs = np.array(axs).reshape(nrows, ncols)
        fig.suptitle(fly_id)

        odor1_et = ft2_et["mfc3_stpt"].to_numpy()
        odor2_et = ft2_et["mfc4_stpt"].to_numpy()
        odor1_replay = ft2_replay["mfc3_stpt"].to_numpy()
        odor2_replay = ft2_replay["mfc4_stpt"].to_numpy()

        for row_idx, lobe in enumerate(lobes):
            time_et = pv2_et["relative_time"].to_numpy()
            time_replay = pv2_replay["relative_time"].to_numpy()

            FF_et = pv2_et[f'0_{lobe}'].to_numpy()
            FF_replay = pv2_replay[f'0_{lobe}'].to_numpy()

            ymin, ymax = finite_minmax(FF_et, FF_replay)

            # ET
            ax = axs[row_idx, 0]
            ax.plot(time_et, FF_et, linewidth=0.5, color=colors[row_idx % len(colors)])
            shade_on_intervals(ax, time_et, odor1_et, mode='span', alpha=0.6, color="#fbceb1")
            shade_on_intervals(ax, time_et, odor2_et, mode='span', alpha=0.6, color="#7ed4e6")
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(lobe, color=markc)
            if xlim is not None:
                ax.set_xlim(xlim)

            # Replay
            ax = axs[row_idx, 1]
            ax.plot(time_replay, FF_replay, linewidth=0.5, color=colors[row_idx % len(colors)])
            shade_on_intervals(ax, time_replay, odor1_replay, mode='span', alpha=0.6, color="#fbceb1", linewidth=4, y=0.95)
            shade_on_intervals(ax, time_replay, odor2_replay,mode='span', alpha=0.6, color="#7ed4e6", linewidth=4, y=0.95)
            ax.set_ylim(ymin, ymax)
            if xlim is not None:
                ax.set_xlim(xlim)

        axs[0, 0].set_title("edge tracking", color=markc)
        axs[0, 1].set_title("replay", color=markc)

        row_idx = len(lobes)
        time_et = ft2_et["relative_time"].to_numpy()
        time_replay = ft2_replay["relative_time"].to_numpy()
        nm_et = ft2_et["net_motion"].to_numpy()
        nm_replay = ft2_replay["net_motion"].to_numpy()
        ymin, ymax = finite_minmax(nm_et, nm_replay)

        ax = axs[row_idx, 0]
        ax.plot(time_et, nm_et, linewidth=0.5, color=markc)
        shade_on_intervals(ax, time_et, odor1_et, mode='span', alpha=0.6, color="#fbceb1")
        shade_on_intervals(ax, time_et, odor2_et, mode='span', alpha=0.6, color="#7ed4e6")
        ax.set_ylim(ymin, ymax)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_ylabel("net motion", color=markc)
        ax = axs[row_idx, 1]
        ax.plot(time_replay, nm_replay, linewidth=0.5, color=markc)
        shade_on_intervals(ax, time_replay, odor1_replay, mode='span', alpha=0.6, color="#fbceb1")
        shade_on_intervals(ax, time_replay, odor2_replay, mode='span', alpha=0.6, color="#7ed4e6")
        ax.set_ylim(ymin, ymax)
        ax1 = axs[row_idx, 1]
        if xlim is not None:
            ax1.set_xlim(xlim)

        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"{fly_id}_et_vs_replay_{xlim}.pdf"), bbox_inches="tight")
        plt.show()        


def et_replay_traj_bw(figure_folder, lobes, colors, size):
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
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=2)
        fig.suptitle(f"{fly_id} et vs replay", fontsize=14)
        data_et = open_pickle(os.path.join(figure_folder, files['et']))
        data_replay = open_pickle(os.path.join(figure_folder, files['replay']))
        df_et = data_et['strip1']['pv2']
        df_replay = data_replay['strip1']['pv2']

        # Shared parameters
        strip_width = 10
        # compute height from data
        yf = max(df_et['ft_posy'].max(), df_replay['ft_posy'].max())
        yo = min(df_et['ft_posy'].min(), df_replay['ft_posy'].min())
        
        axs[0].add_patch(patches.Rectangle((-strip_width / 2, 0), 10, (yf - yo),
                                        facecolor='grey', alpha=0.4))
        axs[0].add_patch(patches.Rectangle(((-strip_width / 2) - 210, 0), 10, (yf - yo),
                                        facecolor='grey', alpha=0.4))
        axs[0].add_patch(patches.Rectangle(((-strip_width / 2) + 210, 0), 10, (yf - yo),
                                           facecolor='grey', alpha=0.4))


        # Loop through the two datasets and plot them in two columns
        for ax, (label, df) in zip(axs, [('ET', df_et), ('Replay', df_replay)]):
            xo = df.iloc[0]['ft_posx']
            yo = df.iloc[0]['ft_posy']

            ax.plot(df['ft_posx'] - xo, df['ft_posy'] - yo, color='lightgrey', label='clean air')
            if 'mfc2_stpt' in df.columns:
                df_odor = df.where(df.mfc2_stpt > 0)
                ax.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo,
                        color='red', label='odor only')

            ax.set_title(label, fontsize=13)
            ax.set_xlabel('x-position (mm)', fontsize=12, color='white')
            ax.set_ylabel('y-position (mm)', fontsize=12, color='white')
            ax.tick_params(which='both', axis='both', labelsize=10, length=3,
                           width=2, color='black', direction='out', left=True, bottom=True)
            for pos in ['right', 'top']:
                ax.spines[pos].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_edgecolor(markc)
            ax.set_aspect('equal', adjustable='datalim')
            ax.legend(frameon=False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/traj'):
            os.makedirs(f'{figure_folder}/fig/traj')
        fig.savefig(f'{figure_folder}/fig/traj/{fly_id}_et_vs_replay.pdf', bbox_inches='tight')


def dLight_traj_bw(figure_folder, lobes, colors, size):
    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue
        savename, _ = os.path.splitext(filename)
        fullpath = os.path.join(figure_folder, filename)
        img_dict = open_pickle(fullpath)
        pv2, ft2 = process_dLight_pickle(img_dict, lobes)
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True)

        mask = ((ft2['mfc3_stpt'] > 0) | (ft2['mfc4_stpt'] > 0)).to_numpy()
        if mask.any():
            start_idx = np.argmax(mask)  # index of first True
            ft2 = ft2.iloc[start_idx:].reset_index(drop=True)
        x = ft2['ft_posx'].to_numpy()
        y = ft2['ft_posy'].to_numpy()
        x, y = fictrac_repair(x, y)
        yf = y.max()
        yo = y.min()
        xo = x[0]
        yo = y[0]

        df_acv = ft2.where(ft2.mfc3_stpt > 0)
        df_oct = ft2.where(ft2.mfc4_stpt > 0)

        x_acv = df_acv['ft_posx'].to_numpy()
        y_acv = df_acv['ft_posy'].to_numpy()
        x_acv, y_acv = fictrac_repair(x_acv, y_acv)

        x_oct = df_oct['ft_posx'].to_numpy()
        y_oct = df_oct['ft_posy'].to_numpy()
        x_oct, y_oct = fictrac_repair(x_oct, y_oct)
        
        strip_width = 10
        if 'et' in filename:
            axs.add_patch(patches.Rectangle((-strip_width / 2, 0), 10, (yf - yo), facecolor='grey', alpha=0.4))
            axs.add_patch(patches.Rectangle(((-strip_width / 2) - 210, 0), 10, (yf - yo), facecolor='grey', alpha=0.4))
            axs.add_patch(patches.Rectangle(((-strip_width / 2) + 210, 0), 10, (yf - yo), facecolor='grey', alpha=0.4))
    
        axs.plot(x - xo, y - yo, color='lightgrey', label='clean air')
        axs.plot(x_acv - xo, y_acv - yo,color='red', label='acv')
        axs.plot(x_oct - xo, y_oct - yo,color='red', label='oct')
        axs.set_xlabel('x-position (mm)', fontsize=12, color=markc)
        axs.set_ylabel('y-position (mm)', fontsize=12, color=markc)
        axs.tick_params(which='both', axis='both', labelsize=10, length=3,
                        width=2, color=markc, direction='out', left=True, bottom=True)

        for pos in ['right', 'top']:
            axs.spines[pos].set_visible(False)
        for spine in axs.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor(markc)
        axs.set_aspect('equal', adjustable='datalim')
        # axs.legend(frameon=False)
        plt.title(filename, fontsize=14)
        plt.tight_layout()
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/traj'):
            os.makedirs(f'{figure_folder}/fig/traj')
        fig.savefig(f'{figure_folder}/fig/traj/{filename}_traj.pdf', bbox_inches='tight')


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


def et_replay_peak_comp_afs(figure_folder, lobes, colors, size):
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

def peaks_per_bout_delta(time, FF, odor_on, baseline_window_s=1.0, baseline_fn=np.nanmedian):
    """
    For each contiguous odor_on bout:
      delta = max(FF in bout) - baseline
    baseline is computed from a window just before bout start.
    """
    time = np.asarray(time, float)
    FF = np.asarray(FF, float)
    odor_on = np.asarray(odor_on).astype(bool)

    bouts = odor_bout_indices(odor_on)
    if len(time) >= 2:
        dt = np.nanmedian(np.diff(time))
    else:
        dt = np.nan

    # convert seconds to samples
    if np.isfinite(dt) and dt > 0:
        n_base = max(1, int(round(baseline_window_s / dt)))
    else:
        n_base = 1

    deltas = []
    for s, e in bouts:
        seg = FF[s:e]
        if not np.isfinite(seg).any():
            deltas.append(np.nan)
            continue

        # baseline window immediately before bout
        bs0 = max(0, s - n_base)
        base_seg = FF[bs0:s]
        if np.isfinite(base_seg).any():
            baseline = float(baseline_fn(base_seg))
        else:
            baseline = float(baseline_fn(seg))  # fallback if no pre-bout baseline

        peak = float(np.nanmax(seg))
        deltas.append(peak - baseline)

    return deltas

def peaks_per_bout_z(time, FF, odor_on, baseline_window_s=1.0,
                         baseline_center_fn=np.nanmedian):
    """
    For each contiguous odor_on bout:
      z_peak = (max(FF in bout) - baseline_center) / baseline_std

    baseline is computed from a window just before bout start (same as peaks_per_bout_delta).
    """
    time = np.asarray(time, float)
    FF = np.asarray(FF, float)
    odor_on = np.asarray(odor_on).astype(bool)

    bouts = odor_bout_indices(odor_on)
    if len(time) >= 2:
        dt = np.nanmedian(np.diff(time))
    else:
        dt = np.nan

    if np.isfinite(dt) and dt > 0:
        n_base = max(1, int(round(baseline_window_s / dt)))
    else:
        n_base = 1

    z_peaks = []
    for s, e in bouts:
        seg = FF[s:e]
        if not np.isfinite(seg).any():
            z_peaks.append(np.nan)
            continue

        bs0 = max(0, s - n_base)
        base_seg = FF[bs0:s]

        if np.isfinite(base_seg).any():
            base_center = float(baseline_center_fn(base_seg))
            mad = float(np.nanmedian(np.abs(base_seg - base_center)))
            base_std = 1.4826 * mad
        else:
            # fallback if no pre-bout baseline
            base_center = float(baseline_center_fn(seg))
            mad = float(np.nanmedian(np.abs(seg - base_center)))
            base_std = 1.4826 * mad

        peak = float(np.nanmax(seg))

        if not np.isfinite(base_std) or base_std == 0:
            z_peaks.append(np.nan)
        else:
            z_peaks.append((peak - base_center) / base_std)

    return z_peaks



def detect_events(odor_on, event_type):
    odor_on = (np.asarray(odor_on) != 0).astype(int)
    d = np.diff(odor_on)
    if event_type == "entry":
        return np.where(d == 1)[0] + 1
    elif event_type == "exit":
        return np.where(d == -1)[0] + 1
    else:
        raise ValueError("event_type must be 'entry' or 'exit'")

def odor_bout_indices(odor_on):
    """Return list of (start_idx, end_idx) for contiguous True runs. end_idx is exclusive."""
    odor_on = np.asarray(odor_on).astype(bool)
    if odor_on.size == 0:
        return []

    edges = np.flatnonzero(np.diff(odor_on.astype(int)) != 0) + 1
    starts = np.r_[0, edges]
    ends   = np.r_[edges, odor_on.size]

    on = odor_on[starts]
    starts, ends = starts[on], ends[on]
    return list(zip(starts, ends))

def exit_bout_indices(odor_on):
    """Return list of (start_idx, end_idx) for contiguous False runs. end_idx is exclusive."""
    odor_on = np.asarray(odor_on).astype(bool)
    if odor_on.size == 0:
        return []

    edges = np.flatnonzero(np.diff(odor_on.astype(int)) != 0) + 1
    starts = np.r_[0, edges]
    ends   = np.r_[edges, odor_on.size]

    off = ~odor_on[starts]
    starts, ends = starts[off], ends[off]
    return list(zip(starts, ends))

def et_replay_oct_peak_comp(figure_folder, lobes, colors, size, zscore=True):
    paired_files = {}
    for filename in os.listdir(figure_folder):
        if filename.endswith(".pkl"):
            if "_et_" in filename:
                prefix = filename.split("_et_")[0]
                paired_files.setdefault(prefix, {})["et"] = filename
            elif "_replay_" in filename:
                prefix = filename.split("_replay_")[0]
                paired_files.setdefault(prefix, {})["replay"] = filename

    for fly_id, files in paired_files.items():
        if "et" not in files or "replay" not in files:
            print(f"Skipping {fly_id}: missing pair")
            continue

        nrows = len(lobes)
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True, nrows=nrows)
        axs = np.ravel(axs)
        fig.suptitle(fly_id)

        data_et = open_pickle(os.path.join(figure_folder, files["et"]))
        data_replay = open_pickle(os.path.join(figure_folder, files["replay"]))
        pv2_et, ft2_et = process_dLight_pickle(data_et, lobes)
        pv2_replay, ft2_replay = process_dLight_pickle(data_replay, lobes)

        odor_et = ft2_et["mfc3_stpt"].to_numpy()
        odor_replay = ft2_replay["mfc4_stpt"].to_numpy()
        time_et = ft2_et["relative_time"].to_numpy()
        time_replay = ft2_replay["relative_time"].to_numpy()
        
        peaks_et = {f"G{i+2}": [] for i in range(len(lobes))}
        peaks_replay = {f"G{i+2}": [] for i in range(len(lobes))}

        for i, lobe in enumerate(lobes):
            FF_et = pv2_et[f'0_{lobe}'].to_numpy()
            FF_replay = pv2_replay[f'0_{lobe}'].to_numpy()
            if zscore:
                peaks_et[f"G{i+2}"] = peaks_per_bout_z(time_et, FF_et, odor_et, baseline_window_s=1.0)
                peaks_replay[f"G{i+2}"] = peaks_per_bout_z(time_replay, FF_replay, odor_replay, baseline_window_s=1.0)
            else:
                peaks_et[f"G{i+2}"] = peaks_per_bout_delta(time_et, FF_et, odor_et, baseline_window_s=1.0)
                peaks_replay[f"G{i+2}"] = peaks_per_bout_delta(time_replay, FF_replay, odor_replay, baseline_window_s=1.0)
            entry_number_et = range(1, len(peaks_et[f"G{i+2}"]) + 1)
            entry_number_replay = range(1, len(peaks_replay[f"G{i+2}"]) + 1)

            axs[i].plot(entry_number_et, peaks_et[f"G{i+2}"], color=colors[i % len(colors)], alpha=0.8)
            axs[i].plot(entry_number_replay, peaks_replay[f"G{i+2}"], color=colors[i % len(colors)], alpha=0.3)
            axs[i].set_ylabel(lobe, color=markc)

        plt.tight_layout()
        plt.show()

        outdir = os.path.join(figure_folder, "fig", "peaks")
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"{fly_id}_peaks_per_entry.pdf"), bbox_inches="tight")


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
            if thresh is None:
                thresh = len(di)  # Use all bouts if thresh is not specified
            for i, (key, df) in enumerate(di.items()):
                if i >= thresh:
                    break  # Stop after the specified number of bouts
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                start_idx = np.searchsorted(time, time_on - pre_post_time)
                end_idx = np.searchsorted(time, time_off + pre_post_time)
                epoch_time = time[start_idx:end_idx] - time[start_idx] + current_x
                concatenated_FF.extend(epoch_FF)
                concatenated_time.extend(epoch_time)
                concatenated_FF.append(np.nan)
                concatenated_time.append(np.nan)
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
            FF = pv2[f'0_{neuron}'].to_numpy()
            time = pv2['relative_time'].to_numpy()
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

def triggered_zFF(figure_folder, neuron, color, tbef=30, taf=30, event_type='entry', first=True):
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
            FF = pv2[f'0_{neuron}'].to_numpy()
            time = pv2['relative_time'].to_numpy()
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


def triggered_zFF_PAM(figure_folder,neuron,color,tbef=30,taf=30,event_type='entry',first=False,zscore_mode="baseline"):
    bin_names = [
        "oct pulses 1–5 (odor)",
        "oct pulses 6–10 (odor + LED)",
        "oct pulses 11–15 (odor)",
        "acv pulses 1–5 (odor)",
        "acv pulses 6–10 (odor + LED)",
        "acv pulses 11–15 (odor)",
    ]

    out_folder = f'{figure_folder}/fig/triggered_zFF_per_file'
    os.makedirs(out_folder, exist_ok=True)

    # --- NEW: collect per-file mean traces (one per bin) for summary ---
    # each element: list of arrays length seg_len (or np.nan if missing)
    perfile_bin_means = [[] for _ in range(6)]

    n_files_used = 0

    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith('.pkl'):
            continue
        print(filename)

        savename, _ = os.path.splitext(filename)
        img_dict = open_pickle(f'{figure_folder}/{filename}')
        pv2, ft2 = process_pickle(img_dict, neuron)
        if pv2 is None or ft2 is None:
            print(f"Skipping file {filename} due to missing data.")
            continue

        FF = pv2[f'0_{neuron}'].to_numpy()
        time = pv2['relative_time'].to_numpy()

        # events
        odor2_on = (ft2['mfc3_stpt'].to_numpy() != 0).astype(int)
        d2 = np.diff(odor2_on)
        if event_type == 'entry':
            son2 = np.where(d2 == 1)[0] + 1
        elif event_type == 'exit':
            son2 = np.where(d2 == -1)[0] + 1
        else:
            raise ValueError("event_type must be 'entry' or 'exit'")

        odor1_on = (ft2['mfc2_stpt'].to_numpy() != 0).astype(int)
        d1 = np.diff(odor1_on)
        if event_type == 'entry':
            son1 = np.where(d1 == 1)[0] + 1
        else:
            son1 = np.where(d1 == -1)[0] + 1

        if len(son2) == 0 and len(son1) == 0:
            print(f"No {event_type} events found in file: {filename}")
            continue

        son2 = son2[:15]
        son1 = son1[:15]
        if first:
            son2 = son2[:1]
            son1 = son1[:1]

        son2_b1, son2_b2, son2_b3 = son2[0:5], son2[5:10], son2[10:15]
        son1_b1, son1_b2, son1_b3 = son1[0:5], son1[5:10], son1[10:15]
        son_bins = [son2_b1, son2_b2, son2_b3, son1_b1, son1_b2, son1_b3]

        # timebase for this file
        tinc = np.mean(np.diff(time))
        idx_bef = int(np.round(float(tbef) / tinc))
        idx_af = int(np.round(float(taf) / tinc))
        seg_len = idx_bef + idx_af + 1
        t = np.linspace(-tbef, taf, seg_len)

        # plot per file
        fig, axs, markc = configure_bw_plot(size=(10, 5), xaxis=True, nrows=2, ncols=3)
        row_ymins = [np.inf, np.inf]
        row_ymaxs = [-np.inf, -np.inf]


        file_had_any_bin = False  # track whether this file contributes to summary at all

        for b, son in enumerate(son_bins):
            ax = axs[b]

            if len(son) == 0:
                ax.set_title(bin_names[b], color=markc, size=12)
                ax.text(0.5, 0.5, "No events", color=markc,
                        ha='center', va='center', transform=ax.transAxes)
                ax.axvline(0, color=markc, linestyle='--', linewidth=1)
                # --- NEW: store NaNs for this bin for this file so lengths stay consistent ---
                perfile_bin_means[b].append(np.full(seg_len, np.nan))
                continue

            mn_mat = np.full((len(son), seg_len), np.nan)
            for i, s in enumerate(son):
                idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
                valid = (idx_array >= 0) & (idx_array < len(FF))
                idx_array = idx_array[valid]
                segment = FF[idx_array]
                mn_mat[i, :len(segment)] = segment

            # normalization
            if zscore_mode in ("none", "raw", None):
                pass

            elif zscore_mode == "global":
                mn_mat = stats.zscore(mn_mat, axis=None, nan_policy='omit')

            elif zscore_mode == "trial":
                mn_mat = stats.zscore(mn_mat, axis=1, nan_policy='omit')

            elif zscore_mode == "baseline":
                baseline = mn_mat[:, :idx_bef]
                mu = np.nanmean(baseline, axis=1, keepdims=True)
                sd = np.nanstd(baseline, axis=1, keepdims=True)

                # avoid divide-by-zero -> keep those trials as NaN
                sd[sd == 0] = np.nan
                mn_mat = (mn_mat - mu) / sd

            else:
                raise ValueError("zscore_mode must be 'none'/'raw', 'global', 'trial', or 'baseline'")
            for r in (0, 1):
                if np.isfinite(row_ymins[r]) and np.isfinite(row_ymaxs[r]):
                    y0, y1 = row_ymins[r], row_ymaxs[r]
                    if y0 == y1:
                        pad = 1.0
                    else:
                        pad = 0.05 * (y1 - y0)
                    y0 -= pad
                    y1 += pad
                    for j in range(r * 3, r * 3 + 3):
                        axs[j].set_ylim(y0, y1)
            masked = np.ma.masked_array(mn_mat, np.isnan(mn_mat))
            plt_mn = np.ma.mean(masked, axis=0)
            plt_sd = np.ma.std(masked, axis=0)
            y_lo = np.nanmin((plt_mn - plt_sd).filled(np.nan))
            y_hi = np.nanmax((plt_mn + plt_sd).filled(np.nan))
            if np.isfinite(y_lo) and np.isfinite(y_hi):
                r = 0 if b < 3 else 1
                row_ymins[r] = min(row_ymins[r], y_lo)
                row_ymaxs[r] = max(row_ymaxs[r], y_hi)

            ax.fill_between(t, plt_mn - plt_sd, plt_mn + plt_sd, color=color, alpha=0.3)
            ax.plot(t, plt_mn, color=color, linewidth=1.5)
            ax.axvline(0, color=markc, linestyle='--', linewidth=1)
            ax.set_title(bin_names[b], color=markc, size=12)

            # --- NEW: store per-file mean trace for summary ---
            perfile_bin_means[b].append(np.array(plt_mn.filled(np.nan)))
            file_had_any_bin = True

        axs[3].set_xlabel('time (s)', size=14, color=markc)
        if zscore_mode in ("none", "raw", None):
            axs[0].set_ylabel('dF/F')
            axs[3].set_ylabel('dF/F')
        else:
            axs[0].set_ylabel('Z-score')
            axs[3].set_ylabel('Z-score')
        plt.tight_layout()

        tag = "first" if first else "every"
        fig.savefig(
            f"{out_folder}/{savename}_{tag}_{event_type}_{zscore_mode}.pdf",
            bbox_inches="tight"
        )
        plt.show()
        plt.close(fig)

        if file_had_any_bin:
            n_files_used += 1

   
    if n_files_used == 0:
        print("No files contributed any events; no summary plot generated.")
        return

    summary_folder = f'{figure_folder}/fig/triggered_zFF_summary'
    os.makedirs(summary_folder, exist_ok=True)

    figS, axsS, markc = configure_bw_plot(size=(10, 5), xaxis=True, nrows=2, ncols=3)
    row_yminsS = [np.inf, np.inf]
    row_ymaxsS = [-np.inf, -np.inf]
    # time axis: uses the last file's seg_len/t (assumes consistent tbef/taf; seg_len may vary if tinc differs)
    # If tinc differs across files, you should resample to a common t before summarizing.
    tS = t

    for b in range(6):
        ax = axsS[b]
        mats = np.vstack(perfile_bin_means[b])  # (n_files, seg_len)

        # drop files that had no events for this bin (all nan)
        keep = ~np.all(np.isnan(mats), axis=1)
        mats = mats[keep, :]

        if mats.shape[0] == 0:
            ax.set_title(bin_names[b], color=markc, size=12)
            ax.text(0.5, 0.5, "No files w/ events", color=markc,
                    ha='center', va='center', transform=ax.transAxes)
            ax.axvline(0, color=markc, linestyle='--', linewidth=1)
            continue

        mean_across_files = np.nanmean(mats, axis=0)
        sem_across_files = np.nanstd(mats, axis=0) / np.sqrt(mats.shape[0])
        y_lo = np.nanmin(mean_across_files - sem_across_files)
        y_hi = np.nanmax(mean_across_files + sem_across_files)
        if np.isfinite(y_lo) and np.isfinite(y_hi):
            r = 0 if b < 3 else 1
            row_yminsS[r] = min(row_yminsS[r], y_lo)
            row_ymaxsS[r] = max(row_ymaxsS[r], y_hi)

        ax.fill_between(tS, mean_across_files - sem_across_files,
                        mean_across_files + sem_across_files,
                        color=color, alpha=0.3)
        ax.plot(tS, mean_across_files, color=color, linewidth=1.5)
        ax.axvline(0, color=markc, linestyle='--', linewidth=1)
        ax.set_title(f"{bin_names[b]}", color=markc, size=12)
    for r in (0, 1):
        if np.isfinite(row_yminsS[r]) and np.isfinite(row_ymaxsS[r]):
            y0, y1 = row_yminsS[r], row_ymaxsS[r]
            if y0 == y1:
                pad = 1.0
            else:
                pad = 0.05 * (y1 - y0)
            y0 -= pad
            y1 += pad
            for j in range(r * 3, r * 3 + 3):
                axsS[j].set_ylim(y0, y1)
    axsS[3].set_xlabel('time (s)', size=14, color=markc)
    axsS[0].set_ylabel('Z-score', size=14, color=markc)
    axsS[3].set_ylabel('Z-score', size=14, color=markc)
    plt.tight_layout()

    tag = "first" if first else "every"
    figS.savefig(
        f"{summary_folder}/SUMMARY_{tag}_{event_type}_{zscore_mode}.pdf",
        bbox_inches="tight"
    )
    plt.show()
    plt.close(figS)

    print(f"Summary plot saved. Files contributing at least one bin: {n_files_used}")


def triggered_upwind_speed(figure_folder,neuron,color,tbef=30,taf=30,event_type='entry',first=False,norm_mode="baseline_subtract"):
    """
    Triggered analysis on upwind speed (ft2['y_velocity']) aligned to odor pulse entry/exit events.
    Produces:
      1) One 2x3 plot per file (mean ± SD across events within that file)
      2) One summary plot across files (mean ± SEM across files of per-file means)

    Bins:
      Odor2/mfc3: pulses 1-5, 6-10, 11-15
      Odor1/mfc2: pulses 1-5, 6-10, 11-15
    """

    bin_names = [
        "oct pulses 1–5",
        "oct pulses 6–10 (+LED)",
        "oct pulses 11–15",
        "acv pulses 1–5",
        "acv pulses 6–10 (+LED)",
        "acv pulses 11–15",
    ]

    out_folder = f'{figure_folder}/fig/triggered_upwind_speed_per_file'
    os.makedirs(out_folder, exist_ok=True)

    # collect per-file mean traces for summary (6 bins)
    perfile_bin_means = [[] for _ in range(6)]
    n_files_used = 0

    last_t = None  # for summary axis (assumes consistent sampling)

    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith('.pkl'):
            continue
        print(filename)

        savename, _ = os.path.splitext(filename)
        img_dict = open_pickle(f'{figure_folder}/{filename}')
        pv2, ft2 = process_pickle(img_dict, neuron)  
        if pv2 is None or ft2 is None:
            print(f"Skipping file {filename} due to missing data.")
            continue

        if 'y_velocity' not in ft2.columns:
            print(f"Skipping {filename}: ft2 has no 'y_velocity' column.")
            continue

        # V = ft2['y_velocity'].to_numpy()
        V = ft2['ft_speed'].to_numpy()
        # V = np.abs(ft2['ft_heading']).to_numpy()
        time = pv2['relative_time'].to_numpy()

        # --- find odor2 (mfc3) events ---
        odor2_on = (ft2['mfc3_stpt'].to_numpy() != 0).astype(int)
        d2 = np.diff(odor2_on)
        if event_type == 'entry':
            son2 = np.where(d2 == 1)[0] + 1
        elif event_type == 'exit':
            son2 = np.where(d2 == -1)[0] + 1
        else:
            raise ValueError("event_type must be 'entry' or 'exit'")

        # --- find odor1 (mfc2) events ---
        odor1_on = (ft2['mfc2_stpt'].to_numpy() != 0).astype(int)
        d1 = np.diff(odor1_on)
        if event_type == 'entry':
            son1 = np.where(d1 == 1)[0] + 1
        else:
            son1 = np.where(d1 == -1)[0] + 1

        if len(son2) == 0 and len(son1) == 0:
            print(f"No {event_type} events found in file: {filename}")
            continue

        # first 15 pulses
        son2 = son2[:15]
        son1 = son1[:15]
        if first:
            son2 = son2[:1]
            son1 = son1[:1]

        # bins
        son2_b1, son2_b2, son2_b3 = son2[0:5], son2[5:10], son2[10:15]
        son1_b1, son1_b2, son1_b3 = son1[0:5], son1[5:10], son1[10:15]
        son_bins = [son2_b1, son2_b2, son2_b3, son1_b1, son1_b2, son1_b3]

        # timebase
        tinc = np.mean(np.diff(time))
        idx_bef = int(np.round(float(tbef) / tinc))
        idx_af = int(np.round(float(taf) / tinc))
        seg_len = idx_bef + idx_af + 1
        t = np.linspace(-tbef, taf, seg_len)
        last_t = t

        # plot per file
        fig, axs, markc = configure_bw_plot(size=(10, 5), xaxis=True, nrows=2, ncols=3)

        file_had_any_bin = False

        for b, son in enumerate(son_bins):
            ax = axs[b]

            if len(son) == 0:
                ax.set_title(bin_names[b], color=markc, size=12)
                ax.text(0.5, 0.5, "No events", color=markc,
                        ha='center', va='center', transform=ax.transAxes)
                ax.axvline(0, color=markc, linestyle='--', linewidth=1)
                perfile_bin_means[b].append(np.full(seg_len, np.nan))
                continue

            mat = np.full((len(son), seg_len), np.nan)

            for i, s in enumerate(son):
                idx = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
                valid = (idx >= 0) & (idx < len(V))
                idx = idx[valid]
                seg = V[idx]
                mat[i, :len(seg)] = seg

            # normalization options for velocity
            if norm_mode == "none":
                pass
            elif norm_mode == "baseline_subtract":
                baseline = mat[:, :idx_bef]
                mu = np.nanmean(baseline, axis=1, keepdims=True)
                mat = mat - mu
            elif norm_mode == "baseline_zscore":
                baseline = mat[:, :idx_bef]
                mu = np.nanmean(baseline, axis=1, keepdims=True)
                sd = np.nanstd(baseline, axis=1, keepdims=True)
                mat = (mat - mu) / sd
            elif norm_mode == "trial_zscore":
                mat = stats.zscore(mat, axis=1, nan_policy='omit')
            else:
                raise ValueError("norm_mode must be 'none', 'baseline_subtract', 'baseline_zscore', or 'trial_zscore'")

            masked = np.ma.masked_array(mat, np.isnan(mat))
            mn = np.ma.mean(masked, axis=0)
            sd = np.ma.std(masked, axis=0)

            ax.fill_between(t, mn - sd, mn + sd, color=color, alpha=0.3)
            ax.plot(t, mn, color=color, linewidth=1.5)
            ax.axvline(0, color=markc, linestyle='--', linewidth=1)
            ax.set_title(bin_names[b], color=markc, size=12)

            perfile_bin_means[b].append(np.array(mn.filled(np.nan)))
            file_had_any_bin = True

        # labels
        axs[3].set_xlabel('time (s)', size=14, color=markc)
        ylab = "Upwind speed" if norm_mode == "none" else "Δ upwind speed" if norm_mode == "baseline_subtract" else "Normalized upwind speed"
        axs[0].set_ylabel(ylab, size=14, color=markc)
        axs[3].set_ylabel(ylab, size=14, color=markc)

        plt.tight_layout()

        tag = "first" if first else "every"
        fig.savefig(
            f"{out_folder}/{savename}_{tag}_{event_type}_{norm_mode}_6bins.pdf",
            bbox_inches="tight"
        )
        plt.show()
        plt.close(fig)

        if file_had_any_bin:
            n_files_used += 1

    # summary
    if n_files_used == 0 or last_t is None:
        print("No files contributed any events; no summary plot generated.")
        return

    summary_folder = f'{figure_folder}/fig/triggered_upwind_speed_summary'
    os.makedirs(summary_folder, exist_ok=True)

    figS, axsS, markc = configure_bw_plot(size=(10, 5), xaxis=True, nrows=2, ncols=3)
    row_yminS = [np.inf, np.inf]
    row_ymaxS = [-np.inf, -np.inf]

    for b in range(6):
        ax = axsS[b]
        mats = np.vstack(perfile_bin_means[b])

        keep = ~np.all(np.isnan(mats), axis=1)
        mats = mats[keep, :]

        if mats.shape[0] == 0:
            ax.set_title(bin_names[b], color=markc, size=12)
            ax.text(0.5, 0.5, "No files w/ events", color=markc,
                    ha='center', va='center', transform=ax.transAxes)
            ax.axvline(0, color=markc, linestyle='--', linewidth=1)
            continue

        mean_files = np.nanmean(mats, axis=0)
        sem_files = np.nanstd(mats, axis=0) / np.sqrt(mats.shape[0])
        r = 0 if b < 3 else 1
        lo = np.nanmin(mean_files - sem_files)
        hi = np.nanmax(mean_files + sem_files)
        if np.isfinite(lo) and np.isfinite(hi):
            row_yminS[r] = min(row_yminS[r], lo)
            row_ymaxS[r] = max(row_ymaxS[r], hi)

        ax.fill_between(last_t, mean_files - sem_files, mean_files + sem_files, color=color, alpha=0.3)
        ax.plot(last_t, mean_files, color=color, linewidth=1.5)
        ax.axvline(0, color=markc, linestyle='--', linewidth=1)
        ax.set_title(f"{bin_names[b]} (n={mats.shape[0]} files)", color=markc, size=12)

    row_axesS = [[axsS[0], axsS[1], axsS[2]], [axsS[3], axsS[4], axsS[5]]]
    for r in (0, 1):
        if np.isfinite(row_yminS[r]) and np.isfinite(row_ymaxS[r]):
            ymin, ymax = row_yminS[r], row_ymaxS[r]
            pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
            for ax in row_axesS[r]:
                ax.set_ylim(ymin - pad, ymax + pad)
    axsS[3].set_xlabel('time (s)', size=14, color=markc)
    ylab = "Upwind speed" if norm_mode == "none" else "Δ upwind speed" if norm_mode == "baseline_subtract" else "Normalized upwind speed"
    axsS[0].set_ylabel(ylab, size=14, color=markc)
    axsS[3].set_ylabel(ylab, size=14, color=markc)

    plt.tight_layout()
    tag = "first" if first else "every"
    figS.savefig(
        f"{summary_folder}/SUMMARY_{tag}_{event_type}_{norm_mode}_6bins_meanSEM_acrossFiles.pdf",
        bbox_inches="tight"
    )
    plt.show()
    plt.close(figS)

    print(f"Summary plot saved. Files contributing at least one bin: {n_files_used}")


def triggered_zFF_exit_minOFF_all(
    figure_folder,
    neuron,
    color,
    taf=30,
    min_off_bout_dur_s=5.0,
    which_odor="both",          # "mfc2", "mfc3", or "both"
    zscore_mode="trial",        # "global", "trial", or "baseline" (baseline falls back; see below)
    first=False                 # if True, only the first qualifying exit per file
):
    """
    Triggered zFF around odor EXIT events (ON->OFF), keeping only exits where the
    subsequent OFF-bout duration >= min_off_bout_dur_s seconds.

    Caveat: acquisition is paused during entries, so tbef is forced to 0.
    """

    tbef = 0.0  # forced: no pre-exit data exists
    out_folder = f'{figure_folder}/fig/triggered_zFF_per_file'
    os.makedirs(out_folder, exist_ok=True)

    perfile_means = []
    n_files_used = 0

    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith('.pkl'):
            continue
        print(filename)

        savename, _ = os.path.splitext(filename)
        img_dict = open_pickle(f'{figure_folder}/{filename}')
        pv2, ft2 = process_pickle(img_dict, neuron)
        if pv2 is None or ft2 is None:
            print(f"Skipping file {filename} due to missing data.")
            continue

        FF = pv2[f'0_{neuron}'].to_numpy()
        time = pv2['relative_time'].to_numpy()
        d, di, do = light_on_off(ft2)
        print(len(di))
        # time increment
        tinc = float(np.nanmean(np.diff(time)))
        if not np.isfinite(tinc) or tinc <= 0:
            print(f"Skipping file {filename} due to invalid time increments.")
            continue

        idx_bef = 0
        idx_af = int(np.round(float(taf) / tinc))
        seg_len = idx_bef + idx_af + 1
        t = np.linspace(-tbef, taf, seg_len)

        def qualifying_exits_by_off_bout_and_led(stpt_col: str) -> np.ndarray:
            """
            Keep exit events that:
            1) Have OFF-bout duration >= min_off_bout_dur_s
            2) Were preceded by an ON-bout during which LED was ON (led1_stpt == 0)
            """
            odor_on = (ft2[stpt_col].to_numpy() != 0).astype(int)
            led_raw = ft2["led1_stpt"].to_numpy()
            led_on = (np.isfinite(led_raw) & (np.abs(led_raw) < 1e-6)).astype(int)

            d = np.diff(odor_on)
            exit_idxs  = np.where(d == -1)[0] + 1   # ON->OFF
            entry_idxs = np.where(d ==  1)[0] + 1   # OFF->ON

            # handle case where file starts already ON
            if odor_on[0] == 1:
                entry_idxs = np.r_[0, entry_idxs]

            if exit_idxs.size == 0:
                return np.array([], dtype=int)

            qual = []
            n = len(odor_on)

            for ex in exit_idxs:
                # ---------- 1) OFF-bout duration check ----------
                next_entries = entry_idxs[entry_idxs > ex]
                if next_entries.size == 0:
                    off_len_samples = n - ex
                else:
                    off_len_samples = next_entries[0] - ex

                off_dur_s = off_len_samples * tinc
                if off_dur_s < float(min_off_bout_dur_s):
                    continue

                # ---------- 2) Preceding ON-bout LED condition ----------
                prev_entries = entry_idxs[entry_idxs < ex]
                if prev_entries.size == 0:
                    continue
                en = prev_entries[-1]

                # LED must be ON for the whole preceding ON bout
                led_segment = led_on[en:ex]   # during the ON bout
                valid = ~np.isnan(led_segment)
                if valid.sum() == 0:
                    continue

                frac_led_on = np.mean(led_segment[valid] == 1)

                # require majority LED ON
                if frac_led_on > 0.5:
                    qual.append(ex)

            return np.array(qual, dtype=int)

        # collect qualifying exits across selected odor channels
        exit_list = []
        if which_odor in ("both", "mfc3"):
            exit_list.append(qualifying_exits_by_off_bout_and_led("mfc3_stpt"))
        if which_odor in ("both", "mfc2"):
            exit_list.append(qualifying_exits_by_off_bout_and_led("mfc2_stpt"))

        exits = np.unique(np.concatenate(exit_list)) if exit_list else np.array([], dtype=int)

        if exits.size == 0:
            print(f"No qualifying exits (OFF bout >= {min_off_bout_dur_s}s) in file: {filename}")
            continue

        if first:
            exits = exits[:1]

        # extract segments (tbef=0 => segment starts at exit)
        mn_mat = np.full((len(exits), seg_len), np.nan)
        for i, s in enumerate(exits):
            idx_array = np.arange(s, s + idx_af + 1, dtype=int)
            valid = (idx_array >= 0) & (idx_array < len(FF))
            idx_array = idx_array[valid]
            segment = FF[idx_array]
            mn_mat[i, :len(segment)] = segment

        # normalization
        if zscore_mode == "global":
            mn_mat = stats.zscore(mn_mat, axis=None, nan_policy='omit')
        elif zscore_mode == "trial":
            mn_mat = stats.zscore(mn_mat, axis=1, nan_policy='omit')
        elif zscore_mode == "baseline":
            # baseline window is empty because tbef=0; fallback to trial z-score
            mn_mat = stats.zscore(mn_mat, axis=1, nan_policy='omit')
        else:
            raise ValueError("zscore_mode must be 'global', 'trial', or 'baseline'")

        masked = np.ma.masked_array(mn_mat, np.isnan(mn_mat))
        plt_mn = np.ma.mean(masked, axis=0)
        plt_sd = np.ma.std(masked, axis=0)

        # per-file plot (single axis)
        fig, ax, markc = configure_bw_plot(size=(6, 4), xaxis=True, nrows=1, ncols=1)
        if isinstance(ax, (list, np.ndarray)):
            ax = ax[0]

        ax.fill_between(t, plt_mn - plt_sd, plt_mn + plt_sd, color=color, alpha=0.3)
        ax.plot(t, plt_mn, color=color, linewidth=1.5)
        ax.axvline(0, color=markc, linestyle='--', linewidth=1)

        ax.set_title(
            f"Qualifying exits: OFF bout ≥ {min_off_bout_dur_s:.1f}s\n"
            f"{which_odor} (n={len(exits)} exits)",
            color=markc, size=12
        )
        ax.set_xlabel('time from exit (s)', size=14, color=markc)
        ax.set_ylabel('Z-score', size=14, color=markc)
        plt.tight_layout()

        tag = "first" if first else "every"
        fig.savefig(
            f"{out_folder}/{savename}_{tag}_exit_OFFmin{min_off_bout_dur_s:g}s_{which_odor}_{zscore_mode}_ALL.pdf",
            bbox_inches="tight"
        )
        plt.show()
        plt.close(fig)

        perfile_means.append(np.array(plt_mn.filled(np.nan)))
        n_files_used += 1

    if n_files_used == 0:
        print("No files contributed any qualifying exits; no summary plot generated.")
        return

    # summary across files (mean ± SEM)
    summary_folder = f'{figure_folder}/fig/triggered_zFF_summary'
    os.makedirs(summary_folder, exist_ok=True)

    mats = np.vstack(perfile_means)
    keep = ~np.all(np.isnan(mats), axis=1)
    mats = mats[keep, :]

    if mats.shape[0] == 0:
        print("Summary: all traces are NaN after filtering; no summary plot generated.")
        return

    mean_across_files = np.nanmean(mats, axis=0)
    sem_across_files = np.nanstd(mats, axis=0) / np.sqrt(mats.shape[0])

    figS, axS, markc = configure_bw_plot(size=(6, 4), xaxis=True, nrows=1, ncols=1)
    if isinstance(axS, (list, np.ndarray)):
        axS = axS[0]

    # uses the last file's t; assumes consistent tinc across files
    tS = t

    axS.fill_between(tS, mean_across_files - sem_across_files,
                     mean_across_files + sem_across_files,
                     color=color, alpha=0.3)
    axS.plot(tS, mean_across_files, color=color, linewidth=1.5)
    axS.axvline(0, color=markc, linestyle='--', linewidth=1)

    axS.set_title(
        f"SUMMARY: exits with OFF bout ≥ {min_off_bout_dur_s:.1f}s, {which_odor}\n"
        f"(n={mats.shape[0]} files)",
        color=markc, size=12
    )
    axS.set_xlabel('time from exit (s)', size=14, color=markc)
    axS.set_ylabel('Z-score', size=14, color=markc)
    plt.tight_layout()

    tag = "first" if first else "every"
    figS.savefig(
        f"{summary_folder}/SUMMARY_{tag}_exit_OFFmin{min_off_bout_dur_s:g}s_{which_odor}_{zscore_mode}_meanSEM_acrossFiles.pdf",
        bbox_inches="tight"
    )
    plt.show()
    plt.close(figS)

    print(f"Summary plot saved. Files used: {mats.shape[0]}")

def triggered_zFF_new(
    figure_folder, rois, tbef=30, taf=30, event_type="entry", first=False,
    colors=None, odor_source="auto", max_events=None, zscore_mode="global",
    summary_mode="sem"  # "sem" or "sd"
):

    n_rois = len(rois)
    out_folder = os.path.join(figure_folder, "fig", "triggered_zFF")
    os.makedirs(out_folder, exist_ok=True)

    # --- NEW: accumulate per-file mean traces for summary plot ---
    summary_traces = {roi: [] for roi in rois}   # each entry: np.array len(t_common)
    t_common = None                              # common timebase for summary

    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue
        savename, _ = os.path.splitext(filename)
        fullpath = os.path.join(figure_folder, filename)
        img_dict = open_pickle(fullpath)
        pv2, ft2 = process_dLight_pickle(img_dict, rois)
        print(list(pv2.columns))

        time = ft2["relative_time"].to_numpy() if "relative_time" in ft2 else pv2["relative_time"].to_numpy()
        tinc = np.nanmean(np.diff(time))
        if not np.isfinite(tinc) or tinc <= 0:
            print(f"Skipping {filename}: bad timebase")
            continue

        idx_bef = int(np.round(float(tbef) / tinc))
        idx_af  = int(np.round(float(taf) / tinc))
        win_len = idx_bef + idx_af + 1
        t = np.linspace(-tbef, taf, win_len)

        odor1 = (ft2["mfc3_stpt"].to_numpy() != 0).astype(int) if "mfc2_stpt" in ft2 else None
        odor2 = (ft2["mfc4_stpt"].to_numpy() != 0).astype(int) if "mfc3_stpt" in ft2 else None

        if odor_source == "auto":
            use = "mfc4" if "oct" in filename.lower() else "mfc3"
        else:
            use = odor_source

        if use == "mfc3":
            if odor1 is None:
                print(f"Skipping {filename}: no mfc2_stpt")
                continue
            son_all = detect_events(odor1, event_type)
        elif use == "mfc4":
            if odor2 is None:
                print(f"Skipping {filename}: no mfc3_stpt")
                continue
            son_all = detect_events(odor2, event_type)
        elif use == "both":
            if odor1 is None and odor2 is None:
                print(f"Skipping {filename}: no odor channels")
                continue
            s1 = detect_events(odor1, event_type) if odor1 is not None else np.array([], dtype=int)
            s2 = detect_events(odor2, event_type) if odor2 is not None else np.array([], dtype=int)
            son_all = np.unique(np.r_[s1, s2])
        else:
            raise ValueError("odor_source must be 'auto', 'mfc2', 'mfc3', or 'both'")

        if son_all.size == 0:
            print(f"No {event_type} events found in file: {filename}")
            continue

        # optional truncation
        if max_events is not None:
            son_all = son_all[:max_events]
        if first:
            son_all = son_all[:1]

        # plot per file
        fig, axs, markc = configure_bw_plot(size=(4 * n_rois, 3), xaxis=True, ncols=n_rois)
        if n_rois == 1:
            axs = np.array([axs])

        for r_idx, roi in enumerate(rois):
            ax = axs[r_idx]
            color = None if colors is None else colors[r_idx % len(colors)]
            FF = pv2[f'0_{roi}'].to_numpy()
            mn_mat = np.full((len(son_all), win_len), np.nan)

            for i, s in enumerate(son_all):
                idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
                valid = (idx_array >= 0) & (idx_array < len(FF))
                idx_array = idx_array[valid]
                seg = FF[idx_array]
                mn_mat[i, :len(seg)] = seg

            if zscore_mode == "global":
                mn_mat = stats.zscore(mn_mat, axis=None, nan_policy="omit")
            elif zscore_mode == "trial":
                mn_mat = stats.zscore(mn_mat, axis=1, nan_policy="omit")
            elif zscore_mode == "baseline":
                baseline = mn_mat[:, :idx_bef]
                mu = np.nanmean(baseline, axis=1, keepdims=True)
                sd = np.nanstd(baseline, axis=1, keepdims=True)
                mn_mat = (mn_mat - mu) / sd
            elif zscore_mode == 'raw':
                pass
            else:
                raise ValueError("zscore_mode must be 'global', 'trial', or 'baseline'")

            masked = np.ma.masked_array(mn_mat, np.isnan(mn_mat))
            plt_mn = np.ma.mean(masked, axis=0)
            plt_sd = np.ma.std(masked, axis=0)

            # --- NEW: store per-file mean trace for summary plot ---
            y = plt_mn.filled(np.nan)  # np.array
            if t_common is None:
                t_common = t.copy()
                y_common = y
            else:
                # if different timebase/window length, interpolate onto t_common
                if (len(t) != len(t_common)) or (not np.allclose(t, t_common, equal_nan=True)):
                    valid = np.isfinite(y)
                    if np.sum(valid) >= 2:
                        y_common = np.interp(t_common, t[valid], y[valid], left=np.nan, right=np.nan)
                    else:
                        y_common = np.full_like(t_common, np.nan, dtype=float)
                else:
                    y_common = y

            summary_traces[roi].append(y_common)

            # original per-file plot
            if color is None:
                ax.fill_between(t, plt_mn - plt_sd, plt_mn + plt_sd, alpha=0.3)
                ax.plot(t, plt_mn, linewidth=1.5)
            else:
                ax.fill_between(t, plt_mn - plt_sd, plt_mn + plt_sd, color=color, alpha=0.3)
                ax.plot(t, plt_mn, color=color, linewidth=1.5)

            ax.axvline(0, color=markc, linestyle="--", linewidth=1)
            ax.set_title(f"{roi}", color=markc, size=14)

        axs[0].set_ylabel("Z-score", color=markc, size=12)
        axs[0].set_xlabel("time (s)", color=markc, size=12)
        tag = "first" if first else "every"
        fname = f"{savename}_{'_'.join(rois)}_{tag}_{event_type}_{zscore_mode}.pdf"
        fig.savefig(os.path.join(out_folder, fname), bbox_inches="tight")
        plt.show()
        plt.close(fig)

    # --- NEW: summary plot across ALL files in the folder ---
    if t_common is None:
        print("No valid files processed; skipping summary plot.")
        return

    figS, axsS, markc = configure_bw_plot(size=(3 * n_rois, 2), xaxis=True, ncols=n_rois)
    if n_rois == 1:
        axsS = np.array([axsS])

    # styling knobs (adjust if you want)
    trace_lw = 1.0
    trace_alpha = 0.5
    mean_lw = 2

    for r_idx, roi in enumerate(rois):
        ax = axsS[r_idx]
        color = None if colors is None else colors[r_idx % len(colors)]

        traces = summary_traces[roi]
        if len(traces) == 0:
            ax.set_title(f"{roi} (no data)", color=markc, size=14)
            ax.axvline(0, color=markc, linestyle="--", linewidth=1)
            continue

        M = np.vstack(traces)  # (n_files, len(t_common))
        mu = np.nanmean(M, axis=0)

        # 1) plot individual traces (thin + transparent)
        for y in M:
            if color is None:
                ax.plot(t_common, y, linewidth=trace_lw, alpha=trace_alpha)
            else:
                ax.plot(t_common, y, color=color, linewidth=trace_lw, alpha=trace_alpha)

        # 2) plot mean trace (thick + solid)
        if color is None:
            ax.plot(t_common, mu, linewidth=mean_lw)
        else:
            ax.plot(t_common, mu, color=color, linewidth=mean_lw)

        ax.axvline(0, color=markc, linestyle="--", linewidth=1)
        ax.set_title(f"{roi} (all files, n={len(traces)})", color=markc, size=14)

    axsS[0].set_ylabel("Z-score", color=markc, size=12)
    axsS[0].set_xlabel("time (s)", color=markc, size=12)
    axsS[0].set_ylim(-4,0.5)

    tag = "first" if first else "every"
    summary_fname = f"SUMMARY_allfiles_{'_'.join(rois)}_{tag}_{event_type}_{zscore_mode}_traces.pdf"
    figS.savefig(os.path.join(out_folder, summary_fname), bbox_inches="tight")
    plt.show()
    plt.close(figS)




def avg_trace(figure_folder,rois,event_type='entry',odor_source="auto", n_pts=200, min_bout_s=1.0,max_bouts=None, colors=None,plot_per_file=True, mode="raw", baseline_frac=0.1,eps=1e-12):
    """
    Create pseudo-time average trace across the duration of each odor-OFF bout.

    Pseudo-time:
      - each OFF bout is mapped to x in [0, 1]
      - ROI signal within bout is interpolated to `n_pts` samples

    mode:
      - "raw": average interpolated bouts directly
      - "baseline_z": z-score EACH BOUT using its own baseline portion
                      (first `baseline_frac` of pseudo-time), then average

    Outputs:
      - saves a SUMMARY pdf in figure_folder/fig/avg_offbouts
      - optionally returns data dict if return_data=True
    """
    if mode not in ("raw", "baseline_z"):
        raise ValueError("mode must be 'raw' or 'baseline_z'")
    if not (0 < baseline_frac < 1):
        raise ValueError("baseline_frac must be between 0 and 1")

    n_rois = len(rois)
    out_folder = os.path.join(figure_folder, "fig", "avg_offbouts")
    os.makedirs(out_folder, exist_ok=True)

    # common pseudo-time axis
    x = np.linspace(0, 1, int(n_pts))
    base_n = max(2, int(np.floor(baseline_frac * len(x))))

    # store one per-file mean trace per ROI (for plotting & grand mean)
    per_file_means = {roi: [] for roi in rois}  # list of (n_pts,) arrays
    processed_files = 0

    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue
        fullpath = os.path.join(figure_folder, filename)
        img_dict = open_pickle(fullpath)
        pv2, ft2 = process_dLight_pickle(img_dict, rois)

        time = ft2["relative_time"].to_numpy() if "relative_time" in ft2 else pv2["relative_time"].to_numpy()
        if time.size < 2:
            print(f"Skipping {filename}: no timebase")
            continue
        tinc = np.nanmean(np.diff(time))
        if not np.isfinite(tinc) or tinc <= 0:
            print(f"Skipping {filename}: bad timebase")
            continue

        odor1 = (ft2["mfc2_stpt"].to_numpy() != 0).astype(bool) if "mfc2_stpt" in ft2 else None
        odor2 = (ft2["mfc3_stpt"].to_numpy() != 0).astype(bool) if "mfc3_stpt" in ft2 else None

        if odor_source == "auto":
            use = "mfc3" if "oct" in filename.lower() else "mfc2"
        else:
            use = odor_source

        if use == "mfc2":
            if odor1 is None:
                print(f"Skipping {filename}: no mfc2_stpt")
                continue
            odor_on = odor1
        elif use == "mfc3":
            if odor2 is None:
                print(f"Skipping {filename}: no mfc3_stpt")
                continue
            odor_on = odor2
        elif use == "both":
            if odor1 is None and odor2 is None:
                print(f"Skipping {filename}: no odor channels")
                continue
            if odor1 is None:
                odor_on = odor2
            elif odor2 is None:
                odor_on = odor1
            else:
                odor_on = (odor1 | odor2)
        else:
            raise ValueError("odor_source must be 'auto', 'mfc2', 'mfc3', or 'both'")

        if event_type == 'entry':
            off_bouts = odor_bout_indices(odor_on)
        else:
            off_bouts = exit_bout_indices(odor_on)

        min_len = int(np.ceil(float(min_bout_s) / float(tinc)))
        off_bouts = [(s, e) for (s, e) in off_bouts if (e - s) >= max(2, min_len)]

        if len(off_bouts) == 0:
            print(f"Skipping {filename}: no OFF bouts >= {min_bout_s}s")
            continue

        if max_bouts is not None:
            off_bouts = off_bouts[:max_bouts]

        file_had_any = False
        for roi in rois:
            col = f"0_{roi}"
            if col not in pv2.columns:
                print(f"File {filename}: missing column {col}")
                continue

            y = pv2[col].to_numpy()
            bout_traces = []

            for (s, e) in off_bouts:
                seg = y[s:e]
                if seg.size < 2:
                    continue

                valid = np.isfinite(seg)
                if np.sum(valid) < 2:
                    continue

                # original segment "pseudo-x" (0..1)
                xx = np.linspace(0, 1, seg.size)

                # interpolate onto common pseudo-time axis
                seg_interp = np.interp(x, xx[valid], seg[valid], left=np.nan, right=np.nan)

                # optional baseline z-scoring per bout
                if mode == "baseline_z":
                    base = seg_interp[:base_n]
                    mu = np.nanmean(base)
                    sd = np.nanstd(base)
                    if not np.isfinite(sd) or sd < eps:
                        # if baseline is flat/invalid, skip this bout to avoid exploding values
                        continue
                    seg_interp = (seg_interp - mu) / sd

                bout_traces.append(seg_interp)

            if len(bout_traces) == 0:
                continue

            bout_mat = np.vstack(bout_traces)      # (n_bouts, n_pts)
            file_mean = np.nanmean(bout_mat, axis=0)

            per_file_means[roi].append(file_mean)
            file_had_any = True

        if file_had_any:
            processed_files += 1

    if processed_files == 0:
        print("No valid files processed; nothing to plot.")
        return

    fig, axs, markc = configure_bw_plot(size=(3 * n_rois, 2), xaxis=True, ncols=n_rois)
    if n_rois == 1:
        axs = np.array([axs])

    trace_lw = 1.0
    trace_alpha = 0.5
    mean_lw = 2

    for r_idx, roi in enumerate(rois):
        ax = axs[r_idx]
        color = None if colors is None else colors[r_idx % len(colors)]

        traces = per_file_means[roi]
        if len(traces) == 0:
            ax.set_title(f"{roi} (no data)", color=markc, size=14)
            ax.set_xlabel("pseudo time (0→1)", color=markc, size=12)
            continue

        M = np.vstack(traces)        # (n_files, n_pts)
        mu = np.nanmean(M, axis=0)   # grand mean across files

        if plot_per_file:
            for yy in M:
                ax.plot(x, yy, color=markc, linewidth=trace_lw, alpha=trace_alpha)

        ax.plot(x, mu, color=markc, linewidth=mean_lw)

        ax.set_title(f'{event_type} bouts', color=markc, size=14)
        ax.set_xlabel("pseudo time (0→1)", color=markc, size=12)

    ylab = "dF/F" if mode == "raw" else "Z-score"
    axs[0].set_ylabel(ylab, color=markc, size=12)

    fname = (f"{event_type}_bouts_{mode}.pdf")
    fig.savefig(os.path.join(out_folder, fname), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _event_durations(arr, son, event_type):
    """Duration (in samples) of the on-period associated with each event index."""
    n = len(arr)
    durations = np.zeros(len(son), dtype=int)
    for i, s in enumerate(son):
        if event_type == "entry":
            j = s
            while j < n and arr[j] == 1:
                j += 1
            durations[i] = j - s
        elif event_type == "exit":
            j = s - 1
            cnt = 0
            while j >= 0 and arr[j] == 1:
                cnt += 1
                j -= 1
            durations[i] = cnt
        else:
            durations[i] = win_len  # fallback: don't filter unknown event types
    return durations


def triggered_zFF_dLight(figure_folder, rois, tbef=30, taf=30, event_type="entry", first=False,
                          colors=None, odor_source="auto", max_events=None, zscore_mode="global",
                          min_duration=None):
    """
    odor_source:
      - "auto": if 'oct' in filename -> mfc3, else -> mfc2 (matches your old intent)
      - "mfc2": use mfc2_stpt
      - "mfc3": use mfc3_stpt
      - "both": use both and merge/sort unique event indices
    zscore_mode:
      - "global" (matches your original: axis=None)
      - "trial"  (zscore per event)
      - "baseline" (zscore using pre-event window)
      - "raw"
    min_duration:
      - minimum event duration (in seconds) required to keep the event.
        Defaults to tbef if not specified (excludes events shorter than the
        pre-event window since they can't be properly baselined).
    """
    n_rois = len(rois)
    out_folder = os.path.join(figure_folder, "fig", "triggered_zFF")
    os.makedirs(out_folder, exist_ok=True)

    if min_duration is None:
        min_duration = tbef

    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue
        savename, _ = os.path.splitext(filename)
        fullpath = os.path.join(figure_folder, filename)
        img_dict = open_pickle(fullpath)
        pv2, ft2 = process_dLight_pickle(img_dict, rois)
        print(list(pv2.columns))

        time = ft2["relative_time"].to_numpy() if "relative_time" in ft2 else pv2["relative_time"].to_numpy()
        tinc = np.nanmean(np.diff(time))
        if not np.isfinite(tinc) or tinc <= 0:
            print(f"Skipping {filename}: bad timebase")
            continue

        idx_bef = int(np.round(float(tbef) / tinc))
        idx_af  = int(np.round(float(taf) / tinc))
        win_len = idx_bef + idx_af + 1
        t = np.linspace(-tbef, taf, win_len)
        min_dur_samples = int(np.round(float(min_duration) / tinc))

        odor1 = (ft2["mfc3_stpt"].to_numpy() != 0).astype(int) if "mfc3_stpt" in ft2 else None
        odor2 = (ft2["mfc4_stpt"].to_numpy() != 0).astype(int) if "mfc4_stpt" in ft2 else None

        if odor_source == "auto":
            use = "mfc4" if "oct" in filename.lower() else "mfc3"
        else:
            use = odor_source

        if use == "mfc3":
            if odor1 is None:
                print(f"Skipping {filename}: no mfc3_stpt")
                continue
            son_all = detect_events(odor1, event_type)
            durations = _event_durations(odor1, son_all, event_type)
        elif use == "mfc4":
            if odor2 is None:
                print(f"Skipping {filename}: no mfc4_stpt")
                continue
            son_all = detect_events(odor2, event_type)
            durations = _event_durations(odor2, son_all, event_type)
        elif use == "both":
            if odor1 is None and odor2 is None:
                print(f"Skipping {filename}: no odor channels")
                continue
            s1 = detect_events(odor1, event_type) if odor1 is not None else np.array([], dtype=int)
            s2 = detect_events(odor2, event_type) if odor2 is not None else np.array([], dtype=int)
            d1 = _event_durations(odor1, s1, event_type) if odor1 is not None else np.array([], dtype=int)
            d2 = _event_durations(odor2, s2, event_type) if odor2 is not None else np.array([], dtype=int)

            # merge, tracking durations alongside indices, then dedupe by index
            combined = np.r_[s1, s2]
            combined_dur = np.r_[d1, d2]
            son_all, uniq_idx = np.unique(combined, return_index=True)
            durations = combined_dur[uniq_idx]
        else:
            raise ValueError("odor_source must be 'auto', 'mfc2', 'mfc3', or 'both'")

        if son_all.size == 0:
            print(f"No {event_type} events found in file: {filename}")
            continue

        # exclude events whose duration is shorter than min_duration (default: tbef)
        keep = durations >= min_dur_samples
        n_excluded = np.sum(~keep)
        if n_excluded > 0:
            print(f"{filename}: excluding {n_excluded} short event(s) (< {min_duration}s)")
        son_all = son_all[keep]

        if son_all.size == 0:
            print(f"No events remain after duration filtering in file: {filename}")
            continue

        # optional truncation
        if max_events is not None:
            son_all = son_all[:max_events]
        if first:
            son_all = son_all[:1]

        # plot per file
        fig, axs, markc = configure_bw_plot(size=(2.5 * n_rois, 3.5), xaxis=True, ncols=n_rois)
        if n_rois == 1:
            axs = np.array([axs])

        any_roi_plotted = False

        for r_idx, roi in enumerate(rois):
            ax = axs[r_idx]
            color = None if colors is None else colors[r_idx % len(colors)]
            FF = pv2[f'0_{roi}'].to_numpy() 
            mn_mat = np.full((len(son_all), win_len), np.nan)

            for i, s in enumerate(son_all):
                idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
                valid = (idx_array >= 0) & (idx_array < len(FF))
                idx_array = idx_array[valid]
                seg = FF[idx_array]
                mn_mat[i, :len(seg)] = seg
            if zscore_mode in ("none", "raw", None):
                pass
            elif zscore_mode == "global":
                mn_mat = stats.zscore(mn_mat, axis=None, nan_policy="omit")
            elif zscore_mode == "trial":
                mn_mat = stats.zscore(mn_mat, axis=1, nan_policy="omit")
            elif zscore_mode == "baseline":
                baseline = mn_mat[:, :idx_bef]
                mu = np.nanmean(baseline, axis=1, keepdims=True)
                sd = np.nanstd(baseline, axis=1, keepdims=True)
                mn_mat = (mn_mat - mu) / sd
            else:
                raise ValueError("zscore_mode must be 'global', 'trial', or 'baseline'")

            masked = np.ma.masked_array(mn_mat, np.isnan(mn_mat))
            plt_mn = np.ma.mean(masked, axis=0)
            plt_sd = np.ma.std(masked, axis=0)

            if color is None:
                ax.fill_between(t, plt_mn - plt_sd, plt_mn + plt_sd, alpha=0.3)
                ax.plot(t, plt_mn, linewidth=1.5)
            else:
                ax.fill_between(t, plt_mn - plt_sd, plt_mn + plt_sd, color=color, alpha=0.3)
                ax.plot(t, plt_mn, color=color, linewidth=1.5)

            ax.axvline(0, color=markc, linestyle="--", linewidth=1)
            ax.set_title(f"{roi}", color=markc, size=14)
            tag = "first" if first else "every"
        axs[0].set_ylabel("Z-score", color=markc, size=12)
        axs[0].set_xlabel("time (s)", color=markc, size=12)
        plt.suptitle(filename)
        tag = "first" if first else "every"
        fname = f"{savename}_{'_'.join(rois)}_{tag}_{event_type}_{zscore_mode}.pdf"
        fig.savefig(os.path.join(out_folder, fname), bbox_inches="tight")
        plt.show()
        plt.close(fig)

def triggered_zFF_dLight_summary(figure_folder, rois, tbef=30, taf=30, event_type="entry", colors=None, odor_source="auto", max_events=None,zscore_mode="global", min_duration=None,per_file_overlay=True, error="sem"):
    n_rois = len(rois)
    out_folder = os.path.join(figure_folder, "fig", "triggered_zFF")
    os.makedirs(out_folder, exist_ok=True)
    if min_duration is None:
        min_duration = tbef
 

    pooled_trials = {roi: [] for roi in rois}
    per_file_means = {roi: [] for roi in rois}
    t_ref = None
 
    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue
        fullpath = os.path.join(figure_folder, filename)
        img_dict = open_pickle(fullpath)
        pv2, ft2 = process_dLight_pickle(img_dict, rois)
 
        time = ft2["relative_time"].to_numpy() if "relative_time" in ft2 else pv2["relative_time"].to_numpy()
        tinc = np.nanmean(np.diff(time))
        if not np.isfinite(tinc) or tinc <= 0:
            print(f"Skipping {filename}: bad timebase")
            continue
 
        idx_bef = int(np.round(float(tbef) / tinc))
        idx_af = int(np.round(float(taf) / tinc))
        win_len = idx_bef + idx_af + 1
        t = np.linspace(-tbef, taf, win_len)
        if t_ref is None:
            t_ref = t
        min_dur_samples = int(np.round(float(min_duration) / tinc))
 
        odor1 = (ft2["mfc3_stpt"].to_numpy() != 0).astype(int) if "mfc3_stpt" in ft2 else None
        odor2 = (ft2["mfc4_stpt"].to_numpy() != 0).astype(int) if "mfc4_stpt" in ft2 else None
 
        if odor_source == "auto":
            use = "mfc4" if "oct" in filename.lower() else "mfc3"
        else:
            use = odor_source
 
        if use == "mfc3":
            if odor1 is None:
                print(f"Skipping {filename}: no mfc3_stpt")
                continue
            son_all = detect_events(odor1, event_type)
            durations = _event_durations(odor1, son_all, event_type)
        elif use == "mfc4":
            if odor2 is None:
                print(f"Skipping {filename}: no mfc4_stpt")
                continue
            son_all = detect_events(odor2, event_type)
            durations = _event_durations(odor2, son_all, event_type)
        elif use == "both":
            if odor1 is None and odor2 is None:
                print(f"Skipping {filename}: no odor channels")
                continue
            s1 = detect_events(odor1, event_type) if odor1 is not None else np.array([], dtype=int)
            s2 = detect_events(odor2, event_type) if odor2 is not None else np.array([], dtype=int)
            d1 = _event_durations(odor1, s1, event_type) if odor1 is not None else np.array([], dtype=int)
            d2 = _event_durations(odor2, s2, event_type) if odor2 is not None else np.array([], dtype=int)
 
            combined = np.r_[s1, s2]
            combined_dur = np.r_[d1, d2]
            son_all, uniq_idx = np.unique(combined, return_index=True)
            durations = combined_dur[uniq_idx]
        else:
            raise ValueError("odor_source must be 'auto', 'mfc2', 'mfc3', or 'both'")
 
        if son_all.size == 0:
            print(f"No {event_type} events found in file: {filename}")
            continue
 
        keep = durations >= min_dur_samples
        n_excluded = np.sum(~keep)
        if n_excluded > 0:
            print(f"{filename}: excluding {n_excluded} short event(s) (< {min_duration}s)")
        son_all = son_all[keep]
 
        if son_all.size == 0:
            print(f"No events remain after duration filtering in file: {filename}")
            continue
 
        if max_events is not None:
            son_all = son_all[:max_events]
 
        for roi in rois:
            FF = pv2[f'0_{roi}'].to_numpy()
            mn_mat = np.full((len(son_all), win_len), np.nan)
 
            for i, s in enumerate(son_all):
                idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
                valid = (idx_array >= 0) & (idx_array < len(FF))
                idx_array = idx_array[valid]
                seg = FF[idx_array]
                mn_mat[i, :len(seg)] = seg
 
            if zscore_mode in ("none", "raw", None):
                pass
            elif zscore_mode == "global":
                mn_mat = stats.zscore(mn_mat, axis=None, nan_policy="omit")
            elif zscore_mode == "trial":
                mn_mat = stats.zscore(mn_mat, axis=1, nan_policy="omit")
            elif zscore_mode == "baseline":
                baseline = mn_mat[:, :idx_bef]
                mu = np.nanmean(baseline, axis=1, keepdims=True)
                sd = np.nanstd(baseline, axis=1, keepdims=True)
                mn_mat = (mn_mat - mu) / sd
            else:
                raise ValueError("zscore_mode must be 'global', 'trial', or 'baseline'")
 
            pooled_trials[roi].append(mn_mat)
 
            masked = np.ma.masked_array(mn_mat, np.isnan(mn_mat))
            per_file_means[roi].append(np.ma.mean(masked, axis=0))
 
    if t_ref is None:
        print("No usable files found in folder.")
        return {}
 
    pooled = {}
    for roi in rois:
        if len(pooled_trials[roi]) == 0:
            pooled[roi] = np.empty((0, len(t_ref)))
        else:
            pooled[roi] = np.vstack(pooled_trials[roi])
 
    fig, axs, markc = configure_white_plot(size=(2.5 * n_rois, 3.5), xaxis=True, ncols=n_rois)
    if n_rois == 1:
        axs = np.array([axs])
 
    for r_idx, roi in enumerate(rois):
        ax = axs[r_idx]
        color = None if colors is None else colors[r_idx % len(colors)]
        mat = pooled[roi]
        n_trials = mat.shape[0]
 
        if n_trials == 0:
            ax.set_title(f"{roi} (no events)", color=markc, size=14)
            continue
 
        masked = np.ma.masked_array(mat, np.isnan(mat))
        plt_mn = np.ma.mean(masked, axis=0)
        plt_sd = np.ma.std(masked, axis=0)
        if error == "sem":
            n_valid = np.ma.count(masked, axis=0)
            n_valid = np.where(n_valid == 0, 1, n_valid)
            err = plt_sd / np.sqrt(n_valid)
        else:
            err = plt_sd
 
        if per_file_overlay:
            for fm in per_file_means[roi]:
                ax.plot(t_ref, fm, color=(color if color else "gray"),
                        linewidth=0.6, alpha=0.35)
 
        if color is None:
            ax.fill_between(t_ref, plt_mn - err, plt_mn + err, alpha=0.3)
            ax.plot(t_ref, plt_mn, linewidth=2)
        else:
            ax.fill_between(t_ref, plt_mn - err, plt_mn + err, color=color, alpha=0.3)
            ax.plot(t_ref, plt_mn, color=color, linewidth=2)
        ax.set_ylim(-4,0.5)
        ax.axvline(0, color=markc, linestyle="--", linewidth=1)
        ax.set_title(f"{roi} (n={n_trials} events, {len(per_file_means[roi])} files)",
                     color=markc, size=12)
 
    axs[0].set_ylabel("Z-score", color=markc, size=12)
    axs[0].set_xlabel("time (s)", color=markc, size=12)
    plt.suptitle(f"Summary across all files ({event_type}, {zscore_mode})")
 
    fname = f"SUMMARY_{'_'.join(rois)}_{event_type}_{zscore_mode}.pdf"
    fig.savefig(os.path.join(out_folder, fname), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def triggered_loc_dLight(figure_folder, rois, cols, tbef=30, taf=30, event_type="entry", first=False,colors=None, odor_source="auto", max_events=None, zscore_mode="global"):
    n_rois = len(rois)
    out_folder = os.path.join(figure_folder, "fig", "triggered_zFF")
    os.makedirs(out_folder, exist_ok=True)
    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue
        savename, _ = os.path.splitext(filename)
        fullpath = os.path.join(figure_folder, filename)
        img_dict = open_pickle(fullpath)
        pv2, ft2 = process_dLight_pickle(img_dict, rois)

        time = ft2["relative_time"].to_numpy() if "relative_time" in ft2 else pv2["relative_time"].to_numpy()
        tinc = np.nanmean(np.diff(time))
        if not np.isfinite(tinc) or tinc <= 0:
            print(f"Skipping {filename}: bad timebase")
            continue

        idx_bef = int(np.round(float(tbef) / tinc))
        idx_af  = int(np.round(float(taf) / tinc))
        win_len = idx_bef + idx_af + 1
        t = np.linspace(-tbef, taf, win_len)

        odor1 = (ft2["mfc2_stpt"].to_numpy() != 0).astype(int) if "mfc2_stpt" in ft2 else None
        odor2 = (ft2["mfc3_stpt"].to_numpy() != 0).astype(int) if "mfc3_stpt" in ft2 else None

        if odor_source == "auto":
            use = "mfc3" if "oct" in filename.lower() else "mfc2"
        else:
            use = odor_source

        if use == "mfc2":
            if odor1 is None:
                print(f"Skipping {filename}: no mfc2_stpt")
                continue
            son_all = detect_events(odor1, event_type)
        elif use == "mfc3":
            if odor2 is None:
                print(f"Skipping {filename}: no mfc3_stpt")
                continue
            son_all = detect_events(odor2, event_type)
        elif use == "both":
            if odor1 is None and odor2 is None:
                print(f"Skipping {filename}: no odor channels")
                continue
            s1 = detect_events(odor1, event_type) if odor1 is not None else np.array([], dtype=int)
            s2 = detect_events(odor2, event_type) if odor2 is not None else np.array([], dtype=int)
            son_all = np.unique(np.r_[s1, s2])
        else:
            raise ValueError("odor_source must be 'auto', 'mfc2', 'mfc3', or 'both'")

        if son_all.size == 0:
            print(f"No {event_type} events found in file: {filename}")
            continue

        if max_events is not None:
            son_all = son_all[:max_events]
        if first:
            son_all = son_all[:1]

        fig, axs, markc = configure_bw_plot(size=(2.5 * len(cols), 3.5), xaxis=True, ncols=len(rois))
        if n_rois == 1:
            axs = np.array([axs])

        any_roi_plotted = False

        for r_idx, col in enumerate(cols):
            ax = axs[r_idx]
            color = markc
            if col == 'ft_heading':
                FF = np.abs(ft2[col]).to_numpy() 
            else:
                FF = ft2[col].to_numpy() 
            mn_mat = np.full((len(son_all), win_len), np.nan)

            for i, s in enumerate(son_all):
                idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
                valid = (idx_array >= 0) & (idx_array < len(FF))
                idx_array = idx_array[valid]
                seg = FF[idx_array]
                mn_mat[i, :len(seg)] = seg

            if zscore_mode == "global":
                mn_mat = stats.zscore(mn_mat, axis=None, nan_policy="omit")
            elif zscore_mode == "trial":
                mn_mat = stats.zscore(mn_mat, axis=1, nan_policy="omit")
            elif zscore_mode == "baseline":
                baseline = mn_mat[:, :idx_bef]
                mu = np.nanmean(baseline, axis=1, keepdims=True)
                sd = np.nanstd(baseline, axis=1, keepdims=True)
                mn_mat = (mn_mat - mu) / sd

            masked = np.ma.masked_array(mn_mat, np.isnan(mn_mat))
            plt_mn = np.ma.mean(masked, axis=0)
            plt_sd = np.ma.std(masked, axis=0)

            if color is None:
                ax.fill_between(t, plt_mn - plt_sd, plt_mn + plt_sd, alpha=0.3)
                ax.plot(t, plt_mn, linewidth=1.5)
            else:
                ax.fill_between(t, plt_mn - plt_sd, plt_mn + plt_sd, color=color, alpha=0.3)
                ax.plot(t, plt_mn, color=color, linewidth=1.5)

            ax.axvline(0, color=markc, linestyle="--", linewidth=1)
            ax.set_title(f"{col}", color=markc, size=14)
            tag = "first" if first else "every"
        axs[0].set_ylabel("Z-score", color=markc, size=12)
        axs[0].set_xlabel("time (s)", color=markc, size=12)
        tag = "first" if first else "every"
        fname = f"{savename}_{'_'.join(cols)}_{tag}_{event_type}_{zscore_mode}.pdf"
        fig.savefig(os.path.join(out_folder, fname), bbox_inches="tight")
        plt.show()
        plt.close(fig)



def int_relative_activity(
    figure_folder,
    rois,
    pre_sec=1.0,
    odor_source="auto",   # "auto", "mfc2", "mfc3", "both"
    colors=None,
    max_bouts=None,
    exclude_nonpositive_baseline=True,
):
    """
    For each file:
      - Detect odor bouts (contiguous odor_on True) using odor_bout_indices().
      - For each bout [s,e) require >= pre_sec available before s; else exclude.
      - Integrate FF in-bout and in the pre_sec window immediately before bout.
      - Normalize: norm_int = I_in / I_pre  (division)
      - Relative activity per ROI: frac = norm_int / sum(norm_int across ROIs)
      - Plot mean frac (+/- SEM) per ROI, per file.

    Notes:
      - Integration is np.trapz(signal, time_segment) on the native timebase.
      - If baseline integral is ~0 (or <=0, if exclude_nonpositive_baseline),
        that ROI for that bout becomes NaN; if all ROIs invalid, bout is skipped.
    """
    n_rois = len(rois)
    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue

        savename, _ = os.path.splitext(filename)
        fullpath = os.path.join(figure_folder, filename)
        img_dict = open_pickle(fullpath)
        pv2, ft2 = process_dLight_pickle(img_dict, rois)
        time = ft2["relative_time"].to_numpy() if "relative_time" in ft2 else pv2["relative_time"].to_numpy()
        
        dt = np.nanmean(np.diff(time))

        pre_n = int(np.round(float(pre_sec) / dt))
        if pre_n < 1:
            pre_n = 1

        odor1 = (ft2["mfc2_stpt"].to_numpy() != 0).astype(int) if "mfc2_stpt" in ft2 else None
        odor2 = (ft2["mfc3_stpt"].to_numpy() != 0).astype(int) if "mfc3_stpt" in ft2 else None
        if odor_source == "auto":
            use = "mfc3" if "oct" in filename.lower() else "mfc2"
        else:
            use = odor_source
        if use == "mfc2":
            if odor1 is None:
                print(f"Skipping {filename}: no mfc2_stpt")
                continue
            odor_on = odor1.astype(bool)
        elif use == "mfc3":
            if odor2 is None:
                print(f"Skipping {filename}: no mfc3_stpt")
                continue
            odor_on = odor2.astype(bool)
        elif use == "both":
            if odor1 is None and odor2 is None:
                print(f"Skipping {filename}: no odor channels")
                continue
            if odor1 is None:
                odor_on = odor2.astype(bool)
            elif odor2 is None:
                odor_on = odor1.astype(bool)
            else:
                odor_on = ((odor1 != 0) | (odor2 != 0))
        else:
            raise ValueError("odor_source must be 'auto', 'mfc2', 'mfc3', or 'both'")

        bouts = exit_bout_indices(odor_on)  # list of (s,e) with e exclusive
        if len(bouts) == 0:
            print(f"No odor bouts found in file: {filename}")
            continue

        if max_bouts is not None:
            bouts = bouts[:max_bouts]
        fractions_per_bout = []
        for (s, e) in bouts:
            s = int(s); e = int(e)
            if s - pre_n < 0:
                continue

            pre_idx  = np.arange(s - pre_n, s, dtype=int)   # [s-pre_n, s)
            bout_idx = np.arange(s, e, dtype=int)           # [s, e)
            if bout_idx.size < 2:
                continue
            t_pre  = time[pre_idx]
            t_bout = time[bout_idx]
            norm_ints = np.full(n_rois, np.nan, dtype=float)
            for r_idx, roi in enumerate(rois):
                FF = pv2[f"0_{roi}"].to_numpy()
                pre_sig  = FF[pre_idx]
                bout_sig = FF[bout_idx]
                I_pre  = np.trapz(pre_sig,  t_pre)
                I_bout = np.trapz(bout_sig, t_bout)
                if not (np.isfinite(I_pre) and np.isfinite(I_bout)):
                    continue
                if exclude_nonpositive_baseline:
                    if I_pre <= 1e-12:
                        continue
                else:
                    if abs(I_pre) < 1e-12:
                        continue
                norm_ints[r_idx] = I_bout / I_pre
            if np.all(~np.isfinite(norm_ints)):
                continue

            denom = np.nansum(norm_ints)
            if (not np.isfinite(denom)) or (abs(denom) < 1e-12):
                continue
            fractions_per_bout.append(norm_ints / denom)

        if len(fractions_per_bout) == 0:
            print(f"{filename}: no valid bouts after requiring >= {pre_sec}s pre-bout.")
            continue

        frac_mat = np.vstack(fractions_per_bout)  # (n_valid_bouts, n_rois)
        frac_mean = np.nanmean(frac_mat, axis=0)
        frac_sem  = np.nanstd(frac_mat, axis=0) / np.sqrt(np.sum(np.isfinite(frac_mat), axis=0).clip(min=1))

        fig, ax, markc = configure_bw_plot(size=(max(4, 0.65 * n_rois + 2), 3),xaxis=True,ncols=1)
        x = np.arange(n_rois)
        frac_mean = np.nanmean(frac_mat, axis=0)
        frac_sem  = np.nanstd(frac_mat, axis=0) / np.sqrt(np.sum(np.isfinite(frac_mat), axis=0).clip(min=1))
        if colors is None:
            line_color = markc
            point_colors = [markc] * n_rois
        else:
            line_color = markc
            point_colors = [colors[i % len(colors)] for i in range(n_rois)]
        ax.plot(x, frac_mean, marker="o", linewidth=2.5, color=line_color, zorder=10)
        ax.errorbar(x, frac_mean, yerr=frac_sem,fmt="none", elinewidth=1.5, capsize=0,color=line_color, zorder=9)

        for i in range(n_rois):
            ax.scatter(x[i], frac_mean[i], s=60, color=point_colors[i], zorder=11)
        ax.set_xticks(x)
        ax.set_xticklabels(rois, rotation=45, ha="right", color=markc)
        ax.set_ylabel("relative activity", color=markc)
        ax.set_title(f"{savename}", color=markc)

        ax.set_ylim(0, 0.5)
        plt.tight_layout()
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/tone'):
            os.makedirs(f'{figure_folder}/fig/tone')
        fig.savefig(f'{figure_folder}/fig/tone/{filename}', bbox_inches='tight')



def int_relative_activity_avg(figure_folder, rois, pre_sec=1.0, odor_source="auto", event_type='entry', max_bouts=None, use_regression=True, metric='integral'):
    """
    metric : 'integral' - use trapz of bout signal (default)
             'peak'     - use nanmax of bout signal
    pre_sec : minimum bout duration (in seconds) required to keep a bout.
              Bouts shorter than pre_sec are excluded.
    """

    n_rois = len(rois)
    group_file_means = {"acv": [], "oct": []}
    fig, ax, markc = configure_white_plot(size=(4,3), xaxis=True, ncols=1)

    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue

        group = "acv" if "acv" in filename else "oct"
        fullpath = os.path.join(figure_folder, filename)
        img_dict = open_pickle(fullpath)
        pv2, ft2 = process_dLight_pickle(img_dict, rois)
        net_motion = ft2['net_motion'].to_numpy()

        time = ft2["relative_time"].to_numpy() if "relative_time" in ft2 else pv2["relative_time"].to_numpy()
        if time is None or len(time) < 3:
            print(f"Skipping {filename}: missing/short timebase"); continue
        dt = np.nanmean(np.diff(time))
        if not np.isfinite(dt) or dt <= 0:
            print(f"Skipping {filename}: bad timebase"); continue

        pre_idx = int(np.round(float(pre_sec) / dt))

        odor1 = (ft2["mfc3_stpt"].to_numpy() != 0).astype(int) if "mfc3_stpt" in ft2 else None
        odor2 = (ft2["mfc4_stpt"].to_numpy() != 0).astype(int) if "mfc4_stpt" in ft2 else None
        use = ("mfc4" if "oct" in filename else "mfc3") if odor_source == "auto" else odor_source

        if use == "mfc3":
            if odor1 is None: print(f"Skipping {filename}: no mfc2_stpt"); continue
            odor_on = odor1.astype(bool)
        elif use == "mfc4":
            if odor2 is None: print(f"Skipping {filename}: no mfc3_stpt"); continue
            odor_on = odor2.astype(bool)
        elif use == "both":
            if odor1 is None and odor2 is None: print(f"Skipping {filename}: no odor channels"); continue
            odor_on = ((odor1 != 0) | (odor2 != 0)) if (odor1 is not None and odor2 is not None) else \
                      (odor2 if odor1 is None else odor1).astype(bool)
        else:
            raise ValueError("odor_source must be 'auto', 'mfc2', 'mfc3', or 'both'")

        bouts = odor_bout_indices(odor_on) if event_type == 'entry' else exit_bout_indices(odor_on)
        if len(bouts) == 0:
            print(f"No odor bouts found in {filename}"); continue

        # exclude bouts shorter than pre_sec
        n_before = len(bouts)
        bouts = [(s, e) for (s, e) in bouts if (int(e) - int(s)) >= pre_idx]
        n_excluded = n_before - len(bouts)
        if n_excluded > 0:
            print(f"{filename}: excluding {n_excluded} bout(s) shorter than {pre_sec}s")
        if len(bouts) == 0:
            print(f"{filename}: no bouts remain after duration filtering"); continue

        if max_bouts is not None:
            bouts = bouts[:max_bouts]

        fractions_per_bout = []
        for (s, e) in bouts:
            s, e = int(s), int(e)
            bout_idx = np.arange(s, e, dtype=int)
            if bout_idx.size < 2:
                continue

            norm_ints = np.full(n_rois, np.nan, dtype=float)
            for r_idx, roi in enumerate(rois):
                FF = pv2[f'0_{roi}'].to_numpy()

                if use_regression:
                    reg = LinearRegression().fit(net_motion.reshape(-1, 1), FF)
                    FF = FF - reg.predict(net_motion.reshape(-1, 1))

                if metric == 'integral':
                    val = np.trapz(FF[bout_idx], time[bout_idx])
                else:  # peak
                    bout_time = time[bout_idx]
                    early_idx = bout_idx[bout_time - bout_time[0] <= 2.0]
                    val = np.nanmax(FF[early_idx]) if early_idx.size > 0 else np.nan

                if np.isfinite(val):
                    norm_ints[r_idx] = val

            if np.all(~np.isfinite(norm_ints)):
                continue
            if metric == 'integral':
                if event_type == 'exit':
                    denom = np.nansum(np.abs(norm_ints))
                elif event_type == 'entry':
                    denom = np.nansum(np.abs(norm_ints))
                if not np.isfinite(denom) or abs(denom) < 1e-12:
                    continue
                fractions_per_bout.append(norm_ints / denom)
            else:  # peak
                fractions_per_bout.append(norm_ints)

        if len(fractions_per_bout) == 0:
            print(f"{filename}: no valid bouts"); continue

        group_file_means[group].append(np.nanmean(np.vstack(fractions_per_bout), axis=0))

    acv_c, oct_c = '#fbceb1', '#7ed4e6'
    x = np.arange(n_rois)

    def plot_group(group_name, c):
        mats = group_file_means[group_name]
        if len(mats) == 0:
            return
        mat   = np.vstack(mats)
        gmean = np.nanmean(mat, axis=0)
        gsem  = np.nanstd(mat, axis=0) / np.sqrt(np.sum(np.isfinite(mat), axis=0).clip(min=1))
        for row in mat:
            ax.scatter(x, row, s=22, color=c, alpha=0.35, zorder=3)
        ax.plot(x, gmean, marker="o", linewidth=2.5, color=c, zorder=10, label=group_name)
        ax.errorbar(x, gmean, yerr=gsem, fmt="none", elinewidth=1.5, capsize=3, color=c, zorder=9)

    plot_group("acv", acv_c)
    plot_group("oct", oct_c)

    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=45, ha="right", color=markc)
    ylabel = "relative activity (integral)" if metric == 'integral' else "peak ΔF/F"
    ax.set_ylabel(ylabel, color=markc)
    ax.set_title(f'{metric} ΔF/F at {event_type}', color=markc)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # --- save ---
    name = 'et_vs_replay_entries' if event_type == 'entry' else 'et_vs_replay_exits'
    save_dir = os.path.join(figure_folder, 'fig', 'tone')
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{name}_{metric}.pdf'), bbox_inches='tight')


_trapz = getattr(np, 'trapezoid', None) or np.trapz

def grouped_odor_response(folder, groups, event_type='entry', min_bout_sec=1.0, use_regression=True,
                          metric='mean', peak_window=2.0, fly_id_fn=None):
    """groups: e.g. {'g2+g3': ['g2','g3'], 'g4+g5': ['g4','g5']}
    Each ROI is z-scored using only odor-free frames (both odor channels off). The z-trace is
    reduced per bout (metric: 'mean', 'integral' = trapz, or 'peak' = max within the first
    peak_window s of the bout), then averaged across the group's ROIs.
    One paired panel per group (acv vs oct, line per fly), all in one figure."""
    assert metric in ('mean', 'integral', 'peak')
    rois = sorted({r for rs in groups.values() for r in rs})
    rows = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith('.pkl'):
            continue
        odor = 'acv' if 'acv' in f else 'oct'
        fly = fly_id_fn(f) if fly_id_fn else '_'.join(
            t for t in re.split(r'[_\-\s]+', f[:-4]) if t.lower() not in ('acv', 'oct', 'et', 'replay'))
        pv, ft = process_dLight_pickle(open_pickle(os.path.join(folder, f)), rois)
        t = (ft if 'relative_time' in ft else pv)['relative_time'].to_numpy()
        dt = np.nanmean(np.diff(t))
        motion = ft['net_motion'].to_numpy().reshape(-1, 1)

        on = {k: ft[k].to_numpy() != 0 for k in ('mfc3_stpt', 'mfc4_stpt') if k in ft}
        odor_on = on['mfc4_stpt' if odor == 'oct' else 'mfc3_stpt']
        base = ~np.any(list(on.values()), axis=0)
        bouts = (odor_bout_indices if event_type == 'entry' else exit_bout_indices)(odor_on)
        bouts = [(int(s), int(e)) for s, e in bouts if (e - s) * dt >= min_bout_sec]

        def resp(z, s, e):
            if metric == 'mean':
                return z[s:e].mean()
            if metric == 'integral':
                return _trapz(z[s:e], t[s:e])
            return np.nanmax(z[s:min(e, s + int(peak_window / dt))])

        z = {}
        for r in rois:
            ff = pv[f'0_{r}'].to_numpy()
            if use_regression:
                ff = ff - LinearRegression().fit(motion, ff).predict(motion)
            z[r] = (ff - np.nanmean(ff[base])) / np.nanstd(ff[base])

        for g, g_rois in groups.items():
            vals = [np.mean([resp(z[r], s, e) for r in g_rois]) for s, e in bouts]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                rows.append(dict(fly=fly, odor=odor, group=g, value=np.mean(vals)))

    df = pd.DataFrame(rows).groupby(['fly', 'odor', 'group'], as_index=False).value.mean()
    print('recordings per fly/odor (a fly needs both acv and oct to be plotted):')
    print(df.groupby(['fly', 'odor']).size().unstack(fill_value=0))

    colors = {'acv': '#fbceb1', 'oct': '#7ed4e6'}
    fig, axs, markc = configure_white_plot(size=(3 * len(groups), 5), xaxis=True, ncols=len(groups))
    axs = np.atleast_1d(axs)

    for ax, g in zip(axs, groups):
        w = df[df.group == g].pivot(index='fly', columns='odor', values='value').dropna()
        p = stats.wilcoxon(w.acv, w.oct).pvalue
        ax.axhline(0, color=markc, ls='--', lw=.8)
        ax.plot([0, 1], w[['acv', 'oct']].T, color=markc, alpha=.4, lw=1)
        for x, o in enumerate(['acv', 'oct']):
            # ax.scatter(np.full(len(w), x), w[o], color=colors[o], s=26, zorder=3)
            ax.vlines(x, w[o].mean() - w[o].sem(), w[o].mean() + w[o].sem(), color=markc, lw=2, zorder=5)
        ax.plot([0, 1], [w.acv.mean(), w.oct.mean()], color=markc, lw=2, zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['acv', 'oct'], color=markc); ax.set_xlim(-.5, 1.5)
        ax.set_ylabel('Δ activity (z-score·s)' if metric == 'integral' else f'{metric} activity (z-score)', color=markc)
        ax.set_title(f'{g}  p={p:.3f} (n={len(w)})', color=markc)
        print(g, w.mean().round(4).to_dict(), f'paired Wilcoxon p={p:.4f}, n={len(w)}')

    lo, hi = min(a.get_ylim()[0] for a in axs), max(a.get_ylim()[1] for a in axs)
    for a in axs:
        a.set_ylim(lo, hi)

    name = f'acv_vs_oct_{event_type}_{metric}_regressed' if use_regression == True else f'acv_vs_oct_{event_type}_{metric}'
    if not os.path.exists(f'{folder}/fig/odor_comps'):
        os.makedirs(f'{folder}/fig/odor_comps')
    fig.savefig(f'{folder}/fig/odor_comps/{name}.pdf', bbox_inches='tight')
    plt.show()

    

def DAN_corr_return_efficiency(figure_folder,roi_num, pre_sec=1.0,odor_source="auto", exclude_nonpositive_baseline=True):
    roiA, roiB = roi_num[0], roi_num[1]

    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue

        file_path = os.path.join(figure_folder, filename)
        img_dict = open_pickle(file_path)
        pv2, ft2 = process_dLight_pickle(img_dict, [roiA, roiB])

        time = pv2["relative_time"].to_numpy()

        if time.size < 3 or not np.all(np.isfinite(time)):
            print(f"Skipping {filename}: bad timebase")
            continue

        dt = np.nanmean(np.diff(time))
        if not np.isfinite(dt) or dt <= 0:
            print(f"Skipping {filename}: bad dt")
            continue

        pre_n = int(np.round(float(pre_sec) / dt))
        pre_n = max(pre_n, 1)

        odor1 = (ft2["mfc2_stpt"].to_numpy() != 0).astype(int) if "mfc2_stpt" in ft2 else None
        odor2 = (ft2["mfc3_stpt"].to_numpy() != 0).astype(int) if "mfc3_stpt" in ft2 else None

        if odor_source == "auto":
            use = "mfc3" if "oct" in filename.lower() else "mfc2"
        else:
            use = odor_source

        if use == "mfc2":
            if odor1 is None:
                print(f"Skipping {filename}: no mfc2_stpt")
                continue
            odor_on = odor1.astype(bool)
        elif use == "mfc3":
            if odor2 is None:
                print(f"Skipping {filename}: no mfc3_stpt")
                continue
            odor_on = odor2.astype(bool)
        elif use == "both":
            if odor1 is None and odor2 is None:
                print(f"Skipping {filename}: no odor channels")
                continue
            if odor1 is None:
                odor_on = odor2.astype(bool)
            elif odor2 is None:
                odor_on = odor1.astype(bool)
            else:
                odor_on = ((odor1 != 0) | (odor2 != 0))
        else:
            raise ValueError("odor_source must be 'auto', 'mfc2', 'mfc3', or 'both'")

        in_bouts  = odor_bout_indices(odor_on)   # list of (s,e), e exclusive
        out_bouts = exit_bout_indices(odor_on)   # list of (s,e), e exclusive

        if len(in_bouts) == 0 or len(out_bouts) == 0:
            print(f"Skipping {filename}: no bouts found")
            continue

        out_by_start = {int(s): (int(s), int(e)) for s, e in out_bouts}

        FF_A = pv2[f"0_{roiA}"].to_numpy()
        FF_B = pv2[f"0_{roiB}"].to_numpy()

        x_eff = []
        y_ratio = []

        for in_s, in_e in in_bouts:
            in_s = int(in_s); in_e = int(in_e)

            # Need pre_sec available before in_s
            if in_s - pre_n < 0:
                continue

            # Find the OUT bout immediately after this IN bout
            nxt_out = out_by_start.get(in_e, None)
            if nxt_out is None:
                continue
            out_s, out_e = nxt_out
            if out_e - out_s < 2:
                continue

            # --- compute return efficiency on OUT bout trajectory ---
            out_df = ft2.iloc[out_s:out_e].copy()

            furthest = get_a_bout_calc(out_df, "furthest_distance_from_plume")
            pathlen  = get_a_bout_calc(out_df, "path_length")

            if not (np.isfinite(furthest) and np.isfinite(pathlen)) or pathlen <= 1e-12:
                continue

            return_eff = float(furthest) / float(pathlen)

            # --- compute normalized integrated activity ratio on IN bout ---
            pre_idx  = np.arange(in_s - pre_n, in_s, dtype=int)
            bout_idx = np.arange(in_s, in_e, dtype=int)
            if bout_idx.size < 2:
                continue

            t_pre  = time[pre_idx]
            t_bout = time[bout_idx]

            Ipre_A  = np.trapz(FF_A[pre_idx],  t_pre)
            Ibout_A = np.trapz(FF_A[bout_idx], t_bout)
            Ipre_B  = np.trapz(FF_B[pre_idx],  t_pre)
            Ibout_B = np.trapz(FF_B[bout_idx], t_bout)

            if not all(np.isfinite(v) for v in [Ipre_A, Ibout_A, Ipre_B, Ibout_B]):
                continue

            if exclude_nonpositive_baseline:
                if (Ipre_A <= 1e-12) or (Ipre_B <= 1e-12):
                    continue
            else:
                if (abs(Ipre_A) < 1e-12) or (abs(Ipre_B) < 1e-12):
                    continue

            normA = Ibout_A / Ipre_A
            normB = Ibout_B / Ipre_B

            if not (np.isfinite(normA) and np.isfinite(normB)) or abs(normB) < 1e-12:
                continue

            ratio = float(normB / normA)

            x_eff.append(return_eff)
            y_ratio.append(ratio)

        fig, ax, markc = configure_bw_plot(size=(5, 5), xaxis=True, ncols=1)

        ax.scatter(x_eff, y_ratio, s=20, alpha=0.8)
        ax.set_title(f"{roiB}/{roiA} normalized AUC ratio vs next OUT return efficiency",
                     color=markc, fontsize=10)
        ax.set_xlabel("return efficiency (furthest distance / path length)", fontsize=12)
        ax.set_ylabel(f"normalized integrated activity ratio: {roiA}/{roiB}", fontsize=12)

        # regression
        if len(x_eff) >= 2:
            x = np.asarray(x_eff, dtype=float)
            y = np.asarray(y_ratio, dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            n_valid = np.count_nonzero(valid)
            if n_valid >= 2:
                xv = x[valid]
                yv = y[valid]
                if (np.nanstd(xv) > 0) and (np.nanstd(yv) > 0):
                    r, p = pearsonr(xv, yv)
                else:
                    r, p = np.nan, np.nan

                # line fit (optional; keep your existing polyfit)
                slope, intercept = np.polyfit(xv, yv, 1)
                x_fit = np.linspace(xv.min(), xv.max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, linewidth=2)

                # R^2 from the same fit (your existing code)
                y_pred = slope * xv + intercept
                ss_res = np.sum((yv - y_pred) ** 2)
                ss_tot = np.sum((yv - np.mean(yv)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                ax.text(
                    0.05, 0.95,
                    f"slope = {slope:.3g}\nR² = {r2:.3f}\nPearson r = {r:.3f}\np = {p:.3g}\n(n={n_valid})",
                    transform=ax.transAxes,
                    va="top"
                )
            # if np.count_nonzero(valid) >= 2:
            #     slope, intercept = np.polyfit(x[valid], y[valid], 1)
            #     x_fit = np.linspace(x[valid].min(), x[valid].max(), 100)
            #     y_fit = slope * x_fit + intercept
            #     ax.plot(x_fit, y_fit, linewidth=2)

            #     y_pred = slope * x[valid] + intercept
            #     ss_res = np.sum((y[valid] - y_pred) ** 2)
            #     ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
            #     r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            #     ax.text(
            #         0.05, 0.95,
            #         f"slope = {slope:.3g}\nR² = {r2:.3f}\n(n={np.count_nonzero(valid)})",
            #         transform=ax.transAxes,
            #         va="top"
            #     )

        fig.suptitle(filename)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        outdir = os.path.join(figure_folder, "fig", "corr")
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"{filename}_returnEff_vs_ratio.pdf"), bbox_inches="tight")
        plt.close(fig)


def et_compartment_corrs(figure_folder, window_s=10, rois=['g2','g3'], colors=['b','k']):
    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue
        savename, _ = os.path.splitext(filename)
        fullpath = os.path.join(figure_folder, filename)
        img_dict = open_pickle(fullpath)
        pv2, ft2 = process_dLight_pickle(img_dict, rois)

        FF_g0 = pv2[f'0_{rois[0]}'].to_numpy()
        FF_g1 = pv2[f'0_{rois[1]}'].to_numpy()
        time  = pv2['relative_time'].to_numpy()

        odor1_on = ft2['mfc2_stpt'].to_numpy()
        odor2_on = ft2['mfc3_stpt'].to_numpy()
        if odor1_on is None and odor2_on is None:
            in_odor = np.zeros_like(FF_g0, dtype=bool)
        else:
            in_odor = np.zeros_like(FF_g0, dtype=bool)
            if odor1_on is not None:
                in_odor |= (odor1_on > 0)
            if odor2_on is not None:
                in_odor |= (odor2_on > 0)
        out_odor = ~in_odor

        fig, axs, markc = configure_bw_plot(size=(10, 5), xaxis=True, nrows=2)
        fig.suptitle(filename, color=markc)

        ax_traces = axs[0]
        ax_corr   = axs[1]

        ax_traces.plot(time, FF_g0, linewidth=0.75, color=colors[0], label=f'{rois[0]}')
        ax_traces.plot(time, FF_g1, linewidth=0.75, color=colors[1], label=f'{rois[1]}')
        shade_on_intervals(ax_traces, time, odor1_on, alpha=0.3, color='#fbceb1')
        shade_on_intervals(ax_traces, time, odor2_on, alpha=0.3, color='#7ed4e6')
        ax_traces.set_ylabel('dF/F', color=markc)
        ax_traces.legend(frameon=False)
        t_corr, r_corr = sliding_pearson(FF_g0, FF_g1, time, window_s, step_s=1)
        ax_corr.plot(t_corr, r_corr, color=markc, linewidth=1)
        ax_corr.axhline(0, linestyle='--', color='gray', linewidth=1)
        shade_on_intervals(ax_corr, time, odor1_on, alpha=0.2, color='#fbceb1')
        shade_on_intervals(ax_corr, time, odor2_on, alpha=0.2, color='#7ed4e6')
        ax_corr.set_ylabel('Pearson r')
        ax_corr.set_xlabel('time (s)')
        ax_corr.set_ylim(-1, 1)
        finite = np.isfinite(FF_g0) & np.isfinite(FF_g1)
        mask_in = in_odor & finite
        if mask_in.sum() >= 3:
            r_in, _ = pearsonr(zscore(FF_g0[mask_in], nan_policy='omit'),zscore(FF_g1[mask_in], nan_policy='omit'))
        
        mask_out = out_odor & finite
        if mask_out.sum() >= 3:
            r_out, _ = pearsonr(zscore(FF_g0[mask_out], nan_policy='omit'),zscore(FF_g1[mask_out], nan_policy='omit'))

        FF_g0_z = zscore(FF_g0, nan_policy='omit')
        FF_g1_z = zscore(FF_g1, nan_policy='omit')
        r, p = pearsonr(FF_g0_z, FF_g1_z)
        ax_corr.set_title(f'in: r={r_in:.3f} | out: r={r_out:.3f} | overall Pearson r={r:.3f}', color=markc, fontsize=10)

        plt.tight_layout()
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/corr'):
            os.makedirs(f'{figure_folder}/fig/corr')
        fig.savefig(f'{figure_folder}/fig/corr/{savename}_{rois[0]}_{rois[1]}_.pdf', bbox_inches='tight')


def et_compartment_corrs_afs(figure_folder,window_s=10,rois=["g2", "g3"],colors=["b", "k"],size=(14,6),step_s=1,):
    paired_files = {}
    for filename in os.listdir(figure_folder):
        if not filename.endswith(".pkl"):
            continue
        if "_et_" in filename:
            prefix = filename.split("_et_")[0]
            paired_files.setdefault(prefix, {})["et"] = filename
        elif "_replay_" in filename:
            prefix = filename.split("_replay_")[0]
            paired_files.setdefault(prefix, {})["replay"] = filename

    outdir = os.path.join(figure_folder, "fig", "corr")
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    for fly_id, files in sorted(paired_files.items()):
        if "et" not in files or "replay" not in files:
            print(f"Skipping {fly_id}: missing pair")
            continue

        data_et = open_pickle(os.path.join(figure_folder, files["et"]))
        data_rp = open_pickle(os.path.join(figure_folder, files["replay"]))
        pv2_et = data_et['strip1']["pv2"]
        pv2_rp = data_rp['strip1']["pv2"]

        FF_et_0 = pv2_et[rois[0]].to_numpy()
        FF_et_1 = pv2_et[rois[1]].to_numpy()
        t_et    = pv2_et['relative_time'].to_numpy()

        FF_rp_0 = pv2_rp[rois[0]].to_numpy()
        FF_rp_1 = pv2_rp[rois[1]].to_numpy()
        t_rp    = pv2_rp['relative_time'].to_numpy()

        odor1_et = pv2_et['mfc2_stpt'].to_numpy()
        odor2_et = pv2_et['mfc3_stpt'].to_numpy()
        odor1_rp = pv2_rp['mfc2_stpt'].to_numpy()
        odor2_rp = pv2_rp['mfc3_stpt'].to_numpy()

        in_et = np.zeros_like(FF_et_0, dtype=bool)
        if odor1_et is not None:
            in_et |= (odor1_et > 0)
        if odor2_et is not None:
            in_et |= (odor2_et > 0)
        out_et = ~in_et

        in_rp = np.zeros_like(FF_rp_0, dtype=bool)
        if odor1_rp is not None:
            in_rp |= (odor1_rp > 0)
        if odor2_rp is not None:
            in_rp |= (odor2_rp > 0)
        out_rp = ~in_rp

        fig, axs, markc = configure_bw_plot(size=size, xaxis=True, nrows=2, ncols=2)
        axs = np.array(axs).reshape(2, 2)
        fig.suptitle(fly_id, color=markc)

        def _panel(col, label, t, FF0, FF1, odor1_on, odor2_on, in_mask, out_mask):
            ax_tr = axs[0, col]
            ax_cr = axs[1, col]

            ax_tr.set_title(label, color=markc)
            ax_tr.plot(t, FF0, linewidth=0.75, color=colors[0], label=f"{rois[0]}")
            ax_tr.plot(t, FF1, linewidth=0.75, color=colors[1], label=f"{rois[1]}")
            shade_on_intervals(ax_tr, t, odor1_on, alpha=0.30, color="#fbceb1")
            shade_on_intervals(ax_tr, t, odor2_on, alpha=0.30, color="#7ed4e6")
            ax_tr.set_ylabel("dF/F", color=markc)
            ax_tr.legend(frameon=False)

            t_corr, r_corr = sliding_pearson(FF0, FF1, t, window_s, step_s=step_s)
            ax_cr.plot(t_corr, r_corr, color=markc, linewidth=1)
            ax_cr.axhline(0, linestyle="--", color="gray", linewidth=1)
            shade_on_intervals(ax_cr, t, odor1_on, alpha=0.20, color="#fbceb1")
            shade_on_intervals(ax_cr, t, odor2_on, alpha=0.20, color="#7ed4e6")
            ax_cr.set_ylabel("Pearson r", color=markc)
            ax_cr.set_xlabel("time (s)", color=markc)
            ax_cr.set_ylim(-1, 1)

            finite = np.isfinite(FF0) & np.isfinite(FF1)

            r_in = np.nan
            r_out = np.nan

            mask_in = in_mask & finite
            if mask_in.sum() >= 3:
                r_in, _ = pearsonr(
                    zscore(FF0[mask_in], nan_policy="omit"),
                    zscore(FF1[mask_in], nan_policy="omit"),
                )

            mask_out = out_mask & finite
            if mask_out.sum() >= 3:
                r_out, _ = pearsonr(
                    zscore(FF0[mask_out], nan_policy="omit"),
                    zscore(FF1[mask_out], nan_policy="omit"),
                )

            mask = np.isfinite(FF_g0) & np.isfinite(FF_g1)
            x = FF_g0[mask]
            y = FF_g1[mask]

            # zscore only after filtering, and only if variance is non-zero
            if x.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
                r, p = np.nan, np.nan
            else:
                xz = (x - np.mean(x)) / np.std(x)
                yz = (y - np.mean(y)) / np.std(y)
                r, p = pearsonr(xz, yz)

            ax_cr.set_title(
                f"in: r={r_in:.3f} | out: r={r_out:.3f} | overall r={r_all:.3f}",
                color=markc,
                fontsize=10,
            )

        _panel(0, "ET", t_et, FF_et_0, FF_et_1, odor1_et, odor2_et, in_et, out_et)
        _panel(1, "Replay", t_rp, FF_rp_0, FF_rp_1, odor1_rp, odor2_rp, in_rp, out_rp)

        plt.tight_layout()
        savename = f"{fly_id}_{rois[0]}_{rois[1]}_et_vs_replay_corr.pdf"
        fig.savefig(os.path.join(outdir, savename), bbox_inches="tight")
        plt.show()
        plt.close(fig)

def activity_corrs(figure_folder, window_s=10, neuron='fsbtn', var='net_motion', colors=['b','k']):
    for filename in sorted(os.listdir(figure_folder)):
        if not filename.endswith(".pkl"):
            continue
        savename, _ = os.path.splitext(filename)
        fullpath = os.path.join(figure_folder, filename)
        img_dict = open_pickle(fullpath)
        pv2, ft2 = process_pickle(img_dict, neuron)

        FF_g0 = ft2[var].to_numpy()
        FF_g1 = pv2[f'0_{neuron}'].to_numpy()
        time  = pv2['relative_time'].to_numpy()

        odor1_on = ft2['mfc2_stpt'].to_numpy()
        odor2_on = ft2['mfc3_stpt'].to_numpy()
        if odor1_on is None and odor2_on is None:
            in_odor = np.zeros_like(FF_g0, dtype=bool)
        else:
            in_odor = np.zeros_like(FF_g0, dtype=bool)
            if odor1_on is not None:
                in_odor |= (odor1_on > 0)
            if odor2_on is not None:
                in_odor |= (odor2_on > 0)
        out_odor = ~in_odor

        fig, axs, markc = configure_bw_plot(size=(10, 5), xaxis=True, nrows=2)
        fig.suptitle(filename, color=markc)

        ax_traces = axs[0]
        ax_corr   = axs[1]

        ax_traces.plot(time, FF_g0, linewidth=0.75, color=colors[0], label=var)
        ax_traces.plot(time, FF_g1, linewidth=0.75, color=colors[1], label=neuron)
        shade_on_intervals(ax_traces, time, odor1_on, alpha=0.3, color='#fbceb1')
        shade_on_intervals(ax_traces, time, odor2_on, alpha=0.3, color='#7ed4e6')
        ax_traces.set_ylabel('dF/F', color=markc)
        ax_traces.legend(frameon=False)
        t_corr, r_corr = sliding_pearson(FF_g0, FF_g1, time, window_s, step_s=1)
        ax_corr.plot(t_corr, r_corr, color=markc, linewidth=1)
        ax_corr.axhline(0, linestyle='--', color='gray', linewidth=1)
        shade_on_intervals(ax_corr, time, odor1_on, alpha=0.2, color='#fbceb1')
        shade_on_intervals(ax_corr, time, odor2_on, alpha=0.2, color='#7ed4e6')
        ax_corr.set_ylabel('Pearson r')
        ax_corr.set_xlabel('time (s)')
        ax_corr.set_ylim(-1, 1)
        finite = np.isfinite(FF_g0) & np.isfinite(FF_g1)
        mask_in = in_odor & finite
        if mask_in.sum() >= 3:
            r_in, _ = pearsonr(zscore(FF_g0[mask_in], nan_policy='omit'),zscore(FF_g1[mask_in], nan_policy='omit'))
        
        mask_out = out_odor & finite
        if mask_out.sum() >= 3:
            r_out, _ = pearsonr(zscore(FF_g0[mask_out], nan_policy='omit'),zscore(FF_g1[mask_out], nan_policy='omit'))

        FF_g0_z = zscore(FF_g0, nan_policy='omit')
        FF_g1_z = zscore(FF_g1, nan_policy='omit')
        r, p = pearsonr(FF_g0_z, FF_g1_z)
        ax_corr.set_title(f'in: r={r_in:.3f} | out: r={r_out:.3f} | overall Pearson r={r:.3f}', color=markc, fontsize=10)

        plt.tight_layout()
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/corr'):
            os.makedirs(f'{figure_folder}/fig/corr')
        fig.savefig(f'{figure_folder}/fig/corr/{savename}_{rois[0]}_{rois[1]}_.pdf', bbox_inches='tight')


def triggered_binned_zFF(figure_folder, neuron, tbef=30, taf=30, event_type='entry', first=True, bin_size=None):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    else:
        color = 'white'  # fallback

    all_mn_mat = []
    max_len = 0

    for filename in os.listdir(figure_folder):
        if not filename.endswith('.pkl'):
            continue

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

        # detect entries / exits
        if event_type == 'entry':
            son = np.where((td[:-1] == False) & (td[1:] == True))[0]
        elif event_type == 'exit':
            son = np.where((td[:-1] == True) & (td[1:] == False))[0]
        else:
            raise ValueError("event_type must be 'entry' or 'exit'")

        if len(son) == 0:
            print(f"No {event_type} events found in file: {filename}")
            continue

        print(f"Found {len(son)} {event_type} events in file: {filename}")

        if first:
            son = son[:1]

        tinc = np.mean(np.diff(pv2['relative_time']))
        idx_bef = int(np.round(float(tbef) / tinc))
        idx_af = int(np.round(float(taf) / tinc))

        ca = FF

        # matrix: events × time_window, initialized with NaNs
        mn_mat = np.full((len(son), idx_bef + idx_af + 1), np.nan)

        for i, s in enumerate(son):
            idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
            idx_array = idx_array[(idx_array >= 0) & (idx_array < len(ca))]

            segment = ca[idx_array]
            mn_mat[i, :len(segment)] = segment  # pad with NaNs automatically

        # NOTE: don't z-score here per file if we're going to bin across files
        all_mn_mat.append(mn_mat)
        max_len = max(max_len, mn_mat.shape[1])

    if len(all_mn_mat) == 0:
        print(f"No {event_type} events found in any file.")
        return

    # ---- pad to same length and stack into one big matrix ----
    padded_mn_mat = []
    for mat in all_mn_mat:
        if mat.shape[1] < max_len:
            padding = np.full((mat.shape[0], max_len - mat.shape[1]), np.nan)
            mat = np.hstack((mat, padding))
        padded_mn_mat.append(mat)

    combined_mn_mat = np.concatenate(padded_mn_mat, axis=0)   # shape: n_events × max_len

    # ---- global Z-score normalization over all values (ignoring NaNs) ----
    combined_z = stats.zscore(combined_mn_mat, axis=None, nan_policy='omit')

    # axis=None returns same shape; just to be safe:
    combined_z = combined_z.reshape(combined_mn_mat.shape)

    t = np.linspace(-tbef, taf, max_len)

    # folder for saving
    figure_folder = f'{figure_folder}/fig/triggered_zFF'
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

    # ======================================================================
    #  CASE 1: binning into groups of `bin_size` events (your new behavior)
    # ======================================================================
    if bin_size is not None:
        total_events = combined_z.shape[0]
        num_bins = total_events // bin_size  # floor division

        if num_bins == 0:
            print(f"Not enough events ({total_events}) to make even one bin of {bin_size}.")
            return

        use_events = num_bins * bin_size
        print(f"Total events: {total_events}, using first {use_events} "
              f"events in {num_bins} bins of {bin_size}.")

        combined_z = combined_z[:use_events, :]

        # setup subplots: one per bin of 5 entries/exits
        fig, axes = plt.subplots(num_bins, 1, figsize=(4, 3 * num_bins), sharex=True)
        if num_bins == 1:
            axes = [axes]

        for b in range(num_bins):
            start = b * bin_size
            end = (b + 1) * bin_size

            bin_mat = combined_z[start:end, :]  # shape: bin_size × max_len
            masked_bin = np.ma.masked_array(bin_mat, np.isnan(bin_mat))

            plt_mn = np.ma.mean(masked_bin, axis=0)
            std = np.ma.std(masked_bin, axis=0)

            ax = axes[b]
            ax.fill_between(t, plt_mn + std, plt_mn - std, color=color, alpha=0.3)
            ax.plot(t, plt_mn, color=color)

            ax.set_ylim(-2, 8)
            ax.plot([0, 0], [-2, 8], color=color, linestyle='--')

            ax.set_ylabel('Z-score', color=color, size=10)
            ax.set_title(
                f'{event_type} events {start + 1}–{end}',
                color='white', size=12
            )

        axes[-1].set_xlabel('time (s)', color='white', size=16)

        fig.tight_layout()

        title_prefix = f"binned_{event_type}_by_{bin_size}"
        if first:
            fig.suptitle(f"first {event_type}, binned by {bin_size}", color='white', size=16, y=1.02)
        else:
            fig.suptitle(f"every {event_type}, binned by {bin_size}", color='white', size=16, y=1.02)

        fname = f'{figure_folder}/{title_prefix}_Zscored.pdf'
        fig.savefig(fname, bbox_inches='tight')
        plt.show()
        return

    # ======================================================================
    #  CASE 2: original behavior – average over all events in one trace
    # ======================================================================
    masked_combined_mn_mat = np.ma.masked_array(combined_z, np.isnan(combined_z))
    plt_mn = np.ma.mean(masked_combined_mn_mat, axis=0)
    std = np.ma.std(masked_combined_mn_mat, axis=0)

    fig, axs, markc = configure_bw_plot(size=(4, 6), xaxis=True)

    plt.fill_between(t, plt_mn + std, plt_mn - std, color=color, alpha=0.3)
    plt.plot(t, plt_mn, color=color)

    plt.ylim(-2, 8)
    plt.plot([0, 0], [-2, 8], color='white', linestyle='--')
    plt.xlabel('time (s)', color='white', size=16)
    plt.ylabel('Z-score', color='white', size=16)

    if first:
        plt.title(f"first {event_type}", color='white', size=16)
        fig.savefig(f'{figure_folder}/ffirst_{event_type}_Zscored.pdf', bbox_inches='tight')
    else:
        plt.title(f"every {event_type}", color='white', size=16)
        fig.savefig(f'{figure_folder}/every_{event_type}_Zscored.pdf', bbox_inches='tight')

    plt.show()

def triggered_zFF_octvsacv(figure_folder, neuron, tbef=30, taf=30, event_type='entry'):
    if neuron == 'mbon09':
        color = '#bd33a4'
    elif neuron == 'mbon21':
        color = '#ff4f00'
    elif neuron == 'mbon30':
        color = '#00cc99'
    else:
        color = 'blue'

    events = {'mfc3': [],'mfc2': []}

    for filename in os.listdir(figure_folder):
        if not filename.endswith('.pkl'):
            continue

        img_dict = open_pickle(os.path.join(figure_folder, filename))
        pv2, ft2 = process_pickle(img_dict, neuron)

        if pv2 is None or ft2 is None:
            print(f"Skipping file {filename}, missing data.")
            continue

        FF = ft2[neuron].to_numpy()
        td = ft2['instrip'].to_numpy()
        tvals = ft2['relative_time'].to_numpy()

        mfc3_state = ft2['mfc3_stpt'].to_numpy()
        mfc2_state = ft2['mfc2_stpt'].to_numpy()
        led_state  = ft2['led1_stpt'].to_numpy()

        # Identify entry/exits
        if event_type == 'entry':
            son = np.where((td[:-1] == False) & (td[1:] == True))[0]
        else:
            son = np.where((td[:-1] == True) & (td[1:] == False))[0]

        if len(son) == 0:
            continue

        tinc = np.mean(np.diff(tvals))
        idx_bef = int(np.round(tbef / tinc))
        idx_af  = int(np.round(taf  / tinc))
        window_len = idx_bef + idx_af + 1

        # Extract triggered segments
        for s in son:
            idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
            idx_array = idx_array[(idx_array >= 0) & (idx_array < len(FF))]
            segment = FF[idx_array]

            led_on = (led_state[s] == 0)  # LED1_ON = 0

            # MFC3
            if mfc3_state[s] > 0:
                events['mfc3'].append((segment, led_on, tvals[s]))

            # MFC2
            if mfc2_state[s] > 0:
                events['mfc2'].append((segment, led_on, tvals[s]))


    def categorize_mfc(event_list):
        if len(event_list) == 0:
            return {'first5': [], 'led_on': [], 'last5': []}

        # Sort by time before slicing
        event_list = sorted(event_list, key=lambda x: x[2])

        # Split
        first5 = event_list[:5]
        last5  = event_list[-5:]
        led_on = [e for e in event_list if e[1]]

        return {
            'first5': first5,
            'led_on': led_on,
            'last5': last5
        }

    cats = {
        'mfc3': categorize_mfc(events['mfc3']),
        'mfc2': categorize_mfc(events['mfc2'])
    }

    def build_stats(event_list):
        if len(event_list) == 0:
            return None, None, None

        segs = [e[0] for e in event_list]
        max_len = max([len(s) for s in segs])

        mats = []
        for s in segs:
            pad = max_len - len(s)
            if pad > 0:
                s = np.concatenate([s, np.full(pad, np.nan)])
            mats.append(s)

        mats = np.vstack(mats)
        mats = stats.zscore(mats, axis=None, nan_policy='omit')

        t = np.linspace(-tbef, taf, max_len)
        mean = np.nanmean(mats, axis=0)
        std  = np.nanstd(mats, axis=0)
        return t, mean, std

    plot_data = {
        ('mfc3','first5'): build_stats(cats['mfc3']['first5']),
        ('mfc3','led_on'): build_stats(cats['mfc3']['led_on']),
        ('mfc3','last5'):  build_stats(cats['mfc3']['last5']),

        ('mfc2','first5'): build_stats(cats['mfc2']['first5']),
        ('mfc2','led_on'): build_stats(cats['mfc2']['led_on']),
        ('mfc2','last5'):  build_stats(cats['mfc2']['last5']),
    }

    fig, ax_dummy, markc = configure_bw_plot(size=(10, 6), xaxis=True)
    fig.clear()
    axs = fig.subplots(2, 3, sharey=True)

    titles = {
        'first5': 'First 5 events',
        'led_on': 'LED1 ON (0)',
        'last5': 'Last 5 events'
    }

    order = [
        ('mfc3','first5',0,0),
        ('mfc3','led_on',0,1),
        ('mfc3','last5', 0,2),
        ('mfc2','first5',1,0),
        ('mfc2','led_on',1,1),
        ('mfc2','last5', 1,2)
    ]

    for key_mfc, key_cat, r, c in order:
        t, mn, std = plot_data[(key_mfc,key_cat)]
        ax = axs[r,c]

        if t is None:
            ax.set_title(f"No events", color='white')
            ax.axis('off')
            continue

        ax.fill_between(t, mn+std, mn-std, color=color, alpha=0.3)
        ax.plot(t, mn, color=color)

        ax.set_title(f"{key_mfc.upper()} — {titles[key_cat]}",
                     color='white', fontsize=11)

        ax.set_ylim(-2, 8)
        ax.plot([0,0],[-2,8],'--',color='white')
        ax.tick_params(colors='white')

        if r == 1:
            ax.set_xlabel('time (s)', color='white')
        if c == 0:
            ax.set_ylabel('Z-score', color='white')

    fig.tight_layout()
    out_folder = os.path.join(figure_folder, 'fig', 'triggered_zFF')
    os.makedirs(out_folder, exist_ok=True)
    outfile = os.path.join(out_folder, f'{event_type}_6panel.pdf')
    fig.savefig(outfile, bbox_inches='tight')

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
            
            FF = ft2.loc[relevant_avel.index, neuron]
            avel_all.extend(relevant_avel.tolist())
            FF_all.extend(FF.tolist())
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
            cmin = np.round(np.percentile(colour[~np.isnan(colour)], 10), decimals=1)  
            cmax = np.round(np.percentile(colour[~np.isnan(colour)], 90), decimals=1) 
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
            _, di, do = inside_outside_acv(ft2)
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


def compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper'):
    """
    Compute raw PVA phase and amplitude from the 16 fsb_upper wedges in ft2.

    Adds two columns to ft2 in place:
        fsb_upper_phase - PVA phase in radians, [-pi, pi]
        fsb_upper_amp   - PVA amplitude

    Returns ft2.
    """
    cols = [f'{i}_{roi_prefix}' for i in range(n_wedges)]
    wedges = ft2[cols].ffill().to_numpy()

    angles = np.arange(n_wedges) * 2*np.pi/n_wedges - np.pi + np.pi/n_wedges

    x = wedges * np.cos(angles)
    y = wedges * np.sin(angles)

    ft2['fsb_upper_phase'] = np.arctan2(y.mean(axis=1), x.mean(axis=1))
    ft2['fsb_upper_amp']   = np.sqrt(y.mean(axis=1)**2 + x.mean(axis=1)**2)

    return ft2

import numpy as np
from scipy.signal import find_peaks


def find_blocks(binary_signal, mergeblocks=True, merge_threshold=5):
    """Find contiguous blocks of 1s. Returns (starts, sizes)."""
    binary_signal = np.asarray(binary_signal).astype(int)
    if binary_signal.sum() == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    d = np.diff(np.concatenate(([0], binary_signal, [0])))
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    sizes  = ends - starts

    if mergeblocks and len(starts) > 1:
        m_starts = [starts[0]]
        m_sizes  = [sizes[0]]
        for s, sz in zip(starts[1:], sizes[1:]):
            gap = s - (m_starts[-1] + m_sizes[-1])
            if gap <= merge_threshold:
                m_sizes[-1] = (s + sz) - m_starts[-1]
            else:
                m_starts.append(s)
                m_sizes.append(sz)
        starts = np.array(m_starts)
        sizes  = np.array(m_sizes)

    return starts, sizes


def interpolate_over_stim(ft2, n_wedges=16, roi_prefix='fsb_upper', pad=5):
    """
    Interpolate over LED-shutter artifacts in the wedge columns of ft2.

    When led1_stpt == 0, the imaging laser shutter is cycling and signal is
    not continuous. This finds peaks (true signal samples) within each stim
    block and linearly interpolates between them, replacing the artifact-laden
    samples in place.

    Parameters
    ----------
    ft2 : DataFrame containing 'led1_stpt' and the wedge columns.
    n_wedges : number of wedge columns named '{i}_{roi_prefix}'.
    roi_prefix : suffix used in the wedge column names.
    pad : samples of padding around each stim block.

    Returns
    -------
    ft2 : same DataFrame, with wedge columns interpolated where applicable.
    stim_mask : bool array, True where the shutter was cycling.
    """
    ft2 = ft2.copy()

    # led1_stpt == 0 -> shutter cycling -> "stim on" for our purposes
    led1 = (ft2['led1_stpt'].to_numpy() - 1) * -1
    led1 = (led1 > 0).astype(int)

    if led1.sum() == 0:
        return ft2, np.zeros(len(ft2), dtype=bool)

    block_starts, block_sizes = find_blocks(led1, mergeblocks=True, merge_threshold=5)

    cols = [f'{i}_{roi_prefix}' for i in range(n_wedges)]
    n = len(ft2)
    # x-axis for np.interp; absolute units don't matter, only spacing.
    t_all = np.arange(n) * 0.1

    for col in cols:
        if col not in ft2.columns:
            continue
        y = ft2[col].to_numpy().astype(float)
        ynew = y.copy()

        # neutralize NaNs for peak finding without losing them in the output copy
        if np.isnan(ynew).any():
            ynew[np.isnan(ynew)] = np.nanmean(y)
        y_for_peaks = np.where(np.isnan(y), 0.0, y)

        for bstart, bsize in zip(block_starts, block_sizes):
            seg = np.arange(bstart - pad, bstart + bsize + pad)
            seg = seg[(seg >= 0) & (seg < n)]
            if len(seg) < 3:
                continue

            ty = y_for_peaks[seg].copy()
            ty -= np.percentile(ty, 30)
            ty[ty < 0] = 0
            tyo = y_for_peaks[seg]
            tt  = t_all[seg]

            prominence = np.percentile(ty, 60) - np.percentile(ty, 40)
            if prominence <= 0:
                prominence = None
            peaks = find_peaks(ty, prominence=prominence)[0]

            # anchor segment endpoints so endpoints don't extrapolate flat
            peaks = np.unique(np.concatenate(([0], peaks, [len(ty) - 1])))

            ynew[seg] = np.interp(tt, tt[peaks], tyo[peaks])

        ft2[col] = ynew

    return ft2, led1.astype(bool)

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


