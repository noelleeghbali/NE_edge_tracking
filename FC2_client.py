
#%%
from imaging_analysis import *
from behavior_analysis import *
from analysis_funs.regression import fci_regmodel
from analysis_funs.optogenetics import opto
from analysis_funs.CX_imaging import CX
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from scipy.interpolate import interp1d
import sys
import pickle
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


#%% heatmap
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'
n_wedges = 16
roi_prefix = 'fsb_upper'
t_start = 0     # start time in seconds (set to None for beginning)
t_end   = 120   # end time in seconds (set to None for end)
plume_colors = {
    'instrip_1':         'tab:blue',   # pink
    'instrip_2':         'tab:cyan',   # blue
    'instrip_ambiguous': (0.6, 0.6, 0.6),   # gray
}

for filename in os.listdir(folder_path):
    if not filename.endswith('.pkl'):
        continue
    img_dict = open_pickle(f'{folder_path}/{filename}')
    pv2, ft2 = process_CX_pickle(img_dict)
    ft2, stim_mask = interpolate_over_stim(ft2, n_wedges=n_wedges, roi_prefix=roi_prefix, pad=5)
    ft2 = compute_fsb_upper_phase(ft2, n_wedges=n_wedges, roi_prefix=roi_prefix)
    ft2 = label_plume_entries(ft2)
    time = pv2['relative_time'].to_numpy()

    # build time-window mask
    tlo = time[0]  if t_start is None else t_start
    thi = time[-1] if t_end   is None else t_end
    mask = (time >= tlo) & (time <= thi)
    if not mask.any():
        print(f'skipping {filename}: no samples in [{tlo}, {thi}]')
        continue

    time_w = time[mask]
    ft2_w  = ft2.loc[mask].reset_index(drop=True)

    wedge_cols = [f'{i}_{roi_prefix}' for i in range(n_wedges)]
    activity = ft2_w[wedge_cols].to_numpy()
    vmin = np.nanpercentile(activity, 5)
    vmax = np.nanpercentile(activity, 95)

    fig, ax, markc = configure_bw_plot(size=(1, 5), xaxis=False)
    pink_cmap = LinearSegmentedColormap.from_list('blackpink', ['black', 'red', 'white'])

    im = ax.imshow(
        activity,
        aspect='auto',
        origin='upper',
        extent=[0.5, n_wedges + 0.5, time_w[-1], time_w[0]],
        cmap=pink_cmap,
        interpolation='none',
        vmin=vmin,
        vmax=vmax,
    )

    box_left  = n_wedges + 1.0
    box_width = 2.0

    for col, color in plume_colors.items():
        sig = ft2_w[col].to_numpy().astype(float)
        d = np.diff(sig)
        ons  = np.where(d > 0)[0] + 1
        offs = np.where(d < 0)[0] + 1
        if sig[0] > 0:
            ons = np.insert(ons, 0, 0)
        if sig[-1] > 0:
            offs = np.append(offs, len(sig) - 1)

        for s, e in zip(ons, offs):
            # clip rectangle times to the window (defensive; mask already did most of this)
            y0 = max(time_w[s], tlo)
            y1 = min(time_w[e], thi)
            if y1 <= y0:
                continue
            rect = Rectangle(
                (box_left, y0),
                box_width,
                y1 - y0,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
            )
            ax.add_patch(rect)

    ax.set_xlim(0.5, box_left + box_width + 0.5)
    ax.set_ylim(thi, tlo)  # origin='upper' => larger time at bottom
    ax.set_xlabel('wedge')
    ax.set_ylabel('time (s)')
    ax.set_xticks([])
    ax.set_title(filename, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, fraction=0.05, ticks=[vmin, vmax])
    cbar.ax.set_xticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])
    cbar.set_label(r'$\Delta F/F_0$', color=markc)
    cbar.ax.xaxis.set_tick_params(color=markc)
    cbar.outline.set_edgecolor(markc)
    plt.setp(cbar.ax.get_xticklabels(), color=markc)
    plt.show()
    out_dir = f'{folder_path}/fig/heatmap'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(f'{out_dir}/{filename}_heatmap_{t_start}_{t_end}.pdf', bbox_inches='tight')


#%% phase change over exit (pseudotime)
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'
for filename in os.listdir(folder_path):
    if filename.endswith('.pkl'):
        img_dict = open_pickle(f'{folder_path}/{filename}')
        pv2, ft2 = process_CX_pickle(img_dict)
        ft2 = compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper')
        time = pv2['relative_time'].to_numpy()
        in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
        off_bouts = exit_bout_indices(in_acv)
        ft2['fsb_upper_phase_deg'] = np.degrees(ft2['fsb_upper_phase'])

        n_pseudo = 75
        pseudotime = np.linspace(0, 1, n_pseudo)
        tail_seconds = None

        phase_deg = ft2['fsb_upper_phase_deg'].to_numpy()
        heading_deg = np.degrees(ft2['ft_heading'].to_numpy())  # assumes ft_heading is in radians

        aligned_phase = []
        aligned_heading = []
        selected_bouts = off_bouts[2:15]

        for start, end in selected_bouts:
            full_bout_phase = phase_deg[start:end]
            full_bout_heading = heading_deg[start:end]
            if len(full_bout_phase) < 2:
                continue

            full_phase_change   = (full_bout_phase   - full_bout_phase[0]   + 180) % 360 - 180
            full_heading_change = (full_bout_heading - full_bout_heading[0] + 180) % 360 - 180

            if tail_seconds is not None:
                bout_time = time[start:end]
                bout_duration = bout_time[-1] - bout_time[0]
                if bout_duration < tail_seconds:
                    continue
                cutoff = bout_time[-1] - tail_seconds
                tail_mask = bout_time >= cutoff
                phase_change   = full_phase_change[tail_mask]
                heading_change = full_heading_change[tail_mask]
            else:
                phase_change   = full_phase_change
                heading_change = full_heading_change

            if len(phase_change) < 2:
                continue

            bout_pseudotime = np.linspace(0, 1, len(phase_change))
            aligned_phase.append(
                interp1d(bout_pseudotime, phase_change,   kind='linear')(pseudotime)
            )
            aligned_heading.append(
                interp1d(bout_pseudotime, heading_change, kind='linear')(pseudotime)
            )

        aligned_phase   = np.array(aligned_phase)
        aligned_heading = np.array(aligned_heading)

        mean_phase   = np.nanmean(aligned_phase,   axis=0)
        sem_phase    = np.nanstd(aligned_phase,    axis=0) / np.sqrt(aligned_phase.shape[0])
        mean_heading = np.nanmean(aligned_heading, axis=0)
        sem_heading  = np.nanstd(aligned_heading,  axis=0) / np.sqrt(aligned_heading.shape[0])

        fig, ax, markc = configure_bw_plot(size=(3,3), xaxis=True)

        for trace in aligned_phase:
            ax.scatter(pseudotime, trace, color='tab:blue', alpha=0.15, s=8)
        for trace in aligned_heading:
            ax.scatter(pseudotime, trace, color='lightgrey', alpha=0.15, s=8)

        # means + SEM bands
        ax.plot(pseudotime, mean_phase, color='tab:blue', lw=2, label='FC2 phase')
        ax.plot(pseudotime, mean_heading, color='lightgrey', lw=2, label='heading')
        ax.axhline(0, color=markc, ls='--', lw=0.5)
        window_label = f'last {tail_seconds}s' if tail_seconds is not None else 'full bout'
        ax.set_xlabel(f'pseudotime ({window_label})', color=markc)
        ax.set_ylabel('Δ from bout onset (deg)', color=markc)
        ax.set_title(f'{filename}, {window_label}')
        ax.legend(frameon=False)

        out_dir = f'{folder_path}/fig/phase_change_heading'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        fig.savefig(f'{out_dir}/{filename}_plume1.pdf', bbox_inches='tight')


#%% combined phase change & heading over pseudotime, both plumes, all files
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez/PAM_all'

n_pseudo = 50
pseudotime = np.linspace(0, 1, n_pseudo)
tail_seconds = None
heading_window_seconds = 1  # last 1s of plume-1 off-bouts used to infer side

bout_groups_idx = {
    'plume 1': slice(2, 15),
    'plume 2': slice(17, 33),
}
group_colors = {
    'plume 1': {'phase': 'tab:blue',  'heading': 'tab:grey'},
    'plume 2': {'phase': 'tab:cyan',  'heading': 'tab:grey'},
}

# accumulate one mean trace per file per plume (for both phase and heading)
per_file = {label: {'phase': [], 'heading': []} for label in bout_groups_idx}

for filename in os.listdir(folder_path):
    if not filename.endswith('.pkl'):
        continue

    img_dict = open_pickle(f'{folder_path}/{filename}')
    pv2, ft2 = process_CX_pickle(img_dict)
    ft2 = compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper')
    time = pv2['relative_time'].to_numpy()
    in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
    off_bouts = exit_bout_indices(in_acv)
    ft2['fsb_upper_phase_deg'] = np.degrees(ft2['fsb_upper_phase'])

    phase_deg = ft2['fsb_upper_phase_deg'].to_numpy()
    heading_rad = ft2['ft_heading'].to_numpy()
    heading_deg = np.degrees(heading_rad)

    # ---- infer plume 1 side from sin(heading) in last 1s of plume-1 off-bouts ----
    plume1_bouts = off_bouts[bout_groups_idx['plume 1']]
    side_samples = []
    for start, end in plume1_bouts:
        if end - start < 2:
            continue
        bout_time = time[start:end]
        if bout_time[-1] - bout_time[0] < heading_window_seconds:
            continue
        cutoff = bout_time[-1] - heading_window_seconds
        mask = bout_time >= cutoff
        side_samples.append(np.sin(heading_rad[start:end][mask]))

    if not side_samples:
        print(f'{filename}: skipped (no usable bouts to infer side)')
        continue
    mean_sin = np.nanmean(np.concatenate(side_samples))
    flip = -1.0 if mean_sin > 0 else 1.0  # plume 1 -> left

    # ---- per-bout interpolation onto pseudotime ----
    def interp_bouts(selected_bouts):
        phase_traces, heading_traces = [], []
        for start, end in selected_bouts:
            full_phase = phase_deg[start:end]
            full_heading = heading_deg[start:end]
            if len(full_phase) < 2:
                continue

            phase_change   = (full_phase   - full_phase[0]   + 180) % 360 - 180
            heading_change = (full_heading - full_heading[0] + 180) % 360 - 180

            if tail_seconds is not None:
                bout_time = time[start:end]
                if bout_time[-1] - bout_time[0] < tail_seconds:
                    continue
                cutoff = bout_time[-1] - tail_seconds
                tail_mask = bout_time >= cutoff
                phase_change   = phase_change[tail_mask]
                heading_change = heading_change[tail_mask]

            if len(phase_change) < 2:
                continue

            # apply side flip
            phase_change   = phase_change * flip
            heading_change = heading_change * flip

            bout_pt = np.linspace(0, 1, len(phase_change))
            phase_traces.append(interp1d(bout_pt, phase_change,   kind='linear')(pseudotime))
            heading_traces.append(interp1d(bout_pt, heading_change, kind='linear')(pseudotime))

        return np.array(phase_traces), np.array(heading_traces)

    for label, sl in bout_groups_idx.items():
        phase_arr, heading_arr = interp_bouts(off_bouts[sl])
        if len(phase_arr) == 0:
            continue
        # one mean trace per file per plume (so each fly contributes equally)
        per_file[label]['phase'].append(np.nanmean(phase_arr, axis=0))
        per_file[label]['heading'].append(np.nanmean(heading_arr, axis=0))

# ---- plot: across-file mean ± SEM for each plume, phase + heading ----
fig, ax, markc = configure_white_plot(size=(5,2), xaxis=True)

for label in bout_groups_idx:
    if label == 'plume 1':
        continue
    phase_list = per_file[label]['phase']
    heading_list = per_file[label]['heading']
    if not phase_list:
        continue

    phase_arr = np.vstack(phase_list)     # (n_files, n_pseudo)
    heading_arr = np.vstack(heading_list)
    n = phase_arr.shape[0]

    phase_mean = np.nanmean(phase_arr, axis=0)
    phase_sem  = np.nanstd(phase_arr, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(phase_mean)
    heading_mean = np.nanmean(heading_arr, axis=0)
    heading_sem  = np.nanstd(heading_arr, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(heading_mean)

    cP = group_colors[label]['phase']
    cH = group_colors[label]['heading']

    ax.plot(pseudotime, phase_mean, color=cP, lw=2, label=f'{label} phase (n={n})')
    ax.fill_between(pseudotime, phase_mean - phase_sem, phase_mean + phase_sem,
                    color=cP, alpha=0.2, lw=0)

    ax.plot(pseudotime, heading_mean, color=cH, lw=2, label=f'{label} heading (n={n})')
    ax.fill_between(pseudotime, heading_mean - heading_sem, heading_mean + heading_sem,
                    color=cH, alpha=0.2, lw=0)

ax.axhline(0, color=markc, ls='--', lw=0.5)

window_label = f'last {tail_seconds}s' if tail_seconds is not None else 'full bout'
ax.set_xlabel(f'pseudotime ({window_label})', color=markc)
ax.set_ylabel('Δ from bout onset', color=markc)
ax.set_title(f'all files, {window_label}')
ax.legend(frameon=False, fontsize=8)

out_dir = f'{folder_path}/fig/phase_change_heading'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
fig.savefig(f'{out_dir}/plume2_all_avg.pdf', bbox_inches='tight')



#%% phase distribution of epg and fc2
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'
for filename in os.listdir(folder_path):
    if filename.endswith('.pkl'):
        img_dict = open_pickle(f'{folder_path}/{filename}')
        pv2, ft2 = process_CX_pickle(img_dict)
        ft2 = compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper')
        time = pv2['relative_time'].to_numpy()
        in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
        off_bouts = exit_bout_indices(in_acv)
        ft2['fsb_upper_phase_deg'] = np.degrees(ft2['fsb_upper_phase'])
 
        tail_seconds = 1
 
        phase_deg = ft2['fsb_upper_phase_deg'].to_numpy()
        heading_deg = np.degrees(ft2['ft_heading'].to_numpy())  # assumes ft_heading is in radians
 
        # collect phase-change and heading-change samples across bouts
        all_phase_changes = []
        all_heading_changes = []
        selected_bouts = off_bouts[2:15]
 
        for start, end in selected_bouts:
            full_bout_phase = phase_deg[start:end]
            full_bout_heading = heading_deg[start:end]
            if len(full_bout_phase) < 2:
                continue
 
            full_phase_change   = (full_bout_phase   - full_bout_phase[0]   + 180) % 360 - 180
            full_heading_change = (full_bout_heading - full_bout_heading[0] + 180) % 360 - 180
 
            if tail_seconds is not None:
                bout_time = time[start:end]
                bout_duration = bout_time[-1] - bout_time[0]
                if bout_duration < tail_seconds:
                    continue
                cutoff = bout_time[-1] - tail_seconds
                tail_mask = bout_time >= cutoff
                phase_change   = full_phase_change[tail_mask]
                heading_change = full_heading_change[tail_mask]
            else:
                phase_change   = full_phase_change
                heading_change = full_heading_change
 
            if len(phase_change) < 2:
                continue
 
            all_phase_changes.append(phase_change)
            all_heading_changes.append(heading_change)
 
        if len(all_phase_changes) == 0:
            continue
 
        all_phase_changes   = np.concatenate(all_phase_changes)
        all_heading_changes = np.concatenate(all_heading_changes)
 
        # drop NaNs just in case
        all_phase_changes   = all_phase_changes[~np.isnan(all_phase_changes)]
        all_heading_changes = all_heading_changes[~np.isnan(all_heading_changes)]
 
        fig, ax, markc = configure_white_plot(size=(5,4), xaxis=True)
 
        bins = np.linspace(-180, 180, 9)  # NUMBER OF BINS
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
 
        phase_counts,   _ = np.histogram(all_phase_changes,   bins=bins, density=True)
        heading_counts, _ = np.histogram(all_heading_changes, bins=bins, density=True)
 
        ax.plot(bin_centers, phase_counts,   color='tab:blue',   lw=2,
                marker='o', ms=4, label='FC2 phase')
        ax.plot(bin_centers, heading_counts, color='tab:orange', lw=2,
                marker='o', ms=4, label='heading')
 
        ax.axvline(0, color=markc, ls='--', lw=0.5)
 
        window_label = f'last {tail_seconds}s' if tail_seconds is not None else 'full bout'
        ax.set_xlabel('Δ from bout onset (deg)', color=markc)
        ax.set_ylabel('density', color=markc)
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.set_title(f'{filename}, {window_label}')
        ax.legend(frameon=False)

#%% plume 1 and plume 2 phase change
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'
for filename in os.listdir(folder_path):
    if filename.endswith('.pkl'):
        img_dict = open_pickle(f'{folder_path}/{filename}')
        pv2, ft2 = process_CX_pickle(img_dict)
        ft2 = compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper')
        time = pv2['relative_time'].to_numpy()
        in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
        off_bouts = exit_bout_indices(in_acv)
        ft2['fsb_upper_phase_deg'] = np.degrees(ft2['fsb_upper_phase'])

        tail_seconds = 3

        phase_deg = ft2['fsb_upper_phase_deg'].to_numpy()
        heading_deg = np.degrees(ft2['ft_heading'].to_numpy())  # assumes ft_heading is in radians

        # define the two bout selections to compare
        bout_groups = {
            'plume 1':  off_bouts[2:15],
            'plume 2': off_bouts[17:30],
        }
        group_colors = {
            'plume 1':  'tab:blue',
            'plume 2': 'tab:cyan',
        }

        def collect_phase_changes(selected_bouts):
            all_phase_changes = []
            for start, end in selected_bouts:
                full_bout_phase = phase_deg[start:end]
                if len(full_bout_phase) < 2:
                    continue

                full_phase_change = (full_bout_phase - full_bout_phase[0] + 180) % 360 - 180

                if tail_seconds is not None:
                    bout_time = time[start:end]
                    bout_duration = bout_time[-1] - bout_time[0]
                    if bout_duration < tail_seconds:
                        continue
                    cutoff = bout_time[-1] - tail_seconds
                    tail_mask = bout_time >= cutoff
                    phase_change = full_phase_change[tail_mask]
                else:
                    phase_change = full_phase_change

                if len(phase_change) < 2:
                    continue

                all_phase_changes.append(phase_change)

            if len(all_phase_changes) == 0:
                return np.array([])

            out = np.concatenate(all_phase_changes)
            return out[~np.isnan(out)]

        fig, ax, markc = configure_white_plot(size=(4,3), xaxis=True)

        bins = np.linspace(-180, 180, 8)  # NUMBER OF BINS
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        plotted_any = False
        for label, selected_bouts in bout_groups.items():
            phase_changes = collect_phase_changes(selected_bouts)
            if len(phase_changes) == 0:
                continue

            counts, _ = np.histogram(phase_changes, bins=bins)
            counts = counts / counts.sum() if counts.sum() > 0 else counts
            ax.plot(bin_centers, counts, color=group_colors[label], lw=2,
                    marker='o', ms=4, label=label)
            plotted_any = True

        if not plotted_any:
            continue

        ax.axvline(0, color=markc, ls='--', lw=0.5)

        window_label = f'last {tail_seconds}s' if tail_seconds is not None else 'full bout'
        ax.set_xlabel('Δ from bout onset (deg)', color=markc)
        ax.set_ylabel('probability', color=markc)
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.set_title(f'{filename}, {window_label}')
        ax.legend(frameon=False)


#%% combined plume 1 vs plume 2 phase change across all files
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez/PAM_train'

tail_seconds = 3
heading_window_seconds = 1  # last 1s of each off-bout used to infer side

bout_groups_idx = {
    'plume 1': slice(2, 15),
    'plume 2': slice(17, 30),
}
group_colors = {
    'plume 1': 'tab:blue',   # normalized to LEFT (negative Δ)
    'plume 2': 'tab:cyan',  # normalized to RIGHT (positive Δ)
}

bins = np.linspace(-180, 180, 10)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# accumulate one probability histogram per file per plume
per_file_probs = {label: [] for label in bout_groups_idx}

for filename in os.listdir(folder_path):
    if not filename.endswith('.pkl'):
        continue

    img_dict = open_pickle(f'{folder_path}/{filename}')
    pv2, ft2 = process_CX_pickle(img_dict)
    ft2 = compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper')
    time = pv2['relative_time'].to_numpy()
    in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
    off_bouts = exit_bout_indices(in_acv)
    ft2['fsb_upper_phase_deg'] = np.degrees(ft2['fsb_upper_phase'])

    phase_deg = ft2['fsb_upper_phase_deg'].to_numpy()
    heading_rad = ft2['ft_heading'].to_numpy()

    # ---- infer plume 1's side from mean sin(heading) in last 1s of plume-1 off-bouts ----
    plume1_bouts = off_bouts[bout_groups_idx['plume 1']]
    side_samples = []
    for start, end in plume1_bouts:
        if end - start < 2:
            continue
        bout_time = time[start:end]
        if bout_time[-1] - bout_time[0] < heading_window_seconds:
            continue
        cutoff = bout_time[-1] - heading_window_seconds
        mask = bout_time >= cutoff
        side_samples.append(np.sin(heading_rad[start:end][mask]))

    if not side_samples:
        print(f'{filename}: skipped (no usable bouts to infer side)')
        continue

    mean_sin = np.nanmean(np.concatenate(side_samples))
    # convention: positive sin(heading) -> "right" plume; we want plume 1 on the LEFT,
    # so flip phase-change sign when plume 1 was on the right.
    flip = -1 if mean_sin > 0 else 1

    # ---- collect phase changes per group, applying the flip ----
    def collect_phase_changes(selected_bouts):
        out = []
        for start, end in selected_bouts:
            full_bout_phase = phase_deg[start:end]
            if len(full_bout_phase) < 2:
                continue
            full_phase_change = (full_bout_phase - full_bout_phase[0] + 180) % 360 - 180

            bout_time = time[start:end]
            bout_duration = bout_time[-1] - bout_time[0]
            if bout_duration < tail_seconds:
                continue
            cutoff = bout_time[-1] - tail_seconds
            tail_mask = bout_time >= cutoff
            phase_change = full_phase_change[tail_mask] * flip

            if len(phase_change) < 2:
                continue
            out.append(phase_change)

        if not out:
            return np.array([])
        arr = np.concatenate(out)
        return arr[~np.isnan(arr)]

    for label, sl in bout_groups_idx.items():
        pc = collect_phase_changes(off_bouts[sl])
        if len(pc) == 0:
            continue
        counts, _ = np.histogram(pc, bins=bins)
        if counts.sum() == 0:
            continue
        per_file_probs[label].append(counts / counts.sum())

# ---- plot: individual files faint + group mean bold ----
fig, ax, markc = configure_bw_plot(size=(4, 3), xaxis=True)

for label, probs_list in per_file_probs.items():
    if not probs_list:
        continue
    color = group_colors[label]
    arr = np.vstack(probs_list)  # (n_files, n_bins)

    # individual files (faint)
    for row in arr:
        ax.plot(bin_centers, row, color=color, lw=0.8, alpha=0.25)

    # group mean +/- SEM
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
    ax.plot(bin_centers, mean, color=color, lw=2, marker='o', ms=4,
            label=f'{label} (n={arr.shape[0]})')
    ax.fill_between(bin_centers, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)

ax.axvline(0, color=markc, ls='--', lw=0.5)
ax.set_xlabel('Δ from bout onset (deg, normalized: plume 1 left / plume 2 right)', color=markc)
ax.set_ylabel('probability', color=markc)
ax.set_xlim(-180, 180)
ax.set_ylim(0, 0.5)
ax.set_xticks(np.arange(-180, 181, 90))
ax.set_title(f'all files, last {tail_seconds}s')
ax.legend(frameon=False)

out_dir = f'{folder_path}/fig/phase_change'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
fig.savefig(f'{out_dir}/plume1_vs_plume2.pdf', bbox_inches='tight')


#%% combined plume 1 vs plume 2 phase change + heading across all files
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'

tail_seconds = 3
heading_window_seconds = 1  # last 1s of each off-bout used to infer side

bout_groups_idx = {
    'plume 1': slice(2, 15),
    'plume 2': slice(17, 30),
}
group_colors = {
    'plume 1': 'tab:blue',   # normalized to LEFT (negative Δ)
    'plume 2': 'tab:cyan',  # normalized to RIGHT (positive Δ)
}

bins = np.linspace(-180, 180, 10)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# accumulate one probability histogram per file per plume (phase + heading)
per_file_probs = {label: [] for label in bout_groups_idx}
per_file_probs_heading = {label: [] for label in bout_groups_idx}

for filename in os.listdir(folder_path):
    if not filename.endswith('.pkl'):
        continue

    img_dict = open_pickle(f'{folder_path}/{filename}')
    pv2, ft2 = process_CX_pickle(img_dict)
    ft2 = compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper')
    time = pv2['relative_time'].to_numpy()
    in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
    off_bouts = exit_bout_indices(in_acv)
    ft2['fsb_upper_phase_deg'] = np.degrees(ft2['fsb_upper_phase'])

    phase_deg = ft2['fsb_upper_phase_deg'].to_numpy()
    heading_rad = ft2['ft_heading'].to_numpy()
    heading_deg = np.degrees(heading_rad)

    # ---- infer plume 1's side from mean sin(heading) in last 1s of plume-1 off-bouts ----
    plume1_bouts = off_bouts[bout_groups_idx['plume 1']]
    side_samples = []
    for start, end in plume1_bouts:
        if end - start < 2:
            continue
        bout_time = time[start:end]
        if bout_time[-1] - bout_time[0] < heading_window_seconds:
            continue
        cutoff = bout_time[-1] - heading_window_seconds
        mask = bout_time >= cutoff
        side_samples.append(np.sin(heading_rad[start:end][mask]))

    if not side_samples:
        print(f'{filename}: skipped (no usable bouts to infer side)')
        continue

    mean_sin = np.nanmean(np.concatenate(side_samples))
    # convention: positive sin(heading) -> "right" plume; we want plume 1 on the LEFT,
    # so flip phase-change sign when plume 1 was on the right.
    flip = -1 if mean_sin > 0 else 1

    # ---- collect changes per group, applying the flip ----
    def collect_changes(selected_bouts, signal_deg):
        out = []
        for start, end in selected_bouts:
            full_bout = signal_deg[start:end]
            if len(full_bout) < 2:
                continue
            full_change = (full_bout - full_bout[0] + 180) % 360 - 180

            bout_time = time[start:end]
            bout_duration = bout_time[-1] - bout_time[0]
            if bout_duration < tail_seconds:
                continue
            cutoff = bout_time[-1] - tail_seconds
            tail_mask = bout_time >= cutoff
            change = full_change[tail_mask] * flip

            if len(change) < 2:
                continue
            out.append(change)

        if not out:
            return np.array([])
        arr = np.concatenate(out)
        return arr[~np.isnan(arr)]

    for label, sl in bout_groups_idx.items():
        # phase change
        pc = collect_changes(off_bouts[sl], phase_deg)
        if len(pc) > 0:
            counts, _ = np.histogram(pc, bins=bins)
            if counts.sum() > 0:
                per_file_probs[label].append(counts / counts.sum())

        # heading change
        hc = collect_changes(off_bouts[sl], heading_deg)
        if len(hc) > 0:
            counts_h, _ = np.histogram(hc, bins=bins)
            if counts_h.sum() > 0:
                per_file_probs_heading[label].append(counts_h / counts_h.sum())

# ---- plot: individual files faint + group mean bold ----
fig, ax, markc = configure_bw_plot(size=(5,4), xaxis=True)

for label, probs_list in per_file_probs.items():
    color = group_colors[label]

    # ---- phase (solid) ----
    if probs_list:
        arr = np.vstack(probs_list)  # (n_files, n_bins)

        # individual files (faint)
        # for row in arr:
            # ax.plot(bin_centers, row, color=color, lw=0.8, alpha=0.25)

        # group mean +/- SEM
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(bin_centers, mean, color=color, lw=2, marker='o', ms=4,
                label=f'{label} phase (n={arr.shape[0]})')
        ax.fill_between(bin_centers, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)

    # ---- heading (dotted) ----
    probs_list_h = per_file_probs_heading[label]
    if probs_list_h:
        arr_h = np.vstack(probs_list_h)

        # individual files (faint, dotted)
        # for row in arr_h:
        #     ax.plot(bin_centers, row, color=color, lw=0.8, alpha=0.25, ls=':')

        # group mean +/- SEM (dotted)
        mean_h = arr_h.mean(axis=0)
        sem_h = arr_h.std(axis=0, ddof=1) / np.sqrt(arr_h.shape[0]) if arr_h.shape[0] > 1 else np.zeros_like(mean_h)
        ax.plot(bin_centers, mean_h, color=color, lw=2, ls=':', marker='s', ms=4,
                label=f'{label} heading (n={arr_h.shape[0]})')
        # ax.fill_between(bin_centers, mean_h - sem_h, mean_h + sem_h, color=color, alpha=0.15, lw=0)

ax.axvline(0, color=markc, ls='--', lw=0.5)
ax.set_xlabel('Δ from bout onset (deg, normalized: plume 1 left / plume 2 right)', color=markc)
ax.set_ylabel('probability', color=markc)
ax.set_xlim(-180, 180)
ax.set_xticks(np.arange(-180, 181, 90))
ax.set_title(f'all files, last {tail_seconds}s')
ax.legend(frameon=False, fontsize=7)

out_dir = f'{folder_path}/fig/phase_change'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
fig.savefig(f'{out_dir}/plume1_vs_plume2_with_heading.pdf', bbox_inches='tight')

#%% average bump shape, last 3s of returns, plume 1 vs plume 2
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez/PAM_all'

n_wedges = 16
tail_seconds = 3
plot_per_fly = True   # <-- toggle per-fly plots

bout_groups_idx = {
    'plume 1': slice(2, 15),
    'plume 2': slice(17, 33),
}
group_colors = {
    'plume 1': 'tab:blue',
    'plume 2': 'tab:cyan',
}

# wedge angles must match those used in compute_fsb_upper_phase
wedge_angles = np.arange(n_wedges) * 2*np.pi/n_wedges - np.pi + np.pi/n_wedges
center_idx = n_wedges // 2

# x-axis: wedge offset from bump peak, in degrees
wedge_offsets_deg = (np.degrees(wedge_angles - wedge_angles[center_idx]) + 180) % 360 - 180
order = np.argsort(wedge_offsets_deg)
x = wedge_offsets_deg[order]

# accumulate one mean bump per fly per plume
per_file_bumps = {label: [] for label in bout_groups_idx}
per_file_names = []

for filename in os.listdir(folder_path):
    if not filename.endswith('.pkl'):
        continue

    img_dict = open_pickle(f'{folder_path}/{filename}')
    pv2, ft2 = process_CX_pickle(img_dict)
    ft2 = compute_fsb_upper_phase(ft2, n_wedges=n_wedges, roi_prefix='fsb_upper')
    time = pv2['relative_time'].to_numpy()
    in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
    off_bouts = exit_bout_indices(in_acv)

    wedge_cols = [f'{i}_fsb_upper' for i in range(n_wedges)]
    wedge_data = ft2[wedge_cols].ffill().to_numpy()
    phase_rad = ft2['fsb_upper_phase'].to_numpy()

    def collect_aligned_bumps(selected_bouts):
        bout_means = []
        for start, end in selected_bouts:
            if end - start < 2:
                continue
            bout_time = time[start:end]
            if bout_time[-1] - bout_time[0] < tail_seconds:
                continue
            cutoff = bout_time[-1] - tail_seconds
            tail_mask = bout_time >= cutoff

            tail_wedges = wedge_data[start:end][tail_mask]
            tail_phase  = phase_rad[start:end][tail_mask]

            if tail_wedges.shape[0] < 2:
                continue

            aligned = np.empty_like(tail_wedges)
            for i in range(tail_wedges.shape[0]):
                if np.isnan(tail_phase[i]):
                    aligned[i] = np.nan
                    continue
                peak_idx = int(np.argmin(np.abs(np.angle(
                    np.exp(1j * (wedge_angles - tail_phase[i]))
                ))))
                shift = center_idx - peak_idx
                aligned[i] = np.roll(tail_wedges[i], shift)

            bout_mean = np.nanmean(aligned, axis=0)
            if not np.all(np.isnan(bout_mean)):
                bout_means.append(bout_mean)
        return bout_means

    # collect per-plume bumps for this fly
    fly_bumps_this_file = {}
    for label, sl in bout_groups_idx.items():
        bumps = collect_aligned_bumps(off_bouts[sl])
        if not bumps:
            continue
        # per-bout array, and the fly's mean across bouts
        bout_arr = np.vstack(bumps)             # (n_bouts, n_wedges)
        fly_mean = np.nanmean(bout_arr, axis=0)
        per_file_bumps[label].append(fly_mean)
        fly_bumps_this_file[label] = (bout_arr, fly_mean)

    if not fly_bumps_this_file:
        continue
    per_file_names.append(filename)

    # ---- per-fly plot ----
    if plot_per_fly:
        fig_f, ax_f, markc_f = configure_white_plot(size=(4, 4), xaxis=True)
        for label, (bout_arr, fly_mean) in fly_bumps_this_file.items():
            color = group_colors[label]
            n_b = bout_arr.shape[0]
            # individual bouts faint
            for row in bout_arr[:, order]:
                ax_f.plot(x, row, color=color, lw=0.6, alpha=0.25)
            # fly mean bold
            ax_f.plot(x, fly_mean[order], color=color, lw=2, marker='o', ms=4,
                      label=f'{label} (n_bouts={n_b})')

        ax_f.axvline(0, color=markc_f, ls='--', lw=0.5)
        ax_f.set_xlabel('wedge offset from bump peak (deg)', color=markc_f)
        ax_f.set_ylabel('wedge activity', color=markc_f)
        ax_f.set_xlim(-180, 180)
        ax_f.set_ylim(0,1)
        ax_f.set_xticks(np.arange(-180, 181, 90))
        ax_f.set_title(f'{filename}, last {tail_seconds}s of returns')
        ax_f.legend(frameon=False)
        out_dir = f'{folder_path}/fig/bump_shape'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        fig_f.savefig(f'{out_dir}/{filename}_plume1_vs_plume2.pdf', bbox_inches='tight')
        plt.close(fig_f)

# ---- group summary plot ----
fig, ax, markc = configure_white_plot(size=(4, 4), xaxis=True)

for label, bumps_list in per_file_bumps.items():
    if not bumps_list:
        continue
    color = group_colors[label]
    arr = np.vstack(bumps_list)[:, order]
    n = arr.shape[0]

    for row in arr:
        ax.plot(x, row, color=color, lw=0.8, alpha=0.25)

    mean = np.nanmean(arr, axis=0)
    sem  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)

    ax.plot(x, mean, color=color, lw=2, marker='o', ms=4, label=f'{label} (n={n})')
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)

ax.axvline(0, color=markc, ls='--', lw=0.5)
ax.set_xlabel('wedge offset from bump peak (deg)', color=markc)
ax.set_ylabel('wedge activity', color=markc)
ax.set_xlim(-180, 180)
ax.set_ylim(0,1)
ax.set_xticks(np.arange(-180, 181, 90))
ax.set_title(f'all flies, last {tail_seconds}s of returns')
ax.legend(frameon=False)

out_dir = f'{folder_path}/fig/bump_shape'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
fig.savefig(f'{out_dir}/plume1_vs_plume2.pdf', bbox_inches='tight')

#%% PVA magnitude comparison: plume 1 vs plume 2
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez/PAM_all'

tail_seconds = 3

bout_groups_idx = {
    'plume 1': slice(2, 15),
    'plume 2': slice(17, 30),
}
group_colors = {
    'plume 1': 'tab:blue',
    'plume 2': 'tab:cyan',
}

# per-file summary: one mean PVA magnitude per plume
per_file_amp = {label: [] for label in bout_groups_idx}
file_labels = []

for filename in os.listdir(folder_path):
    if not filename.endswith('.pkl'):
        continue

    img_dict = open_pickle(f'{folder_path}/{filename}')
    pv2, ft2 = process_CX_pickle(img_dict)
    ft2 = compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper')
    time = pv2['relative_time'].to_numpy()
    in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
    off_bouts = exit_bout_indices(in_acv)

    amp = ft2['fsb_upper_amp'].to_numpy()  # PVA magnitude

    def bout_mean_amps(selected_bouts):
        out = []
        for start, end in selected_bouts:
            if end - start < 2:
                continue
            bt = time[start:end]
            if bt[-1] - bt[0] < tail_seconds:
                tail_mask = np.ones_like(bt, dtype=bool)  # use whole bout if shorter
            else:
                cutoff = bt[-1] - tail_seconds
                tail_mask = bt >= cutoff

            tail_amp = amp[start:end][tail_mask]
            tail_amp = tail_amp[~np.isnan(tail_amp)]
            if len(tail_amp) == 0:
                continue
            out.append(np.mean(tail_amp))
        return np.array(out)

    file_labels.append(filename)
    for label, sl in bout_groups_idx.items():
        bout_amps = bout_mean_amps(off_bouts[sl])
        # one number per fly per plume: mean across that plume's bouts
        per_file_amp[label].append(np.nanmean(bout_amps) if len(bout_amps) else np.nan)

# ---- plot: per-fly PVA magnitude, paired across plumes ----
fig, ax, markc = configure_white_plot(size=(2, 3), xaxis=True)

A1 = np.array(per_file_amp['plume 1'])
A2 = np.array(per_file_amp['plume 2'])
valid = ~(np.isnan(A1) | np.isnan(A2))
A1, A2 = A1[valid], A2[valid]

x1 = np.full_like(A1, 0, dtype=float) + np.random.uniform(-0.05, 0.05, size=A1.shape)
x2 = np.full_like(A2, 1, dtype=float) + np.random.uniform(-0.05, 0.05, size=A2.shape)

# paired lines
for a, b, c, d in zip(x1, A1, x2, A2):
    ax.plot([a, c], [b, d], color='gray', lw=0.8, alpha=0.5)

ax.scatter(x1, A1, color=group_colors['plume 1'], s=40, zorder=3, label='plume 1')
ax.scatter(x2, A2, color=group_colors['plume 2'], s=40, zorder=3, label='plume 2')

# group means
ax.hlines(A1.mean(), -0.15, 0.15, color=group_colors['plume 1'], lw=2)
ax.hlines(A2.mean(),  0.85, 1.15, color=group_colors['plume 2'], lw=2)

ax.set_xticks([0, 1])
ax.set_xticklabels(['plume 1', 'plume 2'])
ax.set_ylabel(f'PVA amplitude', color=markc)

# out_dir = f'{folder_path}/fig/PVA'
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
# fig.savefig(f'{out_dir}/{filename}_plume1_vs_plume2.pdf', bbox_inches='tight')


#%% average bump shape, last 3s of returns, plume 1 vs plume 2 (von Mises fits)
from scipy.optimize import least_squares

folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'

n_wedges = 16
tail_seconds = 3
plot_per_fly = True
adj_r2_threshold = 0.5  # discard timepoints with adjusted R^2 below this

bout_groups_idx = {
    'plume 1': slice(2, 15),
    'plume 2': slice(17, 33),
}
group_colors = {
    'plume 1': 'tab:blue',
    'plume 2': 'tab:cyan',
}

# wedge angles, matching compute_fsb_upper_phase
wedge_angles = np.arange(n_wedges) * 2*np.pi/n_wedges - np.pi + np.pi/n_wedges
center_idx = n_wedges // 2

# x-axis for plotting (wedge offset from bump peak, degrees)
wedge_offsets_deg = (np.degrees(wedge_angles - wedge_angles[center_idx]) + 180) % 360 - 180
order = np.argsort(wedge_offsets_deg)
x = wedge_offsets_deg[order]

# ---- von Mises fit machinery ----
def von_mises(x, alpha, kappa, mu, c):
    return alpha * np.exp(kappa * np.cos(x - mu)) + c

def fit_von_mises(y, angles=wedge_angles):
    """
    Fit f(x) = alpha * exp(kappa * cos(x - mu)) + c by nonlinear least squares
    (trust-region reflective). Returns (params, adj_r2). params is None on failure.
    """
    if np.any(np.isnan(y)):
        return None, np.nan

    # initial guesses
    c0 = np.min(y)
    alpha0 = max((np.max(y) - c0) / np.exp(1.0), 1e-6)  # so alpha*e^kappa~peak when kappa~1
    # PVA-based initial mu
    z = np.sum(y * np.exp(1j * angles))
    mu0 = np.angle(z) if np.abs(z) > 0 else 0.0
    kappa0 = 1.0
    p0 = [alpha0, kappa0, mu0, c0]

    # bounds: alpha > 0, kappa in [0, 50], mu unconstrained (wrap later), c >= 0
    lb = [1e-9, 0.0,   -np.inf, -np.inf]
    ub = [np.inf, 50.0,  np.inf,  np.inf]

    def residuals(p):
        alpha, kappa, mu, c = p
        return von_mises(angles, alpha, kappa, mu, c) - y

    try:
        res = least_squares(residuals, p0, bounds=(lb, ub), method='trf', max_nfev=2000)
    except Exception:
        return None, np.nan

    if not res.success:
        return None, np.nan

    yhat = von_mises(angles, *res.x)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 0:
        return None, np.nan
    r2 = 1 - ss_res / ss_tot
    n, k = len(y), len(p0)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)
    return res.x, adj_r2

def bump_width_amp(alpha, kappa):
    """Bump width (rad) and amplitude from the paper's formulas."""
    if kappa <= 0 or not np.isfinite(kappa):
        return np.nan, np.nan
    # numerically stable cosh: log((exp(k)+exp(-k))/2) = log(cosh(k))
    log_cosh_k = np.log(np.cosh(kappa))
    arg = log_cosh_k / kappa
    if not (-1 <= arg <= 1):
        return np.nan, np.nan
    width = 2 * np.abs(np.arccos(arg))
    amplitude = alpha * (np.exp(kappa) - np.exp(-kappa))
    return width, amplitude

# accumulate per fly
per_file_bumps = {label: [] for label in bout_groups_idx}      # mean aligned bumps (kept for shape plot)
per_file_width = {label: [] for label in bout_groups_idx}      # mean bump width per fly (deg)
per_file_amp   = {label: [] for label in bout_groups_idx}      # mean bump amplitude per fly
per_file_names = []

for filename in os.listdir(folder_path):
    if not filename.endswith('.pkl'):
        continue

    img_dict = open_pickle(f'{folder_path}/{filename}')
    pv2, ft2 = process_CX_pickle(img_dict)
    ft2 = compute_fsb_upper_phase(ft2, n_wedges=n_wedges, roi_prefix='fsb_upper')
    time = pv2['relative_time'].to_numpy()
    in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
    off_bouts = exit_bout_indices(in_acv)

    wedge_cols = [f'{i}_fsb_upper' for i in range(n_wedges)]
    wedge_data = ft2[wedge_cols].ffill().to_numpy()
    phase_rad = ft2['fsb_upper_phase'].to_numpy()

    def collect_for_bouts(selected_bouts):
        """For each bout: aligned mean bump, mean width (deg), mean amp.
        Per-timepoint fits below adj_r2_threshold are discarded."""
        bout_means, bout_widths, bout_amps = [], [], []
        for start, end in selected_bouts:
            if end - start < 2:
                continue
            bt = time[start:end]
            if bt[-1] - bt[0] < tail_seconds:
                continue
            cutoff = bt[-1] - tail_seconds
            tail_mask = bt >= cutoff

            tail_wedges = wedge_data[start:end][tail_mask]
            if tail_wedges.shape[0] < 2:
                continue

            aligned_rows, widths, amps = [], [], []
            for i in range(tail_wedges.shape[0]):
                y = tail_wedges[i]
                if np.any(np.isnan(y)):
                    continue
                params, adj_r2 = fit_von_mises(y)
                if params is None or not np.isfinite(adj_r2) or adj_r2 < adj_r2_threshold:
                    continue
                alpha, kappa, mu, c = params

                # align this timepoint's wedges so the fit peak (mu) lands at center_idx
                peak_idx = int(np.argmin(np.abs(np.angle(
                    np.exp(1j * (wedge_angles - mu))
                ))))
                shift = center_idx - peak_idx
                aligned_rows.append(np.roll(y, shift))

                width, amp = bump_width_amp(alpha, kappa)
                if np.isfinite(width):
                    widths.append(np.degrees(width))
                if np.isfinite(amp):
                    amps.append(amp)

            if not aligned_rows:
                continue
            bout_means.append(np.mean(np.vstack(aligned_rows), axis=0))
            if widths:
                bout_widths.append(np.mean(widths))
            if amps:
                bout_amps.append(np.mean(amps))
        return bout_means, bout_widths, bout_amps

    fly_bumps_this_file = {}
    for label, sl in bout_groups_idx.items():
        bumps, widths, amps = collect_for_bouts(off_bouts[sl])
        if not bumps:
            continue
        bout_arr = np.vstack(bumps)
        fly_mean = np.nanmean(bout_arr, axis=0)
        per_file_bumps[label].append(fly_mean)
        per_file_width[label].append(np.mean(widths) if widths else np.nan)
        per_file_amp[label].append(np.mean(amps) if amps else np.nan)
        fly_bumps_this_file[label] = (bout_arr, fly_mean)

    if not fly_bumps_this_file:
        continue
    per_file_names.append(filename)

    # ---- per-fly bump-shape plot ----
    if plot_per_fly:
        fig_f, ax_f, markc_f = configure_white_plot(size=(4, 4), xaxis=True)
        for label, (bout_arr, fly_mean) in fly_bumps_this_file.items():
            color = group_colors[label]
            n_b = bout_arr.shape[0]
            for row in bout_arr[:, order]:
                ax_f.plot(x, row, color=color, lw=0.6, alpha=0.25)
            ax_f.plot(x, fly_mean[order], color=color, lw=2, marker='o', ms=4,
                      label=f'{label} (n_bouts={n_b})')

        ax_f.axvline(0, color=markc_f, ls='--', lw=0.5)
        ax_f.set_xlabel('wedge offset from bump peak (deg)', color=markc_f)
        ax_f.set_ylabel('wedge activity', color=markc_f)
        ax_f.set_xlim(-180, 180)
        ax_f.set_ylim(0, 1)
        ax_f.set_xticks(np.arange(-180, 181, 90))
        ax_f.set_title(f'{filename}, last {tail_seconds}s of returns')
        ax_f.legend(frameon=False)

        out_dir = f'{folder_path}/fig/bump_shape'
        os.makedirs(out_dir, exist_ok=True)
        fig_f.savefig(f'{out_dir}/{filename}_plume1_vs_plume2.pdf', bbox_inches='tight')
        plt.close(fig_f)

# ---- group bump-shape plot ----
fig, ax, markc = configure_white_plot(size=(4, 4), xaxis=True)
for label, bumps_list in per_file_bumps.items():
    if not bumps_list:
        continue
    color = group_colors[label]
    arr = np.vstack(bumps_list)[:, order]
    n = arr.shape[0]
    for row in arr:
        ax.plot(x, row, color=color, lw=0.8, alpha=0.25)
    mean = np.nanmean(arr, axis=0)
    sem  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    ax.plot(x, mean, color=color, lw=2, marker='o', ms=4, label=f'{label} (n={n})')
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)

ax.axvline(0, color=markc, ls='--', lw=0.5)
ax.set_xlabel('wedge offset from bump peak (deg)', color=markc)
ax.set_ylabel('wedge activity', color=markc)
ax.set_xlim(-180, 180)
ax.set_ylim(0, 1)
ax.set_xticks(np.arange(-180, 181, 90))
ax.set_title(f'all flies, last {tail_seconds}s of returns')
ax.legend(frameon=False)

out_dir = f'{folder_path}/fig/bump_shape'
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f'{out_dir}/plume1_vs_plume2.pdf', bbox_inches='tight')

# ---- per-fly bump width and amplitude, paired across plumes ----
def paired_plot(metric_dict, ylabel, fname_suffix):
    v1 = np.array(metric_dict['plume 1'])
    v2 = np.array(metric_dict['plume 2'])
    valid = ~(np.isnan(v1) | np.isnan(v2))
    v1, v2 = v1[valid], v2[valid]
    if len(v1) == 0:
        print(f'no valid pairs for {ylabel}')
        return

    fig, ax, markc = configure_white_plot(size=(2, 3), xaxis=True)
    x1 = np.zeros_like(v1) + np.random.uniform(-0.05, 0.05, size=v1.shape)
    x2 = np.ones_like(v2)  + np.random.uniform(-0.05, 0.05, size=v2.shape)
    for a, b, c, d in zip(x1, v1, x2, v2):
        ax.plot([a, c], [b, d], color='gray', lw=0.8, alpha=0.5)
    ax.scatter(x1, v1, color=group_colors['plume 1'], s=40, zorder=3, label='plume 1')
    ax.scatter(x2, v2, color=group_colors['plume 2'], s=40, zorder=3, label='plume 2')
    ax.hlines(v1.mean(), -0.15, 0.15, color=group_colors['plume 1'], lw=2)
    ax.hlines(v2.mean(),  0.85, 1.15, color=group_colors['plume 2'], lw=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['plume 1', 'plume 2'])
    ax.set_ylabel(ylabel, color=markc)
    ax.set_title(f'n={len(v1)}')

    from scipy.stats import wilcoxon, ttest_rel
    try:
        _, w_p = wilcoxon(v1, v2)
        _, t_p = ttest_rel(v1, v2)
        ax.text(0.5, 0.02, f'Wilcoxon p={w_p:.3f}\npaired t p={t_p:.3f}',
                transform=ax.transAxes, ha='center', va='bottom', color=markc, fontsize=9)
    except Exception as e:
        print(f'stats skipped: {e}')

    fig.savefig(f'{out_dir}/{fname_suffix}.pdf', bbox_inches='tight')

paired_plot(per_file_width, 'bump width (deg, von Mises fit)', 'bump_width_plume1_vs_plume2')
paired_plot(per_file_amp,   'bump amplitude (von Mises fit)',  'bump_amplitude_plume1_vs_plume2')


#%% bump shape concentration (sharpness) in last 3s, plume 1 vs plume 2
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez/PAM_train'

n_wedges = 16
tail_seconds = 3

bout_groups_idx = {
    'plume 1': slice(2, 15),
    'plume 2': slice(17, 33),
}
group_colors = {
    'plume 1': 'tab:blue',
    'plume 2': 'tab:cyan',
}

# wedge angles must match those used in compute_fsb_upper_phase
wedge_angles = np.arange(n_wedges) * 2*np.pi/n_wedges - np.pi + np.pi/n_wedges

def bump_concentration(wedges_row):
    """
    Per-timepoint bump sharpness, R in [0, 1].
    R = |sum a_i * exp(i*theta_i)| / sum a_i
    Negative or zero wedge values are clipped to 0 so they don't act
    like 'pulling backwards'; this matters for raw F or noisy dF/F.
    """
    a = np.clip(wedges_row, 0, None)
    s = a.sum()
    if s <= 0 or np.isnan(s):
        return np.nan
    z = np.sum(a * np.exp(1j * wedge_angles))
    return np.abs(z) / s

# accumulate one mean concentration per fly per plume
per_file_R = {label: [] for label in bout_groups_idx}
file_labels = []

for filename in os.listdir(folder_path):
    if not filename.endswith('.pkl'):
        continue

    img_dict = open_pickle(f'{folder_path}/{filename}')
    pv2, ft2 = process_CX_pickle(img_dict)
    ft2 = compute_fsb_upper_phase(ft2, n_wedges=n_wedges, roi_prefix='fsb_upper')
    time = pv2['relative_time'].to_numpy()
    in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
    off_bouts = exit_bout_indices(in_acv)

    wedge_cols = [f'{i}_fsb_upper' for i in range(n_wedges)]
    wedge_data = ft2[wedge_cols].ffill().to_numpy()  # (T, n_wedges)

    # precompute per-timepoint concentration for the whole recording
    R_t = np.array([bump_concentration(row) for row in wedge_data])  # (T,)

    def collect_bout_means(selected_bouts):
        bout_means = []
        for start, end in selected_bouts:
            if end - start < 2:
                continue
            bt = time[start:end]
            if bt[-1] - bt[0] < tail_seconds:
                tail_mask = np.ones_like(bt, dtype=bool)
            else:
                cutoff = bt[-1] - tail_seconds
                tail_mask = bt >= cutoff
            tail_R = R_t[start:end][tail_mask]
            tail_R = tail_R[~np.isnan(tail_R)]
            if len(tail_R) == 0:
                continue
            bout_means.append(np.mean(tail_R))
        return np.array(bout_means)

    fly_has_data = False
    for label, sl in bout_groups_idx.items():
        bout_R = collect_bout_means(off_bouts[sl])
        if len(bout_R) == 0:
            per_file_R[label].append(np.nan)
            continue
        per_file_R[label].append(np.mean(bout_R))  # one number per fly per plume
        fly_has_data = True
    if fly_has_data:
        file_labels.append(filename)

# ---- plot: per-fly R, paired across plumes ----
fig, ax, markc = configure_white_plot(size=(3,4), xaxis=True)

R1 = np.array(per_file_R['plume 1'])
R2 = np.array(per_file_R['plume 2'])
valid = ~(np.isnan(R1) | np.isnan(R2))
R1, R2 = R1[valid], R2[valid]

x1 = np.zeros_like(R1) + np.random.uniform(-0.05, 0.05, size=R1.shape)
x2 = np.ones_like(R2)  + np.random.uniform(-0.05, 0.05, size=R2.shape)

# paired lines
for a, b, c, d in zip(x1, R1, x2, R2):
    ax.plot([a, c], [b, d], color='gray', lw=0.8, alpha=0.5)

ax.scatter(x1, R1, color=group_colors['plume 1'], s=40, zorder=3, label='plume 1')
ax.scatter(x2, R2, color=group_colors['plume 2'], s=40, zorder=3, label='plume 2')

# group means
ax.hlines(R1.mean(), -0.15, 0.15, color=group_colors['plume 1'], lw=2)
ax.hlines(R2.mean(),  0.85, 1.15, color=group_colors['plume 2'], lw=2)

ax.set_xticks([0, 1])
ax.set_xticklabels(['plume 1', 'plume 2'])
ax.set_ylabel('bump concentration R\n(sharpness across wedges)', color=markc)
ax.set_ylim(0, 1)
ax.set_title(f'last {tail_seconds}s of returns (n={len(R1)} flies)')

# paired stats
from scipy.stats import wilcoxon, ttest_rel
try:
    w_stat, w_p = wilcoxon(R1, R2)
    t_stat, t_p = ttest_rel(R1, R2)
    ax.text(0.5, 0.02,
            f'Wilcoxon p={w_p:.3f}\npaired t p={t_p:.3f}',
            transform=ax.transAxes, ha='center', va='bottom', color=markc, fontsize=9)
except Exception as e:
    print(f'stats skipped: {e}')


#%% bump amplitude vs entry number, plume 1 vs plume 2
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez/PAM_all'

n_wedges = 16
tail_seconds = 3

plume_entry_ranges = {
    'plume 1': (2, 16),    # entries 2..14 inclusive
    'plume 2': (18, 35),   # entries 17..32 inclusive
}
plume_colors = {
    'plume 1': 'tab:blue',
    'plume 2': 'tab:cyan',
}

# per-fly traces: list of (n_entries_within_plume,) arrays, NaN for missing entries
per_fly_traces = {plume: [] for plume in plume_entry_ranges}

for filename in os.listdir(folder_path):
    if not filename.endswith('.pkl'):
        continue

    img_dict = open_pickle(f'{folder_path}/{filename}')
    pv2, ft2 = process_CX_pickle(img_dict)
    ft2 = compute_fsb_upper_phase(ft2, n_wedges=n_wedges, roi_prefix='fsb_upper')
    time = pv2['relative_time'].to_numpy()
    in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
    off_bouts = exit_bout_indices(in_acv)

    amp = ft2['fsb_upper_amp'].to_numpy()  # PVA amplitude

    for plume, (lo, hi) in plume_entry_ranges.items():
        entry_nums = np.arange(lo, hi)
        trace = np.full(len(entry_nums), np.nan)

        for i, e in enumerate(entry_nums):
            if e >= len(off_bouts):
                continue
            start, end = off_bouts[e]
            if end - start < 2:
                continue
            bt = time[start:end]
            if bt[-1] - bt[0] < tail_seconds:
                tail_mask = np.ones_like(bt, dtype=bool)
            else:
                cutoff = bt[-1] - tail_seconds
                tail_mask = bt >= cutoff

            tail_amp = amp[start:end][tail_mask]
            tail_amp = tail_amp[~np.isnan(tail_amp)]
            if len(tail_amp) == 0:
                continue
            trace[i] = np.mean(tail_amp)

        if not np.all(np.isnan(trace)):
            per_fly_traces[plume].append(trace)

# ---- plot: one panel per plume, individual flies + group mean ± SEM ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, (plume, (lo, hi)) in zip(axes, plume_entry_ranges.items()):
    entry_nums = np.arange(lo, hi)
    x_within = entry_nums - lo + 1   # 1-indexed within-plume entry number
    color = plume_colors[plume]

    traces = per_fly_traces[plume]
    if not traces:
        ax.set_title(f'{plume} (no data)')
        continue
    arr = np.vstack(traces)  # (n_flies, n_entries)

    # individual flies (faint)
    for row in arr:
        ax.plot(x_within, row, color=color, lw=0.8, alpha=0.3)

    # group mean ± SEM, only where we have data
    mean = np.nanmean(arr, axis=0)
    n_per_entry = np.sum(~np.isnan(arr), axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n_per_entry, 1))
    valid = n_per_entry > 0
    ax.plot(x_within[valid], mean[valid], color=color, lw=2.5, marker='o', ms=5,
            label=f'mean (n={arr.shape[0]} flies)')
    ax.fill_between(x_within[valid], (mean - sem)[valid], (mean + sem)[valid],
                    color=color, alpha=0.2, lw=0)

    ax.set_xlabel(f'{plume} entry # (within plume)')
    ax.set_title(plume)
    ax.legend(frameon=False, fontsize=9)

axes[0].set_ylabel(f'bump amplitude (PVA, last {tail_seconds}s)')
fig.suptitle('within-plume time course of bump amplitude')
fig.tight_layout()


#%% mean phase shift in last 3s, per entry number
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'

tail_seconds = 1
heading_window_seconds = 1
plot_per_fly = True

def circ_mean_deg(angles_rad):
    """Circular mean in degrees, ignoring NaN."""
    a = angles_rad[~np.isnan(angles_rad)]
    if len(a) == 0:
        return np.nan
    return np.degrees(np.arctan2(np.nanmean(np.sin(a)), np.nanmean(np.cos(a))))

# accumulate (entry_num, mean_shift_deg) per fly
per_file_entries = []  # list of (filename, entries_dict) where entries_dict[entry_num] = shift_deg

out_dir = f'{folder_path}/fig/phase_shift_per_entry'
os.makedirs(out_dir, exist_ok=True)

for filename in os.listdir(folder_path):
    if not filename.endswith('.pkl'):
        continue

    img_dict = open_pickle(f'{folder_path}/{filename}')
    pv2, ft2 = process_CX_pickle(img_dict)
    ft2 = compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper')
    time = pv2['relative_time'].to_numpy()
    in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
    off_bouts = exit_bout_indices(in_acv)

    phase_rad = ft2['fsb_upper_phase'].to_numpy()
    heading_rad = ft2['ft_heading'].to_numpy()

    # ---- infer plume 1 side from sin(heading) in last 1s of plume-1 off-bouts ----
    side_samples = []
    for start, end in off_bouts[plume1_slice]:
        if end - start < 2:
            continue
        bt = time[start:end]
        if bt[-1] - bt[0] < heading_window_seconds:
            continue
        cutoff = bt[-1] - heading_window_seconds
        mask = bt >= cutoff
        side_samples.append(np.sin(heading_rad[start:end][mask]))

    if not side_samples:
        print(f'{filename}: skipped (no usable bouts to infer side)')
        continue
    mean_sin = np.nanmean(np.concatenate(side_samples))
    flip = -1.0 if mean_sin > 0 else 1.0  # plume 1 -> negative (left)

    # ---- one mean phase shift per bout ----
    entries = {}    # entry_num -> shift_deg
    bout_colors = []  # for per-fly plot

    for entry_num, (start, end) in enumerate(off_bouts):
        if end - start < 2:
            continue
        bout_phase = phase_rad[start:end]
        bt = time[start:end]
        if bt[-1] - bt[0] < tail_seconds:
            tail_mask = np.ones_like(bt, dtype=bool)  # use whole bout
        else:
            cutoff = bt[-1] - tail_seconds
            tail_mask = bt >= cutoff

        # phase change from bout onset, kept circular
        delta = bout_phase[tail_mask] - bout_phase[0]
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        delta = delta * flip

        shift_deg = circ_mean_deg(delta)
        if np.isnan(shift_deg):
            continue
        entries[entry_num] = shift_deg

    if not entries:
        continue
    per_file_entries.append((filename, entries))

    # ---- per-fly plot ----
    if plot_per_fly:
        fig_f, ax_f, markc_f = configure_white_plot(size=(5,2), xaxis=True)
        xs = np.array(sorted(entries.keys()))
        ys = np.array([entries[k] for k in xs])

        # color points by which plume they belong to
        def bout_color(i):
            if plume1_slice.start <= i < plume1_slice.stop:
                return 'tab:blue'
            if plume2_slice.start <= i < plume2_slice.stop:
                return 'tab:cyan'
            return 'lightgray'
        colors = [bout_color(i) for i in xs]

        ax_f.scatter(xs, ys, c=colors, s=40, zorder=3)
        ax_f.plot(xs, ys, color='gray', lw=0.5, alpha=0.5, zorder=1)
        ax_f.axhline(0, color=markc_f, ls='--', lw=0.5)

        ax_f.set_xlabel('entry #', color=markc_f)
        ax_f.set_ylabel(f'mean phase shift in last {tail_seconds}s (deg)', color=markc_f)
        ax_f.set_ylim(-180, 180)
        ax_f.set_yticks(np.arange(-180, 181, 90))
        ax_f.set_title(f'{filename}')
        plt.show()
        # fig_f.savefig(f'{out_dir}/{filename}_phase_shift_per_entry.pdf',
        #               bbox_inches='tight')
        plt.close(fig_f)

# ---- group summary: align by entry number, mean ± SEM across flies ----
if per_file_entries:
    all_entry_nums = sorted({k for _, e in per_file_entries for k in e.keys()})
    n_entries = max(all_entry_nums) + 1
    n_flies = len(per_file_entries)

    # build (n_flies, n_entries) matrix; nan where a fly didn't have that entry
    mat = np.full((n_flies, n_entries), np.nan)
    for fi, (_, entries) in enumerate(per_file_entries):
        for k, v in entries.items():
            mat[fi, k] = v

    # circular mean across flies per entry, with per-entry SEM on the linear values
    # (linear SEM is fine here because shifts are bounded near 0 after side-flip;
    # if entries scatter near +/-180, switch to circular dispersion.)
    counts = np.sum(~np.isnan(mat), axis=0)
    mean = np.full(n_entries, np.nan)
    sem  = np.full(n_entries, np.nan)
    for j in range(n_entries):
        col = mat[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        mean[j] = circ_mean_deg(np.radians(col))
        if len(col) > 1:
            sem[j] = np.std(col, ddof=1) / np.sqrt(len(col))

    fig, ax, markc = configure_white_plot(size=(7, 4), xaxis=True)

    # individual flies, faint
    for fi, (_, entries) in enumerate(per_file_entries):
        xs = np.array(sorted(entries.keys()))
        ys = np.array([entries[k] for k in xs])
        ax.plot(xs, ys, color='gray', lw=0.5, alpha=0.3, marker='o', ms=2)

    # group mean
    valid = ~np.isnan(mean)
    xs_v = np.arange(n_entries)[valid]
    ax.plot(xs_v, mean[valid], color='black', lw=2, marker='o', ms=5,
            label=f'group mean (n={n_flies} flies)')
    ax.fill_between(xs_v, mean[valid] - sem[valid], mean[valid] + sem[valid],
                    color='black', alpha=0.2, lw=0)

    # shade plume 1 / plume 2 entry ranges
    ax.axvspan(plume1_slice.start - 0.5, plume1_slice.stop - 0.5,
               color='tab:blue', alpha=0.08, label='plume 1')
    ax.axvspan(plume2_slice.start - 0.5, plume2_slice.stop - 0.5,
               color='tab:cyan', alpha=0.08, label='plume 2')

    ax.axhline(0, color=markc, ls='--', lw=0.5)
    ax.set_xlabel('entry #', color=markc)
    ax.set_ylabel(f'mean phase shift in last {tail_seconds}s (deg)', color=markc)
    ax.set_ylim(-180, 180)
    ax.set_yticks(np.arange(-180, 181, 90))
    ax.set_title(f'phase shift per entry, all flies')
    ax.legend(frameon=False, loc='best')

    # fig.savefig(f'{out_dir}/group_phase_shift_per_entry.pdf', bbox_inches='tight')


#%% plume 1 and plume 2 bump amplitude over pseudotime (PVA)
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'
n_pseudotime = 100  # number of points along pseudotime axis

for filename in os.listdir(folder_path):
    if filename.endswith('.pkl'):
        img_dict = open_pickle(f'{folder_path}/{filename}')
        pv2, ft2 = process_CX_pickle(img_dict)
        ft2 = compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper')
        time = pv2['relative_time'].to_numpy()
        in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)
        off_bouts = exit_bout_indices(in_acv)

        bump_amp = ft2['fsb_upper_amp'].to_numpy()

        bout_groups = {
            'plume 1': off_bouts[2:],
            'plume 2': off_bouts[:1],
        }
        group_colors = {
            'plume 1': 'tab:blue',
            'plume 2': 'tab:green',
        }

        pseudotime = np.linspace(0, 1, n_pseudotime)

        def collect_interp(selected_bouts):
            traces = []
            for start, end in selected_bouts:
                if end - start < 2:
                    continue
                bout_amp = bump_amp[start:end]
                if np.all(np.isnan(bout_amp)):
                    continue
                # original normalized time within the bout, 0 -> 1
                orig_pt = np.linspace(0, 1, len(bout_amp))
                # drop NaNs before interpolation
                mask = ~np.isnan(bout_amp)
                if mask.sum() < 2:
                    continue
                interp = np.interp(pseudotime, orig_pt[mask], bout_amp[mask])
                traces.append(interp)
            if not traces:
                return None
            return np.vstack(traces)  # (n_bouts, n_pseudotime)

        fig, ax, markc = configure_white_plot(size=(6, 4), xaxis=True)

        plotted_any = False
        for label, selected_bouts in bout_groups.items():
            traces = collect_interp(selected_bouts)
            if traces is None:
                continue
            color = group_colors[label]

            # individual bouts (faint)
            for tr in traces:
                ax.plot(pseudotime, tr, color=color, lw=0.5, alpha=0.25)

            # group mean +/- SEM
            mean = np.nanmean(traces, axis=0)
            sem = np.nanstd(traces, axis=0, ddof=1) / np.sqrt(traces.shape[0])
            ax.plot(pseudotime, mean, color=color, lw=2, label=f'{label} (n={traces.shape[0]})')
            ax.fill_between(pseudotime, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)
            plotted_any = True

        if not plotted_any:
            continue

        ax.set_xlabel('pseudotime (bout onset → end)', color=markc)
        ax.set_ylabel('PVA amplitude', color=markc)
        ax.set_xlim(0, 1)
        ax.set_title(f'{filename}, bump amplitude (PVA) over pseudotime')
        ax.legend(frameon=False)

#%% PVA over time
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'
window_sec = 1.0

for filename in os.listdir(folder_path):
    if filename.endswith('.pkl'):
        img_dict = open_pickle(f'{folder_path}/{filename}')
        pv2, ft2 = process_CX_pickle(img_dict)
        ft2 = compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper')
        time = pv2['relative_time'].to_numpy()
        in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)

        bump_amp = ft2['fsb_upper_amp'].to_numpy()
        entry = np.arange(len(bump_amp))

        # estimate sampling rate from median dt, then convert window to samples
        dt = np.median(np.diff(time))
        win = max(1, int(round(window_sec / dt)))

        # centered rolling mean, NaN-aware
        bump_amp_smooth = (
            pd.Series(bump_amp)
              .rolling(window=win, center=True, min_periods=1)
              .mean()
              .to_numpy()
        )

        fig, ax, markc = configure_white_plot(size=(10, 3), xaxis=True)

        # shade odor-on periods
        if in_acv.any():
            diffs = np.diff(in_acv.astype(int))
            starts = np.where(diffs == 1)[0] + 1
            ends = np.where(diffs == -1)[0] + 1
            if in_acv[0]:
                starts = np.r_[0, starts]
            if in_acv[-1]:
                ends = np.r_[ends, len(in_acv)]
            for s, e in zip(starts, ends):
                ax.axvspan(s, min(e, len(bump_amp) - 1),
                           color='tab:red', alpha=0.15, lw=0)

        ax.plot(entry, bump_amp, color='lightgray', lw=0.5, label='raw')
        ax.plot(entry, bump_amp_smooth, color='k', lw=1,
                label=f'{window_sec:g}s rolling mean')

        ax.set_xlabel('entry number', color=markc)
        ax.set_ylabel('PVA amplitude', color=markc)
        ax.set_xlim(0, len(bump_amp) - 1)
        ax.set_title(f'{filename}, bump amplitude (PVA) over trial')
        ax.legend(frameon=False)


#%% triggered average PVA
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'
window_sec = 1.0       # smoothing window
pre_sec = 3.0          # seconds before entry to show
post_sec = 3.0        # seconds after entry to show (set to 0 if you only want pre)

for filename in os.listdir(folder_path):
    if filename.endswith('.pkl'):
        img_dict = open_pickle(f'{folder_path}/{filename}')
        pv2, ft2 = process_CX_pickle(img_dict)
        ft2 = compute_fsb_upper_phase(ft2, n_wedges=16, roi_prefix='fsb_upper')
        time = pv2['relative_time'].to_numpy()
        in_acv = (ft2['mfc2_stpt'].to_numpy() != 0).astype(bool)

        bump_amp = ft2['fsb_upper_amp'].to_numpy()

        # sampling rate + smoothing
        dt = np.median(np.diff(time))
        win = max(1, int(round(window_sec / dt)))
        bump_amp_smooth = (
            pd.Series(bump_amp)
              .rolling(window=win, center=True, min_periods=1)
              .mean()
              .to_numpy()
        )

        # find plume entry indices: 0 -> 1 transitions in in_acv
        # find plume entry indices: 0 -> 1 transitions in in_acv
        diffs = np.diff(in_acv.astype(int))
        entry_idx = np.where(diffs == 1)[0] + 1
        if in_acv[0]:
            entry_idx = np.r_[0, entry_idx]

        # only keep the first N entries
        max_entries = 15
        entry_idx = entry_idx[:max_entries]

        if len(entry_idx) == 0:
            continue

        pre_samps = int(round(pre_sec / dt))
        post_samps = int(round(post_sec / dt))
        n_total = pre_samps + post_samps + 1
        rel_time = (np.arange(n_total) - pre_samps) * dt  # time relative to entry (s)

        # collect aligned snippets, padding out-of-range with NaN
        snippets = []
        for ei in entry_idx:
            s = ei - pre_samps
            e = ei + post_samps + 1
            snip = np.full(n_total, np.nan)
            src_s = max(s, 0)
            src_e = min(e, len(bump_amp_smooth))
            dst_s = src_s - s
            dst_e = dst_s + (src_e - src_s)
            snip[dst_s:dst_e] = bump_amp_smooth[src_s:src_e]
            snippets.append(snip)

        snippets = np.vstack(snippets)  # (n_entries, n_total)

        # mean +/- SEM across entries (NaN-aware)
        n_valid = np.sum(~np.isnan(snippets), axis=0)
        mean = np.nanmean(snippets, axis=0)
        sem = np.nanstd(snippets, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))

        fig, ax, markc = configure_white_plot(size=(3,4), xaxis=True)

        # individual entries (faint)
        for snip in snippets:
            ax.plot(rel_time, snip, color='tab:blue', lw=0.5, alpha=0.25)

        # group mean +/- SEM
        ax.plot(rel_time, mean, color='tab:blue', lw=2,
                label=f'mean (n={snippets.shape[0]})')
        ax.fill_between(rel_time, mean - sem, mean + sem,
                        color='tab:blue', alpha=0.2, lw=0)

        # mark plume entry
        ax.axvline(0, color=markc, ls='--', lw=1)

        ax.set_xlabel('time relative to plume entry (s)', color=markc)
        ax.set_ylabel('PVA amplitude', color=markc)
        ax.set_xlim(rel_time[0], rel_time[-1])
        ax.set_title(f'{filename}, entry-triggered PVA amplitude')
        ax.legend(frameon=False)


# %%
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'
for filename in os.listdir(folder_path):
    if filename.endswith('.pkl'):
        print(filename)
        img_dict = open_pickle(f'{folder_path}/{filename}')
        pv2, ft2 = process_CX_pickle(img_dict)
        print(list(ft2.columns))
        x,y = fictrac_repair(ft2['ft_posx'],ft2['ft_posy']-yo)
        ft2_odor = ft2.where(ft2['mfc2_stpt']>0)
        xodor,yodor = fictrac_repair(ft2_odor['ft_posx'],ft2_odor['ft_posy']-yo)
        xo = ft2['ft_posx'].iloc[0]
        yo = ft2['ft_posy'].iloc[0]
        plt.plot(x-xo, y-yo, color='k')
        plt.plot(xodor-xo, yodor-yo, color='lightgrey')
        plt.title(filename)

# %%
folder_path = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_FC2/picklez'
for filename in os.listdir(folder_path):
    if filename.endswith('.pkl'):
        print(filename)
        img_dict = open_pickle(f'{folder_path}/{filename}')
        pv2, ft2 = process_CX_pickle(img_dict)
        print(list(ft2.columns))
        x,y = fictrac_repair(ft2['ft_posx'],ft2['ft_posy']-yo)
        ft2_odor = ft2.where(ft2['mfc2_stpt']>0)
        xodor,yodor = fictrac_repair(ft2_odor['ft_posx'],ft2_odor['ft_posy']-yo)
        xo = ft2['ft_posx'].iloc[0]
        yo = ft2['ft_posy'].iloc[0]
        plt.plot(x-xo, y-yo, color='k')
        plt.plot(xodor-xo, yodor-yo, color='lightgrey')
        plt.title(filename)

#%%
folder_path = '/Volumes/LaCie/noelle_imaging/PAM_CsChr_FC2_GC7f/260430/F3/T3/data'
trajectory_plotter_bw(folder_path, 50, 1000, 0, [-500, 500], [0,1000], led='red', hlines=None, select_file= None, plot_type='odor', save=True)
# %%
