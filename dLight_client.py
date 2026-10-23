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
from scipy.stats import zscore, pearsonr
import sys
import pickle
from sklearn.linear_model import LinearRegression


#%% BATCH EXPERIMENT TRACE

folder = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/KC_dLight/picklez/bl_et'
lobes = ['g2', 'g3', 'g4', 'g5'] #, 'b1', 'b2']
lobes = ['g4b', 'g4t', 'g5b', 'g5t'] #, 'b1', 'b2']

colors = ['purple', 'pink', 'blue', 'lightblue'] #, 'orange', 'grey']

pkl_files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pkl'))

for filename in pkl_files:
    print(f"\nProcessing: {filename}")
    data = open_pickle(filename)
    pv2, ft2 = process_dLight_pickle(data, lobes)
 
    fig, axs, markc = configure_bw_plot(size=(8, 6), xaxis=True, nrows=len(lobes) + 1)
    fig.suptitle(filename)

    net_motion = ft2['net_motion'].to_numpy()
    time = pv2['relative_time'].to_numpy()
    odor1_on = ft2['mfc3_stpt'].to_numpy()
    odor2_on = ft2['mfc4_stpt'].to_numpy()
    odor2_on = pd.Series(odor2_on).ffill().bfill().to_numpy()

    for i, lobe in enumerate(lobes):
        FF = pv2[f'0_{lobe}'].to_numpy()
        shade_on_intervals(axs[i], time, odor1_on, mode='span', alpha=0.45, color='#fbceb1')
        shade_on_intervals(axs[i], time, odor2_on, mode='span', alpha=0.45,  color='#7ed4e6')

        axs[i].set_ylabel('dF/F', color=markc)

        # Reshape for sklearn
        loc_reshaped = net_motion.reshape(-1, 1)
        # Fit linear model: FF ~ motion
        reg = LinearRegression().fit(loc_reshaped, FF)
        # Predicted FF due to motion alone
        FF_motion_predicted = reg.predict(loc_reshaped)
        # Residuals = FF with motion contribution removed
        FF_regressed = FF - FF_motion_predicted
        # Add back the mean so the trace stays in a familiar range
        FF_regressed = FF_regressed + FF.mean()
        print(f"  {lobe} R² (variance explained by motion): {reg.score(loc_reshaped, FF):.3f}")

        axs[i].plot(time, FF, linewidth=0.75, color=colors[i])

    ax = axs[len(lobes)]
    ax.plot(time, net_motion, linewidth=1, color=markc)

    d = filename.split(os.path.sep)
    fly = d[-3] + '_' + d[-2] + '_' + d[-1]
    fly = fly.replace('.pkl', '')

    plt.tight_layout()
    if not os.path.exists(f'{folder}/fig/traces'):
        os.makedirs(f'{folder}/fig/traces')
    fig.savefig(f'{folder}/fig/traces/{fly}_trace.pdf', bbox_inches='tight')


#%% EXPERIMENT TRACE
figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/KC_dLight/analysis'
# colors = ['#7fffd4', '#08e8de','#ff55a3', '#ff1493', 'white', 'grey']
fig, axs, markc = configure_bw_plot(size=(8,6), xaxis=True, nrows=len(lobes)+1)
fig.suptitle(filename)

net_motion = ft2['net_motion'].to_numpy()
time = pv2['relative_time'].to_numpy()
odor1_on = ft2['mfc3_stpt'].to_numpy()
odor2_on = ft2['mfc4_stpt'].to_numpy()


for i, lobe in enumerate(lobes):
    FF = pv2[f'0_{lobe}'].to_numpy()
    shade_on_intervals(axs[i], time, odor1_on, alpha=0.3, color='#fbceb1')
    shade_on_intervals(axs[i], time, odor2_on, alpha=0.3, color='#d473d4')    
    axs[i].set_ylabel('dF/F', color=markc)

    # Reshape for sklearn
    loc_reshaped = net_motion.reshape(-1, 1)
    # Fit linear model: FF ~ loc
    reg = LinearRegression().fit(loc_reshaped, FF)
    # Predicted FF due to motion alone
    FF_motion_predicted = reg.predict(loc_reshaped)
    # Residuals = FF with motion contribution removed
    FF_regressed = FF - FF_motion_predicted
    # Optionally, add back the mean so the trace stays in a familiar range
    FF_regressed = FF_regressed + FF.mean()
    print(f"R² (variance explained by motion): {reg.score(loc_reshaped, FF):.3f}")

    axs[i].plot(time, FF_regressed, linewidth=0.75, color=colors[i])
    # axs[i].plot(time, FF_regressed, linewidth=0.75, color=colors[i])

    

ax = axs[len(lobes)]
ax.plot(time, net_motion, linewidth=1, color=markc)
d = filename.split(os.path.sep)
fly = d[-3] + '_' + d[-2] + '_' + d[-1]
plt.tight_layout()
plt.show()
# if not os.path.exists(f'{figure_folder}/fig/traces'):
#     os.makedirs(f'{figure_folder}/fig/traces')
# fig.savefig(f'{figure_folder}/fig/traces/{fly}_trace.pdf', bbox_inches='tight')

#%% rebaseline if needed
savedir = os.path.join('/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/KC_dLight/picklez')
lobes = ['g4b', 'g4t', 'g5b', 'g5t']
rebaseline(savedir, lobes, span=500)


  #%% EXPERIMENT TRACE + COMPARTMENT CORRELATION

# fig, axs, markc = configure_bw_plot(size=(10, 7),xaxis=True,nrows=2)

# fig.suptitle(filename)

# ax_traces = axs[0]
# ax_corr   = axs[1]

# # --- data ---
# FF_g2 = pv2['0_g2'].to_numpy()
# FF_g4 = pv2['0_g4'].to_numpy()
# time  = pv2['relative_time'].to_numpy()

# odor1_on = ft2['mfc2_stpt'].to_numpy()
# odor2_on = ft2['mfc3_stpt'].to_numpy()

# # --- top: raw traces ---
# ax_traces.plot(time, FF_g2, linewidth=1, color=colors[1], label='g2')
# ax_traces.plot(time, FF_g4, linewidth=1, color=colors[3], label='g4')

# shade_on_intervals(ax_traces, time, odor1_on, alpha=0.3, color='#fbceb1')
# shade_on_intervals(ax_traces, time, odor2_on, alpha=0.3, color='#d473d4')

# ax_traces.set_ylabel('dF/F', color=markc)
# ax_traces.legend(frameon=False)

# # --- sliding correlation ---
# t_corr, r_corr = sliding_pearson(
#     FF_g2, FF_g4, time,
#     window_s=20,
#     step_s=1
# )

# ax_corr.plot(t_corr, r_corr, color=markc, linewidth=1)
# ax_corr.axhline(0, linestyle='--', color='gray', linewidth=1)

# # keep odor shading aligned
# shade_on_intervals(ax_corr, time, odor1_on, alpha=0.2, color='#fbceb1')
# shade_on_intervals(ax_corr, time, odor2_on, alpha=0.2, color='#7ed4e6')

# ax_corr.set_ylabel('pearson r', color=markc)
# ax_corr.set_xlabel('time (s)', color=markc)
# ax_corr.set_ylim(-1, 1)

# # --- overall Pearson (descriptive only) ---
# FF_g2_z = zscore(FF_g2, nan_policy='omit')
# FF_g4_z = zscore(FF_g4, nan_policy='omit')

# r, p = pearsonr(FF_g2_z, FF_g4_z)
# print(f"Pearson r = {r:.3f}, p = {p:.3e}")

# plt.tight_layout()
# plt.show()




#%%
# et_replay_traces_bw_new(figure_folder, lobes, colors, size = (12,6))
# et_replay_traces_ne(figure_folder, lobes, colors, size = (12,6))
# et_replay_auc_comp(figure_folder, lobes, colors, size=(6,6))
# et_replay_peak_comp(figure_folder, lobes, colors, oct=True, size=(6,6))
# et_replay_tuning(figure_folder, lobes, colors, size=(10,4))
et_replay_traj_bw(figure_folder, lobes, colors, size=(10,5))

# %%
figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/FB4R/ne/replay/oct'

# lobes = ['g2',  'g3', 'g4', 'g5'] #, 'b1', 'b2']
# lobes = ['g4b',  'g4t', 'g5b', 'g5t'] #, 'b1', 'b2']
groups = {'PPL': ['g2', 'g3'], 'PAM': ['g4', 'g5']}


colors = ['white', '#08e8de','#ff55a3', '#ff1493'] #, '#9966cc', '#ffbcd9']
# lobes = ['g2', 'g4']
# colors = ['#7fffd4', '#ff55a3']

# et_replay_traces_bw_new(figure_folder, ['fsbtn'], colors, size = (14,4), xlim=(100,400))
# et_replay_oct_peak_comp(figure_folder, lobes, colors, size=(8,6),zscore=True)
# triggered_zFF_dLight(figure_folder, lobes, tbef=2, taf=2, event_type="entry", first=False,colors=colors, odor_source="auto", max_events=None, zscore_mode="raw")
triggered_zFF_dLight_summary(figure_folder, lobes, tbef=2, taf=2, event_type="entry", colors=colors, odor_source="auto", max_events=None,zscore_mode="raw", min_duration=0.5,per_file_overlay=True, error="sem")
# triggered_zFF_dLight_summary(figure_folder, lobes, tbef=2, taf=2, event_type="entry", colors=colors, odor_source="auto", max_events=None,zscore_mode="baseline", min_duration=0.5,per_file_overlay=True, error="sem")
# triggered_loc_dLight(figure_folder, lobes, cols=['net_motion', 'y_velocity', 'ang_velocity', 'ft_heading'], tbef=3, taf=3, event_type="entry", first=False,colors=colors, odor_source="auto", max_events=None, zscore_mode="trial")
# DAN_corr_return_efficiency(figure_folder,lobes, pre_sec=1.0,odor_source="auto", exclude_nonpositive_baseline=True)
# int_relative_activity_avg(figure_folder, lobes, pre_sec=1.0, odor_source="auto", event_type='entry', max_bouts=None, use_regression=False,metric='integral')
# int_relative_activity_avg(figure_folder, lobes, pre_sec=1.0, odor_source="auto", event_type='exit', max_bouts=None, use_regression=True,metric='peak')
# DAN_corr(figure_folder, lobes)
# et_compartment_corrs(figure_folder, window_s=10, rois=lobes,colors=colors)
# dLight_traj_bw(figure_folder, lobes, colors, size=(6,6))

# grouped_odor_response(figure_folder, groups, event_type='entry',min_bout_sec=1.0, use_regression=True, metric='integral', peak_window=2.0)
# 

# %% ANDY REPLAY DATA
figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/KCdLight_replay1'
lobes = ['net_motion', 'G3'] #, 'b1', 'b2']
colors = ['white', '#08e8de'] #, '#9966cc', '#ffbcd9']
# et_compartment_corrs_afs(figure_folder,window_s=5,rois=lobes,colors=colors,size=(12,4),step_s=1)
# et_replay_traj_bw(figure_folder, lobes, colors, size=(10,5))

# %%
