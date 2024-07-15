# %%
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal as sg
import os
import matplotlib.animation as animation
from scipy.spatial.distance import euclidean
import matplotlib.ticker as ticker
import warnings
from scipy import interpolate
from scipy import stats
import seaborn as sns
import statistics
import csv
from behavior_analysis import *

folder = "/Users/noelleeghbali/Desktop/exp/tethered_behavior/summer_2024_exp/analysis/gKCs_GtACR"
figure_folder = "/Users/noelleeghbali/Desktop/exp/tethered_behavior/summer_2024_exp/analysis/gKCs_GtACR/figure"
folders = [folder]
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)
n_flies = 9
c1 = '#1e90ff'
c2 = '#ff1dce'
c3 = '#ff3800'

# %% Upwind distance tracked when LED on at y=0 or y=500
keys_list = [f'Fly{i}' for i in range(1, n_flies)]  
fig, axs = plt.subplots(figsize=(3, 5))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
fig.patch.set_facecolor('black')
axs.set_facecolor('black')
grouped_data = {}
for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.log'):
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)
            df_odor = df[df['odor_on']]
            first_on_index = df_odor.index[0]
            post_df = df.loc[first_on_index:]
            xo = post_df.iloc[0]['ft_posx']
            yo = post_df.iloc[0]['ft_posy']
            for key in keys_list:
                if key in filename:
                    group_key = key
                    break
            if group_key not in grouped_data:
                grouped_data[group_key] = {'insidpre': None, 'plume': None}
            if 'inside' in filename:
                grouped_data[group_key]['inside'] = (df_odor['ft_posy'].iloc[-1] - yo)
            elif 'plume' in filename:
                grouped_data[group_key]['plume'] = (df_odor['ft_posy'].iloc[-1] - yo)
inside_values = [data['inside'] for data in grouped_data.values()]
plume_values = [data['plume'] for data in grouped_data.values()]
noise = 0.05  # Adjust the noise level as needed
x_inside = np.random.normal(1, noise, size=len(inside_values))
x_plume = np.random.normal(2, noise, size=len(plume_values))
plt.scatter(x_inside, inside_values, color=c1, alpha=0.5)
plt.scatter(x_plume, plume_values, color=c2, alpha=0.5)
# plt.scatter(1, avg_r1, color='#c1ffc1', marker="_",  alpha=1)
# plt.scatter(2, avg_r2, color='#6f00ff',  marker="_", alpha=1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
axs.spines['bottom'].set_color('white')
axs.spines['left'].set_color('white')
plt.gca().spines['left'].set_linewidth(2)
plt.ylabel('distance tracked (mm)', fontsize=18, color='white')
plt.yticks(fontsize=14, color='white')
axs.set_xticks([1, 2])
axs.set_xticklabels(['y=500', 'y=0'], fontsize=16, color='white',rotation=45)
plt.xlim(0.5, 2.5)
plt.tight_layout()
plt.show()
savename = 'insidevsplume_uwdisplacement_bw.pdf'
#fig.savefig(os.path.join(figure_folder, savename))


# %% Returns/m when LED on at y=0 or y=500
keys_list = [f'Fly{i}' for i in range(1, n_flies)]  
fig, axs = plt.subplots(figsize=(3, 5))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
fig.patch.set_facecolor('black')
axs.set_facecolor('black')
grouped_data = {}
for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.log'):
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)
            df_odor = df[df['odor_on']]
            first_on_index = df_odor.index[0]
            post_df = df.loc[first_on_index:]
            xo = post_df.iloc[0]['ft_posx']
            yo = post_df.iloc[0]['ft_posy']
            ypos = df['ft_posy'] - yo
            b1_df = df[(ypos >= 0) & (ypos < 500)]
            b2_df = df[(ypos >= 500) & (ypos < 1000)]
            d, d_in, d_out = inside_outside(post_df)
            d1, d_in1, d_out1 = inside_outside(b1_df)
            d2, d_in2, d_out2 = inside_outside(b2_df)
            pl = get_a_bout_calc(post_df, 'path_length') / 1000
            returns_insidepre = 0
            returns_insidepost = 0
            returns_plume = 0
            for key in keys_list:
                if key in filename:
                    group_key = key
                    break
            if group_key not in grouped_data:
                grouped_data[group_key] = {'insidepre': None, 'insidepost': None, 'plume': None}
            if 'inside' in filename:
                for key, df in d_out1.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                        returns_insidepre+=1
                grouped_data[group_key]['insidepre'] = returns_insidepre/pl
                for key, df in d_out2.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                        returns_insidepost+=1
                grouped_data[group_key]['insidepost'] = returns_insidepost/pl
            elif 'plume' in filename:
                for key, df in d_out.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                        returns_plume+=1
                grouped_data[group_key]['plume'] = returns_plume/pl
insidepre_values = [data['insidepre'] for data in grouped_data.values()]
insidepost_values = [data['insidepost'] for data in grouped_data.values()]
plume_values = [data['plume'] for data in grouped_data.values()]
noise = 0.05  # Adjust the noise level as needed
x_insidepre = np.random.normal(1, noise, size=len(insidepre_values))
x_insidepost = np.random.normal(2, noise, size=len(insidepost_values))
x_plume = np.random.normal(3, noise, size=len(plume_values))
plt.scatter(x_insidepre, insidepre_values, color=c1, alpha=0.5)
plt.scatter(x_insidepost, insidepost_values, color=c2, alpha=0.5)
plt.scatter(x_plume, plume_values, color=c3, alpha=0.5)
# plt.scatter(1, avg_r1, color='#c1ffc1', marker="_",  alpha=1)
# plt.scatter(2, avg_r2, color='#6f00ff',  marker="_", alpha=1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
axs.spines['bottom'].set_color('white')
axs.spines['left'].set_color('white')
plt.gca().spines['left'].set_linewidth(2)
plt.ylabel('returns per meter', fontsize=18, color='white')
plt.yticks(fontsize=14, color='white')
axs.set_xticks([1, 2, 3])
axs.set_xticklabels(['no LED', 'y=500', 'y=0'], fontsize=16, color='white',rotation=45)
plt.xlim(0.5, 3.5)
plt.tight_layout()
plt.show()
savename = 'insidevsplume_returnsperm_bw.pdf'
#fig.savefig(os.path.join(figure_folder, savename))


# %% Upwind speed at entry
buf_pts = 100
R1_entries = {}
R2_entries = {}
for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.log'):
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)
            df_odor = df[df['odor_on']]
            first_on_index = df_odor.index[0]  
            post_df = df.loc[first_on_index:]
            xo = post_df.iloc[0]['ft_posx']
            yo = post_df.iloc[0]['ft_posy']
            ypos = df['ft_posy'] - yo
            if 'inside' in filename:
                d1, d_in1, d_out1 = inside_outside(post_df)
            elif 'plume' in filename:
                d2, d_in2, d_out2 = inside_outside(post_df)
            # Create entry epochs for the current fly around the specific time event
            idxs = np.arange(first_on_index - buf_pts, first_on_index + buf_pts)
            entries = df['y-vel'][idxs]
            if 'inside' in filename:
                R1_entries[filename] = entries  # Directly assign epoch to the fly
            elif 'plume' in filename:
                R2_entries[filename] = entries  # Directly assign epoch to the fly
t = (np.arange(len(idxs)) - buf_pts) / 10
fig, axs = plt.subplots(1, 1, figsize=(8, 5))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
fig.patch.set_facecolor('black')
axs.set_facecolor('black')
plt.axvline(x=0, linestyle='--', color='white')
average_R1 = np.mean(list(R1_entries.values()), axis=0)
average_R2 = np.mean(list(R2_entries.values()), axis=0)
axs.plot(t, average_R1, color=c1, linewidth=2, label='y=500')
axs.plot(t, average_R2, color=c2, linewidth=2, label='y=0')
R1_error = stats.sem(list(R1_entries.values()), axis=0)
R2_error = stats.sem(list(R2_entries.values()), axis=0)
axs.fill_between(t, average_R1 + R1_error, average_R1 - R1_error, color=c1, alpha=0.5)
axs.fill_between(t, average_R2 + R2_error, average_R2 - R2_error, color=c2, alpha=0.5)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['bottom'].set_linewidth(2)
axs.spines['left'].set_linewidth(2)
axs.spines['bottom'].set_color('white')
axs.spines['left'].set_color('white')
plt.yticks(fontsize=14, color='white')
plt.xticks(fontsize=14, color='white')
plt.xlabel('time (s)', fontsize=18, color='white')
plt.ylabel('upwind speed(mm/s)', fontsize=18, color='white')
fig.tight_layout()
plt.legend()
plt.show()
#fig.savefig(os.path.join(figure_folder, 'uw_speed_entry_insidevsplume.pdf'))

# %%
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
b1_means = []
b2_means = []

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.log'):
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)
            df = calculate_trav_dir(df)
            df_odor = df[df['odor_on']]
            first_on_index = df_odor.index[0]
            post_df = df.loc[first_on_index:]
            xo = post_df.iloc[0]['ft_posx']
            yo = post_df.iloc[0]['ft_posy']
            ypos = df['ft_posy'] - yo
            b1_df = df[(ypos < 1000)]
            b2_df = df[(ypos >= 500) & (ypos <= 1000)]
            d1, d_in1, d_out1 = inside_outside(b1_df)
            d2, d_in2, d_out2 = inside_outside(b2_df)
            fly_b1_means = []
            fly_b2_means = []
            if 'plume' in filename:
                for key, df in d_out1.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                        df = get_last_second(df)
                        b1_circmean = circmean_heading(df, fly_b1_means)
                        b1_x, b1_y = pol2cart(1, b1_circmean)
                        b1_means.append((b1_x, b1_y))
                    fly_b1_mean = stats.circmean(fly_b1_means, low=-np.pi, high=np.pi, axis=None, nan_policy='omit')
                    plt.polar([0, fly_b1_mean], [0, 1], color='black', alpha=0.3, solid_capstyle='round')
            if 'inside' in filename:
                for key, df in d_out1.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1: 
                        df = get_last_second(df)
                        b2_circmean = circmean_heading(df, fly_b2_means)
                        b2_x, b2_y = pol2cart(1, b2_circmean)
                        b2_means.append((b2_x, b2_y))
                    fly_b2_mean = stats.circmean(fly_b2_means, low=-np.pi, high=np.pi, axis=None, nan_policy='omit')
                    plt.polar([0, fly_b2_mean], [0, 1], color='#0bdf51', alpha=0.3, solid_capstyle='round')

# Convert Cartesian coordinates to arrays for easier manipulation
b1_means = np.array(b1_means)
b2_means = np.array(b2_means)

# Check if arrays are empty before computing summary vectors
if b1_means.size > 0:
    summary_b1_vector = np.mean(b1_means, axis=0)
    summary_b1_rho, summary_b1_phi = cart2pol(summary_b1_vector[0], summary_b1_vector[1])
    plt.polar([0, summary_b1_phi], [0, summary_b1_rho], color='black', alpha=1, linewidth=3, label='ctrl', solid_capstyle='round')
if b2_means.size > 0:
    summary_b2_vector = np.mean(b2_means, axis=0)
    summary_b2_rho, summary_b2_phi = cart2pol(summary_b2_vector[0], summary_b2_vector[1])
    plt.polar([0, summary_b2_phi], [0, summary_b2_rho], color='#0bdf51', alpha=1, linewidth=3, label='led on', solid_capstyle='round')

ax.grid(False)
ax.set_yticklabels([])
ax.set_theta_zero_location('N')
plt.title('entry angles')
plt.show()
savename = 'entry_angle_gKC_inhibition_inside.pdf'
#fig.savefig(os.path.join(figure_folder, savename))
# %%
