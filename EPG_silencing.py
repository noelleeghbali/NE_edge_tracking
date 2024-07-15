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

folder = "/Users/noelleeghbali/Desktop/exp/tethered_behavior/summer_2024_exp/MAIN/EPG_GtACR"
figure_folder = "/Users/noelleeghbali/Desktop/exp/tethered_behavior/summer_2024_exp/analysis/EPG_GtACR/figure"
folders = [folder]
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

# %% Average inside vs. outside trajectories (inhibition inside)
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
data = {
    'ctrl': {'x_in': [], 'y_in': [], 'x_out': [], 'y_out': [], 'color': 'black'},
    'led': {'x_in': [], 'y_in': [], 'x_out': [], 'y_out': [], 'color': '#0bdf51'}
}
for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.log') and 'inside' in filename:
            logfile = os.path.join(folder, filename)
            df = open_log(logfile) # This might be ambiguous...
            df_odor=df[df['odor_on']]
            first_on_index = df_odor.index[0]   
            post_df = df.loc[first_on_index:]
            yo = yo = post_df.iloc[0]['ft_posy']
            ypos = df['ft_posy'] - yo
            
            for key, cond in zip(['ctrl', 'led'], [(ypos < 500), (ypos > 500) & (ypos < 1500)]):
                df_subset = df[cond]
                d, d_on, d_off = inside_outside(df_subset)
                
                x_in, y_in = average_trajectory_in(d_on, pts=5000)
                x_out, y_out = average_trajectory_out(d_off, pts=5000)
                
                if len(x_in) > 0:
                    data[key]['x_in'].append(np.array(x_in))
                if len(y_in) > 0:
                    data[key]['y_in'].append(np.array(y_in))
                if len(x_out) > 0:
                    data[key]['x_out'].append(np.array(x_out))
                if len(y_out) > 0:
                    data[key]['y_out'].append(np.array(y_out))

                if len(x_out) > 0 and len(y_out) > 0:
                    axs.plot(x_out, y_out, color=data[key]['color'], alpha=0.2, linewidth=0.5)
                if len(x_in) > 0 and len(y_in) > 0:
                    axs.plot(x_in, y_in, color=data[key]['color'], alpha=0.2, linewidth=0.5)

# Calculate and plot the averages
for key in ['ctrl', 'led']:  # Plot ctrl first, then led
    if data[key]['x_in'] and data[key]['y_in']:  # Check if the lists are not empty
        x_in = np.vstack([arr for arr in data[key]['x_in'] if arr.size > 0])
        y_in = np.vstack([arr for arr in data[key]['y_in'] if arr.size > 0])
        avg_x_in = np.mean(x_in, axis=0)
        avg_y_in = np.mean(y_in, axis=0)
        axs.plot(avg_x_in, avg_y_in, color=data[key]['color'], alpha=1, linewidth=0.5, label=f'Inside {key.upper()}')

    if data[key]['x_out'] and data[key]['y_out']:  # Check if the lists are not empty
        x_out = np.vstack([arr for arr in data[key]['x_out'] if arr.size > 0])
        y_out = np.vstack([arr for arr in data[key]['y_out'] if arr.size > 0])
        avg_x_out = np.mean(x_out, axis=0)
        avg_y_out = np.mean(y_out, axis=0)
        axs.plot(avg_x_out, avg_y_out, color=data[key]['color'], alpha=1, linewidth=0.5, label=f'Outside {key.upper()}')

axs.set_aspect('equal')
axs.set_frame_on(False)
axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

# scale bar
scale_length = 5  # Length of the scale bar in data units
scale_position = 0.1  # Position of the scale bar along the y-axis
axs.plot([0, 5], [-20, -20], color='black', linewidth=2)
axs.text(2.5, -19.9, '5 mm', ha='center', va='bottom', fontsize=18)
plt.vlines(x=0, ymin=-19, ymax=19, colors='k', linestyles=(0, (5, 10)), linewidth=1)
plt.show()

savename = 'EPG_inhibition_inside_avg_trajs.pdf'
fig.savefig(os.path.join(figure_folder, savename))

# %% Returns per meter (inhibition outside)
fig, axs = plt.subplots(figsize=(3,5))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
fig.patch.set_facecolor('black')  
axs.set_facecolor('black')  

returns_b1 = []
returns_b2 = []
for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.log') and 'outside' in filename:
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)
            df_odor=df[df['odor_on']]
            first_on_index = df_odor.index[0]  
            post_df = df.loc[first_on_index:]
            xo = post_df.iloc[0]['ft_posx']
            yo = post_df.iloc[0]['ft_posy']
            ypos = df['ft_posy'] - yo
            df[(ypos >= 0) & (ypos < 500)]
            b1_df = df[(ypos >= 0) & (ypos < 500)]
            b2_df = df[ypos>=500]
            d1, d_in1, d_out1 = inside_outside(b1_df)
            d2, d_in2, d_out2 = inside_outside(b2_df)
            returns1=0
            returns2=0
            pl1 = get_a_bout_calc(b1_df, 'path_length') / 1000
            pl2 = get_a_bout_calc(b2_df, 'path_length') / 1000
            for key, df in d_out1.items():
                if return_to_edge(df):
                    returns1+=1
            returns_b1.append(returns1/pl1)
            for key, df in d_out2.items():
                if return_to_edge(df):
                    returns2+=1
            returns_b2.append(returns2/pl2) # list of returns in this block for each fly
b1_avg = sum(returns_b1) / len(returns_b1)
b2_avg = sum(returns_b2) / len(returns_b2)
noise = 0.05  # Adjust the noise level as needed
x_b1 = np.random.normal(1, noise, size=len(returns_b1))
x_b2 = np.random.normal(2, noise, size=len(returns_b2))
plt.scatter(x_b1, returns_b1, color='grey', alpha=0.5)
plt.scatter(x_b2, returns_b2, color='#0bdf51', alpha=0.5)
plt.plot([1,2],[b1_avg, b2_avg], color='white')
#  Plot averages as single symbols
# plt.scatter(1, acv_avg, color='none', edgecolor='white', marker='o',linewidth=2, s=100)
# plt.scatter(2, oct_avg, color='none', edgecolor='white', marker='o',linewidth=2, s=100)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
axs.spines['bottom'].set_color('white')
axs.spines['left'].set_color('white')
plt.gca().spines['left'].set_linewidth(2)
plt.ylabel('returns per meter', fontsize=18, color='white')
plt.yticks(fontsize=14, color='white')
axs.set_xticks([1,2])
axs.set_xticklabels(['ctrl', 'led on'], fontsize=16, color='white')
plt.xlim(0.5, 2.5)
plt.tight_layout()

plt.show()
savename='EPG_inhibition_outside_returns_per_meter.pdf'
fig.savefig(os.path.join(figure_folder, savename))

# %% Average entry angle (inhibition inside)
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
            b1_df = df[(ypos >= 0) & (ypos < 500)]
            b2_df = df[ypos >= 500]
            d1, d_in1, d_out1 = inside_outside(b1_df)
            d2, d_in2, d_out2 = inside_outside(b2_df)
            fly_b1_means = []
            fly_b2_means = []
            for key, df in d_in1.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                    df = get_last_second(df)
                    b1_circmean = circmean_heading(df, fly_b1_means)
                    b1_x, b1_y = pol2cart(1, b1_circmean)
                    b1_means.append((b1_x, b1_y))
                fly_b1_mean = stats.circmean(fly_b1_means, low=-np.pi, high=np.pi, axis=None, nan_policy='omit')
                plt.polar([0, fly_b1_mean], [0, 1], color='black', alpha=0.3, solid_capstyle='round')
            for key, df in d_in2.items():
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
plt.title('exit angles')
plt.show()
savename = 'exit_angle_EPG_inhibition_inside.pdf'
fig.savefig(os.path.join(figure_folder, savename))
# %%
