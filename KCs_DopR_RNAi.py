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

folder = "/Users/noelleeghbali/Desktop/exp/tethered_behavior/summer_2024_exp/analysis/KCs_DopR_RNAi"
figure_folder = "/Users/noelleeghbali/Desktop/exp/tethered_behavior/summer_2024_exp/analysis/KCs_DopR_RNAi/figure"
folders = [folder]
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

# %% Returns per meter (DopR1 vs DopR2)
fig, axs = plt.subplots(figsize=(3,5))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
fig.patch.set_facecolor('black')  
axs.set_facecolor('black')  

returns_r1 = []
returns_r2 = []
r1_n = 0
r2_n = 0
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
            d, d_in, d_out = inside_outside(post_df)
            returns1=0
            returns2=0
            pl = get_a_bout_calc(post_df, 'path_length') / 1000

            if 'Dop1R1' in filename:
                for key, df in d_out.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                        returns1+=1
                returns_r1.append(returns1/pl)
            if 'Dop1R2' in filename:
                for key, df in d_out.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                        returns2+=1
                returns_r2.append(returns2/pl) # list of returns in this block for each fly
r1_avg = sum(returns_r1) / len(returns_r1)
r2_avg = sum(returns_r2) / len(returns_r2)
noise = 0.05  # Adjust the noise level as needed
x_r1 = np.random.normal(1, noise, size=len(returns_r1))
x_r2 = np.random.normal(2, noise, size=len(returns_r2))
plt.scatter(x_r1, returns_r1, color='#c1ffc1', alpha=0.5)
plt.scatter(x_r2, returns_r2, color='#6f00ff', alpha=0.5)
plt.plot([1,2],[r1_avg, r2_avg], color='white')
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
axs.set_xticklabels(['DopR1', 'DopR2'], fontsize=16, color='white', rotation=45)
plt.xlim(0.5, 2.5)
plt.tight_layout()
plt.show()
savename='DopR_RNAi_returns_per_meter.pdf'
#fig.savefig(os.path.join(figure_folder, savename))


# %% Upwind displacement (LED at start vs at y=500)
fig, axs = plt.subplots(figsize=(3, 5))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
fig.patch.set_facecolor('black')
axs.set_facecolor('black')
ytrack_r1 = []
ytrack_r2 = []
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
            xpos = np.abs(post_df['ft_posx'] - xo)
            if 'Dop1R1' in filename:
                ytrack = df_odor['ft_posy'].iloc[-1] - yo
                ytrack_r1.append(ytrack / 1000)
            if 'Dop1R2' in filename:
                ytrack = df_odor['ft_posy'].iloc[-1] - yo
                ytrack_r2.append(ytrack / 1000)
avg_r1 = sum(ytrack_r1) / len(ytrack_r1) if len(ytrack_r1) > 0 else 0
avg_r2 = sum(ytrack_r2) / len(ytrack_r2) if len(ytrack_r2) > 0 else 0
noise = 0.05  # Adjust the noise level as needed
x_r1 = np.random.normal(1, noise, size=len(ytrack_r1))
x_r2 = np.random.normal(2, noise, size=len(ytrack_r2))
plt.hlines(y=0, xmin=-10, xmax=10, colors='white', linestyles='--', linewidth=1)
plt.scatter(x_r1, ytrack_r1, color='#c1ffc1', alpha=0.5)
plt.scatter(x_r2, ytrack_r2, color='#6f00ff', alpha=0.5)
plt.scatter(1, avg_r1, color='#c1ffc1', marker="_",  alpha=1)
plt.scatter(2, avg_r2, color='#6f00ff',  marker="_", alpha=1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
axs.spines['bottom'].set_color('white')
axs.spines['left'].set_color('white')
plt.gca().spines['left'].set_linewidth(2)
plt.ylabel('upwind displacement (m)', fontsize=18, color='white')
plt.yticks(fontsize=14, color='white')
axs.set_xticks([1, 2])
axs.set_xticklabels(['DopR1', 'DopR2'], fontsize=16, color='white',rotation=45)
plt.xlim(0.5, 2.5)
plt.tight_layout()
plt.show()
savename = 'R1vsR2_uwdisplacement_bw.pdf'
#fig.savefig(os.path.join(figure_folder, savename))


# %% X-position distribution
xpos_r1 = []
xpos_r2 = []
for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.log'):
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)
            df_odor=df[df['odor_on']]
            first_on_index = df_odor.index[0]   
            post_df = df.loc[first_on_index:]
            xo = post_df.iloc[0]['ft_posx']
            yo = post_df.iloc[0]['ft_posy']
            ypos = df['ft_posy'] - yo
            xpos = np.abs(df['ft_posx'] - xo)
            if 'Dop1R1' in filename:
                xpos_r1.append(xpos)
            if 'Dop1R2' in filename:
                xpos_r2.append(xpos)
compiled_r1 = pd.concat(xpos_r1, axis=0, ignore_index=True)
compiled_r2= pd.concat(xpos_r2, axis=0, ignore_index=True)
fig, axs = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor('black')  
axs.set_facecolor('black')  
sns.kdeplot(compiled_r1, label='DopR1', color='#c1ffc1', fill=False, cut=0)
sns.kdeplot(compiled_r2, label='DopR2', color='#6f00ff', fill=False, cut=0)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks([0, 25, 500], fontsize=14, color='white')
plt.yticks([0, 0.02], fontsize=14, color='white')
plt.legend()

# Add labels and title
plt.axvline(x=25, color='lightgrey')
plt.xlabel('x-position (mm)', fontsize=18, color='white')
plt.ylabel('probability', fontsize=18, color='white')
plt.xlim(0, 500)
plt.tight_layout()
plt.show()
savename = 'xpos_distribution_R1vsR2.pdf'
#fig.savefig(os.path.join(figure_folder, savename))

# %% Upwind speed at entry
buf_pts = 60
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
            if 'Dop1R1' in filename:
                d1, d_in1, d_out1 = inside_outside(post_df)
            elif 'Dop1R2' in filename:
                d2, d_in2, d_out2 = inside_outside(post_df)
            # Create entry epochs for the current fly around the specific time event
            idxs = np.arange(first_on_index - buf_pts, first_on_index + buf_pts)
            entries = df['y-vel'][idxs]
            if 'Dop1R1' in filename:
                R1_entries[filename] = entries  # Directly assign epoch to the fly
            elif 'Dop1R2' in filename:
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
axs.plot(t, average_R1, color='#c1ffc1', linewidth=2, label='DopR1')
axs.plot(t, average_R2, color='#6f00ff', linewidth=2, label='DopR2')
R1_error = stats.sem(list(R1_entries.values()), axis=0)
R2_error = stats.sem(list(R2_entries.values()), axis=0)
axs.fill_between(t, average_R1 + R1_error, average_R1 - R1_error, color='#c1ffc1', alpha=0.5)
axs.fill_between(t, average_R2 + R2_error, average_R2 - R2_error, color='#6f00ff', alpha=0.5)

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
#fig.savefig(os.path.join(figure_folder, 'uw_speed_entry_R1vR2.pdf'))
# %%
return_efficiency(folder, savename='testofre', size=(3,5), groups=2, keywords=['Dop1R1', 'Dop1R2'], colors=['#c1ffc1', '#6f00ff'])
# %%
block_return_efficiency(folder, savename='testofbre', size=(3,5), cutoff=350, labels=['LED on inside', 'LED on outisde'], colors=['grey', 'green'])
# %%
block_xpos_distribution(folder, savename = 'testkde', size=(6,5), cutoff=500, labels=['LED off', 'LED on'], colors=['grey', 'green'])
# %%
