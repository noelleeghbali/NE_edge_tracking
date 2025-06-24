import pandas as pd
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib import cm
from PIL import Image
from scipy import interpolate
from scipy import stats
from scipy.stats import linregress, sem, circmean, pearsonr, spearmanr
from scipy.stats import ttest_rel, wilcoxon, ttest_ind, kruskal, shapiro, normaltest, mannwhitneyu
import os
import seaborn as sns
import re
import scikit_posthocs as sp

def open_log(logfile):  # Generate a dataframe from the contents of a log file.
    df = pd.read_table(logfile, delimiter='[,]', engine='python')
    new = df["timestamp -- motor_step_command"].str.split("--", n=1, expand=True)
    df["timestamp"]= new[0]
    df["motor_step_command"]=new[1]
    df.drop(columns=["timestamp -- motor_step_command"], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format="%m/%d/%Y-%H:%M:%S.%f ")
    first_time = df['timestamp'].iloc[0]
    df['seconds'] = (df['timestamp']- first_time).dt.total_seconds()
    if (df['mfc2_stpt'] == 0).all():
        df['odor_on'] = df.mfc3_stpt>0
    if (df['mfc3_stpt'] == 0).all():
        df['odor_on'] = df.mfc2_stpt>0
    else:
        df['odor_on'] = (df['mfc2_stpt'] > 0) | (df['mfc3_stpt'] > 0)
    del_t = np.mean(np.diff(df.seconds))
    effective_rate = 1/del_t
    yv = np.gradient(df.ft_posy)*effective_rate
    df['y-vel'] = yv 
    xv = np.gradient(df.ft_posx)*effective_rate
    df['x-vel'] = xv
    xv = np.gradient(df.ft_posx, df.seconds)
    yv = np.gradient(df.ft_posy, df.seconds)
    speed = np.sqrt(xv**2+yv**2)
    df['speed'] = speed
    heading = ((df['ft_heading']+math.pi) % (2*math.pi))-math.pi
    df['transformed_heading'] = np.rad2deg(heading)
    try:
        angvel = np.abs(df.ft_yaw)*effective_rate
    except:
        angvel = np.abs(df.df_yaw)*effective_rate
    speed = np.sqrt(xv**2+yv**2)
    df['abs_angvel'] = angvel
    netmotion = np.sqrt(df.ft_roll**2+df.ft_pitch**2+df.ft_yaw**2)*effective_rate
    df['net_motion'] = netmotion
    return df

def light_on_off(df):  # For an experiment dataframe, split the trajectory into light on and light off bouts.
    d_on = {}
    d_off = {}
    # Group the df by consecutive 'led1_setpt' values, creating a dictionary of sub-dfs for each unique sequence
    d_total = dict([*df.groupby(df['led1_stpt'].ne(df['led1_stpt'].shift()).cumsum())])
    for bout in d_total:
        # If any 'led1_stpt' value in the group equals 1, add it to the d_off dictionary
        if (d_total[bout]['led1_stpt'] == 1).any():
            d_off[bout] = d_total[bout]
        # Otherwise, add the group to the d_on dictionary
        else:
            d_on[bout] = d_total[bout]
    return d_total, d_on, d_off

def calculate_trav_dir(df):
    x = df.ft_posx
    y = df.ft_posy
    dir = np.arctan2(np.gradient(y), np.gradient(x))
    df['trav_dir'] = dir
    return df

def calculate_net_motion(df):
    del_t = np.mean(np.diff(df.seconds))
    effective_rate = 1/del_t
    netmotion = np.sqrt(df.ft_roll**2+df.ft_pitch**2+df.ft_yaw**2) * effective_rate
    df['net_motion'] = netmotion
    return df

def circmean_heading(df, means_list):
    if 'ft_heading' in df.columns:
        heading_data = df['ft_heading']
    elif 'heading' in df.columns:
        heading_data = df['heading']
    else:
        raise KeyError("Neither 'ft_heading' nor 'heading' found in DataFrame.")
    circmean_value = stats.circmean(heading_data, low=-np.pi, high=np.pi, axis=None, nan_policy='omit')
    means_list.append(circmean_value)
    return circmean_value

def compute_mean_entry_heading(d, imaging=True):
    headings = []
    for key, df in d.items():
        if imaging:
            if df['relative_time'].iloc[-1] - df['relative_time'].iloc[0] >= 1:
                df = get_last_second(df)
                circmean_heading(df, headings)
        else:
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                df = get_last_second(df)
                circmean_heading(df, headings)

    if headings:
        headings_rad = np.array(headings)  # assumed to be in radians already
        vector = np.mean(np.exp(1j * headings_rad))
        mean_heading = np.angle(vector)
        strength = np.abs(vector)  # vector length (between 0 and 1)
        return mean_heading, strength
    else:
        return None, None



def circular_mean(angles):
    """Compute the circular mean of a list of angles in degrees."""
    angles_rad = np.deg2rad(angles)
    sin_sum = np.sum(np.sin(angles_rad))
    cos_sum = np.sum(np.cos(angles_rad))
    mean_angle_rad = np.arctan2(sin_sum, cos_sum)
    mean_angle_deg = np.rad2deg(mean_angle_rad)
    return mean_angle_deg

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

# def get_last_second(df):
#     last_time = df['seconds'].iloc[-1]
#     return df[df['seconds'] >= (last_time - 1)]

def mean_resultant_vector_length(angles):
    # angles should be in radians
    n = len(angles)
    if n == 0:
        return np.nan  # Or handle as you see fit
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = np.sqrt(C**2 + S**2) / n
    return R

def get_last_second(df):
    # Find the first occurrence of the 'seconds' column, d/t duplicates
    seconds_idx = next(i for i, col in enumerate(df.columns) if col == 'seconds')
    seconds = df.iloc[:, seconds_idx]
    last_time = seconds.iloc[-1]
    return df[seconds >= (last_time - 1)]
    
def inside_outside(df):
    di = {}
    do = {}
    d = dict([*df.groupby(df['instrip'].ne(df['instrip'].shift()).cumsum())])
    for bout in d:
        if d[bout].instrip.any():
            di[bout]=d[bout]
        else:
            do[bout]=d[bout]
    return d, di, do

def inside_outside_acv(df):
    di = {}
    do = {}
    d = dict([*df.groupby((df['mfc2_stpt'] > 0).ne((df['mfc2_stpt'] > 0).shift()).cumsum())])
    for bout in d:
        if (d[bout]['mfc2_stpt'] > 0).any():
            di[bout]=d[bout]
        else:
            do[bout]=d[bout]
    return d, di, do

def inside_outside_oct(df):
    di = {}
    do = {}
    d = dict([*df.groupby((df['mfc3_stpt'] > 0).ne((df['mfc3_stpt'] > 0).shift()).cumsum())])
    for bout in d:
        if (d[bout]['mfc3_stpt'] > 0).any():
            di[bout]=d[bout]
        else:
            do[bout]=d[bout]
    return d, di, do

def return_to_edge(df):
    xpos = df.ft_posx.to_numpy()
    if np.abs(xpos[-1]-xpos[0])<1:
        return True
    else:
        return False

def return_to_edge_angle(df, angle=0):
    xpos = df.ft_posx.to_numpy()
    ypos = df.ft_posy.to_numpy()
    theta = np.radians(angle)
    projected_pos = xpos * np.cos(theta) + ypos * np.sin(theta)
    return np.abs(projected_pos[-1] - projected_pos[0]) < 1

def light_on_off(df):
    d_on = dict()
    d_off = dict()
    d = dict([*df.groupby(df['led1_stpt'].ne(df['led1_stpt'].shift()).cumsum())])
    for bout in d:
        if (d[bout]['led1_stpt'] == 1).any():
            d_off[bout]=d[bout]
        else:
            d_on[bout]=d[bout]
    return d, d_on, d_off
    
def exclude_lost_tracking(data, thresh=10):
    jumps = np.sqrt(np.gradient(data.ft_posy)**2+np.gradient(data.ft_posx)**2)
    resets, _ = sg.find_peaks(jumps, thresh)
    #resets = resets + 10
    l_mod = np.concatenate(([0], resets.tolist(), [len(data)-1]))
    l_mod = l_mod.astype(int)
    list_of_dfs = [data.iloc[l_mod[n]:l_mod[n+1]] for n in range(len(l_mod)-1)]
    if len(list_of_dfs)>1:
        data = max(list_of_dfs, key=len)
        data.reset_index()
        print('LOST TRACKING, SELECTION MADE',)
    return data

def average_trajectory_in(dict, pts=5000):
    """
    find the averge (x,y) trajectory for a dict of inside or outside trajectories
    excludes first and last trajectories, excludes trajectories that don't
    return to the edge. flips trajectories to align all inside and outside.
    """
    if len(dict)<3:
        avg_x, avg_y=[],[]
    else:
        numel = len(dict)-2
        avg = np.zeros((numel, pts, 2))
        for i, key in enumerate(list(dict.keys())[1:-2]):
            df = dict[key]
            if len(df)>10:
                x = df.ft_posx.to_numpy()
                x = x-x[-1]
                if np.abs(x[0]-x[-1])<1: # fly must make it back to edge
                    x = -np.sign(np.mean(x))*x
                    y = df.ft_posy.to_numpy()
                    y = y-y[-1]
                    t = np.arange(len(x))
                    t_common = np.linspace(t[0], t[-1], pts)
                    fx = interpolate.interp1d(t, x)
                    fy = interpolate.interp1d(t, y)
                    avg[i,:,0] = fx(t_common)
                    avg[i,:,1] = fy(t_common)
        avg_x = np.mean(avg[:,:,0], axis=0)
        avg_y = np.mean(avg[:,:,1], axis=0)
    return avg_x, avg_y

def average_trajectory_out(dict, pts=5000):
    """
    find the averge (x,y) trajectory for a dict of inside or outside trajectories
    excludes first and last trajectories, excludes trajectories that don't
    return to the edge. flips trajectories to align all inside and outside.
    """
    if len(dict)<3:
        avg_x, avg_y=[],[]
    else:
        numel = len(dict)-2
        avg = np.zeros((numel, pts, 2))
        for i, key in enumerate(list(dict.keys())[1:-2]):
            df = dict[key]
            if len(df)>10:
                x = df.ft_posx.to_numpy()
                x = x-x[0]
                if np.abs(x[0]-x[-1]) < 1: # fly must make it back to edge
                    x = np.sign(np.mean(x))*x
                    y = df.ft_posy.to_numpy()
                    y = y-y[0]
                    t = np.arange(len(x))
                    t_common = np.linspace(t[0], t[-1], pts)
                    fx = interpolate.interp1d(t, x)
                    fy = interpolate.interp1d(t, y)
                    avg[i,:,0] = fx(t_common)
                    avg[i,:,1] = fy(t_common)
        avg_x = np.mean(avg[:,:,0], axis=0)
        avg_y = np.mean(avg[:,:,1], axis=0)
    return avg_x, avg_y

def mirror(angle):
    angle = np.mod(angle, 2*np.pi)
    if angle > np.pi:
        angle -= 2*np.pi  # bring to [-π, π]
    if angle < 0:
        angle = -angle  # mirror to right side
    return angle

def exp_parameters(folder_path):  # Create variables for visualization
    folder = folder_path
    figure_folder = f'{folder_path}/traj'
    # If a folder for storing figures does not exist, make one
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    # Initialize an empty list to store results for each experiment
    params_list = []
    # Create a dataframe for every logfile in the folder
    for filename in os.listdir(folder):
        if filename.endswith('.log'):
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)
            # print(df.columns())
            # print(df['mfc2_stpt'].unique().tolist())
            # print(df['mfc3_stpt'].unique().tolist())
            # Filter the df to include datapoints only when odor is being delivered, assigning df_odor based
            # on which mass flow controller is active, along with a corresponding plume color.
            if (df['mfc2_stpt'] == 0).all():
                df_odor = df.where(df.mfc3_stpt > 0)
                if 'mch' in filename or 'mho' in filename or 'ben' in filename:
                    plume_color = '#d473d4'
                elif 'oct' in filename:
                    plume_color = '#7ed4e6'
            if (df['mfc3_stpt'] == 0).all():
                df_odor = df.where(df.mfc2_stpt > 0)
                if 'mch' in filename or 'mho' in filename or 'ben' in filename:
                    plume_color = '#d473d4'
                else:
                    plume_color = '#fbceb1'
            if (df['mfc2_stpt'] == 0).all() and (df['mfc3_stpt'] == 0).all():
                df_odor = None
                plume_color = 'white'
            # elif len(df['mfc2_stpt'].unique().tolist())>1 and len(df['mfc3_stpt'].unique().tolist())>1:
            #     df_oct = df_odor = df.where(df.mfc3_stpt > 0)
            #     df_acv = df_odor = df.where(df.mfc2_stpt > 0)
            #     plume_color = ['#fbceb1', '#7ed4e6']
            # Filter the df to include datapoints only when optogenetic light is being delivered (light = on when led1 = 0.0)
            df_light = df.where(df.led1_stpt == 0.0)
            if df_odor is None:
                print('no df_odor')
                # Create an index for the first instance of light on (exp start), and filter the df to start at this index
                first_on_index = df[df['led1_stpt'] == 0.0].index[0]
                exp_df = df.loc[first_on_index:]
                bl_df = df.loc[:first_on_index]

                # Establish coordinates of the subject's origin at the exp start
                xo = exp_df.iloc[0]['ft_posx']
                yo = exp_df.iloc[0]['ft_posy']
                # Append the results for the current file to the list
                params_list.append([figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color])
            # elif df_oct is not None and df_acv is not None:
            #     print('multiple odors')
            #     first_on_index = df[df['instrip']].index[0] 
            #     exp_df = df.loc[first_on_index:]
            #     xo = exp_df.iloc[0]['ft_posx']
            #     yo = exp_df.iloc[0]['ft_posy']
            #     params_list.append([figure_folder, filename, df, df_oct, df_acv, df_light, exp_df, xo, yo, plume_color])
            else:
                # Create an index for the first instance of odor on (exp start), and filter the df to start at this index
                first_on_index = df[df['instrip']].index[0]
                exp_df = df.loc[first_on_index:]
                bl_df = df.loc[:first_on_index]
                # Establish coordinates of the subject's origin at the exp start
                xo = exp_df.iloc[0]['ft_posx']
                yo = exp_df.iloc[0]['ft_posy']
                # Append the results for the current file to the list
                params_list.append([figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color])

    return params_list

def exp_parameters_multiodor(folder_path):
    folder = folder_path
    figure_folder = f'{folder_path}/traj'
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    params_list = []
    for filename in os.listdir(folder):
        if filename.endswith('.log'):
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)
            mfc2_vals = df['mfc2_stpt'].unique().tolist()
            mfc3_vals = df['mfc3_stpt'].unique().tolist()
            df_light = df.where(df.led1_stpt == 0.0)
            if len(mfc2_vals) > 0 and len(mfc3_vals) > 0:
                df_odor1 = df.where(df.mfc2_stpt > 0)
                df_odor2 = df.where(df.mfc3_stpt > 0)
                if 'mch' in filename or 'ben' in filename or 'mho' in filename:
                    plume_colors = ['#d473d4', '#7ed4e6']
                else:
                    plume_colors = ['#fbceb1', '#7ed4e6']
                first_on_index = df[df['instrip']].index[0]
                exp_df = df.loc[first_on_index:]
                xo = exp_df.iloc[0]['ft_posx']
                yo = exp_df.iloc[0]['ft_posy']
                params_list.append(['multiodor', figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors])
            else:
                df_odor = df.where((df.mfc2_stpt > 0) | (df.mfc3_stpt > 0))
                plume_color = ['#7ed4e6']
                first_on_index = df[df['instrip']].index[0]
                exp_df = df.loc[first_on_index:]
                xo = exp_df.iloc[0]['ft_posx']
                yo = exp_df.iloc[0]['ft_posy']
                params_list.append(['singleodor', figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color])

    return params_list

def kw_stats(group_data, keywords):
    if all(len(group_data[keyword]) > 0 for keyword in keywords):
        stat, p_value = stats.ttest_ind(group_data[keywords[0]], group_data[keywords[1]], equal_var=False)
        if p_value < 0.001:
            significance = '***'
            p_text = f'p < 0.001 {significance}'
        elif p_value < 0.01:
            significance = '**'
            p_text = f'p = {p_value:.3f} {significance}'
        elif p_value < 0.05:
            significance = '*'
            p_text = f'p = {p_value:.3f} {significance}'
        else:
            significance = 'n.s.'
            p_text = f'p = {p_value:.3f} {significance}'
        return p_value, p_text, significance
    else:
        print("Not enough data to perform statistical test.")
        return None, None, None

def configure_white_plot(size=(4,6), xaxis=False):
    fig, axs = plt.subplots(1, 1, figsize=size)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    fig.patch.set_facecolor('white')  # Set background to black
    axs.set_facecolor('white')  # Set background of plotting area to black
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.rcParams['text.color'] = 'black'
    axs.spines['bottom'].set_visible(xaxis)
    axs.spines['bottom'].set_color('black')
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_color('black')
    axs.spines['left'].set_linewidth(2)
    axs.tick_params(axis='x', colors='black')
    axs.tick_params(axis='y', colors='black')
    if xaxis:
        axs.xaxis.label.set_color('black')
        axs.tick_params(axis='x', colors='black')
    plt.gca().spines['left'].set_linewidth(2)
    markc = 'black'
    return fig, axs, markc

def configure_bw_plot(size=(4, 6), xaxis=False, nrows=1, ncols=1):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)
    markc = 'white'
    # Flatten axs if it's a 2D array (in case of multiple rows/cols)
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
        for ax in axs:
            ax.set_facecolor('black')  # Plot area background
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(xaxis)
            ax.spines['bottom'].set_color('white')
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_color('white')
            ax.spines['left'].set_linewidth(2)
            ax.tick_params(axis='x', colors='white', width=2)
            ax.tick_params(axis='y', colors='white', width=2)
            if xaxis:
                ax.xaxis.label.set_color('white')
    else:
        axs = axs
        axs.set_facecolor('black')  # Plot area background
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['bottom'].set_visible(xaxis)
        axs.spines['bottom'].set_color('white')
        axs.spines['bottom'].set_linewidth(2)
        axs.spines['left'].set_color('white')
        axs.spines['left'].set_linewidth(2)
        axs.tick_params(axis='x', colors='white', width=2)
        axs.tick_params(axis='y', colors='white', width=2)
        if xaxis:
            axs.xaxis.label.set_color('white')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    fig.patch.set_facecolor('black')  # Figure background
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    return fig, axs, markc

def style_ax(ax, markc, title, xtick_labels, ylabel, xlim):
    ax.tick_params(which='both', axis='both', labelsize=12, length=3, width=2,
                   color='black', direction='out', left=True, bottom=True)
    ax.tick_params(axis='both', colors='white')

    for pos in ['right', 'top']:
        ax.spines[pos].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor(markc)

    ax.set_ylabel(ylabel, fontsize=18, color='white')
    ax.set_yticklabels([f"{tick:.1f}" for tick in ax.get_yticks()],
                       fontsize=14, color=markc)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(xtick_labels, fontsize=16, color=markc, rotation=45)
    ax.set_xlim(xlim)
    ax.set_title(title, fontsize=14, color='white')


def configure_white_polar(size=(6, 6), grid=True):
    fig, axs = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=size)
    markc = 'black'
    # Set plotting settings for dark theme
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    fig.patch.set_facecolor('white')  
    axs.set_facecolor('white') 
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    if grid:
        axs.grid(True, color='black', linestyle='-', linewidth=1)  # Enable gridlines with white color
    else:
        axs.grid(False)
    axs.plot(np.linspace(0, 2*np.pi, 100), np.ones(100), color='black', lw=2)  # Circle outline at radius = 1
    axs.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))  # 8 ticks every 45 degrees
    axs.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], fontsize=14, color='black')
    axs.set_yticklabels([])
    axs.set_theta_zero_location('N')
    axs.set_theta_direction(-1)
    return fig, axs, markc


def configure_bw_polar(size=(6, 6), grid=True):
    fig, axs = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=size)
    markc = 'white'
    # Set plotting settings for dark theme
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    fig.patch.set_facecolor('black')  
    axs.set_facecolor('black') 
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    if grid:
        axs.grid(True, color='white', linestyle='-', linewidth=1)  # Enable gridlines with white color
    else:
        axs.grid(False)
    axs.plot(np.linspace(0, 2*np.pi, 100), np.ones(100), color='white', lw=2)  # Circle outline at radius = 1
    axs.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))  # 8 ticks every 45 degrees
    axs.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], fontsize=14, color='white')
    axs.set_yticklabels([])
    axs.set_theta_zero_location('N')
    axs.set_theta_direction(-1)
    return fig, axs, markc


def trajectory_plotter(folder_path, strip_width, strip_length, plume_start, xlim, ylim, led, hlines=[], select_file=None, plot_type='odor', baseline=True, save=False):
    params_list = exp_parameters(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bdf51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        # Create a figure and set the font to Arial
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        # In an odor plume, plot the trajectory when the animal is in the odor
        if plot_type == 'odor':
            if baseline:
                plt.plot(df['ft_posx'] - xo, df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            else:
                plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
            plt.gca().add_patch(patches.Rectangle((-strip_width / 2, plume_start), strip_width, strip_length, facecolor=plume_color, alpha=0.5))
            savename = filename + '_odor_trajectory.pdf'
        # In a light plume, plot the trajectroy when the animal is in the light
        elif plot_type == 'odorless':
            if baseline:
                plt.plot(df['ft_posx'] - xo, df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            else:
                plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='lightgrey', label='base trajectory')
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on')
            plt.gca().add_patch(patches.Rectangle((-strip_width / 2, plume_start), strip_width, strip_length, facecolor=plume_color, edgecolor='lightgrey'))
            savename = filename + '_strip_trajectory.pdf'
        if hlines is not None:
            for i in (1, len(hlines)):
                plt.hlines(y=hlines[i - 1], xmin=-100, xmax=100, colors='k', linestyles='--', linewidth=1)
        # Set axes, labels, and title
        plt.xlim(xlim)
        plt.ylim(ylim)
        # plt.legend()
        plt.title(filename, fontsize=14)
        axs.set_xlabel('x-position (mm)', fontsize=14, color='white')
        axs.set_ylabel('y-position (mm)', fontsize=14, color='white')
        # Further customization
        axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            axs.spines[pos].set_visible(False)
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in axs.spines.items():
            spine.set_linewidth(2)
        for spine in axs.spines.values():
            spine.set_edgecolor('black')
        # Save and show the plot
        if save:
            plt.savefig(os.path.join(figure_folder, savename))
        else:
            plt.show()

def trajectory_plotter_bw(folder_path, strip_width, strip_length, plume_start, xlim, ylim, led, hlines=[], select_file=None, plot_type='odor', save=False):
    params_list = exp_parameters(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_bw_plot(size=(10,10), xaxis=True)
        # In an odor plume, plot the trajectory when the animal is in the odor
        if plot_type == 'odor':
            plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
            plt.gca().add_patch(patches.Rectangle((-strip_width / 2, plume_start), strip_width, strip_length, facecolor=plume_color, alpha=0.3))
            savename = filename + '_odor_trajectory_bw.pdf'
        # In a light plume, plot the trajectroy when the animal is in the light
        elif plot_type == 'odorless':
            plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='base trajectory')
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on')
            plt.gca().add_patch(patches.Rectangle((-strip_width / 2, plume_start), strip_width, strip_length, facecolor='black', edgecolor='lightgrey'))
            savename = filename + '_strip_trajectory_bw.pdf'
        if hlines is not None:
            for i in (1, len(hlines)):
                plt.hlines(y=hlines[i - 1], xmin=-100, xmax=100, colors='white', linestyles='--', linewidth=1)
        # Set axes, labels, and title
        plt.xlim(xlim)
        plt.ylim(ylim)
        #plt.legend()
        plt.title(filename, fontsize=14, color='white')
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
        # Further customization
        axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            axs.spines[pos].set_visible(False)
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in axs.spines.items():
            spine.set_linewidth(2)
        for spine in axs.spines.values():
            spine.set_edgecolor('white')
        axs.tick_params(axis='both', colors='white')
        # Save and show the plot
        if save:
            plt.savefig(os.path.join(figure_folder, savename))
        else:
            plt.show()

def multiodor_trajectory_plotter(folder_path, strip_width, strip_length, plume_start, xlim, ylim, led, hlines=[], select_file=None, plot_type='odor', save=False):
    params_list = exp_parameters_multiodor(folder_path)
    if led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_bw_plot(size=(10,10), xaxis=True)
        # In an odor plume, plot the trajectory when the animal is in the odor
        plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        plt.plot(df_odor1['ft_posx'] - xo, df_odor1['ft_posy'] - yo, color=plume_colors[0], label='odor only', linewidth=2)
        plt.plot(df_odor2['ft_posx'] - xo, df_odor2['ft_posy'] - yo, color=plume_colors[1], label='odor only', linewidth=2)
        plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
        plt.gca().add_patch(patches.Rectangle((-strip_width / 2, plume_start), strip_width, strip_length, facecolor='grey', alpha=0.3))
        savename = filename + '_odor_trajectory_bw.pdf'
        # Set axes, labels, and title
        plt.xlim(xlim)
        plt.ylim(ylim)
        #plt.legend()
        plt.title(filename, fontsize=14, color='white')
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
        # Further customization
        axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            axs.spines[pos].set_visible(False)
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in axs.spines.items():
            spine.set_linewidth(2)
        for spine in axs.spines.values():
            spine.set_edgecolor('white')
        axs.tick_params(axis='both', colors='white')
        # Save and show the plot
        if save:
            plt.savefig(os.path.join(figure_folder, savename))
        else:
            plt.show()

def jumping_plotter_bw(folder_path, led, stripwidth=50, jumpthresh=0, select_file=None, save=False): #
    params_list = exp_parameters(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_white_plot(size=(10,10), xaxis=True)
        d, d_in, d_out = inside_outside(exp_df)
        if jumpthresh is None:
            for key, df in d_in.items():
                if key == 1:
                    x = -(stripwidth/2)
                else:
                    x = df['ft_posx'].min() - xo
                y = df['ft_posy'].min() - yo
                height = (df['ft_posy'].max() - df['ft_posy'].min())
                plume = patches.Rectangle((x, y),stripwidth,height,facecolor='#fbceb1',alpha=0.5)
                axs.add_patch(plume)
        plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=1)
        plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=1)
        # if 'ctrl' not in filename:
            # plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=1)
        axs.set_aspect('equal', adjustable='datalim') 
        savename = filename + '_odor_trajectory_white.pdf'
        plt.title(filename, fontsize=14, color=markc)
        plt.show()
        if not os.path.exists(f'{folder_path}/traj/'):
            os.makedirs(f'{folder_path}/traj/')
        fig.savefig(f'{folder_path}/traj/{savename}.pdf', bbox_inches='tight')

def jumping_return_rate(folder_path, led, select_file=None, save=False):
    params_list = exp_parameters(folder_path)
    ctrl = []
    exp = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_white_plot(size=(10,10), xaxis=True)
        d, d_in, d_out = inside_outside(exp_df)
        returns = len(d_in) - 1
        last_odor = df_odor.index[-1]
        exp_df = exp_df.loc[:last_odor]
        pl = get_a_bout_calc(exp_df, 'path_length')
        if 'ctrl' in filename:
            ctrl.append(returns/pl)
        else:
            exp.append(returns/pl)
    plt.scatter([1] * len(all_plume1_bout), all_plume1_bout, color=colors[0], zorder=20)
    plt.scatter([2] * len(all_plume2_bout), all_plume2_bout, color=colors[1], zorder=20)
    avg_ctrl = np.mean(ctrl)
    avg_exp = np.mean(exp)
    
    plt.title(filename, fontsize=14, color=markc)
    plt.show()
    if not os.path.exists(f'{folder_path}/traj/'):
        os.makedirs(f'{folder_path}/traj/')
    fig.savefig(f'{folder_path}/traj/{savename}.pdf', bbox_inches='tight')

        
def zigzag_plotter_bw(folder_path, led, select_file=None, save=False):
    params_list = exp_parameters(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_white_plot(size=(10,10), xaxis=True)
        if 'CM' in folder_path:
            print(list(df.columns))
            df['exp_phase'] = df['exp_phase'].fillna(0)
            change_points = df['exp_phase'].ne(df['exp_phase'].shift())
            change_indices = df.index[change_points].tolist()
            print(change_indices)
            groups = (df['exp_phase'] != df['exp_phase'].shift()).cumsum()
            dfs = [group_df for _, group_df in df.groupby(groups)]
            plume1_df = dfs[1]
            plume2_df = dfs[3]
        else:
            d, d_on, d_off = light_on_off(exp_df)
            plume1_df = d_off[list(d_off.keys())[0]]
            plume1_start = plume1_df[plume1_df['instrip']].index.min()
            plume1_df = plume1_df.loc[plume1_start:]
            plume2_df = d_off[list(d_off.keys())[-1]]
            plume2_start = plume2_df[plume2_df['instrip']].index.min()
            plume2_df = plume2_df.loc[plume2_start:]
        plume_adj = 25 / np.sqrt(2)
        width=50
        mask1 = plume1_df['instrip'].fillna(False)
        height1 = (plume1_df.loc[mask1, 'ft_posx'].max() - plume1_df.loc[mask1, 'ft_posx'].min()) * np.sqrt(2) + 30
        mask2 = plume2_df['instrip'].fillna(False)
        height2 = (plume2_df.loc[mask2, 'ft_posx'].max() - plume2_df.loc[mask2, 'ft_posx'].min()) * np.sqrt(2) + 30
        if 'left' in filename:
            angle1 = 45
            angle2 = -45
            x1 = plume1_df.iloc[0]['ft_posx'] - xo - plume_adj
            y1 = plume1_df.iloc[0]['ft_posy'] - yo - plume_adj
            x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
            y2 = plume2_df.iloc[0]['ft_posy'] - yo + plume_adj
        elif 'RR' in filename:
            angle1 = -45
            angle2 = -45
            x1 = plume1_df.iloc[0]['ft_posx'] - xo - plume_adj
            y1 = plume1_df.iloc[0]['ft_posy'] - yo + plume_adj
            x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
            y2 = plume2_df.iloc[0]['ft_posy'] - yo + plume_adj
        else:
            angle1 = -45
            angle2 = 45
            x1 = plume1_df.iloc[0]['ft_posx'] - xo - plume_adj
            y1 = plume1_df.iloc[0]['ft_posy'] - yo + plume_adj
            x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
            y2 = plume2_df.iloc[0]['ft_posy'] - yo - plume_adj
        plume1 = patches.Rectangle((x1, y1),width,height1,angle=angle1,facecolor='#fbceb1',alpha=0.5)
        plume2 = patches.Rectangle((x2, y2),width,height2,angle=angle2,facecolor='#fbceb1',alpha=0.5)
        axs.add_patch(plume1)
        axs.add_patch(plume2)
        # plt.plot(plume1_df['ft_posx'] - xo, plume1_df['ft_posy'] - yo, color='green', label='clean air', linewidth=2)
        # plt.plot(plume2_df['ft_posx'] - xo, plume2_df['ft_posy'] - yo, color='blue', label='clean air', linewidth=2)
        plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        if 'ctrl' not in filename and 'CM' not in folder_path:
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
        savename = filename + '_odor_trajectory_white.pdf'
        #plt.legend()
        plt.title(filename, fontsize=14, color=markc)
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
        # Further customization
        axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            axs.spines[pos].set_visible(False)
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in axs.spines.items():
            spine.set_linewidth(2)
        for spine in axs.spines.values():
            spine.set_edgecolor(markc)
        axs.tick_params(axis='both', colors=markc)
        axs.set_aspect('equal', adjustable='datalim') 
        # Save and show the plot
        if save:
            plt.savefig(os.path.join(figure_folder, savename))
        else:
            plt.show()

def multiodor_zigzag_plotter_bw(folder_path, led, select_file=None, save=False):
    params_list = exp_parameters_multiodor(folder_path)
    if led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_bw_plot(size=(10,10), xaxis=True)
        d, d_on, d_off = light_on_off(exp_df)
        plume1_df = d_off[list(d_off.keys())[0]]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        plume2_df = d_off[list(d_off.keys())[-1]]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        plume_adj = 25 / np.sqrt(2)
        x1 = plume1_df.iloc[0]['ft_posx'] - xo - plume_adj
        y1 = plume1_df.iloc[0]['ft_posy'] - yo + plume_adj
        x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
        y2 = plume2_df.iloc[0]['ft_posy'] - yo - plume_adj
        width=50
        height1 = (plume1_df['ft_posx'].max() - plume1_df['ft_posx'].min())  * np.sqrt(2)
        height2 = np.abs(plume2_df['ft_posx'].max() - plume2_df['ft_posx'].min()) * np.sqrt(2)
        plume1 = patches.Rectangle((x1, y1),width,height1,angle=-45,facecolor=plume_colors[0],alpha=0.5)
        plume2 = patches.Rectangle((x2, y2),width,height2,angle=45,facecolor=plume_colors[0],alpha=0.5)
        axs.add_patch(plume1)
        axs.add_patch(plume2)
        plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        plt.plot(df_odor1['ft_posx'] - xo, df_odor1['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
        savename = filename + '_odor_trajectory_bw.pdf'
        #plt.legend()
        plt.title(filename, fontsize=14, color='white')
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
        # Further customization
        axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            axs.spines[pos].set_visible(False)
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in axs.spines.items():
            spine.set_linewidth(2)
        for spine in axs.spines.values():
            spine.set_edgecolor('white')
        axs.tick_params(axis='both', colors='white')
        axs.set_aspect('equal', adjustable='datalim') 
        # Save and show the plot
        if save:
            plt.savefig(os.path.join(figure_folder, savename))
        else:
            plt.show()

def training_plotter_bw(folder_path, led, select_file=None, save=False):
    params_list = exp_parameters(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_white_plot(size=(8,8), xaxis=True)
        _, d_on, d_off = light_on_off(df)
        _, d_in, d_out = inside_outside_oct(df)
        # print(len(d_in))
        pre_df = d_off[list(d_off.keys())[0]]
        plume_df = d_off[list(d_off.keys())[-1]]
        plume_start = plume_df[plume_df['instrip']].index.min()
        plume_df = plume_df.loc[plume_start:]
        plume_adj = 25 / np.sqrt(2)
        xplume = plume_df.iloc[0]['ft_posx'] - xo - plume_adj
        yplume = plume_df.iloc[0]['ft_posy'] - yo + plume_adj
        width=50
        height = (plume_df['ft_posx'].max() - plume_df['ft_posx'].min())  * np.sqrt(2)
        plume = patches.Rectangle((xplume, yplume),width,height,angle=-45,facecolor=plume_color,alpha=0.5)
        axs.add_patch(plume)
        plt.plot(df['ft_posx'] - xo, df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
        savename = filename + '_odor_trajectory_white.pdf'
        #plt.legend()
        plt.title(filename, fontsize=14, color=markc)
        axs.set_xlim(-500,1100)
        axs.set_ylim(-200,1400)
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
        # Further customization
        axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            axs.spines[pos].set_visible(False)
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in axs.spines.items():
            spine.set_linewidth(2)
        for spine in axs.spines.values():
            spine.set_edgecolor(markc)
        axs.set_aspect('equal', adjustable='datalim') 
        axs.tick_params(axis='both', colors=markc)
        # Save and show the plot
        if save:
            plt.savefig(os.path.join(figure_folder, savename))
        else:
            plt.show()

def oct_training_return_efficiency(folder_path, savename, groups=2, multiodor=True, size=(3,5), labels=['L training', 'R training'], colors=['lightgrey', '#7ed4e6']):
    if multiodor:
        params_list = exp_parameters_multiodor(folder_path)
    else:
        params_list = exp_parameters(folder_path)
    returns = {f'returns_r{i+1}': [] for i in range(groups)}
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    for this_experiment in params_list:
        if this_experiment[0] == 'multiodor':
            experiment_type, figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        else:
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, d_on, d_off = light_on_off(df)
        _, d_in, d_out = inside_outside(df)
        plume_df = d_off[list(d_off.keys())[-1]]
        plume_start = plume_df[plume_df['instrip']].index.min()
        plume_df = plume_df.loc[plume_start:]
        _, d_in, d_out = inside_outside(plume_df)
        pl = get_a_bout_calc(plume_df, 'path_length') / 1000
        counts = {f'returns_{i+1}': 0 for i in range(groups)}
        if multiodor:
            if 'mch' in filename:
                for key, df in d_out.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(df, angle=-45):
                        counts['returns_2'] += 1
                returns['returns_r2'].append(counts['returns_2'] / pl)
            elif 'RR' in filename:
                for key, df in d_out.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(df, angle=-45):
                        counts['returns_1'] += 1
                returns['returns_r1'].append(counts['returns_1'] / pl)
        if not multiodor:
            if 'LR' in filename:
                for key, df in d_out.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(df, angle=-45):
                        counts['returns_1'] += 1
                returns['returns_r1'].append(counts['returns_1'] / pl)
            else:
                for key, df in d_out.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(df, angle=-45):
                        counts['returns_2'] += 1
                returns['returns_r2'].append(counts['returns_2'] / pl)

    averages = [np.mean(returns[f'returns_r{i+1}']) if returns[f'returns_r{i+1}'] else 0 for i in range(groups)]
    std_devs = [np.std(returns[f'returns_r{i+1}']) if returns[f'returns_r{i+1}'] else 0 for i in range(groups)]
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(returns[f'returns_r{i+1}'])) for i in range(groups)]
    for i in range(groups):
        plt.scatter(x_values[i], returns[f'returns_r{i+1}'], color=colors[i])
        plt.hlines(averages[i], xmin=i+0.9, xmax=i+1.1, colors=markc, linewidth=2)  # Mean line
        # plt.errorbar(i+1, averages[i], yerr=std_devs[i], color=markc, capsize=0)  # Std deviation
    # plt.plot(range(1, groups+1), averages, color='white')
    data1 = returns['returns_r1']
    data2 = returns['returns_r2']
    stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')

    # Annotate p-value on the plot
    p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(data1, default=0), max(data2, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)
    # Further customization
    axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color=markc, direction='out', left=True, bottom=True)
    for pos in ['right', 'top']:
        axs.spines[pos].set_visible(False)
    plt.tight_layout()
    sns.despine(offset=10)
    for _, spine in axs.spines.items():
        spine.set_linewidth(2)
    for spine in axs.spines.values():
        spine.set_edgecolor(markc)
    axs.tick_params(axis='both', colors=markc)
    plt.ylabel('returns per meter', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks(range(1, groups+1))
    axs.set_xticklabels([labels[0], labels[1]], fontsize=16, color=markc, rotation=45)
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    # Save and show the plot
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def oct_training_pre_isi(folder_path, savename, groups=2, multiodor=True, bout='in', size=(3,5), labels=['L training', 'R training'], colors=['lightgrey', '#7ed4e6']):
    if multiodor:
        params_list = exp_parameters_multiodor(folder_path)
        isis_mch = []
        isis_oct = []
    else:
        params_list = exp_parameters(folder_path)
        isis_LR = []
        isis_RR = []
        all_isis = []
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    for this_experiment in params_list:
        if multiodor:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        else:
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, d_on, d_off = light_on_off(df)
        _, d_in, d_out = inside_outside_oct(df)
        plume_df = d_off[list(d_off.keys())[-1]]
        plume_start = plume_df[plume_df['instrip']].index.min()
        pre_df = df.loc[:plume_start]
        _, d_on, d_off = light_on_off(pre_df)
      
        if bout == 'in':
            d_bout = d_on
        elif bout == 'out':
            d_bout = dict(list(d_off.items())[1:])

        if multiodor:
            if 'mch' in filename:
                for key, df in list(d_bout.items())[:-1]:
                    isi = df['seconds'].iloc[-1] - df['seconds'].iloc[0]    
                    isis_mch.append(isi)
            else:
                for key, df in list(d_bout.items())[:-1]:
                    isi = df['seconds'].iloc[-1] - df['seconds'].iloc[0]    
                    isis_oct.append(isi)
        if not multiodor:
            if 'LR' in filename:
                for key, df in list(d_bout.items())[:-1]:
                    isi = df['seconds'].iloc[-1] - df['seconds'].iloc[0]    
                    if isi > 400:
                        print(filename)
                    isis_LR.append(isi)
                    all_isis.append(isi)
            else:
                for key, df in list(d_bout.items())[:-1]:
                    isi = df['seconds'].iloc[-1] - df['seconds'].iloc[0]    
                    isis_RR.append(isi)
                    all_isis.append(isi)
    if multiodor:
        avg_mch, std_mch = np.mean(isis_mch), np.std(isis_mch)
        avg_oct, std_oct = np.mean(isis_oct), np.std(isis_oct)
        noise = 0.05
        x_mch = np.random.normal(1, noise, size=len(isis_mch))
        x_oct = np.random.normal(2, noise, size=len(isis_oct))
        plt.scatter(x_mch, isis_mch, color=colors[0], alpha=0.8)
        plt.scatter(x_oct, isis_oct, color=colors[1], alpha=0.8)
        plt.hlines(avg_mch, xmin=0.9, xmax=1.1, colors=markc, linewidth=2)  # Mean line
        plt.hlines(avg_oct, xmin=1.9, xmax=2.1, colors=markc, linewidth=2)  # Mean line
        plt.errorbar(1, avg_mch, yerr=std_mch, color=markc, capsize=0)
        plt.errorbar(2, avg_oct, yerr=std_oct, color=markc, capsize=0)
    else:
        avg_LR, std_LR = np.mean(isis_LR), np.std(isis_LR)
        avg_RR, std_RR = np.mean(isis_RR), np.std(isis_RR)
        avg_all, std_all = np.mean(all_isis), np.std(all_isis)
        print(avg_all)
        print(std_all)
        noise = 0.05
        x_LR = np.random.normal(1, noise, size=len(isis_LR))
        x_RR = np.random.normal(2, noise, size=len(isis_RR))
        plt.scatter(x_LR, isis_LR, color=colors[0], alpha=0.8)
        plt.scatter(x_RR, isis_RR, color=colors[1], alpha=0.8)
        plt.hlines(avg_LR, xmin=0.9, xmax=1.1, colors=markc, linewidth=2)  # Mean line
        plt.hlines(avg_RR, xmin=1.9, xmax=2.1, colors=markc, linewidth=2)  # Mean line
        plt.errorbar(1, avg_LR, yerr=std_LR, color=markc, capsize=0)
        plt.errorbar(2, avg_RR, yerr=std_RR,  color=markc, capsize=0)
    # Further customization
    axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
    for pos in ['right', 'top']:
        axs.spines[pos].set_visible(False)
    plt.tight_layout()
    sns.despine(offset=10)
    for _, spine in axs.spines.items():
        spine.set_linewidth(2)
    for spine in axs.spines.values():
        spine.set_edgecolor('white')
    axs.tick_params(axis='both', colors='white')
    if bout == 'in':
        plt.ylabel('average pulse length (s)', fontsize=18, color='white')
    elif bout == 'out':
        plt.ylabel('average ISI (s)', fontsize=18, color='white')
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks(range(1, groups+1))
    axs.set_xticklabels([labels[0], labels[1]], fontsize=16, color=markc, rotation=45)
    plt.xlim(0.5, 2.5)
    plt.tight_layout()
    plt.show()
    # Save and show the plot
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def operant_training_pre_angles(folder_path, savename, multiodor=True, angle='entry', size=(5,5), colors=['lightgrey', '#ff355e']):
    fig, axs, markc = configure_bw_polar(size=size, grid=False)
    if multiodor:
        params_list = exp_parameters_multiodor(folder_path)
        angles_mch = []
        angles_oct = []
    else:
        params_list = exp_parameters(folder_path)
        angles_LR = []
        angles_RR = []
    for this_experiment in params_list:
        if multiodor:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        else:
            figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
            
        _, d_in, d_out = inside_outside_oct(df)
        _, d_on, d_off = light_on_off(df)
        plume_df = d_off[list(d_off.keys())[-1]]
        plume_start = plume_df[plume_df['instrip']].index.min()
        pre_df = df.loc[:plume_start]
        _, d_on, d_off = light_on_off(pre_df)
        if angle == 'entry':
            d_bout = dict(list(d_off.items())[1:])
        elif angle == 'exit':
            d_bout = d_on
        print(len(d_bout))
        if multiodor:
            for key, df in list(d_bout.items())[:-1]:
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                    df = get_last_second(df)
                    if 'mch' in filename:
                        circmean = circmean_heading(df, angles_mch)  # Appends inside function
                        p1x, p1y = pol2cart(1, circmean)  # Convert to Cartesian
                        plt.polar([0, circmean], [0, 1], color=colors[0], alpha=0.6, linewidth=1, solid_capstyle='round')  # Old ray plot
                    else:
                        circmean = circmean_heading(df, angles_oct)  # Appends inside function
                        p1x, p1y = pol2cart(1, circmean)  # Convert to Cartesian
                        plt.polar([0, circmean], [0, 1], color=colors[1], alpha=0.6, linewidth=1, solid_capstyle='round')  # Old ray plot
        else:
            for key, df in list(d_bout.items())[:-1]:
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                    df = get_last_second(df)
                    if 'LR' in filename:
                        circmean = circmean_heading(df, angles_LR)  # Appends inside function
                        p1x, p1y = pol2cart(1, circmean)  # Convert to Cartesian
                        plt.polar([0, circmean], [0, 1], color=colors[0], alpha=0.6, linewidth=1, solid_capstyle='round')  # Old ray plot
                    elif 'RR' in filename:
                        print('appending to RR')
                        circmean = circmean_heading(df, angles_RR)  # Appends inside function
                        p1x, p1y = pol2cart(1, circmean)  # Convert to Cartesian
                        plt.polar([0, circmean], [0, 1], color=colors[1], alpha=0.6, linewidth=1, solid_capstyle='round')  # Old ray plot
    if multiodor:
        p1_means = np.array(angles_mch)
        p2_means = np.array(angles_oct)
        label1 = 'mch plume'
        label2 = 'oct plume'
    else:
        print(len(angles_RR))
        p1_means = np.array(angles_LR)
        p2_means = np.array(angles_RR)
        label1 = 'L training'
        label2 = 'R training'
    if len(p1_means) > 0:
        print('p1 good')
        summary_p1_vector = np.mean(np.column_stack((np.cos(p1_means), np.sin(p1_means))), axis=0)
        summary_p1_rho, summary_p1_phi = cart2pol(summary_p1_vector[0], summary_p1_vector[1])
        plt.polar([0, summary_p1_phi], [0, summary_p1_rho], color=colors[0], alpha=1, linewidth=3, label=label1, solid_capstyle='round')
    if len(p2_means) > 0:
        print('p2 good')
        summary_p2_vector = np.mean(np.column_stack((np.cos(p2_means), np.sin(p2_means))), axis=0)
        summary_p2_rho, summary_p2_phi = cart2pol(summary_p2_vector[0], summary_p2_vector[1])
        plt.polar([0, summary_p2_phi], [0, summary_p2_rho], color=colors[1], alpha=1, linewidth=3, label=label2, solid_capstyle='round')
    plt.title(f'training pulse {angle} angles', color=markc, fontsize=18)
    plt.legend(prop={'size': 10}, frameon=False, loc='upper left', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def multiodor_training_plotter_bw(folder_path, led, select_file=None, save=False):
    params_list = exp_parameters_multiodor(folder_path)
    if led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor1, df_odor2, df_light, bl_df, exp_df, xo, yo, plume_colors = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_white_plot(size=(8,8), xaxis=True)
        _, d_on, d_off = light_on_off(df)
        _, d_in, d_out = inside_outside_oct(df)
        # print(len(d_in))
        # pre_df = d_off[list(d_off.keys())[0]]
        plume_df = d_off[list(d_off.keys())[-1]]
        plume_start = plume_df[plume_df['instrip']].index.min()
        plume_df = plume_df.loc[plume_start:]
        plume_adj = 25 / np.sqrt(2)
        xplume = plume_df.iloc[0]['ft_posx'] - xo - plume_adj
        yplume = plume_df.iloc[0]['ft_posy'] - yo + plume_adj
        width=50
        height = (plume_df['ft_posx'].max() - plume_df['ft_posx'].min())  * np.sqrt(2)
        plume = patches.Rectangle((xplume, yplume),width,height,angle=-45,facecolor=plume_colors[0],alpha=0.5)
        axs.add_patch(plume)
        plt.plot(df['ft_posx'] - xo, df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        plt.plot(df_odor1['ft_posx'] - xo, df_odor1['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        plt.plot(df_odor2['ft_posx'] - xo, df_odor2['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
        savename = filename + '_odor_trajectory_white.pdf'
        #plt.legend()
        plt.title(filename, fontsize=14, color=markc)
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
        axs.set_xlim(-500,1100)
        axs.set_ylim(-200,1400)
        # Further customization
        axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            axs.spines[pos].set_visible(False)
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in axs.spines.items():
            spine.set_linewidth(2)
        for spine in axs.spines.values():
            spine.set_edgecolor(markc)
        axs.tick_params(axis='both', colors=markc)
        axs.set_aspect('equal', adjustable='datalim') 
        # Save and show the plot
        if save:
            plt.savefig(os.path.join(figure_folder, savename))
        else:
            plt.show()

def disappearing_trajectory_plotter_bw(folder_path, led, select_file=None, save=False):
    params_list = exp_parameters(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_bw_plot(size=(8,8), xaxis=True)
        y_final = df_odor['ft_posy'].dropna().iloc[-1] - yo
        x_final = df_odor['ft_posx'].dropna().iloc[-1] - xo
        plt.gca().add_patch(patches.Rectangle((-25,0), 50, y_final, facecolor=plume_color, alpha=0.5))
        # if x_final < 1:
        #     axs.axvline(x=-25, ymin=0, ymax = y_final, color='grey', linewidth=1)
        # elif x_final> 1:
        #     axs.axvline(x=25, ymin=0, ymax = y_final, color='grey', linewidth=1)
        # In an odor plume, plot the trajectory when the animal is in the odor
        plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
        savename = filename + '_odor_trajectory_bw.pdf'
        #plt.legend()
        plt.title(filename, fontsize=14, color='white')
        # Further customization
        axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            axs.spines[pos].set_visible(False)        
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in axs.spines.items():
            spine.set_linewidth(2)
        for spine in axs.spines.values():
            spine.set_edgecolor('white')
        axs.tick_params(axis='both', colors='white')
        axs.set_aspect('equal', adjustable='datalim')
        # Save and show the plot
        if save:
            plt.savefig(os.path.join(figure_folder, savename))
        else:
            plt.show()


def zigzag_return_efficiency(folder_path, savename, size=(3,5), colors=['lightgrey', '#ff355e']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    re1 = []
    re2 = []
    re2c = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        if 'CM' in folder_path:
            print(list(df.columns))
            df['exp_phase'] = df['exp_phase'].fillna(0)
            change_points = df['exp_phase'].ne(df['exp_phase'].shift())
            change_indices = df.index[change_points].tolist()
            print(change_indices)
            groups = (df['exp_phase'] != df['exp_phase'].shift()).cumsum()
            dfs = [group_df for _, group_df in df.groupby(groups)]
            plume1_df = dfs[1]
            plume2_df = dfs[3]
        else:
            _, d_on, d_off = light_on_off(exp_df)
            plume1_df = d_off[list(d_off.keys())[0]]
            plume2_df = d_off[list(d_off.keys())[-1]]
        _, d_in1, d_out1 = inside_outside_acv(plume1_df)
        _, d_in2, d_out2 = inside_outside_acv(plume2_df)
        pl1 = get_a_bout_calc(plume1_df, 'path_length') / 1000
        pl2 = get_a_bout_calc(plume2_df, 'path_length') / 1000
        returns1 = -1
        returns2 = -1
        if 'left' in filename:
            angle1 = 45
            angle2 = -45
        if 'RR' in filename:
            angle1 = -45
            angle2 = -45
        else:
            angle1 = -45
            angle2 = 45
        for key, df in d_out1.items():
            if return_to_edge_angle(df, angle1):
                returns1 += 1
                cw = get_a_bout_calc(df, 'x_distance_from_plume')
                pl = get_a_bout_calc(df, 'path_length')
        re1.append(returns1/pl1)
        for key, df in d_out2.items():
            if return_to_edge_angle(df, angle2):
                returns2 += 1
        if 'ctrl' in filename:        
            re2c.append(returns2/pl2)
        else:
            re2.append(returns2/pl2)
        # if 'ctrl' in filename:
        #     plt.plot([1,2], [returns1/pl1, returns2/pl2], color=colors[0], zorder=0)
    noise = 0.05
    plt.scatter(1 + np.random.normal(0, noise, len(re1)), re1, color=colors[0])
    plt.scatter(2 + np.random.normal(0, noise, len(re2c)), re2c, color=colors[1])
    plt.scatter(3 + np.random.normal(0, noise, len(re2)), re2, color=colors[2])
    # plt.scatter([1] * len(re1), re1, color=colors[0])
    # plt.scatter([2] * len(re2c), re2c, color=colors[1], edgecolor=colors[0])
    print(re1)
    print(re2c)
    print(re2)
    avg_re1, std_re1 = np.mean(re1), np.std(re1)
    avg_re2c, std_re2c = np.mean(re2c), np.std(re2c)
    avg_re2, std_re2 = np.mean(re2), np.std(re2)
    plt.hlines(avg_re1, 0.9, 1.1, colors=markc, linewidth=2)
    plt.hlines(avg_re2c, 1.9, 2.1, colors=markc, linewidth=2)
    plt.hlines(avg_re2, 2.9, 3.1, colors=markc, linewidth=2)
    # plt.errorbar(1, avg_re1, yerr=std_re1, color=markc, capsize=0)
    # plt.errorbar(2, avg_re2c, yerr=std_re2c, color=markc, capsize=0)
    # plt.errorbar(3, avg_re2, yerr=std_re2, color=markc, capsize=0)
    plt.ylabel('returns per meter', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    # axs.set_xticks([1,1.5,2.5])
    # axs.set_xticklabels(['plume 1', 'plume 2 (LED off)', 'plume 2 (LED on)'], fontsize=16, color=markc, rotation=45)
    axs.tick_params(axis='x', length=0)
    plt.xlim(0.5, 3.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def zigzag_reentry_uws(folder_path, savename, size=(3,5), colors=['lightgrey', '#ff355e'], sample_rate=20, buf_pts=4):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    time = np.linspace(-buf_pts, buf_pts, buf_pts * sample_rate * 2)
    uws_ctrl = []
    uws_exp = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, d_on, d_off = light_on_off(exp_df)
        try:
            # Use last light-off event for re-entry analysis
            plume2_df = d_off[list(d_off.keys())[-1]]
            plume2_df = plume2_df[plume2_df['instrip']]  # Only in-plume points
            if plume2_df.empty:
                continue  # Skip if no reentry
            # Get plume re-entry onset time
            onset_time = plume2_df.index[0]
            half_window = buf_pts * sample_rate
            if onset_time in df.index:
                onset_idx = df.index.get_loc(onset_time)
                if onset_idx >= half_window and onset_idx + half_window < len(df):
                    segment = df.iloc[onset_idx - half_window : onset_idx + half_window]
                    uws_segment = segment['y-vel'].values
                    # Normalize using pre-onset baseline
                    baseline = np.mean(uws_segment[:half_window])
                    norm_uws = uws_segment - baseline
                    if 'ctrl' in filename.lower():
                        uws_ctrl.append(norm_uws)
                    else:
                        uws_exp.append(norm_uws)
                    print(f"found reentry segment in {filename}")
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
    if uws_ctrl:
        mean_ctrl = np.mean(uws_ctrl, axis=0)
        sem_ctrl = sem(uws_ctrl, axis=0)
        axs.plot(time, mean_ctrl, color=colors[0], lw=2, label='LED off')
        axs.fill_between(time, mean_ctrl - sem_ctrl, mean_ctrl + sem_ctrl, color=colors[0], alpha=0.2)
    if uws_exp:
        mean_exp = np.mean(uws_exp, axis=0)
        sem_exp = sem(uws_exp, axis=0)
        axs.plot(time, mean_exp, color=colors[1], lw=2, label='LED on')
        axs.fill_between(time, mean_exp - sem_exp, mean_exp + sem_exp, color=colors[1], alpha=0.2)
    axs.axvline(0, color=markc, linestyle='--')
    axs.set_xlabel("time (s)", color=markc, fontsize=14)
    axs.set_ylabel("upwind speed (mm/s)", color=markc, fontsize=14)
    axs.set_title("upwind speed at reentry", color=markc, fontsize=16)
    axs.legend(prop={'size': 10}, frameon=False)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def zigzag_return_efficiency_fc(folder_path, savename, size=(3,5), colors=['lightgrey', '#ff355e']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    # re1 = []
    re2 = []
    re2c = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        _, d_on, d_off = light_on_off(exp_df)
        plume1_df = d_off[list(d_off.keys())[0]]
        plume2_df = d_off[list(d_off.keys())[-1]]
        _, d_in1, d_out1 = inside_outside_acv(plume1_df)
        _, d_in2, d_out2 = inside_outside_acv(plume2_df)
        pl1 = get_a_bout_calc(plume1_df, 'path_length') / 1000
        pl2 = get_a_bout_calc(plume2_df, 'path_length') / 1000
        returns1 = 0
        returns2 = 0
        for key, df in d_out1.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(df, angle=-45):
                returns1 += 1
        # re1.append(returns1/pl1)
        for key, df in d_out2.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(df, angle=45):
                returns2 += 1
        if 'ctrl' in filename:
            re2c.append((returns2/pl2)/(returns1/pl1))
        else:
            re2.append((returns2/pl2)/(returns1/pl1))
    # plt.scatter([1]*len(re1), re1, color=colors[0], alpha=0.5)
    re2c_mean = np.mean(re2c)
    re2c_sem = np.std(re2c) / np.sqrt(len(re2c))
    re2_mean = np.mean(re2)
    re2_sem = np.std(re2) / np.sqrt(len(re2))
    axs.axhline(1, color=markc, linestyle='--', linewidth=1)
    plt.scatter(1 + np.random.normal(0, 0.05, len(re2c)), re2c, color=colors[0])
    plt.scatter(2 + np.random.normal(0, 0.05, len(re2)), re2, color=colors[1])
    plt.ylabel('plume 1: plume 2\nreturns per meter', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks([1, 2])
    axs.set_xticklabels(['LED off', 'LED on'], fontsize=18, color=markc, rotation=45)
    axs.tick_params(axis='x', length=0)
    plt.xlim(0.5, 2.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')
    
def zigzag_training_traj(folder_path, savename, size=(6,6)):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=2)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        _, d_on, d_off = light_on_off(exp_df)
        plume1_df = d_off[list(d_off.keys())[0]]
        plume2_df = d_off[list(d_off.keys())[-1]]
        end_of_plume1 = plume1_df.index[-1]
        start_of_plume2 = plume2_df.index[0]
        training_df = exp_df.loc[end_of_plume1:start_of_plume2]
        training_df_odor = training_df.where(df.mfc2_stpt > 0)
        xo = training_df.iloc[0]['ft_posx']
        yo = training_df.iloc[0]['ft_posy']
        if 'ctrl' in filename:
            axs[0].plot(training_df['ft_posx'] - xo, training_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            axs[0].plot(training_df_odor['ft_posx'] - xo, training_df_odor['ft_posy'] - yo, color='#fbceb1', label='odor', linewidth=2)
            axs[0].set_title('LED off', fontsize=14, color='white')

        else:
            axs[1].plot(training_df['ft_posx'] - xo, training_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            axs[1].plot(training_df_odor['ft_posx'] - xo, training_df_odor['ft_posy'] - yo, color='#ff355e', label='odor + light', linewidth=2)
            axs[1].set_title('LED on', fontsize=14, color='white')
        
    xlims = [ax.get_xlim() for ax in axs]
    ylims = [ax.get_ylim() for ax in axs]
    xmin = min(x[0] for x in xlims)
    xmax = max(x[1] for x in xlims)
    ymin = min(y[0] for y in ylims)
    ymax = max(y[1] for y in ylims)
    x_range = xmax - xmin
    y_range = ymax - ymin
    max_range = max(x_range, y_range)
    xmid = (xmax + xmin) / 2
    ymid = (ymax + ymin) / 2
    new_xlim = (xmid - max_range / 2, xmid + max_range / 2)
    new_ylim = (ymid - max_range / 2, ymid + max_range / 2)

    for ax in axs:
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.set_xlabel('x-position (mm)', fontsize=14)
        ax.set_ylabel('y-position (mm)', fontsize=14)
        ax.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        ax.tick_params(axis='both', colors='white')
        ax.set_aspect('equal', adjustable='datalim') 
        for pos in ['right', 'top']:
            ax.spines[pos].set_visible(False)
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in ax.spines.items():
            spine.set_linewidth(2)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def zigzag_reentry_traj(folder_path, size=(6,6)):
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        _, d_on, d_off = light_on_off(exp_df)
        plume1_df = d_off[list(d_off.keys())[0]]
        plume2_df = d_off[list(d_off.keys())[-1]]
        _, d_in, d_out = inside_outside(plume2_df)
        print(len(d_out))
        reentry_df = d_out[list(d_out.keys())[1]]
        xo = reentry_df.iloc[0]['ft_posx']
        yo = reentry_df.iloc[0]['ft_posy']
        if return_to_edge_angle(reentry_df, angle=45):
            x1 = 0
            y1 = 0
            x2 = reentry_df.iloc[-1]['ft_posx'] - xo
            y2 = reentry_df.iloc[-1]['ft_posy'] - yo
            radius = 0.1 * np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        else:
            x1 = 10 / np.sqrt(2)
            y1 = -10 / np.sqrt(2)
            x2 = -10 / np.sqrt(2)
            y2 = 10 / np.sqrt(2)
            radius = 0.15 * np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        axs.plot([x1, x2], [y1, y2], linestyle='--', color='white', linewidth=1.5)
        
        if 'ctrl' in filename:
            axs.plot(reentry_df['ft_posx'] - xo, reentry_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            axs.set_title('LED off', fontsize=14, color='white')
            circle = patches.Circle((0, 0), radius=radius, edgecolor='none', facecolor='#fbceb1', linewidth=2)
            axs.add_patch(circle)
        else:
            axs.plot(reentry_df['ft_posx'] - xo, reentry_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            axs.set_title('LED on', fontsize=14, color='white')
            circle = patches.Circle((0, 0), radius=radius, edgecolor='none', facecolor='#ff355e', linewidth=2)
            axs.add_patch(circle)
        savename = filename + '_reentry_trajectory_bw.pdf'

        axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            axs.spines[pos].set_visible(False)        
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in axs.spines.items():
            spine.set_linewidth(2)
        for spine in axs.spines.values():
            spine.set_edgecolor('white')
        axs.tick_params(axis='both', colors='white')
        axs.set_aspect('equal', adjustable='datalim')
            
        if not os.path.exists(f'{folder_path}/fig/'):
            os.makedirs(f'{folder_path}/fig/')
        fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def zigzag_training_stats(folder_path, savename, labels, size=(3,5), colors=['#fbceb1', '#ff355e']):
    params_list = exp_parameters(folder_path)
    training_dur = []
    training_dur_ctrl = []
    pre_plume_dur = []
    pre_plume_dur_ctrl = []
    pre_plume_tortuosity = []
    pre_plume_tortuosity_ctrl = []
    pre_plume_speed = []
    pre_plume_speed_ctrl = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        _, d_on, d_off = light_on_off(exp_df)
        plume1_df = d_off[list(d_off.keys())[0]]
        plume2_df = d_off[list(d_off.keys())[-1]]
        end_of_plume1 = plume1_df.index[-1]
        start_of_plume2 = plume2_df.index[0]
        training_df = exp_df.loc[end_of_plume1:start_of_plume2]
        _, train_on, train_off = light_on_off(training_df)

        training_bout = train_on[2]
        pre_plume_bout = train_off[3]
        x_i = pre_plume_bout['ft_posx'].iloc[0]
        x_f = pre_plume_bout['ft_posx'].iloc[-1]
        y_i = pre_plume_bout['ft_posy'].iloc[0]
        y_f = pre_plume_bout['ft_posy'].iloc[-1]
        pre_disp = ((x_f - x_i)**2 + (y_f - y_i)**2)**0.5
        pre_dist = get_a_bout_calc(pre_plume_bout, 'path_length')
        pre_tortuosity = pre_dist / pre_disp
        avg_speed = np.mean(pre_plume_bout['speed'])

        if 'RR' in filename or 'ctrl' in filename:
            training_dur_ctrl.append(training_bout['seconds'].iloc[-1] - training_bout['seconds'].iloc[0])
            pre_plume_dur_ctrl.append(pre_plume_bout['seconds'].iloc[-1] - pre_plume_bout['seconds'].iloc[0])
            pre_plume_tortuosity_ctrl.append(pre_tortuosity)
            pre_plume_speed_ctrl.append(avg_speed)
        else:
            training_dur.append(training_bout['seconds'].iloc[-1] - training_bout['seconds'].iloc[0])
            pre_plume_dur.append(pre_plume_bout['seconds'].iloc[-1] - pre_plume_bout['seconds'].iloc[0])
            pre_plume_tortuosity.append(pre_tortuosity)
            pre_plume_speed.append(avg_speed)

    fig, axs, markc = configure_bw_plot(size=size, xaxis=False, ncols=4)
    noise = 0.05
    datasets = [
        (axs[3], pre_plume_speed, pre_plume_speed_ctrl, 'speed', 'mm/s', (0.5, 2.5)),
        (axs[2], pre_plume_tortuosity, pre_plume_tortuosity_ctrl, 'tortuosity', 'distance/\ndisplacement', (0.5, 2.5)),
        (axs[1], pre_plume_dur, pre_plume_dur_ctrl, "re-entry latency", "time(s)", (0.5, 2.5)),
        (axs[0], training_dur, training_dur_ctrl, "pulse length", "time (s)", (0.5, 2.5))
    ]
    for ax, data, ctrl_data, title, ylabel, xlim in datasets:
        x_ctrl = np.random.normal(1, noise, size=len(ctrl_data))
        x = np.random.normal(2, noise, size=len(data))
        avg, std = np.mean(data), np.std(data)
        avg_ctrl, std_ctrl = np.mean(ctrl_data), np.std(ctrl_data)
        ax.scatter(x_ctrl, ctrl_data, color=colors[0], alpha=0.5)
        ax.scatter(x, data, color=colors[1], alpha=0.8)
        ax.hlines(avg_ctrl, 0.9, 1.1, colors=markc, linewidth=2)
        ax.hlines(avg, 1.9, 2.1, colors=markc, linewidth=2)
        ax.errorbar(1, avg_ctrl, yerr=std_ctrl, color=markc, capsize=0)
        ax.errorbar(2, avg, yerr=std, color=markc, capsize=0)
        style_ax(ax, markc, title=title, xtick_labels=labels, ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def zigzag_first_return_efficiency(folder_path, savename, size=(3,5), colors=['lightgrey', '#ff355e']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    dist1_list, pl1_list = [], []
    dist2_list, pl2_list = [], []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment 
        _, d_on, d_off = light_on_off(exp_df)
        plume1_df = d_off[list(d_off.keys())[0]]
        plume2_df = d_off[list(d_off.keys())[-1]]
        _, d_in1, d_out1 = inside_outside_acv(plume1_df)
        _, d_in2, d_out2 = inside_outside_acv(plume2_df)
        if 'RR' in filename:
            angle=-45
        else:
            angle=45
        for key, df in d_out2.items():
            
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(df, angle=angle):
                dist2 = get_a_bout_calc(df, 'furthest_distance_from_plume')
                pl2 = get_a_bout_calc(df, 'path_length')
                if 'RR' in filename:
                    dist1_list.append(dist2)
                    pl1_list.append(pl2)
                    plt.scatter(dist2, pl2, color=colors[0], alpha=0.5)
                else:
                    print(filename)
                    dist2_list.append(dist2)
                    pl2_list.append(pl2)
                    plt.scatter(dist2, pl2, color=colors[2], alpha=0.5)
    dist1_array, pl1_array = np.array(dist1_list), np.array(pl1_list)
    dist2_array, pl2_array = np.array(dist2_list), np.array(pl2_list)
    if len(dist1_array) > 1:
        slope1, intercept1, _, _, _ = linregress(dist1_array, pl1_array)
        x_fit1 = np.linspace(min(dist1_array), max(dist1_array), 100)
        y_fit1 = slope1 * x_fit1 + intercept1
        plt.plot(x_fit1, y_fit1, color=colors[0], linestyle='-', linewidth=2, label='RR')
    if len(dist2_array) > 1:
        slope2, intercept2, _, _, _ = linregress(dist2_array, pl2_array)
        x_fit2 = np.linspace(min(dist2_array), max(dist2_array), 100)
        y_fit2 = slope2 * x_fit2 + intercept2
        plt.plot(x_fit2, y_fit2, color=colors[2], linestyle='-', linewidth=2, label='RL')
    plt.ylabel('total path length (mm)', fontsize=18, color=markc)
    plt.xlabel('furthest distance from plume (mm)', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    plt.xticks(fontsize=14, color=markc)
    axs.plot(np.linspace(0, 1000, 100), 2*(np.linspace(0, 1000, 100)), linestyle='--', color=markc)
    axs.set_xlim(-10, 200)
    legend = plt.legend(prop={'size': 10}, frameon=False)
    plt.setp(legend.get_texts(), color='white')
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def zigzag_alt_return_efficiency(folder_path, savename, size=(3,5), colors=['lightgrey', '#ff355e']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    dist1_list, pl1_list = [], []
    dist2_list, pl2_list = [], []
    dist2c_list, pl2c_list = [], []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        _, d_on, d_off = light_on_off(exp_df)
        plume1_df = d_off[list(d_off.keys())[0]]
        plume2_df = d_off[list(d_off.keys())[-1]]
        _, d_in1, d_out1 = inside_outside_acv(plume1_df)
        _, d_in2, d_out2 = inside_outside_acv(plume2_df)
        for key, df in d_out1.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(df, angle=-45):
                dist1 = get_a_bout_calc(df, 'furthest_distance_from_plume')
                pl1 = get_a_bout_calc(df, 'path_length')
                plt.scatter(dist1, pl1, color=colors[0], alpha=0.8)
                dist1_list.append(dist1)
                pl1_list.append(pl1)
        for key, df in d_out2.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(df, angle=45):
                if 'ctrl' in filename:
                    dist2c = get_a_bout_calc(df, 'furthest_distance_from_plume')
                    pl2c = get_a_bout_calc(df, 'path_length')
                    # plt.scatter(dist2c, pl2c, color=colors[1], alpha=0.5)
                    dist2c_list.append(dist2c)
                    pl2c_list.append(pl2c)
                else:
                    dist2 = get_a_bout_calc(df, 'furthest_distance_from_plume')
                    pl2 = get_a_bout_calc(df, 'path_length')
                    plt.scatter(dist2, pl2, color=colors[2], alpha=0.8)
                    dist2_list.append(dist2)
                    pl2_list.append(pl2)
    dist1_array, pl1_array = np.array(dist1_list), np.array(pl1_list)
    dist2_array, pl2_array = np.array(dist2_list), np.array(pl2_list)
    dist2c_array, pl2c_array = np.array(dist2c_list), np.array(pl2c_list)
    if len(dist1_array) > 1:
        slope1, intercept1, _, _, _ = linregress(dist1_array, pl1_array)
        x_fit1 = np.linspace(min(dist1_array), max(dist1_array), 100)
        y_fit1 = slope1 * x_fit1 + intercept1
        plt.plot(x_fit1, y_fit1, color=colors[0], linestyle='-', linewidth=2, label='plume 1')
    # if len(dist2c_array) > 1:
    #     slope2c, intercept2c, _, _, _ = linregress(dist2c_array, pl2c_array)
    #     x_fit2c = np.linspace(min(dist2c_array), max(dist2c_array), 100)
    #     y_fit2c = slope2c * x_fit2c + intercept2c
    #     plt.plot(x_fit2c, y_fit2c, color=colors[1], linestyle='-', linewidth=2, label='plume 2 ctrl')
    if len(dist2_array) > 1:
        slope2, intercept2, _, _, _ = linregress(dist2_array, pl2_array)
        x_fit2 = np.linspace(min(dist2_array), max(dist2_array), 100)
        y_fit2 = slope2 * x_fit2 + intercept2
        plt.plot(x_fit2, y_fit2, color=colors[2], linestyle='-', linewidth=2, label='plume 2')
    plt.ylabel('total path length (mm)', fontsize=18, color=markc)
    plt.xlabel('furthest distance from plume (mm)', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    plt.xticks(fontsize=14, color=markc)
    axs.plot(np.linspace(0, 1000, 100), 2*(np.linspace(0, 1000, 100)), linestyle='--', color=markc)
    axs.set_xlim(-10, 200)
    legend = plt.legend(prop={'size': 10}, frameon=False)
    plt.setp(legend.get_texts(), color=markc)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def zigzag_entry_angles_all(folder_path, savename, size=(5, 5), colors=['lightgrey', '#ff355e']):
    all_plume1_bout = []
    all_plume2_bout = []
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment  
        _, d_on, d_off = light_on_off(exp_df)
        plume1_df = d_off[list(d_off.keys())[0]]
        plume2_df = d_off[list(d_off.keys())[-1]]
        _, d_in1, d_out1 = inside_outside_acv(plume1_df)
        _, d_in2, d_out2 = inside_outside_acv(plume2_df)
        for key, df in d_out1.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                df = get_last_second(df)
                p1_circmean = circmean_heading(df, all_plume1_bout)
                
        for key, df in d_out2.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                df = get_last_second(df)
                p2_circmean = circmean_heading(df, all_plume2_bout)

    fig1, axs1, markc1 = configure_bw_polar(size=size, grid=False)
    for angle in all_plume1_bout:
        plt.polar([0, angle], [0, 1], color=colors[0], alpha=0.2, linewidth=1, solid_capstyle='round')
    if len(all_plume1_bout) > 0:
        summary_vector1 = np.mean(np.column_stack((np.cos(all_plume1_bout), np.sin(all_plume1_bout))), axis=0)
        rho1, phi1 = cart2pol(summary_vector1[0], summary_vector1[1])
        plt.polar([0, phi1], [0, rho1], color=colors[0], alpha=1, linewidth=3, label='plume 1', solid_capstyle='round')
    plt.title('Plume 1 avg entry angle', color=markc1, fontsize=18)
    plt.show()
    fig2, axs2, markc2 = configure_bw_polar(size=size, grid=False)
    for angle in all_plume2_bout:
        plt.polar([0, angle], [0, 1], color=colors[1], alpha=0.2, linewidth=1, solid_capstyle='round')
    if len(all_plume2_bout) > 0:
        summary_vector2 = np.mean(np.column_stack((np.cos(all_plume2_bout), np.sin(all_plume2_bout))), axis=0)
        rho2, phi2 = cart2pol(summary_vector2[0], summary_vector2[1])
        plt.polar([0, phi2], [0, rho2], color=colors[1], alpha=1, linewidth=3, label='plume 2', solid_capstyle='round')
    plt.title('Plume 2 avg entry angle', color=markc2, fontsize=18)
    plt.show()
    fig_folder = os.path.join(folder_path, 'fig')
    os.makedirs(fig_folder, exist_ok=True)
    fig1.savefig(os.path.join(fig_folder, f'{savename}_plume1.pdf'), bbox_inches='tight')
    fig2.savefig(os.path.join(fig_folder, f'{savename}_plume2.pdf'), bbox_inches='tight')

def zigzag_entry_angle_strength(folder_path, savename, size=(5, 5), colors=['lightgrey', '#ff355e']):
    all_plume1_bout = []
    all_plume2_bout = []
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment  
        if 'ctrl' not in filename:
            _, d_on, d_off = light_on_off(exp_df)
            plume1_df = d_off[list(d_off.keys())[0]]
            plume2_df = d_off[list(d_off.keys())[-1]]
            _, d_in1, d_out1 = inside_outside_acv(plume1_df)
            _, d_in2, d_out2 = inside_outside_acv(plume2_df)
            plume1_bout = []
            plume2_bout = []
            for key, df in d_out1.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                    df = get_last_second(df)
                    p1_circmean = circmean_heading(df, plume1_bout)
                    
            for key, df in d_out2.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                    df = get_last_second(df)
                    p2_circmean = circmean_heading(df, plume2_bout)
            plt.plot([1,2], [(mean_resultant_vector_length(plume1_bout)),(mean_resultant_vector_length(plume2_bout))], color=colors[0], zorder=0)
            all_plume1_bout.append(mean_resultant_vector_length(plume1_bout))
            all_plume2_bout.append(mean_resultant_vector_length(plume2_bout))
    plt.scatter([1] * len(all_plume1_bout), all_plume1_bout, color=colors[0], zorder=20)
    plt.scatter([2] * len(all_plume2_bout), all_plume2_bout, color=colors[1], zorder=20)
    avg_p1, std_p1 = np.mean(all_plume1_bout), np.std(all_plume1_bout)
    avg_p2, std_p2 = np.mean(all_plume2_bout), np.std(all_plume2_bout)
    # plt.hlines(avg_p1, 0.9, 1.1, colors=markc, linewidth=2)
    # plt.hlines(avg_p2, 1.9, 2.1, colors=markc, linewidth=2)
    stat, p_value = wilcoxon(all_plume1_bout, all_plume2_bout)
    p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(all_plume1_bout, default=0), max(all_plume2_bout, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)
    plt.ylabel('entry angle strength', fontsize=18, color=markc)
    plt.yticks([0, 1], fontsize=14, color=markc)
    axs.tick_params(axis='x', length=0)          # Remove x-axis tick marks
    axs.set_xticklabels([])   
    plt.xlim(0.5, 2.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

    
def oct_training_travel_from_plume(folder_path, savename, size=(5, 5), colors=['lightgrey', '#ff355e'], keywords=['oct', 'Ltrain', 'Rtrain']):
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=size)
    params_list = exp_parameters(folder_path)
    plume_dfs = {key: [] for key in keywords}
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        if 'training' in filename:
            _, don, doff = light_on_off(exp_df)
            plume_df = doff[list(doff.keys())[-1]]
            plume_start = plume_df[plume_df['instrip']].index.min()
            plume_df = plume_df.loc[plume_start:]
        else:
            plume_df = exp_df
        plume_df = calculate_trav_dir(plume_df)
        _, d_in, d_out = inside_outside(plume_df)
        if len(d_out) > 0:
            first_exit = d_out[list(d_out.keys())[0]]
            end_idx_label = first_exit.index[-1]
            end_idx_pos = plume_df.index.get_loc(end_idx_label)
            plume_df = plume_df.iloc[:end_idx_pos]
        if 'LR' in filename or 'mch' in filename:
            key = keywords[1]  # 'L training'
        elif 'RR' in filename:
            key = keywords[2]  # 'R training'
        else:
            key = keywords[0]  # 'no training'
        plume_dfs[key].append(plume_df)
    for i, df in enumerate(plume_dfs):
        ax = axs[i]
        color = colors[i]
        title = keywords[i]
        headings = df['trav_dir'].dropna().values
        for theta in headings:
            ax.plot([0, theta], [0, 1], color=color, alpha=0.1, linewidth=1)
        mean_vector = np.mean(np.column_stack((np.cos(headings), np.sin(headings))), axis=0)
        mean_rho, mean_phi = cart2pol(mean_vector[0], mean_vector[1])
        ax.plot([0, mean_phi], [0, mean_rho], color=color, linewidth=3)
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.show()
    # Save the plot to a file
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def return_efficiency(folder_path, savename, size=(3,5), groups=2, keywords = ['Dop1R1', 'Dop1R2'], colors=['#c1ffc1', '#6f00ff']):
    fig, axs = configure_bw_plot(size=size, xaxis=False)
    returns = {f'returns_r{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        d, d_in, d_out = inside_outside(exp_df)
        pl = get_a_bout_calc(exp_df, 'path_length') / 1000
        counts = {f'returns{i+1}': 0 for i in range(groups)}
        for i in range(groups):
            if keywords[i] in filename:
                for key, df in d_out.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                        counts[f'returns{i+1}'] += 1
                returns[f'returns_r{i+1}'].append(counts[f'returns{i+1}'] / pl)
    averages = [sum(returns[f'returns_r{i+1}']) / len(returns[f'returns_r{i+1}']) if returns[f'returns_r{i+1}'] else 0 for i in range(groups)]
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(returns[f'returns_r{i+1}'])) for i in range(groups)]
    for i in range(groups):
        plt.scatter(x_values[i], returns[f'returns_r{i+1}'], color=colors[i % len(colors)], alpha=0.5)
    plt.plot(range(1, groups+1), averages, color='white')
    # for i in range(groups):
    #     plt.scatter(i+1, averages[i], color='none', edgecolor='white', marker='o', linewidth=2, s=100)
    plt.ylabel('returns per meter', fontsize=18, color='white')
    plt.yticks(fontsize=14, color='white')
    axs.set_xticks(range(1, groups+1))
    axs.set_xticklabels(keywords, fontsize=16, color='white', rotation=45)
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')



def plume_onset_uws(folder_path, savename, multiodor=True, size=(6,5), sample_rate=20, buf_pts=4,
                    keywords=['no training', 'L training', 'R training'],
                    colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    time = np.linspace(-buf_pts, buf_pts, buf_pts * sample_rate * 2)
    uws_traces = {kw: [] for kw in keywords}
    if multiodor:
        params_list = exp_parameters_multiodor(folder_path)
    else:
        params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        if multiodor:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        else:
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        _, di, do = inside_outside(exp_df)
        _, don, doff = light_on_off(exp_df)
        print(filename)
        try:
            if 'training' in filename:
                plume_df = doff[list(doff.keys())[-1]]
                plume_start = plume_df[plume_df['instrip']].index.min()
                plume_df = plume_df.loc[plume_start:]
                _, d_in, d_out = inside_outside(plume_df)
                first_key = list(d_in.keys())[0]
                df_in = d_in[first_key]
                if 'LR' in filename or 'mch' in filename:
                    key = keywords[1]
                elif 'RR' in filename:
                    key = keywords[2]
                else:
                    key = keywords[0]
            else:
                first_key = list(di.keys())[0]
                df_in = di[first_key]
                key = keywords[0]
            onset_time = df_in.index[0]
            half_window = buf_pts * sample_rate
            if onset_time in df.index:
                onset_idx = df.index.get_loc(onset_time)
                # Make sure enough data exists before and after
                if onset_idx >= half_window and onset_idx + half_window < len(df):
                    segment = df.iloc[onset_idx - half_window : onset_idx + half_window]
                    uws_segment = segment['y-vel'].values
                    baseline = np.mean(uws_segment[:half_window])
                    norm_uws = uws_segment - baseline
                    uws_traces[key].append(norm_uws)
                    print('found segment')
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
    for key, traces in uws_traces.items():
        if traces:
            mean_trace = np.mean(traces, axis=0)
            sem_trace = sem(traces, axis=0)
            color = colors[keywords.index(key)]
            axs.plot(time, mean_trace, color=color, lw=2, label=key)
            axs.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace,
                             color=color, alpha=0.2)
    axs.axvline(0, color=markc, linestyle='--')
    axs.set_xlabel("time (s)", color=markc, fontsize=14)
    axs.set_ylabel("upwind speed (mm/s)", color=markc, fontsize=14)
    plt.legend(prop={'size': 10}, frameon=False, loc='upper left', bbox_to_anchor=(1.1, 1.1))
    plt.title("upwind speed at plume onset", color=markc, fontsize=16)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def oct_training_plume_onset_traj(folder_path, savename, multiodor=True, size=(6,5), time=True, keywords=['no training', 'L training', 'R training'], colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=(size[0], size[1]*2), xaxis=True, nrows=2, ncols=3)
    axs = np.array(axs).reshape(2, 3)
    if multiodor:
        params_list = exp_parameters_multiodor(folder_path)
    else:
        params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        if this_experiment[0] == 'multiodor':
            experiment_type, figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        else:
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment   
        if 'training' in filename:
            _, don, doff = light_on_off(exp_df)
            plume_df = doff[list(doff.keys())[-1]]
            plume_start = plume_df[plume_df['instrip']].index.min()
            plume_df = plume_df.loc[plume_start:]
        else:
            plume_df = exp_df
        _, d_in, d_out = inside_outside(plume_df)
        # print(len(d_out))
        if (len(d_out)) > 0:
            # first_exit = d_out[list(d_out.keys())[0]] 
            # end_idx_label = first_exit.index[-1]
            # end_idx_pos = plume_df.index.get_loc(end_idx_label)
            # plume_df = plume_df.iloc[:end_idx_pos]
            plume_df['rel_seconds'] = plume_df['seconds'] - plume_df['seconds'].iloc[0]
            plume_df = plume_df[plume_df['rel_seconds'] <= 30]
            xo = plume_df.iloc[0]['ft_posx']
            yo = plume_df.iloc[0]['ft_posy']
            end_x = plume_df.iloc[-1]['ft_posx'] 
            end_y = plume_df.iloc[-1]['ft_posy'] 
            dx = end_x - xo
            dy = end_y - yo
            if 'mch' in filename:
                plume_df_odor = plume_df.where(plume_df.mfc2_stpt > 0)
            else:
                plume_df_odor = plume_df.where(plume_df.mfc3_stpt > 0)
            if 'LR' in filename:
                axs[0,1].plot(plume_df['ft_posx'] - xo, plume_df['ft_posy'] - yo, color='grey', linewidth=1)
                axs[0,1].plot(plume_df_odor['ft_posx'] - xo, plume_df_odor['ft_posy'] - yo, color=colors[1],  linewidth=1)
                axs[0,1].set_title(keywords[1], fontsize=14, color='white')
                axs[1,1].arrow(0, 0, dx, dy, head_width=10, color=colors[1], alpha=0.8)
            elif 'RR' in filename:
                axs[0,2].plot(plume_df['ft_posx'] - xo, plume_df['ft_posy'] - yo, color='grey', linewidth=1)
                axs[0,2].plot(plume_df_odor['ft_posx'] - xo, plume_df_odor['ft_posy'] - yo, color= colors[2], linewidth=1)
                axs[0,2].set_title(keywords[2], fontsize=14, color='white')
                axs[1,2].arrow(0, 0, dx, dy, head_width=10, color=colors[1], alpha=0.8)
            else:
                axs[0,0].plot(plume_df['ft_posx'] - xo, plume_df['ft_posy'] - yo, color='grey', linewidth=1)
                axs[0,0].plot(plume_df_odor['ft_posx'] - xo, plume_df_odor['ft_posy'] - yo, color=colors[0], linewidth=1)
                axs[0,0].set_title(keywords[0], fontsize=14, color='white')
                axs[1,0].arrow(0, 0, dx, dy, head_width=10, color=colors[0], alpha=0.8)
    
    xlims = [ax.get_xlim() for row in axs for ax in row]
    ylims = [ax.get_ylim() for row in axs for ax in row]
    xmin = min(x[0] for x in xlims)
    xmax = max(x[1] for x in xlims)
    ymin = min(y[0] for y in ylims)
    ymax = max(y[1] for y in ylims)
    x_range = xmax - xmin
    y_range = ymax - ymin
    max_range = max(x_range, y_range)
    xmid = (xmax + xmin) / 2
    ymid = (ymax + ymin) / 2
    new_xlim = (xmid - max_range / 2, xmid + max_range / 2)
    new_ylim = (ymid - max_range / 2, ymid + max_range / 2)

    for ax in axs[1]:
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
    for ax in axs.flat:
        ax.set_xlabel('x-position (mm)', fontsize=14)
        ax.set_ylabel('y-position (mm)', fontsize=14)
        ax.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        ax.tick_params(axis='both', colors='white')
        ax.set_aspect('equal', adjustable='datalim') 
        for pos in ['right', 'top']:
            ax.spines[pos].set_visible(False)
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in ax.spines.items():
            spine.set_linewidth(2)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def avg_plume_onset_uws(folder_path, savename, multiodor=True, size=(6,5), sample_rate=10, buf_pts=4,
                        keywords=['no training', 'L training', 'R training'],
                        colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    uws_avg = {kw: [] for kw in keywords}
    if multiodor:
        params_list = exp_parameters_multiodor(folder_path)
    else:
        params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        if multiodor:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        else:
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        _, di, do = inside_outside(exp_df)
        _, don, doff = light_on_off(exp_df)
        try:
            if 'training' in filename:
                plume_df = doff[list(doff.keys())[-1]]
                plume_start = plume_df[plume_df['instrip']].index.min()
                plume_df = plume_df.loc[plume_start:]
                _, d_in, d_out = inside_outside(plume_df)
                first_key = list(d_in.keys())[0]
                df_in = d_in[first_key]
                if 'LR' in filename or 'mch' in filename:
                    key = keywords[1]
                elif 'RR' in filename:
                    key = keywords[2]
                else:
                    key = keywords[0]
            else:
                first_key = list(di.keys())[0]
                df_in = di[first_key]
                key = keywords[0]
            onset_time = df_in.index[0]
            half_window = buf_pts * sample_rate
            if onset_time in df.index:
                onset_idx = df.index.get_loc(onset_time)
                # Ensure enough data exists before and after
                if onset_idx >= half_window and onset_idx + half_window < len(df):
                    segment = df.iloc[onset_idx - half_window : onset_idx + half_window]
                    uws_segment = segment['y-vel'].values
                    post_onset_avg = np.mean(uws_segment[half_window:])  # second half = post-onset
                    uws_avg[key].append(post_onset_avg)
                    print('found segment')
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
    for i, key in enumerate(keywords):
        x_noise = np.random.uniform(-0.05, 0.05, len(uws_avg[key]))
        axs.scatter(np.full(len(uws_avg[key]), i) + x_noise, uws_avg[key], color=colors[i])
    avg_values = [np.mean(uws_avg[key]) if uws_avg[key] else np.nan for key in keywords]
    std_values = [np.std(uws_avg[key]) if len(uws_avg[key]) > 1 else 0 for key in keywords]
    for i, (mean, error) in enumerate(zip(avg_values, std_values)):
        axs.hlines(mean, i - 0.1, i + 0.1, colors=markc, linewidth=2)
        # axs.errorbar(i, mean, yerr=error, color=markc, capsize=0)
    axs.set_ylabel("avg. upwind speed (mm/s)", color=markc, fontsize=14)
    axs.set_title(f'avg. upwind speed\n{buf_pts}s after plume onset', color=markc, fontsize=16)
    axs.set_xticks(range(len(keywords)))
    axs.set_xticklabels(keywords, fontsize=14, rotation=45)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def disappearing_cw_vs_returns(folder_path, savename, keyword='light', size=(5,5)):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    light_disp_list = []
    light_returns_list = []
    ctrl_disp_list = []
    ctrl_returns_list = []
    light_disp_list = []
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        d, d_in, d_out = inside_outside(exp_df)
        odor_off_idx = df_odor.dropna().index[-1]
        disappeared_df = exp_df[odor_off_idx:]
        if keyword in filename:
            # calculate crosswind displacement after plume disappears
            light_disp_list.append(np.abs((disappeared_df['ft_posx'].iloc[-1] - disappeared_df['ft_posx'].iloc[0])))
            # count number of returns to plume
            returns = 0
            for key, df in d_out.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                    returns += 1
            light_returns_list.append(returns)
        elif keyword not in filename:
            # calculate crosswind displacement after plume disappears
            ctrl_disp_list.append(np.abs((disappeared_df['ft_posx'].iloc[-1] - disappeared_df['ft_posx'].iloc[0])))
            # count number of returns to plume
            returns = 0
            for key, df in d_out.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                    returns += 1
            ctrl_returns_list.append(returns)
    plt.scatter(ctrl_returns_list, ctrl_disp_list, color='lightgrey', edgecolor='none', marker='o', linewidth=2, s=50)
    plt.scatter(light_returns_list, light_disp_list, color='#ff355e', edgecolor='none', marker='o', linewidth=2, s=50)
    plt.xlabel('returns to plume', fontsize=18, color=markc)
    plt.ylabel('crosswind displacement (mm)', fontsize=18, color=markc)
    plt.xticks(fontsize=14, color='white')
    plt.yticks(fontsize=14, color='white')
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def disappearing_cw_vs_uw(folder_path, savename, keyword='light', size=(5,5)):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    light_uw_disp_list = []
    light_cw_disp_list = []
    ctrl_uw_disp_list = []
    ctrl_cw_disp_list = []
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        d, d_in, d_out = inside_outside(exp_df)
        odor_off_idx = df_odor.dropna().index[-1]
        tracking_df = exp_df[:odor_off_idx]
        if keyword in filename:
            # calculate upwind displacement in plume
            light_uw_disp_list.append(tracking_df['ft_posy'].max() - tracking_df['ft_posy'].min())
            # calculate crosswind displacement after plume disappears
            disappeared_df = exp_df[odor_off_idx:]
            light_cw_disp_list.append(np.abs(disappeared_df['ft_posx'].iloc[-1] - disappeared_df['ft_posx'].iloc[0]))
        elif keyword not in filename:
            # calculate upwind displacement in plume
            ctrl_uw_disp_list.append(tracking_df['ft_posy'].max() - tracking_df['ft_posy'].min())
            # calculate crosswind displacement after plume disappears
            disappeared_df = exp_df[odor_off_idx:]
            ctrl_cw_disp_list.append(np.abs(disappeared_df['ft_posx'].iloc[-1] - disappeared_df['ft_posx'].iloc[0]))
    plt.scatter(light_uw_disp_list, light_cw_disp_list, color='#ff355e', edgecolor='none', marker='o', linewidth=2, s=50)
    plt.scatter(ctrl_uw_disp_list, ctrl_cw_disp_list, color='lightgrey', edgecolor='none', marker='o', linewidth=2, s=50)
    plt.xlabel('upwind displacement (mm)', fontsize=18, color=markc)
    plt.ylabel('crosswind displacement (mm)', fontsize=18, color=markc)
    plt.xticks(fontsize=14, color='white')
    plt.yticks(fontsize=14, color='white')
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def cc_return_efficiency(folder_path, savename, size=(5, 5), groups=3, keywords=['ctrl', 'crisscross_oct'], colors=['7ed4e6','#ff355e']):
    fig, axs, markc = configure_white_plot(size=size, xaxis=False)
    returns = {f'returns_r{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        ypos = exp_df['ft_posy']
        b2_start = exp_df[(ypos > 350) & (exp_df['odor_on'] == False)].index[0]
        exp_df = exp_df.loc[b2_start:]
        d, d_in, d_out = inside_outside(exp_df)
        pl = get_a_bout_calc(exp_df, 'path_length') / 1000
        counts = {f'returns{i+1}': 0 for i in range(groups)}
        for i in range(groups):
            if keywords[i] in filename:
                for key, df in d_out.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                        counts[f'returns{i+1}'] += 1
                returns[f'returns_r{i+1}'].append(counts[f'returns{i+1}'] / pl)
    
    # Calculate averages
    averages = [sum(returns[f'returns_r{i+1}']) / len(returns[f'returns_r{i+1}']) if returns[f'returns_r{i+1}'] else 0 for i in range(groups)]
    
    # Perform statistical comparison
    if groups == 2:  # Statistical comparison is only meaningful with 2 groups
        t_stat, p_value = ttest_ind(returns['returns_r1'], returns['returns_r2'])
        p_text = f'p = {p_value:.3g}'
    else:
        p_text = None

    # Generate random noise for scatter
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(returns[f'returns_r{i+1}'])) for i in range(groups)]
    std_devs = [np.std(returns[f'returns_r{i+1}']) if returns[f'returns_r{i+1}'] else 0 for i in range(groups)]

    # Plot individual data points and group averages
    for i in range(groups):
        plt.scatter(x_values[i], returns[f'returns_r{i+1}'], color=colors[i % len(colors)], alpha=0.5)
        bar = plt.bar(i + 1, averages[i], facecolor='none', edgecolor='markc', linewidth=2, alpha=1, width=0.5)
        plt.errorbar(i + 1, averages[i], yerr=std_devs[i], fmt='none', ecolor='markc', elinewidth=2, capsize=0)

    # Add p-value to the plot
    if p_text:
        plt.text(1.5, max(max(returns['returns_r1']), max(returns['returns_r2'])) + 0.2, p_text, ha='center', fontsize=14)

    # Customize the plot
    plt.ylabel('returns per meter', fontsize=18)
    plt.yticks(fontsize=14)
    axs.set_xticks(range(1, groups+1))
    axs.set_xticklabels(keywords, fontsize=16, rotation=45)
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()

    # Show and save the plot
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def octs_return_efficiency(folder_path, savename, size=(5, 5), groups=3, plotc='black', keywords=['plume', 'ctrl', 'crisscross_oct'], colors=['7ed4e6','#ff355e', '#ff355e']):
    if plotc == 'black':
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    elif plotc == 'white':
        fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    returns = {f'returns_r{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        counts = {f'returns{i+1}': 0 for i in range(groups)}
        for i in range(groups):
            if keywords[i] in filename:
                if keywords[i] == 'plume' or keywords[i] == 'ctrl':
                    d, d_in, d_out = inside_outside(exp_df)
                    pl = get_a_bout_calc(exp_df, 'path_length') / 1000
                    for key, df in d_out.items():
                        if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                            counts[f'returns{i+1}'] += 1
                    returns[f'returns_r{i+1}'].append(counts[f'returns{i+1}'] / pl)
                elif keywords[i] == 'crisscross_oct':
                    ypos = exp_df['ft_posy']
                    b2_start = exp_df[(ypos > 350) & (exp_df['odor_on'] == False)].index[0]
                    exp_df = exp_df.loc[b2_start:]
                    d, d_in, d_out = inside_outside(exp_df)
                    pl = get_a_bout_calc(exp_df, 'path_length') / 1000
                    for key, df in d_out.items():
                        if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                            counts[f'returns{i+1}'] += 1
                    returns[f'returns_r{i+1}'].append(counts[f'returns{i+1}'] / pl)
    
    averages = [sum(returns[f'returns_r{i+1}']) / len(returns[f'returns_r{i+1}']) if returns[f'returns_r{i+1}'] else 0 for i in range(groups)]
    # Plot individual data points and group averages
    alphas = [0.5, 0.5, 1]
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(returns[f'returns_r{i+1}'])) for i in range(groups)]
    for i in range(groups):
        plt.scatter(x_values[i], returns[f'returns_r{i+1}'], color=colors[i % len(colors)], alpha=alphas[i])
    plt.plot([1,2,3], averages, color=markc, linewidth=1)
    plt.ylabel('returns per meter', color=markc, fontsize=18)
    plt.yticks(fontsize=14)
    axs.set_xticks(range(1, groups+1))
    axs.set_xticklabels(keywords, fontsize=16, rotation=45)
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def uw_tracking(folder_path, savename, size=(3,5), groups=2, keywords = ['Dop1R1', 'Dop1R2'], colors=['#c1ffc1', '#6f00ff']):
    fig, axs = configure_bw_plot(size=size, xaxis=False)
    dists = {f'dists_{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000] # cut off at 1000
        df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000] # cut off at 1000
        d, d_in, d_out = inside_outside(exp_df)
        for i in range(groups):
            if keywords[i] in filename:
                uw = df_odor['ft_posy'].max() - yo
                dists[f'dists_{i+1}'].append(uw/1000)
    averages = [sum(dists[f'dists_{i+1}']) / len(dists[f'dists_{i+1}']) if dists[f'dists_{i+1}'] else 0 for i in range(groups)]
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(dists[f'dists_{i+1}'])) for i in range(groups)]
    for i in range(groups):
        plt.scatter(x_values[i], dists[f'dists_{i+1}'], color=colors[i % len(colors)], alpha=0.5)
    plt.plot(range(1, groups+1), averages, color='white')
    # for i in range(groups):
    #     plt.scatter(i+1, averages[i], color='none', edgecolor='white', marker='o', linewidth=2, s=100)
    plt.ylabel('distance tracked (m)', fontsize=18, color='white')
    plt.yticks(fontsize=14, color='white')
    axs.set_xticks(range(1, groups+1))
    axs.set_xticklabels(keywords, fontsize=16, color='white', rotation=45)
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def walking_speed_id(folder_path, savename, size=(3,5), groups=2, keywords=['ctrl', 'light'], colors=['grey', '#0bdf51'], plotc="white"):
    if plotc=='white':
        fig, axs, markc = configure_white_plot(size=size, xaxis=False)
    if plotc=='black':
        fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    subject_data = {}
    group_data = {keyword: [] for keyword in keywords}  # Collect uw values for each group
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]  # Cut off at 1000
        df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000]  # Cut off at 1000 
        # Extract subject identifier from filename using regex
        match = re.search(r'Fly\d+', filename)
        if match:
            subject_id = match.group()
        else:
            continue  # Skip if no subject ID is found
        if subject_id not in subject_data:
            subject_data[subject_id] = {}
        ws = df_odor['speed']
        for i in range(groups):
            if keywords[i] in filename:
                subject_data[subject_id][keywords[i]] = ws.mean()
                group_data[keywords[i]].append(ws.mean())  # Collect walking speed value for statistical testing
    # Calculate averages for plotting
    averages = []
    for keyword in keywords:
        ws_values = group_data[keyword]
        avg = sum(ws_values) / len(ws_values) if ws_values else 0
        averages.append(avg)
    p_value, p_text, significance = kw_stats(group_data, keywords)
    noise = 0.05
    for subject_id, data in subject_data.items():
        x = []
        y = []
        x_noisy = []
        for i, keyword in enumerate(keywords):
            if keyword in data:
                x_val = i + 1
                y_val = data[keyword]
                x_noise = x_val + np.random.normal(0, noise)
                x.append(x_val)
                y.append(y_val)
                x_noisy.append(x_noise)
                axs.scatter(x_noise, y_val, color=colors[i % len(colors)], alpha=0.5)
        if len(x) > 1:
            axs.plot(x_noisy, y, color='grey', alpha=0.5)
        axs.plot(range(1, groups + 1), averages, color=markc, linewidth=2)
    if p_value is not None:
        max_y = max(max(group_data[keyword]) for keyword in keywords)
        y_pos = max_y + 0.1 * max_y  # Adjust y position as needed
        # axs.text((1 + groups) / 2, y_pos, p_text, ha='center', va='bottom', fontsize=14, color=markc)
        # axs.plot([1, groups], [y_pos*0.95, y_pos*0.95], color=markc, linewidth=1)
    else:
        print("Cannot annotate plot due to insufficient data.")
    axs.set_ylabel('avg. walking speed (mm/s)', fontsize=18)
    axs.tick_params(axis='y', labelsize=14)
    axs.set_xticks(range(1, groups + 1))
    axs.set_xticklabels(keywords, fontsize=16, rotation=45)
    axs.set_xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def uw_tracking_id(folder_path, savename, size=(3,5), groups=2, keywords=['ctrl', 'light'], colors=['grey', '#0bdf51'], plotc="white"):
    if plotc=='white':
        fig, axs, markc = configure_white_plot(size=size, xaxis=False)
    if plotc=='black':
        fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    subject_data = {}
    group_data = {keyword: [] for keyword in keywords}  # Collect uw values for each group
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]  # Cut off at 1000
        df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000]  # Cut off at 1000 
        # Extract subject identifier from filename using regex
        match = re.search(r'Fly\d+', filename)
        if match:
            subject_id = match.group()
        else:
            continue  # Skip if no subject ID is found
        if subject_id not in subject_data:
            subject_data[subject_id] = {}
        uw = (df_odor['ft_posy'].max() - yo) / 1000  # Convert to meters
        for i in range(groups):
            if keywords[i] in filename:
                subject_data[subject_id][keywords[i]] = uw
                group_data[keywords[i]].append(uw)  # Collect uw value for statistical testing
    # Calculate averages for plotting
    averages = []
    for keyword in keywords:
        uw_values = group_data[keyword]
        avg = sum(uw_values) / len(uw_values) if uw_values else 0
        averages.append(avg)
    # Call the run_stats function
    p_value, p_text, significance = kw_stats(group_data, keywords)
    noise = 0.05
    for subject_id, data in subject_data.items():
        x = []
        y = []
        x_noisy = []
        for i, keyword in enumerate(keywords):
            if keyword in data:
                x_val = i + 1
                y_val = data[keyword]
                x_noise = x_val + np.random.normal(0, noise)
                x.append(x_val)
                y.append(y_val)
                x_noisy.append(x_noise)
                axs.scatter(x_noise, y_val, color=colors[i % len(colors)], alpha=0.5)
        if len(x) > 1:
            axs.plot(x_noisy, y, color='grey', alpha=0.5)
    axs.plot(range(1, groups + 1), averages, color=markc, linewidth=2)
    if p_value is not None:
        max_y = max(max(group_data[keyword]) for keyword in keywords)
        y_pos = max_y + 0.1 * max_y  # Adjust y position as needed
        # axs.text((1 + groups) / 2, y_pos, p_text, ha='center', va='bottom', fontsize=14, color=markc)
        # axs.plot([1, groups], [y_pos*0.95, y_pos*0.95], color=markc, linewidth=1)
    else:
        print("Cannot annotate plot due to insufficient data.")
    axs.tick_params(axis='y', labelsize=14)
    axs.set_xticks(range(1, groups + 1))
    axs.set_xticklabels(keywords, fontsize=16, rotation=45)
    axs.set_ylabel('distance tracked (m)', color=markc, fontsize=18)
    axs.set_xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def uw_tracking_id(folder_path, savename, size=(3,5), groups=2, keywords=['ctrl', 'light'], colors=['grey', '#0bdf51'], plotc="white"):
    if plotc=='white':
        fig, axs, markc = configure_white_plot(size=size, xaxis=False)
    if plotc=='black':
        fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    subject_data = {}
    group_data = {keyword: [] for keyword in keywords}  # Collect uw values for each group
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]  # Cut off at 1000
        df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000]  # Cut off at 1000 
        # Extract subject identifier from filename using regex
        match = re.search(r'Fly\d+', filename)
        if match:
            subject_id = match.group()
        else:
            continue  # Skip if no subject ID is found
        if subject_id not in subject_data:
            subject_data[subject_id] = {}
        uw = (df_odor['ft_posy'].max() - yo) / 1000  # Convert to meters
        for i in range(groups):
            if keywords[i] in filename:
                subject_data[subject_id][keywords[i]] = uw
                group_data[keywords[i]].append(uw)  # Collect uw value for statistical testing
    # Calculate averages for plotting
    averages = []
    for keyword in keywords:
        uw_values = group_data[keyword]
        avg = sum(uw_values) / len(uw_values) if uw_values else 0
        averages.append(avg)
    # Call the run_stats function
    p_value, p_text, significance = kw_stats(group_data, keywords)
    noise = 0.05
    for subject_id, data in subject_data.items():
        x = []
        y = []
        x_noisy = []
        for i, keyword in enumerate(keywords):
            if keyword in data:
                x_val = i + 1
                y_val = data[keyword]
                x_noise = x_val + np.random.normal(0, noise)
                x.append(x_val)
                y.append(y_val)
                x_noisy.append(x_noise)
                axs.scatter(x_noise, y_val, color=colors[i % len(colors)], alpha=0.5)
        if len(x) > 1:
            axs.plot(x_noisy, y, color='grey', alpha=0.5)
    axs.plot(range(1, groups + 1), averages, color=markc, linewidth=2)
    if p_value is not None:
        max_y = max(max(group_data[keyword]) for keyword in keywords)
        y_pos = max_y + 0.1 * max_y  # Adjust y position as needed
        # axs.text((1 + groups) / 2, y_pos, p_text, ha='center', va='bottom', fontsize=14, color=markc)
        # axs.plot([1, groups], [y_pos*0.95, y_pos*0.95], color=markc, linewidth=1)
    else:
        print("Cannot annotate plot due to insufficient data.")
    axs.tick_params(axis='y', labelsize=14)
    axs.set_xticks(range(1, groups + 1))
    axs.set_xticklabels(keywords, fontsize=16, rotation=45)
    axs.set_ylabel('distance tracked (m)', color=markc, fontsize=18)
    axs.set_xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def pulse_return_efficiency(folder_path, savename, size=(5, 5), groups=2, colors=['white','red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    returns = {f'returns_r{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        #last_on_index = exp_df[exp_df['ft_posy'] - yo 
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000] # account for active tracking only
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        pl = get_a_bout_calc(exp_df, 'path_length') / 1000
        counts = {f'returns_{i+1}': 0 for i in range(groups)}
        if len(d_on) < 1:
            continue
            # for key, df in d_out.items():
            #     if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
            #         counts['returns_1'] += 1
            # returns['returns_r1'].append(counts['returns_1'] / pl)
        elif len(d_on) == 1:
            for key, df in d_out.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                    counts['returns_1'] += 1
            returns['returns_r2'].append(counts['returns_2'] / pl)
        elif len(d_on) > 1:
            for key, df in d_out.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                    counts['returns_1'] += 1
            returns['returns_r3'].append(counts['returns_3'] / pl)
            
    averages = [sum(returns[f'returns_r{i+1}']) / len(returns[f'returns_r{i+1}']) if returns[f'returns_r{i+1}'] else 0 for i in range(groups)]
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(returns[f'returns_r{i+1}'])) for i in range(groups)]
    for i in range(groups):
        plt.scatter(x_values[i], returns[f'returns_r{i+1}'], color='grey', alpha=0.5)
    plt.plot(range(1, groups+1), averages, color=markc)
    # for i in range(groups):
    #     plt.scatter(i+1, averages[i], color='none', edgecolor='white', marker='o', linewidth=2, s=100)
    plt.ylabel('returns per meter', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks(range(1, groups+1))
    axs.set_xticklabels(['single activation', 'multiple activation'], fontsize=16, color=markc, rotation=45)
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def pulse_entry_angle(folder_path, savename, size=(5, 5), colors=['white', 'red']):
    fig, axs, markc = configure_bw_polar(size=size, grid=False)
    params_list = exp_parameters(folder_path)
    single_entries = []
    multi_entries = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]  # Filter for active tracking
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        if len(d_on) == 1:
            single_entry, strength = compute_mean_entry_heading(d_out, imaging=False)
            single_entry = mirror(single_entry)
            single_entries.append(single_entry)
            plt.polar([0, single_entry], [0, strength], color=colors[0], alpha=0.5, linewidth=1, solid_capstyle='round')
        elif len(d_on) > 1:
            multi_entry, strength = compute_mean_entry_heading(d_out, imaging=False)
            multi_entry = mirror(multi_entry)
            multi_entries.append(multi_entry)
            plt.polar([0, multi_entry], [0, strength], color=colors[1], alpha=0.5, linewidth=1, solid_capstyle='round')
        else:
            continue
    print(len(single_entries))
    print(len(multi_entries))
    if len(single_entries) > 0:
        summary_single_vector = np.mean(np.column_stack((np.cos(single_entries), np.sin(single_entries))), axis=0)
        summary_single_rho, summary_single_phi = cart2pol(summary_single_vector[0], summary_single_vector[1])
        plt.polar([0, summary_single_phi], [0, summary_single_rho], color=colors[0], alpha=1, linewidth=3, label='single act.', solid_capstyle='round')
        
    if len(multi_entries) > 0:
        summary_multi_vector = np.mean(np.column_stack((np.cos(multi_entries), np.sin(multi_entries))), axis=0)
        summary_multi_rho, summary_multi_phi = cart2pol(summary_multi_vector[0], summary_multi_vector[1])
        plt.polar([0, summary_multi_phi], [0, summary_multi_rho], color=colors[1], alpha=1, linewidth=3, label='multi act.', solid_capstyle='round')
    plt.title('avg entry angle', color=markc, fontsize=18)
    # plt.legend(loc='upper right')
    plt.show()
    fig_folder = os.path.join(folder_path, 'fig')
    os.makedirs(fig_folder, exist_ok=True)
    fig.savefig(os.path.join(fig_folder, f'{savename}.pdf'), bbox_inches='tight')

def pulse_entry_angle_strength(folder_path, savename, size=(5, 5), colors=['white', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    single_strengths = []
    multi_strengths = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]  # Filter for active tracking
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        if len(d_on) == 1 and len(d_out) > 3:
            single_entry, strength = compute_mean_entry_heading(d_out, imaging=False)
            single_strengths.append(strength)     
            if strength > 0.9 or strength < 0.3:
                print(filename)
        elif len(d_on) > 1 and len(d_out) > 3:
            multi_entry, strength = compute_mean_entry_heading(d_out, imaging=False)
            multi_strengths.append(strength)  
            if strength > 0.9 or strength < 0.3:
                print(filename)
        else:   
            continue
    axs.scatter(np.full(len(single_strengths), 1), single_strengths, color=colors[0])
    axs.scatter(np.full(len(multi_strengths), 2), multi_strengths, color=colors[1])
    axs.set_xlim(0.5, 2.5)
    axs.set_ylim(0, 1.05)
    axs.set_xticks([1, 2])
    axs.set_xticklabels(['single', 'multi'], color=markc)
    axs.set_ylabel('entry strength (MRL)', color=markc)
    plt.tight_layout()
    plt.show()
    plt.show()
    fig_folder = os.path.join(folder_path, 'fig')
    os.makedirs(fig_folder, exist_ok=True)
    fig.savefig(os.path.join(fig_folder, f'{savename}.pdf'), bbox_inches='tight')


def alt_pulse_return_efficiency(folder_path, savename, size=(5, 5), groups=3, colors=['white','red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    dist1_list = []
    pl1_list = []
    dist2_list = []
    pl2_list = []
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]  # Cut off at y=1000
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        if len(d_on) < 1:
                    continue
        elif len(d_on) == 1:
            for key, df in d_out.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1 and return_to_edge(df):
                    dist = get_a_bout_calc(df, 'furthest_distance_from_plume')
                    pl = get_a_bout_calc(df, 'path_length')
                    plt.scatter(dist, pl, color=colors[0], alpha=0.5)
                    dist1_list.append(dist)
                    pl1_list.append(pl)
        elif len(d_on) > 1:
            print(filename)
            for key, df in d_out.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1 and return_to_edge(df):
                    dist = get_a_bout_calc(df, 'furthest_distance_from_plume')
                    pl = get_a_bout_calc(df, 'path_length')
                    plt.scatter(dist, pl, color=colors[1], alpha=0.5)
                    dist2_list.append(dist)
                    pl2_list.append(pl)
    dist1_array, pl1_array = np.array(dist1_list), np.array(pl1_list)
    dist2_array, pl2_array = np.array(dist2_list), np.array(pl2_list)
    if len(dist1_array) > 1:
        slope1, intercept1, _, _, _ = linregress(dist1_array, pl1_array)
        x_fit1 = np.linspace(min(dist1_array), max(dist1_array), 100)
        y_fit1 = slope1 * x_fit1 + intercept1
        plt.plot(x_fit1, y_fit1, color=colors[0], linestyle='-', linewidth=2, label='single')
    if len(dist2_array) > 1:
        slope2, intercept2, _, _, _ = linregress(dist2_array, pl2_array)
        x_fit2 = np.linspace(min(dist2_array), max(dist2_array), 100)
        y_fit2 = slope2 * x_fit2 + intercept2
        plt.plot(x_fit2, y_fit2, color=colors[1], linestyle='-', linewidth=2, label='multiple')
    plt.ylabel('total path length (mm)', fontsize=18, color=markc)
    plt.xlabel('furthest distance from plume (mm)', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    plt.xticks(fontsize=14, color=markc)
    axs.plot(np.linspace(0, 1000, 100), 2*(np.linspace(0, 1000, 100)), linestyle='--', color=markc)
    axs.set_xlim(-10, 200)
    legend = plt.legend(prop={'size': 10}, frameon=False)
    plt.setp(legend.get_texts(), color='white')
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')
            
  
def alt_pulse_return_efficiency_light(folder_path, savename, size=(5, 5), groups=2):
    fig, axs = configure_bw_plot(size=size, xaxis=True)
    returns = {f'returns_r{i+1}': [] for i in range(groups)}
    file_counts = {f'returns_r{i+1}': 0 for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]  # Cut off at y=1000
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        # if len(d_on) < 1:
        #     returns_key = 'returns_r1'
        # elif len(d_on) == 1:
        #     returns_key = 'returns_r2'
        # elif len(d_on) > 1:
        #     returns_key = 'returns_r3'
        # file_counts[returns_key] += 1

        # Create a combined list of bouts sorted by start time
        bouts = []
        for key, df_bout in d_in.items():
            bouts.append({'key': key, 'start_time': df_bout['seconds'].iloc[0], 'end_time': df_bout['seconds'].iloc[-1], 'label': 'in', 'df': df_bout})
        for key, df_bout in d_out.items():
            bouts.append({'key': key, 'start_time': df_bout['seconds'].iloc[0], 'end_time': df_bout['seconds'].iloc[-1], 'label': 'out', 'df': df_bout})
        bouts.sort(key=lambda x: x['start_time'])

        # Process each bout and check the preceding d_in bout
        for idx, bout in enumerate(bouts):
            if bout['label'] == 'out':
                if idx > 0:
                    preceding_bout = bouts[idx - 1]
                    if preceding_bout['label'] == 'in':
                        # Check if led1_stpt is 0.0 throughout the preceding d_in bout
                        if (preceding_bout['df']['led1_stpt'] == 1.0).any():
                            df = bout['df']
                            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                                cw = get_a_bout_calc(df, 'x_distance_from_plume')
                                pl = get_a_bout_calc(df, 'path_length')
                                returns['returns_r1'].append((cw, pl / 1000))  # cw in mm, pl in meters
                        elif (preceding_bout['df']['led1_stpt'] == 0.0).any():
                            df = bout['df']
                            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                                cw = get_a_bout_calc(df, 'x_distance_from_plume')
                                pl = get_a_bout_calc(df, 'path_length')
                                returns['returns_r2'].append((cw, pl / 1000))  # cw in mm, pl in meters
    colors = {'returns_r1': 'grey', 'returns_r2': 'red'}
    labels = {'returns_r1': 'no activation', 'returns_r2': 'activation'}
    slopes = {}
    for returns_key, data in returns.items():
        if data:
            print('found data')
            cw_values = [item[0] for item in data]  # x-values (distance away from plume)
            pl_values = [item[1] for item in data]  # y-values (total path length in meters)
            slope, intercept, r_value, p_value, std_err = linregress(cw_values, pl_values)
            slopes[returns_key] = slope  # Store the slope
            #n_points = file_counts[returns_key]
            label_with_slope = f"{labels[returns_key]} (slope={(slope*1000):.2f}"
            #label_with_slope = f"{labels[returns_key]} (slope={(slope*1000):.2f}, n={n_points} flies)"
            axs.scatter(cw_values, pl_values, color=colors[returns_key], label=label_with_slope)
    plt.ylabel('total path length (m)', fontsize=18, color='white')
    plt.xlabel('distance away from plume (mm)', fontsize=18, color='white')
    plt.yticks(fontsize=14, color='white')
    axs.plot(np.linspace(0, 1000, 100), .002*(np.linspace(0, 1000, 100)), linestyle='--', color='white')
    axs.set_xlim(-10,300)
    legend = plt.legend(prop={'size': 10}, frameon=False) 
    plt.setp(legend.get_texts(), color='white')
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def pulse_uw_tracking(folder_path, savename, size=(5, 5), plotc='black', groups=3):
    if plotc=='white':
        fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    if plotc=='black':
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    dists = {f'dists_{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000] # cut off at 1000
        df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000] # cut off at 1000
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        if len(d_on) < 1:
            uw = df_odor['ft_posy'].max() - yo
            dists['dists_1'].append(uw/1000)
        elif len(d_on) == 1:
            uw = df_odor['ft_posy'].max() - yo
            dists['dists_2'].append(uw/1000)
        elif len(d_on) > 1:
            uw = df_odor['ft_posy'].max() - yo
            dists['dists_3'].append(uw/1000)
    averages = [sum(dists[f'dists_{i+1}']) / len(dists[f'dists_{i+1}']) if dists[f'dists_{i+1}'] else 0 for i in range(groups)]
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(dists[f'dists_{i+1}'])) for i in range(groups)]
    colors = ['grey', '#ff355e', '#ff355e']
    alphas = [0.5, 0.5, 1]
    for i in range(groups):
        plt.scatter(x_values[i], dists[f'dists_{i+1}'], color=colors[i], alpha=alphas[i])
        plt.hlines(averages[i], xmin=i+0.9, xmax=i+1.1, colors=markc, linewidth=2)  # Mean line
    stat, p_value = kruskal(dists['dists_1'], dists['dists_2'], dists['dists_3'])
    print(f"Kruskal-Wallis H-test: H = {stat:.3f}, p = {p_value:.4f}")
    print(dists['dists_1'])
    print(dists['dists_2'])
    print(dists['dists_3'])
    plt.ylabel('distance tracked (m)', color=markc, fontsize=18)
    plt.yticks(fontsize=14)
    axs.set_xticks(range(1, groups+1))
    # axs.set_xticklabels(['no activation', 'single activation', 'multiple activation'], fontsize=16, color=markc, rotation=45)
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def pulse_cw_dist(folder_path, savename, size=(5, 5), plotc='black', groups=3):
    if plotc=='white':
        fig, axs, markc = configure_white_plot(size=size, xaxis=True)
        markc='black'
    if plotc=='black':
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
        markc='white'
    dists = {f'dists_{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]  # cut off at 1000
        df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000]  # cut off at 1000
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        cw_list = []
        for key, df_segment in d_out.items():
            if df_segment['seconds'].iloc[-1] - df_segment['seconds'].iloc[0] >= 0.5: #and return_to_edge(df_segment):  
                cw = get_a_bout_calc(df_segment, 'x_distance_from_plume')
                cw_list.append(cw)
        if cw_list:
            average_cw = sum(cw_list) / len(cw_list)
        else:
            average_cw = 0  # You can choose to skip this file if no 'cw' values
        if len(d_on) < 1:
            dists['dists_1'].append(average_cw)
        elif len(d_on) == 1:
            dists['dists_2'].append(average_cw)
        elif len(d_on) > 1:
            dists['dists_3'].append(average_cw)
    averages = [sum(dists[f'dists_{i+1}']) / len(dists[f'dists_{i+1}']) if dists[f'dists_{i+1}'] else 0 for i in range(groups)]
    print(averages)
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(dists[f'dists_{i+1}'])) for i in range(groups)]
    colors = ['grey', '#ff355e', '#ff355e']
    stat, p_value = kruskal(dists['dists_1'], dists['dists_2'], dists['dists_3'])
    print(f"Kruskal-Wallis H-test: H = {stat:.3f}, p = {p_value:.4f}")
    print(dists['dists_1'])
    print(dists['dists_2'])
    print(dists['dists_3'])
    alphas = [0.5, 0.5, 1]
    for i in range(groups):
        plt.scatter(
            x_values[i], 
            dists[f'dists_{i+1}'], 
            color=colors[i], 
            alpha=alphas[i]
        )
        plt.hlines(averages[i], xmin=i + 0.9, xmax=i + 1.1, colors=markc, linewidth=2)  # Mean line
    plt.ylabel('distance from plume (mm)', color=markc, fontsize=18)
    plt.yticks(fontsize=14)
    axs.set_xticks(range(1, groups+1))
    # axs.set_xticklabels(['no activation', 'single activation', 'multiple activation'], color=markc, fontsize=16, rotation=45)
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    fig_folder = f'{folder_path}/fig/'
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig.savefig(f'{fig_folder}/{savename}.pdf', bbox_inches='tight')

def pulse_cw_vs_uw_dist(folder_path, savename, size=(5, 5), plotc='black'):
    if plotc=='white':
        fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    if plotc=='black':
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    all_uw = []
    all_cw = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]  # cut off at 1000
        df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000]  # cut off at 1000
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        uw = df_odor['ft_posy'].max() - yo
        cw_list = []
        for key, df in d_out.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):  
                cw = get_a_bout_calc(df, 'x_distance_from_plume')
                cw_list.append(cw)
        if len(d_on) > 1:
            plt.scatter(np.mean(cw_list), uw, color='#ff355e')
            all_uw.append(uw)
            all_cw.append(np.mean(cw_list))
        if len(d_on) == 1:
            plt.scatter(np.mean(cw_list), uw, color='#ff355e', alpha=0.5)
            all_uw.append(uw)
            all_cw.append(np.mean(cw_list))
    all_uw = np.array(all_uw)
    all_cw = np.array(all_cw)
    valid_mask = ~np.isnan(all_uw) & ~np.isnan(all_cw)
    all_uw = all_uw[valid_mask]
    all_cw = all_cw[valid_mask]

    if len(all_uw) > 1 and len(all_cw) > 1:
        pearson_corr, p_val = pearsonr(all_cw, all_uw)
        spearman_corr, sp_p_val = spearmanr(all_cw, all_uw)
        print(f"Pearson correlation: r = {pearson_corr:.3f}, p = {p_val:.3g}")
        print(f"Spearman correlation: r = {spearman_corr:.3f}, p = {sp_p_val:.3g}")
    
    plt.ylabel('distance tracked (mm)', color=markc, fontsize=14)
    plt.xlabel('avg. distance from edge (mm)', color=markc, fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    fig_folder = f'{folder_path}/fig/'
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig.savefig(f'{fig_folder}/{savename}.pdf', bbox_inches='tight')

def pulse_cw_dist_VR(folder_path, savename, size=(5, 5), plotc='black', colors=['red', 'white']):
    if plotc=='white':
        fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    if plotc=='black':
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    all_reinf = []
    all_non = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]  # cut off at 1000
        df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000]  # cut off at 1000
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        reinf_list = []
        non_list = []
        if len(d_on) > 1:
            print(filename)
            reinf_idxs = list(d_on.keys())
            reinf_idxs = [int(idx) for idx in reinf_idxs]
            reinf_idxs = [idx + 1 for idx in reinf_idxs]
            print(reinf_idxs)
            print(list(d_out.keys()))
            for key, df_segment in d_out.items():
                if return_to_edge(df_segment):  
                    cw = get_a_bout_calc(df_segment, 'x_distance_from_plume')
                    if int(key) in reinf_idxs:
                        reinf_list.append(cw)
                    else:
                        non_list.append(cw)
            if len(non_list) >= 1:
                average_reinf = sum(reinf_list) / len(reinf_list)
                all_reinf.append(average_reinf)
                average_non = sum(non_list) / len(non_list)
                all_non.append(average_non)
                plt.plot([1,2], [average_reinf, average_non], color=markc, zorder=0)
                # print(filename)
                print(average_reinf / average_non)
                print
        
    plt.scatter([1]*len(all_reinf), all_reinf, color=colors[0], alpha=0.7, zorder=10)
    plt.scatter([2]*len(all_non), all_non, color=colors[1], alpha=0.7, zorder=10)
    
    plt.ylabel('distance from plume (mm)', color=markc, fontsize=18)
    axs.set_yscale('log')
    plt.yticks(fontsize=14)
    axs.set_xticks([1,2])
    axs.set_xticklabels(['reinforced', 'not reinforced'], color=markc, fontsize=16, rotation=45)
    plt.xlim(0.5, 2.5)
    plt.tight_layout()
    plt.show()
    fig_folder = f'{folder_path}/fig/'
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig.savefig(f'{fig_folder}/{savename}.pdf', bbox_inches='tight')

def pulse_cw_dist_VR_indiv(folder_path, savename, size=(5, 5), plotc='black', colors=['red', 'white']):
    if plotc=='white':
        fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    if plotc=='black':
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    reinf_list = []
    non_list = []
    flies=0
    total=0
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]  # cut off at 1000
        df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000]  # cut off at 1000
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        # reinf_list = []
        # non_list = []
        if len(d_on) > 1:
            flies+=1
            reinf_idxs = list(d_on.keys())
            reinf_idxs = [int(idx) for idx in reinf_idxs]
            reinf_idxs = [idx + 1 for idx in reinf_idxs]
            print(reinf_idxs)
            print(list(d_out.keys()))
            for key, df_segment in d_out.items():
                if df_segment['seconds'].iloc[-1] - df_segment['seconds'].iloc[0] >= 0.5 and return_to_edge(df_segment): 
                    total +=1 
                    cw = get_a_bout_calc(df_segment, 'path_length')
                    if int(key) in reinf_idxs:
                        reinf_list.append(cw)
                    else:
                        non_list.append(cw)
    average_reinf, std_reinf = np.mean(reinf_list), np.std(reinf_list)
    average_non, std_non = np.mean(non_list), np.std(non_list)
    # plt.plot([1,2], [average_reinf, average_non], color=markc, zorder=0)
    print(f'total: {total}')
    print(f'flies: {flies}')
    plt.scatter(1 + np.random.normal(0, 0.05, len(reinf_list)), reinf_list, color=colors[0], alpha=0.7)
    plt.scatter(2 + np.random.normal(0, 0.05, len(non_list)), non_list, color=colors[1], alpha=0.7)
    
    plt.hlines(average_reinf, xmin=0.9, xmax=1.1, colors=markc, linewidth=2)  # Mean line
    plt.hlines(average_non, xmin=1.9, xmax=2.1, colors=markc, linewidth=2)  # Mean line
    plt.errorbar(1, average_reinf, yerr=std_reinf, color=markc, capsize=0)
    plt.errorbar(2, average_non, yerr=std_non, color=markc, capsize=0)

    # Test normality of both groups
    stat_reinf, p_reinf = shapiro(reinf_list)
    stat_non, p_non = shapiro(non_list)

    print(f"Shapiro test - Reinforced: p = {p_reinf:.3e}")
    print(f"Shapiro test - Non-Reinforced: p = {p_non:.3e}")

    if p_reinf > 0.05 and p_non > 0.05:
        # Both distributions are normal – use t-test
        stat, p_value = ttest_ind(reinf_list, non_list, equal_var=False)
        test_type = "Independent t-test"
    else:
        # At least one distribution is non-normal – use Mann-Whitney U
        stat, p_value = mannwhitneyu(reinf_list, non_list, alternative='two-sided')
        test_type = "Mann-Whitney U test"

    print(f"{test_type}: statistic = {stat:.3f}, p = {p_value:.3e}")
    plt.ylabel('return length (mm)', color=markc, fontsize=18)
    axs.set_yscale('log')
    plt.yticks(fontsize=14)
    axs.set_xticks([1,2])
    axs.set_xticklabels(['reinforced', 'not reinforced'], color=markc, fontsize=16, rotation=45)
    plt.xlim(0.5, 2.5)
    plt.tight_layout()
    plt.show()
    fig_folder = f'{folder_path}/fig/'
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig.savefig(f'{fig_folder}/{savename}.pdf', bbox_inches='tight')


def reinforcement_hist_scatter(folder_path, savename='reinforcement_scatter', size=(10, 8), colors=('gray', 'red')):
    import numpy as np
    params_list = exp_parameters(folder_path)

    all_reinf = []
    all_non = []
    fly_labels = []

    for i, this_experiment in enumerate(params_list):
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000]
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)

        if len(d_on) > 1:
            reinf_idxs = [int(idx) + 1 for idx in d_on.keys()]
            reinf_path_lengths = []
            non_path_lengths = []
            for key, df_segment in d_out.items():
                if df_segment['seconds'].iloc[-1] - df_segment['seconds'].iloc[0] >= 0.5 and return_to_edge(df_segment):
                    path_length = get_a_bout_calc(df_segment, 'x_distance_from_plume')
                    if int(key) in reinf_idxs:
                        reinf_path_lengths.append(path_length)
                    else:
                        non_path_lengths.append(path_length)
            if len(non_path_lengths) >=1:
                all_reinf.append(reinf_path_lengths)
                all_non.append(non_path_lengths)
                fly_labels.append(filename)

    # Begin plotting
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    for i, (reinf, non) in enumerate(zip(all_reinf, all_non)):
        y_val = i + 1
        axs.scatter(reinf, [y_val]*len(reinf), color=colors[0], label='reinforced' if i == 0 else "", alpha=0.8)
        axs.scatter([-x for x in non], [y_val]*len(non), color=colors[1], label='non-reinforced' if i == 0 else "", alpha=0.8)
        
    axs.axvline(0, color=markc, linewidth=1)
    axs.set_yticks(range(1, len(fly_labels)+1))
    axs.set_xlabel('distance from plume (mm)', fontsize=14)
    
    # ax.legend()
    # sns.despine()
    plt.tight_layout()
    plt.show()
    figure_folder = os.path.join(folder_path, 'figure')
    os.makedirs(figure_folder, exist_ok=True)
    fig.savefig(os.path.join(figure_folder, f'{savename}.pdf'))
    
           
            

def pulse_xpos_distribution(folder_path, savename, size=(5, 5), groups=3):
    fig, axs = configure_bw_plot(size=size, xaxis=True)
    dists = {f'dists_{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000] # cut off at 1000
        df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000] # cut off at 1000
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        xpos = np.abs(df['ft_posx'] - xo)
        if len(d_on) < 1:
            dists['dists_1'].append(xpos)
        elif len(d_on) == 1:
            dists['dists_2'].append(xpos)
        elif len(d_on) > 1:
            dists['dists_3'].append(xpos)
    compiled_r1 = pd.concat(dists['dists_1'], axis=0, ignore_index=True)
    compiled_r2 = pd.concat(dists['dists_2'], axis=0, ignore_index=True)
    compiled_r3 = pd.concat(dists['dists_3'], axis=0, ignore_index=True)
    axs.axvline(x=25, color='white', linewidth=1)
    sns.kdeplot(compiled_r1, label='no activation', color='grey', fill=False, cut=0, ax=axs)
    sns.kdeplot(compiled_r2, label='single activation', color='#ff355e', alpha=0.5, fill=False, cut=0, ax=axs)
    sns.kdeplot(compiled_r3, label='multiple activation', color='#ff355e', fill=False, cut=0, ax=axs)
    plt.xticks([0, 250, 500], fontsize=14, color='white')  # Adjust tick locations as needed
    plt.yticks([0, 0.02], fontsize=14, color='white')  # Adjust tick locations as needed
    axs.set_xlim(left=0)  # Start the x-axis at 0
    axs.set_xlabel('x position', fontsize=16, color='white')  # Label for x-axis
    axs.set_ylabel('probability', fontsize=16, color='white')  # Label for y-axis
    legend = plt.legend(prop={'size': 14}, frameon=False)  # frameon=False removes the legend box
    axs.spines['left'].set_position(('data', 0))
    legend = plt.legend(prop={'size': 10}, frameon=False) 
    plt.setp(legend.get_texts(), color='white')
    plt.show()
    plt.tight_layout()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')
    

def block_return_efficiency(folder_path, savename, size=(3,5), cutoff=500, labels = ['LED off', 'LED on'], colors=['#c1ffc1', '#6f00ff']):
    # Right now hard coded for only 2 blocks
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    params_list = exp_parameters(folder_path)
    rpm_b1 = []
    rpm_b2 = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        # print(filename)
        ypos = exp_df['ft_posy'] - yo
        if '500' in filename:
            b1_df = exp_df[(ypos >= 0) & (ypos < cutoff)]
            b2_df = exp_df[(ypos >= cutoff)] # still not ideal
            b2_start = b2_df['seconds'].iloc[0]
        else:
            b1_df = exp_df[(ypos >= 0) & (ypos < 350)]
            b2_start = exp_df[(ypos > 350) & (exp_df['odor_on'] == False)].index[0]
            b2_df = exp_df.loc[b2_start:]
        d1, d_in1, d_out1 = inside_outside(b1_df)
        d2, d_in2, d_out2 = inside_outside(b2_df)
        pl1 = get_a_bout_calc(b1_df, 'path_length') / 1000
        pl2 = get_a_bout_calc(b2_df, 'path_length') / 1000
        returns_b1 = 0
        returns_b2 = 0
        for key, df in d_out1.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                    returns_b1+=1
        rpm_b1.append(returns_b1 / pl1)
        for key, df in d_out2.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                    returns_b2+=1
        rpm_b2.append(returns_b2 / pl2)  
        if (returns_b2 / pl2) > (returns_b1 / pl1):
            print(filename)
        plt.plot([1,2], [(returns_b1/pl1), (returns_b2/pl2)], color='lightgrey', alpha=0.5)
    b1_avg = sum(rpm_b1) / len(rpm_b1)
    b2_avg = sum(rpm_b2) / len(rpm_b2)  
    plt.scatter([1]*len(rpm_b1), rpm_b1, color=colors[0], alpha=0.5)
    plt.scatter([2]*len(rpm_b2), rpm_b2, color=colors[1], alpha=0.5)
    plt.plot([1,2],[b1_avg, b2_avg], color='white')
    plt.ylabel('returns per meter', fontsize=18, color='white')
    axs.set_xticks((1,2))
    axs.set_xticklabels(labels, fontsize=16, color='white', rotation=45)
    plt.xlim(0.5, 2.5)
    plt.tight_layout() 
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}', bbox_inches='tight')

def b2_baseline_comp(folder_path, savename, size=(3,5), cutoff=500, labels = ['baseline', 'first 2 min.', 'last 2 min.'], colors=['#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    params_list = exp_parameters(folder_path)
    # Empty lists for tortuosities across all experiments
    bl = []
    early_b2 = []
    late_b2 = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        ypos = exp_df['ft_posy'] - yo
        if '500' in filename:
            b2_df = exp_df[(ypos >= cutoff)] # still not ideal
            b2_start = b2_df['seconds'].iloc[0]
        else:
            b2_start = exp_df[(ypos > 350) & (exp_df['odor_on'] == False)].index[0]
            b2_df = exp_df.loc[b2_start:]
       
        b2_end = b2_df['seconds'].iloc[-1]
        early_b2_df = b2_df[b2_df['seconds'] <= (b2_start + 120)]
        late_b2_df = b2_df[b2_df['seconds'] >= (b2_end - 120)]
    
        bl.append(get_a_bout_calc(bl_df, 'tortuosity'))
        early_b2.append(get_a_bout_calc(early_b2_df, 'tortuosity'))
        late_b2.append(get_a_bout_calc(late_b2_df, 'tortuosity'))

    bl_avg = sum(bl) / len(bl)
    early_b2_avg = sum(early_b2) / len(early_b2)  
    late_b2_avg = sum(late_b2) / len(late_b2)  

    plt.scatter([1]*len(bl), bl, color=colors[0], alpha=0.5)
    plt.scatter([2]*len(early_b2), early_b2, color=colors[1], alpha=0.5)
    plt.scatter([3]*len(late_b2), late_b2, color=colors[2], alpha=0.5)

    # plt.plot([1,2],[b1_avg, b2_avg], color='white')
    plt.ylabel('tortuosity', fontsize=18, color='white')
    axs.set_xticks((1,2,3))
    axs.set_xticklabels(labels, fontsize=16, color='white', rotation=45)
    plt.xlim(0.5, 3.5)
    plt.tight_layout() 
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}', bbox_inches='tight')


def b2_baseline_traj(folder_path, savename, size=(3,5), cutoff=500, labels = ['baseline', 'first 2 min.', 'last 2 min.'], colors=['#c1ffc1', '#6f00ff'], palette='husl'):
    params_list = exp_parameters(folder_path)
    num_files = len(params_list)
    unique_colors = sns.color_palette(palette, num_files)
    fig, axs, markc = configure_bw_plot(size=(size[0], size[1]*2), xaxis=True, nrows=2, ncols=2)
    axs = np.array(axs).reshape(2, 2)
    for i, this_experiment in enumerate(params_list):
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        ypos = exp_df['ft_posy'] - yo
        color = unique_colors[i]
        if '500' in filename:
            b2_start = exp_df[(ypos > cutoff) & (exp_df['odor_on'] == False)].index[0]
        else:
            b2_start = exp_df[(ypos > 350) & (exp_df['odor_on'] == False)].index[0]
            
        b2_df = exp_df.loc[b2_start:]
        xo_bl = bl_df['ft_posx'].iloc[0]
        yo_bl = bl_df['ft_posy'].iloc[0]
        dx_bl = xo - xo_bl
        dy_bl = yo - yo_bl
        xo_b2 = b2_df['ft_posx'].iloc[0]
        xf_b2 = b2_df['ft_posx'].iloc[-1]
        yo_b2 = b2_df['ft_posy'].iloc[0]
        yf_b2 = b2_df['ft_posy'].iloc[-1]
        dx_b2 = xf_b2 - xo_b2
        dy_b2 = yf_b2 - yo_b2

        axs[0,0].plot(bl_df['ft_posx'] - xo, bl_df['ft_posy'] - yo, color=color, label='clean air', linewidth=2)
        axs[0,0].set_title(labels[0], fontsize=14, color=markc)
        axs[1,0].arrow(0, 0, dx_bl, dy_bl, head_width=15, linewidth=2, color=color, alpha=0.8)
        axs[0,1].plot(b2_df['ft_posx'] - xo, b2_df['ft_posy'] - yo_b2, color=color, label='clean air', linewidth=2)
        axs[0,1].set_title(labels[1], fontsize=14, color=markc)
        axs[1,1].arrow(0, 0, dx_b2, dy_b2, head_width=15, linewidth=2, color=color, alpha=0.8)

    xlims = [ax.get_xlim() for row in axs for ax in row]
    ylims = [ax.get_ylim() for row in axs for ax in row]
    xmin = min(x[0] for x in xlims)
    xmax = max(x[1] for x in xlims)
    ymin = min(y[0] for y in ylims)
    ymax = max(y[1] for y in ylims)
    x_range = xmax - xmin
    y_range = ymax - ymin
    max_range = max(x_range, y_range)
    xmid = (xmax + xmin) / 2
    ymid = (ymax + ymin) / 2
    new_xlim = (xmid - max_range / 2, xmid + max_range / 2)
    new_ylim = (ymid - max_range / 2, ymid + max_range / 2)

    for ax in axs[1]:
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
    for ax in axs.flat:
        ax.set_xlabel('x-position (mm)', fontsize=14)
        ax.set_ylabel('y-position (mm)', fontsize=14)
        ax.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
        ax.tick_params(axis='both', colors=markc)
        ax.set_aspect('equal', adjustable='datalim') 
        for pos in ['right', 'top']:
            ax.spines[pos].set_visible(False)
        plt.tight_layout()
        sns.despine(offset=10)
        for _, spine in ax.spines.items():
            spine.set_linewidth(2)
        for spine in ax.spines.values():
            spine.set_edgecolor(markc)

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}', bbox_inches='tight')

def b2_baseline_traj_individual(folder_path, savename_prefix, size=(3,5), cutoff=500, labels=['baseline', 'first 2 min.'], colors=['#c1ffc1', '#6f00ff']):
    params_list = exp_parameters(folder_path)

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment

        fig, axs, markc = configure_bw_plot(size=(size[0], size[1]*2), xaxis=True, nrows=2, ncols=2)
        axs = np.array(axs).reshape(2, 2)

        ypos = exp_df['ft_posy'] - yo
        if '500' in filename:
            b2_df = exp_df[(ypos >= cutoff)]
            b2_start = b2_df['seconds'].iloc[0]
        else:
            b2_start = exp_df[(ypos > 350) & (exp_df['odor_on'] == False)].index[0]
            b2_df = exp_df.loc[b2_start:]

        # Baseline movement
        xo_bl = bl_df['ft_posx'].iloc[0]
        yo_bl = bl_df['ft_posy'].iloc[0]
        dx_bl = xo - xo_bl
        dy_bl = yo - yo_bl

        # B2 movement
        xo_b2 = b2_df['ft_posx'].iloc[0]
        xf_b2 = b2_df['ft_posx'].iloc[-1]
        yo_b2 = b2_df['ft_posy'].iloc[0]
        yf_b2 = b2_df['ft_posy'].iloc[-1]
        dx_b2 = xf_b2 - xo_b2
        dy_b2 = yf_b2 - yo_b2

        axs[0,0].plot(bl_df['ft_posx'] - xo, bl_df['ft_posy'] - yo, color=colors[0], linewidth=2)
        axs[0,0].set_title(labels[0], fontsize=14, color=markc)
        axs[1,0].arrow(0, 0, dx_bl, dy_bl, head_width=15,linewidth=2, color=colors[0], alpha=0.8)

        axs[0,1].plot(b2_df['ft_posx'] - xo, b2_df['ft_posy'] - yo_b2, color=colors[1], linewidth=2)
        axs[0,1].set_title(labels[1], fontsize=14, color=markc)
        axs[1,1].arrow(0, 0, dx_b2, dy_b2, head_width=15,linewidth=2, color=colors[1], alpha=0.8)

        # Normalizing axis ranges
        xlims = [ax.get_xlim() for row in axs for ax in row]
        ylims = [ax.get_ylim() for row in axs for ax in row]
        xmin = min(x[0] for x in xlims)
        xmax = max(x[1] for x in xlims)
        ymin = min(y[0] for y in ylims)
        ymax = max(y[1] for y in ylims)
        x_range = xmax - xmin
        y_range = ymax - ymin
        max_range = max(x_range, y_range)
        xmid = (xmax + xmin) / 2
        ymid = (ymax + ymin) / 2
        new_xlim = (xmid - max_range / 2, xmid + max_range / 2)
        new_ylim = (ymid - max_range / 2, ymid + max_range / 2)

        for ax in axs[1]:
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
        for ax in axs.flat:
            ax.set_xlabel('x-position (mm)', fontsize=14)
            ax.set_ylabel('y-position (mm)', fontsize=14)
            ax.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
            ax.tick_params(axis='both', colors=markc)
            ax.set_aspect('equal', adjustable='datalim')
            for pos in ['right', 'top']:
                ax.spines[pos].set_visible(False)
            plt.tight_layout()
            sns.despine(offset=10)
            for _, spine in ax.spines.items():
                spine.set_linewidth(2)
            for spine in ax.spines.values():
                spine.set_edgecolor(markc)
        plt.show()
        # Save individual plot
        if not os.path.exists(f'{folder_path}/fig/'):
            os.makedirs(f'{folder_path}/fig/')
        clean_name = os.path.splitext(os.path.basename(filename))[0]
        fig.savefig(f'{folder_path}/fig/{savename_prefix}_{clean_name}.png', bbox_inches='tight')
        plt.close(fig)


def block_xpos_distribution(folder_path, savename, size=(6,5), cutoff=500, labels = ['LED off', 'LED on'], colors=['#c1ffc1', '#6f00ff']):
    # Right now hard coded for only 2 blocks
    # Need to time cutoff rather than ypos cutoff
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    xpos_dist_b1 = []
    xpos_dist_b2 = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        print(filename)
        ypos = exp_df['ft_posy'] - yo
        cutoff_condition = ypos >= cutoff
        cutoff_idx = cutoff_condition.idxmax()
        print(cutoff_idx)
        b1_df = exp_df[:cutoff_idx]
        print(b1_df['ft_posy'])
        b2_df = exp_df[cutoff_idx:]
        print(b2_df['ft_posy'])
        xpos_b1 = (b1_df['ft_posx'] - xo)
        xpos_b2 = (b2_df['ft_posx'] - xo)
        xpos_dist_b1.append(xpos_b1)
        xpos_dist_b2.append(xpos_b2)
    compiled_b1 = pd.concat(xpos_dist_b1, axis=0, ignore_index=True)
    compiled_b2 = pd.concat(xpos_dist_b2, axis=0, ignore_index=True)      
    sns.kdeplot(compiled_b1, label=labels[0], color=colors[0], fill=False, cut=0)
    sns.kdeplot(compiled_b2, label=labels[1], color=colors[1], fill=False, cut=0) 
    min_x, max_x = min(compiled_b1.min(), compiled_b2.min()), max(compiled_b1.max(), compiled_b2.max())
    axs.set_xlim([-300, 300])
    plt.axvline(x=25, color='lightgrey')
    plt.axvline(x=-25, color='lightgrey')
    plt.xlabel('x-position (mm)', fontsize=18, color='white')
    plt.ylabel('density', fontsize=18, color='white')
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_linewidth(2)
    axs.spines['bottom'].set_color('white')
    axs.spines['left'].set_color('white')
    plt.xticks([min_x, max_x], fontsize=14, color='white')
    plt.yticks([0, 0.03], fontsize=14, color='white')
    legend = plt.legend()  # Get the legend object
    legend.get_frame().set_facecolor('none')  # Set the facecolor to none (transparent)
    legend.get_frame().set_edgecolor('none')
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}', bbox_inches='tight')

def avg_entry_angle(folder, window):
    fig, axs = plt.subplots(subplot_kw={'projection': 'polar'})
    b1_means = []
    b2_means = []
    for filename in os.listdir(folder):
        if filename.endswith('.log'):
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)
            df = calculate_trav_dir(df)
            df_odor = df[df['odor_on']]
            first_on_index = df_odor.index[0]
            df = df.loc[first_on_index:]
            xo = df.iloc[0]['ft_posx']
            yo = df.iloc[0]['ft_posy']
            ypos = df['ft_posy'] - yo
            b1_df = df[(ypos >= 0) & (ypos < 500)]
            b2_df = df[ypos >= 500]
            d1, d_in1, d_out1 = inside_outside(b1_df)
            d2, d_in2, d_out2 = inside_outside(b2_df)
            fly_b1_means = []
            fly_b2_means = []
            for key, df in d_out1.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                    df = get_last_second(df)
                    b1_circmean = circmean_heading(df, fly_b1_means)
                    b1_x, b1_y = pol2cart(1, b1_circmean)
                    b1_means.append((b1_x, b1_y))
                fly_b1_mean = stats.circmean(fly_b1_means, low=-np.pi, high=np.pi, axis=None, nan_policy='omit')
                plt.polar([0, fly_b1_mean], [0, 1], color='black', alpha=0.3, solid_capstyle='round')
            for key, df in d_out2.items():
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

def disappearing_bias_minutes(folder_path):
    fig, axs, markc = configure_white_polar(size=(6, 6), grid=True)
    params_list = exp_parameters(folder_path)
    dur = None
    circ_means_all = []
    cmap = cm.get_cmap('Reds', 10)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        print(filename)
        ypos = df['ft_posy'] - yo
        odor_off_idx = df_odor.dropna().index[-1]
        disappeared_df = exp_df[:odor_off_idx]
        duration = (disappeared_df['seconds'].iloc[-1] - disappeared_df['seconds'].iloc[0])
        if dur is None or duration < dur:
            dur = duration
            dur = int(dur // 60)
        fly_means = []
        for minute in range(dur):
            start_time = disappeared_df['seconds'].iloc[0] + (minute * 60)
            end_time = disappeared_df['seconds'].iloc[0] + ((minute + 1) * 60)
            minute_df = disappeared_df.loc[(disappeared_df['seconds'] >= start_time) & (disappeared_df['seconds'] < end_time)]
            minute_circmean = (circmean_heading(minute_df, fly_means))
            color = cmap(minute / dur)
            # plot each fly's average heading for each minute
            axs.scatter(minute_circmean, minute, color=color, s=10, label=f'{minute} minute')
        # fly_mean = stats.circmean(fly_means, low=-np.pi, high=np.pi, axis=None, nan_policy='omit')
        # axs.scatter(flymean, minute, color='red', s=10, label=f'{minute} minute')
    axs.set_title('heading after plume disappears', va='bottom')
    axs.set_theta_zero_location('N')  
    axs.set_theta_direction(-1)  
    # axs.legend()
    plt.show()

def get_a_bout_calc(df, data_type):
    def path_length(x, y):
        n = len(x)
        lv = [np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2) for i in range (1,n)]
        lt = np.cumsum(lv)
        L = sum(lv)
        return lt, L

    # Based on the string input 'data_type', return a specified calculation
    if data_type == 'x_distance_from_plume':
        return df['ft_posx'].max() - df['ft_posx'].min()
    if data_type == 'duration':
        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        return duration.total_seconds()
    if data_type == 'avg_speed':
        return df['speed'].mean()
    if data_type == 'path_length':
        df.reset_index(drop=True, inplace=True)
        lt, pl = path_length(df.ft_posx, df.ft_posy)
        return pl
    if data_type == 'tortuosity':
        df.reset_index(drop=True, inplace=True)
        lt, pl = path_length(df.ft_posx, df.ft_posy)
        x_i = df['ft_posx'].iloc[0]
        x_f = df['ft_posx'].iloc[-1]
        y_i = df['ft_posy'].iloc[0]
        y_f = df['ft_posy'].iloc[-1]
        disp = ((x_f - x_i)**2 + (y_f - y_i)**2)**0.5
        tort = pl / disp
        return tort
    if data_type == 'furthest_distance_from_plume':
        x1, y1 = df['ft_posx'].iloc[0], df['ft_posy'].iloc[0]
        x2, y2 = df['ft_posx'].iloc[-1], df['ft_posy'].iloc[-1]
        distances = np.abs((y2 - y1) * df['ft_posx'] - (x2 - x1) * df['ft_posy'] + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        dist = distances.max()
        return dist
    return None


def store_bout_data(strip_dict, data_type, strip_status='in', hist=True):
    # Based on the string input 'data_type' and the location of the fly
    # relative to a plume, return bout calcs in a list
    data = []
    for key, df in strip_dict.items():
        value = get_a_bout_calc(df, data_type)
        # If this data is to be used in a histogram, flip the sign of the 'in' data
        # so that an abutted (or mirrored) histogram can be generated
        if hist and strip_status == 'in':
            value = -value
        # Otherwise, like for some other type of plot, just append normally
        data.append(value)
    return data


def create_bout_df(folder, data_type, plot_type):
    # Create a dataframe for every logfile in the folder
    for filename in os.listdir(folder):
        if filename.endswith('.log'):
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)

            # Generate in/out dictionaries based on plot type
            if plot_type == 'odorless':
                d_total, d_in, d_out = light_on_off(df)
            elif plot_type == 'odor':
                d_total, d_in, d_out = odor_on_off(df)
            # This is just based on the nature of the experiment. Basically remove the first
            # 'out' key because that's the recorded baseline. Remove the last 'out' key
            # to eliminate "exits". Remove the first 'in' key because the animal has
            # not directly navigated into the plume when the stimulus first comes on
            d_out.pop(list(d_out.keys())[-1], None)  # Remove last key
            d_out.pop(list(d_out.keys())[0], None)  # Remove first key
            d_in.pop(2, None)  # Remove first key
            # Assign lists generated by store_bout_data
            in_data = store_bout_data(d_in, data_type, 'in', hist=True)
            out_data = store_bout_data(d_out, data_type, 'out', hist=True)
    data = {
        'data': in_data + out_data,
        'condition': ['in'] * len(in_data) + ['out'] * len(out_data)
    }
    return pd.DataFrame(data)


def plot_histograms(folder_path, boutdf, plot_variable, group_variable, group_values, group_colors, title, x_label, y_label, x_limits=None):
    figure_folder = f'{folder_path}/figure'
    # Create a figure and add a subplot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    offset = 0.5  # Offset the histograms so that values at 0 do not overlap
    # Assign the number of bins based on the range of the data
    vmin, vmax = boutdf[plot_variable].min(), boutdf[plot_variable].max()
    nbins = int(vmax - vmin)
    # Plotting histograms for each group
    # Iterate through the conditions
    for color, condition in zip(group_colors, group_values):
        df = boutdf[boutdf[group_variable] == condition]
        if df.empty:
            continue  # Skip if no data for this condition
        hist_vals = df[plot_variable].values
        # Apply an offset
        if condition == 'out':
            hist_vals += offset
        # Apply a negative offset
        elif condition == 'in':
            hist_vals -= offset
        bins = np.linspace(vmin, vmax, nbins)
        # Use the adjusted values for plotting
        ax.hist(hist_vals, bins, facecolor=color, edgecolor='none', rwidth=0.95)
    # Customize the plot
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=18)
    if x_limits:
        ax.set_xlim(x_limits)
    # Further customization
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    sns.set_theme(style='whitegrid')
    ax.tick_params(which='both', axis='both', labelsize=16, length=3, width=2, color='black', direction='out', left=True, bottom=True)
    for pos in ['right', 'top']:
        ax.spines[pos].set_visible(False)
    sns.despine(offset=10)
    for _, spine in ax.spines.items():
        spine.set_linewidth(2)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    plt.tight_layout()
    # Save the plot
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    fig.savefig(os.path.join(figure_folder, title + '.pdf'))


def plot_scatter(folder_path, boutdf, plot_variable, group_variable, group_values, group_colors, title, x_label, y_label, x_limits=None):
    figure_folder = f'{folder_path}/Figure'

    fig, ax1 = plt.subplots(figsize=(8, 3))
    sns.stripplot(data=boutdf, x=plot_variable, ax=ax1,
                      y=group_variable, edgecolor='none', dodge=False,
                      alpha=0.5, palette=group_colors, linewidth=0)
    ax1.grid(False)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    if x_limits:
        ax1.set_xlim(x_limits)
    ax1.set_yticks(range(1, len(boutdf[group_variable].unique()) + 1))
    ax1.set_yticklabels([str(i) if i % 5 == 0 else '' for i in range(1, len(boutdf[group_variable].unique()) + 1)])

    # Further customization for a cleaner look
    for ax in fig.axes:
        ax.tick_params(which='both', axis='both', labelsize=18, length=3, width=2, color='black', direction='out', left=False, bottom=False)
        for pos in ['right', 'top']:
            ax.spines[pos].set_visible(False)
    plt.tight_layout()
    sns.despine(offset=10)
    for _, spine in ax1.spines.items():
        spine.set_linewidth(2)
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')

    # Save the plot
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    fig.savefig(os.path.join(figure_folder, title + '.pdf'))

def plot_circmean_heading(df, means_list):
    x = df.ft_posx.to_numpy()
    x = x - x[-1]
    if np.abs(x[0] - x[-1]):
        circmean_value = stats.circmean(df.ft_heading, low=-np.pi, high=np.pi, axis=None, nan_policy='omit')
        means_list.append(circmean_value)
        return circmean_value