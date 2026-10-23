import pandas as pd
import cv2
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from PIL import Image
from scipy import interpolate
from scipy import stats
from scipy.stats import linregress, sem, circmean, pearsonr, spearmanr, norm
from scipy.stats import ttest_rel, wilcoxon, ttest_ind, kruskal, shapiro, normaltest, mannwhitneyu, ks_2samp
from sklearn.linear_model import LinearRegression
import os, re, datetime
import seaborn as sns
import re
import scikit_posthocs as sp


def open_log(logfile):  # Generate a dataframe from the contents of a log file.
    df = pd.read_table(logfile, delimiter='[,]', engine='python', encoding='latin-1')
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


def save_exp_logs(folder_path): 
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        base_name = os.path.basename(filename)          
        short_name = base_name.split('_')[0]       
        if not os.path.exists(f'{folder_path}/logs/'):
            os.makedirs(f'{folder_path}/logs/')
        exp_df.to_csv(f'{folder_path}/logs/{short_name}.log', index=False)


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
        if len(headings) == 1:
            print('one return')
        headings_rad = np.array(headings)  # assumed to be in radians already
        vector = np.mean(np.exp(1j * headings_rad))
        mean_heading = np.angle(vector)
        strength = np.abs(vector)  # vector length (between 0 and 1)
        return headings, mean_heading, strength
    else:
        return None, None

def mean_crosswind_component(angles_rad, min_n=3):
    """Mean of |sin(heading)| — 0 = upwind/downwind, 1 = pure crosswind.
    Assumes ft_heading = 0 is upwind, in radians."""
    a = np.asarray(angles_rad, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < min_n:
        return 0
    return float(np.mean(np.abs(np.sin(a))))

def mean_upwind_component(angles_rad, min_n=3):
    a = np.asarray(angles_rad, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < min_n:
        return 0
    return float(np.mean(np.abs(np.cos(a))))

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

def finite_minmax(*arrays):
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    return (vals.min(), vals.max()) if vals.size else (-1.0, 1.0)

def mean_resultant_vector_length(angles):
    # angles should be in radians
    n = len(angles)
    if n == 0:
        return np.nan  # Or handle as you see fit
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = np.sqrt(C**2 + S**2) / n
    return R

def get_entry_df(do_df, di_df, exp_df, window=0.5, first_entry=False):
    """
    Return a dataframe for entry trajectory:
    - Normal case: last `window` seconds of do + first `window` seconds of di
    - First entry: last `window` seconds from exp_df before di starts + first `window` seconds of di
    """
    if di_df is None:
        return None
    
    di_start = di_df['seconds'].iloc[0]
    
    if first_entry:
        # Use exp_df for the half-second before first di
        pre_half = exp_df[(exp_df['seconds'] >= di_start - window) &
                          (exp_df['seconds'] < di_start)]
        di_half = di_df[di_df['seconds'] <= di_start + window]
        entry_df = pd.concat([pre_half, di_half])
        return entry_df if len(entry_df) > 1 else None
    
    if do_df is None:
        return None
    
    # Normal case: do + di
    do_end = do_df['seconds'].iloc[-1]
    do_half = do_df[do_df['seconds'] >= do_end - window]
    di_half = di_df[di_df['seconds'] <= di_start + window]
    
    entry_df = pd.concat([do_half, di_half])
    return entry_df if len(entry_df) > 1 else None

def get_last_second(df):
    # Find the first occurrence of the 'seconds' column, d/t duplicates
    seconds_idx = next(i for i, col in enumerate(df.columns) if col == 'seconds')
    seconds = df.iloc[:, seconds_idx]
    last_time = seconds.iloc[-1]
    return df[seconds >= (last_time - 0.5)]

def inside_outside_xpos(df, edge=25):
    di = {}
    do = {}
    start_x = df['ft_posx'].iloc[0]

    # Split the dataframe into contiguous bouts whenever 'instrip' changes
    d = dict([*df.groupby(df['instrip'].ne(df['instrip'].shift()).cumsum())])

    for bout, bout_df in d.items():
        # Compute difference from starting x position
        bout_df = bout_df.copy()  # avoid SettingWithCopy warnings
        bout_df['instrip'] = (bout_df['ft_posx'].sub(start_x).abs() < edge)

        if bout_df['instrip'].any():
            di[bout] = bout_df
        else:
            do[bout] = bout_df

    return d, di, do

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

def inside_outside_odor(df):
    di = {}
    do = {}
    d = dict([*df.groupby(df['odor_on'].ne(df['odor_on'].shift()).cumsum())])
    for bout in d:
        if d[bout].odor_on.any():
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

def inside_outside_img(df):
    di = {}
    dout = {}

    instrip = df["instrip"].to_numpy()

    # run IDs in numpy
    changes = np.empty(len(instrip), dtype=bool)
    changes[0] = True
    changes[1:] = instrip[1:] != instrip[:-1]
    bout_ids = np.cumsum(changes) - 1  # start at 0

    # group using bout_ids (positional)
    groups = dict([*df.groupby(bout_ids)])

    for bout, g in groups.items():
        # positions of rows in this group (NOT index labels)
        pos = df.index.get_indexer(g.index)
        # get_indexer returns -1 if something can't be found (shouldn't happen)
        if (pos < 0).any():
            raise ValueError("Index alignment issue: some group indices not found in df.index")

        if instrip[pos].any():
            di[bout] = g
        else:
            dout[bout] = g

    return groups, di, dout


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
        if filename.endswith('.log') and not filename.startswith('._'):
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)
            if (df['mfc2_stpt'] == 0).all():
                df_odor = df.where(df.mfc3_stpt > 0)
                if 'mch' in filename or 'mho' in filename or 'ben' in filename:
                    plume_color = '#d473d4'
                elif 'oct' in filename:
                    plume_color = '#7ed4e6'
                else:
                    plume_color = None
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
                params_list.append([figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color])
            # elif df_oct is not None and df_acv is not None:
            #     print('multiple odors')
            #     first_on_index = df[df['instrip']].index[0] 
            #     exp_df = df.loc[first_on_index:]
            #     xo = exp_df.iloc[0]['ft_posx']
            #     yo = exp_df.iloc[0]['ft_posy']
            #     params_list.append([figure_folder, filename, df, df_oct, df_acv, df_light, exp_df, xo, yo, plume_color])
            else:
                # Create an index for the first instance of odor on (exp start), and filter the df to start at this index
                if 'instrip' in df.columns:
                    first_on_index = df[df['instrip']].index[0]
                else: 
                    first_on_index = df[df['odor_on']].index[0]
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
            print(filename)
            logfile = os.path.join(folder, filename)
            df = open_log(logfile)
            mfc2_vals = df['mfc2_stpt'].unique().tolist()
            mfc3_vals = df['mfc3_stpt'].unique().tolist()
            df_light = df.where(df.led1_stpt == 0.0)
            if len(mfc2_vals) > 1 and len(mfc3_vals) > 1:
                df_odor1 = df.where(df.mfc2_stpt > 0)
                df_odor2 = df.where(df.mfc3_stpt > 0)
                if 'mch' in filename or 'ben' in filename or 'mho' in filename:
                    plume_colors = ['#d473d4', '#7ed4e6']
                else:
                    plume_colors = ['#fbceb1', '#7ed4e6']
                first_on_index = df[df['instrip']].index[0]
                bl_df = df.loc[:first_on_index]
                exp_df = df.loc[first_on_index:]
                xo = exp_df.iloc[0]['ft_posx']
                yo = exp_df.iloc[0]['ft_posy']
                params_list.append([figure_folder, filename, df, df_odor1, df_odor2, df_light, bl_df, exp_df, xo, yo, plume_colors])
            else:
                print('single odor')
                df_odor = df.where((df.mfc2_stpt > 0) | (df.mfc3_stpt > 0))
                if 'mch' in filename:
                    plume_color = '#d473d4'
                else:
                    plume_color = '#7ed4e6'
                first_on_index = df[df['instrip']].index[0]
                bl_df = df.loc[:first_on_index]
                exp_df = df.loc[first_on_index:]
                xo = exp_df.iloc[0]['ft_posx']
                yo = exp_df.iloc[0]['ft_posy']
                params_list.append([figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color])

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

def configure_white_plot(size=(4, 6), xaxis=False, nrows=1, ncols=1):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)
    markc = 'black'
    # Flatten axs if it's a 2D array (in case of multiple rows/cols)
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
        for ax in axs:
            ax.set_facecolor('white')  # Plot area background
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(1)
            ax.tick_params(axis='x', colors='black', width=1)
            ax.tick_params(axis='y', colors='black', width=1)
            if xaxis:
                ax.xaxis.label.set_color('black')
                ax.spines['bottom'].set_visible(xaxis)
                ax.spines['bottom'].set_color('black')
                ax.spines['bottom'].set_linewidth(1)
            if not xaxis:
                ax.tick_params(axis='x', which='both', length=0)
                ax.spines['bottom'].set_visible(False)
    else:
        axs = axs
        axs.set_facecolor('white')  # Plot area background
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['left'].set_color('black')
        axs.spines['left'].set_linewidth(1)
        axs.tick_params(axis='x', colors='black', width=1)
        axs.tick_params(axis='y', colors='black', width=1)
        if xaxis:
            axs.xaxis.label.set_color('black')
            axs.spines['bottom'].set_visible(xaxis)
            axs.spines['bottom'].set_color('black')
            axs.spines['bottom'].set_linewidth(1)
        if not xaxis:
            axs.tick_params(axis='x', which='both', length=0)
            axs.spines['bottom'].set_visible(False)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    fig.patch.set_facecolor('white')  # Figure background
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
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
            ax.spines['left'].set_color('white')
            ax.spines['left'].set_linewidth(1)
            ax.tick_params(axis='x', colors='white', width=1)
            ax.tick_params(axis='y', colors='white', width=1)
            if xaxis:
                ax.xaxis.label.set_color('white')
                ax.spines['bottom'].set_visible(xaxis)
                ax.spines['bottom'].set_color('white')
                ax.spines['bottom'].set_linewidth(1)
            if not xaxis:
                ax.tick_params(axis='x', which='both', length=0)
                ax.spines['bottom'].set_visible(False)
                
    else:
        axs = axs
        axs.set_facecolor('black')  # Plot area background
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['left'].set_color('white')
        axs.spines['left'].set_linewidth(1)
        axs.tick_params(axis='x', colors='white', width=1)
        axs.tick_params(axis='y', colors='white', width=1)
        if xaxis:
            axs.xaxis.label.set_color('white')
            axs.spines['bottom'].set_visible(xaxis)
            axs.spines['bottom'].set_color('white')
            axs.spines['bottom'].set_linewidth(1)
        if not xaxis:
            axs.tick_params(axis='x', which='both', length=0)
            axs.spines['bottom'].set_visible(False)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    fig.patch.set_facecolor('black')  # Figure background
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    return fig, axs, markc

def style_ax(ax, markc, title, xtick_labels, xticks, ylabel, xlim):
    ax.tick_params(which='both', axis='both', labelsize=12, length=3, width=0.5,
                   color=markc, direction='out', left=True, bottom=True)
    ax.tick_params(axis='both', colors=markc)

    for pos in ['right', 'top']:
        ax.spines[pos].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor(markc)

    ax.set_ylabel(ylabel, fontsize=14, color=markc)
    for label in ax.get_yticklabels():
        label.set_color(markc)
        label.set_fontsize(14)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=14, color=markc, rotation=45)
    ax.set_xlim(xlim)
    ax.set_title(title, fontsize=14, color=markc)   

 
def linregress_r2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan, np.nan  # not enough points
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return m, b, r2

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
        if disp < 0.1:
            tort = 1
        return tort
        

    if data_type == 'furthest_distance_from_plume':
        x1, y1 = df['ft_posx'].iloc[0], df['ft_posy'].iloc[0]
        x2, y2 = df['ft_posx'].iloc[-1], df['ft_posy'].iloc[-1]
        distances = np.abs((y2 - y1) * df['ft_posx'] - (x2 - x1) * df['ft_posy'] + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        dist = distances.max()
        return dist
    return None


def trajectory_plotter(folder_path, strip_width, strip_length, plume_start, xlim, ylim, led, hlines=[], angle=0, select_file=None, plot_type='odor', baseline=True, save=False):
    params_list = exp_parameters(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bdf51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        # Create a figure and set the font to Arial
        fig, axs, markc = configure_white_plot(size=(10,10),xaxis=True)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        # In an odor plume, plot the trajectory when the animal is in the odor
        if plot_type == 'odor':
            if baseline:
                plt.plot(df['ft_posx'] - xo, df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            else:
                plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='#6acff6', label='clean air', linewidth=2)
            plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='red', label='odor only', linewidth=2)
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
            if np.abs(angle) > 0:
                plume_adj = (strip_width/2) / np.sqrt(2)
                mask = exp_df['instrip'].fillna(False)
                height = (exp_df.loc[mask, 'ft_posx'].max() - exp_df.loc[mask, 'ft_posx'].min()) * np.sqrt(2) + 30
                x1 = exp_df.iloc[0]['ft_posx'] - xo - plume_adj
                y1 = exp_df.iloc[0]['ft_posy'] - yo + plume_adj
                plt.gca().add_patch(patches.Rectangle((x1, y1),strip_width,height,angle=angle,facecolor=plume_color,alpha=0.5))
            else:
                plt.gca().add_patch(patches.Rectangle((-strip_width / 2, plume_start), strip_width, strip_length, facecolor='grey', alpha=0.5))
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
        if angle == 0:
            plt.xlim(xlim)
            plt.ylim(ylim)
        else:
            axs.set_aspect('equal', adjustable='datalim') 
        # plt.legend()
        plt.title(filename, fontsize=14)
        axs.set_xlabel('x-position (mm)', fontsize=14, color=markc)
        axs.set_ylabel('y-position (mm)', fontsize=14, color=markc)
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
        # Save and show the plot
        if save:
            plt.savefig(os.path.join(figure_folder, savename))
        else:
            plt.show()


def plot_gaussian_alpha_plume(center_x, start_y, strip_width, strip_length, plume_color, ax, max_alpha=1.0):
    # Parameters
    resolution = 0.1  # mm per step along width
    half_width = strip_width / 2
    sigma = 25 / np.sqrt(2 * np.log(2))  # Gaussian sigma from FWHM = 25 mm
    x_vals = np.arange(-half_width, half_width + resolution, resolution)
    norm_alpha = norm(loc=0, scale=sigma)
    
    # Plot thin vertical strips across the width
    for x in x_vals:
        alpha = norm_alpha.pdf(x) / norm_alpha.pdf(0) * max_alpha  # Normalize to max_alpha
        rect = patches.Rectangle(
            (center_x + x, start_y),
            resolution,
            strip_length,
            facecolor=plume_color,
            edgecolor=None,
            alpha=alpha,
            linewidth=0
        )
        ax.add_patch(rect)

def plot_gaussian_plume_vector(center_x, start_y, strip_width, strip_length, ax,
                               fwhm=25, resolution=0.1,
                               cmap='Greys_r', min_val=0.2, max_val=0.7):
    half_width = strip_width / 2
    sigma = fwhm / np.sqrt(2 * np.log(2))
    x_vals = np.arange(-half_width, half_width + resolution, resolution)
    norm_alpha = norm(loc=0, scale=sigma)

    peak = norm_alpha.pdf(0)
    intensities = norm_alpha.pdf(x_vals) / peak  # values 0..1

    # Rescale intensities into [min_val, max_val]
    scaled_t = min_val + intensities * (max_val - min_val)

    colormap = cm.get_cmap(cmap)

    for x, t in zip(x_vals, scaled_t):
        face = colormap(t)
        face = (face[0], face[1], face[2], 1.0)
        rect = patches.Rectangle((center_x + x, start_y),
                                 resolution, strip_length,
                                 facecolor=face, edgecolor='none')
        rect.set_antialiased(False)
        ax.add_patch(rect)


def gaussian_plotter(folder_path, strip_width, strip_length, plume_start, xlim, ylim, select_file):
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_white_plot(size=(10,10),xaxis=True)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='#6acff6', label='clean air', linewidth=2)
        plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='red', label='odor only', linewidth=2)

        plot_gaussian_plume_vector(
            center_x=0,
            start_y=plume_start,
            strip_width=strip_width,
            strip_length=strip_length,
            ax=plt.gca(),
            fwhm=25, 
            resolution=0.1, 
            cmap='Greys',
            min_val=0.0,
            max_val=0.4
        )

        savename = filename + 'gaussian_trajectory.pdf'

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(filename, fontsize=14, color=markc)
        axs.set_xlabel('x-position (mm)', fontsize=14, color=markc)
        axs.set_ylabel('y-position (mm)', fontsize=14, color=markc)
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
        plt.savefig(os.path.join(figure_folder, savename))
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
        # print(bl_df['seconds'].iloc[-1]-bl_df['seconds'].iloc[0])
        # In an odor plume, plot the trajectory when the animal is in the odor
        if plot_type == 'odor':
            plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
            plt.gca().add_patch(patches.Rectangle((-strip_width / 2, plume_start), strip_width, strip_length, facecolor=plume_color, alpha=0.5))
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

def paired_plotter(folder_path, led='none', strip_width=50, strip_length=1000, plume_start=0):
    params_list = exp_parameters(folder_path)
    led_map = {'none': 'grey', 'red': '#ff355e', 'green': '#0bda51'}
    ledc = led_map.get(led.lower(), 'grey')

    fly_dict = {}
    for exp in params_list:
        figure_folder, filename, *rest = exp
        m = re.search(r'(fly\d+)', filename, flags=re.IGNORECASE)
        if not m:
            continue
        fly = m.group(1)
        cond = 'exp' if 'light' in filename.lower() else 'ctrl' 
        if cond is None:
            continue
        fly_dict.setdefault(fly, {})[cond] = exp

    for fly, conds in fly_dict.items():
        if 'exp' not in conds or 'ctrl' not in conds:
            print(f"[jumping_paired_plotter] {fly}: missing 'exp' or 'ctrl' → skipping.")
            continue

        exp_data = conds['exp']
        ctrl_data = conds['ctrl']
        (_, exp_filename, _, df_odor_exp, df_light_exp, _, exp_df, xo_exp, yo_exp, plume_color_exp) = exp_data
        (_, ctrl_filename, _, df_odor_ctrl, df_light_ctrl, _, ctrl_df, xo_ctrl, yo_ctrl, plume_color_ctrl) = ctrl_data

        fig, axs, markc = configure_bw_plot(size=(12, 6), nrows=1, ncols=2, xaxis=True)
        if hasattr(axs, 'flatten'):
            ax_exp, ax_ctrl = axs.flatten()
        else:  
            ax_exp, ax_ctrl = axs[0], axs[1]

        def plot_traj(ax, df_main, df_odor, df_light, xo, yo, plume_color, label_prefix):
            d, d_in, _ = inside_outside(df_main)
            ax.add_patch(patches.Rectangle((-strip_width / 2, plume_start), strip_width, strip_length, facecolor=plume_color, alpha=0.5))
            ax.plot(df_main['ft_posx'] - xo, df_main['ft_posy'] - yo, color='grey', linewidth=1)
            ax.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', linewidth=1)
            ax.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label=f'{label_prefix} light', linewidth=1)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(label_prefix)
            ax.tick_params(labelsize=8)

        plot_traj(ax_exp, exp_df, df_odor_exp, df_light_exp, xo_exp, yo_exp, plume_color_exp, 'exp')
        plot_traj(ax_ctrl, ctrl_df, df_odor_ctrl, df_light_ctrl, xo_ctrl, yo_ctrl, plume_color_ctrl,'ctrl')

        for ax in (ax_exp, ax_ctrl):
            ax.set_xlim(-500,500)
            ax.set_ylim(-100,1000)

        fig.suptitle(fly, fontsize=14, color=markc)
        handles, labels = ax_exp.get_legend_handles_labels()
        plt.show()
        if not os.path.exists(f'{folder_path}/paired/'):
            os.makedirs(f'{folder_path}/paired/')
        fig.savefig(f'{folder_path}/paired/{fly}.pdf', bbox_inches='tight')


def jumping_plotter_bw(folder_path, led, stripwidth=50, select_file=None, save=False): #
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
        for key, df in d_in.items():
            if key == 1:
                x = -(stripwidth/2)
            else:
                x_entry = df['ft_posx'].iloc[:5].values
                entry_dx = x_entry[-1] - x_entry[0]
                xi = df['ft_posx'].iloc[0] - xo
                x = xi - stripwidth if entry_dx < 0 else xi 
            y = df['ft_posy'].min() - yo
            height = (df['ft_posy'].max() - df['ft_posy'].min())
            plume = patches.Rectangle((x, y),stripwidth,height,facecolor=plume_color, alpha=0.5)
            axs.add_patch(plume)
        plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=1)
        plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=1)
        # if 'ctrl' not in filename:
        #     plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=1)
        axs.set_aspect('equal', adjustable='datalim') 
        savename = filename + '_odor_trajectory_bw'
        plt.title(filename, fontsize=14, color=markc)
        plt.show()
        if not os.path.exists(f'{folder_path}/traj/'):
            os.makedirs(f'{folder_path}/traj/')
        fig.savefig(f'{folder_path}/traj/{savename}.pdf', bbox_inches='tight')


def jumping_paired_plotter(folder_path, led='none', stripwidth=50):

    params_list = exp_parameters(folder_path)

    led_map = {'none': 'grey', 'red': '#ff355e', 'green': '#0bda51'}
    ledc = led_map.get(led.lower(), 'grey')

    # ----------  Pair files by Fly tag  --------------------------------------
    fly_dict = {}
    for exp in params_list:
        figure_folder, filename, *rest = exp

        # Extract Fly number (expects e.g. "Fly7")
        m = re.search(r'(Fly\d+)', filename, flags=re.IGNORECASE)
        if not m:
            continue
        fly = m.group(1)

        # Detect condition
        cond = 'exp' if 'exp' in filename.lower() else 'ctrl' if 'ctrl' in filename.lower() else None
        if cond is None:
            continue

        fly_dict.setdefault(fly, {})[cond] = exp

    # ----------  Generate paired plots  --------------------------------------
    for fly, conds in fly_dict.items():
        if 'exp' not in conds or 'ctrl' not in conds:
            print(f"[jumping_paired_plotter] {fly}: missing 'exp' or 'ctrl' → skipping.")
            continue

        exp_data = conds['exp']
        ctrl_data = conds['ctrl']

        # Unpack tuple structure produced by exp_parameters -------------------
        (_, exp_filename, _, df_odor_exp, _, _, exp_df, xo_exp, yo_exp, plume_color_exp) = exp_data
        (_, ctrl_filename, _, df_odor_ctrl, _, _, ctrl_df, xo_ctrl, yo_ctrl, plume_color_ctrl) = ctrl_data

        # ----------  Create figure with configure_white_plot  -----------------
        fig, axs, markc = configure_white_plot(size=(12, 6), nrows=1, ncols=2, xaxis=True)
        # `axs` is a 1×2 array‑like → flatten to ensure consistent indexing
        if hasattr(axs, 'flatten'):
            ax_exp, ax_ctrl = axs.flatten()
        else:  # already a list/tuple
            ax_exp, ax_ctrl = axs[0], axs[1]

        def plot_traj(ax, df_main, df_odor, xo, yo, plume_color, label_prefix):
            d, d_in, _ = inside_outside(df_main)

            # Shade plume rectangles
            for key, din in d_in.items():
                if key == 1:
                    x_rect = -(stripwidth / 2)
                else:
                    x_entry = din['ft_posx'].iloc[:5].values
                    entry_dx = x_entry[-1] - x_entry[0]
                    xi = din['ft_posx'].iloc[0] - xo
                    x_rect = xi - stripwidth if entry_dx < 0 else xi
                y_rect = din['ft_posy'].min() - yo
                height_rect = din['ft_posy'].max() - din['ft_posy'].min()
                ax.add_patch(patches.Rectangle((x_rect, y_rect), stripwidth, height_rect, facecolor=plume_color, alpha=0.5))

            # Main trajectory & odor segment
            ax.plot(df_main['ft_posx'] - xo, df_main['ft_posy'] - yo, color='grey', linewidth=1)
            ax.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', linewidth=1)
            # Light segment if re‑enabled
            # ax.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label=f'{label_prefix} light', linewidth=1)

            ax.set_aspect('equal', adjustable='box')
            ax.set_title(label_prefix)
            ax.tick_params(labelsize=8)

        plot_traj(ax_exp, exp_df, df_odor_exp, xo_exp, yo_exp, plume_color_exp, 'exp')
        plot_traj(ax_ctrl, ctrl_df, df_odor_ctrl, xo_ctrl, yo_ctrl, plume_color_ctrl,'ctrl')

        # Ensure consistent limits after all plotting
        ax_exp.relim()
        ax_ctrl.relim()
        ax_exp.autoscale_view()
        ax_ctrl.autoscale_view()

        # Get combined limits
        xmins = [ax.get_xlim()[0] for ax in (ax_exp, ax_ctrl)]
        xmaxs = [ax.get_xlim()[1] for ax in (ax_exp, ax_ctrl)]
        ymins = [ax.get_ylim()[0] for ax in (ax_exp, ax_ctrl)]
        ymaxs = [ax.get_ylim()[1] for ax in (ax_exp, ax_ctrl)]

        xmin, xmax = min(xmins), max(xmaxs)
        ymin, ymax = min(ymins), max(ymaxs)

        for ax in (ax_exp, ax_ctrl):
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        fig.suptitle(f'{fly}: exp vs ctrl', fontsize=14, color=markc)
        handles, labels = ax_exp.get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc='upper right')
        plt.show()
        if not os.path.exists(f'{folder_path}/paired/'):
            os.makedirs(f'{folder_path}/paired/')
        fig.savefig(f'{folder_path}/paired/{fly}.pdf', bbox_inches='tight')

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

def dir_training_plotter(folder_path, led, select_file=None):
    params_list = exp_parameters_multiodor(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        if len(this_experiment) == 10:
            multi = False
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        elif len(this_experiment) == 11:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, bl_df, exp_df, xo, yo, plume_colors = this_experiment
            multi = True
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_white_plot(size=(10,10), xaxis=True)
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        plume2_df = exp_df[exp_df['exp_phase']==3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        plume_adj = 25 / np.sqrt(2)
        width=50
        mask1 = plume1_df['instrip'].fillna(False)
        height1 = (plume1_df.loc[mask1, 'ft_posx'].max() - plume1_df.loc[mask1, 'ft_posx'].min()) * np.sqrt(2) + 30
        mask2 = plume2_df['instrip'].fillna(False)
        height2 = (plume2_df.loc[mask2, 'ft_posx'].max() - plume2_df.loc[mask2, 'ft_posx'].min()) * np.sqrt(2) + 30
        if multi:
            plume_color1 = plume_colors[1]
            plume_color2 = plume_colors[0]
        else:
            plume_color1 = plume_color
            plume_color2 = plume_color
        
        if 'ctrl' in filename:
            if 'LR' in filename or 'LL' in filename:
                x1 = -plume_adj
                y1 = -plume_adj
                angle1 = 45
                angle2 = 0
            elif 'RR' in filename or 'RL' in filename:
                x1 = -plume_adj
                y1 = plume_adj
                angle1 = -45
                angle2 = 0 
        else:
            if 'LR' in filename:
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
            elif 'RL' in filename:
                angle1 = -45
                angle2 = 45
                x1 = plume1_df.iloc[0]['ft_posx'] - xo - plume_adj
                y1 = plume1_df.iloc[0]['ft_posy'] - yo + plume_adj
                x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
                y2 = plume2_df.iloc[0]['ft_posy'] - yo - plume_adj
            elif 'LL' in filename:
                angle1 = 45
                angle2 = 45
                x1 = plume1_df.iloc[0]['ft_posx'] - xo - plume_adj
                y1 = plume1_df.iloc[0]['ft_posy'] - yo - plume_adj
                x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
                y2 = plume2_df.iloc[0]['ft_posy'] - yo - plume_adj
        
        if 'ctrl' not in filename:
            plume1 = patches.Rectangle((x1, y1),width,height1,angle=angle1,facecolor=plume_color1,alpha=0.5)
            axs.add_patch(plume1)
            plume2 = patches.Rectangle((x2, y2),width,height2,angle=angle2,facecolor=plume_color2,alpha=0.5)
            axs.add_patch(plume2)
        else:
            plume1 = patches.Rectangle((x1, y1),width,300,angle=angle1,facecolor=plume_color1,alpha=0.5)
            axs.add_patch(plume1)
        
        plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        if multi:
            plt.plot(df_odor1['ft_posx'] - xo, df_odor1['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
            plt.plot(df_odor2['ft_posx'] - xo, df_odor2['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)

        else:
            plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        if 'ctrl' not in filename:
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
        
        savename = filename + '_odor_trajectory.pdf'
        plt.title(filename, fontsize=14, color=markc)
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
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
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, savename))
        # plt.show()



def dir_training_plotter_paired(folder_path, led, select_file=None):
    params_list = exp_parameters(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    grouped_experiments = defaultdict(list)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if select_file and filename != select_file:
            continue
        match = re.search(r'[Ff]ly\d+', filename)
        if match:
            fly_id = match.group()
        else:
            continue  # skip files without Fly ID
        grouped_experiments[fly_id].append(this_experiment)

    for fly_id, experiments in grouped_experiments.items():
        if len(experiments) < 2:
            continue  # only plot if there are at least 2 experiments

        fig, axs, markc = configure_bw_plot(size=(16,8), xaxis=True, ncols=2)
        axs = axs.flatten()

        # --- Global plume2 height across experiments ---
        plume2_heights = []
        for this_experiment in experiments:
            _, filename, _, df_odor, df_light, _, exp_df, xo, yo, plume_color = this_experiment
            plume2_df = exp_df[exp_df['exp_phase'] == 3]
            plume2_start = plume2_df[plume2_df['instrip']].index.min()
            plume2_df = plume2_df.loc[plume2_start:]
            mask2 = plume2_df['instrip'].fillna(False)
            height2 = (plume2_df.loc[mask2, 'ft_posx'].max() - plume2_df.loc[mask2, 'ft_posx'].min()) * np.sqrt(2) + 30
            plume2_heights.append(height2)
        max_plume2_height = max(plume2_heights)

        # --- Global x/y limits ---
        all_x, all_y = [], []
        for this_experiment in experiments:
            _, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
            all_x.extend(list(exp_df['ft_posx'] - xo))
            all_y.extend(list(exp_df['ft_posy'] - yo))

        x_pad = (max(all_x) - min(all_x)) * 0.1
        y_pad = (max(all_y) - min(all_y)) * 0.1
        xmin, xmax = min(all_x) - x_pad, max(all_x) + x_pad
        ymin, ymax = min(all_y) - y_pad, max(all_y) + y_pad

        # --- Plot each experiment ---
        for ax, this_experiment in zip(axs, experiments):
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment

            plume1_df = exp_df[exp_df['exp_phase'] == 1]
            plume1_start = plume1_df[plume1_df['instrip']].index.min()
            plume1_df = plume1_df.loc[plume1_start:]
            plume2_df = exp_df[exp_df['exp_phase'] == 3]
            plume2_start = plume2_df[plume2_df['instrip']].index.min()
            plume2_df = plume2_df.loc[plume2_start:]
            plume_adj = 25 / np.sqrt(2)
            width = 50
            mask1 = plume1_df['instrip'].fillna(False)
            height1 = (plume1_df.loc[mask1, 'ft_posx'].max() - plume1_df.loc[mask1, 'ft_posx'].min()) * np.sqrt(2) + 30
            height2 = max_plume2_height  # force equal plume2 length

            # Determine angles & offsets
            if 'LR' in filename:
                angle1, angle2 = 45, -45
                x1 = plume1_df.iloc[0]['ft_posx'] - xo - plume_adj
                y1 = plume1_df.iloc[0]['ft_posy'] - yo - plume_adj
                x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
                y2 = plume2_df.iloc[0]['ft_posy'] - yo + plume_adj
            elif 'RR' in filename:
                angle1, angle2 = -45, -45
                x1 = plume1_df.iloc[0]['ft_posx'] - xo - plume_adj
                y1 = plume1_df.iloc[0]['ft_posy'] - yo + plume_adj
                x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
                y2 = plume2_df.iloc[0]['ft_posy'] - yo + plume_adj
            elif 'RL' in filename:
                angle1, angle2 = -45, 45
                x1 = plume1_df.iloc[0]['ft_posx'] - xo - plume_adj
                y1 = plume1_df.iloc[0]['ft_posy'] - yo + plume_adj
                x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
                y2 = plume2_df.iloc[0]['ft_posy'] - yo - plume_adj
            elif 'LL' in filename:
                angle1, angle2 = 45, 45
                x1 = plume1_df.iloc[0]['ft_posx'] - xo - plume_adj
                y1 = plume1_df.iloc[0]['ft_posy'] - yo - plume_adj
                x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
                y2 = plume2_df.iloc[0]['ft_posy'] - yo - plume_adj

            # Draw rectangles
            plume1 = patches.Rectangle((x1, y1), width, height1, angle=angle1,
                                       facecolor=plume_color, alpha=0.5)
            plume2 = patches.Rectangle((x2, y2), width, height2, angle=angle2,
                                       facecolor=plume_color, alpha=0.5)
            ax.add_patch(plume1)
            ax.add_patch(plume2)
            ax.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
            ax.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
            ax.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
            ax.set_title(filename, fontsize=14, color=markc)
            ax.set_xlabel('x-position (mm)', fontsize=12, color=markc)
            ax.set_ylabel('y-position (mm)', fontsize=12, color=markc)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal', adjustable='box')
            sns.despine(ax=ax, offset=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        savename = f"{fly_id}_odor_trajectory.pdf"
        plt.savefig(os.path.join(figure_folder, savename))
        plt.show()



def dir_training_return_efficiency(folder_path, savename, groups=['AA', 'BA'], size=(3,5), colors=['lightgrey', '#7ed4e6']):
    params_list = exp_parameters_multiodor(folder_path)
    data_dict = {}
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    for this_experiment in params_list:
        if len(this_experiment) == 10:
            multi = False
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        elif len(this_experiment) == 11:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, bl_df, exp_df, xo, yo, plume_colors = this_experiment
            multi = True
        if 'ctrl' in filename:
            continue
        if multi:
            label = groups[1] if 'mch' in filename else groups[0]
        else:
            label = groups[0] if 'LL' in filename or 'RR' in filename else groups[1]


        match = re.search(r'Fly(\d+)', filename)
        if not match:
            continue
        fly_id = int(match.group(1))

        t_match = re.search(r'(\d{8}-\d{6})', filename)
        plume2_df = exp_df[exp_df['exp_phase']==3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        _, d_in, d_out = inside_outside(plume2_df)
        if 'RR' in filename:
            angle = -45
        elif 'LL' in filename:
            angle = 45

        # entries = len(d_in) - 1
        entries = 0
        for key, df in d_out.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5:
                if return_to_edge_angle(df, angle=angle):
                    if (df['ft_posy'].iloc[-1] > df['ft_posy'].iloc[0]):
                        entries += 1
        pl = get_a_bout_calc(plume2_df, 'path_length') / 1000
    
        data_dict.setdefault(fly_id, {})[label] = entries / pl

    data1 = []
    data2 = []
    for fly_id, data in data_dict.items():
        print(fly_id, data)
    
    for fly_id, data in data_dict.items():
        plt.plot([1,2], [data[groups[0]], data[groups[1]]], color=markc, alpha=0.8)
        data1.append(data[groups[0]])
        data2.append(data[groups[1]])

    stat, p_value = wilcoxon(data1, data2)
    p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(data1, default=0), max(data2, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)
    
    axs.plot([1,2], [np.mean(data1), np.mean(data2)], color=colors[1], linewidth=3)
    sem1 = sem(data1) 
    sem2 = sem(data2)
    axs.errorbar(1, np.mean(data1), yerr=sem1, color=colors[1], capsize=0, linewidth=3)
    axs.errorbar(2, np.mean(data2), yerr=sem2, color=colors[1], capsize=0, linewidth=3)

    for pos in ['right', 'top']:
        axs.spines[pos].set_visible(False)
    plt.tight_layout()
    sns.despine(offset=10)
    for _, spine in axs.spines.items():
        spine.set_linewidth(2)
    for spine in axs.spines.values():
        spine.set_edgecolor(markc)
    plt.ylabel('returns per meter', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks(range(1, len(groups)+1))
    axs.set_xticks([1, 2])
    axs.set_xticklabels(groups, fontsize=16, color=markc)
    axs.tick_params(axis='x', which='both', length=0)
    axs.spines['bottom'].set_visible(False)
    plt.xlim(0.7, len(groups) + 0.3)
    plt.tight_layout()
    plt.show()

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def dir_training_return_eff_comp(folder_path, savename, groups=['AA', 'BA'], size=(3,5), colors=['lightgrey', '#7ed4e6']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        plume2_df = exp_df[exp_df['exp_phase']==3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]

        _, d_in1, d_out1 = inside_outside(plume1_df)
        _, d_in2, d_out2 = inside_outside(plume2_df)
        
        uw_entries = 0

        if 'ctrl' in filename:
            entries = len(d_in1) - 1
            for key, df in d_out1.items():
                if df['ft_posy'].iloc[-1] > df['ft_posy'].iloc[0]:
                    uw_entries +=1
            pl = get_a_bout_calc(plume1_df, 'path_length') / 1000
            data1.append(uw_entries/pl)
        else:
            entries = len(d_in2) - 1
            for key, df in d_out2.items():
                if df['ft_posy'].iloc[-1] > df['ft_posy'].iloc[0]:
                    uw_entries +=1
            pl = get_a_bout_calc(plume2_df, 'path_length') / 1000
            if 'pulse' in filename:
                data2.append(uw_entries/pl)
                print(filename)
            elif 'LL' in filename or 'RR' in filename:
                data4.append(uw_entries/pl)
            else:
                data3.append(uw_entries/pl)

    plt.scatter(1 + np.random.normal(0, 0.05, len(data1)), data1, color=colors[0])
    plt.scatter(2 + np.random.normal(0, 0.05, len(data2)), data2, color=colors[1])     
    plt.scatter(3 + np.random.normal(0, 0.05, len(data3)), data3, color=colors[2])     
    plt.scatter(4 + np.random.normal(0, 0.05, len(data3)), data4, color=colors[3])     

    mean1, sem1 = np.mean(np.array(data1)), sem(data1) 
    mean2, sem2 = np.mean(np.array(data2)), sem(data2)
    mean3, sem3 = np.mean(np.array(data3)), sem(data3)
    mean4, sem4 = np.mean(np.array(data4)), sem(data4)
    axs.errorbar(1, np.mean(data1), yerr=sem1, color=markc, capsize=0, linewidth=3)
    axs.errorbar(2, np.mean(data2), yerr=sem2, color=markc, capsize=0, linewidth=3)
    axs.errorbar(3, np.mean(data3), yerr=sem3, color=markc, capsize=0, linewidth=3)
    axs.errorbar(4, np.mean(data4), yerr=sem4, color=markc, capsize=0, linewidth=3)
    axs.hlines(mean1, 0.9, 1.1, colors=markc, linewidth=3)
    axs.hlines(mean2, 1.9, 2.1, colors=markc, linewidth=3)
    axs.hlines(mean3, 2.9, 3.1, colors=markc, linewidth=3)
    axs.hlines(mean4, 3.9, 4.1, colors=markc, linewidth=3)

    for pos in ['right', 'top']:
        axs.spines[pos].set_visible(False)
    plt.tight_layout()
    sns.despine(offset=10)
    for _, spine in axs.spines.items():
        spine.set_linewidth(2)
    for spine in axs.spines.values():
        spine.set_edgecolor(markc)
    plt.ylabel('returns per meter', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks(range(1, len(groups)+1))
    axs.set_xticks([1, 2, 3, 4])
    axs.set_xticklabels(groups, fontsize=16, color=markc)
    axs.tick_params(axis='x', which='both', length=0)
    axs.spines['bottom'].set_visible(False)
    plt.xlim(0.7, len(groups) + 0.3)
    plt.tight_layout()
    plt.show()
    print(f'nt: {data1}')
    print(f'ol: {data2}')
    print(f'ba: {data3}')
    print(f'aa: {data4}')
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def dir_training_prog_comp(folder_path, savename, groups=['AA', 'BA'], size=(3,5), colors=['lightgrey', '#7ed4e6']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        plume2_df = exp_df[exp_df['exp_phase']==3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        plume1_odor = plume1_df[plume1_df.mfc3_stpt > 0]   
        plume2_odor = plume2_df[plume2_df.mfc3_stpt > 0]  

        if 'ctrl' in filename:
            dist = np.abs((plume1_odor['ft_posx'].iloc[-1] - plume1_odor['ft_posx'].iloc[0]) / 1000) * math.sqrt(2)
            data1.append(dist)
        else:
            dist = np.abs((plume2_odor['ft_posx'].iloc[-1] - plume2_odor['ft_posx'].iloc[0]) / 1000) * math.sqrt(2)
            if 'pulse' in filename:
                data2.append(dist)
            elif 'LL' in filename or 'RR' in filename:
                data4.append(dist)
            else:
                data3.append(dist)

    plt.scatter(1 + np.random.normal(0, 0.05, len(data1)), data1, color=colors[0])
    plt.scatter(2 + np.random.normal(0, 0.05, len(data2)), data2, color=colors[1])     
    plt.scatter(3 + np.random.normal(0, 0.05, len(data3)), data3, color=colors[2])     
    plt.scatter(4 + np.random.normal(0, 0.05, len(data4)), data4, color=colors[3])     

    mean1, sem1 = np.mean(np.array(data1)), sem(data1) 
    mean2, sem2 = np.mean(np.array(data2)), sem(data2)
    mean3, sem3 = np.mean(np.array(data3)), sem(data3)
    mean4, sem4 = np.mean(np.array(data4)), sem(data4)
    axs.errorbar(1, np.mean(data1), yerr=sem1, color=markc, capsize=0, linewidth=3)
    axs.errorbar(2, np.mean(data2), yerr=sem2, color=markc, capsize=0, linewidth=3)
    axs.errorbar(3, np.mean(data3), yerr=sem3, color=markc, capsize=0, linewidth=3)
    axs.errorbar(4, np.mean(data4), yerr=sem4, color=markc, capsize=0, linewidth=3)
    axs.hlines(mean1, 0.9, 1.1, colors=markc, linewidth=3)
    axs.hlines(mean2, 1.9, 2.1, colors=markc, linewidth=3)
    axs.hlines(mean3, 2.9, 3.1, colors=markc, linewidth=3)
    axs.hlines(mean4, 3.9, 4.1, colors=markc, linewidth=3)

    for pos in ['right', 'top']:
        axs.spines[pos].set_visible(False)
    plt.tight_layout()
    sns.despine(offset=10)
    for _, spine in axs.spines.items():
        spine.set_linewidth(1)
    for spine in axs.spines.values():
        spine.set_edgecolor(markc)
    plt.ylabel('distance tracked (m)', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks(range(1, len(groups)+1))
    axs.set_xticks([1, 2, 3, 4])
    axs.set_xticklabels(groups, fontsize=16, color=markc)
    axs.tick_params(axis='x', which='both', length=0)
    axs.spines['bottom'].set_visible(False)
    plt.xlim(0.7, len(groups) + 0.3)
    plt.tight_layout()
    plt.show()
    print(f'nt: {data1}')
    print(f'ol: {data2}')
    print(f'ba: {data3}')
    print(f'aa: {data4}')
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def dir_training_progression(folder_path, savename, groups=['AA', 'BA'], size=(3,5), colors=['lightgrey', '#7ed4e6']):
    if 'mch' in folder_path:
        params_list = exp_parameters_multiodor(folder_path)
        multi = True
    else:
        params_list = exp_parameters(folder_path)
        multi = False    
    data_dict = {}
    fig, axs, markc = configure_white_plot(size=size, xaxis=False)
    for this_experiment in params_list:
        if len(this_experiment) == 9:
            figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        elif len(this_experiment) == 10:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        if 'ctrl' in filename:
            continue
        if multi:
            label = groups[1] if 'mch' in filename else groups[0]
        else:
            label = groups[0] if 'LL' in filename or 'RR' in filename else groups[1]

        match = re.search(r'Fly(\d+)', filename)
        if not match:
            continue
        fly_id = int(match.group(1))

        t_match = re.search(r'(\d{8}-\d{6})', filename)
        plume2_df = exp_df[exp_df['exp_phase']==3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        _, d_in, d_out = inside_outside(plume2_df)
        if 'mch' in filename:
            plume2_df_odor = df[df.mfc2_stpt > 0]
        else:
            plume2_df_odor = df[df.mfc3_stpt > 0]

        dist = (plume2_df_odor['ft_posx'].iloc[-1] - plume2_df_odor['ft_posx'].iloc[0]) / 1000
        if 'LL' in filename or 'RL' in filename:
            dist = -dist
        
        data_dict.setdefault(fly_id, {})[label] = dist * math.sqrt(2)

    data1 = []
    data2 = []
    for fly_id, data in data_dict.items():
        plt.plot([1,2], [data[groups[0]], data[groups[1]]], color=markc, alpha=0.8)
        data1.append(data[groups[0]])
        data2.append(data[groups[1]])

    stat, p_value = wilcoxon(data1, data2)
    p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(data1, default=0), max(data2, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)
    
    axs.plot([1,2], [np.mean(data1), np.mean(data2)], color=colors[1], linewidth=3)
    sem1 = sem(data1) 
    sem2 = sem(data2)
    axs.errorbar(1, np.mean(data1), yerr=sem1, color=colors[1], capsize=0, linewidth=3)
    axs.errorbar(2, np.mean(data2), yerr=sem2, color=colors[1], capsize=0, linewidth=3)

    for pos in ['right', 'top']:
        axs.spines[pos].set_visible(False)
    plt.tight_layout()
    sns.despine(offset=10)
    for _, spine in axs.spines.items():
        spine.set_linewidth(2)
    for spine in axs.spines.values():
        spine.set_edgecolor(markc)
    plt.ylabel('progression along plume axis (m)', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks(range(1, len(groups)+1))
    axs.set_xticks([1, 2])
    axs.set_xticklabels(groups, fontsize=16, color=markc)
    axs.tick_params(axis='x', which='both', length=0)
    axs.spines['bottom'].set_visible(False)
    plt.xlim(0.7, len(groups) + 0.3)
    plt.tight_layout()
    plt.show()

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def dir_training_return_efficiency_ctrl(folder_path, savename, groups=['MBON', 'DAN'], size=(3,5), colors=['lightgrey', '#7ed4e6']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    ctrls = []
    exps = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if 'LR' in filename or 'RL' in filename:
            continue
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        if (exp_df['exp_phase'] == 3).any():
            plume2_df = exp_df[exp_df['exp_phase'] == 3]
            plume2_start = plume2_df[plume2_df['instrip']].index.min()
            plume2_df = plume2_df.loc[plume2_start:]
        # _, d_in, d_out = inside_outside(plume1_df)
        # pl = get_a_bout_calc(plume1_df, 'path_length') / 1000
        _, d_in, d_out = inside_outside(plume2_df)
        pl = get_a_bout_calc(plume2_df, 'path_length') / 1000

        entries = len(d_in) -1
        # for key, df in d_out.items():
        #     if 'LL' in filename or 'LR' in filename:
        #         if return_to_edge_angle(df, angle=45):
        #             entries +=1
        #     elif 'RL' in filename or 'RR' in filename:
        #         if return_to_edge_angle(df, angle=-45):
        #             entries +=1
    
        if 'MBON' in filename:
            ctrls.append(entries/pl)
        else:
            exps.append(entries/pl)
        # else:
        #     if 'ctrl' in filename:
        #         ctrls.append(0)
    print(ctrls)
    print(exps)
    plt.scatter(1 + np.random.normal(0, 0.05, len(exps)), exps, color=colors[1])
    plt.scatter(2 + np.random.normal(0, 0.05, len(ctrls)), ctrls, color=colors[0])       
    ctrl, exp = np.array(ctrls), np.array(exps)    
    ctrl_avg, ctrl_sem = np.mean(ctrl), sem(ctrl)
    exp_avg, exp_sem = np.mean(exp), sem(exp)
    axs.hlines(ctrl_avg, 1.9, 2.1, colors=markc, linewidth=3)
    axs.hlines(exp_avg, 0.9, 1.1, colors=markc, linewidth=3)
    axs.errorbar(2, ctrl_avg, yerr=ctrl_sem, color=markc, capsize=0, linewidth=3)
    axs.errorbar(1, exp_avg, yerr=exp_sem, color=markc, capsize=0, linewidth=3)
    style_ax(axs, markc, title=None, xtick_labels=['exp', 'ctrl'], xticks=[1,2], ylabel='returns per meter', xlim=(0.7,2.3))
    plt.tight_layout()
    plt.show()

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')



def dir_training_bias_paired(folder_path, savename, size, avgcolor):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    lefts=[]
    rights=[]
    grouped_experiments = defaultdict(list)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        match = re.search(r'[Ff]ly\d+', filename)
        if match:
            fly_id = match.group()
        else:
            continue  # skip files without Fly ID
        grouped_experiments[fly_id].append(this_experiment)

    for fly_id, experiments in grouped_experiments.items():
        left, right = None, None  # reset for each fly
        for this_experiment in experiments:
            _, filename, _, df_odor, df_light, _, exp_df, xo, yo, plume_color = this_experiment
            if 'ctrl' in filename:
                continue
            plume1_df = exp_df[exp_df['exp_phase'] == 1]
            plume2_df = exp_df[exp_df['exp_phase'] == 3]
            start_x = plume1_df.iloc[-1]['ft_posx']
            start_y = plume1_df.iloc[-1]['ft_posy']
            # start_x = plume2_df.iloc[0]['ft_posx']
            # start_y = plume2_df.iloc[0]['ft_posy']
            end_x = plume2_df.iloc[-1]['ft_posx']
            end_y = plume2_df.iloc[-1]['ft_posy']
            dx, dy = end_x - start_x, end_y - start_y
            angle = np.degrees(np.arctan2(dx, abs(dy)))

            if 'LR' in filename or 'LL' in filename:
                left = angle
                lefts.append(angle)
            elif 'RL' in filename or 'RR' in filename:
                right = angle
                rights.append(angle)

        # plot only if we have both sides
        if left is not None and right is not None:
            axs.plot([1, 2], [left, right], linestyle='-', color=markc, label=fly_id, alpha=0.8)
    lefts = np.array(lefts)
    rights = np.array(rights)
    left_avg, left_sem = np.mean(lefts), sem(lefts)
    right_avg, right_sem = np.mean(rights), sem(rights)
    axs.plot([1,2], [left_avg, right_avg], color=avgcolor, linewidth=3)
    axs.errorbar(1, left_avg, yerr=left_sem, color=avgcolor, capsize=0, linewidth=3)
    axs.errorbar(2, right_avg, yerr=right_sem, color=avgcolor, capsize=0, linewidth=3)
    stat, p_value = wilcoxon(lefts, rights)
    p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(lefts, default=0), max(rights, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)

    axs.axhline(0, color=markc, linestyle='--', linewidth=1)
    plt.yticks(fontsize=14, color=markc)
    plt.ylabel('net travel direction (º)', fontsize=18, color=markc)
    axs.set_ylim([90, 90])
    axs.set_yticks([-90, 0, 90])
    axs.set_yticklabels(['-90', '0', '+90'], color=markc)
    plt.xlim(0.7, 2.3)
    axs.set_xticks([1, 2])
    axs.set_xticklabels(['left', 'right'], fontsize=16, color=markc)
    axs.tick_params(axis='x', which='both', length=0)
    axs.spines['bottom'].set_visible(False)
    # plt.legend()
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def dir_training_bias_paired_polar(folder_path, savename, cond, size, avgcolor):
    params_list = exp_parameters_multiodor(folder_path)
    fig, axs, markc = configure_white_polar(size=size, grid=False)
    lefts = []
    rights = []
    pulses = []
    grouped_experiments = defaultdict(list)
    for this_experiment in params_list:
        if len(this_experiment) == 10:
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        elif len(this_experiment) == 11:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, bl_df, exp_df, xo, yo, plume_colors = this_experiment
    
        left, right = None, None  # reset for each fly
        if 'ctrl' in filename:
            continue
        plume2_df = exp_df[exp_df['exp_phase'] == 3]
        if cond == 'baseline':
            df = bl_df
        else:
            df = plume2_df
        start_x = df.iloc[0]['ft_posx']
        start_y = df.iloc[0]['ft_posy']
        end_x = df.iloc[-1]['ft_posx']
        end_y = df.iloc[-1]['ft_posy']
        dx, dy = end_x - start_x, end_y - start_y
        angle = np.arctan2(dx, dy)   # NOTE the swap
        angle = angle % (2*np.pi)
        # if 'LL' in filename: # or 'LL' in filename:
        #     left = angle
        #     lefts.append(angle)
        # if 'RL' in filename: # or 'RR' in filename:
            # right = angle
            # rights.append(angle)
        if 'pulse' in filename:
            pulse = angle
            pulses.append(pulse)

        # if left is not None:
        #     plt.polar([0, left], [0, 1], color=avgcolor, alpha=0.8, linewidth=1)
        # if right is not None:
        #     plt.polar([0, right], [0, 1], color=avgcolor, alpha=0.8, linewidth=1)
        if pulse is not None:
            plt.polar([0, pulse], [0, 1], color=avgcolor, alpha=0.8, linewidth=1)

    if len(lefts) > 0:
        v = np.mean(np.column_stack((np.cos(lefts), np.sin(lefts))), axis=0)
        rho, phi = cart2pol(v[0], v[1])
        plt.polar([0, phi], [0, rho], color=avgcolor, alpha=1, linewidth=5, solid_capstyle='round')

    if len(rights) > 0:
        v = np.mean(np.column_stack((np.cos(rights), np.sin(rights))), axis=0)
        rho, phi = cart2pol(v[0], v[1])
        plt.polar([0, phi], [0, rho], color=avgcolor, alpha=1, linewidth=5, solid_capstyle='round')
    
    if len(pulses) > 0:
        v = np.mean(np.column_stack((np.cos(pulses), np.sin(pulses))), axis=0)
        rho, phi = cart2pol(v[0], v[1])
        plt.polar([0, phi], [0, rho], color=avgcolor, alpha=1, linewidth=5, solid_capstyle='round')

    plt.tight_layout()
    plt.show()

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')
    

def dir_training_bias_change_odor_vs_dir(folder_path, savename, size, colors):
    def wrap180(angle):
        return (angle + 180) % 360 - 180

    params_list = exp_parameters_multiodor(folder_path)
    print("number of experiments:", len(params_list))
    fig, axs, markc = configure_white_plot(size=size, xaxis=False)
    BA = []
    OM = []

    for this_experiment in params_list:
        if len(this_experiment) == 10:
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
            multi = False
        elif len(this_experiment) == 11:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, bl_df, exp_df, xo, yo, plume_colors = this_experiment
            multi = True

        print(filename)

        # calculate entry angle
        plume1_df = exp_df[exp_df['exp_phase'] == 1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        d, di, do = inside_outside(plume1_df)
        print(len(do))
        headings, mean_heading, ea_strength = compute_mean_entry_heading(do, imaging=False)
        entry_angle = np.rad2deg(mean_heading)
        
        print(entry_angle)

        # calculate net travel direction
        plume2_df = exp_df[exp_df['exp_phase'] == 3]
        start_x = plume2_df.iloc[0]['ft_posx']
        start_y = plume2_df.iloc[0]['ft_posy']
        end_x = plume2_df.iloc[-1]['ft_posx']
        end_y = plume2_df.iloc[-1]['ft_posy']
        dx, dy = end_x - start_x, end_y - start_y
        trav_angle = np.arctan2(dx, dy)
        travel_angle = np.rad2deg(trav_angle)
        print(travel_angle)

        shift = wrap180(travel_angle - entry_angle)
        if 'LL' in filename or 'LR' in filename:
            shift = -shift
        if multi:
            OM.append(shift)
        else:
            BA.append(shift)

    plt.scatter(1 + np.random.normal(0, 0.05, len(BA)), BA, color=colors[0])
    plt.scatter(2 + np.random.normal(0, 0.05, len(OM)), OM, color=colors[1])
    mean1, sem1 = np.mean(np.array(BA)), sem(BA)
    mean2, sem2 = np.mean(np.array(OM)), sem(OM)
    axs.errorbar(1, mean1, yerr=sem1, color=markc, capsize=0, linewidth=3)
    axs.errorbar(2, mean2, yerr=sem2, color=markc, capsize=0, linewidth=3)
    axs.hlines(mean1, 0.9, 1.1, colors=markc, linewidth=3)
    axs.hlines(mean2, 1.9, 2.1, colors=markc, linewidth=3)

    stat, p_value = mannwhitneyu(BA, OM, alternative='two-sided')
    p_text = f'p = {p_value:.4g}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(BA, default=0), max(OM, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)

    axs.axhline(0, color=markc, linewidth=1, linestyle='--', alpha=0.5)
    plt.ylabel('entry → travel shift (°)', fontsize=16, color=markc)
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks([1, 2])
    axs.set_xticklabels(['BA', 'OM'], fontsize=16, color=markc)
    axs.tick_params(axis='x', which='both', length=0)
    axs.spines['bottom'].set_visible(False)
    plt.ylim(-180, 180)
    plt.xlim(0.7, 2.3)
    plt.tight_layout()
    plt.show()

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def dir_training_bias_change_BA_vs_AA(folder_path, savename, size, colors):
    def wrap180(angle):
        return (angle + 180) % 360 - 180
    params_list = exp_parameters(folder_path)
    print("number of experiments:", len(params_list))

    fig, axs, markc = configure_white_plot(size=size, xaxis=False)
    BA = []
    AA = []

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if 'ctrl' in filename:
            continue
        print(filename)

        # calculate entry angle
        plume1_df = exp_df[exp_df['exp_phase'] == 1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        d, di, do = inside_outside(plume1_df)
        headings, mean_heading, ea_strength = compute_mean_entry_heading(do, imaging=False)
        entry_angle = np.rad2deg(mean_heading)
        
        print(entry_angle)

        # calculate net travel direction
        plume2_df = exp_df[exp_df['exp_phase'] == 3]
        start_x = plume2_df.iloc[0]['ft_posx']
        start_y = plume2_df.iloc[0]['ft_posy']
        end_x = plume2_df.iloc[-1]['ft_posx']
        end_y = plume2_df.iloc[-1]['ft_posy']
        dx, dy = end_x - start_x, end_y - start_y
        trav_angle = np.arctan2(dx, dy)
        travel_angle = np.rad2deg(trav_angle)
        print(travel_angle)

        shift = wrap180(travel_angle - entry_angle)
        if 'RL' in filename or 'LR' in filename:
            if 'LR' in filename:
                shift = -shift
            BA.append(shift)
        elif 'RR' in filename or 'LL' in filename:
            if 'LL' in filename:
                shift = -shift
            AA.append(shift)

    plt.scatter(1 + np.random.normal(0, 0.05, len(BA)), BA, color=colors[0])
    plt.scatter(2 + np.random.normal(0, 0.05, len(AA)), AA, color=colors[1])
    mean1, sem1 = np.mean(np.array(BA)), sem(BA)
    mean2, sem2 = np.mean(np.array(AA)), sem(AA)
    axs.errorbar(1, mean1, yerr=sem1, color=markc, capsize=0, linewidth=3)
    axs.errorbar(2, mean2, yerr=sem2, color=markc, capsize=0, linewidth=3)
    axs.hlines(mean1, 0.9, 1.1, colors=markc, linewidth=3)
    axs.hlines(mean2, 1.9, 2.1, colors=markc, linewidth=3)

    stat, p_value = mannwhitneyu(BA, AA, alternative='two-sided')
    p_text = f'p = {p_value:.4g}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(BA, default=0), max(AA, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)

    axs.axhline(0, color=markc, linewidth=1, linestyle='--', alpha=0.5)
    plt.ylabel('entry → travel shift (°)', fontsize=16, color=markc)
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks([1, 2])
    axs.set_xticklabels(['BA', 'AA'], fontsize=16, color=markc)
    axs.tick_params(axis='x', which='both', length=0)
    axs.spines['bottom'].set_visible(False)
    plt.ylim(-180, 180)
    plt.xlim(0.7, 2.3)
    plt.tight_layout()
    plt.show()

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def dir_training_bias_paired_ctrl(folder_path, savename, size, colors=['lightgrey', 'blue']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    lefts=[]
    rights=[]
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, filename, _, df_odor, df_light, _, exp_df, xo, yo, plume_color = this_experiment
        if 'ctrl' not in filename:
            continue
        start_x = exp_df.iloc[0]['ft_posx']
        start_y = exp_df.iloc[0]['ft_posy']
        end_x = exp_df.iloc[-1]['ft_posx']
        end_y = exp_df.iloc[-1]['ft_posy']
        dx, dy = end_x - start_x, end_y - start_y
        angle = np.degrees(np.arctan2(dx, abs(dy)))

        if 'LR' in filename or 'LL' in filename:
            lefts.append(angle)
        elif 'RL' in filename or 'RR' in filename:
            rights.append(angle)

    plt.scatter(1 + np.random.normal(0, 0.05, len(lefts)), lefts, color=colors[0], alpha=0.8)
    plt.scatter(2 + np.random.normal(0, 0.05, len(rights)), rights, color=colors[1], alpha=0.8)
    lefts = np.array(lefts)
    rights = np.array(rights)
    left_avg, left_sem = np.mean(lefts), sem(lefts)
    right_avg, right_sem = np.mean(rights), sem(rights)
    axs.hlines(left_avg, 0.9, 1.1, color=markc, linewidth=3)
    axs.hlines(right_avg, 1.9, 2.1, color=markc, linewidth=3)
    axs.errorbar(1, left_avg, yerr=left_sem, color=markc, capsize=0, linewidth=3)
    axs.errorbar(2, right_avg, yerr=right_sem, color=markc, capsize=0, linewidth=3)
    stat, p_value = mannwhitneyu(lefts, rights, alternative='two-sided')
    p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(lefts, default=0), max(rights, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)

    axs.axhline(0, color=markc, linestyle='--', linewidth=1)
    plt.yticks(fontsize=14, color=markc)
    plt.ylabel('net travel direction (º)', fontsize=18, color=markc)
    axs.set_ylim([90, 90])
    axs.set_yticks([-90, 0, 90])
    axs.set_yticklabels(['-90', '0', '+90'], color=markc)
    plt.xlim(0.7, 2.3)
    axs.set_xticks([1, 2])
    axs.set_xticklabels(['left', 'right'], fontsize=16, color=markc)
    axs.tick_params(axis='x', which='both', length=0)
    axs.spines['bottom'].set_visible(False)
    # plt.legend()
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')
        


def dir_training_entry_angles(folder_path, savename, size=(5,5), colors=['lightgrey', '#ff355e']):
    params_list = exp_parameters_multiodor(folder_path)
    fig, axs, markc = configure_white_polar(size=size, grid=False)
    angles_L = [] #exit
    angles_R = [] #entry
    for this_experiment in params_list:
        if len(this_experiment) == 10:
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        elif len(this_experiment) == 11:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, bl_df, exp_df, xo, yo, plume_colors = this_experiment
    
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if 'ctrl' in filename:
            continue
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        plume2_df = exp_df[exp_df['exp_phase']==3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        _, di1, do1 = inside_outside(plume1_df)
        _, di2, do2 = inside_outside(plume2_df)
        for k in sorted(di1.keys()):  # iterate over 1, 3, 5, ...
            if k == 1:  # first entry has no preceding do
                entry_df = get_entry_df(None, di1[k], exp_df, window=0.5, first_entry=True)
            else:  # use preceding do (key = k-1)
                entry_df = get_entry_df(do1.get(k-1), di1[k], exp_df, window=0.5)
            if entry_df is not None:
                if 'LR' in filename or 'LL' in filename:
                    circmean = circmean_heading(entry_df, angles_R)
                    # circmean = (-circmean) % (2*np.pi)   # mirror
                    angles_R[-1] = circmean  # overwrite the last appended value
                    plt.polar([0, circmean], [0, 1], color=colors[0], alpha=0.6, linewidth=1)
                # if 'RL' in filename or 'RR' in filename:
                #     circmean = circmean_heading(entry_df, angles_R)
                #     plt.polar([0, circmean], [0, 1], color=colors[0], alpha=0.6, linewidth=1)
        for k in sorted(do1.keys()):  # iterate over 1, 3, 5, ...
            entry_df = get_entry_df(di1.get(k-1), do1[k], exp_df, window=0.5)
            if entry_df is not None:
                if 'LR' in filename or 'LL' in filename:
                    circmean = circmean_heading(entry_df, angles_L)
                    circmean = (-circmean) % (2*np.pi)   # mirror
                    angles_L[-1] = circmean  # overwrite the last appended value
                    # plt.polar([0, circmean], [0, 1], color=colors[1], alpha=0.6, linewidth=1)
                elif 'RR' in filename or 'RL' in filename:
                    circmean = circmean_heading(entry_df, angles_L)
                    # plt.polar([0, circmean], [0, 1], color=colors[1], alpha=0.6, linewidth=1)
        
    if len(angles_L) > 0:
        summary_p1_vector = np.mean(np.column_stack((np.cos(angles_L), np.sin(angles_L))), axis=0)
        summary_p1_rho, summary_p1_phi = cart2pol(summary_p1_vector[0], summary_p1_vector[1])
        plt.polar([0, summary_p1_phi], [0, summary_p1_rho], color=colors[1], alpha=1, linewidth=5, solid_capstyle='round')
    if len(angles_R) > 0:
        summary_p2_vector = np.mean(np.column_stack((np.cos(angles_R), np.sin(angles_R))), axis=0)
        summary_p2_rho, summary_p2_phi = cart2pol(summary_p2_vector[0], summary_p2_vector[1])
        plt.polar([0, summary_p2_phi], [0, summary_p2_rho], color=colors[0], alpha=1, linewidth=5, solid_capstyle='round')
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def dir_training_plume_onset_traj(folder_path, savename, size=(6,5), time=True, keywords=['no training', 'L training', 'R training'], colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=(size[0], size[1]*2), xaxis=True, nrows=2, ncols=2)
    axs = np.array(axs).reshape(2, 2)
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment   
        if filename == '09102025-112628_PAM_Chr_oct_dir_training_LR_Fly10_over.log' or filename == '09102025-121050_PAM_Chr_oct_dir_training_RL_Fly9.log' or 'ctrl' in filename:
            continue
        
        plume2_df = exp_df[exp_df['exp_phase']==3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        _, d_in, d_out = inside_outside(plume2_df)
    
        if (len(d_out)) > 0:
            first_exit = d_out[list(d_out.keys())[0]] 
            end_idx_label = first_exit.index[-1]
            end_idx_pos = plume2_df.index.get_loc(end_idx_label)
            plume_df = plume2_df.iloc[:end_idx_pos]
            # plume_df = plume2_df
            xo = plume_df.iloc[0]['ft_posx']
            yo = plume_df.iloc[0]['ft_posy']
            end_x = plume_df.iloc[-1]['ft_posx'] 
            end_y = plume_df.iloc[-1]['ft_posy'] 
            dx = end_x - xo
            dy = end_y - yo
            plume_df_odor = plume_df.where(plume_df.mfc3_stpt > 0)
            if 'LR' in filename or 'LL' in filename:
                axs[0,0].plot(plume_df['ft_posx'] - xo, plume_df['ft_posy'] - yo, color='grey', linewidth=1)
                axs[0,0].plot(plume_df_odor['ft_posx'] - xo, plume_df_odor['ft_posy'] - yo, color=colors[0],  linewidth=1)
                axs[0,0].set_title(keywords[0], fontsize=14, color=markc)
                axs[1,0].arrow(0, 0, dx, dy, head_width=10, color=colors[0], alpha=0.8)
            elif 'RL' in filename or 'RR' in filename:
                axs[0,1].plot(plume_df['ft_posx'] - xo, plume_df['ft_posy'] - yo, color='grey', linewidth=1)
                axs[0,1].plot(plume_df_odor['ft_posx'] - xo, plume_df_odor['ft_posy'] - yo, color= colors[1], linewidth=1)
                axs[0,1].set_title(keywords[1], fontsize=14, color=markc)
                axs[1,1].arrow(0, 0, dx, dy, head_width=10, color=colors[1], alpha=0.8)
            
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
        ax.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color=markc, direction='out', left=True, bottom=True)
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
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def dir_training_test_onset_vector(folder_path, savename, size=(6,5), colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=2)
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment   
        if (exp_df['exp_phase'] == 3).any():
            plume2_df = exp_df[exp_df['exp_phase']==3]
            plume2_start = plume2_df[plume2_df['instrip']].index.min()
            plume2_df = plume2_df.loc[plume2_start:]
            _, d_in, d_out = inside_outside(plume2_df)
        else:
            continue
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        plume1_end = plume1_df.index.max()

        vector_df = exp_df.loc[plume1_end:plume2_start]

        xo = vector_df.iloc[0]['ft_posx']
        yo = vector_df.iloc[0]['ft_posy']
        end_x = vector_df.iloc[-1]['ft_posx'] 
        end_y = vector_df.iloc[-1]['ft_posy'] 
        dx = end_x - xo
        dy = end_y - yo

        if 'ctrl' in filename:
            if 'LL' in filename or 'LR' in filename:
                axs[0].arrow(0, 0, -dx, dy, head_width=3, color=colors[0], alpha=0.8)
            else:
                axs[0].arrow(0, 0, dx, dy, head_width=3, color=colors[0], alpha=0.8)
        else:
            if 'LL' in filename or 'LR' in filename:
                axs[1].arrow(0, 0, -dx, dy, head_width=3, color=colors[1], alpha=0.8)
            else:
                axs[1].arrow(0, 0, dx, dy, head_width=3, color=colors[1], alpha=0.8)
            
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
    for ax in axs.flat:
        ax.set_xlabel('x-position (mm)', fontsize=14)
        ax.set_ylabel('y-position (mm)', fontsize=14)
        ax.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color=markc, direction='out', left=True, bottom=True)
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
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def dir_training_plume_onset_first10_traj(folder_path, savename, size=(6,5), delay=15, colors=['#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment   
        if 'ctrl' in filename:
            continue
        plume2_df = exp_df[exp_df['exp_phase'] == 3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        t0 = plume2_df['seconds'].iloc[0]
        plume_df = plume2_df[plume2_df['seconds'] <= t0 + delay] # first 10 secs
        # plume_df=plume2_df
        plume_df_odor = plume_df.where(plume_df.mfc3_stpt > 0)
        xo = plume_df.iloc[0]['ft_posx']
        yo = plume_df.iloc[0]['ft_posy']
        end_x = plume_df.iloc[-1]['ft_posx'] 
        end_y = plume_df.iloc[-1]['ft_posy'] 
        dx = end_x - xo
        dy = end_y - yo
        
        if 'LR' in filename or 'LL' in filename:
            axs.plot(plume_df['ft_posx'] - xo, plume_df['ft_posy'] - yo, color=colors[0], linewidth=1)
            axs.arrow(0, 0, dx, dy, head_width=15,color=colors[0], alpha=0.8)
        # if 'RL' in filename or 'RR' in filename:
        #     axs.plot(plume_df['ft_posx'] - xo, plume_df['ft_posy'] - yo, color=colors[1], linewidth=1)
        #     axs.arrow(0, 0, dx, dy, head_width=15, color=colors[1], alpha=0.8)
    axs.set_xlabel('x-position (mm)', fontsize=14)
    axs.set_ylabel('y-position (mm)', fontsize=14)
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
    plt.tight_layout()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def dir_training_plume_training_traj(folder_path, savename, size=(6,5), colors=['#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=2)
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment   
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        _, di, do = inside_outside(plume1_df)
        # if (exp_df['exp_phase'] == 3).any():
        #     plume2_df = exp_df[exp_df['exp_phase'] == 3]
        #     plume2_start = plume2_df[plume2_df['instrip']].index.min()
        #     plume2_df = plume2_df.loc[plume2_start:]
        #     _, di, do = inside_outside(plume2_df)
        for key, plume_df in di.items():
            x_end = plume_df['ft_posx'].iloc[-1]
            y_end = plume_df['ft_posy'].iloc[-1]
            x_shifted = plume_df['ft_posx'] - x_end
            y_shifted = plume_df['ft_posy'] - y_end
            if 'LL' in filename or 'LR' in filename:
                x_shifted = -x_shifted
            if 'PAM' in filename:
                axs[0].plot(x_shifted, y_shifted, color=colors[0], linewidth=1, alpha=0.65)
            else:
                axs[1].plot(x_shifted, y_shifted, color=colors[1], linewidth=1, alpha=0.65)
        # for key, plume_df in list(do.items())[:1]:
        for key, plume_df in do.items():
            x_start = plume_df['ft_posx'].iloc[0]
            y_start = plume_df['ft_posy'].iloc[0]
            x_shifted = plume_df['ft_posx'] - x_start
            y_shifted = plume_df['ft_posy'] - y_start
            # flip for LL / LR
            if 'LL' in filename or 'LR' in filename:
                x_shifted = -x_shifted
            if 'PAM' in filename:
                axs[0].plot(x_shifted, y_shifted, color='lightgrey', linewidth=1, alpha=0.65)
            else:
                axs[1].plot(x_shifted, y_shifted, color='lightgrey', linewidth=1, alpha=0.65)
    xlims = [ax.get_xlim() for ax in axs]
    ylims = [ax.get_ylim() for ax in axs]

    xmin = min(x[0] for x in xlims)
    xmax = max(x[1] for x in xlims)
    ymin = min(y[0] for y in ylims)
    ymax = max(y[1] for y in ylims)

    for ax in axs:  # loop through each subplot axis
        ax.set_xlabel('x-position (mm)', fontsize=14)
        ax.set_ylabel('y-position (mm)', fontsize=14)
        ax.tick_params(which='both', axis='both', labelsize=12, length=3, width=2,
                    color=markc, direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            ax.spines[pos].set_visible(False)
        for _, spine in ax.spines.items():
            spine.set_linewidth(2)
            spine.set_edgecolor(markc)
        ax.tick_params(axis='both', colors=markc)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def dir_training_xpos_dist(folder_path, savename, size=(6,5), ctrl=True, colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    xpos_left = []
    xpos_right = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment          
        if not ctrl:
            if 'ctrl' in filename:
                continue
        elif ctrl:
            if 'ctrl' not in filename:
                continue
        
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume2_df = exp_df[exp_df['exp_phase']==3]

        if ctrl:
            xpos = exp_df['ft_posx'] - plume1_df.iloc[-1]['ft_posx']
        elif not ctrl:
            xpos = plume2_df['ft_posx'] - plume1_df.iloc[-1]['ft_posx']

        if 'LR' in filename or 'LL' in filename:
            xpos_left.append(xpos)
        elif 'RL' in filename or 'RR' in filename:
            xpos_right.append(xpos)
    
    xpos_left = pd.concat(xpos_left, axis=0, ignore_index=True)
    xpos_right = pd.concat(xpos_right, axis=0, ignore_index=True)
    sns.kdeplot(xpos_left, color=colors[0], shade=True, cut=0, label='left')
    sns.kdeplot(xpos_right, color=colors[1], shade=True, cut=0, label='right')
    plt.axvline(x=0, linestyle='--',color=markc)
    plt.xlim(-600,600)
    plt.xlabel('x-position (mm)', color=markc, fontsize=14)
    plt.ylabel('probability', color=markc, fontsize=14)
    plt.legend(frameon=False)
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')



def dir_training_heading_dist(folder_path, savename, size=(6,5), ctrl=True, colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    xpos_left = []
    xpos_right = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment          
        if not ctrl:
            if 'ctrl' in filename:
                continue
        elif ctrl:
            if 'ctrl' not in filename:
                continue
        
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume2_df = exp_df[exp_df['exp_phase']==3]

        heading = plume2_df['transformed_heading'] 

        if 'LR' in filename or 'LL' in filename:
            xpos_left.append(heading)
        elif 'RL' in filename or 'RR' in filename:
            xpos_right.append(heading)
    
    xpos_left = pd.concat(xpos_left, axis=0, ignore_index=True)
    xpos_right = pd.concat(xpos_right, axis=0, ignore_index=True)

    bins = np.arange(-180, 185, 10)

    axs.hist(xpos_left, bins=bins, color=colors[0], alpha=0.5, density=True, edgecolor=colors[0])
    axs.hist(xpos_right, bins=bins, color=colors[1], alpha=0.5, density=True, edgecolor=colors[1])

    plt.axvline(x=0, linestyle='--',color=markc)
    plt.xlim(-180,180)
    axs.set_xticks([-90, 90])
    plt.xlabel('heading (º)', color=markc, fontsize=14)
    plt.ylabel('probability', color=markc, fontsize=14)
    plt.legend(frameon=False)
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def dir_training_p1_heading_dist(folder_path, savename, size=(6,5), colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    heading_ctrl = []
    heading_exp = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment          
        
        plume1_df = exp_df[exp_df['exp_phase'] == 1]
        plume1_df_odor = plume1_df.where(plume1_df.mfc3_stpt > 0)
        heading = plume1_df_odor['transformed_heading'] 

        if 'ctrl' in filename:
            heading_ctrl.append(heading)
        else:
            heading_exp.append(heading)
    
    heading_ctrl = pd.concat(heading_ctrl, axis=0, ignore_index=True)
    heading_exp = pd.concat(heading_exp, axis=0, ignore_index=True)

    bins = np.arange(-180, 185, 10)

    axs.hist(heading_ctrl, bins=bins, color=colors[0], alpha=0.5, density=True, edgecolor=colors[0])
    axs.hist(heading_exp, bins=bins, color=colors[1], alpha=0.5, density=True, edgecolor=colors[1])

    # plt.axvline(x=0, linestyle='--',color=markc)
    plt.xlim(-180,180)
    axs.set_xticks([-90, 90])
    plt.xlabel('heading (º)', color=markc, fontsize=14)
    plt.ylabel('probability', color=markc, fontsize=14)
    plt.legend(frameon=False)
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def dir_training_plume_onset_uws(folder_path, savename, multiodor=True, size=(6,5), sample_rate=20, buf_pts=4, keywords=['no training', 'L training', 'R training'], colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
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
        if 'ctrl' in filename:
            _, di, do = inside_outside(exp_df)
            first_key = list(di.keys())[0]
            df_in = di[first_key]
            key = keywords[0]
        else:
            plume2_df = exp_df[exp_df['exp_phase']==3]
            plume2_start = plume2_df[plume2_df['instrip']].index.min()
            plume2_df = plume2_df.loc[plume2_start:]
            _, d_in2, d_out2 = inside_outside(plume2_df)
            first_key = list(d_in2.keys())[0]
            df_in = d_in2[first_key]
            if ('RR' in filename or 'LL' in filename):
                key = keywords[1]
            elif ('LR' in filename or 'RL' in filename):
                key = keywords[2]      
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


def opto_uws_trace(folder_path, savename, size=(6,5), sample_rate=20, pre=4, post=4,
                    keywords=['no training', 'L training', 'R training'],
                    colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    pre_pts = int(pre * sample_rate)
    post_pts = int(post * sample_rate)
    time = np.linspace(-pre, post, pre_pts + post_pts)
    uws_traces = {kw: [] for kw in keywords}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if 'light' in filename:
            key = keywords[1]
        else:
            key = keywords[0]
        d, d_in, d_out = inside_outside(exp_df)
        first_key = list(d_in.keys())[0]
        df_in = d_in[first_key]

        onset_time = df_in.index[0]
        if onset_time in df.index:
            onset_idx = df.index.get_loc(onset_time)
            # Make sure enough data exists before and after
            if onset_idx >= pre_pts and onset_idx + post_pts < len(df):
                segment = df.iloc[onset_idx - pre_pts : onset_idx + post_pts]
                uws_segment = segment['y-vel'].values
                baseline = np.mean(uws_segment[:pre_pts])
                norm_uws = uws_segment - baseline
                uws_traces[key].append(norm_uws)

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
    # plt.legend(prop={'size': 10}, frameon=False, loc='upper left', bbox_to_anchor=(1.1, 1.1))
    plt.title("upwind speed at plume onset", color=markc, fontsize=16)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def dir_training_avg_plume_onset_uws(folder_path, savename, size=(6,5), baseline_subtract=True, sample_rate=10, pre=4, post=4, keywords=['no training', 'BA', 'AA'], colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    uws_avg = {kw: [] for kw in keywords}
    uws_dur_avg = {kw: [] for kw in keywords}
    params_list = exp_parameters(folder_path)
    pre_pts = int(round(pre * sample_rate))
    post_pts = int(round(post * sample_rate))
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        if 'ctrl' in filename:
            _, di, do = inside_outside(exp_df)
            first_key = list(di.keys())[0]
            df_in = di[first_key]
            key = keywords[0]
        elif 'pulse' in filename:
            _, di, do = inside_outside(df)
            first_key = list(di.keys())[1]
            df_in = di[first_key]
            key = keywords[1] 
            print('OL found')
        else: 
            plume2_df = exp_df[exp_df['exp_phase']==3]
            plume2_start = plume2_df[plume2_df['instrip']].index.min()
            plume2_df = plume2_df.loc[plume2_start:]
            _, d_in2, d_out2 = inside_outside(plume2_df)
            first_key = list(d_in2.keys())[0]
            df_in = d_in2[first_key]
            if 'mch' in filename: # if ('RR' in filename or 'LL' in filename):
                key = keywords[3]
            else: # #('LR' in filename or 'RL' in filename):
                key = keywords[2] 
            # elif 'pulse' in filename:
            #     print('OL found')
            #     key = keywords[1]     
            
        onset_time = df_in.index[0]
        # half_window = buf_pts * sample_rate
        if onset_time in df.index:
            onset_idx = df.index.get_loc(onset_time)
            if onset_idx >= pre_pts and onset_idx + post_pts < len(df):
                if baseline_subtract:
                    segment = df.iloc[onset_idx - pre_pts : onset_idx + post_pts]
                    uws_segment = segment['y-vel'].values
                    baseline = np.mean(uws_segment[:pre_pts])
                    trace = uws_segment - baseline
                else:
                    segment = df.iloc[onset_idx : onset_idx + post_pts]
                    uws_segment = segment['y-vel'].values
                    trace = uws_segment.copy()
                uws_dur = segment['seconds'].iloc[-1] - segment['seconds'].iloc[0]
                post_onset_avg = np.mean(trace)  # second half = post-onset
                uws_avg[key].append(post_onset_avg)
                # uws_dur_avg[key].append(uws_dur)
    print(uws_avg)    
    for i, key in enumerate(keywords):
        x_noise = np.random.uniform(-0.05, 0.05, len(uws_avg[key]))
        axs.scatter(np.full(len(uws_avg[key]), i) + x_noise, uws_avg[key], color=colors[i])
    avg_values = [np.mean(uws_avg[key]) if uws_avg[key] else np.nan for key in keywords]
    avg_durs = [np.mean(uws_dur_avg[key]) if uws_dur_avg[key] else np.nan for key in keywords]
    std_values = [sem(uws_avg[key]) if len(uws_avg[key]) > 1 else 0 for key in keywords]
    for i, (mean, error) in enumerate(zip(avg_values, std_values)):
        axs.hlines(mean, i - 0.1, i + 0.1, colors=markc, linewidth=2)
        axs.errorbar(i, mean, yerr=error, color=markc, capsize=0)
    # axs.hlines(y=0, xmin=0, xmax=2.5, color=markc, linestyles='--', linewidth=1)
    axs.set_ylabel("avg. upwind speed (mm/s)", color=markc, fontsize=14)
    # axs.set_title(f'avg. upwind speed\n{buf_pts}s after plume onset', color=markc, fontsize=16)
    axs.set_xticks(range(len(keywords)))
    axs.set_xticklabels(keywords, fontsize=14)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')



def dir_training_avg_plume_onset_uws_paired(folder_path, savename, size=(6,5), baseline_subtract=True, sample_rate=10, pre=4, post=4, keywords=['no training', 'BA', 'AA'], colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    params_list = exp_parameters_multiodor(folder_path)
    fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    data_dict = {}
    pre_pts = int(round(pre * sample_rate))
    post_pts = int(round(post * sample_rate))

    for this_experiment in params_list:
        if len(this_experiment) == 10:
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        elif len(this_experiment) == 11:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, bl_df, exp_df, xo, yo, plume_colors = this_experiment

        plume2_df = exp_df[exp_df['exp_phase']==3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        _, d_in2, d_out2 = inside_outside(plume2_df)
        first_key = list(d_in2.keys())[0]
        df_in = d_in2[first_key]
        if 'mch' in filename:
            key = keywords[1]
        else:
            key = keywords[0]

        match = re.search(r'Fly(\d+)', filename)
        if not match:
            continue
        fly_id = int(match.group(1))

        onset_time = df_in.index[0]
        if onset_time in df.index:
            onset_idx = df.index.get_loc(onset_time)
            if onset_idx >= pre_pts and onset_idx + post_pts < len(df):
                if baseline_subtract and pre_pts > 0:
                    segment = df.iloc[onset_idx - pre_pts : onset_idx + post_pts]
                    uws_segment = segment['y-vel'].values
                    baseline = np.mean(uws_segment[:pre_pts])
                    trace = uws_segment - baseline
                else:
                    segment = df.iloc[onset_idx : onset_idx + post_pts]
                    uws_segment = segment['y-vel'].values
                    trace = uws_segment.copy()
                post_onset_avg = np.mean(trace)
                data_dict.setdefault(fly_id, {})[key] = post_onset_avg

    data1 = []
    data2 = []
    for fly_id, data in data_dict.items():
        if keywords[0] not in data or keywords[1] not in data:
            continue  # need both conditions for a paired comparison
        axs.plot([1,2], [data[keywords[0]], data[keywords[1]]], color=markc, alpha=0.8)
        data1.append(data[keywords[0]])
        data2.append(data[keywords[1]])
    print(data1)
    print(data2)
    stat, p_value = wilcoxon(data1, data2)
    p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(data1, default=0), max(data2, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)

    axs.plot([1,2], [np.mean(data1), np.mean(data2)], color=colors[1], linewidth=3)
    sem1 = sem(data1)
    sem2 = sem(data2)
    axs.errorbar(1, np.mean(data1), yerr=sem1, color=colors[1], capsize=0, linewidth=3)
    axs.errorbar(2, np.mean(data2), yerr=sem2, color=colors[1], capsize=0, linewidth=3)

    for pos in ['right', 'top']:
        axs.spines[pos].set_visible(False)
    plt.tight_layout()
    sns.despine(offset=10)
    for _, spine in axs.spines.items():
        spine.set_linewidth(2)
    for spine in axs.spines.values():
        spine.set_edgecolor(markc)
    plt.ylabel('avg. upwind speed (mm/s)', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks([1, 2])
    axs.set_xticklabels(keywords[:2], fontsize=16, color=markc)
    axs.tick_params(axis='x', which='both', length=0)
    axs.spines['bottom'].set_visible(False)
    plt.xlim(0.7, 2.3)
    plt.tight_layout()
    plt.show()

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def dir_training_inside_outside_avg_plume_onset_uws(folder_path, savename, size=(6,5), colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    all_uws1 = []
    all_uws2 = []
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        _, d_in, d_out = inside_outside(exp_df)
        uws1 = []
        uws2 = []
        for key, df in d_in.items():
            uws = np.mean(df['y-vel'])
            if int(df['exp_phase'].iloc[-1]) == 1:
                uws1.append(uws)
            elif int(df['exp_phase'].iloc[-1]) == 3:
                uws2.append(uws)
        uws1_avg = sum(uws1) / len(uws1)
        uws2_avg = sum(uws2) / len(uws2)
        all_uws1.append(uws1_avg)
        all_uws2.append(uws2_avg)
        plt.plot([1,2], [uws1_avg, uws2_avg], color=markc)

    ctrl, exp = np.array(all_uws1), np.array(all_uws2)    
    ctrl_avg, ctrl_sem = np.mean(ctrl), sem(ctrl)
    exp_avg, exp_sem = np.mean(exp), sem(exp)
    plt.plot([1,2], [ctrl_avg, exp_avg], color=colors[1], linewidth=3)
    axs.errorbar(1, ctrl_avg, yerr=ctrl_sem, color=colors[1], capsize=0, linewidth=3)
    axs.errorbar(2, exp_avg, yerr=exp_sem, color=colors[1], capsize=0, linewidth=3)
    stat, p_value = wilcoxon(ctrl, exp)
    p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(ctrl, default=0), max(exp, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)
    style_ax(axs, markc, title=None, xtick_labels=['LED on', 'LED off'], xticks=[1,2], ylabel='avg. upwind speed', xlim=(0.7,2.3))
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def dir_training_stats(folder_path, savename, size=(6,5), colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=6)
    params_list = exp_parameters(folder_path)
    data = {metric: {'exp': [], 'ctrl': []} for metric in ['latencies', 'res', 'dists', 'eas', 'vels', 'uwes']}
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment    
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        _, di, do = inside_outside(plume1_df)    
        plume2_df = exp_df[exp_df['exp_phase']==3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        group = 'ctrl' if 'ctrl' in filename else 'exp'
        if group == 'exp':
            print(plume1_df['seconds'].iloc[-1] - plume2_df['seconds'].iloc[0])
        lats = []
        for key, df in do.items():
            lat = df['seconds'].iloc[-1] - df['seconds'].iloc[0]
            lats.append(lat)      
        lats_avg = sum(lats) / len(lats)
        data['latencies'][group].append(lats_avg)
        for k in sorted(do.keys()):  
            if k == 2: 
                duration = do[k]['seconds'].iloc[-1] - do[k]['seconds'].iloc[0]
                if group == 'exp':
                    print(f'do[{k}] duration: {duration}')

        indurs = []
        for key, df in di.items():
            lat = df['seconds'].iloc[-1] - df['seconds'].iloc[0]
            indurs.append(lat)      
        indur_avg = sum(indurs) / len(indurs)
        # if group == 'exp':
            # print(f'in: {indur_avg}')
            # print(f'out: {lats_avg}')
        
        res = []
        for key, df in do.items():
            dist = get_a_bout_calc(df, 'furthest_distance_from_plume') 
            pl = get_a_bout_calc(df, 'path_length')
            if pl > 0:
                re = dist/pl
                res.append(re)
        data['res'][group].append(sum(res) / len(res))

        dists = []
        for key, df in do.items():
            # dist = (plume1_df['ft_posx'].iloc[-1] - plume1_df['ft_posx'].iloc[0]) / 1000
            dist = (df['ft_posx'].iloc[-1] - df['ft_posx'].iloc[0]) / 1000
            if 'LL' in filename or 'LR' in filename:
                dist = -dist
            dists.append(dist * math.sqrt(2))
        # data['dists'][group].append(dist * math.sqrt(2))
        data['dists'][group].append(sum(dists) / len(dists))

        ea = []
        if len(di) > 1:
            for k in sorted(di.keys()):  
                if k == 1: 
                    entry_df = get_entry_df(None, di[k], exp_df, window=0.5, first_entry=True)
                    duration = di[k]['seconds'].iloc[-1] - di[k]['seconds'].iloc[0]
                    if group == 'exp':
                        print(f'di[{k}] duration: {duration}')
                else:  
                    entry_df = get_entry_df(do.get(k-1), di[k], exp_df, window=0.5)
                if entry_df is not None:
                    circmean = circmean_heading(entry_df, ea)
            data['eas'][group].append(mean_resultant_vector_length(ea))
            

        vels = []
        for key, df in do.items():
            vel = np.mean(df['abs_angvel'])
            vels.append(vel)
        data['vels'][group].append(sum(vels) / len(vels))

        uw_exits = 0
        if len(di) > 1:
            for key, df in di.items():
                if np.abs(df['ft_posx'].iloc[-1] - xo) < np.abs(df['ft_posy'].iloc[-1] - yo):
                    uw_exits += 1
                else:
                    print(filename)
                    print('downwind')
            data['uwes'][group].append(uw_exits/len(di))


    datasets = [
        (axs[0], 'latencies', 'reentry latency (s)', (0.5, 1.5)),
        (axs[1], 'res', 'return efficiency', (0.5, 1.5)),
        (axs[2], 'dists', 'progression (m)', (0.5, 1.5)),
        (axs[3], 'eas', 'entry angle strength', (0.5, 1.5)),
        (axs[4], 'vels', 'angular velocity (rad/s)', (0.5, 1.5)),
        (axs[5], 'uwes', 'upwind exit rate', (0.5, 1.5)),
    ]        
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = data[key]['ctrl']
        exp_vals = data[key]['exp']

        x_ctrl = np.random.normal(0.7, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(1.22, noise, size=len(exp_vals))
        ax.scatter(x_ctrl, ctrl_vals, color=colors[0], alpha=0.8)
        ax.scatter(x_exp, exp_vals, color=colors[1], alpha=0.8)

        stat, p_value = mannwhitneyu(ctrl_vals, exp_vals, alternative='two-sided')
        p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
        ax.text(1, max(max(ctrl_vals), max(exp_vals)) * 1.1,
                p_text, ha='center', va='bottom', fontsize=10, color=markc)

        mean_ctrl = np.mean(ctrl_vals) if ctrl_vals else 0
        mean_exp = np.mean(exp_vals) if exp_vals else 0
        sem_ctrl = sem(ctrl_vals) if len(ctrl_vals) > 1 else 0
        sem_exp = sem(exp_vals) if len(exp_vals) > 1 else 0
        ax.hlines(mean_ctrl, 0.6, 0.8, color=markc, linewidth=3)
        ax.hlines(mean_exp, 1.1, 1.3, color=markc, linewidth=3)
        ax.errorbar(0.7, mean_ctrl, yerr=sem_ctrl, color=markc, capsize=0, linewidth=3)
        ax.errorbar(1.2, mean_exp, yerr=sem_exp, color=markc, capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['ctrl', 'exp'], xticks=[0.7,1.2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def dir_test_stats(folder_path, savename, size=(6,5), colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, nrows=3)
    params_list = exp_parameters(folder_path)
    all_lats = {}
    all_res  = {}
    all_pls = {}
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment    
        if 'ctrl' in filename or 'LR' in filename or 'RL' in filename:
            continue
        plume2_df = exp_df[exp_df['exp_phase'] == 3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        # _, di, do = inside_outside(plume2_df)    
        _, di, do = inside_outside(exp_df)    

        exits = 0

        lats, res, pls = [], [], []
        for key, df in do.items():
            lat = df['seconds'].iloc[-1] - df['seconds'].iloc[0]     
            dist = get_a_bout_calc(df, 'furthest_distance_from_plume') 
            pl = get_a_bout_calc(df, 'path_length')
            if pl > 0:
                lats.append(lat) 
                re = dist/pl
                res.append(re)
                pls.append(pl/1000)
                exits += 1
     
        exits = np.arange(1, exits + 1)
        axs[0].scatter(exits, lats, s=10, alpha=0.6, color=colors[1])
        # axs[0].plot(exits, lats, color=colors[1], alpha=0.4, linewidth=0.8)

        axs[1].scatter(exits, res, s=10, alpha=0.6, color=colors[1])
        # axs[1].plot(exits, res, color=colors[1], alpha=0.4, linewidth=0.8)

        axs[2].scatter(exits, pls, s=10, alpha=0.6, color=colors[1])
        # axs[2].plot(exits, pls, color=colors[1], alpha=0.4, linewidth=0.8)

        for i, (lat, re, pl) in enumerate(zip(lats, res, pls), start=1):
            if i not in all_lats:
                all_lats[i] = []
                all_res[i]  = []
                all_pls[i]  = []
            all_lats[i].append(lat)
            all_res[i].append(re)
            all_pls[i].append(pl)
    
    mean_lats = {k: np.mean(v) for k, v in all_lats.items() if len(v) > 2}
    mean_res = {k: np.mean(v) for k, v in all_res.items() if len(v) > 2}   
    mean_pls = {k: np.mean(v) for k, v in all_pls.items() if len(v) > 2}   

    if mean_lats:
        axs[0].plot(list(mean_lats.keys()), list(mean_lats.values()), 
                    color=colors[1], linewidth=2, marker='o')
    if mean_res:
        axs[1].plot(list(mean_res.keys()), list(mean_res.values()), 
                    color=colors[1], linewidth=2, marker='o')
    if mean_pls:
        axs[2].plot(list(mean_pls.keys()), list(mean_pls.values()), 
                    color=colors[1], linewidth=2, marker='o')

    axs[0].set_ylabel('return latency (s)', color=markc, fontsize=12)
    axs[1].set_ylabel('return efficiency', color=markc, fontsize=12)
    axs[2].set_ylabel('pathlength (m)', color=markc, fontsize=12)

    axs[0].tick_params(axis='both', labelsize=10)
    axs[1].tick_params(axis='both', labelsize=10)      
    axs[2].tick_params(axis='both', labelsize=10)
 
   
    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def tracking_over_time(folder_path, savename, size=(6,5), colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, nrows=3)
    params_list = exp_parameters(folder_path)
    all_lats = {}
    all_res  = {}
    all_pls = {}
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment    
        _, di, do = inside_outside_odor(exp_df)    

        exits = 0

        lats, res, pls = [], [], []
        for key, df in list(do.items())[:30]:
            if return_to_edge(df) and df['seconds'].iloc[-1] - df['seconds'].iloc[0] < 200:
                lat = df['seconds'].iloc[-1] - df['seconds'].iloc[0]     
                dist = get_a_bout_calc(df, 'furthest_distance_from_plume') 
                pl = get_a_bout_calc(df, 'path_length')
                if pl > 0:
                    lats.append(lat) 
                    re = dist/pl
                    res.append(re)
                    pls.append(pl/1000)
                    exits += 1
     
        exits = np.arange(1, exits + 1)
        axs[0].scatter(exits, lats, s=10, alpha=0.3, color=colors[1])
        # axs[0].plot(exits, lats, color=colors[1], alpha=0.4, linewidth=0.8)

        axs[1].scatter(exits, res, s=10, alpha=0.3, color=colors[1])
        # axs[1].plot(exits, res, color=colors[1], alpha=0.4, linewidth=0.8)

        axs[2].scatter(exits, pls, s=10, alpha=0.3, color=colors[1])
        # axs[2].plot(exits, pls, color=colors[1], alpha=0.4, linewidth=0.8)

        for i, (lat, re, pl) in enumerate(zip(lats, res, pls), start=1):
            if i not in all_lats:
                all_lats[i] = []
                all_res[i]  = []
                all_pls[i]  = []
            all_lats[i].append(lat)
            all_res[i].append(re)
            all_pls[i].append(pl)
    
    mean_lats = {k: np.mean(v) for k, v in all_lats.items() if len(v) > 2}
    mean_res = {k: np.mean(v) for k, v in all_res.items() if len(v) > 2}   
    mean_pls = {k: np.mean(v) for k, v in all_pls.items() if len(v) > 2}   

    if mean_lats:
        axs[0].plot(list(mean_lats.keys()), list(mean_lats.values()), 
                    color=colors[0], linewidth=2, marker='o')
    if mean_res:
        axs[1].plot(list(mean_res.keys()), list(mean_res.values()), 
                    color=colors[0], linewidth=2, marker='o')
    if mean_pls:
        axs[2].plot(list(mean_pls.keys()), list(mean_pls.values()), 
                    color=colors[0], linewidth=2, marker='o')

    axs[0].set_ylabel('return latency (s)', color=markc, fontsize=12)
    axs[1].set_ylabel('return efficiency', color=markc, fontsize=12)
    axs[2].set_ylabel('pathlength (m)', color=markc, fontsize=12)

    axs[0].tick_params(axis='both', labelsize=10)
    axs[1].tick_params(axis='both', labelsize=10)      
    axs[2].tick_params(axis='both', labelsize=10)
 
    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def tracking_over_time_stats(folder_path, savename, size=(3,5), colors=['#c1ffc1', 'red']):
    # IN PROGRESS!!
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=2)
    params_list = exp_parameters(folder_path)
    data = {metric: {'on': [], 'off': []} for metric in ['res', 'eas']}
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, d_in, d_out = inside_outside_odor(exp_df)
        
        res_on = []
        res_off = []
        for key, df in d_out.items():
            dist = get_a_bout_calc(df, 'furthest_distance_from_plume') 
            pl = get_a_bout_calc(df, 'path_length')
            if pl > 0:
                re = dist/pl
                if key-1 in d_on:
                    print('light on')
                    res_on.append(re)
                else:
                    print('light off')
                    res_off.append(re)   
        if len(res_on) > 0 and len(res_off) > 0:
            data['res']['on'].append(sum(res_on)/len(res_on))
            data['res']['off'].append(sum(res_off)/len(res_off))
            axs[0].plot([0.7,1.2],[(sum(res_on)/len(res_on)), (sum(res_off)/len(res_off))], color=markc, linewidth=1, alpha=0.8)

        ea_on = []
        ea_off = []
        for k in sorted(d_in.keys()):  
            if k == 1: 
                entry_df = get_entry_df(None, d_in[k], exp_df, window=0.5, first_entry=True)
            else:  
                entry_df = get_entry_df(d_out.get(k-1), d_in[k], exp_df, window=0.5)
            if entry_df is not None:
                if k-2 in sorted(d_on.keys()):
                    circmean = circmean_heading(entry_df, ea_on)
                else:
                    circmean = circmean_heading(entry_df, ea_off)
        if len(ea_on) > 1 and len(ea_off) > 1 and len(d_out) > 1:
            print(filename)
            data['eas']['on'].append(mean_resultant_vector_length(ea_on))
            data['eas']['off'].append(mean_resultant_vector_length(ea_off))
            axs[1].plot([0.7,1.2],[(mean_resultant_vector_length(ea_on)), (mean_resultant_vector_length(ea_off))], color=markc, linewidth=1, alpha=0.8)

    datasets = [
        (axs[0], 'res', 'return efficiency', (0.5, 1.5)),
        (axs[1], 'eas', 'entry angle strength', (0.5, 1.5)),
    ]
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = data[key]['on']
        exp_vals = data[key]['off']

        x_ctrl = np.random.normal(0.7, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(1.2, noise, size=len(exp_vals))
        # ax.scatter(x_ctrl, ctrl_vals, color=colors[1], alpha=1)
        # ax.scatter(x_exp, exp_vals, color=colors[0], alpha=0.4)

        stat, p_value = wilcoxon(ctrl_vals, exp_vals, alternative='two-sided')
        p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
        ax.text(1, max(max(ctrl_vals), max(exp_vals)) * 1.1,
                p_text, ha='center', va='bottom', fontsize=10, color=markc)

        mean_ctrl = np.mean(ctrl_vals) if ctrl_vals else 0
        mean_exp = np.mean(exp_vals) if exp_vals else 0
        sem_ctrl = sem(ctrl_vals) if len(ctrl_vals) > 1 else 0
        sem_exp = sem(exp_vals) if len(exp_vals) > 1 else 0
        ax.plot([0.7,1.2], [mean_ctrl, mean_exp], color=colors[1], linewidth=3)
        # ax.hlines(mean_ctrl, 0.6, 0.8, color=markc, linewidth=3)
        # ax.hlines(mean_exp, 1.1, 1.3, color=markc, linewidth=3)
        ax.errorbar(0.7, mean_ctrl, yerr=sem_ctrl, color=colors[1], capsize=0, linewidth=3)
        ax.errorbar(1.2, mean_exp, yerr=sem_exp, color=colors[1], capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['on', 'off'], xticks=[0.7,1.2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def dir_training_progression_cor(folder_path, savename, size=(3,5), colors=['lightgrey', '#7ed4e6']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment  
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        if 'RL' in filename or 'LR' in filename or 'ctrl' in filename:
            continue
        if (exp_df['exp_phase'] == 3).any():
            plume2_df = exp_df[exp_df['exp_phase'] == 3]
            plume2_start = plume2_df[plume2_df['instrip']].index.min()
            plume2_df = plume2_df.loc[plume2_start:]
        else:
            continue
        _, di, do = inside_outside(plume1_df)

        plume1_odor = plume1_df.loc[plume1_df.mfc3_stpt > 0]
        plume2_odor = plume2_df.loc[plume2_df.mfc3_stpt > 0]

        # dist1 = np.abs((plume1_odor['ft_posx'].iloc[-1] - plume1_odor['ft_posx'].iloc[0]) / 1000) * math.sqrt(2)
        dist2 = np.abs((plume2_odor['ft_posx'].iloc[-1] - plume2_odor['ft_posx'].iloc[0]) / 1000) * math.sqrt(2)
        
        # ea = []
        # for k in sorted(di.keys()):  
        #     if k == 1: 
        #         entry_df = get_entry_df(None, di[k], exp_df, window=0.5, first_entry=True)
        #     else:  
        #         entry_df = get_entry_df(do.get(k-1), di[k], exp_df, window=0.5)
        #     if entry_df is not None:
        #         circmean = circmean_heading(entry_df, ea)
        # dist1 = (mean_resultant_vector_length(ea))

        dur = []
        for key, df in di.items():  
            dur.append(df['seconds'].iloc[-1] - df['seconds'].iloc[0])
        dist1 = sum(dur)/len(dur)

        color = colors[0] if 'ctrl' in filename else colors[1]
        plt.scatter(dist1, dist2, color=color)
        print(f'dist1: {dist1}')
        print(f'dist2: {dist2}')

    plt.xlabel('cum. reinforcement duration (s)', color=markc)
    plt.ylabel('distance tracked test  (m)', color=markc)
    plt.tight_layout()
    plt.show()

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

      
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
        print(filename)
        fig, axs, markc = configure_white_plot(size=(8,8), xaxis=True)
        _, d_on, d_off = light_on_off(df)
        _, d_in, d_out = inside_outside_oct(df)
        # print(len(d_in))
        pre_df = d_off[list(d_off.keys())[0]]
        plume_df = d_off[list(d_off.keys())[-1]]
        plume_start = plume_df[plume_df['instrip']].index.min()
        plume_df = plume_df.loc[plume_start:]
        plume_adj = 25 / np.sqrt(2)
        width=50
        height = (plume_df['ft_posx'].max() - plume_df['ft_posx'].min())  * np.sqrt(2)
        if '_L' in filename:
            angle = 45
            xplume = plume_df.iloc[0]['ft_posx'] - xo - plume_adj
            yplume = plume_df.iloc[0]['ft_posy'] - yo - plume_adj
        elif '_R' in filename:
            angle = -45
            xplume = plume_df.iloc[0]['ft_posx'] - xo - plume_adj
            yplume = plume_df.iloc[0]['ft_posy'] - yo + plume_adj

        plume = patches.Rectangle((xplume, yplume),width,height,angle=angle,facecolor=plume_color,alpha=0.5)
        axs.add_patch(plume)
        
        plt.plot(df['ft_posx'] - xo, df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        if 'ctrl' in filename:
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=plume_color, label='light on', linewidth=2)
        else:
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
        savename = filename + '_odor_trajectory_white.pdf'
        #plt.legend()
        plt.title(filename, fontsize=14, color=markc)
        # axs.set_xlim(-500,1100)
        # axs.set_ylim(-200,1400)
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

def training_1p_plotter_bw(folder_path, led, select_file=None, save=False):
    params_list = exp_parameters(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if select_file and filename != select_file:
            continue
        print(filename)
        fig, axs, markc = configure_white_plot(size=(8,8), xaxis=True)
        _, d_in, d_out = inside_outside_oct(df)
        plume_df = d_in[list(d_in.keys())[1]]
        plume_start = plume_df[plume_df['instrip']].index.min()
        plume_df = exp_df.loc[plume_start:]
        plume_adj = 25 / np.sqrt(2)
        width=50
        height = (plume_df['ft_posx'].max() - plume_df['ft_posx'].min())  * np.sqrt(2)
        if '_L' in filename:
            angle = 45
            xplume = plume_df.iloc[0]['ft_posx'] - xo - plume_adj
            yplume = plume_df.iloc[0]['ft_posy'] - yo - plume_adj
        elif '_R' in filename:
            angle = -45
            xplume = plume_df.iloc[0]['ft_posx'] - xo - plume_adj
            yplume = plume_df.iloc[0]['ft_posy'] - yo + plume_adj

        plume = patches.Rectangle((xplume, yplume),width,height,angle=angle,facecolor=plume_color,alpha=0.5)
        axs.add_patch(plume)
        
        plt.plot(df['ft_posx'] - xo, df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        if 'ctrl' in filename:
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=plume_color, label='light on', linewidth=2)
        else:
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
        savename = filename + '_odor_trajectory_white.pdf'
        plt.title(filename, fontsize=14, color=markc)
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
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
        if save:
            plt.savefig(os.path.join(figure_folder, savename))
        else:
            plt.show()


def trajectory_plotter_45_bw(folder_path, led, select_file=None, save=False):
    params_list = exp_parameters(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if select_file and filename != select_file:
            continue
        print(filename)
        fig, axs, markc = configure_white_plot(size=(8,8), xaxis=True)
        plume_adj = 25 / np.sqrt(2)
        width=50
        height = (exp_df['ft_posx'].max() - exp_df['ft_posx'].min())  * np.sqrt(2)
        if '_L' in filename:
            angle = 45
            xplume = exp_df.iloc[0]['ft_posx'] - xo - plume_adj
            yplume = exp_df.iloc[0]['ft_posy'] - yo - plume_adj
        elif '_R' in filename:
            angle = -45
            xplume = exp_df.iloc[0]['ft_posx'] - xo - plume_adj
            yplume = exp_df.iloc[0]['ft_posy'] - yo + plume_adj

        plume = patches.Rectangle((xplume, yplume),width,1000,angle=angle,facecolor=plume_color,alpha=0.5)
        axs.add_patch(plume)
        
        plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
       
        savename = filename + '_trajectory.pdf'
        plt.title(filename, fontsize=14, color=markc)
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
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
    print(len(data1))
    print(len(data2))

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

def oct_training_return_efficiency_ctrl(folder_path, savename, groups=4, size=(3,5), labels=['L ctrl', 'L exp', 'R ctrl', 'R exp'], colors=['lightgrey', '#7ed4e6'], alphas=[0.5, 0.5, 1, 1]):
    params_list = exp_parameters(folder_path)
    returns = {f'returns_r{i+1}': [] for i in range(groups)}
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, d_on, d_off = light_on_off(df)
        _, d_in, d_out = inside_outside(df)
        plume_df = d_off[list(d_off.keys())[-1]]
        plume_start = plume_df[plume_df['instrip']].index.min()
        plume_df = plume_df.loc[plume_start:]
        _, d_in, d_out = inside_outside(plume_df)
        pl = get_a_bout_calc(plume_df, 'path_length') / 1000
        count = 0
        for key, bout_df in d_out.items():
            if bout_df['seconds'].iloc[-1] - bout_df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(bout_df, angle=-45):
                count += 1
        efficiency = count / pl if pl else 0
        if 'LR' in filename and 'ctrl' in filename:
            group_index = 0  # LR_ctrl
        elif 'LR' in filename:
            group_index = 1  # LR
        elif 'RR' in filename and 'ctrl' in filename:
            group_index = 2  # RR_ctrl
        elif 'RR' in filename:
            group_index = 3  # RR
        else:
            continue  # Skip unclassified files

        returns[f'returns_r{group_index+1}'].append(efficiency)

    averages = [np.mean(returns[f'returns_r{i+1}']) if returns[f'returns_r{i+1}'] else 0 for i in range(groups)]
    std_devs = [np.std(returns[f'returns_r{i+1}']) if returns[f'returns_r{i+1}'] else 0 for i in range(groups)]
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(returns[f'returns_r{i+1}'])) for i in range(groups)]
    for i in range(groups):
        plt.scatter(x_values[i], returns[f'returns_r{i+1}'], color=colors[i], alpha=alphas[i], label=labels[i])
        plt.hlines(averages[i], xmin=i+0.9, xmax=i+1.1, colors=markc, linewidth=2)
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
    axs.set_xticklabels(labels, fontsize=16, color=markc, rotation=45)
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    for i in range(groups):
        group_name = labels[i]
        return_list = returns[f'returns_r{i+1}']
        print(f"\nGroup: {group_name} (n={len(return_list)})")
        print(return_list)
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
        sem_mch = sem(isis_mch, nan_policy='omit')
        sem_oct = sem(isis_oct, nan_policy='omit')
        noise = 0.05
        x_mch = np.random.normal(1, noise, size=len(isis_mch))
        x_oct = np.random.normal(2, noise, size=len(isis_oct))
        plt.scatter(x_mch, isis_mch, color=colors[0], alpha=0.8)
        plt.scatter(x_oct, isis_oct, color=colors[1], alpha=0.8)
        plt.hlines(avg_mch, xmin=0.9, xmax=1.1, colors=markc, linewidth=2)  # Mean line
        plt.hlines(avg_oct, xmin=1.9, xmax=2.1, colors=markc, linewidth=2)  # Mean line
        plt.errorbar(1, avg_mch, yerr=sem_mch, color=markc, capsize=0)
        plt.errorbar(2, avg_oct, yerr=sem_oct, color=markc, capsize=0)
    else:
        avg_LR, std_LR = np.mean(isis_LR), np.std(isis_LR)
        avg_RR, std_RR = np.mean(isis_RR), np.std(isis_RR)
        avg_all, std_all = np.mean(all_isis), np.std(all_isis)
        print(avg_all)
        print(std_all)
        sem_LR = sem(isis_LR, nan_policy='omit')
        sem_RR = sem(isis_RR, nan_policy='omit')
        noise = 0.05
        x_LR = np.random.normal(1, noise, size=len(isis_LR))
        x_RR = np.random.normal(2, noise, size=len(isis_RR))
        plt.scatter(x_LR, isis_LR, color=colors[0], alpha=0.8)
        plt.scatter(x_RR, isis_RR, color=colors[1], alpha=0.8)
        plt.hlines(avg_LR, xmin=0.9, xmax=1.1, colors=markc, linewidth=2)  # Mean line
        plt.hlines(avg_RR, xmin=1.9, xmax=2.1, colors=markc, linewidth=2)  # Mean line
        plt.errorbar(1, avg_LR, yerr=sem_LR, color=markc, capsize=0)
        plt.errorbar(2, avg_RR, yerr=sem_RR,  color=markc, capsize=0)
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
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
            
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
        summary_p1_vector = np.mean(np.column_stack((np.cos(p1_means), np.sin(p1_means))), axis=0)
        summary_p1_rho, summary_p1_phi = cart2pol(summary_p1_vector[0], summary_p1_vector[1])
        plt.polar([0, summary_p1_phi], [0, summary_p1_rho], color=colors[0], alpha=1, linewidth=3, label=label1, solid_capstyle='round')
    if len(p2_means) > 0:
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

def OL_training_angles(folder_path, savename, size=(5,5), colors=['lightgrey', '#ff355e']):
    fig, axs, markc = configure_white_polar(size=size, grid=False)
    params_list = exp_parameters(folder_path)
    angles_entry = []
    angles_exit = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, d_in, d_out = inside_outside_oct(df)
        _, d_on, d_off = light_on_off(df)
        plume_df = d_off[list(d_off.keys())[-1]]
        plume_start = plume_df[plume_df['instrip']].index.min()
        pre_df = df.loc[:plume_start]
        _, d_on, d_off = light_on_off(pre_df)
        d_entry = dict(list(d_off.items())[1:])
        d_exit = d_on
        print(len(d_entry))
        print(len(d_exit))
      
        for key, df in list(d_entry.items())[:-1]:
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                df = get_last_second(df)
                circmean = circmean_heading(df, angles_entry)  # Appends inside function
                p1x, p1y = pol2cart(1, circmean)  # Convert to Cartesian
                plt.polar([0, circmean], [0, 1], color=colors[0], alpha=0.6, linewidth=1, solid_capstyle='round')  # Old ray plot
        for key, df in list(d_exit.items())[:-1]:
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                df = get_last_second(df)
                circmean = circmean_heading(df, angles_exit)  # Appends inside function
                p1x, p1y = pol2cart(1, circmean)  # Convert to Cartesian
                # plt.polar([0, circmean], [0, 1], color=colors[1], alpha=0.6, linewidth=1, solid_capstyle='round')  # Old ray plot
    p1_means = np.array(angles_entry)
    p2_means = np.array(angles_exit)
    label1 = 'entry'
    label2 = 'exit'
    if len(p1_means) > 0:
        summary_p1_vector = np.mean(np.column_stack((np.cos(p1_means), np.sin(p1_means))), axis=0)
        summary_p1_rho, summary_p1_phi = cart2pol(summary_p1_vector[0], summary_p1_vector[1])
        plt.polar([0, summary_p1_phi], [0, summary_p1_rho], color=colors[0], alpha=1, linewidth=3, label=label1, solid_capstyle='round')
    if len(p2_means) > 0:
        summary_p2_vector = np.mean(np.column_stack((np.cos(p2_means), np.sin(p2_means))), axis=0)
        summary_p2_rho, summary_p2_phi = cart2pol(summary_p2_vector[0], summary_p2_vector[1])
        # plt.polar([0, summary_p2_phi], [0, summary_p2_rho], color=colors[1], alpha=1, linewidth=3, label=label2, solid_capstyle='round')
    plt.title(f'training pulse entry/exit angles', color=markc, fontsize=18)
    plt.legend(prop={'size': 10}, frameon=False, loc='upper left', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def dual_odor_training_plotter(folder_path, led, select_file=None):
    params_list = exp_parameters_multiodor(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_white_plot(size=(10,10), xaxis=True)
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        plume2_df = exp_df[exp_df['exp_phase']==3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        plume3_df = exp_df[exp_df['exp_phase']==5]
        plume3_start = plume3_df[plume3_df['instrip']].index.min()
        plume3_df = plume3_df.loc[plume3_start:]
    
        train1 = exp_df[exp_df['exp_phase']==2]
        train2 = exp_df[exp_df['exp_phase']==4]
        print(filename)
        print(train1.iloc[-1]['seconds'] - train1.iloc[0]['seconds'])
        print(train2.iloc[-1]['seconds'] - train2.iloc[0]['seconds'])

        plume_adj = 25 / np.sqrt(2)
        width=50

        mask1 = plume1_df['instrip'].fillna(False)
        height1 = (plume1_df.loc[mask1, 'ft_posx'].max() - plume1_df.loc[mask1, 'ft_posx'].min()) * np.sqrt(2) + 30
        mask2 = plume2_df['instrip'].fillna(False)
        height2 = (plume2_df.loc[mask2, 'ft_posx'].max() - plume2_df.loc[mask2, 'ft_posx'].min()) * np.sqrt(2) + 30
        mask3 = plume3_df['instrip'].fillna(False)
        height3 = (plume3_df.loc[mask3, 'ft_posx'].max() - plume3_df.loc[mask3, 'ft_posx'].min()) * np.sqrt(2) + 30
        angle1 = 45
        angle2 = -45
        if 'Rtest' in filename:
            angle3 = -45
            x3 = plume3_df.iloc[0]['ft_posx'] - xo - plume_adj
            y3 = plume3_df.iloc[0]['ft_posy'] - yo + plume_adj
        elif 'Ltest' in filename:
            angle3 = 45
            x3 = plume3_df.iloc[0]['ft_posx'] - xo - plume_adj
            y3 = plume3_df.iloc[0]['ft_posy'] - yo - plume_adj
        
        x1 = plume1_df.iloc[0]['ft_posx'] - xo - plume_adj
        y1 = plume1_df.iloc[0]['ft_posy'] - yo - plume_adj
        x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
        y2 = plume2_df.iloc[0]['ft_posy'] - yo + plume_adj
            
        plume1 = patches.Rectangle((x1, y1),width,height1,angle=angle1,facecolor=plume_colors[1],alpha=0.5)
        axs.add_patch(plume1)
        plume2 = patches.Rectangle((x2, y2),width,height2,angle=angle2,facecolor=plume_colors[0],alpha=0.5)
        axs.add_patch(plume2)
        plume3 = patches.Rectangle((x3, y3),width,height3,angle=angle3,facecolor=plume_colors[1],alpha=0.5)
        axs.add_patch(plume3)
        
        plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        # plt.plot(plume1_df['ft_posx'] - xo, plume1_df['ft_posy'] - yo, color='pink', label='clean air', linewidth=2)
        # plt.plot(plume2_df['ft_posx'] - xo, plume2_df['ft_posy'] - yo, color='blue', label='clean air', linewidth=2)
        # plt.plot(plume3_df['ft_posx'] - xo, plume3_df['ft_posy'] - yo, color='green', label='clean air', linewidth=2)
        plt.plot(df_odor1['ft_posx'] - xo, df_odor1['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        plt.plot(df_odor2['ft_posx'] - xo, df_odor2['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=2)
        plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=2)
       
        savename = filename + '_odor_trajectory.pdf'
        plt.title(filename, fontsize=14, color=markc)
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
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
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, savename))


def two_odor_training_plotter(folder_path, led, select_file=None):
    params_list = exp_parameters_multiodor(folder_path)
    if led == 'none':
        ledc = 'grey'
    elif led == 'red':
        ledc = '#ff355e'
    elif led == 'green':
        ledc = '#0bda51'
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_white_plot(size=(10,10), xaxis=True)
        plume1_df = exp_df[exp_df['exp_phase']==1]
        plume1_start = plume1_df[plume1_df['instrip']].index.min()
        plume1_df = plume1_df.loc[plume1_start:]
        
        plume2_df = exp_df[exp_df['exp_phase']==3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        plume_adj = 25 / np.sqrt(2)
        ## find transition point in plume 1 and ID odor
        first_row1 = plume1_df.iloc[0]
        if first_row1['mfc2_stpt'] > 0:
            plume_color1a = plume_colors[0]
            plume_color1b = plume_colors[1]
            first_mfc = 'mfc2'  
        else:
            plume_color1a = plume_colors[1]
            plume_color1b = plume_colors[0]
            first_mfc = 'mfc3'
        print(first_mfc)
        ## find second plume color
        first_row2 = plume2_df.iloc[0]
        if first_row2['mfc2_stpt'] > 0:
            plume_color2 = plume_colors[0]
        else:
            plume_color2 = plume_colors[1]
          
        transition_idx = (plume1_df.loc[(plume1_df['instrip']) & (plume1_df[f'{first_mfc}_stpt'] > 0)].index.max())
        plume1a_df = plume1_df.loc[:transition_idx]
        plume1b_df = plume1_df.loc[transition_idx+1:]

        width=50
        mask1a = plume1a_df['instrip'].fillna(False)
        height1a = (plume1a_df.loc[mask1a, 'ft_posx'].max() - plume1a_df.loc[mask1a, 'ft_posx'].min()) * np.sqrt(2) 
        mask1b = plume1b_df['instrip'].fillna(False)
        height1b = (plume1b_df.loc[mask1b, 'ft_posx'].max() - plume1b_df.loc[mask1b, 'ft_posx'].min()) * np.sqrt(2) 
        mask2 = plume2_df['instrip'].fillna(False)
        height2 = (plume2_df.loc[mask2, 'ft_posx'].max() - plume2_df.loc[mask2, 'ft_posx'].min()) * np.sqrt(2) + 30

        print(len(plume1a_df))
        print(len(plume1b_df))

        if 'RR' in filename:
            angle1a = -45
            angle1b = -45
            angle2 = -45
            x1a = plume1a_df.iloc[0]['ft_posx'] - xo - plume_adj
            y1a = plume1a_df.iloc[0]['ft_posy'] - yo + plume_adj
            x1b = plume1b_df.iloc[0]['ft_posx'] - xo - plume_adj
            y1b = plume1b_df.iloc[0]['ft_posy'] - yo + plume_adj
            x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
            y2 = plume2_df.iloc[0]['ft_posy'] - yo + plume_adj
        
        elif 'LL' in filename:
            angle1a = 45
            angle1b = 45
            angle2 = 45
            x1a = plume1a_df.iloc[0]['ft_posx'] - xo - plume_adj
            y1a = plume1a_df.iloc[0]['ft_posy'] - yo - plume_adj
            x1b = plume1b_df.iloc[0]['ft_posx'] - xo - plume_adj
            y1b = plume1b_df.iloc[0]['ft_posy'] - yo - plume_adj
            x2 = plume2_df.iloc[0]['ft_posx'] - xo - plume_adj
            y2 = plume2_df.iloc[0]['ft_posy'] - yo - plume_adj
        
 
        # plume1a = patches.Rectangle((x1a, y1a),width,height1a,angle=angle1a,facecolor=plume_color1a,alpha=0.5)
        # axs.add_patch(plume1a)
        # plume1b = patches.Rectangle((x1b, y1b),width,height1b,angle=angle1b,facecolor=plume_color1b,alpha=0.5)
        # axs.add_patch(plume1b)
        plume2 = patches.Rectangle((x2, y2),width,height2,angle=angle2,facecolor=plume_color2,alpha=0.5)
        axs.add_patch(plume2)
        
        plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air', linewidth=2)
        plt.plot(df_odor1['ft_posx'] - xo, df_odor1['ft_posy'] - yo, color=plume_colors[0], label='odor only', linewidth=2)
        # plt.plot(df_odor2['ft_posx'] - xo, df_odor2['ft_posy'] - yo, color=plume_colors[1], label='odor only', linewidth=2)       
        
        savename = filename + '_odor_trajectory.pdf'
        plt.title(filename, fontsize=14, color=markc)
        axs.set_xlabel('x-position (mm)', fontsize=14)
        axs.set_ylabel('y-position (mm)', fontsize=14)
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
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, savename))


def multiodor_dir_test_stats(folder_path, savename, size=(6,5), colors=['lightgrey', '#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, nrows=3)
    params_list = exp_parameters(folder_path)
    all_lats = {}
    all_res  = {}
    all_pls = {}
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment    
        if len(this_experiment) == 9:
            multi = False
            figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        elif len(this_experiment) == 10:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
            multi = True
        plume2_df = exp_df[exp_df['exp_phase'] == 3]
        plume2_start = plume2_df[plume2_df['instrip']].index.min()
        plume2_df = plume2_df.loc[plume2_start:]
        _, di, do = inside_outside(plume2_df)    

        # EDIT THIS
        lats = []
        for key, df in do.items():
            lat = df['seconds'].iloc[-1] - df['seconds'].iloc[0]
            lats.append(lat)      
        lats_avg = sum(lats) / len(lats)
        data['latencies'][group].append(lats_avg)

        res = []
        for key, df in do.items():
            dist = get_a_bout_calc(df, 'furthest_distance_from_plume') 
            pl = get_a_bout_calc(df, 'path_length')
            if pl > 0:
                re = dist/pl
                res.append(re)
        data['res'][group].append(sum(res) / len(res))

        dists = []
        for key, df in do.items():
            # dist = (plume1_df['ft_posx'].iloc[-1] - plume1_df['ft_posx'].iloc[0]) / 1000
            dist = (df['ft_posx'].iloc[-1] - df['ft_posx'].iloc[0]) / 1000
            if 'LL' in filename or 'LR' in filename:
                dist = -dist
            dists.append(dist * math.sqrt(2))
        # data['dists'][group].append(dist * math.sqrt(2))
        data['dists'][group].append(sum(dists) / len(dists))

        ea = []
        if len(di) > 1:
            for k in sorted(di.keys()):  
                if k == 1: 
                    entry_df = get_entry_df(None, di[k], exp_df, window=0.5, first_entry=True)
                else:  
                    entry_df = get_entry_df(do.get(k-1), di[k], exp_df, window=0.5)
                if entry_df is not None:
                    circmean = circmean_heading(entry_df, ea)
            data['eas'][group].append(mean_resultant_vector_length(ea))

        vels = []
        for key, df in do.items():
            vel = np.mean(df['abs_angvel'])
            vels.append(vel)
        data['vels'][group].append(sum(vels) / len(vels))

        uw_exits = 0
        if len(di) > 1:
            for key, df in di.items():
                if np.abs(df['ft_posx'].iloc[-1] - xo) < np.abs(df['ft_posy'].iloc[-1] - yo):
                    uw_exits += 1
                else:
                    print(filename)
                    print('downwind')
            data['uwes'][group].append(uw_exits/len(di))


    datasets = [
        (axs[0], 'latencies', 'reentry latency (s)', (0.5, 1.5)),
        (axs[1], 'res', 'return efficiency', (0.5, 1.5)),
        (axs[2], 'dists', 'progression (m)', (0.5, 1.5)),
        (axs[3], 'eas', 'entry angle strength', (0.5, 1.5)),
        (axs[4], 'vels', 'angular velocity (rad/s)', (0.5, 1.5)),
        (axs[5], 'uwes', 'upwind exit rate', (0.5, 1.5)),
    ]        
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = data[key]['ctrl']
        exp_vals = data[key]['exp']

        x_ctrl = np.random.normal(0.7, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(1.22, noise, size=len(exp_vals))
        ax.scatter(x_ctrl, ctrl_vals, color=colors[0], alpha=0.8)
        ax.scatter(x_exp, exp_vals, color=colors[1], alpha=0.8)

        stat, p_value = mannwhitneyu(ctrl_vals, exp_vals, alternative='two-sided')
        p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
        ax.text(1, max(max(ctrl_vals), max(exp_vals)) * 1.1,
                p_text, ha='center', va='bottom', fontsize=10, color=markc)

        mean_ctrl = np.mean(ctrl_vals) if ctrl_vals else 0
        mean_exp = np.mean(exp_vals) if exp_vals else 0
        sem_ctrl = sem(ctrl_vals) if len(ctrl_vals) > 1 else 0
        sem_exp = sem(exp_vals) if len(exp_vals) > 1 else 0
        ax.hlines(mean_ctrl, 0.6, 0.8, color=markc, linewidth=3)
        ax.hlines(mean_exp, 1.1, 1.3, color=markc, linewidth=3)
        ax.errorbar(0.7, mean_ctrl, yerr=sem_ctrl, color=markc, capsize=0, linewidth=3)
        ax.errorbar(1.2, mean_exp, yerr=sem_exp, color=markc, capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['ctrl', 'exp'], xticks=[0.7,1.2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
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
            df['exp_phase'] = df['exp_phase'].fillna(0)
            change_points = df['exp_phase'].ne(df['exp_phase'].shift())
            change_indices = df.index[change_points].tolist()
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
        # plt.plot([0.5,1.5], [returns1/pl1, returns2/pl2], color=colors[0], zorder=0)
    noise = 0.05
    plt.scatter(0.5 + np.random.normal(0, noise, len(re1)), re1, color=colors[0])
    plt.scatter(1.2 + np.random.normal(0, noise, len(re2c)), re2c, color=colors[1])
    plt.scatter(1.9 + np.random.normal(0, noise, len(re2)), re2, color=colors[2])
    # plt.scatter([0.5] * len(re1), re1, color=colors[0])
    # plt.scatter([1.5] * len(re2c), re2c, color=colors[1], edgecolor=colors[0])
    print(re1)
    print(re2c)
    print(re2)
    avg_re1, std_re1 = np.mean(re1), sem(re1)
    avg_re2c, std_re2c = np.mean(re2c), sem(re2c)
    avg_re2, std_re2 = np.mean(re2), sem(re2)
    plt.hlines(avg_re1, 0.4, 0.6, colors=markc, linewidth=3)
    plt.hlines(avg_re2c, 1.1, 1.3, colors=markc, linewidth=3)
    plt.hlines(avg_re2, 1.8, 2.0, colors=markc, linewidth=3)
    # plt.plot([0.5,1.5], [avg_re1, avg_re2], color=markc, linewidth=3)
    plt.errorbar(0.5, avg_re1, yerr=std_re1, color=markc, capsize=0, linewidth=3)
    plt.errorbar(1.2, avg_re2c, yerr=std_re2c, color=markc, capsize=0, linewidth=3)
    plt.errorbar(1.5, avg_re2, yerr=std_re2, color=markc, capsize=0, linewidth=3)
    plt.ylabel('returns per meter', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    # axs.set_xticks([1,1.5,2.5])
    # axs.set_xticklabels(['plume 1', 'plume 2 (LED off)', 'plume 2 (LED on)'], fontsize=16, color=markc, rotation=45)
    axs.tick_params(axis='x', length=0)
    # plt.xlim(0.5, 3.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def zigzag_return_velocity(folder_path, savename, size=(3,5), colors=['lightgrey', '#ff355e']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    v1_avg = []
    v2_avg = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        if 'ctrl' in filename:
            continue 
        print(filename)
        _, d_on, d_off = light_on_off(exp_df)
        plume1_df = d_off[list(d_off.keys())[0]]
        plume2_df = d_off[list(d_off.keys())[-1]]
        _, d_in1, d_out1 = inside_outside_acv(plume1_df)
        _, d_in2, d_out2 = inside_outside_acv(plume2_df)

        if 'left' in filename:
            angle1 = 45
            angle2 = -45
        if 'RR' in filename:
            angle1 = -45
            angle2 = -45
        else:
            angle1 = -45
            angle2 = 45

        v1 = []
        v2 = []
        for key, df in d_out1.items():
            if return_to_edge_angle(df, angle1):
                start_pos = df['ft_posx'].iloc[0]
                dist_from_start = (df['ft_posx'] - start_pos).abs()
                idx_far = dist_from_start.idxmax()
                df_inbound = df.loc[idx_far:]
                avg_inbound_speed = df_inbound['speed'].mean()      
                # avg_inbound_speed = df['speed'].mean()                
                v1.append(avg_inbound_speed)

        for key, df in d_out2.items():
            if return_to_edge_angle(df, angle2):
                start_pos = df['ft_posx'].iloc[0]
                dist_from_start = (df['ft_posx'] - start_pos).abs()
                idx_far = dist_from_start.idxmax()
                df_inbound = df.loc[idx_far:]
                avg_inbound_speed = df_inbound['speed'].mean()  
                # avg_inbound_speed = df['speed'].mean()  
                v2.append(avg_inbound_speed)
    
        v1_avg.append(sum(v1) / len(v1))
        v2_avg.append(sum(v2) / len(v2))

    plt.scatter([1] * len(v1_avg), v1_avg, color=colors[0])
    plt.scatter([2] * len(v2_avg), v2_avg, color=colors[1], edgecolor=colors[0])
    plt.plot([1,2], [v1_avg, v2_avg], color= colors[0], zorder=0)

    avg_v1, std_v1 = np.mean(v1_avg), sem(v1_avg)
    avg_v2, std_v2 = np.mean(v2_avg), sem(v2_avg)
    stat, p_value = wilcoxon(v1_avg, v2_avg)
    p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(v1_avg, default=0), max(v2_avg, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)
    plt.hlines(avg_v1, 0.9, 1.1, colors=markc, linewidth=3)
    plt.hlines(avg_v2, 1.9, 2.1, colors=markc, linewidth=3)
    plt.errorbar(1, avg_v1, yerr=std_v1, color=markc, capsize=0, linewidth=3)
    plt.errorbar(2, avg_v2, yerr=std_v2, color=markc, capsize=0, linewidth=3)
    plt.ylabel('return speed', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    axs.tick_params(axis='x', length=0)
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
    print(re2c)
    print(re2)
    re2c_mean = np.mean(re2c)
    re2c_sem = np.std(re2c) / np.sqrt(len(re2c))
    re2_mean = np.mean(re2)
    re2_sem = np.std(re2) / np.sqrt(len(re2))
    axs.axhline(1, color=markc, linestyle='--', linewidth=1)
    plt.scatter(1 + np.random.normal(0, 0.05, len(re2c)), re2c, color=colors[0])
    plt.scatter(2 + np.random.normal(0, 0.05, len(re2)), re2, color=colors[1])
    plt.ylabel('plume 1: plume 2\nreturns per meter', fontsize=18, color=markc)
    plt.hlines(re2c_mean, 0.9, 1.1, colors=markc, linewidth=3)
    plt.hlines(re2_mean, 1.9, 2.1, colors=markc, linewidth=3)
    plt.errorbar(1, re2c_mean, yerr=re2c_sem, color=markc, capsize=0, linewidth=3)
    plt.errorbar(2, re2_mean, yerr=re2_sem, color=markc, capsize=0, linewidth=3)
    plt.yticks(fontsize=14, color=markc)
    axs.set_xticks([1, 2])
    # axs.set_xticklabels(['LED off', 'LED on'], fontsize=18, color=markc, rotation=45)
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


def zigzag_alt_return_efficiency_paired(folder_path, savename, size=(3,5), colors=['lightgrey', '#ff355e']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False)
    re1_list, re2_list = [], []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        _, d_on, d_off = light_on_off(exp_df)
        plume1_df = d_off[list(d_off.keys())[0]]
        plume2_df = d_off[list(d_off.keys())[-1]]
        _, d_in1, d_out1 = inside_outside_acv(plume1_df)
        _, d_in2, d_out2 = inside_outside_acv(plume2_df)
        re1s = []
        re2s = []
        for key, df in d_out1.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(df, angle=-45):
                dist1 = get_a_bout_calc(df, 'furthest_distance_from_plume')
                pl1 = get_a_bout_calc(df, 'path_length')
                re1_bout = dist1/pl1
                re1s.append(re1_bout)
        for key, df in d_out2.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge_angle(df, angle=45):
                dist2 = get_a_bout_calc(df, 'furthest_distance_from_plume')
                pl2 = get_a_bout_calc(df, 'path_length')
                re2_bout = dist2/pl2
                re2s.append(re2_bout)
        re1 = sum(re1s) / len(re1s)
        re2 = sum(re2s) / len(re2s)
        re1_list.append(re1)
        re2_list.append(re2)
        plt.plot([1,2], [re1, re2], color=colors[0], zorder=20)
    plt.scatter([1] * len(re1_list), re1_list, color=colors[0], zorder=20)
    plt.scatter([2] * len(re2_list), re2_list, color=colors[1], zorder=20)
    re1_array, re2_array = np.array(re1_list), np.array(re2_list)
    avg1, sem1 = np.mean(re1_array), sem(re1_array)
    avg2, sem2 = np.mean(re2_array), sem(re2_array)
    stat, p_value = wilcoxon(re1_array, re2_array)
    p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(re1_array, default=0), max(re2_array, default=0)) * 1.1, p_text,
         ha='center', va='bottom', fontsize=14, color=markc)
    plt.ylabel('return efficiency', fontsize=18, color=markc)
    plt.yticks(fontsize=14, color=markc)
    plt.xticks(fontsize=14, color=markc)
    axs.tick_params(axis='x', length=0)         
    axs.set_xticklabels([])   
    plt.xlim(0.5, 2.5)
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
        _, d_on, d_off = light_on_off(exp_df)
        plume1_df = d_off[list(d_off.keys())[0]]
        plume2_df = d_off[list(d_off.keys())[-1]]
        _, d_in1, d_out1 = inside_outside_acv(plume1_df)
        _, d_in2, d_out2 = inside_outside_acv(plume2_df)
        plume1_bout = []
        plume2_bout = []
        if 'ctrl' in filename:
            print(filename)
            print(len(d_out2))
            if len(d_out2) <= 3:
                all_plume1_bout.append(0)
            else:
                for key, df in d_out2.items():
                    if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                        df = get_last_second(df)
                        p1_circmean = circmean_heading(df, plume1_bout)
                print(mean_resultant_vector_length(plume1_bout))
                all_plume1_bout.append(mean_resultant_vector_length(plume1_bout))

        elif 'ctrl' not in filename:
            for key, df in d_out2.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                    df = get_last_second(df)
                    p2_circmean = circmean_heading(df, plume2_bout)
            # plt.plot([1,2], [(mean_resultant_vector_length(plume1_bout)),(mean_resultant_vector_length(plume2_bout))], color=colors[0], zorder=0)
            all_plume2_bout.append(mean_resultant_vector_length(plume2_bout))
    noise = 0.05
    x_1 = np.random.normal(1, noise, size=len(all_plume1_bout))
    x_2 = np.random.normal(2, noise, size=len(all_plume2_bout))
    plt.scatter(x_1, all_plume1_bout, color=colors[0])
    plt.scatter(x_2, all_plume2_bout, color=colors[1])
    avg_p1, std_p1 = np.mean(all_plume1_bout), np.std(all_plume1_bout)
    avg_p2, std_p2 = np.mean(all_plume2_bout), np.std(all_plume2_bout)
    plt.hlines(avg_p1, 0.9, 1.1, colors=markc, linewidth=2)
    plt.hlines(avg_p2, 1.9, 2.1, colors=markc, linewidth=2)
    # stat, p_value = wilcoxon(all_plume1_bout, all_plume2_bout)
    stat, p_value = mannwhitneyu(all_plume1_bout, all_plume2_bout, alternative='two-sided')
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


def speed_vs_heading_diff(folder_path, savename, size=(5, 4), grp='ctrl', colors=['lightgrey', '#ff355e'],bin_width_deg=15, speed_pct=20):
    """
    compute the per-fly entry angle (circular mean of
    headings in the last second of each outside-bout) for plume 1 and plume 2.
    Then, using outside-of-odor frames where speed > per-fly 20th percentile,
    plot mean forward speed vs (heading - entry_angle) wrapped to [-180, 180].
    Mean +/- SEM across flies, separately for plume 1 and plume 2.
    """
    bin_edges = np.arange(-180, 180 + bin_width_deg, bin_width_deg)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    exp_curves = []
    ctrl_curves = []

    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_white_plot(size=size, xaxis=True)

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, d_in, d_out = inside_outside(exp_df)
        bout_headings, mean_heading, _ = compute_mean_entry_heading(d_out, imaging=False)
        entry = circmean(bout_headings) if len(bout_headings) else np.nan
        str = f'{np.rad2deg(entry):.1f}°' if not np.isnan(entry) else 'n/a'
        print(f'{filename}  |  entry angle: {str}')

        def fly_curve(d_out, entry_angle_rad):
            if np.isnan(entry_angle_rad) or len(d_out) == 0:
                return None

            return_segments = []
            for _, bout_df in d_out.items():
                if len(bout_df) < 2:
                    continue

                # perpendicular distance from the start->end line, per frame
                # (matches get_a_bout_calc(..., 'furthest_distance_from_plume') logic)
                x1, y1 = bout_df['ft_posx'].iloc[0], bout_df['ft_posy'].iloc[0]
                x2, y2 = bout_df['ft_posx'].iloc[-1], bout_df['ft_posy'].iloc[-1]
                denom = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if denom == 0:
                    continue
                distances = np.abs(
                    (y2 - y1) * bout_df['ft_posx']
                    - (x2 - x1) * bout_df['ft_posy']
                    + x2 * y1 - y2 * x1
                ) / denom

                far_idx = int(np.argmax(distances.values))
                return_leg = bout_df.iloc[far_idx:]
                if len(return_leg) >= 2:
                    return_segments.append(return_leg)

            if len(return_segments) == 0:
                return None

            outside = pd.concat(return_segments, ignore_index=True)
            if 'speed' not in outside.columns or len(outside) == 0:
                return None

            # per-fly 20th percentile filter on the return-leg frames
            thresh = np.nanpercentile(outside['speed'].values, speed_pct)
            outside = outside[outside['speed'] > thresh]
            if len(outside) == 0:
                return None

            # heading diff wrapped to [-180, 180]
            diff_rad = np.angle(np.exp(1j * (outside['ft_heading'].values - entry_angle_rad)))
            diff_deg = np.rad2deg(diff_rad)
            speeds = outside['speed'].values

            curve = np.full(n_bins, np.nan)
            for i in range(n_bins):
                mask = (diff_deg >= bin_edges[i]) & (diff_deg < bin_edges[i + 1])
                if mask.any():
                    curve[i] = np.mean(speeds[mask])
            return curve

        fcurve = fly_curve(d_out, entry)
        if fcurve is not None:
            if 'plume' in filename:
                ctrl_curves.append(fcurve)
            else:
                exp_curves.append(fcurve)

    exp_curves = np.array(exp_curves) if len(exp_curves) else np.empty((0, n_bins))
    ctrl_curves = np.array(ctrl_curves) if len(ctrl_curves) else np.empty((0, n_bins))

    def mean_sem(arr):
        with np.errstate(invalid='ignore'):
            mean = np.nanmean(arr, axis=0)
            n = np.sum(~np.isnan(arr), axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
        return mean, sem

    m1, s1 = mean_sem(ctrl_curves)
    m2, s2 = mean_sem(exp_curves)

    axs.plot(bin_centers, m1, color=colors[0], linewidth=2, label='ctrl')
    axs.fill_between(bin_centers, m1 - s1, m1 + s1, color=colors[0], alpha=0.3, linewidth=0)
    axs.plot(bin_centers, m2, color=colors[1], linewidth=2, label='exp')
    axs.fill_between(bin_centers, m2 - s2, m2 + s2, color=colors[1], alpha=0.3, linewidth=0)

    axs.set_xlabel('heading − goal angle (º)', fontsize=14, color=markc)
    axs.set_ylabel('walking speed (mm/s)', fontsize=14, color=markc)
    axs.set_xticks([-180, -90, 0, 90, 180])
    axs.tick_params(colors=markc, labelsize=12)
    axs.set_xlim(-180, 180)
    axs.set_ylim(2, 10)
    axs.legend(frameon=False, fontsize=12, labelcolor=markc)
    plt.tight_layout()
    plt.show()

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def zigzag_speed_vs_heading_diff(folder_path, savename, size=(5, 4), grp='ctrl', colors=['lightgrey', '#ff355e'],bin_width_deg=15, speed_pct=20):
    """
    compute the per-fly entry angle (circular mean of
    headings in the last second of each outside-bout) for plume 1 and plume 2.
    Then, using outside-of-odor frames where speed > per-fly 20th percentile,
    plot mean forward speed vs (heading - entry_angle) wrapped to [-180, 180].
    Mean +/- SEM across flies, separately for plume 1 and plume 2.
    """
    bin_edges = np.arange(-180, 180 + bin_width_deg, bin_width_deg)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    # Per-fly binned speed curves: rows = flies, cols = bins
    p1_curves = []
    p2_curves = []

    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_white_plot(size=size, xaxis=True)

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if grp == 'ctrl':
            if 'ctrl' not in filename:
                continue
        elif grp == 'exp':
            if 'ctrl' in filename:
                continue
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

        # --- per-fly entry angles (circular mean over last-second bout headings) ---
        p1_bout_headings = []
        p2_bout_headings = []
        for _, bout_df in d_out1.items():
            if bout_df['seconds'].iloc[-1] - bout_df['seconds'].iloc[0] >= 1:
                circmean_heading(get_last_second(bout_df), p1_bout_headings)
        for _, bout_df in d_out2.items():
            if bout_df['seconds'].iloc[-1] - bout_df['seconds'].iloc[0] >= 1:
                circmean_heading(get_last_second(bout_df), p2_bout_headings)

        if len(p1_bout_headings) == 0 and len(p2_bout_headings) == 0:
            continue

        # circular mean of the bout-circmeans -> per-fly entry angle (radians)
        # circmean_heading appears to append in radians; if it appends degrees, swap to np.deg2rad first.
        p1_entry = circmean(p1_bout_headings) if len(p1_bout_headings) else np.nan
        p2_entry = circmean(p2_bout_headings) if len(p2_bout_headings) else np.nan
        p1_str = f'{np.rad2deg(p1_entry):.1f}°' if not np.isnan(p1_entry) else 'n/a'
        p2_str = f'{np.rad2deg(p2_entry):.1f}°' if not np.isnan(p2_entry) else 'n/a'
        print(f'{filename}  |  plume 1 entry angle: {p1_str}  |  plume 2 entry angle: {p2_str}')

        def fly_curve(d_out, entry_angle_rad):
            if np.isnan(entry_angle_rad) or len(d_out) == 0:
                return None

            return_segments = []
            for _, bout_df in d_out.items():
                if len(bout_df) < 2:
                    continue

                # perpendicular distance from the start->end line, per frame
                # (matches get_a_bout_calc(..., 'furthest_distance_from_plume') logic)
                x1, y1 = bout_df['ft_posx'].iloc[0], bout_df['ft_posy'].iloc[0]
                x2, y2 = bout_df['ft_posx'].iloc[-1], bout_df['ft_posy'].iloc[-1]
                denom = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if denom == 0:
                    continue
                distances = np.abs(
                    (y2 - y1) * bout_df['ft_posx']
                    - (x2 - x1) * bout_df['ft_posy']
                    + x2 * y1 - y2 * x1
                ) / denom

                far_idx = int(np.argmax(distances.values))
                return_leg = bout_df.iloc[far_idx:]
                if len(return_leg) >= 2:
                    return_segments.append(return_leg)

            if len(return_segments) == 0:
                return None

            outside = pd.concat(return_segments, ignore_index=True)
            if 'speed' not in outside.columns or len(outside) == 0:
                return None

            # per-fly 20th percentile filter on the return-leg frames
            thresh = np.nanpercentile(outside['speed'].values, speed_pct)
            outside = outside[outside['speed'] > thresh]
            if len(outside) == 0:
                return None

            # heading diff wrapped to [-180, 180]
            diff_rad = np.angle(np.exp(1j * (outside['ft_heading'].values - entry_angle_rad)))
            diff_deg = np.rad2deg(diff_rad)
            speeds = outside['speed'].values

            curve = np.full(n_bins, np.nan)
            for i in range(n_bins):
                mask = (diff_deg >= bin_edges[i]) & (diff_deg < bin_edges[i + 1])
                if mask.any():
                    curve[i] = np.mean(speeds[mask])
            return curve

        c1 = fly_curve(d_out1, p1_entry)
        c2 = fly_curve(d_out2, p2_entry)
        if c1 is not None:
            p1_curves.append(c1)
        if c2 is not None:
            p2_curves.append(c2)

    p1_curves = np.array(p1_curves) if len(p1_curves) else np.empty((0, n_bins))
    p2_curves = np.array(p2_curves) if len(p2_curves) else np.empty((0, n_bins))

    def mean_sem(arr):
        with np.errstate(invalid='ignore'):
            mean = np.nanmean(arr, axis=0)
            n = np.sum(~np.isnan(arr), axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
        return mean, sem

    m1, s1 = mean_sem(p1_curves)
    m2, s2 = mean_sem(p2_curves)

    axs.plot(bin_centers, m1, color=colors[0], linewidth=2, label='plume 1')
    axs.fill_between(bin_centers, m1 - s1, m1 + s1, color=colors[0], alpha=0.3, linewidth=0)
    axs.plot(bin_centers, m2, color=colors[1], linewidth=2, label='plume 2')
    axs.fill_between(bin_centers, m2 - s2, m2 + s2, color=colors[1], alpha=0.3, linewidth=0)

    axs.set_xlabel('heading − goal angle (º)', fontsize=14, color=markc)
    axs.set_ylabel('walking speed (mm/s)', fontsize=14, color=markc)
    axs.set_xticks([-180, -90, 0, 90, 180])
    axs.tick_params(colors=markc, labelsize=12)
    axs.set_xlim(-180, 180)
    axs.set_ylim(2, 10)
    axs.legend(frameon=False, fontsize=12, labelcolor=markc)
    plt.tight_layout()
    plt.show()

    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def zigzag_speed_vs_heading(folder_path, savename, size=(5, 4), grp='ctrl',
                             colors=['lightgrey', '#ff355e'],
                             bin_width_deg=15, speed_pct=20):
    """
    Using outside-of-odor return-leg frames where speed > per-fly 20th percentile,
    plot mean walking speed vs heading (deg, wrapped to [-180, 180]).
    Mean +/- SEM across flies, separately for plume 1 and plume 2.
    """
    bin_edges = np.arange(-180, 180 + bin_width_deg, bin_width_deg)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    p1_curves = []
    p2_curves = []

    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_white_plot(size=size, xaxis=True)

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if grp == 'ctrl':
            if 'ctrl' not in filename:
                continue
        elif grp == 'exp':
            if 'ctrl' in filename:
                continue

        if 'CM' in folder_path:
            df['exp_phase'] = df['exp_phase'].fillna(0)
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

        def fly_curve(d_out):
            if len(d_out) == 0:
                return None

            return_segments = []
            for _, bout_df in d_out.items():
                if len(bout_df) < 2:
                    continue

                x1, y1 = bout_df['ft_posx'].iloc[0], bout_df['ft_posy'].iloc[0]
                x2, y2 = bout_df['ft_posx'].iloc[-1], bout_df['ft_posy'].iloc[-1]
                denom = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if denom == 0:
                    continue
                distances = np.abs(
                    (y2 - y1) * bout_df['ft_posx']
                    - (x2 - x1) * bout_df['ft_posy']
                    + x2 * y1 - y2 * x1
                ) / denom

                far_idx = int(np.argmax(distances.values))
                return_leg = bout_df.iloc[far_idx:]
                if len(return_leg) >= 2:
                    return_segments.append(return_leg)

            if len(return_segments) == 0:
                return None

            outside = pd.concat(return_segments, ignore_index=True)
            if 'speed' not in outside.columns or len(outside) == 0:
                return None

            thresh = np.nanpercentile(outside['speed'].values, speed_pct)
            outside = outside[outside['speed'] > thresh]
            if len(outside) == 0:
                return None

            # raw heading wrapped to [-180, 180]
            heading_rad = np.angle(np.exp(1j * outside['ft_heading'].values))
            heading_deg = np.rad2deg(heading_rad)
            speeds = outside['speed'].values

            curve = np.full(n_bins, np.nan)
            for i in range(n_bins):
                mask = (heading_deg >= bin_edges[i]) & (heading_deg < bin_edges[i + 1])
                if mask.any():
                    curve[i] = np.mean(speeds[mask])
            return curve

        c1 = fly_curve(d_out1)
        c2 = fly_curve(d_out2)
        if c1 is not None:
            p1_curves.append(c1)
        if c2 is not None:
            p2_curves.append(c2)

    p1_curves = np.array(p1_curves) if len(p1_curves) else np.empty((0, n_bins))
    p2_curves = np.array(p2_curves) if len(p2_curves) else np.empty((0, n_bins))

    def mean_sem(arr):
        with np.errstate(invalid='ignore'):
            mean = np.nanmean(arr, axis=0)
            n = np.sum(~np.isnan(arr), axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
        return mean, sem

    m1, s1 = mean_sem(p1_curves)
    m2, s2 = mean_sem(p2_curves)

    axs.plot(bin_centers, m1, color=colors[0], linewidth=2, label='plume 1')
    axs.fill_between(bin_centers, m1 - s1, m1 + s1, color=colors[0], alpha=0.3, linewidth=0)
    axs.plot(bin_centers, m2, color=colors[1], linewidth=2, label='plume 2')
    axs.fill_between(bin_centers, m2 - s2, m2 + s2, color=colors[1], alpha=0.3, linewidth=0)

    axs.set_xlabel('heading (º)', fontsize=14, color=markc)
    axs.set_ylabel('walking speed (mm/s)', fontsize=14, color=markc)
    axs.set_xticks([-180, -90, 0, 90, 180])
    axs.tick_params(colors=markc, labelsize=12)
    axs.set_xlim(-180, 180)
    axs.set_ylim(2, 10)
    axs.legend(frameon=False, fontsize=12, labelcolor=markc)
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



def plume_onset_uws(folder_path, savename, multiodor=True, size=(6,5), sample_rate=20, buf_pts=4, keywords=['no training', 'L training', 'R training'], colors=['lightgrey', '#c1ffc1', '#6f00ff']):
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
                # if '45_ctrl' in filename:
                #     key = keywords[0]
                #     print(filename)
                if ('LR' in filename and 'ctrl' not in filename) or 'mch' in filename:
                    key = keywords[1]
                elif ('RR' in filename and 'ctrl' not in filename):
                    key = keywords[2]
            elif 'ctrl' in filename:
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


def pulse_onset_uws(folder_path, savename, multiodor=True, size=(6,5), sample_rate=35, pre=5, post=10, colors=['grey', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    pre_pts = int(round(pre * sample_rate))
    post_pts = int(round(post * sample_rate))
    n_points = pre_pts + post_pts
    time = np.linspace(-pre, post, n_points)
    
    traces = []
    if multiodor:
        params_list = exp_parameters_multiodor(folder_path)
    else:
        params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        if multiodor:
            figure_folder, filename, df, df_odor1, df_odor2, df_light, exp_df, xo, yo, plume_colors = this_experiment
        else:
            figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment        
        if 'ctrl' not in filename:
            continue
        _, di, do = inside_outside(exp_df)
        print(list(di.keys()))
        first_key = list(di.keys())[0]
        df_in = di[first_key]
        
        onset_time = df_in.index[0]
        if onset_time in df.index:
            onset_idx = df.index.get_loc(onset_time)
            if onset_idx >= pre_pts and onset_idx + post_pts < len(df):
                segment = df.iloc[onset_idx - pre_pts : onset_idx + post_pts]
                uws_segment = segment['y-vel'].values
                # uws_segment = np.abs(segment['transformed_heading']).values
                if len(uws_segment) == n_points:  # safety check
                    axs.plot(time, uws_segment, color=colors[0], lw=0.5, alpha=0.5)
                    traces.append(uws_segment)

    mean_trace = np.mean(traces, axis=0)
    sem_trace = sem(traces, axis=0)
    axs.plot(time, mean_trace, color=colors[1], lw=2)
    axs.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace, color=colors[1], alpha=0.2)
    axs.axvline(0, color=markc, linewidth=0.5, linestyle='--')
    axs.axvline(5.5, color=markc, linewidth=0.5, linestyle='--')
    axs.set_xlabel("time (s)", color=markc, fontsize=12)
    axs.set_ylabel("upwind speed (mm/s)", color=markc, fontsize=12)
    plt.title("upwind speed at pulse onset", color=markc, fontsize=12)
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

def avg_plume_onset_uws(folder_path, savename, multiodor=True, size=(6,5), sample_rate=10, buf_pts=4, keywords=['no training', 'L training', 'R training'], colors=['lightgrey', '#c1ffc1', '#6f00ff']):
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
                # if '45_ctrl' in filename:
                #     key = keywords[0]
                if (('LR' in filename or 'RL' in filename) and ('ctrl' not in filename)) or 'mch' in filename:
                    key = keywords[1]
                elif ('RR' in filename or 'LL' in filename) and ('ctrl' not in filename):
                    key = keywords[2]      
            elif 'ctrl' in filename:
                print(filename)
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


def CM_disappearing_memory(folder_path, savename, color, size=(5,5)):
    fig, axs, markc = configure_white_plot(size=size, xaxis=True, ncols=4)
    params_list = exp_parameters(folder_path)
    entries_list, strengths_list, perc_list, tort_list = [], [], [], []

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        d, d_in, d_out = inside_outside(exp_df)
        last_out_key = list(d_out.keys())[-1]
        off_df = d_out[last_out_key]
        print(filename)

        num_entries = len(d_in) - 1

        def circ_diff_deg(a, b):
            return (a - b + 180) % 360 - 180
        
        angle, strength = compute_mean_entry_heading(d_out, imaging=False)
        angle = np.rad2deg(angle)

        entry_angle = float(angle)  # make sure it's scalar in degrees
        off_df['ft_heading'] = np.rad2deg(off_df['ft_heading'])
        off_heading = off_df['ft_heading'].astype(float).to_numpy()

        diff = np.abs(circ_diff_deg(off_heading, entry_angle))
        mask = diff <= 22.5
        percentage_in_window = (mask.sum() / len(off_heading)) * 100
        print(f'percentage of frames in 45º wedge: {percentage_in_window:.2f}%')

        entries_list.append(num_entries)
        strengths_list.append(strength)
        perc_list.append(percentage_in_window)

        tort = get_a_bout_calc(off_df, 'tortuosity')
        tort_list.append(tort)

        # plot entries vs. travel %
        axs[0].scatter(num_entries, percentage_in_window, color=color)
        axs[0].set_title('time in entry quadrant vs.\nnumber of entries', color=markc, fontsize=14)
        axs[0].set_xlabel('number of entries', color=markc, fontsize=12)
        axs[0].set_ylabel('% time in quadrant', color=markc, fontsize=12)
        # plot entry angle strength vs. travel %
        axs[1].scatter(strength, percentage_in_window, color=color)
        axs[1].set_title('time in entry quadrant vs.\nentry angle strength', color=markc, fontsize=14)
        axs[1].set_xlabel('mean resultant vector', color=markc, fontsize=12)
        axs[1].set_ylabel('% time in quadrant', color=markc, fontsize=12)
        # plot entries vs. tortuosity
        axs[2].scatter(num_entries, tort, color=color)
        axs[2].set_title('tortuosity vs.\nnumber of entries', color=markc, fontsize=14)
        axs[2].set_xlabel('number of entries', color=markc, fontsize=12)
        axs[2].set_ylabel('tortuosity', color=markc, fontsize=12)
        # plot entry angle strength vs. tortuosity
        axs[3].scatter(strength, tort, color=color)
        axs[3].set_title('tortuosity vs.\nentry angle strength', color=markc, fontsize=14)
        axs[3].set_xlabel('mean resultant vector', color=markc, fontsize=12)
        axs[3].set_ylabel('tortuosity', color=markc, fontsize=12)

    x1 = np.asarray(entries_list, dtype=float)
    x2 = np.asarray(strengths_list, dtype=float)
    y1  = np.asarray(perc_list, dtype=float)
    y2  = np.asarray(tort_list, dtype=float)

    m1, b1, r2_1 = linregress_r2(x1, y1)
    m2, b2, r2_2 = linregress_r2(x2, y1)
    m3, b3, r2_3 = linregress_r2(x1, y2)
    m4, b4, r2_4 = linregress_r2(x2, y2)

    if np.isfinite(r2_1):
        xs = np.linspace(x1.min(), x1.max(), 100)
        axs[0].plot(xs, m1*xs + b1, linewidth=1.5, color=markc)
    if np.isfinite(r2_2):
        xs = np.linspace(x2.min(), x2.max(), 100)
        axs[1].plot(xs, m2*xs + b2, linewidth=1.5, color=markc)
    if np.isfinite(r2_3):
        xs = np.linspace(x1.min(), x1.max(), 100)
        axs[2].plot(xs, m3*xs + b3, linewidth=1.5, color=markc)
    if np.isfinite(r2_4):
        xs = np.linspace(x2.min(), x2.max(), 100)
        axs[3].plot(xs, m4*xs + b4, linewidth=1.5, color=markc)

    print(r2_1)
    print(r2_2)
    print(r2_3)
    print(r2_4)
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

def pulse_uw_tracking(folder_path, savename, size=(5, 5), plotc='black', groups=3, colors=['red', 'red', 'red']):
    if plotc=='white':
        fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    if plotc=='black':
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    dists = {f'dists_{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000] # cut off at 1000
        dl, d_on, d_off = light_on_off(exp_df)
        if df_odor is not None:
            df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000] # cut off at 1000
            uw = df_odor['ft_posy'].max() - yo
            if len(d_on) < 1:
                # print(f'oct plume: {filename}')
                dists['dists_1'].append((uw/1000))
            elif len(d_on) == 1:
                # print(f'oct+LED plume: {filename}')
                dists['dists_3'].append((uw/1000))
                
        elif df_odor is None:
            # print(f'LED plume: {filename}')
            uw = exp_df.loc[exp_df['led1_stpt'] == 0.0, 'ft_posy'].max() - yo
            returns = len(d_on)
            dists['dists_2'].append((uw/1000))
        
    averages = [sum(dists[f'dists_{i+1}']) / len(dists[f'dists_{i+1}']) if dists[f'dists_{i+1}'] else 0 for i in range(groups)]
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(dists[f'dists_{i+1}'])) for i in range(groups)]
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
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def pulse_rpm(folder_path, savename, size=(5, 5), plotc='black', groups=3, colors=['red', 'red', 'red']):
    if plotc=='white':
        fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    if plotc=='black':
        fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    rpms = {f'rpms_{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000] # cut off at 1000
        dl, d_on, d_off = light_on_off(exp_df)
        pl = get_a_bout_calc(exp_df, 'path_length') / 1000
        if df_odor is not None:
            df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000] # cut off at 1000
            d, di, do = inside_outside(exp_df)
            returns = 0
            for key, df in di.items():
                if df['ft_posy'].iloc[-1] > df['ft_posy'].iloc[0]:
                    returns +=1
            if len(d_on) < 1:
                rpms['rpms_1'].append((returns/pl))
            elif len(d_on) == 1:
                rpms['rpms_3'].append((returns/pl))
                
        elif df_odor is None:
            # print(f'LED plume: {filename}')
            returns = 0
            for key, df in d_on.items():
                if df['ft_posy'].iloc[-1] > df['ft_posy'].iloc[0]:
                    returns +=1
            rpms['rpms_2'].append((returns/pl))
        
    averages = [sum(rpms[f'rpms_{i+1}']) / len(rpms[f'rpms_{i+1}']) if rpms[f'rpms_{i+1}'] else 0 for i in range(groups)]
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(rpms[f'rpms_{i+1}'])) for i in range(groups)]
    alphas = [0.5, 0.5, 1]
    for i in range(groups):
        plt.scatter(x_values[i], rpms[f'rpms_{i+1}'], color=colors[i], alpha=alphas[i])
        plt.hlines(averages[i], xmin=i+0.9, xmax=i+1.1, colors=markc, linewidth=2)  # Mean line
    stat, p_value = kruskal(rpms['rpms_1'], rpms['rpms_2'], rpms['rpms_3'])
    print(f"Kruskal-Wallis H-test: H = {stat:.3f}, p = {p_value:.4f}")
    print(rpms['rpms_1'])
    print(rpms['rpms_2'])
    print(rpms['rpms_3'])
    plt.ylabel('upwind returns/meter', color=markc, fontsize=18)
    plt.yticks(fontsize=14)
    axs.set_xticks(range(1, groups+1))
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
    

def block_comparisons(folder_path, savename, size=(3,5), groups=5, labels = ['LED off', 'LED on'], colors=['#c1ffc1', '#6f00ff']):
    # Right now hard coded for only 2 blocks
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False, ncols=groups)
    paired_data = {metric: {} for metric in ['returns_old', 'speeds', 'fvel', 'angvels', 'uw_disp', 'distance_tracked', 'x_dist']}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        mask = (exp_df['led1_stpt'] == 0) & (exp_df['instrip'] == True)
        last_idx = exp_df.index[mask][-1]
        b1 = exp_df.loc[:last_idx].copy()
        b2 = exp_df.loc[last_idx:].copy()
        pair_id = filename
        for block_label, block_df in [('b1', b1), ('b2', b2)]:
            _, d_in, d_out = inside_outside(block_df)
            pl = get_a_bout_calc(block_df, 'path_length') / 1000.0  # m
            returns = 0
            for key, df in d_in.items():
                if df['ft_posy'].iloc[-1] > df['ft_posy'].iloc[0]:
                    returns +=1
            speed = float(np.nanmean(block_df['speed']))
            fvel  = float(np.nanmean(block_df['y-vel']))
            angvel = float(np.nanmean(block_df['abs_angvel']))
            ydisp = (block_df['ft_posy'].max() - block_df['ft_posy'].min()) / 1000.0  # m
            yo = block_df['ft_posy'].iloc[0]
            uw_disps = []
            for _, seg_df in d_out.items():
                if len(seg_df) > 1:
                    uw_disps.append(abs((seg_df['ft_posy'].iloc[-1] - seg_df['ft_posy'].iloc[0]) / 1000.0))
            uw_disp = float(np.nanmean(uw_disps)) if len(uw_disps) else np.nan
            mask = block_df['mfc3_stpt'] == 0
            xpos = np.abs(block_df['ft_posx'] - xo)
            xpos = xpos.where(mask)
            x_dist = xpos.max()
            print(filename)
            print(x_dist)

            # avoid divide-by-zero
            returns_old = (returns / pl) if (pl and not np.isnan(pl) and pl > 0) else np.nan

            paired_data['returns_old'].setdefault(pair_id, {})[block_label] = returns_old
            paired_data['speeds'].setdefault(pair_id, {})[block_label] = speed
            paired_data['fvel'].setdefault(pair_id, {})[block_label] = fvel
            paired_data['angvels'].setdefault(pair_id, {})[block_label] = angvel
            paired_data['uw_disp'].setdefault(pair_id, {})[block_label] = uw_disp
            paired_data['x_dist'].setdefault(pair_id, {})[block_label] = x_dist

    # --- plotting (paired lines per file) ---
    datasets = [
        (axs[5], 'x_dist',     'max. crosswind distance (mm)'),
        (axs[4], 'uw_disp',     '|distance upwind| (m)'),
        (axs[3], 'returns_old', 'returns per meter'),
        (axs[2], 'speeds',      'avg. speed (mm/s)'),
        (axs[1], 'fvel',        'upwind speed (mm/s)'),
        (axs[0], 'angvels',     'ang. velocity (rad/s)'),
    ]
    noise = 0.05
    for ax, key, ylabel in datasets:
        b1_vals, b2_vals = [], []
        for pair_id, data in paired_data[key].items():
            if ('b1' in data) and ('b2' in data):
                v1, v2 = data['b1'], data['b2']
                if np.isnan(v1) or np.isnan(v2):
                    continue
                b1_vals.append(v1)
                b2_vals.append(v2)
                ax.plot([1, 2], [v1, v2], color=colors[0], linewidth=1, alpha=0.8)
        x1 = np.random.normal(1, noise, size=len(b1_vals))
        x2 = np.random.normal(2, noise, size=len(b2_vals))

        if len(b1_vals) >= 1 and len(b1_vals) == len(b2_vals):
            try:
                stat, p_value = wilcoxon(b1_vals, b2_vals)
                p_text = f'p = {p_value:.5g}' if p_value >= 0.001 else 'p < 0.001'
            except Exception:
                p_text = 'p = n/a'
        else:
            p_text = 'p = n/a'

        y_max = max(max(b1_vals, default=0), max(b2_vals, default=0))
        ax.text(1.5, y_max * 1.1 if y_max != 0 else 0.1, p_text,
                ha='center', va='bottom', fontsize=10, color=markc)
        if len(b1_vals):
            m1 = float(np.mean(b1_vals)); e1 = float(sem(b1_vals)) if len(b1_vals) > 1 else 0.0
        else:
            m1, e1 = np.nan, 0.0
        if len(b2_vals):
            m2 = float(np.mean(b2_vals)); e2 = float(sem(b2_vals)) if len(b2_vals) > 1 else 0.0
        else:
            m2, e2 = np.nan, 0.0
        ax.plot([1, 2], [m1, m2], color=colors[1], linewidth=3)
        ax.errorbar(1, m1, yerr=e1, color=colors[1], capsize=0, linewidth=3)
        ax.errorbar(2, m2, yerr=e2, color=colors[1], capsize=0, linewidth=3)
        style_ax(ax, markc, None,xtick_labels=list(labels),xticks=[1, 2],ylabel=ylabel,xlim=(0.5,2.5))
    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    out_dir = os.path.join(folder_path, "fig")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{savename}.pdf"), bbox_inches="tight")


def b2_comparisons(folder_path,savename,size=(3, 5),groups=5,labels=['cc','ctrl'],colors=('#c1ffc1', '#6f00ff')):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False, ncols=groups)
    group_data = {metric: {0: [], 1: []} for metric in['returns_old', 'speeds', 'fvel', 'cvel', 'uw_disp', 'x_dist']}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if 'oct_plume' in filename:
            continue
        if 'strip' in filename:
            continue
        cc = bool(((exp_df['led1_stpt'] == 0) & (exp_df['instrip'] == False)).any())
  
        g = 0 if cc else 1
        mask = (exp_df['led1_stpt'] == 0) & (exp_df['instrip'] == True)
        last_idx = exp_df.index[mask][-1]
        b1 = exp_df.loc[:last_idx].copy()
        b2 = exp_df.loc[last_idx:].copy()

        d, d_in, d_out = inside_outside(b2)
        pl = get_a_bout_calc(b2, 'path_length') / 1000
        returns = 0
        for key, df in d_in.items():
            if df['ft_posy'].iloc[-1] > df['ft_posy'].iloc[0]:
                returns +=1
        speed = float(np.nanmean(b2['speed']))
        fvel = float(np.nanmean(b2['y-vel']))
        cvel = np.abs(float(np.nanmean(b2['x-vel'])))
        ydisp = (b2['ft_posy'].max() - b2['ft_posy'].min()) / 1000.0  # m (not plotted in your original, but kept)
        in_mask = (b2['instrip'] == True)
        odor_fvel = float(np.nanmean(b2.loc[in_mask, 'x-vel'])) if in_mask.any() else np.nan
        uw_disps = 0
        for key, seg_df in d_in.items():
            if len(seg_df) > 1:
                uw_disps += ((seg_df['ft_posy'].iloc[-1] - seg_df['ft_posy'].iloc[0]) / 1000.0)
        # uw_disp = float(np.nanmean(uw_disps)) if len(uw_disps) else np.nan
        mask = block_df['mfc3_stpt'] == 0
        xpos = np.abs(block_df['ft_posx'] - xo)
        xpos = xpos.where(mask)
        x_dist = xpos.max()
        returns_old = (returns / pl) if (pl and not np.isnan(pl) and pl > 0) else np.nan
        metrics = {
            'returns_old': returns_old,
            'speeds': speed,
            'fvel': fvel,
            'cvel': cvel,
            'uw_disp': uw_disps,
            'x_dist': x_dist,
        }
        for k, v in metrics.items():
            if v is None or np.isnan(v):
                continue
            group_data[k][g].append(v)

    datasets = [
        (axs[5], 'x_dist',          'max. crosswind dist (mm)'),
        (axs[4], 'uw_disp',          'upwind displacement in odor (m)'),
        (axs[3], 'returns_old',      'returns per meter'),
        (axs[2], 'speeds',           'avg. speed (mm/s)'),
        (axs[1], 'fvel',             'upwind speed (mm/s)'),
        (axs[0], 'cvel',          'crosswind speed (mm/s)'),
    ]
    noise = 0.05
    for ax, key, ylabel in datasets:
        g0 = np.array(group_data[key][0], dtype=float)
        g1 = np.array(group_data[key][1], dtype=float)
        x0 = np.random.normal(1, noise, size=len(g0))
        x1 = np.random.normal(2, noise, size=len(g1))

        if len(g0):
            ax.scatter(x0, g0, alpha=0.8, color=colors[0], edgecolor='none')
        if len(g1):
            ax.scatter(x1, g1, alpha=0.8, color=colors[1], edgecolor='none')
        if len(g0) >= 1 and len(g1) >= 1:
            try:
                stat, p_value = mannwhitneyu(g0, g1, alternative='two-sided')
                p_text = f'p = {p_value:.4g}' if p_value >= 0.001 else 'p < 0.001'
            except Exception:
                p_text = 'p = n/a'
        else:
            p_text = 'p = n/a'
        y_max = max(np.nanmax(g0) if len(g0) else 0, np.nanmax(g1) if len(g1) else 0)
        ax.text(1.5, y_max * 1.1 if y_max != 0 else 0.1, p_text,
                ha='center', va='bottom', fontsize=10, color=markc)
        m0 = float(np.nanmean(g0)) if len(g0) else np.nan
        e0 = float(sem(g0, nan_policy='omit')) if len(g0) > 1 else 0.0
        m1 = float(np.nanmean(g1)) if len(g1) else np.nan
        e1 = float(sem(g1, nan_policy='omit')) if len(g1) > 1 else 0.0
        ax.hlines(m0, 0.9, 1.1, color=markc, linewidth=3)
        ax.hlines(m1, 1.9, 2.1, color=markc, linewidth=3)
        ax.errorbar(1, m0, yerr=e0, color=markc, capsize=0, linewidth=3)
        ax.errorbar(2, m1, yerr=e1, color=markc, capsize=0, linewidth=3)
        style_ax(ax, markc, None,xtick_labels=list(labels),xticks=[1, 2],ylabel=ylabel,xlim=(0.5, 2.5))
    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    out_dir = os.path.join(folder_path, "fig")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{savename}.pdf"), bbox_inches="tight")


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

def b2_comparisons(folder_path,savename,size=(3, 5),groups=5,labels=['cc','ctrl'],colors=('#c1ffc1', '#6f00ff')):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False, ncols=groups)
    group_data = {metric: {0: [], 1: []} for metric in['returns_old', 'speeds', 'fvel', 'cvel', 'uw_disp', 'x_dist']}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if 'oct_plume' in filename:
            continue
        if 'strip' in filename:
            continue
        cc = bool(((exp_df['led1_stpt'] == 0) & (exp_df['instrip'] == False)).any())
  
        g = 0 if cc else 1
        mask = (exp_df['led1_stpt'] == 0) & (exp_df['instrip'] == True)
        last_idx = exp_df.index[mask][-1]
        b1 = exp_df.loc[:last_idx].copy()
        b2 = exp_df.loc[last_idx:].copy()

        d, d_in, d_out = inside_outside(b2)
        pl = get_a_bout_calc(b2, 'path_length') / 1000
        returns = 0
        for key, df in d_in.items():
            if df['ft_posy'].iloc[-1] > df['ft_posy'].iloc[0]:
                returns +=1
        speed = float(np.nanmean(b2['speed']))
        fvel = float(np.nanmean(b2['y-vel']))
        cvel = np.abs(float(np.nanmean(b2['x-vel'])))
        ydisp = (b2['ft_posy'].max() - b2['ft_posy'].min()) / 1000.0  # m (not plotted in your original, but kept)
        in_mask = (b2['instrip'] == True)
        odor_fvel = float(np.nanmean(b2.loc[in_mask, 'x-vel'])) if in_mask.any() else np.nan
        uw_disps = 0
        xpos = np.abs(b2['ft_posx'] - xo)
        x_dist = xpos.max()
        for key, seg_df in d_in.items():
            if len(seg_df) > 1:
                uw_disps += ((seg_df['ft_posy'].iloc[-1] - seg_df['ft_posy'].iloc[0]) / 1000.0)
        # uw_disp = float(np.nanmean(uw_disps)) if len(uw_disps) else np.nan
        
        returns_old = (returns / pl) if (pl and not np.isnan(pl) and pl > 0) else np.nan
        metrics = {
            'returns_old': returns_old,
            'speeds': speed,
            'fvel': fvel,
            'cvel': cvel,
            'uw_disp': uw_disps,
            'x_dist': x_dist
        }
        for k, v in metrics.items():
            if v is None or np.isnan(v):
                continue
            group_data[k][g].append(v)

    datasets = [
        (axs[5], 'x_dist',          'max. croswind dist. (mm)'),
        (axs[4], 'uw_disp',          'upwind displacement in odor (m)'),
        (axs[3], 'returns_old',      'returns per meter'),
        (axs[2], 'speeds',           'avg. speed (mm/s)'),
        (axs[1], 'fvel',             'upwind speed (mm/s)'),
        (axs[0], 'cvel',          'crosswind speed (mm/s)'),
    ]
    noise = 0.05
    for ax, key, ylabel in datasets:
        g0 = np.array(group_data[key][0], dtype=float)
        g1 = np.array(group_data[key][1], dtype=float)
        x0 = np.random.normal(1, noise, size=len(g0))
        x1 = np.random.normal(2, noise, size=len(g1))

        if len(g0):
            ax.scatter(x0, g0, alpha=0.8, color=colors[0], edgecolor='none')
        if len(g1):
            ax.scatter(x1, g1, alpha=0.8, color=colors[1], edgecolor='none')
        if len(g0) >= 1 and len(g1) >= 1:
            try:
                stat, p_value = mannwhitneyu(g0, g1, alternative='two-sided')
                p_text = f'p = {p_value:.4g}' if p_value >= 0.001 else 'p < 0.001'
            except Exception:
                p_text = 'p = n/a'
        else:
            p_text = 'p = n/a'
        y_max = max(np.nanmax(g0) if len(g0) else 0, np.nanmax(g1) if len(g1) else 0)
        ax.text(1.5, y_max * 1.1 if y_max != 0 else 0.1, p_text,
                ha='center', va='bottom', fontsize=10, color=markc)
        m0 = float(np.nanmean(g0)) if len(g0) else np.nan
        e0 = float(sem(g0, nan_policy='omit')) if len(g0) > 1 else 0.0
        m1 = float(np.nanmean(g1)) if len(g1) else np.nan
        e1 = float(sem(g1, nan_policy='omit')) if len(g1) > 1 else 0.0
        ax.hlines(m0, 0.9, 1.1, color=markc, linewidth=3)
        ax.hlines(m1, 1.9, 2.1, color=markc, linewidth=3)
        ax.errorbar(1, m0, yerr=e0, color=markc, capsize=0, linewidth=3)
        ax.errorbar(2, m1, yerr=e1, color=markc, capsize=0, linewidth=3)
        style_ax(ax, markc, None,xtick_labels=list(labels),xticks=[1, 2],ylabel=ylabel,xlim=(0.5, 2.5))
    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    out_dir = os.path.join(folder_path, "fig")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{savename}.pdf"), bbox_inches="tight")


def b2_inside_outside(folder_path, savename, size=(3,5), cutoff=500, labels = ['baseline', 'first 2 min.', 'last 2 min.'], colors=['#c1ffc1', '#6f00ff']):
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


def block_xpos_distribution(folder_path, savename, size=(6,5), labels = ['LED off', 'LED on'], colors=['#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    xpos_dist_b1 = []
    xpos_dist_b2 = []
    b1_meds = []
    b2_meds = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        mask = (exp_df['led1_stpt'] == 0) & (exp_df['instrip'] == True)
        last_idx = exp_df.index[mask][-1]
        b1 = exp_df.loc[:last_idx].copy()
        b2 = exp_df.loc[last_idx:].copy()
        xpos_b1 = (b1['ft_posx'] - xo)
        xpos_b2 = (b2['ft_posx'] - xo)
        xpos_dist_b1.append(np.abs(xpos_b1))
        xpos_dist_b2.append(np.abs(xpos_b2))
        b1_meds.append(np.median(np.abs(xpos_b1)))
        b2_meds.append(np.median(np.abs(xpos_b2)))
    compiled_b1 = pd.concat(xpos_dist_b1, axis=0, ignore_index=True)
    compiled_b2 = pd.concat(xpos_dist_b2, axis=0, ignore_index=True)      
    sns.kdeplot(compiled_b1, label=labels[0], color=colors[0], fill=False, cut=0)
    sns.kdeplot(compiled_b2, label=labels[1], color=colors[1], fill=False, cut=0) 
    stat, p = wilcoxon(b1_meds, b2_meds)
    print(f"Wilcoxon signed-rank: stat={stat:.3f}, p={p:.4g}")
    stat, p = ks_2samp(compiled_b1, compiled_b2)
    print(f"KS statistic = {stat:.3f}, p = {p:.10e}")
    min_x, max_x = min(compiled_b1.min(), compiled_b2.min()), max(compiled_b1.max(), compiled_b2.max())
    axs.set_xlim([0, 500])
    plt.axvline(x=25, color='lightgrey')
    # plt.axvline(x=-25, color='lightgrey')
    plt.xlabel('x-position (mm)', fontsize=18, color='white')
    plt.ylabel('density', fontsize=18, color='white')
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_linewidth(2)
    axs.spines['bottom'].set_color('white')
    axs.spines['left'].set_color('white')
    plt.xticks([0, 500], fontsize=14, color='white')
    plt.yticks([0, 0.03], fontsize=14, color='white')
    legend = plt.legend()  # Get the legend object
    legend.get_frame().set_facecolor('none')  # Set the facecolor to none (transparent)
    legend.get_frame().set_edgecolor('none')
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def cc_xpos_distribution(folder_path, savename, size=(6,5), labels = ['LED off', 'LED on'], colors=['#c1ffc1', '#6f00ff']):
    fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    xpos_dist_b1 = []
    xpos_dist_b2 = []
    b1_meds = []
    b2_meds = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if 'oct_plume' in filename or 'strip' in filename or 'pulse' in filename:
            continue
        mask = (exp_df['led1_stpt'] == 0) & (exp_df['instrip'] == True)
        last_idx = exp_df.index[mask][-1]
        b1 = exp_df.loc[:last_idx].copy()
        b2 = exp_df.loc[last_idx:].copy()
        xpos_b2 = (b2['ft_posx'] - xo)
        if 'ctrl' in filename:
            xpos_dist_b1.append(np.abs(xpos_b2))
            b1_meds.append(np.median(np.abs(xpos_b2)))
        else:
            xpos_dist_b2.append(np.abs(xpos_b2))
            b2_meds.append(np.median(np.abs(xpos_b2)))
    compiled_b1 = pd.concat(xpos_dist_b1, axis=0, ignore_index=True)
    compiled_b2 = pd.concat(xpos_dist_b2, axis=0, ignore_index=True)      
    sns.kdeplot(compiled_b1, label=labels[0], color=colors[0], fill=False, cut=0)
    sns.kdeplot(compiled_b2, label=labels[1], color=colors[1], fill=False, cut=0) 
    stat, p = mannwhitneyu(b1_meds, b2_meds, alternative='two-sided')
    print(f"Mann-Whitney U: stat={stat:.3f}, p={p:.4g}")
    stat, p = ks_2samp(compiled_b1, compiled_b2)
    print(f"KS statistic = {stat:.3f}, p = {p:.10e}")
    min_x, max_x = min(compiled_b1.min(), compiled_b2.min()), max(compiled_b1.max(), compiled_b2.max())
    axs.set_xlim([0, 250])
    plt.axvline(x=25, color='lightgrey')
    # plt.axvline(x=-25, color='lightgrey')
    plt.xlabel('x-position (mm)', fontsize=18, color=markc)
    plt.ylabel('density', fontsize=18, color=markc)
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_linewidth(2)
    axs.spines['bottom'].set_color(markc)
    axs.spines['left'].set_color(markc)
    plt.xticks([0, 300], fontsize=14, color=markc)
    plt.yticks([0, 0.03], fontsize=14, color=markc)
    legend = plt.legend()  # Get the legend object
    legend.get_frame().set_facecolor('none')  # Set the facecolor to none (transparent)
    legend.get_frame().set_edgecolor('none')
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def PAM_returns(folder_path, savename, size=(6,5), colors=['lightgrey', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    flydata = []

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if 'pulse' in filename or 'ctrl' in filename:
            _, d_on, d_off = light_on_off(exp_df)
            _, d_in, d_out = inside_outside(exp_df)
            n_reinf = len(d_on)
            n_returns = 0
            for key, df in d_out.items():
                if return_to_edge(df):
                    n_returns += 1 
            if n_returns < 1:
                n_returns = 1
            fly_id = filename  
            flydata.append({
                "fly_id": fly_id,
                "returns": n_returns,
                "reinforcements": n_reinf,
            })

    flies_sorted = sorted(flydata, key=lambda d: (d["reinforcements"], d["returns"]))
    for i, d in enumerate(flies_sorted, start=1):
        n_returns = d["returns"]
        n_reinf = d["reinforcements"]
        ys = np.arange(1, n_returns + 1)
        xs = np.full_like(ys, i, dtype=float)
        pam = ys <= n_reinf
        axs.scatter(xs[~pam], ys[~pam], s=10, color=colors[0], edgecolors='none')
        axs.scatter(xs[pam],  ys[pam], s=10, color=colors[1], edgecolors='none')
    axs.set_ylabel("returns", color=markc)
    axs.set_xlim(0.5, len(flies_sorted) + 0.5)
    axs.set_xticks(range(1, len(flies_sorted)+1))
    axs.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    outdir = os.path.join(folder_path, "fig")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{savename}.pdf"), bbox_inches='tight')


def cc_uws(folder_path, savename, size=(6,5), sample_rate=20, tbef=2, taf=2, labels=['ctrl', 'cc'], colors=['lightgrey', 'red'], baseline_subtract=True):
    fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    n_bef = int(round(tbef * sample_rate))
    n_aft = int(round(taf * sample_rate))
    time = (np.arange(n_bef + n_aft) - n_bef) / sample_rate
    params_list = exp_parameters(folder_path)
    all_traces = {'ctrl': [], 'exp': []}

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if 'oct_plume' in filename or 'strip' in filename or 'pulse' in filename:
            continue
        mask = (exp_df['led1_stpt'] == 0) & (exp_df['instrip'] == True)
        if not np.any(mask):
            continue

        last_idx = exp_df.index[mask][-1]
        b2 = exp_df.loc[last_idx:].copy()
        ins = b2['instrip'].astype(bool)
        entry_mask = ins & (~ins.shift(1, fill_value=False))
        entry_times = b2.index[entry_mask].tolist()
        if len(entry_times) == 0:
            continue

        for onset_time in entry_times:
            onset_idx = df.index.get_loc(onset_time)
            if onset_idx < n_bef or (onset_idx + n_aft) >= len(df):
                continue
            segment = df.iloc[onset_idx - n_bef : onset_idx + n_aft]
            uws_segment = np.abs(segment['y-vel']).values

            if baseline_subtract:
                baseline = np.mean(uws_segment[:n_bef])
                trace = uws_segment - baseline
            else:
                trace = uws_segment.copy()
            if len(trace) != (n_bef + n_aft):
                continue
            group = 'ctrl' if 'ctrl' in filename else 'exp'
            all_traces[group].append(trace)
    for key, label, color in zip(['ctrl', 'exp'], labels, colors):
        traces = all_traces[key]
        if len(traces) == 0:
            continue
        traces = np.asarray(traces)  # shape: (n_traces, n_timepoints)
        mean_trace = np.mean(traces, axis=0)
        sem_trace = sem(traces, axis=0) if len(traces) > 1 else np.zeros_like(mean_trace)
        axs.plot(time, mean_trace, color=color, lw=2, label=f"{label} (n={len(traces)})")
        axs.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace,
                         color=color, alpha=0.2)
    axs.axvline(0, color=markc, linestyle='--')
    axs.set_xlabel("time (s)", color=markc, fontsize=14)
    axs.set_ylabel("upwind speed (mm/s)", color=markc, fontsize=14)
    axs.set_title("upwind speed change at plume entry", color=markc, fontsize=14)
    plt.legend(prop={'size': 10}, frameon=False, loc='upper left', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.show()
    outdir = os.path.join(folder_path, "fig")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{savename}.pdf"), bbox_inches='tight')


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

def oct_acv_comparisons(folder_path, savename, size=(3,5), side='inside', colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=4)
    params_list = exp_parameters(folder_path)
    data = {metric: {'acv': [], 'oct': []} for metric in ['speed', 'uw_dist', 'pl', 'time']}
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        group = 'oct' if 'oct' in filename else 'acv'
        _, d_in, d_out = inside_outside_odor(exp_df)
        if len(d_in) < 3:
            continue
        if side == 'inside':
            bout_dict = d_in
        elif side == 'outside':
            bout_dict = d_out

        speed_vals, uw_dists, pl_vals, time_vals = [], [], [], []
        for key, bout_df in bout_dict.items():
            speed_vals.append(float(bout_df['speed'].mean()))
            uw_dists.append(float(bout_df['ft_posy'].iloc[-1] - bout_df['ft_posy'].iloc[0]))
            pl_vals.append(float(get_a_bout_calc(bout_df, 'path_length')/1000))
            time_vals.append(float(get_a_bout_calc(bout_df, 'duration')))
        
        metrics = {
                'speed': float(np.mean(speed_vals)),
                'uw_dist': float(np.mean(uw_dists)),
                'pl': float(np.mean(pl_vals)),
                'time': float(np.mean(time_vals)),
            }
        for metric, value in metrics.items():
            if np.isfinite(value):
                data[metric][group].append(value)
    datasets = [
        (axs[0], 'speed', f'{side} speed (mm/s)', (0.5, 1.5)),
        (axs[1], 'uw_dist', f'{side} upwind distance (mm)', (0.5, 1.5)),
        (axs[2], 'pl', f'{side} pathlength (m)', (0.5, 1.5)),
        (axs[3], 'time', f'{side} time (s)', (0.5, 1.5)),
    ]
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = data[key]['acv']
        exp_vals = data[key]['oct']

        x_ctrl = np.random.normal(0.7, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(1.2, noise, size=len(exp_vals))
        ax.scatter(x_ctrl, ctrl_vals, color=colors[0], alpha=0.6)
        ax.scatter(x_exp, exp_vals, color=colors[1], alpha=0.6)

        stat, p_value = mannwhitneyu(ctrl_vals, exp_vals, alternative='two-sided')
        p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
        ax.text(1, max(max(ctrl_vals), max(exp_vals)) * 1.1,
                p_text, ha='center', va='bottom', fontsize=10, color=markc)

        mean_ctrl = np.mean(ctrl_vals) if ctrl_vals else 0
        mean_exp = np.mean(exp_vals) if exp_vals else 0
        sem_ctrl = sem(ctrl_vals) if len(ctrl_vals) > 1 else 0
        sem_exp = sem(exp_vals) if len(exp_vals) > 1 else 0
        ax.hlines(mean_ctrl, 0.6, 0.8, color=markc, linewidth=3)
        ax.hlines(mean_exp, 1.1, 1.3, color=markc, linewidth=3)
        ax.errorbar(0.7, mean_ctrl, yerr=sem_ctrl, color=markc, capsize=0, linewidth=3)
        ax.errorbar(1.2, mean_exp, yerr=sem_exp, color=markc, capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['acv', 'oct'], xticks=[0.7,1.2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')
        
def oct_led_comparisons(folder_path, savename, size=(3,5), colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=4)
    params_list = exp_parameters(folder_path)
    data = {metric: {'on': [], 'off': []} for metric in ['res', 'eas', 'dists', 'pls']}
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, d_in, d_out = inside_outside_odor(exp_df)
        _, d_on, d_off = light_on_off(exp_df)
        if len(d_on) <= 1:
            continue
        
        res_on, ea_on, pls_on, dists_on = [],[],[],[]
        res_off, ea_off, pls_off, dists_off = [],[],[],[]

        for key, df in d_out.items():
            if return_to_edge(df):
                dist = get_a_bout_calc(df, 'furthest_distance_from_plume') 
                xdist = get_a_bout_calc(df, 'x_distance_from_plume') 
                pl = get_a_bout_calc(df, 'path_length')
                if pl > 0:
                    re = dist/pl
                    if key-1 in d_on:
                        res_on.append(re)
                        dists_on.append(dist)
                        pls_on.append(pl/1000)
                    else:
                        res_off.append(re)   
                        dists_off.append(dist)
                        pls_off.append(pl/1000)

        if len(res_on) > 0 and len(res_off) > 0:
            data['res']['on'].append(sum(res_on)/len(res_on))
            data['res']['off'].append(sum(res_off)/len(res_off))
            axs[0].plot([0.7,1.2],[(sum(res_on)/len(res_on)), (sum(res_off)/len(res_off))], color=markc, linewidth=1, alpha=0.8)
        if len(res_on) > 0 and len(res_off) > 0:
            data['pls']['on'].append(sum(pls_on)/len(pls_on))
            data['pls']['off'].append(sum(pls_off)/len(pls_off))
            axs[2].plot([0.7,1.2],[(sum(pls_on)/len(pls_on)), (sum(pls_off)/len(pls_off))], color=markc, linewidth=1, alpha=0.8)
        if len(res_on) > 0 and len(res_off) > 0:
            data['dists']['on'].append(sum(dists_on)/len(dists_on))
            data['dists']['off'].append(sum(dists_off)/len(dists_off))
            axs[3].plot([0.7,1.2],[(sum(dists_on)/len(dists_on)), (sum(dists_off)/len(dists_off))], color=markc, linewidth=1, alpha=0.8)
        
        

        for k in sorted(d_in.keys()):  
            if k == 1: 
                entry_df = get_entry_df(None, d_in[k], exp_df, window=0.5, first_entry=True)
            else:  
                entry_df = get_entry_df(d_out.get(k-1), d_in[k], exp_df, window=0.5)
            if entry_df is not None:
                if k-2 in sorted(d_on.keys()):
                    circmean = circmean_heading(entry_df, ea_on)
                else:
                    circmean = circmean_heading(entry_df, ea_off)

        if len(ea_on) > 1 and len(ea_off) > 1 and len(d_out) > 1:
            data['eas']['on'].append(mean_resultant_vector_length(ea_on))
            data['eas']['off'].append(mean_resultant_vector_length(ea_off))
            axs[1].plot([0.7,1.2],[(mean_resultant_vector_length(ea_on)), (mean_resultant_vector_length(ea_off))], color=markc, linewidth=1, alpha=0.8)
    
    print(data)
    datasets = [
        (axs[0], 'res', 'return efficiency', (0.5, 1.5)),
        (axs[1], 'eas', 'entry angle strength', (0.5, 1.5)),
        (axs[2], 'pls', 'pathlength(m)', (0.5, 1.5)),
        (axs[3], 'dists', 'dist. from plume (mm)', (0.5, 1.5)),
    ]
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = data[key]['on']
        exp_vals = data[key]['off']

        x_ctrl = np.random.normal(0.7, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(1.2, noise, size=len(exp_vals))

        stat, p_value = wilcoxon(ctrl_vals, exp_vals, alternative='two-sided')
        p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
        ax.text(1, max(max(ctrl_vals), max(exp_vals)) * 1.1,
                p_text, ha='center', va='bottom', fontsize=10, color=markc)

        mean_ctrl = np.mean(ctrl_vals) if ctrl_vals else 0
        mean_exp = np.mean(exp_vals) if exp_vals else 0
        sem_ctrl = sem(ctrl_vals) if len(ctrl_vals) > 1 else 0
        sem_exp = sem(exp_vals) if len(exp_vals) > 1 else 0
        ax.plot([0.7,1.2], [mean_ctrl, mean_exp], color=colors[1], linewidth=3)
        ax.errorbar(0.7, mean_ctrl, yerr=sem_ctrl, color=colors[1], capsize=0, linewidth=3)
        ax.errorbar(1.2, mean_exp, yerr=sem_exp, color=colors[1], capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['on', 'off'], xticks=[0.7,1.2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def oct_led_re_scatter(folder_path, savename, size=(3,5), colors=['lightgrey', '#ff355e']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    dist1_list, pl1_list = [], []
    dist2_list, pl2_list = [], []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment     
        _, d_in, d_out = inside_outside_odor(exp_df)   
        _, d_on, d_off = light_on_off(exp_df)
        if len(d_on) <= 1:
            continue

        for key, df in d_out.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                dist = get_a_bout_calc(df, 'furthest_distance_from_plume')
                pl = get_a_bout_calc(df, 'path_length')
                if key-1 in d_on:
                    plt.scatter(dist, dist/pl, color=colors[1], alpha=0.8)
                    dist2_list.append(dist)
                    pl2_list.append(dist/pl)
                else:
                    plt.scatter(dist, dist/pl, color=colors[0], alpha=0.8)
                    dist1_list.append(dist)
                    pl1_list.append(dist/pl)
       
    dist1_array, pl1_array = np.array(dist1_list), np.array(pl1_list)
    dist2_array, pl2_array = np.array(dist2_list), np.array(pl2_list)
    if len(dist1_array) > 1:
        slope1, intercept1, r1, _, _ = linregress(dist1_array, pl1_array)
        r2_off = r1 ** 2
        x_fit1 = np.linspace(min(dist1_array), max(dist1_array), 100)
        y_fit1 = slope1 * x_fit1 + intercept1
        plt.plot(x_fit1, y_fit1, color=colors[0], linestyle='-', linewidth=2, label='LED off')
    if len(dist2_array) > 1:
        slope2, intercept2, r2, _, _ = linregress(dist2_array, pl2_array)
        r2_on = r2 ** 2
        x_fit2 = np.linspace(min(dist2_array), max(dist2_array), 100)
        y_fit2 = slope2 * x_fit2 + intercept2
        plt.plot(x_fit2, y_fit2, color=colors[1], linestyle='-', linewidth=2, label='LED on')
    print(f"LED off  : R² = {r2_off:.4f}")
    print(f"LED on   : R² = {r2_on:.4f}")
    plt.ylabel('return efficiency', fontsize=16, color=markc)
    plt.xlabel('furthest distance from plume (mm)', fontsize=16, color=markc)
    plt.yticks(fontsize=14, color=markc)
    plt.xticks(fontsize=14, color=markc)
    # axs.plot(np.linspace(0, 1000, 100), 2*(np.linspace(0, 1000, 100)), linestyle='--', color=markc)
    axs.set_xlim(-10, 200)
    axs.set_ylim(0, 0.5)
    legend = plt.legend(prop={'size': 10}, frameon=False)
    plt.setp(legend.get_texts(), color=markc)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def oct_pulse_comparisons(folder_path, savename, size=(3,5), colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=4)
    params_list = exp_parameters(folder_path)
    data = {metric: {'single': [], 'multi': []} for metric in ['rpm', 'res', 'eas', 'lats']}
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, d_in, d_out = inside_outside_odor(exp_df)
        _, d_on, d_off = light_on_off(exp_df)
      
        if len(d_on) == 1 and len(d_out) > 3:
            group = 'single'
        elif len(d_on) > 1 and len(d_out) > 3:
            group='multi'

        data['rpm'][group].append((len(d_in)-1) / (get_a_bout_calc(exp_df, 'path_length')/1000))

        res = []
        for key, df in d_out.items():
            dist = get_a_bout_calc(df, 'furthest_distance_from_plume') 
            pl = get_a_bout_calc(df, 'path_length')
            if pl > 0:
                re = dist/pl
                res.append(re)
        data['res'][group].append(sum(res)/len(res))
        
        ea = []
        for k in sorted(d_in.keys()):  
            if k == 1: 
                entry_df = get_entry_df(None, d_in[k], exp_df, window=0.5, first_entry=True)
            else:  
                entry_df = get_entry_df(d_out.get(k-1), d_in[k], exp_df, window=0.5)
            if entry_df is not None:
                circmean = circmean_heading(entry_df, ea)
        data['eas'][group].append(mean_resultant_vector_length(ea))
        
        lats = []
        for key, df in d_out.items():
            start_pos = df['ft_posx'].iloc[0]

            # 2. Distance from start
            dist_from_start = (df['ft_posx'] - start_pos).abs()

            # 3. Index where fly is farthest from plume
            idx_far = dist_from_start.idxmax()

            # 4. Inbound segment: from farthest point to end
            df_inbound = df.loc[idx_far:]

            # 5. Average inbound speed using existing speed column
            avg_inbound_speed = df_inbound['speed'].mean()

            lats.append(avg_inbound_speed)
        lats_avg = sum(lats) / len(lats)
        data['lats'][group].append(lats_avg)

    datasets = [
        (axs[0], 'rpm', 'returns per meter', (0.5, 1.5)),
        (axs[1], 'res', 'return efficiency', (0.5, 1.5)),
        (axs[2], 'eas', 'entry angle strength', (0.5, 1.5)),
        (axs[3], 'lats', 'return latency (s)', (0.5, 1.5)),
    ]
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = data[key]['single']
        exp_vals = data[key]['multi']

        x_ctrl = np.random.normal(0.7, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(1.2, noise, size=len(exp_vals))
        ax.scatter(x_ctrl, ctrl_vals, color=colors[1], alpha=0.4)
        ax.scatter(x_exp, exp_vals, color=colors[1])

        stat, p_value = mannwhitneyu(ctrl_vals, exp_vals, alternative='two-sided')
        p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
        ax.text(1, max(max(ctrl_vals), max(exp_vals)) * 1.1,
                p_text, ha='center', va='bottom', fontsize=10, color=markc)

        mean_ctrl = np.mean(ctrl_vals) if ctrl_vals else 0
        mean_exp = np.mean(exp_vals) if exp_vals else 0
        sem_ctrl = sem(ctrl_vals) if len(ctrl_vals) > 1 else 0
        sem_exp = sem(exp_vals) if len(exp_vals) > 1 else 0
        ax.hlines(mean_ctrl, 0.6, 0.8, color=markc, linewidth=3)
        ax.hlines(mean_exp, 1.1, 1.3, color=markc, linewidth=3)
        ax.errorbar(0.7, mean_ctrl, yerr=sem_ctrl, color=markc, capsize=0, linewidth=3)
        ax.errorbar(1.2, mean_exp, yerr=sem_exp, color=markc, capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['single', 'multi'], xticks=[0.7,1.2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def oct_time_comparisons(folder_path, savename, size=(3,5), colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=5)
    params_list = exp_parameters(folder_path)
    
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, d_in, d_out = inside_outside_odor(exp_df)
        _, d_on, d_off = light_on_off(exp_df)
        if len(d_on) < 3:
            continue
        data = {metric: [] for metric in ['rpm', 'res', 'eas', 'lats', 'dist']}
        
        dur = 0
        for key, df in d_on.items():
            dur += df['seconds'].iloc[-1] - df['seconds'].iloc[0]

        data['rpm'].append((len(d_in)-1) / (get_a_bout_calc(exp_df, 'path_length')/1000))
        data['dist'].append((df_odor['ft_posy'].max() - df_odor['ft_posy'].min())/1000)

        res = []
        for key, df in d_out.items():
            dist = get_a_bout_calc(df, 'furthest_distance_from_plume') 
            pl = get_a_bout_calc(df, 'path_length')
            if pl > 0:
                re = dist/pl
                res.append(re)
        data['res'].append(sum(res)/len(res))
        
        ea = []
        for k in sorted(d_in.keys()):  
            if k == 1: 
                entry_df = get_entry_df(None, d_in[k], exp_df, window=0.5, first_entry=True)
            else:  
                entry_df = get_entry_df(d_out.get(k-1), d_in[k], exp_df, window=0.5)
            if entry_df is not None:
                circmean = circmean_heading(entry_df, ea)
        data['eas'].append(mean_resultant_vector_length(ea))
        
        lats = []
        for key, df in d_out.items():
            lat = df['seconds'].iloc[-1] - df['seconds'].iloc[0]
            lats.append(lat)      
        lats_avg = sum(lats) / len(lats)
        data['lats'].append(lats_avg)

        datasets = [
            (axs[0], 'rpm', 'returns per meter', colors[0]),
            (axs[1], 'res', 'return efficiency',  colors[1]),
            (axs[2], 'eas', 'entry angle strength',  colors[2]),
            (axs[3], 'lats', 'return latency (s)',  colors[3]),
            (axs[4], 'dist', 'distance tracked (m)',  colors[4])
        ]

        for ax, key, ylabel, color in datasets:
            ax.scatter(dur, sum(data[key])/len(data[key]), color=color)
            ax.set_ylabel(ylabel, fontsize=14, color=markc)
            ax.set_xlabel('reinforcement dur. (s)', fontsize=14)
        
    sns.despine(offset=10)
    fig.tight_layout()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def oct_dist_comparisons(folder_path, savename, size=(3,5), colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=5)
    params_list = exp_parameters(folder_path)
    
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        _, d_in, d_out = inside_outside_odor(exp_df)
        _, d_on, d_off = light_on_off(exp_df)
        if len(d_on) < 3:
            continue
        data = {metric: [] for metric in ['rpm', 'res', 'eas', 'lats', 'dist']}

        dist = 0
        for key, df in d_on.items():
            dist += df['ft_posy'].iloc[-1] - df['ft_posy'].iloc[0]

        data['rpm'].append((len(d_in)-1) / (get_a_bout_calc(exp_df, 'path_length')/1000))
        data['dist'].append((df_odor['ft_posy'].max() - df_odor['ft_posy'].min())/1000)

        res = []
        for key, df in d_out.items():
            dist = get_a_bout_calc(df, 'furthest_distance_from_plume') 
            pl = get_a_bout_calc(df, 'path_length')
            if pl > 0:
                re = dist/pl
                res.append(re)
        data['res'].append(sum(res)/len(res))
        
        ea = []
        for k in sorted(d_in.keys()):  
            if k == 1: 
                entry_df = get_entry_df(None, d_in[k], exp_df, window=0.5, first_entry=True)
            else:  
                entry_df = get_entry_df(d_out.get(k-1), d_in[k], exp_df, window=0.5)
            if entry_df is not None:
                circmean = circmean_heading(entry_df, ea)
        data['eas'].append(mean_resultant_vector_length(ea))
        
        lats = []
        for key, df in d_out.items():
            lat = df['seconds'].iloc[-1] - df['seconds'].iloc[0]
            lats.append(lat)      
        lats_avg = sum(lats) / len(lats)
        data['lats'].append(lats_avg)

        datasets = [
            (axs[0], 'rpm', 'returns per meter', colors[0]),
            (axs[1], 'res', 'return efficiency',  colors[1]),
            (axs[2], 'eas', 'entry angle strength',  colors[2]),
            (axs[3], 'lats', 'return latency (s)',  colors[3]),
            (axs[4], 'dist', 'distance tracked (m)',  colors[4])
        ]

        for ax, key, ylabel, color in datasets:
            ax.scatter(dist, sum(data[key])/len(data[key]), color=color)
            ax.set_ylabel(ylabel, fontsize=14, color=markc)
            ax.set_xlabel('reinforcement dist. (mm)', fontsize=14)
        
    sns.despine(offset=10)
    fig.tight_layout()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

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

def jumping_comparisons(folder_path, savename, size=(3,5), groups=2, keywords = ['ctrl', 'ablated'], colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_white_plot(size=size, xaxis=False, ncols=groups)
    paired_data = {metric: {} for metric in ['tort', 'returns_old', 'speeds', 'fvel', 'angvels', 'uw_disp']}
    params_list = exp_parameters(folder_path)
    time_stamps = {} 
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        d, d_in, d_out = inside_outside(exp_df)
        pl = get_a_bout_calc(exp_df, 'path_length') / 1000
        returns = len(d_in) - 1
        tort = get_a_bout_calc(exp_df, 'tortuosity')
        speed = np.mean(exp_df['speed'])
        fvel = np.mean(exp_df['y-vel'])
        angvel = np.mean(exp_df['abs_angvel'])
        ydisp = (exp_df['ft_posy'].max() - exp_df['ft_posy'].min()) / 1000
        uw_disps = []
        for key, df in d_out.items():
            uw_disps.append(np.abs((exp_df['ft_posy'].iloc[-1] - exp_df['ft_posy'].iloc[0]) / 1000))
        uw_disp = sum(uw_disps) / len(uw_disps)

        # Extract Fly# using regex
        match = re.search(r'Fly(\d+)', filename)
        if not match:
            continue
        fly_id = int(match.group(1))
        label  = 'ctrl' if keywords[0] in filename else 'exp'
        t_match = re.search(r'(\d{8}-\d{6})', filename)
        if t_match:
            ts_str = t_match.group(1)
            ts = datetime.datetime.strptime(ts_str, '%m%d%Y-%H%M%S')
            time_stamps.setdefault(fly_id, {})[label] = ts
        
        for metric, value in (
            ("returns_old", returns / pl),
            ("tort", tort),
            ("speeds", speed),
            ("fvel", fvel),
            ("angvels", angvel),
            ("uw_disp", uw_disp)
        ):
            paired_data[metric].setdefault(fly_id, {})[label] = value

    datasets = [
        (axs[5], 'uw_disp', '|distance upwind| (m)', (0.8, 2.2)),
        (axs[4], 'returns_old', 'returns per meter ', (0.8, 2.2)),
        (axs[3], 'tort', 'tortuosity', (0.8, 2.2)),
        (axs[2], 'speeds', 'avg. speed (mm/s)', (0.8, 2.2)),
        (axs[1], 'fvel', 'upwind speed (mm/s)', (0.8, 2.2)),
        (axs[0], 'angvels', 'ang. velocity (rad/s)', (0.8, 2.2))
    ]
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = []
        exp_vals = []
        for fly_id, data in paired_data[key].items():
            if 'ctrl' in data and 'exp' in data:
                ts_ctrl = time_stamps.get(fly_id, {}).get('ctrl')
                ts_exp  = time_stamps.get(fly_id, {}).get('exp')
                if ts_exp > ts_ctrl:
                    linecolor = colors[0] #'blue'
                else:
                    linecolor=colors[0]
                ctrl_vals.append(data['ctrl'])
                exp_vals.append(data['exp'])
                ax.plot([1, 2], [data['ctrl'], data['exp']], color=linecolor, linewidth=1, alpha=0.5)  # Line between pairs
        x_ctrl = np.random.normal(1, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(2, noise, size=len(exp_vals))
        stat, p_value = wilcoxon(ctrl_vals, exp_vals)
        p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
        ax.text(1.5, max(max(ctrl_vals, default=0), max(exp_vals, default=0)) * 1.1, p_text,
            ha='center', va='bottom', fontsize=10, color=markc)
        # ax.hlines(np.mean(ctrl_vals), 0.9, 1.1, colors=markc, linewidth=2)
        # ax.hlines(np.mean(exp_vals), 1.9, 2.1, colors=markc, linewidth=2)
        ax.plot([1,2], [np.mean(ctrl_vals), np.mean(exp_vals)], color=colors[1], linewidth=3)
        sem_ctrl = sem(ctrl_vals) if len(ctrl_vals) > 1 else 0
        sem_exp = sem(exp_vals) if len(exp_vals) > 1 else 0
        ax.errorbar(1, np.mean(ctrl_vals), yerr=sem_ctrl, color=colors[1], capsize=0, linewidth=3)
        ax.errorbar(2, np.mean(exp_vals), yerr=sem_exp, color=colors[1], capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['off', 'on'], xticks=[1,2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def jumping_comparisons_tracking(folder_path, savename, size=(3,5), groups=2, keywords = ['ctrl', 'ablated'], colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_white_plot(size=size, xaxis=False, ncols=groups)
    paired_data = {metric: {} for metric in ['first_latency','return_lens','entry_angle_strength', 'crossovers']}
    params_list = exp_parameters(folder_path)
    time_stamps = {} 
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        d, d_in, d_out = inside_outside(exp_df)
        if len(d_out) < 2:
            continue
        lengths = 0
        crossovers = 0
        for key, df in d_out.items():
            lengths += df['seconds'].iloc[-1] - df['seconds'].iloc[0]
        for key, df in d_in.items():
            if df['ft_posx'].iloc[-1] - df['ft_posx'].iloc[0] > 40:
                crossovers +=1
        first_exit = d_out[list(d_out.keys())[0]]
        first_latency = first_exit['seconds'].iloc[-1] - first_exit['seconds'].iloc[0]
        return_length = lengths / len(d)
        entry_angle, strength = compute_mean_entry_heading(d_out, imaging=False)
      
        # Extract Fly# using regex
        match = re.search(r'Fly(\d+)', filename)
        if not match:
            continue
        fly_id = int(match.group(1))
        label  = 'ctrl' if keywords[0] in filename else 'exp'
        t_match = re.search(r'(\d{8}-\d{6})', filename)
        if t_match:
            ts_str = t_match.group(1)
            ts = datetime.datetime.strptime(ts_str, '%m%d%Y-%H%M%S')
            time_stamps.setdefault(fly_id, {})[label] = ts
        
        for metric, value in (
            ('first_latency', first_latency),
            ('return_lens', return_length),
            ('entry_angle_strength', strength),
            #('crossovers', crossovers),
        ):
            paired_data[metric].setdefault(fly_id, {})[label] = value

    datasets = [
        (axs[0], 'return_lens', 'avg. return length (s)', (0.8, 2.2)),
        (axs[1], 'first_latency', 'first return latency (s)', (0.8, 2.2)),
        (axs[2], 'entry_angle_strength', 'entry angle strength', (0.8, 2.2)),
        #(axs[3], 'crossovers', 'crossovers', (0.8, 2.2)),
    ]
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = []
        exp_vals = []
        for fly_id, data in paired_data[key].items():
            if 'ctrl' in data and 'exp' in data:
                ts_ctrl = time_stamps.get(fly_id, {}).get('ctrl')
                ts_exp  = time_stamps.get(fly_id, {}).get('exp')
                if ts_exp > ts_ctrl:
                    linecolor = colors[0] #'blue'
                else:
                    linecolor=colors[0]
                ctrl_vals.append(data['ctrl'])
                exp_vals.append(data['exp'])
                ax.plot([1, 2], [data['ctrl'], data['exp']], color=linecolor, linewidth=1, alpha=0.5)  # Line between pairs
        x_ctrl = np.random.normal(1, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(2, noise, size=len(exp_vals))
        stat, p_value = wilcoxon(ctrl_vals, exp_vals)
        p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
        ax.text(1.5, max(max(ctrl_vals, default=0), max(exp_vals, default=0)) * 1.1, p_text,
            ha='center', va='bottom', fontsize=10, color=markc)
        # ax.hlines(np.mean(ctrl_vals), 0.9, 1.1, colors=markc, linewidth=2)
        # ax.hlines(np.mean(exp_vals), 1.9, 2.1, colors=markc, linewidth=2)
        ax.plot([1,2], [np.mean(ctrl_vals), np.mean(exp_vals)], color=colors[1], linewidth=3)
        sem_ctrl = sem(ctrl_vals) if len(ctrl_vals) > 1 else 0
        sem_exp = sem(exp_vals) if len(exp_vals) > 1 else 0
        ax.errorbar(1, np.mean(ctrl_vals), yerr=sem_ctrl, color=colors[1], capsize=0, linewidth=3)
        ax.errorbar(2, np.mean(exp_vals), yerr=sem_exp, color=colors[1], capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['off', 'on'], ylabel=ylabel, xlim=xlim)
    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def jumping_entry_angles(folder_path, savename, size=(5, 5), colors=['#ff355e', 'lightgrey']):
    fly_data = defaultdict(lambda: {'exp': [], 'ctrl': []})
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment  
        d, d_in, d_out = inside_outside(exp_df)
        if len(d_out) < 2:
            continue
        # Extract fly ID and type from filename
        fly_match = re.search(r'(Fly\d+)', filename)
        type_match = 'exp' if 'exp' in filename else 'ctrl'
        if not fly_match:
            continue
        fly_id = fly_match.group(1)
        for key, df in d_out.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 1:
                df = get_last_second(df)
                circ_angle = circmean_heading(df, [])
                if circ_angle is not None:
                    fly_data[fly_id][type_match].append(circ_angle)

    # Create a polar plot for each fly that has both exp and ctrl data
    for fly_id, angles in fly_data.items():
        if angles['exp'] and angles['ctrl']:
            fig, ax, markc = configure_white_polar(size=size, grid=False)
            for angle in angles['exp']:
                plt.polar([0, angle], [0, 1], color=colors[1], alpha=0.2, linewidth=1, solid_capstyle='round')
            summary_vector_exp = np.mean(np.column_stack((np.cos(angles['exp']), np.sin(angles['exp']))), axis=0)
            rho_exp, phi_exp = cart2pol(*summary_vector_exp)
            plt.polar([0, phi_exp], [0, rho_exp], color=colors[1], linewidth=3, label='exp', solid_capstyle='round')
            for angle in angles['ctrl']:
                plt.polar([0, angle], [0, 1], color=colors[0], alpha=0.2, linewidth=1, solid_capstyle='round')
            summary_vector_ctrl = np.mean(np.column_stack((np.cos(angles['ctrl']), np.sin(angles['ctrl']))), axis=0)
            rho_ctrl, phi_ctrl = cart2pol(*summary_vector_ctrl)
            plt.polar([0, phi_ctrl], [0, rho_ctrl], color=colors[0], linewidth=3, label='ctrl', solid_capstyle='round')
            # plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
            plt.title(f'{fly_id} entry angles', fontsize=14)
            fig_folder = os.path.join(folder_path, 'fig')
            os.makedirs(fig_folder, exist_ok=True)
            plt.show()
            fig.savefig(os.path.join(fig_folder, f'{savename}_{fly_id}.pdf'), bbox_inches='tight')
            plt.close(fig)


def INC_plotter(folder_path, led, stripwidth=50, spacing=250, select_file=None, save=False):
    params_list = exp_parameters(folder_path)
    ledc = {
        'none': 'grey',
        'red': '#ff355e',
        'green': '#0bda51'
    }.get(led, 'grey')
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if select_file and filename != select_file:
            continue
        fig, axs, markc = configure_white_plot(size=(10, 10), xaxis=True)
        x_pos = exp_df['ft_posx'] - xo
        min_x, max_x = x_pos.min(), x_pos.max()
        base_left_edge = -(stripwidth/2)  # center plume starts 
        first_start = base_left_edge + spacing * np.floor((min_x - base_left_edge) / spacing)
        last_start = base_left_edge + spacing * np.ceil((max_x - base_left_edge) / spacing)
        plume_starts = np.arange(first_start, last_start + 1, spacing)
        y_pos = exp_df['ft_posy'] - yo
        y_min = y_pos.min()
        height = y_pos.max() - y_min
        for start_x in plume_starts:
            plume = patches.Rectangle(
                (start_x, y_min),
                stripwidth,
                height,
                facecolor='#fbceb1',
                edgecolor='none', alpha=0.5
            )
            axs.add_patch(plume)
        plt.scatter(0, 0, s=150, color='red')
        plt.plot(x_pos, y_pos, color='grey', label='clean air', linewidth=1)
        plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only', linewidth=1)
        # if 'ctrl' not in filename:
        #     plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on', linewidth=1)
        axs.set_aspect('equal', adjustable='datalim')
        plt.title(filename, fontsize=14, color=markc)
        plt.show()
        out_dir = os.path.join(folder_path, 'traj')
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f'{filename}_odor_trajectory_white.pdf'), bbox_inches='tight')

def xpos_dist(folder_path, savename, size):
    fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    xpos_all = []
    xpos_ctrl_all = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        d, d_in, d_out = inside_outside(exp_df)
        xo = exp_df.iloc[0]['ft_posx']
        xpos = np.abs(df['ft_posx'] - xo)

        if 'acv' in filename:
            xpos_ctrl_all.append(xpos)
        else:
            xpos_all.append(xpos)

    xpos_all = pd.concat(xpos_all, axis=0, ignore_index=True)
    xpos_ctrl_all = pd.concat(xpos_ctrl_all, axis=0, ignore_index=True)
    sns.kdeplot(xpos_all, color='red', shade=False, cut=0, label='graded')
    sns.kdeplot(xpos_ctrl_all, color='grey', shade=False, cut=0, label='sharp')
    sigma = 25.0 / np.sqrt(2.0 * np.log(2.0))
    xgrid = np.linspace(0, 500, 1000)
    gauss = np.exp(-(xgrid**2) / (2.0 * sigma**2))  # peak = 1 at x=0
    gauss_deriv = np.abs((-xgrid / (sigma**2)) * gauss)

    axs.plot(xgrid, gauss * 0.3 * axs.get_ylim()[1], linestyle='--', color='black', label='odor conc.')
    axs.plot(xgrid, gauss_deriv * 0.3 * axs.get_ylim()[1] / np.max(gauss_deriv), linestyle='--', color='blue', label='d/dx odor conc.')

    plt.xticks([0, 25, 80, 200], color=markc, fontsize=12)
    plt.yticks([0, 0.02], color=markc, fontsize=12)
    plt.axvline(x=25, color='lightgrey')
    plt.xlabel('x-position (mm)', color=markc, fontsize=14)
    plt.ylabel('probability', color=markc, fontsize=14)
    plt.xlim(0, 200)
    plt.legend(frameon=False)
    plt.show()
    fig.savefig(os.path.join(figure_folder, f'{savename}.pdf'))

def xdist_dist(folder_path, savename, size, colors):
    fig, axs, markc = configure_white_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    xpos_all = []
    xpos_ctrl_all = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        d, d_in, d_out = inside_outside(exp_df)
        
        for key, df in d_out.items():
            if return_to_edge(df) and df['seconds'].iloc[-1]-df['seconds'].iloc[0] > 0.5:
                xdist = get_a_bout_calc(df, 'x_distance_from_plume')
                if 'light' not in filename:
                    xpos_ctrl_all.append(xdist)
                else:
                    xpos_all.append(xdist)

    # xpos_all = pd.concat(xpos_all, axis=0, ignore_index=True)
    # xpos_ctrl_all = pd.concat(xpos_ctrl_all, axis=0, ignore_index=True)
    sns.kdeplot(xpos_all, color=colors[1], shade=True, cut=0, label='LED on')
    sns.kdeplot(xpos_ctrl_all, color=colors[0], shade=True, cut=0, label='LED off')
    # plt.xticks([0, 25, 80, 200], color=markc, fontsize=12)
    plt.yticks([0, 0.02], color=markc, fontsize=12)
    # plt.axvline(x=25, color='lightgrey')
    plt.xlabel('x-distance from plume (mm)', color=markc, fontsize=14)
    plt.ylabel('probability', color=markc, fontsize=14)
    # plt.xlim(0, 200)
    plt.legend(frameon=False)
    plt.show()
    fig.savefig(os.path.join(figure_folder, f'{savename}.pdf'))

def yprog_time(folder_path, savename, size, colors):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        t0 = exp_df['seconds'].iloc[0]

        if 'light' not in filename:
            plt.plot(exp_df['seconds'] - t0, exp_df['ft_posy'] - yo, color=colors[0], linewidth=1, alpha=0.8)
        else:
            plt.plot(exp_df['seconds'] - t0, exp_df['ft_posy'] - yo, color=colors[1], linewidth=1, alpha=0.8)
      
    plt.hlines(y=0, xmin=0, xmax=900, colors='k', linestyles='--', linewidth=0.75)  
    plt.xticks([0, 900], color=markc, fontsize=12)
    plt.yticks([0, 500, 1000], color=markc, fontsize=12)
    plt.xlabel('time (s)', color=markc, fontsize=14)
    plt.ylabel('y-position (mm)', color=markc, fontsize=14)
    plt.xlim(0, 900)
    plt.ylim(-100, 1000)
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def gaussian_vel_hist(folder_path, savename, size, bin_width):
    fig, ax, markc = configure_white_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)

    xs, ys = [], []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment

        x0 = exp_df.iloc[0]['ft_posx']
        xpos = np.abs(exp_df['ft_posx'].to_numpy() - x0)
        yvel = exp_df['speed'].to_numpy()
        conc = exp_df['mfc2_stpt'] / (exp_df['mfc2_stpt'] + exp_df['mfc1_stpt'])

        # yvel = np.abs(exp_df['x-vel'].to_numpy())

        mask = np.isfinite(xpos) & np.isfinite(yvel)
        xs.append(xpos[mask])
        ys.append(yvel[mask])

    xpos = np.concatenate(xs)
    yvel = np.concatenate(ys)

    x_min, x_max = xpos.min(), xpos.max()
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    bin_indices = np.digitize(xpos, bins)

    bin_centers = []
    mean_y = []
    sem_y = []  

    for i in range(1, len(bins)):
        sel = (bin_indices == i)
        if np.any(sel):
            bin_centers.append(0.5*(bins[i-1] + bins[i]))
            mean_y.append(np.mean(yvel[sel]))
            sem_y.append(np.std(yvel[sel]) / np.sqrt(sel.sum()))

    bin_centers = np.array(bin_centers)
    mean_y = np.array(mean_y)
    sem_y = np.array(sem_y)

    ax.errorbar(bin_centers, mean_y, yerr=sem_y, fmt='-o', capsize=1)
    ax.set_xlabel("x position (mm)", fontsize=14)
    ax.set_ylabel("avg. speed (mm/s)",fontsize=14)
    ax.set_xticks([0, 25, 80])
    ax.tick_params(axis='x', labelsize=12, colors=markc)
    ax.tick_params(axis='y', labelsize=12, colors=markc)
    ax.axvline(x=25, color='lightgrey')
    sigma = 25.0 / np.sqrt(2.0 * np.log(2.0))
    xgrid = np.linspace(0, 500, 1000)
    gauss = np.exp(-(xgrid**2) / (2.0 * sigma**2))  # peak = 1 at x=0
    if 'ctrl' not in folder_path:
        ax.plot(xgrid, gauss * 0.2 * ax.get_ylim()[1], linestyle='--', color='black', label='odor conc.')
    ax.set_xlim(0, 80)
    plt.tight_layout()
    plt.show()
    out_path = os.path.join(folder_path, f'{savename}.pdf')
    fig.savefig(out_path, dpi=300)

def silencing_comparisons(folder_path, savename, size=(3,5), groups=2, keyword = 'light', colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_white_plot(size=size, xaxis=False, ncols=groups)
    paired_data = {metric: {} for metric in ['returns', 'returns_uw', 'dist_tracked', 'uw_disp', 'xdist', 'uw_comp', 'ea_strength']}
    params_list = exp_parameters(folder_path)
    time_stamps = {} 
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        print(filename)
        d, d_in, d_out = inside_outside(exp_df)
        d_outside = exp_df.where(exp_df.mfc2_stpt==0)
        pl = get_a_bout_calc(exp_df, 'path_length') / 1000
        duration = exp_df['seconds'].iloc[-1] - exp_df['seconds'].iloc[0]
        uw_returns = 0
        returns = 0

        angvel = np.mean(d_outside['abs_angvel'])
        uw_tracking = (df_odor['ft_posy'].max() - yo) / 1000
        if uw_tracking > 1:
            uw_tracking = 1
        
        if len(d_in) > 2:
            headings, mean_heading, ea_strength = compute_mean_entry_heading(d_out, imaging=False)
            upwind_component = mean_crosswind_component(headings, min_n=1)

            uw_disps = []
            durs = []
            res = []
            xdists = []

            for key, df in d_out.items():
                if return_to_edge(df) and df['seconds'].iloc[-1] > df['seconds'].iloc[0] > 0.5:
                    returns +=1
                    uw_disps.append(((df['ft_posy'].iloc[-1] - df['ft_posy'].iloc[0]) / 1000))
                    durs.append(df['seconds'].iloc[-1] - df['seconds'].iloc[0])
                    xdists.append(get_a_bout_calc(df, 'x_distance_from_plume')) 
                    if df['ft_posy'].iloc[-1] > df['ft_posy'].iloc[0]:
                        uw_returns +=1
                    dist = get_a_bout_calc(df, 'furthest_distance_from_plume') 
                    plen = get_a_bout_calc(df, 'path_length')
                    if pl > 0:
                        r_e = dist/plen
                        res.append(r_e)
            uw_disp = (sum(uw_disps) / len(uw_disps)) * 1000
            durs = (sum(durs) / len(durs)) 
            reff = sum(res)/len(res)
            xdist = (sum(xdists) / len(xdists)) 
            
        else:
            ea_strength, uw_disp, reff = 0, 0, 0 
            end_df = d_out[list(d_out.keys())[-1]]
            durs = end_df['seconds'].iloc[-1] - end_df['seconds'].iloc[0]
            xdist = np.abs(exp_df['ft_posx'].iloc[-1] - xo)
            upwind_component = 0

        match = re.search(r'Fly(\d+)', filename)
        if not match:
            continue
        fly_id = int(match.group(1))
        label  = 'exp' if keyword in filename else 'ctrl'
        t_match = re.search(r'(\d{8}-\d{6})', filename)
        if t_match:
            ts_str = t_match.group(1)
            ts = datetime.datetime.strptime(ts_str, '%m%d%Y-%H%M%S')
            time_stamps.setdefault(fly_id, {})[label] = ts
        
        for metric, value in (
            ("returns", returns / pl),
            ("returns_uw", uw_returns / pl),
            ("dist_tracked", uw_tracking),
            ("uw_disp", uw_disp),
            ("xdist", xdist),
            ('uw_comp', upwind_component),
            ('ea_strength', ea_strength)
        ):
            paired_data[metric].setdefault(fly_id, {})[label] = value

    datasets = [
        (axs[6], 'ea_strength', 'entry angle strength', (0.8, 2.2)),
        (axs[5], 'uw_comp', 'crosswind component', (0.8, 2.2)),
        (axs[4], 'xdist', 'max. x-dist. from plume (mm)', (0.8, 2.2)),
        (axs[3], 'uw_disp', 'distance upwind (mm)', (0.8, 2.2)),
        (axs[2], 'dist_tracked', 'distance tracked (m)', (0.8, 2.2)),
        (axs[1], 'returns_uw', 'upwind returns per meter', (0.8, 2.2)),
        (axs[0], 'returns', 'returns per meter', (0.8, 2.2))
    ]
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = []
        exp_vals = []
        for fly_id, data in paired_data[key].items():
            if 'ctrl' in data and 'exp' in data:
                ts_ctrl = time_stamps.get(fly_id, {}).get('ctrl')
                ts_exp  = time_stamps.get(fly_id, {}).get('exp')
                if ts_exp > ts_ctrl:
                    linecolor = colors[0] 
                else:
                    linecolor=colors[0]
                ctrl_vals.append(data['ctrl'])
                exp_vals.append(data['exp'])
                ax.plot([1, 2], [data['ctrl'], data['exp']], color=linecolor, linewidth=1, alpha=0.5)  # Line between pairs
        x_ctrl = np.random.normal(1, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(2, noise, size=len(exp_vals))
        stat, p_value = wilcoxon(ctrl_vals, exp_vals)
        p_text = f'p = {p_value:.4g}'
        ax.text(1.5, max(max(ctrl_vals, default=0), max(exp_vals, default=0)) * 1.1, p_text,
            ha='center', va='bottom', fontsize=10, color=markc)
        ax.plot([1,2], [np.mean(ctrl_vals), np.mean(exp_vals)], color=colors[1], linewidth=3)
        sem_ctrl = sem(ctrl_vals) if len(ctrl_vals) > 1 else 0
        sem_exp = sem(exp_vals) if len(exp_vals) > 1 else 0
        ax.errorbar(1, np.mean(ctrl_vals), yerr=sem_ctrl, color=colors[1], capsize=0, linewidth=3)
        ax.errorbar(2, np.mean(exp_vals), yerr=sem_exp, color=colors[1], capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['off', 'on'], xticks=[1,2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def unpaired_comparisons(folder_path,savename,size=(3, 5),groups=5,labels=['cc','ctrl'],colors=('#c1ffc1', '#6f00ff')):
    fig, axs, markc = configure_white_plot(size=size, xaxis=False, ncols=groups)
    group_data = {metric: {0: [], 1: []} for metric in['returns_old', 'speeds', 'fvel', 'cvel', 'uw_disp', 'dist']}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment  
        if labels[0] in filename:
            g = 0  
            df_odor = df[df.mfc2_stpt > 0]
        else:
            g = 1
            df_odor = df[df.mfc2_stpt > 0]
        d, d_in, d_out = inside_outside(exp_df)
        pl = get_a_bout_calc(exp_df, 'path_length') / 1000
        returns = 0
        print(df_odor)
        xdist = (df_odor['ft_posx'].iloc[-1] - df_odor['ft_posx'].iloc[0]) / 1000
        dist = xdist * math.sqrt(2)
        if 'LL' in filename:
            dist = -dist
        print (dist)

        for key, df in d_in.items():
            if df['ft_posy'].iloc[-1] > df['ft_posy'].iloc[0]:
                returns +=1
        speed = float(np.nanmean(exp_df['speed']))
        fvel = float(np.nanmean(exp_df['y-vel']))
        cvel = np.abs(float(np.nanmean(exp_df['x-vel'])))
        ydisp = (exp_df['ft_posy'].max() - exp_df['ft_posy'].min()) / 1000.0  # m (not plotted in your original, but kept)
        in_mask = (exp_df['instrip'] == True)
        odor_fvel = float(np.nanmean(exp_df.loc[in_mask, 'y-vel'])) if in_mask.any() else np.nan
        uw_disps = 0
        xpos = np.abs(exp_df['ft_posx'] - xo)
        x_dist = xpos.max()
        for key, seg_df in d_in.items():
            if len(seg_df) > 1:
                uw_disps += ((seg_df['ft_posy'].iloc[-1] - seg_df['ft_posy'].iloc[0]) / 1000.0)
        # uw_disp = float(np.nanmean(uw_disps)) if len(uw_disps) else np.nan
        
        returns_old = (returns / pl) if (pl and not np.isnan(pl) and pl > 0) else np.nan
        metrics = {
            'returns_old': returns_old,
            'speeds': speed,
            'fvel': odor_fvel,
            'cvel': cvel,
            'uw_disp': uw_disps,
            'dist': dist
        }
        for k, v in metrics.items():
            if v is None or np.isnan(v):
                continue
            group_data[k][g].append(v)

    datasets = [
        (axs[5], 'dist',          'distance tracked (m)'),
        (axs[4], 'uw_disp',          'upwind displacement in odor (m)'),
        (axs[3], 'returns_old',      'returns per meter'),
        (axs[2], 'speeds',           'avg. speed (mm/s)'),
        (axs[1], 'fvel',             'upwind speed (mm/s)'),
        (axs[0], 'cvel',          'crosswind speed (mm/s)'),
    ]
    noise = 0.05
    for ax, key, ylabel in datasets:
        g0 = np.array(group_data[key][0], dtype=float)
        g1 = np.array(group_data[key][1], dtype=float)
        x0 = np.random.normal(1, noise, size=len(g0))
        x1 = np.random.normal(2, noise, size=len(g1))

        if len(g0):
            ax.scatter(x0, g0, alpha=0.8, color=colors[0], edgecolor='none')
        if len(g1):
            ax.scatter(x1, g1, alpha=0.8, color=colors[1], edgecolor='none')
        if len(g0) >= 1 and len(g1) >= 1:
            try:
                stat, p_value = mannwhitneyu(g0, g1, alternative='two-sided')
                p_text = f'p = {p_value:.4g}' if p_value >= 0.001 else 'p < 0.001'
            except Exception:
                p_text = 'p = n/a'
        else:
            p_text = 'p = n/a'
        y_max = max(np.nanmax(g0) if len(g0) else 0, np.nanmax(g1) if len(g1) else 0)
        ax.text(1.5, y_max * 1.1 if y_max != 0 else 0.1, p_text,
                ha='center', va='bottom', fontsize=10, color=markc)
        m0 = float(np.nanmean(g0)) if len(g0) else np.nan
        e0 = float(sem(g0, nan_policy='omit')) if len(g0) > 1 else 0.0
        m1 = float(np.nanmean(g1)) if len(g1) else np.nan
        e1 = float(sem(g1, nan_policy='omit')) if len(g1) > 1 else 0.0
        ax.hlines(m0, 0.9, 1.1, color=markc, linewidth=3)
        ax.hlines(m1, 1.9, 2.1, color=markc, linewidth=3)
        ax.errorbar(1, m0, yerr=e0, color=markc, capsize=0, linewidth=3)
        ax.errorbar(2, m1, yerr=e1, color=markc, capsize=0, linewidth=3)
        style_ax(ax, markc, None,xtick_labels=list(labels),xticks=[1, 2],ylabel=ylabel,xlim=(0.5, 2.5))
    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    out_dir = os.path.join(folder_path, "fig")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{savename}.pdf"), bbox_inches="tight")


def unpaired_bl_comparisons(folder_path,savename,size=(3, 5),groups=5,labels=['cc','ctrl'],colors=('#c1ffc1', '#6f00ff')):
    fig, axs, markc = configure_white_plot(size=size, xaxis=False, ncols=groups)
    group_data = {metric: {0: [], 1: []} for metric in['speed', 'fvel', 'cvel', 'angvel', 'tort', ]}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment    
        if labels[0] in filename:
            g = 0  
        else:
            g = 1
        x_i = bl_df['ft_posx'].iloc[0]
        x_f = bl_df['ft_posx'].iloc[-1]
        y_i = bl_df['ft_posy'].iloc[0]
        y_f = bl_df['ft_posy'].iloc[-1]
        disp = ((x_f - x_i)**2 + (y_f - y_i)**2)**0.5
        pl = get_a_bout_calc(bl_df, 'path_length') / 1000
        tort = pl / disp
        speed = float(np.nanmean(bl_df['speed']))
        fvel = float(np.nanmean(bl_df['y-vel']))
        cvel = np.abs(float(np.nanmean(bl_df['x-vel'])))
        angvel = float(np.nanmean(bl_df['abs_angvel']))

        metrics = {
            'speed': speed,
            'fvel': fvel,
            'cvel': cvel,
            'angvel': angvel,
            'tort': tort,
        }

        for k, v in metrics.items():
            if v is None or np.isnan(v):
                continue
            group_data[k][g].append(v)

    datasets = [
        (axs[4], 'speed',         'avg. speed (mm/s)'),
        (axs[3], 'fvel',          'upwind speed (mm/s)'),
        (axs[2], 'cvel',          'crosswind speed (mm/s)'),
        (axs[1], 'angvel',        'ang. velocity (rad/s)'),
        (axs[0], 'tort',          'tortuosity (a.u.)'),
    ]
    noise = 0.05
    for ax, key, ylabel in datasets:
        g0 = np.array(group_data[key][0], dtype=float)
        g1 = np.array(group_data[key][1], dtype=float)
        x0 = np.random.normal(1, noise, size=len(g0))
        x1 = np.random.normal(2, noise, size=len(g1))

        if len(g0):
            ax.scatter(x0, g0, alpha=0.8, color=colors[0], edgecolor='none')
        if len(g1):
            ax.scatter(x1, g1, alpha=0.8, color=colors[1], edgecolor='none')
        if len(g0) >= 1 and len(g1) >= 1:
            try:
                stat, p_value = mannwhitneyu(g0, g1, alternative='two-sided')
                p_text = f'p = {p_value:.4g}' if p_value >= 0.001 else 'p < 0.001'
            except Exception:
                p_text = 'p = n/a'
        else:
            p_text = 'p = n/a'
        y_max = max(np.nanmax(g0) if len(g0) else 0, np.nanmax(g1) if len(g1) else 0)
        ax.text(1.5, y_max * 1.1 if y_max != 0 else 0.1, p_text,
                ha='center', va='bottom', fontsize=10, color=markc)
        m0 = float(np.nanmean(g0)) if len(g0) else np.nan
        e0 = float(sem(g0, nan_policy='omit')) if len(g0) > 1 else 0.0
        m1 = float(np.nanmean(g1)) if len(g1) else np.nan
        e1 = float(sem(g1, nan_policy='omit')) if len(g1) > 1 else 0.0
        ax.hlines(m0, 0.9, 1.1, color=markc, linewidth=3)
        ax.hlines(m1, 1.9, 2.1, color=markc, linewidth=3)
        ax.errorbar(1, m0, yerr=e0, color=markc, capsize=0, linewidth=3)
        ax.errorbar(2, m1, yerr=e1, color=markc, capsize=0, linewidth=3)
        style_ax(ax, markc, None,xtick_labels=list(labels),xticks=[1, 2],ylabel=ylabel,xlim=(0.5, 2.5))
    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    out_dir = os.path.join(folder_path, "fig")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{savename}.pdf"), bbox_inches="tight")


def silencing_withinfly_comparisons(folder_path, savename, size=(3, 4), colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False, ncols=2)
    paired_data = {metric: {} for metric in ['reff', 'ea_strength']}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        print(filename)

        mask = (exp_df['led1_stpt'] == 0) & (exp_df['instrip'] == True)
        cutoff_idx = exp_df.index[mask][0]
        df_off = exp_df.loc[cutoff_idx:].copy()
        df_on = exp_df.loc[:cutoff_idx].copy()

        for label, sub_df in (('off', df_off), ('on', df_on)):
            d, d_in, d_out = inside_outside(sub_df)

            res = []
            if len(d_in) > 2:
                headings, mean_heading, ea_strength = compute_mean_entry_heading(d_out, imaging=False)
                for key, bout_df in d_out.items():
                    if return_to_edge(bout_df) and bout_df['seconds'].iloc[-1] - bout_df['seconds'].iloc[0] > 0.5:
                        dist = get_a_bout_calc(bout_df, 'furthest_distance_from_plume')
                        plen = get_a_bout_calc(bout_df, 'path_length')
                        if plen > 0:
                            res.append(dist / plen)
            else:
                ea_strength = 0

            reff = np.mean(res) if len(res) > 0 else np.nan

            paired_data['reff'].setdefault(filename, {})[label] = reff
            paired_data['ea_strength'].setdefault(filename, {})[label] = ea_strength

    datasets = [
        (axs[0], 'reff', 'return efficiency', (0.8, 2.2)),
        (axs[1], 'ea_strength', 'entry angle strength (1-R)', (0.8, 2.2)),
    ]

    for ax, key, ylabel, xlim in datasets:
        off_vals = []
        on_vals = []
        for filename, data in paired_data[key].items():
            if 'off' not in data or 'on' not in data:
                continue
            if np.isnan(data['off']) or np.isnan(data['on']):
                print(f'  {key}: skipping {filename}, missing value on one side')
                continue
            off_vals.append(data['off'])
            on_vals.append(data['on'])
            ax.plot([1, 2], [data['off'], data['on']], color=colors[0], linewidth=1, alpha=0.5)

        if len(off_vals) < 2:
            print(f'  {key}: not enough paired flies to plot/test, skipping stats')
            continue

        stat, p_value = wilcoxon(off_vals, on_vals)
        p_text = f'p = {p_value:.4g}'
        ax.text(1.5, max(max(off_vals, default=0), max(on_vals, default=0)) * 1.1, p_text,
                 ha='center', va='bottom', fontsize=10, color=markc)
        ax.plot([1, 2], [np.mean(off_vals), np.mean(on_vals)], color=colors[1], linewidth=3)
        sem_off = sem(off_vals) if len(off_vals) > 1 else 0
        sem_on = sem(on_vals) if len(on_vals) > 1 else 0
        ax.errorbar(1, np.mean(off_vals), yerr=sem_off, color=colors[1], capsize=0, linewidth=3)
        ax.errorbar(2, np.mean(on_vals), yerr=sem_on, color=colors[1], capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['off', 'on'], xticks=[1, 2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def opto_comparisons_unpaired(folder_path, savename, size=(3,5), groups=2, keyword='light', colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False, ncols=groups)
    group_data = {metric: {'ctrl': [], 'exp': []} for metric in
                  ['returns', 'returns_uw', 'dist_tracked', 'uw_disp', 'xdist', 'uw_comp', 'ea_strength']}
    params_list = exp_parameters(folder_path)

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        print(filename)
        d, d_in, d_out = inside_outside(exp_df)
        d_outside = exp_df.where(exp_df.mfc2_stpt == 0)
        pl = get_a_bout_calc(exp_df, 'path_length') / 1000
        duration = exp_df['seconds'].iloc[-1] - exp_df['seconds'].iloc[0]
        uw_returns = 0
        returns = 0

        angvel = np.mean(d_outside['abs_angvel'])
        uw_tracking = (df_odor['ft_posy'].max() - yo) / 1000
        if uw_tracking > 1:
            uw_tracking = 1

        if len(d_in) > 2:
            headings, mean_heading, ea_strength = compute_mean_entry_heading(d_out, imaging=False)
            upwind_component = mean_crosswind_component(headings, min_n=1)

            uw_disps = []
            durs = []
            res = []
            xdists = []

            for key, df in d_out.items():
                if return_to_edge(df) and df['seconds'].iloc[-1] > df['seconds'].iloc[0] > 0.5:
                    returns += 1
                    uw_disps.append(((df['ft_posy'].iloc[-1] - df['ft_posy'].iloc[0]) / 1000))
                    durs.append(df['seconds'].iloc[-1] - df['seconds'].iloc[0])
                    xdists.append(get_a_bout_calc(df, 'x_distance_from_plume'))
                    if df['ft_posy'].iloc[-1] > df['ft_posy'].iloc[0]:
                        uw_returns += 1
                    dist = get_a_bout_calc(df, 'furthest_distance_from_plume')
                    plen = get_a_bout_calc(df, 'path_length')
                    if pl > 0:
                        r_e = dist / plen
                        res.append(r_e)
            uw_disp = (sum(uw_disps) / len(uw_disps)) * 1000
            durs = (sum(durs) / len(durs))
            reff = sum(res) / len(res)
            xdist = (sum(xdists) / len(xdists))

        else:
            ea_strength, uw_disp, reff = 0, 0, 0
            end_df = d_out[list(d_out.keys())[-1]]
            durs = end_df['seconds'].iloc[-1] - end_df['seconds'].iloc[0]
            xdist = np.abs(exp_df['ft_posx'].iloc[-1] - xo)
            upwind_component = 0

        label = 'exp' if keyword in filename else 'ctrl'

        for metric, value in (
            ("returns", returns / pl),
            ("returns_uw", uw_returns / pl),
            ("dist_tracked", uw_tracking),
            ("uw_disp", uw_disp),
            ("xdist", xdist),
            ('uw_comp', upwind_component),
            ('ea_strength', ea_strength)
        ):
            group_data[metric][label].append(value)

    datasets = [
        (axs[6], 'ea_strength', 'entry angle strength', (0.8, 2.2)),
        (axs[5], 'uw_comp', 'crosswind component', (0.8, 2.2)),
        (axs[4], 'xdist', 'max. x-dist. from plume (mm)', (0.8, 2.2)),
        (axs[3], 'uw_disp', 'distance upwind (mm)', (0.8, 2.2)),
        (axs[2], 'dist_tracked', 'distance tracked (m)', (0.8, 2.2)),
        (axs[1], 'returns_uw', 'upwind returns per meter', (0.8, 2.2)),
        (axs[0], 'returns', 'returns per meter', (0.8, 2.2))
    ]
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = group_data[key]['ctrl']
        exp_vals = group_data[key]['exp']

        x_ctrl = np.random.normal(1, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(2, noise, size=len(exp_vals))
        ax.scatter(x_ctrl, ctrl_vals, color=colors[0], alpha=0.8, s=15, zorder=2)
        ax.scatter(x_exp, exp_vals, color=colors[1], alpha=0.8, s=15, zorder=2)

        stat, p_value = mannwhitneyu(ctrl_vals, exp_vals, alternative='two-sided')
        p_text = f'p = {p_value:.4g}'
        ax.text(1.5, max(max(ctrl_vals, default=0), max(exp_vals, default=0)) * 1.1, p_text,
                ha='center', va='bottom', fontsize=10, color=markc)
        mean_ctrl = np.mean(ctrl_vals)
        mean_exp = np.mean(exp_vals)
        sem_ctrl = sem(ctrl_vals) 
        sem_exp = sem(exp_vals) 
        ax.hlines(mean_ctrl, 0.9, 1.1, colors=markc, linewidth=3)
        ax.hlines(mean_exp, 1.9, 2.1, colors=markc, linewidth=3)
        ax.errorbar(1, np.mean(ctrl_vals), yerr=sem_ctrl, color=markc, capsize=0, linewidth=3)
        ax.errorbar(2, np.mean(exp_vals), yerr=sem_exp, color=markc, capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['off', 'on'], xticks=[1, 2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def silencing_inout_comparisons(folder_path, savename, size=(3,5), groups=4, side ='inside', colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True, ncols=groups)
    paired_data = {metric: {} for metric in ['speed', 'torts', 'pl', 'angvel']}
    params_list = exp_parameters(folder_path)
    time_stamps = {} 
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        d, d_in, d_out = inside_outside(exp_df)
        if side == 'inside':
            bout_dict = d_in
        elif side == 'outside':
            bout_dict = d_out
        match = re.search(r'Fly(\d+)', filename)
        if not match:
            continue
        fly_id = int(match.group(1))
        label  = 'exp' if 'light' in filename else 'ctrl'
        t_match = re.search(r'(\d{8}-\d{6})', filename)
        
        if not bout_dict:  
            metrics = {'speed': 0.0, 'uw_dist': 0.0, 'pl': 0.0, 'time': 0.0}
        else:
            speed_vals, tort_vals, pl_vals, av_vals = [], [], [], []
        for key, bout_df in bout_dict.items(): # NO RETURN TO EDGE CONDITION
            if bout_df['seconds'].iloc[-1] - bout_df['seconds'].iloc[0] > 0.5:     
                speed_vals.append(float(bout_df['speed'].mean()))
                tort_vals.append(float(get_a_bout_calc(bout_df, 'tortuosity')))
                pl_vals.append(float(get_a_bout_calc(bout_df, 'path_length')/1000))
                av_vals.append(float(bout_df['abs_angvel'].mean()))
        
            metrics = {
                'speed': float(np.mean(speed_vals)),
                'torts': float(np.mean(tort_vals)),
                'pl': float(np.mean(pl_vals)),
                'angvel': float(np.mean(av_vals)),
            }

        # Assign metrics to this fly/label
        for metric, value in metrics.items():
            paired_data[metric].setdefault(fly_id, {})[label] = value
        datasets = [
            (axs[0], 'speed', f'{side} speed (mm/s)', (0.8, 2.2)),
            (axs[1], 'torts', f'{side} tortuosity (a.u.)', (0.8, 2.2)),
            (axs[2], 'pl', f'{side} pathlength (m)', (0.8, 2.2)),
            (axs[3], 'angvel', f'{side} ang. velocity (rad/s)', (0.8, 2.2)),
        ]
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = []
        exp_vals = []
        for fly_id, data in paired_data[key].items():
            if 'exp' in data and 'ctrl' in data:
                ctrl_vals.append(data['ctrl'])
                exp_vals.append(data['exp'])
                ax.plot([1, 2], [data['ctrl'], data['exp']], color=markc, linewidth=1, alpha=0.5)  # Line between pairs
        x_ctrl = np.random.normal(1, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(2, noise, size=len(exp_vals))
        stat, p_value = wilcoxon(ctrl_vals, exp_vals)
        p_text = f'p = {p_value:.4g}' if p_value >= 0.001 else 'p < 0.001'
        ax.text(1.5, max(max(ctrl_vals, default=0), max(exp_vals, default=0)) * 1.1, p_text,
            ha='center', va='bottom', fontsize=10, color=markc)
        # ax.hlines(np.mean(ctrl_vals), 0.9, 1.1, colors=markc, linewidth=2)
        # ax.hlines(np.mean(exp_vals), 1.9, 2.1, colors=markc, linewidth=2)
        ax.plot([1,2], [np.mean(ctrl_vals), np.mean(exp_vals)], color=colors[1], linewidth=3)
        sem_ctrl = sem(ctrl_vals) if len(ctrl_vals) > 1 else 0
        sem_exp = sem(exp_vals) if len(exp_vals) > 1 else 0
        ax.errorbar(1, np.mean(ctrl_vals), yerr=sem_ctrl, color=colors[1], capsize=0, linewidth=3)
        ax.errorbar(2, np.mean(exp_vals), yerr=sem_exp, color=colors[1], capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['ctrl', 'exp'], xticks=[1,2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def silencing_re_scatter(folder_path, savename, size=(3,5), colors=['lightgrey', '#ff355e']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    dist1_list, pl1_list = [], []
    dist2_list, pl2_list = [], []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment     
        _, d_in, d_out = inside_outside_odor(exp_df)   
        if len(d_in) <= 1:
            continue

        for key, df in d_out.items():
            if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                dist = get_a_bout_calc(df, 'x_distance_from_plume')
                pl = get_a_bout_calc(df, 'path_length')
                if dist > 1:
                    if 'light' in filename:
                        plt.scatter(dist, dist/pl, color=colors[1], alpha=0.8)
                        dist2_list.append(dist)
                        pl2_list.append(dist/pl)
                    else:
                        plt.scatter(dist, dist/pl, color=colors[0], alpha=0.8)
                        dist1_list.append(dist)
                        pl1_list.append(dist/pl)
       
    dist1_array, pl1_array = np.array(dist1_list), np.array(pl1_list)
    dist2_array, pl2_array = np.array(dist2_list), np.array(pl2_list)
    if len(dist1_array) > 1:
        slope1, intercept1, r1, _, _ = linregress(dist1_array, pl1_array)
        r2_off = r1 ** 2
        x_fit1 = np.linspace(min(dist1_array), max(dist1_array), 100)
        y_fit1 = slope1 * x_fit1 + intercept1
        plt.plot(x_fit1, y_fit1, color=colors[0], linestyle='-', linewidth=2, label='LED off')
    if len(dist2_array) > 1:
        slope2, intercept2, r2, _, _ = linregress(dist2_array, pl2_array)
        r2_on = r2 ** 2
        x_fit2 = np.linspace(min(dist2_array), max(dist2_array), 100)
        y_fit2 = slope2 * x_fit2 + intercept2
        plt.plot(x_fit2, y_fit2, color=colors[1], linestyle='-', linewidth=2, label='LED on')
    print(f"LED off  : R² = {r2_off:.4f}")
    print(f"LED on   : R² = {r2_on:.4f}")
    plt.ylabel('return efficiency', fontsize=16, color=markc)
    plt.xlabel('furthest distance from plume (mm)', fontsize=16, color=markc)
    plt.yticks(fontsize=14, color=markc)
    plt.xticks(fontsize=14, color=markc)
    # axs.plot(np.linspace(0, 1000, 100), 2*(np.linspace(0, 1000, 100)), linestyle='--', color=markc)
    axs.set_xlim(-10, 200)
    axs.set_ylim(0, 0.5)
    legend = plt.legend(prop={'size': 10}, frameon=False)
    plt.setp(legend.get_texts(), color=markc)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def silencing_angles(folder_path, savename, size=(5,5), group='exp', colors=['lightgrey', '#ff355e']):
    fig, axs, markc = configure_white_polar(size=size, grid=False)
    params_list = exp_parameters(folder_path)
    entry_angles = [] 
    exit_angles = [] 
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if group == 'ctrl':
            if 'light' in filename:
                continue 
        elif group == 'exp':
            if 'light' not in filename:
                continue 
        _, di, do = inside_outside(exp_df)
        entries = []
        exits = []
        for k in sorted(di.keys()):  
            if k == 1:  
                continue
            else:  
                entry_df = get_entry_df(do.get(k-1), di[k], exp_df, window=0.5)
            if entry_df is not None:
                circmean = circmean_heading(entry_df, entries)
                circmean = mirror(circmean)
                entries[-1] = circmean 
                # plt.polar([0, circmean], [0, 1], color=colors[0], alpha=0.4, linewidth=0.5)
              
        for k in sorted(do.keys()):  
            entry_df = get_entry_df(di.get(k-1), do[k], exp_df, window=0.5)
            if entry_df is not None:
                circmean = circmean_heading(entry_df, exits)
                circmean = mirror(circmean)
                exits[-1] = circmean 
                # plt.polar([0, circmean], [0, 1], color=colors[0], alpha=0.4, linewidth=0.5)

        if len(exits) > 0:
            v = np.mean(np.column_stack((np.cos(exits), np.sin(exits))),axis=0)
            rho_f, phi_f = cart2pol(v[0], v[1])
            exit_angles.append(phi_f)
            plt.polar([0, phi_f], [0, 1], color=colors[1], alpha=0.4, linewidth=0.5)

        if len(entries) > 0:
            v = np.mean(np.column_stack((np.cos(entries), np.sin(entries))),axis=0)
            rho_f, phi_f = cart2pol(v[0], v[1])

            entry_angles.append(phi_f)
            plt.polar([0, phi_f], [0, 1], color=colors[0], alpha=0.4, linewidth=0.5)
              
    if len(exit_angles) > 0:
        summary_p1_vector = np.mean(np.column_stack((np.cos(exit_angles), np.sin(exit_angles))), axis=0)
        summary_p1_rho, summary_p1_phi = cart2pol(summary_p1_vector[0], summary_p1_vector[1])
        plt.polar([0, summary_p1_phi], [0, summary_p1_rho], color=colors[1], alpha=1, linewidth=5, solid_capstyle='round')
    if len(entry_angles) > 0:
        summary_p2_vector = np.mean(np.column_stack((np.cos(entry_angles), np.sin(entry_angles))), axis=0)
        summary_p2_rho, summary_p2_phi = cart2pol(summary_p2_vector[0], summary_p2_vector[1])
        plt.polar([0, summary_p2_phi], [0, summary_p2_rho], color=colors[0], alpha=1, linewidth=5, solid_capstyle='round')
    
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def average_trajectories(folder_path, savename, size=(5,5), groups=['ctrl', 'exp'],colors=['blue', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    data = {
        'ctrl': {'x_in': [], 'y_in': [], 'x_out': [], 'y_out': [], 'color': colors[0]},
        'exp': {'x_in': [], 'y_in': [], 'x_out': [], 'y_out': [], 'color': colors[1]}
    }

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        for group in groups:
            if group == 'ctrl' and 'light' in filename:
                continue 
            elif group == 'exp' and 'light' not in filename:
                continue 
            d, di, do = inside_outside(exp_df)
            x_in, y_in = average_trajectory_in(di, pts=5000)
            x_out, y_out = average_trajectory_out(do, pts=5000)
            if len(x_in) > 0:
                data[group]['x_in'].append(np.array(x_in))
            if len(y_in) > 0:
                data[group]['y_in'].append(np.array(y_in))
            if len(x_out) > 0:
                data[group]['x_out'].append(np.array(x_out))
            if len(y_out) > 0:
                data[group]['y_out'].append(np.array(y_out))

            if len(x_out) > 0 and len(y_out) > 0:
                axs.plot(x_out, y_out, color=data[group]['color'], alpha=0.5, linewidth=0.5)
            if len(x_in) > 0 and len(y_in) > 0:
                axs.plot(x_in, y_in, color=data[group]['color'], alpha=0.5, linewidth=0.5)

    for group in groups:
        if data[group]['x_in'] and data[group]['y_in']:  
            x_in = np.vstack([arr for arr in data[group]['x_in'] if arr.size > 0])
            y_in = np.vstack([arr for arr in data[group]['y_in'] if arr.size > 0])
            avg_x_in = np.mean(x_in, axis=0)
            avg_y_in = np.mean(y_in, axis=0)
            axs.plot(avg_x_in, avg_y_in, color=data[group]['color'], alpha=1, linewidth=1)

        if data[group]['x_out'] and data[group]['y_out']:  
            x_out = np.vstack([arr for arr in data[group]['x_out'] if arr.size > 0])
            y_out = np.vstack([arr for arr in data[group]['y_out'] if arr.size > 0])
            avg_x_out = np.mean(x_out, axis=0)
            avg_y_out = np.mean(y_out, axis=0)
            axs.plot(avg_x_out, avg_y_out, color=data[group]['color'], alpha=1, linewidth=1)
    
    # axs.set_aspect('equal')
    axs.axvline(0, color=markc, linewidth=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def ET_comparisons(folder_path, savename, size=(3,5), groups=2, colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_bw_plot(size=size, xaxis=False, ncols=groups)
    params_list = exp_parameters(folder_path)
    all_inside = {metric: [] for metric in ['plume_dists', 'orthog_dists', 'speeds', 'plume_speeds', 'orthog_speeds', 'time']}
    all_outside = {metric: [] for metric in ['plume_dists', 'orthog_dists', 'speeds', 'plume_speeds', 'orthog_speeds', 'time']}

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if 'strip' in filename:
            d, d_in, d_out = light_on_off(exp_df)
        else:
            if (exp_df['mfc3_stpt'] == 0).all():
                d, d_in, d_out = inside_outside_acv(exp_df)
            if (exp_df['mfc2_stpt'] == 0).all():
                d, d_in, d_out = inside_outside_oct(exp_df)
        if len(d_in) <=1:
            continue

        inside_data = {metric: [] for metric in all_inside}
        outside_data = {metric: [] for metric in all_outside}

        for key, df in d.items():
            ydisp = np.abs(df['ft_posy'].iloc[-1] - df['ft_posy'].iloc[0])
            xdisp = np.abs((df['ft_posx'] - df['ft_posx'].iloc[0])).max()
            speed = np.mean(df['speed'])
            yvel = np.mean(df['y-vel'])
            xvel = np.mean(np.abs(df['x-vel']))
            duration = get_a_bout_calc(df, 'duration')
            # ydisp = df['ft_posy'].iloc[-1] - df['ft_posy'].iloc[0]
            # xdisp = np.abs(df['ft_posx'].iloc[-1] - df['ft_posx'].iloc[0])
            # speed = np.mean(df['speed'])
            # yvel = np.mean(df['y-vel'])
            # xvel = np.mean(df['x-vel'])

            if key % 2 > 0:
                inside_data['time'].append(duration)
                inside_data['plume_dists'].append(ydisp)
                inside_data['orthog_dists'].append(xdisp)
                inside_data['speeds'].append(speed)
                inside_data['plume_speeds'].append(yvel)
                inside_data['orthog_speeds'].append(xvel)
            else:
                if return_to_edge(df):
                    outside_data['time'].append(duration)
                    outside_data['plume_dists'].append(ydisp)
                    outside_data['orthog_dists'].append(xdisp)
                    outside_data['speeds'].append(speed)
                    outside_data['plume_speeds'].append(yvel)
                    outside_data['orthog_speeds'].append(xvel)

        # Average per fly
        inside_avg = {metric: np.nanmean(values) if len(values) else np.nan
              for metric, values in inside_data.items()}
        outside_avg = {metric: np.nanmean(values) if len(values) else np.nan
              for metric, values in outside_data.items()}

        for metric in all_inside:
            all_inside[metric].append(inside_avg[metric])
            all_outside[metric].append(outside_avg[metric])

        datasets = [
            (axs[0], 'time', 'time (s)', (0.8, 2.2)),
            (axs[1], 'plume_dists', 'plume distance (mm)', (0.8, 2.2)),
            (axs[2], 'orthog_dists', 'orthogonal distance (mm)', (0.8, 2.2)),
            (axs[3], 'speeds', 'speed (mm/s)', (0.8, 2.2)),
            (axs[4], 'plume_speeds', 'plume speed (mm/s)', (0.8, 2.2)),
            (axs[5], 'orthog_speeds', 'orthogonal speed (mm/s)',(0.8, 2.2))
        ]

        for ax, metric, ylabel, xlim in datasets:ax.plot([1, 2], [outside_avg[metric], inside_avg[metric]], color=markc, linewidth=1)

    for ax, metric, ylabel, xlim in datasets:
        in_vals = np.array(all_inside[metric])
        out_vals = np.array(all_outside[metric])
        mask = np.isfinite(in_vals) & np.isfinite(out_vals)
        in_vals = in_vals[mask]
        out_vals = out_vals[mask]

        print(f"{metric}: n_pairs={mask.sum()}")  
    
        mean_in = np.nanmean(in_vals)
        mean_out = np.nanmean(out_vals)
        sem_in = sem(in_vals, nan_policy='omit')
        sem_out = sem(out_vals, nan_policy='omit')
        ax.errorbar(1, mean_out, yerr=sem_out, color=colors[0], capsize=0, linewidth=3)
        ax.errorbar(2, mean_in, yerr=sem_in, color=colors[1], capsize=0, linewidth=3)
        ax.hlines(mean_out, 0.9, 1.1, colors=colors[0], linewidth=3)
        ax.hlines(mean_in, 1.9, 2.1, colors=colors[1], linewidth=3)
        y_all = np.concatenate([in_vals, out_vals])
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size == 0:
            ylim = (0, 1)
        else:
            y_min, y_max = y_all.min(), y_all.max()
            margin = 0.1 * (y_max - y_min) if y_max != y_min else 1
            ylim = (y_min - margin, y_max + margin)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlim(xlim)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['out', 'in'])
        if len(in_vals) > 0 and len(out_vals) > 0:
            stat, p = wilcoxon(out_vals, in_vals, alternative='two-sided')
            stars = ''
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                stars = 'ns'
            max_y = max(np.max(in_vals), np.max(out_vals))
            y_text = max_y + 0.05 * (ylim[1] - ylim[0])
            ax.text(1.5, y_text, f'{stars} (p={p:.4g})', ha='center', va='bottom', fontsize=9)
    
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def orco_acv_comparisons(folder_path, savename, size=(3,5), colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_white_plot(size=size, xaxis=False)
    params_list = exp_parameters(folder_path)
    acv_inside = []
    orco_inside = []

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        print(filename)
        if 'strip' in filename:
            d, d_in, d_out = light_on_off(exp_df)
        else:
            if (exp_df['mfc3_stpt'] == 0).all():
                d, d_in, d_out = inside_outside_acv(exp_df)
            if (exp_df['mfc2_stpt'] == 0).all() and (df['mfc3_stpt'] == 0).all():
                d, d_in, d_out = inside_outside_oct(exp_df)
        
        durations = []
        for key, df in d_in.items():
            if return_to_edge(df):
                duration = get_a_bout_calc(df, 'duration')
                if 'strip' in filename:
                    durations.append(duration)
                else:
                    durations.append(duration)

        avg_duration = sum(durations) / len(durations)
        if 'strip' in filename:
            orco_inside.append(avg_duration)
        else:
            acv_inside.append(avg_duration)

    noise = 0.05
    x_acv = np.random.normal(1, noise, size=len(acv_inside))
    x_orco = np.random.normal(2, noise, size=len(orco_inside))
    stat, p_value = mannwhitneyu(acv_inside, orco_inside, alternative='two-sided')
    p_text = f'p = {p_value:.4g}' if p_value >= 0.001 else 'p < 0.001'
    axs.text(1.5, max(max(acv_inside, default=0), max(orco_inside, default=0)) * 1.1, p_text,
        ha='center', va='bottom', fontsize=10, color=markc)
    mean_acv = np.nanmean(acv_inside)
    mean_orco = np.nanmean(orco_inside)
    sem_acv = sem(acv_inside, nan_policy='omit')
    sem_orco = sem(orco_inside, nan_policy='omit')
    axs.errorbar(1, mean_acv, yerr=sem_acv, color=markc, capsize=0, linewidth=3)
    axs.errorbar(2, mean_orco, yerr=sem_orco, color=markc, capsize=0, linewidth=3)
    axs.hlines(mean_acv, 0.9, 1.1, colors=markc, linewidth=3)
    axs.hlines(mean_orco, 1.9, 2.1, colors=markc, linewidth=3)
    plt.scatter(x_acv, acv_inside, color=colors[0], alpha=0.6)
    plt.scatter(x_orco, orco_inside, color=colors[1], alpha=0.6)
    axs.tick_params(axis='x', which='both', length=0)
    axs.spines['bottom'].set_visible(False)
    # axs.set_xticks([1, 2])
    style_ax(axs, markc, title=None, xtick_labels=['acv', 'light strip'], xticks=[1,2], ylabel='avg. time inside (s)', xlim=(0.7,2.3))
    sns.despine(offset=10)
    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')



def silencing_return_efficiency(folder_path, savename, size=(3,5), colors=['lightgrey', '#7ed4e6']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    off_avg = []
    on_avg = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        print(filename)
        mask = (exp_df['led1_stpt'] == 0) & (exp_df['instrip'] == True)
        cutoff_idx = exp_df.index[mask][0]
        df_off =  exp_df.loc[cutoff_idx:].copy()
        df_on =  exp_df.loc[:cutoff_idx].copy()
        _, off_i, off_o = inside_outside(df_off)
        _, on_i, on_o = inside_outside(df_on)
        print(off_o.keys())
        print(on_o.keys())

        res_off = []
        res_on = []

        for key, df in off_o.items():
            if return_to_edge(df):
                # if df['seconds'].iloc[-1] - df['seconds'].iloc[0] > 0.5:
                    xdist_off = get_a_bout_calc(df, 'x_distance_from_plume') 
                    pl_off = get_a_bout_calc(df, 'path_length') 
                    res_off.append(xdist_off/pl_off)
        for key, df in on_o.items():
            if return_to_edge(df):
                #  if df['seconds'].iloc[-1] - df['seconds'].iloc[0] > 0.5:
                    xdist_on = get_a_bout_calc(df, 'x_distance_from_plume') 
                    pl_on = get_a_bout_calc(df, 'path_length') 
                    res_on.append(xdist_on/pl_on)

        fly_off_avg = np.mean(res_off)
        fly_on_avg = np.mean(res_on)
        plt.scatter(fly_off_avg, fly_on_avg, color=colors[0], alpha=0.6)
        off_avg.append(fly_off_avg)
        on_avg.append(fly_on_avg)
    
    off_array, on_array = np.array(off_avg), np.array(on_avg)
    slope, intercept, r, _, _ = linregress(off_array, on_array)
    r2 = r ** 2
    print(r2)
    x_fit = np.linspace(min(off_array), max(off_array), 100)
    y_fit = slope * x_fit + intercept
    # plt.plot(x_fit, y_fit, color=markc, linestyle='-', linewidth=2)

    max_val = max(off_array.max(), on_array.max())
    plt.plot([0, max_val], [0, max_val], color='grey', linestyle='--', linewidth=1, zorder=1)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.ylabel('return efficiency, LED off', fontsize=12, color=markc)
    plt.xlabel('return efficiency, LED on', fontsize=12, color=markc)
    plt.yticks(fontsize=12, color=markc)
    plt.xticks(fontsize=12, color=markc)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')


def silencing_re_vs_entries(folder_path, savename, size=(3,5), colors=['lightgrey', '#7ed4e6']):
    params_list = exp_parameters(folder_path)
    fig, axs, markc = configure_bw_plot(size=size, xaxis=True)
    n_returns_off = []
    fold_change = []

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.copy()
        ypos = exp_df['ft_posy'] - yo
        over_mask = ypos > 1000
        if over_mask.any():
            trunc_idx = exp_df.index[over_mask][0]
            exp_df = exp_df.loc[:trunc_idx].copy()

        mask = (exp_df['led1_stpt'] == 0) & (exp_df['instrip'] == True)
        cutoff_idx = exp_df.index[mask][0]
        df_off = exp_df.loc[cutoff_idx:].copy()
        df_on = exp_df.loc[:cutoff_idx].copy()
        _, off_i, off_o = inside_outside(df_off)
        _, on_i, on_o = inside_outside(df_on)
   
        res_off = []
        res_on = []


        for key, df in off_o.items():
            if return_to_edge(df):
                # if df['seconds'].iloc[-1] - df['seconds'].iloc[0] > 0.5:
                xdist_off = get_a_bout_calc(df, 'x_distance_from_plume')
                pl_off = get_a_bout_calc(df, 'path_length')
                if pl_off > 0:
                    res_off.append(xdist_off / pl_off)

        for key, df in on_o.items():
            if return_to_edge(df):
                # if df['seconds'].iloc[-1] - df['seconds'].iloc[0] > 0.5:
                xdist_on = get_a_bout_calc(df, 'x_distance_from_plume')
                pl_on = get_a_bout_calc(df, 'path_length')
                if pl_on > 0:
                    res_on.append(xdist_on / pl_on)

        fly_off_avg = np.mean(res_off)
        fly_on_avg = np.mean(res_on)
        n_off = len(res_off)
        print(filename)
        print(n_off)
        fc = fly_on_avg / fly_off_avg
        plt.scatter(n_off, fc, color=colors[0], alpha=0.6)
        n_returns_off.append(n_off)
        fold_change.append(fc)

    n_array, fc_array = np.array(n_returns_off), np.array(fold_change)
    slope, intercept, r, _, _ = linregress(n_array, fc_array)
    r2 = r ** 2
    print(r2)
    x_fit = np.linspace(n_array.min(), n_array.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color=markc, linestyle='-', linewidth=1)
    plt.axhline(1, color='grey', linestyle='--', linewidth=1, zorder=1)
    plt.xlabel('number of returns, LED off', fontsize=12, color=markc)
    plt.ylabel('return eff. fold change', fontsize=12, color=markc)
    plt.yticks(fontsize=12, color=markc)
    plt.xticks(fontsize=12, color=markc)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def orco_lightstrip_ypos_comparison(folder_path, savename, size=(3,6), keyword='NW', colors=['black', '#6acff6']):
    fig, axs, markc = configure_white_plot(size=size, xaxis=False)
    params_list = exp_parameters(folder_path)
    yavg_nw_list = []
    yavg_wind_list = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        ypos = df_light['ft_posy'] - yo
        yavg = np.mean(ypos)
        if keyword in filename:
            yavg_nw_list.append(yavg)
        else:
            yavg_wind_list.append(yavg)

    noise = 0.05
    x_nw = np.random.normal(1, noise, size=len(yavg_nw_list))
    x_wind = np.random.normal(2, noise, size=len(yavg_wind_list))

    plt.scatter(x_nw, yavg_nw_list, color=colors[0], alpha=0.6, label='NW')
    plt.scatter(x_wind, yavg_wind_list, color=colors[1], alpha=0.6, label='Wind')

    mean_nw = np.mean(yavg_nw_list) if len(yavg_nw_list) else np.nan
    mean_wind = np.mean(yavg_wind_list) if len(yavg_wind_list) else np.nan
    sem_nw = sem(yavg_nw_list, nan_policy='omit')
    sem_wind = sem(yavg_wind_list, nan_policy='omit')
    axs.errorbar(1, mean_nw, yerr=sem_nw, color=markc, capsize=0, linewidth=3)
    axs.errorbar(2, mean_wind, yerr=sem_wind, color=markc, capsize=0, linewidth=3)
    axs.hlines(mean_nw, 0.9, 1.1, colors=markc, linewidth=3)
    axs.hlines(mean_wind, 1.9, 2.1, colors=markc, linewidth=3)
    stat, p_value = mannwhitneyu(yavg_nw_list, yavg_wind_list, alternative='two-sided')
    p_text = f'p = {p_value:.4g}'
    axs.text(1.5, max(max(yavg_nw_list, default=0), max(yavg_wind_list, default=0)) * 1.1, p_text,
        ha='center', va='bottom', fontsize=10, color=markc)

    axs.tick_params(axis='x', which='both', length=0)
    axs.spines['bottom'].set_visible(False)
    # axs.set_xticks([1, 2])
    style_ax(axs, markc, title=None, xtick_labels=['no wind', 'wind'], xticks=[1,2], ylabel='avg. y-position (mm)', xlim=(0.7,2.3))
    sns.despine(offset=10)
    sns.despine(offset=10)
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def orco_acv_ET_comparisons(folder_path, savename, size=(12,6), colors=['#c1ffc1', 'red']):
    params_list = exp_parameters(folder_path)

    metrics = ['plume_dists', 'orthog_dists', 'speeds', 'plume_speeds', 'orthog_speeds', 'time']
    labels = ['plume distance (mm)', 'orthogonal distance (mm)', 'speed (mm/s)', 'plume speed (mm/s)', 'orthogonal speed (mm/s)', 'time (s)']

    data = {
        'inside': {'strip': {m: [] for m in metrics}, 'nostrip': {m: [] for m in metrics}},
        'outside': {'strip': {m: [] for m in metrics}, 'nostrip': {m: [] for m in metrics}}
    }

    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        if 'strip' in filename:
            d, d_in, d_out = light_on_off(exp_df)
        else:
            if (exp_df['mfc3_stpt'] == 0).all():
                d, d_in, d_out = inside_outside_acv(exp_df)
            if (exp_df['mfc2_stpt'] == 0).all() and (df['mfc3_stpt'] == 0).all():
                d, d_in, d_out = inside_outside_oct(exp_df)
        print(filename)
        print(len(d))
        print(len(d_in))
        print(len(d_out))

        group = 'strip' if 'strip' in filename else 'nostrip'

        inside_data = {m: [] for m in metrics}
        outside_data = {m: [] for m in metrics}

        for key, df in d.items():
            ydisp = np.abs(df['ft_posy'].iloc[-1] - df['ft_posy'].iloc[0])
            xdisp = np.abs((df['ft_posx'] - df['ft_posx'].iloc[0]))
            xdisp = np.max(xdisp)
            speed = np.mean(df['speed'])
            yvel = np.mean(df['y-vel'])
            xvel = (np.mean(np.abs(df['x-vel']))) 
            duration = get_a_bout_calc(df, 'duration')

            if key % 2 > 0:
                if return_to_edge(df):
                    inside_data['plume_dists'].append(ydisp)
                    inside_data['orthog_dists'].append(xdisp)
                    inside_data['speeds'].append(speed)
                    inside_data['plume_speeds'].append(yvel)
                    inside_data['orthog_speeds'].append(xvel)
                    inside_data['time'].append(duration)
            else:
                outside_data['orthog_dists'].append(xdisp)
                outside_data['orthog_speeds'].append(xvel)
                outside_data['speeds'].append(speed)
                outside_data['plume_speeds'].append(yvel)
                if return_to_edge(df):
                    outside_data['plume_dists'].append(ydisp)
                    outside_data['time'].append(duration)
                    

        # Per-fly averages
        inside_avg  = {m: np.nanmean(v) if len(v) else np.nan for m, v in inside_data.items()}
        outside_avg = {m: np.nanmean(v) if len(v) else np.nan for m, v in outside_data.items()}

        for m in metrics:
            data['inside'][group][m].append(inside_avg[m])
            data['outside'][group][m].append(outside_avg[m])

    fig, axs, markc = configure_white_plot(size=size, nrows=2, ncols=len(metrics))
    axs = np.array(axs).reshape(2, len(metrics))
    noise = 0.05

    for col, (metric, ylabel) in enumerate(zip(metrics, labels)):
        # ----- Inside (top row) -----
        ax_in = axs[0, col]
        n_in = np.array(data['inside']['nostrip'][metric])
        s_in = np.array(data['inside']['strip'][metric])

        n_in = n_in[np.isfinite(n_in)]
        s_in = s_in[np.isfinite(s_in)]

        x1 = np.random.normal(1, noise, len(n_in))
        x2 = np.random.normal(2, noise, len(s_in))

        if len(n_in) > 0 and len(s_in) > 0:
            stat, p = mannwhitneyu(n_in, s_in, alternative='two-sided')
            p_text = f"p={p:.4g}" if p >= 0.001 else "p<0.001"
        else:
            p_text = "n/a"
            
        ax_in.scatter(x1, n_in, color=colors[0], alpha=0.6)
        ax_in.scatter(x2, s_in, color=colors[1], alpha=0.6)
        ax_in.errorbar(1, np.nanmean(n_in), yerr=sem(n_in, nan_policy='omit'), color='black', linewidth=2)
        ax_in.errorbar(2, np.nanmean(s_in), yerr=sem(s_in, nan_policy='omit'), color='black', linewidth=2)
        ax_in.hlines(np.nanmean(n_in), 0.9, 1.1, colors='black', linewidth=2)
        ax_in.hlines(np.nanmean(s_in), 1.9, 2.1, colors='black', linewidth=2)
        ax_in.set_title(ylabel, fontsize=12)
        ax_in.set_xticks([1, 2])
        ax_in.set_xticklabels(['acv', 'LED strip'])
        ymax = np.nanmax(np.concatenate([n_in, s_in])) if len(n_in)+len(s_in) > 0 else 1
        ax_in.text(1.5, ymax * 1.05, p_text, ha='center', fontsize=9)

        # ----- Outside (bottom row) -----
        ax_out = axs[1, col]
        n_out = np.array(data['outside']['nostrip'][metric])
        s_out = np.array(data['outside']['strip'][metric])

        n_out = n_out[~np.isnan(n_out)]
        s_out = s_out[~np.isnan(s_out)]

        x3 = np.random.normal(1, noise, len(n_out))
        x4 = np.random.normal(2, noise, len(s_out))

        if len(n_out) > 0 and len(s_out) > 0:
            stat2, p2 = mannwhitneyu(n_out, s_out, alternative='two-sided')
            p_text2 = f"p={p2:.4g}" if p2 >= 0.001 else "p<0.001"
        else:
            p_text2 = "n/a"

        ax_out.scatter(x3, n_out, color=colors[0], alpha=0.6)
        ax_out.scatter(x4, s_out, color=colors[1], alpha=0.6)
        ax_out.errorbar(1, np.nanmean(n_out), yerr=sem(n_out, nan_policy='omit'), color='black', linewidth=2)
        ax_out.errorbar(2, np.nanmean(s_out), yerr=sem(s_out, nan_policy='omit'), color='black', linewidth=2)
        ax_out.hlines(np.nanmean(n_out), 0.9, 1.1, colors='black', linewidth=2)
        ax_out.hlines(np.nanmean(s_out), 1.9, 2.1, colors='black', linewidth=2)
        ax_out.set_xticks([1, 2])
        ax_out.set_xticklabels(['acv', 'LED strip'])
        ymax2 = np.nanmax(np.concatenate([n_out, s_out])) if len(n_out)+len(s_out) > 0 else 1
        ax_out.text(1.5, ymax2 * 1.05, p_text2, ha='center', fontsize=9)

    axs[0,0].set_ylabel('inside', fontsize=12)
    axs[1,0].set_ylabel('outside', fontsize=12)

    fig.tight_layout()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')



def gaussian_comparisons(folder_path, savename, size=(3,5), groups=4, side ='inside', impose_edge=True, keywords = ['ctrl', 'ablated'], colors=['#c1ffc1', 'red']):
    fig, axs, markc = configure_white_plot(size=size, xaxis=False, ncols=groups)
    paired_data = {metric: {} for metric in ['speed', 'uw_dist', 'pl', 'time']}
    params_list = exp_parameters(folder_path)
    time_stamps = {} 
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, bl_df, exp_df, xo, yo, plume_color = this_experiment
        d, d_in, d_out = inside_outside(exp_df)
        _, d_in25, d_out25 = inside_outside_xpos(exp_df, 25)
        if impose_edge and side == 'inside':
            bout_dict = d_in25
        elif impose_edge and side == 'outside':
            bout_dict = d_out25
        elif not impose_edge and side == 'inside': 
            bout_dict = d_in
        elif not impose_edge and side == 'outside': 
            bout_dict = d_out
         
        match = re.search(r'Fly(\d+)', filename)
        if not match:
            continue
        fly_id = int(match.group(1))
        label  = 'sharp' if 'acv' in filename else 'graded'
        t_match = re.search(r'(\d{8}-\d{6})', filename)
        
        if not bout_dict:  
            metrics = {'speed': 0.0, 'uw_dist': 0.0, 'pl': 0.0, 'time': 0.0}
        else:
            speed_vals, uw_dists, pl_vals, time_vals = [], [], [], []
        for key, bout_df in bout_dict.items():
            speed_vals.append(float(bout_df['speed'].mean()))
            uw_dists.append(float(bout_df['ft_posy'].iloc[-1] - bout_df['ft_posy'].iloc[0])/1000)
            pl_vals.append(float(get_a_bout_calc(bout_df, 'path_length')/1000))
            time_vals.append(float(get_a_bout_calc(bout_df, 'duration')))
        
            metrics = {
                'speed': float(np.mean(speed_vals)),
                'uw_dist': float(np.mean(uw_dists)),
                'pl': float(np.mean(pl_vals)),
                'time': float(np.mean(time_vals)),
            }

        # Assign metrics to this fly/label
        for metric, value in metrics.items():
            paired_data[metric].setdefault(fly_id, {})[label] = value
        datasets = [
            (axs[0], 'speed', f'{side} speed (mm/s)', (0.8, 2.2)),
            (axs[1], 'uw_dist', f'{side} upwind distance (mm)', (0.8, 2.2)),
            (axs[2], 'pl', f'{side} pathlength (m)', (0.8, 2.2)),
            (axs[3], 'time', f'{side} time (s)', (0.8, 2.2)),
        ]
    noise = 0.05
    for ax, key, ylabel, xlim in datasets:
        ctrl_vals = []
        exp_vals = []
        for fly_id, data in paired_data[key].items():
            if 'sharp' in data and 'graded' in data:
                ctrl_vals.append(data['sharp'])
                exp_vals.append(data['graded'])
                ax.plot([1, 2], [data['sharp'], data['graded']], color=markc, linewidth=1, alpha=0.5)  # Line between pairs
        x_ctrl = np.random.normal(1, noise, size=len(ctrl_vals))
        x_exp = np.random.normal(2, noise, size=len(exp_vals))
        stat, p_value = wilcoxon(ctrl_vals, exp_vals)
        p_text = f'p = {p_value:.4g}' if p_value >= 0.001 else 'p < 0.001'
        ax.text(1.5, max(max(ctrl_vals, default=0), max(exp_vals, default=0)) * 1.1, p_text,
            ha='center', va='bottom', fontsize=10, color=markc)
        # ax.hlines(np.mean(ctrl_vals), 0.9, 1.1, colors=markc, linewidth=2)
        # ax.hlines(np.mean(exp_vals), 1.9, 2.1, colors=markc, linewidth=2)
        ax.plot([1,2], [np.mean(ctrl_vals), np.mean(exp_vals)], color=colors[1], linewidth=3)
        sem_ctrl = sem(ctrl_vals) if len(ctrl_vals) > 1 else 0
        sem_exp = sem(exp_vals) if len(exp_vals) > 1 else 0
        ax.errorbar(1, np.mean(ctrl_vals), yerr=sem_ctrl, color=colors[1], capsize=0, linewidth=3)
        ax.errorbar(2, np.mean(exp_vals), yerr=sem_exp, color=colors[1], capsize=0, linewidth=3)
        style_ax(ax, markc, None, xtick_labels=['sharp', 'graded'], xticks=[1,2], ylabel=ylabel, xlim=xlim)

    sns.despine(offset=10)
    fig.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')
   


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    if len(centers) < 2:
        step = 1.0
    else:
        step = float(np.diff(centers).mean())
    return np.concatenate(([centers[0] - step/2], centers + step/2))

def _hist2d_from_df(df, xcol, ycol, x_centers, y_centers, x0=0.0, y0=0.0):
    """2D histogram using bin centers; returns array shaped (len(y), len(x)) for imshow."""
    x = np.abs(df[xcol].to_numpy() - x0)
    y = df[ycol].to_numpy() - y0
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    xe = _centers_to_edges(x_centers)
    ye = _centers_to_edges(y_centers)
    H, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    return H.T  # rows=y, cols=x for imshow

def occupancy_heatmap_all_files(
    folder_path,
    xlim, ylim,                 # e.g., (-200, 200), (0, 1000)
    res=5.0,                    # bin size in your position units (mm)
    xcol="ft_posx",
    ycol="ft_posy",
    include=("exp_df","df_odor"),  # which tables to include per experiment
    normalize=True,             # True -> probability map; False -> raw counts
    percentile_range=(0, 95),   # for color scaling on nonzero bins
    cmap="Greys",
    figure_size=(8,10),
    save_to=None                # path to save (e.g., ".../all_files_occupancy.pdf"); None -> just show
):
    """
    Build a SINGLE occupancy heatmap by summing over all files/experiments in folder_path.
    Uses exp_parameters(folder_path) to get per-file DataFrames + (xo, yo) offsets.
    """
    params_list = exp_parameters(folder_path)

    # shared grid for everyone (guarantees comparability)
    x_centers = np.arange(xlim[0], xlim[1] + res, res, dtype=float)
    y_centers = np.arange(ylim[0], ylim[1] + res, res, dtype=float)

    # accumulator
    H_all = np.zeros((len(y_centers), len(x_centers)), dtype=float)

    # sum occupancy from requested phases across all files
    for this_experiment in params_list:
        (
            figure_folder, filename, df, df_odor, df_light,
            bl_df, exp_df, xo, yo, plume_color
        ) = this_experiment

        # pick which per-file DataFrames to include
        name_to_df = {
            "df": df,
            "df_odor": df_odor,
            "df_light": df_light,
            "bl_df": bl_df,
            "exp_df": exp_df,
        }
        for key in include:
            dfi = name_to_df.get(key, None)
            if dfi is not None and len(dfi):
                # dfi = dfi.copy()
                # dfi[xcol] = dfi[xcol] - xo
                # dfi[ycol] = dfi[ycol] - yo
                # dfi[xcol] = np.abs(dfi[xcol])
                H_all += _hist2d_from_df(dfi, xcol, ycol, x_centers, y_centers, x0=xo, y0=yo)

    # normalize if desired
    if normalize and H_all.sum() > 0:
        H_plot = H_all / H_all.sum()
    else:
        H_plot = H_all

    # robust color scaling on nonzero entries
    nz = H_plot[H_plot > 0]
    if nz.size:
        vmin = np.percentile(nz, percentile_range[0])
        vmax = np.percentile(nz, percentile_range[1])
        if vmax <= vmin:
            vmin, vmax = nz.min(), nz.max()
    else:
        vmin, vmax = 0.0, 1.0

    # --- figure ---
    fig, ax = plt.subplots(figsize=figure_size)
    extent = (x_centers.min(), x_centers.max(), y_centers.min(), y_centers.max())
    im = ax.imshow(
        H_plot,
        cmap=cmap,
        origin="lower",
        extent=extent,
        vmin=vmin, vmax=vmax,
        aspect="auto",
        rasterized=True
    )
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("x-position (mm)")
    ax.set_ylabel("y-position (mm)")
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("occupancy" + (" (prob.)" if normalize else ""))

    sns.despine()
    fig.tight_layout()
    plt.show()
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    fig.savefig(save_to, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {save_to}")
    

    return H_all, (x_centers, y_centers)