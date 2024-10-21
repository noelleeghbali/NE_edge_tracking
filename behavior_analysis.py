import pandas as pd
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from PIL import Image
from scipy import interpolate
from scipy import stats
import os
import seaborn as sns


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
    x = df.ft_posx.to_numpy()
    x = x - x[-1]
    circmean_value = stats.circmean(df.ft_heading, low=-np.pi, high=np.pi, axis=None, nan_policy='omit')
    means_list.append(circmean_value)
    return circmean_value

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

def get_last_second(df):
    last_time = df['seconds'].iloc[-1]
    return df[df['seconds'] >= (last_time - 1)]
    
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
            # Filter the df to include datapoints only when odor is being delivered, assigning df_odor based
            # on which mass flow controller is active, along with a corresponding plume color.
            if (df['mfc2_stpt'] == 0).all():
                df_odor = df.where(df.mfc3_stpt > 0)
                plume_color = '#7ed4e6'
            if (df['mfc3_stpt'] == 0).all():
                df_odor = df.where(df.mfc2_stpt > 0)
                plume_color = '#fbceb1'
            if (df['mfc2_stpt'] == 0).all() and (df['mfc3_stpt'] == 0).all():
                df_odor = None
                plume_color = 'white'
            # Filter the df to include datapoints only when optogenetic light is being delivered (light = on when led1 = 0.0)
            df_light = df.where(df.led1_stpt == 0.0)
            if df_odor is None:
                # Create an index for the first instance of light on (exp start), and filter the df to start at this index
                first_on_index = df[df['led1_stpt'] == 0.0].index[0]
                exp_df = df.loc[first_on_index:]
                # Establish coordinates of the subject's origin at the exp start
                xo = exp_df.iloc[0]['ft_posx']
                yo = exp_df.iloc[0]['ft_posy']
                # Append the results for the current file to the list
                params_list.append([figure_folder, filename, df_odor, df_light, exp_df, xo, yo, plume_color])
            else:
                # Create an index for the first instance of odor on (exp start), and filter the df to start at this index
                first_on_index = df[df['odor_on']].index[0]
                exp_df = df.loc[first_on_index:]
                # Establish coordinates of the subject's origin at the exp start
                xo = exp_df.iloc[0]['ft_posx']
                yo = exp_df.iloc[0]['ft_posy']
                # Append the results for the current file to the list
                params_list.append([figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color])

    return params_list

def configure_bw_plot(size=(4,6), xaxis=False):
    fig, axs = plt.subplots(1, 1, figsize=size)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    fig.patch.set_facecolor('black')  # Set background to black
    axs.set_facecolor('black')  # Set background of plotting area to black
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.rcParams['text.color'] = 'white'
    axs.spines['bottom'].set_visible(xaxis)
    axs.spines['bottom'].set_color('white')
    axs.spines['left'].set_color('white')
    axs.spines['left'].set_linewidth(2)
    axs.tick_params(axis='x', colors='white')
    axs.tick_params(axis='y', colors='white')
    if xaxis:
        axs.xaxis.label.set_color('white')
        axs.tick_params(axis='x', colors='white')
    plt.gca().spines['left'].set_linewidth(2)
    return fig, axs

def trajectory_plotter(folder_path, strip_width, strip_length, plume_start, xlim, ylim, led, hlines=[], select_file=None, plot_type='odor', save=False):
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
            plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air')
            plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only')
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on')
            plt.gca().add_patch(patches.Rectangle((-strip_width / 2, plume_start), strip_width, strip_length, facecolor=plume_color, alpha=0.5))
            savename = filename + '_odor_trajectory.pdf'
        # In a light plume, plot the trajectroy when the animal is in the light
        elif plot_type == 'odorless':
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
        plt.legend()
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
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        # If a file is specified and the current file is not the specified one, skip to the next iteration
        if select_file and filename != select_file:
            continue
        fig, axs = configure_bw_plot(size=(10,10), xaxis=True)
        # In an odor plume, plot the trajectory when the animal is in the odor
        if plot_type == 'odor':
            plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='grey', label='clean air')
            plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='lightgrey', label='odor only')
            plt.plot(df_light['ft_posx'] - xo, df_light['ft_posy'] - yo, color=ledc, label='light on')
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

def pulse_return_efficiency(folder_path, savename, size=(5, 5), groups=3):
    fig, axs = configure_bw_plot(size=size, xaxis=False)
    returns = {f'returns_r{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        #last_on_index = exp_df[exp_df['ft_posy'] - yo 
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000] # account for active tracking only
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        pl = get_a_bout_calc(exp_df, 'path_length') / 1000
        counts = {f'returns_{i+1}': 0 for i in range(groups)}
        if len(d_on) < 1:
            for key, df in d_out.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                    counts['returns_1'] += 1
            returns['returns_r1'].append(counts['returns_1'] / pl)
        elif len(d_on) == 1:
            for key, df in d_out.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                    counts['returns_2'] += 1
            returns['returns_r2'].append(counts['returns_2'] / pl)
        elif len(d_on) > 1:
            for key, df in d_out.items():
                if df['seconds'].iloc[-1] - df['seconds'].iloc[0] >= 0.5 and return_to_edge(df):
                    counts['returns_3'] += 1
            returns['returns_r3'].append(counts['returns_3'] / pl)
            
    averages = [sum(returns[f'returns_r{i+1}']) / len(returns[f'returns_r{i+1}']) if returns[f'returns_r{i+1}'] else 0 for i in range(groups)]
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(returns[f'returns_r{i+1}'])) for i in range(groups)]
    for i in range(groups):
        plt.scatter(x_values[i], returns[f'returns_r{i+1}'], color='grey', alpha=0.5)
    plt.plot(range(1, groups+1), averages, color='white')
    # for i in range(groups):
    #     plt.scatter(i+1, averages[i], color='none', edgecolor='white', marker='o', linewidth=2, s=100)
    plt.ylabel('returns per meter', fontsize=18, color='white')
    plt.yticks(fontsize=14, color='white')
    axs.set_xticks(range(1, groups+1))
    axs.set_xticklabels(['no activation', 'single activation', 'multiple activation'], fontsize=16, color='white', rotation=45)
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

def pulse_uw_tracking(folder_path, savename, size=(5, 5), groups=3):
    fig, axs = configure_bw_plot(size=size, xaxis=False)
    dists = {f'dists_{i+1}': [] for i in range(groups)}
    params_list = exp_parameters(folder_path)
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        exp_df = exp_df.loc[exp_df['ft_posy'] - yo <= 1000] # cut off at 1000
        df_odor = df_odor.loc[df_odor['ft_posy'] - yo <= 1000] # cut off at 1000
        de, d_in, d_out = inside_outside(exp_df)
        dl, d_on, d_off = light_on_off(exp_df)
        if len(d_on) < 1:
            uw = df_odor['ft_posy'].max() - yo
            dists['dists_1'].append(uw)
        elif len(d_on) == 1:
            uw = df_odor['ft_posy'].max() - yo
            dists['dists_2'].append(uw)
        elif len(d_on) > 1:
            uw = df_odor['ft_posy'].max() - yo
            dists['dists_3'].append(uw)
    averages = [sum(dists[f'dists_{i+1}']) / len(dists[f'dists_{i+1}']) if dists[f'dists_{i+1}'] else 0 for i in range(groups)]
    noise = 0.05
    x_values = [np.random.normal(i+1, noise, size=len(dists[f'dists_{i+1}'])) for i in range(groups)]
    for i in range(groups):
        plt.scatter(x_values[i], dists[f'dists_{i+1}'], color='grey', alpha=0.5)
    plt.plot(range(1, groups+1), averages, color='white')
    # for i in range(groups):
    #     plt.scatter(i+1, averages[i], color='none', edgecolor='white', marker='o', linewidth=2, s=100)
    plt.ylabel('distance tracked (mm)', fontsize=18, color='white')
    plt.yticks(fontsize=14, color='white')
    axs.set_xticks(range(1, groups+1))
    axs.set_xticklabels(['no activation', 'single activation', 'multiple activation'], fontsize=16, color='white', rotation=45)
    plt.xlim(0.5, groups + 0.5)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}.pdf', bbox_inches='tight')

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
    fig, axs = configure_bw_plot(size=size, xaxis=False)
    params_list = exp_parameters(folder_path)
    rpm_b1 = []
    rpm_b2 = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        ypos = exp_df['ft_posy'] - yo
        b1_df = exp_df[(ypos >= 0) & (ypos < cutoff)]
        b2_df = exp_df[(ypos >= cutoff) & (ypos < 1000)]
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
    b1_avg = sum(rpm_b1) / len(rpm_b1)
    b2_avg = sum(rpm_b2) / len(rpm_b2)  
    noise = 0.05
    x_b1=np.random.normal(1, noise, size=len(rpm_b1))
    x_b2=np.random.normal(2, noise, size=len(rpm_b2))
    plt.scatter(x_b1, rpm_b1, color=colors[0], alpha=0.5)
    plt.scatter(x_b2, rpm_b2, color=colors[1], alpha=0.5)
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

def block_xpos_distribution(folder_path, savename, size=(6,5), cutoff=500, labels = ['LED off', 'LED on'], colors=['#c1ffc1', '#6f00ff']):
    # Right now hard coded for only 2 blocks
    fig, axs = configure_bw_plot(size=size, xaxis=True)
    params_list = exp_parameters(folder_path)
    xpos_dist_b1 = []
    xpos_dist_b2 = []
    for this_experiment in params_list:
        figure_folder, filename, df, df_odor, df_light, exp_df, xo, yo, plume_color = this_experiment
        ypos = exp_df['ft_posy'] - yo
        b1_df = exp_df[(ypos >= 0) & (ypos < cutoff)]
        b2_df = exp_df[(ypos >= cutoff) & (ypos < 1000)]
        xpos_b1 = np.abs(b1_df['ft_posx'] - xo)
        xpos_b2 = np.abs(b2_df['ft_posx'] - xo)
        xpos_dist_b1.append(xpos_b1)
        xpos_dist_b2.append(xpos_b2)
    compiled_b1 = pd.concat(xpos_dist_b1, axis=0, ignore_index=True)
    compiled_b2 = pd.concat(xpos_dist_b2, axis=0, ignore_index=True)      
    sns.kdeplot(compiled_b1, label=labels[0], color=colors[0], fill=False, cut=0)
    sns.kdeplot(compiled_b2, label=labels[1], color=colors[1], fill=False, cut=0) 
    min_x, max_x = min(compiled_b1.min(), compiled_b2.min()), max(compiled_b1.max(), compiled_b2.max())
    axs.set_xlim([0, max_x])
    plt.axvline(x=25, color='lightgrey')
    plt.xlabel('x-position (mm)', fontsize=18, color='white')
    plt.ylabel('density', fontsize=18, color='white')
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_linewidth(2)
    axs.spines['bottom'].set_color('white')
    axs.spines['left'].set_color('white')
    plt.xticks([0, 25, max_x], fontsize=14, color='white')
    plt.yticks([0, 0.03], fontsize=14, color='white')
    plt.legend()
    plt.show()
    print(compiled_b1.head(), compiled_b1.min())
    print(compiled_b2.head(), compiled_b2.min())    
    if not os.path.exists(f'{folder_path}/fig/'):
        os.makedirs(f'{folder_path}/fig/')
    fig.savefig(f'{folder_path}/fig/{savename}', bbox_inches='tight')

def avg_entry_angle(folder, window):
    fig, axs = plt.subplots(subplot_kw={'projection': 'polar'})
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