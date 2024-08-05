import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pingouin as pg
from src.drive import drive as dr
from src.utilities import funcs as fn
from src.utilities import plotting as pl
from src.utilities import imaging as im
from scipy import interpolate, stats
from scipy.optimize import minimize
# from numba import jit
import seaborn as sns
import time
importlib.reload(dr)
importlib.reload(im)
importlib.reload(pl)
importlib.reload(fn)


sns.set(font="Arial")
sns.set(font_scale=0.6)
sns.set_style("white")
sns.set_style("ticks")

class closed_loop():
    """
    class for comparing constant concentration and gradient plumes

    excluding this trial (and naive replay) because fly does not track horizontal plume
    01252024-192235_horizontal_replay_Fly5.log
    01262024-183812_replay_pulses_Fly4.log
    """
    def __init__(self, directory='M1', experiment = 'constant'):
        d = dr.drive_hookup()
        # your working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'
        elif directory == 'Andy':
            self.cwd = '/Volumes/Andy/GitHub/edge-tracking'
        self.experiment = experiment

        # specify which Google sheet to pull log files from
        self.sheet_id = '1FjRjwelSbCZM6wQ1JaFMuDGuFKy4K11BBv4bM_QXOZY'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df

        # specify pickle folder and pickle name
        self.picklefol = os.path.join(self.cwd, 'data/horizontal_replay/pickles')
        if not os.path.exists(self.picklefol):
            os.makedirs(self.picklefol)
        self.picklesname = os.path.join(self.picklefol, 'closed_loop.p')

        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/horizontal_replay')
        if not os.path.exists(self.figurefol):
            os.makedirs(self.figurefol)

        # download new log folders
        self.logfol = '/Volumes/Andy/logs'
        d = dr.drive_hookup()
        #d.download_logs_to_local('/Volumes/Andy/logs')

    def split_trajectories(self):
        # dict where all log files are stored
        all_data = {}
        accept = []
        for i, log in enumerate(self.sheet.logfile):
            # specify trial type
            trial_type = self.sheet.condition.iloc[i]
            # read in each log file
            data = fn.read_log(os.path.join(self.logfol, log))
            # if the tracking was lost, select correct segment

            data = fn.exclude_lost_tracking(data)

            # contingencies
            if log == '01232024-125101_Horizontalreplay_Fly3.log':
                data = data.iloc[:35040] # cropped because the fly flung the ball -- speed 10x normal

            mfcs = self.sheet.mfcs.iloc[i]
            if mfcs == '1':
                data['instrip'] = np.where(np.abs(data.mfc2_stpt)>0.01, True, False)
            # consolidate short in and short out periods
            data = fn.consolidate_in_out(data)

            # append speeds to dataframe
            data = fn.calculate_speeds(data)

            # crop after the replay
            if trial_type=='cl':
                ix_final = data[data['mode']=='replay'].index[-1]
                data = data.iloc[:ix_final]

            # split trajectories into inside and outside component
            d, di, do = fn.inside_outside(data)

            dict_temp = {"data": data,
                        "d": d,
                        "di": di,
                        "do": do,
                        "trial_type":trial_type
                        }
            all_data[log] = dict_temp
        # pickle everything
        fn.save_obj(all_data, self.picklesname)


    def load_trajectories(self):
        """
        open the pickled data stored from split_trajectories()
        """
        all_data = fn.load_obj(self.picklesname)
        return all_data

    def plot_individual_trajectories(self):
        
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')

        all_data = self.load_trajectories()
        for log in list(all_data.keys()):
            print(log)
            temp = all_data[log]
            data = temp['data']
            fig, axs = plt.subplots(1,1)
            pl.plot_trajectory(data, axs)
            pl.plot_trajectory_odor(data, axs)
            # axs.plot([0,100], [0,0], 'k')
            axs.axis('equal')
            # axs.axis('off')
            fig.suptitle(log)
            fig.savefig(os.path.join(self.figurefol, 'trajectory_'+log.replace('.log', '.pdf')), transparent=True)
        return temp

    def plot_individual_trajectories_live_vs_replay(self):
        """
        Check to make sure that the live and replay makes sense
        """
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')

        all_data = self.load_trajectories()
        for log in list(all_data.keys()):
            if all_data[log]['trial_type'] == 'cl':
                print(log)
                temp = all_data[log]
                data = temp['data']
                df1 = data[data['mode']=='live']
                df2 = data[data['mode']=='replay']
                fig, axs = plt.subplots(1,1)

                pl.plot_trajectory(df1, axs)
                pl.plot_trajectory(df2, axs, color = 'blue')
                pl.plot_trajectory_odor(df1, axs, color=pl.inside_color)
                pl.plot_trajectory_odor(df2, axs, color=pl.inside_color)
                axs.plot([0,100], [0,0], 'k')
                axs.axis('equal')
                axs.axis('off')
                fig.suptitle(log)
                fig.savefig(os.path.join(self.figurefol, 'trajectory_'+log.replace('.log', '.pdf')), transparent=True)

                # also plot the odor profile
                t0 = df1.seconds.iloc[0]
                fig, axs = plt.subplots(1,1, figsize=(6,1))
                axs.plot(df1['seconds']-t0, df1['instrip'])
                axs.plot(df2['seconds']-t0, df2['instrip'], color='blue')
                axs.plot([0,30],[0,0], color='k') # 30 second bar
                axs.spines[['right', 'top']].set_visible(False)
                axs.set_yticks([0,1])
                axs.set_yticklabels(['outside', 'inside'])
                fig.tight_layout()
                fig.savefig(os.path.join(self.figurefol, 'odor_signal_'+log.replace('.log', '.pdf')))
        return temp
    
    def entry_direction(self):
        def calc_entry_heading(df, t0):
            df['ft_heading'] = fn.wrap(df['ft_heading'])
            df['seconds'] = df['seconds']-df['seconds'].iloc[0]
            t = df['seconds'].to_numpy()
            tmax = df['seconds'].iloc[-1]
            if tmax<t0:
                heading = fn.circmean(df.ft_heading)
            else:
                ix = np.argmin(np.abs(t-(tmax-t0)))
                heading = fn.circmean(df.ft_heading.iloc[ix:])
            return heading

        def calc_outside_heading(df):
            df['ft_heading'] = fn.wrap(df['ft_heading'])
            heading = fn.circmean(df.ft_heading)
            return heading
        
        def calc_displacement(df):
            delx = df['ft_posx'].iloc[-1]-df['ft_posx'].iloc[0]
            dely = df['ft_posy'].iloc[-1]-df['ft_posy'].iloc[0]
            angle = np.arctan2(dely,delx)
            angle = fn.conv_cart_upwind(np.array([angle]))
            return angle
        
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('ticks')


        all_data = self.load_trajectories()
        num_fly = 0

        colors=['blue', 'darkgreen']

        for fly in list(self.sheet.fly.unique()):
            # scatter plot
            fig, axs = plt.subplots(1,5,figsize=(12,3))

            df = self.sheet.loc[self.sheet['fly']==fly]
            log = df.loc[df['condition']=='cl'].logfile.item()
            savebase = log.replace('.log', '')
            temp = all_data[log]
            df_t = temp['data']
            for p, mode in enumerate(['live', 'replay']):
                heading_entry = []
                heading_outside = []
                displacement_outside = []
                df_m = df_t[df_t['mode']==mode]
                _,_,do = fn.inside_outside(df_m)
                pl.plot_trajectory(df=df_m, axs=axs[0],color=colors[p])
                pl.plot_trajectory_odor(df=df_m, axs=axs[0])
                for key in list(do.keys())[:-1]:
                    
                    df1 = do[key]
                    df2 = do[key+2]

                    heading_entry.append(calc_entry_heading(df1, t0=1))
                    heading_outside.append(calc_outside_heading(df2))
                    displacement_outside.append(calc_displacement(df2))


                savebase = log.replace('.log', '')

                axs[p+1].plot([0,0], [-np.pi, np.pi], linestyle='dashed', color='k', alpha=0.5)
                axs[p+1].plot([-np.pi, np.pi], [0,0], linestyle='dashed', color='k', alpha=0.5)
                axs[p+1].scatter(heading_entry, displacement_outside,color=colors[p])
                axs[p+1].set_ylim(-np.pi, np.pi)
                axs[p+1].set_xlim(-np.pi, np.pi)
                axs[p+1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
                axs[p+1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
                axs[p+1].set_xlabel('entry heading (prev 1 sec.)')
                axs[p+1].set_ylabel('outside displacement vector')
            
            log = df.loc[df['condition']=='replay'].logfile.item()
            temp = all_data[log]
            df_t = temp['data']
            heading_entry = []
            heading_outside = []
            displacement_outside = []
            _,_,do = fn.inside_outside(df_t)
            for key in list(do.keys())[:-1]:
                
                df1 = do[key]
                df2 = do[key+2]

                heading_entry.append(calc_entry_heading(df1, t0=1))
                heading_outside.append(calc_outside_heading(df2))
                displacement_outside.append(calc_displacement(df2))

            pl.plot_trajectory(df=df_t, axs=axs[4])
            pl.plot_trajectory_odor(df=df_t, axs=axs[4])
            
            axs[p+2].plot([0,0], [-np.pi, np.pi], linestyle='dashed', color='k', alpha=0.5)
            axs[p+2].plot([-np.pi, np.pi], [0,0], linestyle='dashed', color='k', alpha=0.5)
            axs[p+2].scatter(heading_entry, heading_outside,color=pl.inside_color)
            axs[p+2].set_ylim(-np.pi, np.pi)
            axs[p+2].set_xlim(-np.pi, np.pi)
            axs[p+2].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
            axs[p+2].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
            axs[p+2].set_xlabel('entry heading (prev 1 sec.)')
            axs[p+2].set_ylabel('outside displacement vector')
        
            fig.tight_layout()
            fig.savefig(os.path.join(self.figurefol, savebase+'_scatter.pdf'))

    def average_trajectory(self, pts = 10000):
        """
        - calculate the average trajectory for each animal that successfully edge tracks
        - calculate average trajectory for naive flies
        - save the average trajectory as .p
        """

        def average_x_y(d):
            x_interp, y_interp = [],[]
            for key in list(d.keys()):
                temp = d[key]
                if len(temp)>10:
                    temp = fn.find_cutoff(temp)
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()
                    x0 = x[0]
                    y0 = y[0]
                    x = x-x0
                    y = y-y0
                t = np.arange(len(x))
                t_common = np.linspace(t[0], t[-1], pts)
                fx = interpolate.interp1d(t, x)
                fy = interpolate.interp1d(t, y)
                x_interp.append(fx(t_common))
                y_interp.append(fy(t_common))
            return x_interp, y_interp

        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('ticks')


        all_data = self.load_trajectories()
        num_fly = 0
        def average_template():
            template = {
            'x_in':[],
            'y_in':[],
            'x_out':[],
            'y_out':[]
            }
            return template
        
        all_averages = {
            'live': average_template(),
            'replay': average_template(),
            'naive': average_template()
        }

        for fly in list(self.sheet.fly.unique()):
            fig_cl, axs_cl = plt.subplots(1,1, figsize = (9,3), subplot_kw={}) # plot for closed loop replay
            fig_naive, axs_naive = plt.subplots(1,1) # plot for naive replay
            fig, axs = plt.subplots(1,3) # plot for average trajectory
            
            df = self.sheet.loc[self.sheet['fly']==fly]
            log = df.loc[df['condition']=='cl'].logfile.item()
            savebase = log.replace('.log', '')
            temp = all_data[log]
            df_t = temp['data']
            pl.plot_trajectory(df=df_t, axs = axs_cl)
            pl.plot_trajectory_odor(df = df_t, axs = axs_cl)
            
            for p, mode in enumerate(['live', 'replay']):
                avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
                df_m = df_t[df_t['mode']==mode]

                # flip traj. depending on which direction the fly goes
                x = df_m.ft_posx.to_numpy()
                if mode == 'live':
                    # find where the odor turns on
                    ix = df_m.index[df_m.instrip==True].tolist()[0]
                    x0 = x[ix]
                    xend = x[-1]
                if xend<x0: # if the flies go to the left, flip them so they go to the right
                    x = -x
                    df_m.loc[:,('ft_posx')] = x
                
                # break up to inside and outside
                _,di,do = fn.inside_outside(df_m)

                # Need to exclude the last live outside bout before it switches to replay
                if mode == 'live':
                    last_bout = list(do.keys())[-1]
                    do.pop(last_bout)
                
                for io, d in zip(['in', 'out'],[di,do]):
                    d.pop(list(d.keys())[0]) # remove the first outside and inside segments
                    if io=='in':
                        avg_x_in, avg_y_in = average_x_y(d)
                        for i in np.arange(len(avg_x_in)):
                            axs[p].plot(avg_x_in[i], avg_y_in[i], pl.inside_color, linewidth = 0.25)
                        axs[p].plot(np.mean(avg_x_in, axis=0), np.mean(avg_y_in, axis=0), pl.inside_color)
                        x_end = np.mean(avg_x_in, axis=0)[-1]
                        y_end = np.mean(avg_y_in, axis=0)[-1]
                        all_averages[mode]['x_in'].append(np.mean(avg_x_in, axis=0))
                        all_averages[mode]['y_in'].append(np.mean(avg_y_in, axis=0))
                    elif io=='out':
                        avg_x_out, avg_y_out = average_x_y(d)
                        for i in np.arange(len(avg_x_out)):
                            axs[p].plot(avg_x_out[i]+x_end, avg_y_out[i]+y_end, pl.outside_color, linewidth=0.25)
                        axs[p].plot(np.mean(avg_x_out, axis=0)+x_end, np.mean(avg_y_out, axis=0)+y_end, pl.outside_color)
                        all_averages[mode]['x_out'].append(np.mean(avg_x_out, axis=0))
                        all_averages[mode]['y_out'].append(np.mean(avg_y_out, axis=0))
                    axs[p].axis('equal')
            
            # pull in the naive replay and plot the average trajectory
            log = df.loc[df['condition']=='replay'].logfile.item()
            temp = all_data[log]
            df_t = temp['data']

            pl.plot_trajectory(df=df_t, axs = axs_naive)
            pl.plot_trajectory_odor(df = df_t, axs = axs_naive)

            avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
            _,di,do = fn.inside_outside(df_t)
            for io, d in zip(['in', 'out'],[di,do]):
                d.pop(list(d.keys())[0]) # remove the first outside and inside segments
                if io=='in':
                    avg_x_in, avg_y_in = average_x_y(d)
                    for i in np.arange(len(avg_x_in)):
                        axs[p+1].plot(avg_x_in[i], avg_y_in[i], pl.inside_color, linewidth=0.25)
                    axs[p+1].plot(np.mean(avg_x_in, axis=0), np.mean(avg_y_in, axis=0), pl.inside_color)
                    x_end = np.mean(avg_x_in, axis=0)[-1]
                    y_end = np.mean(avg_y_in, axis=0)[-1]
                    all_averages['naive']['x_in'].append(np.mean(avg_x_in, axis=0))
                    all_averages['naive']['y_in'].append(np.mean(avg_y_in, axis=0))
                elif io=='out':
                    avg_x_out, avg_y_out = average_x_y(d)
                    for i in np.arange(len(avg_x_out)):
                        axs[p+1].plot(avg_x_out[i]+x_end, avg_y_out[i]+y_end, pl.outside_color, linewidth=0.25)
                    axs[p+1].plot(np.mean(avg_x_out, axis=0)+x_end, np.mean(avg_y_out, axis=0)+y_end, pl.outside_color)
                    all_averages['naive']['x_out'].append(np.mean(avg_x_out, axis=0))
                    all_averages['naive']['y_out'].append(np.mean(avg_y_out, axis=0))
                axs[p+1].axis('equal')
        
        # plot the individual animal averages and an average of averages
        fig, axs = plt.subplots(1,3, figsize = (9,3))
        for i, mode in enumerate(['live','replay','naive']):
            d_temp = all_averages[mode]
            x_end = np.mean(d_temp['x_in'], axis=0)[-1]
            y_end = np.mean(d_temp['y_in'], axis=0)[-1]
            for l in np.arange(len(d_temp['x_in'])):
                axs[i].plot(d_temp['x_in'][l], d_temp['y_in'][l], color=pl.inside_color, linewidth=0.5)
                axs[i].plot(d_temp['x_out'][l]+x_end, d_temp['y_out'][l]+y_end, color=pl.outside_color, linewidth=0.5)
            axs[i].plot(np.mean(d_temp['x_in'], axis=0), np.mean(d_temp['y_in'], axis=0), color=pl.inside_color)
            axs[i].plot(np.mean(d_temp['x_out'], axis=0)+x_end, np.mean(d_temp['y_out'], axis=0)+y_end, color=pl.outside_color)
            axs[i].plot([0,10], [0,0], color='k')
            axs[i].axis('off')
        fig.savefig(os.path.join(self.figurefol, 'all_averages.pdf'))

        fn.save_obj(all_averages, os.path.join(self.picklefol,'average_trajectories.p'))
        return all_averages
    
    def plot_average_heading(self):
        all_averages = fn.load_obj(os.path.join(self.picklefol,'average_trajectories.p'))


        angles = {
            'live': [],
            'replay': [],
            'naive': []
        }
        fig, axs = plt.subplots(1,3, figsize=(9,3), subplot_kw={'projection':'polar'})
        for i,a in enumerate(axs.flatten()):
            a.plot(np.linspace(0,2*np.pi, num=50), np.ones(50), 'k', linewidth=0.5)

        for i, condition in enumerate(['live', 'replay', 'naive']):
            temp = all_averages[condition]
            x_all = temp['x_out']
            y_all = temp['y_out']
            angles = []
            for m in np.arange(len(x_all)):
                x = x_all[m]
                y = y_all[m]
                x_avg = np.mean(x)
                y_avg = np.mean(y)
                ang_temp = np.arctan2(y_avg, x_avg)
                angles.append(ang_temp)
                axs[i].plot(ang_temp, 1, '.', color='k')
                axs[i].plot([ang_temp,ang_temp], [0,1], alpha=0.5, linewidth=0.5, color='k')
            axs[i].axis('off')
            axs[i].plot([stats.circmean(angles), stats.circmean(angles)],[0, 1], color='k', linewidth=3)
        fig.savefig(os.path.join(self.figurefol, 'average_position_directions.pdf'))