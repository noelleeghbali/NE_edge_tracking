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
#from numba import jit
import seaborn as sns
import time
importlib.reload(dr)
importlib.reload(im)
importlib.reload(pl)
importlib.reload(fn)
os.getcwd()

class reverse_gradient:
    def __init__(self, directory='M1'):
        d = dr.drive_hookup()
        # your working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'

        # specify which Google sheet to pull log files from
        self.sheet_id = '1GsaAiICMIZrz3YL5-9nh9oWrTSFaXL4JAJf6ofRNfrs'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df

        # specify pickle folder and pickle name
        self.picklefol = os.path.join(self.cwd, 'data/reverse_gradient/pickles')
        if not os.path.exists(self.picklefol):
            os.makedirs(self.picklefol)
        self.picklesname = os.path.join(self.picklefol, 'et_manuscript_reverse_gradient.p')

        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/reverse_gradient')
        if not os.path.exists(self.figurefol):
            os.makedirs(self.figurefol)

        # download new log folders
        self.logfol = '/Volumes/Andy/logs'
        d = dr.drive_hookup()
        #d.download_logs_to_local('/Volumes/Andy/logs')


    def split_trajectories(self):
        # dict where all log files are stored
        all_data = {}
        for i, log in enumerate(self.sheet.logfile):
            # specify trial type
            trial_type = self.sheet.condition.iloc[i]
            # read in each log file
            data = fn.read_log(os.path.join(self.logfol, log))
            # if the tracking was lost, select correct segment
            data = fn.exclude_lost_tracking(data, thresh=10)
            # specificy when the fly is in the strip for old mass flow controllers
            mfcs = self.sheet.mfcs.iloc[i]
            if mfcs == '0':
                data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
            # consolidate short in and short out periods
            data = fn.consolidate_in_out(data)

            #calculate speeds
            data = fn.calculate_speeds(data)
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

    def compare_tracking(self):
        """
        calculate basic metrics constant and gradient plumes for paired experiment
        - Also look at inbound vs outbound
        """
        # some functions for calculating necessary metrics

        def inbound_outbound_angle(x, y, simplified):
            """
            calculate the inbound and outbound angle for a given inside or outside trajectory
            """
            def calculate_angle(vector_1, vector_2):
                unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = np.arccos(dot_product)
                return angle
            x = x-x[0]
            y = y-y[0]
            start_x = x[0]
            start_y = y[0]
            return_x = x[-1]
            return_y = y[-1]

            out_x = simplified[1,0] # x end of the first line segment
            out_y = simplified[1,1] # y end of the first line segment
            in_x = simplified[-2,0] # x start of the last line segment
            in_y = simplified[-2,1] # y start of the last line segment
            
            # vectors used to calculate the inside angles with three line simplification
            vec_out = [out_x-start_x, out_y-start_y]
            edge_vec_out = [return_x-start_x, return_y-start_y]
            vec_in = [in_x-return_x, in_y-return_y]
            edge_vec_in = [start_x-return_x, start_y-return_y]

            # calculate the angle
            angle_out = np.rad2deg(calculate_angle(vec_out, edge_vec_out))
            angle_in = np.rad2deg(calculate_angle(vec_in, edge_vec_in))
            return [angle_out, angle_in]

        def inbound_outbound_lengths(x ,y, simplified):
            """
            inbound and outbound path length
            """
            out_ix = np.where(x==simplified[1,0])[0][0]
            in_ix = np.where(x==simplified[-2,0])[0][0]
            _,length_out = fn.path_length(x[0:out_ix+1], y[0:out_ix+1])
            _,length_in = fn.path_length(x[in_ix:], y[in_ix:])
            e_out = np.max(np.abs(x[0:out_ix+1]))/length_out
            e_in = np.max(np.abs(x[in_ix:]))/length_in

            return [length_out, length_in, e_out, e_in]

        def inbound_outbound_time(x,y,t, simplified):
            """
            inbound and outbound time
            """
            out_ix = np.where(x==simplified[1,0])[0][0]
            in_ix = np.where(x==simplified[-2,0])[0][0]
            t_out = np.mean(t[0:out_ix+1]-t[0])
            t_in = np.mean(t[in_ix:]-t[in_ix])

            return [t_out, t_in]
        
        def inbound_outbound_speed(x,y,s,simplified):
            """
            inbound and outbound speed
            """
            out_ix = np.where(x==simplified[1,0])[0][0]
            in_ix = np.where(x==simplified[-2,0])[0][0]
            s_out = np.mean(s[0:out_ix+1])
            s_in = np.mean(s[in_ix:])

            return [s_out, s_in]
        
        def inbound_outbound_velocities(x,y,xv,yv, simplified):
            """
            inbound_outbound speeds
            """
            out_ix = np.where(x==simplified[1,0])[0][0]
            in_ix = np.where(x==simplified[-2,0])[0][0]
            xv_out = np.mean(xv[0:out_ix+1])
            xv_in = np.mean(xv[in_ix:])
            yv_out = np.mean(yv[0:out_ix+1])
            yv_in = np.mean(yv[in_ix:])
            if len(simplified)==4:
                xv_mid = np.mean(xv[out_ix:in_ix])
                yv_mid = np.mean(yv[out_ix:in_ix])
            else:
                xv_mid = np.nan
                yv_mid = np.nan

            return [xv_out, xv_mid, xv_in, yv_out, yv_mid, yv_in]
        

        # interpolation points
        pts=10000

        # load preprocessed data
        preprocessed = self.load_trajectories()

        # list of dictionaries for all inside and outside results
        all_results = []


        # go through each fly and all its trials
        for log_name in list(preprocessed.keys()):
            plot=False
            temp = preprocessed[log_name]
            fly = 1 # to be filled in
            trial_type = temp['trial_type']
            # log = temp['data']
            di = temp['di']
            do = temp['do']
            # d = temp['d']

            # calculate all inside parameters
            p = {
            'fly':fly,
            'trial_type':trial_type,
            'filename': log_name,
            't':[],
            'pathl':[],
            'speed':[],
            'up_vel':[],
            'cross_speed':[],
            'x_dist':[],
            'y_dist':[],
            'x_interp': [],
            'y_interp': [],
            't_interp': [],
            's_interp': [],
            'xv_interp': [],
            'yv_interp': [],
            'out_angle': [],
            'in_angle':[],
            'out_dist':[],
            'in_dist':[],
            'out_e':[],
            'in_e':[],
            'out_t': [],
            'in_t': [],
            'out_s': [],
            'in_s': [],
            'xv_out': [],
            'xv_mid': [],
            'xv_in': [],
            'xv_out': [],
            'xv_mid': [],
            'xv_in': [],
            }

            for i, dio in enumerate([di,do]):
                params = p.copy()
                if i==0:
                    params['io']='in'
                else:
                    params['io']='out'
                for bout in list(dio.keys()):
                    df_bout = dio[bout]
                    if fn.return_to_edge(df_bout): # only calculating information for bouts that return to the edge
                        x = df_bout.ft_posx.to_numpy()
                        y = df_bout.ft_posy.to_numpy()
                        t = df_bout.seconds.to_numpy()
                        t = t-t[0]
                        del_t = np.mean(np.gradient(t))
                        s = df_bout.speed.to_numpy()

                        # time
                        t_bout = t[-1]-t[0]
                        params['t'].append(t_bout)

                        # pathlength
                        _,pathl = fn.path_length(x,y)
                        params['pathl'].append(pathl)

                        # x and y distances
                        x = -np.abs(x-x[0]) # center on x=0 and flip to be on the left
                        y = y-y[0]
                        x_dist = np.min(x)
                        y_dist = y[-1]-y[0]
                        params['x_dist'].append(x_dist)
                        params['y_dist'].append(y_dist)

                        # speeds
                        params['speed'].append(np.mean(df_bout.speed))
                        params['up_vel'].append(np.mean(df_bout.yv))
                        params['cross_speed'].append(np.mean(np.abs(df_bout.xv)))
                        yv = df_bout.yv.to_numpy()
                        xv = np.gradient(x)/del_t

                        
                        # interpolate x,y,t variables
                        t_pts = np.arange(len(x))
                        t_common = np.linspace(t_pts[0], t_pts[-1], pts)
                        fx = interpolate.interp1d(t_pts, x)
                        fy = interpolate.interp1d(t_pts, y)
                        ft = interpolate.interp1d(t_pts, t)
                        fs = interpolate.interp1d(t_pts, s)
                        fxv = interpolate.interp1d(t_pts, xv)
                        fyv = interpolate.interp1d(t_pts, yv)

                        params['x_interp'].append(fx(t_common))
                        params['y_interp'].append(fy(t_common))
                        params['t_interp'].append(ft(t_common))
                        params['s_interp'].append(fs(t_common))
                        params['xv_interp'].append(fxv(t_common))
                        params['yv_interp'].append(fyv(t_common))

                        if plot and (params['io']=='out'):
                            fig, axs = plt.subplots(1,1)
                            axs.plot(x,y)
                            axs.text(0,0,str([angle_out, angle_in]))
                            axs.axis('equal')
                            plot = False
                # average over all bouts for the same animal
                for pkey in list(params.keys()):
                    if type(params[pkey])==list:
                        params[pkey] = np.mean(params[pkey], axis=0)
                        
                
                # inbound and outbound parameters
                x_interp = params['x_interp']
                y_interp = params['y_interp']
                t_interp = params['t_interp']
                s_interp = params['s_interp']
                xv_interp = params['xv_interp']
                yv_interp = params['yv_interp']

                xy0 = np.concatenate((x_interp[:,None],y_interp[:,None]),axis=-1)
                simplified = fn.rdp_pts(xy0) 

                [params['out_angle'], params['in_angle']] = inbound_outbound_angle(x_interp, y_interp, simplified)
                [params['out_dist'], params['in_dist'], params['out_e'], params['in_e']] = inbound_outbound_lengths(x_interp,y_interp, simplified)
                [params['out_t'], params['in_t']] = inbound_outbound_time(x_interp,y_interp,t_interp, simplified)
                [params['out_s'], params['in_s']] = inbound_outbound_speed(x_interp,y_interp, s_interp, simplified)
                [params['xv_out'], params['xv_mid'], params['xv_in'], params['yv_out'], params['yv_mid'], params['yv_in']] = inbound_outbound_velocities(x_interp,y_interp, xv_interp, yv_interp, simplified)


                all_results.append(params)

            df = pd.DataFrame(all_results)
            fn.save_obj(df, os.path.join(self.picklefol, 'compare_tracking.p'))
        return df

    def load_comparison(self):
        # load the processed comparison data
        if os.path.exists(os.path.join(self.picklefol, 'compare_tracking.p')):
            df = fn.load_obj(os.path.join(self.picklefol, 'compare_tracking.p'))
        else:
            df = self.compare_tracking()
        return df
    
    def plot_comparison(self):
        """
        plot comparison between gradient and constant for all parameters
        """
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        colors = [pl.inside_color, pl.inside_color, pl.outside_color, pl.outside_color]

        df = self.load_comparison()
        metrics = ['t',
                   'pathl',
                   'speed',
                   'up_vel',
                   'y_dist',
                   'cross_speed',
                   'x_dist']
        labels = ['time (s)', 
                  'path length (mm)', 
                  'speed (mm/s)', 
                  'upwind velocity (mm/s)',
                  'y distance (mm)', 
                  'crosswind speed (mm/s)', 
                  'x distance (mm)']
        scale = 1.5
        # make one plot with all subplots, and three smaller plots for making the figure
        #fig, axs = plt.subplots(1,len(metrics), figsize=(scale*len(metrics), scale))
        #fig.tight_layout()
        # small figure #1
        fig1, axs1 = plt.subplots(1,3, figsize=(scale*3, scale))
        # small figure #2
        fig2, axs2= plt.subplots(1,2, figsize=(scale*2, scale))
        # small figure #3
        fig3, axs3 = plt.subplots(1,2, figsize=(scale*2, scale))
        figs = [fig1, fig2, fig3]
        xpos = np.array([0,1]) # generalize this
        for fig in figs:
            fig.tight_layout()
            sns.despine(fig)
        axs = np.concatenate((axs1, axs2, axs3))
        for i, metric in enumerate(metrics):
            sns.stripplot(data=df, x='io', y=metric, hue='trial_type', hue_order=['constant','reverse'],palette=['black', 'grey'], dodge=True, size=4, alpha=0.4, ax=axs[i])
            sns.pointplot(data=df, x='io', y=metric, hue='trial_type', hue_order=['constant','reverse'],palette=['black', 'grey'], dodge=.4, ax=axs[i], join=False, markers=['_','_'], errwidth=2.0)




            axs[i].get_legend().remove()
            axs[i].tick_params(axis='x', pad=-3)
            axs[i].tick_params(axis='y', pad=-3)
            axs[i].set_ylabel(labels[i])
            axs[i].set_xlabel('')
            label_pos = axs[i].xaxis.get_label().get_position()
            print(label_pos)
            #axs_sub[i]=axs[i].copy()
        fig1.savefig(os.path.join(self.figurefol, 'cnst_v_rev_1.pdf'))
        fig2.savefig(os.path.join(self.figurefol, 'cnst_v_rev_2.pdf'))
        fig3.savefig(os.path.join(self.figurefol, 'cnst_v_rev_3.pdf'))

    def statistical_tests(self):
        df = self.compare_tracking()
        metrics = ['t','pathl','speed','up_vel','cross_speed','x_dist','y_dist']
        labels = ['time (s)', 'path length (mm)', 'speed (mm/s)', 'upwind velocity (mm/s)', 'x speed (mm/s)', 'x distance (mm)', 'y distance (mm)']
        for dv in metrics:
            print('%%%%%%%%%DEPENDENT VARIABLE = ', dv, '%%%%%%%%%')
            print(pg.mixed_anova(data=df, dv=dv, within = 'io', between='trial_type', subject='filename').to_markdown())
            print('')
            print(pg.pairwise_ttests(data=df, dv='t', within = 'io', between='trial_type', subject='filename').to_markdown())
            print('')
            print('')

    def load_trajectories(self):
        all_data = fn.load_obj(self.picklesname)
        return all_data

    def plot_individual_trajectories(self):
        all_data = self.load_trajectories()
        for i, log in enumerate(self.sheet.log):
            print(log)
            df = all_data[log]['data']
            fig, axs = plt.subplots(1,1)
            pl.plot_vertical_edges(df, axs)
            pl.plot_trajectory(df, axs)
            pl.plot_trajectory_odor(df, axs)
            axs.axis('equal')
            fig.suptitle(log)

    def plot_example_trajectory(self):
        """
        plot an example of the trajectories for the constant odor plume and the decreasing odor plume
        """
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        all_data = self.load_trajectories()
        examples = ['05302022-190526_constantOdor_Fly0.log', '05302022-194930_reversegradient_Fly0.log']
        fig, axs = plt.subplots(1,2,figsize=(2,3))
        j=0
        for i, log in enumerate(self.sheet.logfile):
            if log in examples:
                # load the data
                df = all_data[log]['data']

                # crop the trajectory so that it starts once the odor turn on
                ix0 = df[df.mfc2_stpt>0].index[0]
                df = df.iloc[ix0:]
                
                # reset the position to (0,0)
                df.loc[:,'ft_posx'] = df.ft_posx-df.ft_posx.iloc[0]
                df.loc[:,'ft_posy'] = df.ft_posy-df.ft_posy.iloc[0]

                # scalebar
                if j==0:
                    axs[j].plot([-25,25], [-30,-30], 'k')
                    axs[j].text(-60, -80, '50 mm')

                # plot the plume schematic
                if j==0:
                    axs[j] = pl.plot_plume_corridor(axs[j], type='constant')
                elif j==1:
                    axs[j] = pl.plot_plume_corridor(axs[j], type='decreasing')

                # need to crop the trajectory that wanders very far away
                if np.max(df.ft_posx)>50:
                    ix = df[df.ft_posx>50].index[0]
                    df = df.iloc[:ix]
                axs[j] = pl.plot_trajectory(df, axs[j], linewidth=0.5)
                axs[j] = pl.plot_trajectory_odor(df, axs[j], linewidth=0.5)
                axs[j].axis('equal')
                axs[j].set_ylim([-50,1000])
                axs[j].axis('off')
                j+=1
        fig.savefig(os.path.join(self.figurefol, 'decreasing_gradient_example_trajectories.pdf'))

    def distance_tracked_upwind(self):
        all_data = self.load_trajectories()
        """
        plot an example of the trajectories for the constant odor plume and the decreasing odor plume
        """
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')

        df = self.sheet
        upwind_dist=[]
        for fly in df.fly.unique():
            df_fly = df[df.fly==fly]
            for i, log in enumerate(df_fly.logfile):
                # find where odor turns on
                data = all_data[log]['data']
                ix0 = data.index[data.instrip==True].tolist()[0]
                data = data.iloc[ix0:]
                data.ft_posy = data.ft_posy-data.ft_posy.iloc[0]
                data_instrip = data.mask(data.instrip==False)
                y = data_instrip.ft_posy.to_numpy()
                experiment = all_data[log]['trial_type']
                if experiment=='constant':
                    y_constant = np.nanmax(y)/1000
                elif experiment=='reverse':
                    y_reverse = np.nanmax(y)/1000
                else:
                    print('NO EXPERIMENT FOUND')
                    return
            upwind_dist.append([y_constant, y_reverse])
        upwind_dist = np.array(upwind_dist)
        fig, axs = plt.subplots(1,1, figsize=(1,1.6))
        axs = pl.paired_plot(axs, upwind_dist[:,0],upwind_dist[:,1],color1=pl.inside_color, color2=pl.outside_color,
            indiv_pts=True, scatter_scale=0, alpha=1, indiv_markers=False, mean_line=False, log=False)
        axs.set_xticks([0,1])
        axs.set_xticklabels(['constant', 'reverse'], rotation=45)
        axs.set_yticks([0,0.5,1])
        axs.set_ylabel('dist. tracked up plume (m)')
        axs.set_ylim(0,1.2)
        sns.despine()
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'decreasing_gradient_paired_plot.pdf'))


