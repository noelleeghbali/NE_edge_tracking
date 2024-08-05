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
from numpy_ext import rolling_apply
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



class constant_gradient():
    """
    class for comparing constant concentration and gradient plumes
    """
    def __init__(self, directory='M1', experiment = 'constant'):
        d = dr.drive_hookup()
        # your working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'
        elif directory == 'Andy'
            self.cwd = '/Volumes/Andy/GitHub/edge-tracking'
        self.experiment = experiment

        # specify which Google sheet to pull log files from
        self.sheet_id = '1Is1t3UtMAycrvpSMvEf6j2Gpc4b5jkEdm7yTIEAxfw8'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df

        # specify pickle folder and pickle name
        self.picklefol = os.path.join(self.cwd, 'data/constant_gradient/pickles')
        if not os.path.exists(self.picklefol):
            os.makedirs(self.picklefol)
        self.picklesname = os.path.join(self.picklefol, 'et_manuscript_cons_v_gradient.p')

        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/constant_gradient')
        if not os.path.exists(self.figurefol):
            os.makedirs(self.figurefol)

        # download new log folders
        self.logfol = '/Volumes/Andy/logs'
        d = dr.drive_hookup()
        #d.download_logs_to_local('/Volumes/Andy/logs')

    def split_trajectories(self):
        """
        Pre-process all data and save in dictionary
        """
        # dict where all log files are stored
        all_data = {}
        accept = []
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
            # select for edge-tracking trials
            if not fn.select_et(data):
                print('REJECT: ',log)
                continue
            else:
                print('ACCEPT: ',log)
                accept.append(log)
            # append speeds to dataframe
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
        # save the log files used
        df = pd.DataFrame(data = accept, columns = ['logs'])
        df.to_csv(os.path.join(self.picklefol, 'accepted_logs.csv'))

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
            temp = all_data[log]
            data = temp['data']
            fig, axs = plt.subplots(1,1)
            pl.plot_trajectory(data, axs)
            pl.plot_trajectory_odor(data, axs)
            axs.axis('equal')
            fig.suptitle(log)
            fig.savefig(os.path.join(self.figurefol, 'trajectory_'+log.replace('.log', '.pdf')), transparent=True)

    def plot_trajectories(self, experiment = 'constant', pts = 10000):
        """
        plot the average trajectories.  Normalize them by divinding
        """
        all_data = self.load_trajectories()
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = [],[],[],[]
        for i, log in enumerate(list(all_data.keys())):
            # print(self.sheet.experiment.iloc[i] == experiment)
            # if self.sheet.experiment.iloc[i] != experiment:
            #     continue
            avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
            temp = all_data[log]
            do = temp['do']
            di = temp['di']
            df = temp['data']
            df_instrip = df.where(df.instrip==True)
            count = 0
            for key in list(do.keys())[1:]:
                temp = do[key]
                if len(temp)>10:
                    temp = fn.find_cutoff(temp)
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()
                    x0 = x[0]
                    y0 = y[0]
                    x = x-x0
                    y = y-y0
                    # condition: fly must make it back to the edge. rotate trajectory to check
                    if np.abs(x[-1]-x[0])<1:
                        count+=1
                        if np.mean(x)>0: # align insides to the right and outsides to the left
                            x = -x
                        t = np.arange(len(x))
                        t_common = np.linspace(t[0], t[-1], pts)
                        fx = interpolate.interp1d(t, x)
                        fy = interpolate.interp1d(t, y)
                        #axs.plot(fx(t_common), fy(t_common))
                        avg_x_out.append(fx(t_common))
                        avg_y_out.append(fy(t_common))
            for key in list(di.keys())[1:]:
                temp = di[key]
                if len(temp)>10:
                    temp = fn.find_cutoff(temp)
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()
                    x0 = x[0]
                    y0 = y[0]
                    x = x-x0
                    y = y-y0
                    if np.abs(x[-1]-x[0])<1:
                        if np.mean(x)<0: # align insides to the right and outsides to the left
                            x = -x
                        t = np.arange(len(x))
                        t_common = np.linspace(t[0], t[-1], pts)
                        fx = interpolate.interp1d(t, x)
                        fy = interpolate.interp1d(t, y)
                        #axs.plot(fx(t_common), fy(t_common))
                        avg_x_in.append(fx(t_common))
                        avg_y_in.append(fy(t_common))

            # convert lists to arrays for normalization step below
            avg_x_in = np.array(avg_x_in)
            avg_x_out = np.array(avg_x_out)
            avg_y_in = np.array(avg_y_in)
            avg_y_out = np.array(avg_y_out)

            if count>6: # condition: each trajectory needs more than three outside trajectories
                print(log)
                x_traj = df.ft_posx.to_numpy()
                y_traj = df.ft_posy.to_numpy()
                x_traj_in = df_instrip.ft_posx.to_numpy()
                y_traj_in = df_instrip.ft_posy.to_numpy()
                fig, axs = plt.subplots(1,3)
                axs[0].plot(x_traj, y_traj)
                axs[0].plot(x_traj_in, y_traj_in, 'r')

                xmax, ymax = [], []
                for i in np.arange(len(avg_x_out)):
                    xmax.append(np.max(np.abs(avg_x_out[i])))
                    ymax.append(np.max(np.abs(avg_y_out[i])))
                    axs[1].plot(avg_x_out[i], avg_y_out[i], 'k', alpha=0.1)
                    axs[2].plot(avg_x_out[i]/xmax[-1], avg_y_out[i]/ymax[-1], 'k', alpha=0.1)
                xmax = np.array(xmax)
                xmax = xmax[:,np.newaxis]
                ymax = np.array(ymax)
                ymax = ymax[:, np.newaxis]
                axs[1].plot(np.mean(avg_x_out, axis=0),np.mean(avg_y_out, axis=0), color='k')
                axs[2].plot(np.mean(avg_x_out/xmax, axis=0),np.mean(avg_y_out/ymax, axis=0), color='k')
                animal_avg_x_out.append(np.mean(avg_x_out/xmax, axis=0))
                animal_avg_y_out.append(np.mean(avg_y_out/ymax, axis=0))
                
                xmax, ymax = [], []
                for i in np.arange(len(avg_x_in)):
                    xmax.append(np.max(np.abs(avg_x_in[i])))
                    ymax.append(np.max(np.abs(avg_y_in[i])))
                    axs[1].plot(avg_x_in[i], avg_y_in[i], 'r', alpha=0.1)
                    axs[2].plot(avg_x_in[i]/xmax[-1], avg_y_in[i]/ymax[-1], 'r', alpha=0.1)
                xmax = np.array(xmax)
                xmax = xmax[:,np.newaxis]
                ymax = np.array(ymax)
                ymax = ymax[:, np.newaxis]
                axs[1].plot(np.mean(avg_x_in, axis=0),np.mean(avg_y_in, axis=0), color='r')
                axs[2].plot(np.mean(avg_x_in/xmax, axis=0),np.mean(avg_y_in/ymax, axis=0), color='r')
                animal_avg_x_in.append(np.mean(avg_x_in/xmax, axis=0))
                animal_avg_y_in.append(np.mean(avg_y_in/ymax, axis=0))
                fig.suptitle(log)
                fig.savefig(os.path.join(self.figurefol, log.replace('.log', '.pdf')), transparent=True)


        # save the average trajectories
        fn.save_obj([animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out], os.path.join(self.picklefol, self.experiment+'_'+'average_trajectories.p'))


        # make an average of the averages for each fly
        fig, axs = plt.subplots(1,1)

        for i in np.arange(len(animal_avg_x_in)):
            axs.plot(animal_avg_x_in[i], animal_avg_y_in[i], 'r', alpha=0.1)
        axs.plot(np.mean(animal_avg_x_in, axis=0), np.mean(animal_avg_y_in, axis=0), 'r')
        exit_x = np.mean(animal_avg_x_in, axis=0)[-1]
        exit_y = np.mean(animal_avg_y_in, axis=0)[-1]

        for i in np.arange(len(animal_avg_x_out)):
            axs.plot(animal_avg_x_out[i]+exit_x, animal_avg_y_out[i]+exit_y, 'k', alpha=0.1)
        axs.plot(np.mean(animal_avg_x_out+exit_x, axis=0), np.mean(animal_avg_y_out+exit_y, axis=0), 'k')

        # draw the plume boundary line at the appropriate angle
        max_y_out = np.max(animal_avg_y_out)+exit_y
        max_y_in = np.max(animal_avg_y_in)
        max_y = np.max((max_y_out,max_y_in))

        axs.axis('equal')
        fig.savefig(os.path.join(self.figurefol, experiment+'_all_averages.pdf'), transparent = True)
        return axs

    def plot_plume_fuzziness(self, t_decay):
        """
        plot the plume "fuzziness"

        """
        all_data = self.load_trajectories()

        D = []

        for i, log in enumerate(list(all_data.keys())):
            temp = all_data[log]
            do = temp['do']
            di = temp['di']
            for key in list(do.keys()):
                temp = do[key]
                temp = fn.find_cutoff(temp)
                x = temp.ft_posx.to_numpy()
                y = temp.ft_posy.to_numpy()
                x0 = x[0]
                y0 = y[0]
                x = x-x0
                y = y-y0
                t = temp.seconds.to_numpy()
                t=t-t[0]
                if len(t)>t_decay:
                    if np.abs(x[-1]-x[0])<1: #returns to edge
                        if np.mean(x)>0: # align insides to the right and outsides to the left
                            x = -x
                        ix_t = np.argmin(np.abs(t-t_decay))
                        xi = x[ix_t]
                        yi = y[ix_t]
                        if xi<0:
                            D.append({
                                'condition': 'out',
                                'x': x,
                                'y': y,
                                'xi': xi,
                                'yi': yi,
                                'ix': ix_t
                            })
            for key in list(di.keys()):
                temp = di[key]
                temp = fn.find_cutoff(temp)
                x = temp.ft_posx.to_numpy()
                y = temp.ft_posy.to_numpy()
                x0 = x[0]
                y0 = y[0]
                x = x-x0
                y = y-y0
                t = temp.seconds.to_numpy()
                t=t-t[0]
                if len(t)>t_decay:
                    if np.abs(x[-1]-x[0])<1: #returns to edge
                        if np.mean(x)<0: # align insides to the right and outsides to the left
                            x = -x
                        ix_t = np.argmin(np.abs(t-t_decay))
                        xi = x[ix_t]
                        yi = y[ix_t]
                        if xi>0:
                            D.append({
                                'condition': 'in',
                                'x': x,
                                'y': y,
                                'xi': xi,
                                'yi': yi,
                                'ix': ix_t
                            })
        df = pd.DataFrame(D)

        # figure showing point in trajectory where odor is "on" as defined by t_decay
        fig, axs = plt.subplots(2,1, figsize=(2,3), sharex=True, gridspec_kw={'height_ratios': [1,0.5]})
        for color, cond in zip([pl.inside_color, pl.outside_color],['in', 'out']):
            df_temp = df[df.condition==cond]
            df_temp = df_temp.sample(n=20, random_state=2)
            for index, row in df_temp.iterrows():
                axs[0].plot(row.x, row.y, color, linewidth=0.5, alpha=0.5)
                axs[0].plot(row.x[0:row.ix+1], row.y[0:row.ix+1], color='k', linewidth=0.5)
                axs[0].plot(row.xi, row.yi, '.', color=color)
        axs[0].set_ylim(-5,15)
        axs[0].set_xlim(-10,10)
        axs[0].plot([0,0], [-5,15], color='k', linewidth=0.5)
        axs[0].plot(0,0,'.', color='k')
        axs[0].set_xticks([-5,0,5])
        axs[0].set_yticks([0,5])
        axs[0].set_ylabel('y position (mm)')
        axs[0].spines.bottom.set_bounds(-5,5)
        axs[0].spines.left.set_bounds(0,5)
        sns.despine(ax=axs[0])

        sns.histplot(data=df, x='xi', hue='condition', hue_order=['out', 'in'], palette=[pl.outside_color, pl.inside_color], element='step', stat='density', ax=axs[1])
        axs[1].set_xticks([-5,0,5])
        axs[1].spines.bottom.set_bounds(-5,5)
        axs[1].set_xlabel('x position (mm)')
        axs[1].set_yticks([0,0.1])
        axs[1].spines.left.set_bounds(0,0.1)
        axs[1].get_legend().remove()
        sns.despine(ax=axs[1])
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'plumefuzziness_on_pts.pdf')) 

        #cumulative distribution of x positions where odor is "on" as defined by t_decay
        fig, axs = plt.subplots(1,1, figsize=(2,2))
        df['xi'] = np.abs(df['xi'])
        sns.histplot(data=df, x='xi', hue='condition', hue_order=['out', 'in'], palette=[pl.outside_color, pl.inside_color], 
                     element='step', ax=axs, cumulative=True, stat="density", common_norm=False, fill=False)
        axs.set_ylabel('Density')
        axs.set_xlabel('x position (mm)')
        axs.set_yticks([0,0.5,1])
        axs.set_xticks([0,2,5])
        axs.spines.bottom.set_bounds(0,5)
        axs.get_legend().remove()
        sns.despine()
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'plume_fuzziness_cumulative.pdf'))

    def compare_tracking_norm(self):
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
        
        def constant_path(a,x,y):
            _,L = fn.path_length(a*x, a*y)
            return np.abs(1-L)
        
        # number of two point and three point
        two = 0
        three = 0

        # interpolation points
        pts=10000

        # load preprocessed data
        preprocessed = self.load_trajectories()

        # list of dictionaries for all inside and outside results
        all_results = []

        # make a folder for the images of simplifications
        individual_figures = os.path.join(self.figurefol,'path_simplifications')
        if not os.path.exists(individual_figures):
            os.makedirs(individual_figures)

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

            for i, dio in enumerate([do]):
                params = p.copy()
                if i==0:
                    params['io']='out'
                else:
                    params['io']='in'
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

                        # normalize each trajectory so that its pathlength is 1
                        xnorm = fx(t_common)
                        ynorm = fy(t_common)
                        a = 1/pathl
                        res = minimize(constant_path, a, method='nelder-mead',
                                    args=(xnorm,ynorm), options={'xatol': 0.01})
                        a = res.x

                        params['x_interp'].append(a*xnorm)
                        params['y_interp'].append(a*ynorm)
                        params['t_interp'].append(ft(t_common)/np.max(ft(t_common)))
                        params['s_interp'].append(fs(t_common)/np.max(np.abs(fs(t_common))))
                        params['xv_interp'].append(fxv(t_common)/np.max(np.abs(fxv(t_common))))
                        params['yv_interp'].append(fyv(t_common)/np.max(np.abs(fyv(t_common))))

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

                # RDP
                # simplified, num = fn.rdp_pts(xy0, nodes=4)
                simplified = fn.rdp_pts_optimize(xy0, nodes=4)

                # farthest point
                # ix = np.argmin(xy0[:,0])
                # simplified = np.array([xy0[0,:],xy0[ix,:], xy0[-1,:]])

                # top and bottom 90%
                # simplified = np.array([xy0[0,:], xy0[1000,:], xy0[9000,:], xy0[-1,:]])

                savename = log_name.replace('.log', '.pdf')
                
                # if num == 'two':
                #     two+=1
                #     savename = '2_'+log_name.replace('.log', '.pdf')
                #     print('2 = '+str(len(do)))
                # elif num == 'three':
                #     three+=1
                #     savename = '3_'+log_name.replace('.log', '.pdf')
                #     print('3 = '+str(len(do)))

                # make a figure for each individual
                fig, axs = plt.subplots(1,1, figsize=(2,2))
                axs.plot(xy0[:,0], xy0[:,1])
                axs.plot(simplified[:,0], simplified[:,1])
                axs.axis('equal')
                fig.savefig(os.path.join(individual_figures, savename))
                
                [params['out_angle'], params['in_angle']] = inbound_outbound_angle(x_interp, y_interp, simplified)
                [params['out_dist'], params['in_dist'], params['out_e'], params['in_e']] = inbound_outbound_lengths(x_interp,y_interp, simplified)
                [params['out_t'], params['in_t']] = inbound_outbound_time(x_interp,y_interp,t_interp, simplified)
                [params['out_s'], params['in_s']] = inbound_outbound_speed(x_interp,y_interp, s_interp, simplified)
                [params['xv_out'], params['xv_mid'], params['xv_in'], params['yv_out'], params['yv_mid'], params['yv_in']] = inbound_outbound_velocities(x_interp,y_interp, xv_interp, yv_interp, simplified)


                all_results.append(params)

            df = pd.DataFrame(all_results)
            fn.save_obj(df, os.path.join(self.picklefol, 'compare_tracking.p'))
        print('two line simplification = ', str(two),'. three line simplification =  ', str(three))
        return df    
    
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
        
        def return_results_template():
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
            return p
        
        # number of two point and three point
        two = 0
        three = 0

        # interpolation points
        pts=10000

        # load preprocessed data
        preprocessed = self.load_trajectories()

        # list of dictionaries for all inside and outside results
        all_results = []

        # make a folder for the images of simplifications
        individual_figures = os.path.join(self.figurefol,'path_simplifications')
        if not os.path.exists(individual_figures):
            os.makedirs(individual_figures)



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

            for i, dio in enumerate([di,do]):
                params = return_results_template()
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
                # simplified, num = fn.rdp_pts(xy0)

                # if num == 'two':
                #     two+=1
                #     savename = '2_'+log_name.replace('.log', '.pdf')
                #     print('2 = '+str(len(do)))
                # elif num == 'three':
                #     three+=1
                #     savename = '3_'+log_name.replace('.log', '.pdf')
                #     print('3 = '+str(len(do)))

                ix = np.argmin(xy0[:,0])
                simplified = np.array([xy0[0,:], xy0[ix,:], xy0[-1,:]])
                # simplified = fn.rdp_pts_optimize_g(xy0, nodes=4)

                # make a figure for each individual
                # fig, axs = plt.subplots(1,1, figsize=(2,2))
                # axs.plot(xy0[:,0], xy0[:,1])
                # axs.plot(simplified[:,0], simplified[:,1])
                # axs.axis('equal')
                # fig.savefig(os.path.join(individual_figures, savename))
                # plt.close(fig)

                
                

                [params['out_angle'], params['in_angle']] = inbound_outbound_angle(x_interp, y_interp, simplified)
                [params['out_dist'], params['in_dist'], params['out_e'], params['in_e']] = inbound_outbound_lengths(x_interp,y_interp, simplified)
                [params['out_t'], params['in_t']] = inbound_outbound_time(x_interp,y_interp,t_interp, simplified)
                [params['out_s'], params['in_s']] = inbound_outbound_speed(x_interp,y_interp, s_interp, simplified)
                [params['xv_out'], params['xv_mid'], params['xv_in'], params['yv_out'], params['yv_mid'], params['yv_in']] = inbound_outbound_velocities(x_interp,y_interp, xv_interp, yv_interp, simplified)


                all_results.append(params)

            df = pd.DataFrame(all_results)
            fn.save_obj(df, os.path.join(self.picklefol, 'compare_tracking.p'))
        print('two line simplification = ', str(two),'. three line simplification =  ', str(three))
        return df

    def load_comparison(self):
        # load the processed comparison data
        if os.path.exists(os.path.join(self.picklefol, 'compare_tracking.p')):
            df = fn.load_obj(os.path.join(self.picklefol, 'compare_tracking.p'))
        else:
            df = self.compare_tracking()
        return df
    
    def plot_comparison_old(self):
        """
        plot comparison between gradient and constant for all parameters
        """
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        colors = [pl.inside_color, pl.inside_color, pl.outside_color, pl.outside_color]

        df = self.compare_tracking()
        metrics = ['t','pathl','speed','up_vel','cross_speed','x_dist','y_dist']
        labels = ['time (s)', 'path length (mm)', 'speed (mm/s)', 'upwind velocity (mm/s)','y distance (mm)', 'x speed (mm/s)', 'x distance (mm)']
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
        for fig in figs:
            fig.tight_layout()
            sns.despine(fig)
        axs = np.concatenate((axs1, axs2, axs3))
        for i, metric in enumerate(metrics):
            dots = sns.stripplot(data=df, x='io', y=metric, hue='trial_type', dodge=True, size=4, alpha=0.7, ax=axs[i])
            box = sns.boxplot(data=df, x='io', y=metric, hue='trial_type', dodge=True, ax=axs[i])
            #return dots
            b=0
            for box in box.patches:
                if type(box) == matplotlib.patches.PathPatch:
                    box.set_color(colors[b])
                    if b%2==0:
                        alpha = 0.7
                    else:
                        alpha = 0.5
                        box.set_hatch('/////')
                        box.set_fill(False)
                    box.set_alpha(alpha)
                    b+=1
            for j in np.arange(0,4):
                dots.collections[j].set_color(colors[j])


            axs[i].get_legend().remove()
            axs[i].tick_params(axis='x', pad=-3)
            axs[i].tick_params(axis='y', pad=-3)
            axs[i].set_ylabel(labels[i])
            axs[i].set_xlabel('')
            #axs_sub[i]=axs[i].copy()
        fig1.savefig(os.path.join(self.figurefol, 'grad_v_cnst_1.pdf'))
        fig2.savefig(os.path.join(self.figurefol, 'grad_v_cnst_2.pdf'))
        fig3.savefig(os.path.join(self.figurefol, 'grad_v_cnst_3.pdf'))

    def plot_comparison(self):
        """
        plot comparison between gradient and constant for all parameters
        """
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        colors = [pl.inside_color, pl.inside_color, pl.outside_color, pl.outside_color]

        df = self.compare_tracking()
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

            sns.stripplot(data=df, x='io', y=metric, hue='trial_type', hue_order=['increasing gradient','constant odor'],palette=['black', 'grey'], dodge=True, size=4, alpha=0.4, ax=axs[i])
            sns.pointplot(data=df, x='io', y=metric, hue='trial_type', hue_order=['increasing gradient','constant odor'],palette=['black', 'grey'], dodge=.4, ax=axs[i], join=False, markers=['_','_'], errwidth=2.0)


            axs[i].get_legend().remove()
            axs[i].tick_params(axis='x', pad=-3)
            axs[i].tick_params(axis='y', pad=-3)
            axs[i].set_ylabel(labels[i])
            axs[i].set_xlabel('')
            label_pos = axs[i].xaxis.get_label().get_position()
            print(label_pos)
            #axs_sub[i]=axs[i].copy()
        fig1.savefig(os.path.join(self.figurefol, 'grad_v_cnst_1.pdf'))
        fig2.savefig(os.path.join(self.figurefol, 'grad_v_cnst_2.pdf'))
        fig3.savefig(os.path.join(self.figurefol, 'grad_v_cnst_3.pdf'))

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

    def plot_two_examples(self):
        """
        make an example plot with constant and gradient plumes and trajectories from the same fly
        """
        import matplotlib.cm as cm
        from matplotlib import colors
        constant_log = '10282020-213508_constantOdor2_Fly7.log'
        gradient_log = '10282020-212819_gradient1_Fly7.log'
        all_data = self.load_trajectories()

        # load gradient trace
        dg = all_data[gradient_log]
        df_g = dg['data']


        # load constant trace
        dc = all_data[constant_log]
        df_c = dc['data']

        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        fig,axs = plt.subplots(1,2,figsize = (2,4))

        # plot gradient strip
        axs[0].imshow(np.linspace(0, 1, 256).reshape(-1, 1),cmap=cm.Greys,extent=[-25, 25, 0, 1000], origin='lower')
        axs[0].plot([-25, 25], [-10,-10], 'k')
        axs[0].plot([-50, -50], [0,50], 'k')
        axs[0].text(-50,-50,str(50)+' mm', fontsize=7, font='Arial')
        axs[0] = pl.plot_trajectory(df_g, axs=axs[0])
        axs[0] = pl.plot_trajectory_odor(df_g, axs=axs[0])
        axs[0].axis('equal')
        axs[0].axis('off')
        axs[0].set_ylim([axs[0].get_ylim()[0], 1000]) #stop traces at y=1000
        xl = axs[0].get_xlim()
        yl = axs[0].get_ylim()

        # plot constant strip
        color = cm.Greys(np.arange(0,1000, 1))[85]
        cmap = colors.ListedColormap(color)
        axs[1].imshow(np.linspace(0, 1, 256).reshape(-1, 1), cmap = cmap,extent=[-25, 25, 0, 1000], origin='lower')
        axs[1] = pl.plot_trajectory(df_c, axs=axs[1])
        axs[1] = pl.plot_trajectory_odor(df_c, axs=axs[1])
        axs[1].set_xlim(xl)
        axs[1].set_ylim(yl)
        axs[1].axis('off')

        fig.savefig(os.path.join(self.figurefol, 'gradient_v_constant_plume.pdf'))

    def plot_comparison_inside_outside(self):
        """
        comparison for looking at the upwind distance inside and outside
        - eventually for plotting all other metrics inside and outside
        """
        # metrics
        # duration
        # crosswind range
        # upwind speed
        # crosswind speed

        metrics = ['x_dist', 'y_dist', 'speed', 'up_vel', 'cross_speed']
        metric_names = ['crosswind distance (mm)', 'upwind distance (mm)', 'speed (mm/s)', 'upwind speed (mm/s)', 'crosswind speed (mm/s)']

        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        sns.set_style("ticks")
        fig,axs = plt.subplots(1,len(metrics),figsize = (len(metrics),1.5))


        # load the data
        df = self.load_comparison()
        df_in = df[df['io']=='in']
        df_out = df[df['io']=='out']
        for i, metric in enumerate(metrics):
            print(metric_names[i])

            y_in = np.abs(df_in[metric].to_numpy())
            y_out = np.abs(df_out[metric].to_numpy())


            pl.paired_plot(axs[i], y_in,y_out, color1=pl.inside_color, color2=pl.outside_color,
                indiv_pts=True, scatter_scale=0, alpha=1, indiv_markers=False, mean_line=False)

            axs[i].spines['top'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['left'].set_linewidth(0.75)

            axs[i].set_ylabel(metric_names[i])
            axs[i].set_xticks([0,1])
            axs[i].set_xticklabels(['odor', 'air'], rotation=45)
            axs[i].xaxis.set_tick_params(width=0)
            axs[i].yaxis.set_tick_params(width=0.75)
            axs[i].yaxis.set_tick_params(length=2.5)
            

            [t.set_color(i) for (i,t) in
            zip([pl.inside_color, pl.outside_color],axs[i].xaxis.get_ticklabels())]
            print(' ')

        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'vertical_plume_inside_outside_upwind.pdf'))

    def inbound_outbound(self):
        """
        function for plotting inbound and outbound metrics
        """
        # load the data
        if os.path.exists(os.path.join(self.picklefol, 'compare_tracking.p')):
            df = fn.load_obj(os.path.join(self.picklefol, 'compare_tracking.p'))
        else:
            df = self.compare_tracking()

        
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('ticks')

        df = df[df['io']=='out']
        params = ['angle', 'dist', 't', 's', 'e']
        param_labels = ['angle (o)', 'distance (mm)', 'time (s)', 'speed (mm/s)', 'efficiency (mm/mm)']

        # for i, row in df.iterrows():
        #     fig, axs = plt.subplots(1,1)
        #     x = row['x_interp']
        #     y = row['y_interp']
        #     axs.plot(x,y)
        #     axs.set_title(str(row['out_angle'])+ '   '+str(row['in_angle']))
        #     axs.axis('equal')

        fig,axs = plt.subplots(1,len(params),figsize = (1.2*len(params), 1.2))
        for i, param in enumerate(params):
            outbound = 'out_'+param
            inbound = 'in_'+param
            outbound = df[outbound].to_numpy()
            inbound = df[inbound].to_numpy()
            if param == 'dist':
                set_log=True
            else:
                set_log=False
            pl.paired_plot(axs[i], outbound,inbound,color1=pl.inside_color, color2=pl.outside_color,
                indiv_pts=True, scatter_scale=0, alpha=1, indiv_markers=False, mean_line=False, log=set_log)
            
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].set_xticks([0,1])
            axs[i].set_xticklabels(labels = ['outbound', 'inbound'], rotation = 45)
            axs[i].set_ylabel(param_labels[i])
            
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'vertical_inbound_outbound.pdf'))
        
        fig, axs = plt.subplots(1,1, figsize=(2,2))
        params = ['out', 'mid', 'in']
        color = ['#ADD8E6', 'g', '#00008B']
        xv_avg = []
        yv_avg = []
        for i, param in enumerate(params):
            xv = 'xv_'+param
            yv = 'yv_'+param
            axs.plot(df[xv], df[yv], '.', color=color[i], alpha=1)
            xv_avg.append(np.mean(df[xv]))
            yv_avg.append(np.mean(df[yv]))
        axs.plot(xv_avg, yv_avg, '.-',color= 'k')
        axs.set_xlim(-10,10)
        axs.set_ylim(-10,10)
        axs.set_xticks([-10,0,10])
        axs.set_yticks([-10,0,10])
        axs.plot([-10, 10], [0,0], 'k', alpha=0.3)
        axs.plot([0,0], [-10, 10], 'k', alpha=0.3)
        axs.set_xlabel('x velocity (mm/s)')
        axs.set_ylabel('y velocity (mm/s)')
        fig.tight_layout()
        sns.despine()
        fig.savefig(os.path.join(self.figurefol, 'vertical_inbound_outbound_scatter.pdf'))

    def plot_outside_curvature(self):
        """
        3/6/2024 in progress, trying to show that curvature increases then decreases as fly returns
        """

        def calculate_curvature(x,y):
            dx = np.gradient(x)
            dy = np.gradient(y)
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            c = np.abs(dx*d2y-d2x*dy)/(dx**2+dy**2)**1.5
            return c
        
        all_curvatures = []
        
        # load the data
        if os.path.exists(os.path.join(self.picklefol, 'compare_tracking.p')):
            df = fn.load_obj(os.path.join(self.picklefol, 'compare_tracking.p'))
        else:
            df = self.compare_tracking()
        df = df.loc[df['io']=='out', :]
        for indx, row in df.iterrows():
            fig, axs = plt.subplots(1,1)
            x = row['x_interp']
            y = row['y_interp']
            axs.plot(x,y)
            x = fn.savgolay_smooth(x)
            y = fn.savgolay_smooth(y)
            c = rolling_apply(calculate_curvature,10,x,y)
            c = np.mean(c, axis=1)
            axs.plot(x,y)
            pl.colorline(axs=axs,x=x,y=y,z=c, norm=[0,0.01*np.max(c)])
            all_curvatures.append(c)
        fig,axs = plt.subplots(1,1)
        axs.plot(np.nanmean(all_curvatures, axis=0))

    def plot_outside_curvature2(self):
        """
        trying to use angular RDP to look at turn angles and run lengths to show more directed returns
        """
        def calculate_curvature(x,y):
            dx = np.gradient(x)
            dy = np.gradient(y)
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            c = np.abs(dx*d2y-d2x*dy)/((dx*dx+dy*dy)**1.5)
            return c
        def interpolate_var(var, pts=100):
            t_pts = np.arange(len(var))
            t_common = np.linspace(t_pts[0], t_pts[-1], pts)
            fvar = interpolate.interp1d(t_pts, var)
            return fvar(t_common)
        
        # load preprocessed data
        preprocessed = self.load_trajectories()


        # go through each fly and all its trials
        for log_name in list(preprocessed.keys()):
            all_angles = []
            all_lengths = []
            temp = preprocessed[log_name]
            do = temp['do']
            # d = temp['d']
            for bout in list(do.keys()):
                df_bout = do[bout]
                if fn.return_to_edge(df_bout): # only calculating information for bouts that return to the edge
                    x = df_bout.ft_posx.to_numpy()
                    y = df_bout.ft_posy.to_numpy()
                    xy0 = np.concatenate((x[:,None],y[:,None]),axis=-1)
                    simplified, _, angles, L = fn.rdp_simp_heading_angles_len(x,y, epsilon=1)
                    if len(simplified)>10:
                        try:
                            all_angles.append(interpolate_var(angles))
                        except:
                            print(simplified)
                            fig, axs = plt.subplots(1,1)
                            axs.plot(simplified[:,0], simplified[:,1])
                        all_lengths.append(interpolate_var(L))
                        # fig, axs = plt.subplots(1,3)
                        # axs[0].plot(x,y)
                        # axs[1].plot(np.abs(angles))
                        # axs[2].plot(L)
            all_angles = np.abs(all_angles)
            fig, axs = plt.subplots(2,1)
            for i in np.arange(len(all_angles)):
                axs[0].plot(all_angles[i], 'k', alpha=0.01)
            axs[0].plot(np.mean(all_angles,axis=0))
            
            for i in np.arange(len(all_lengths)):
                axs[1].plot(all_lengths[i], 'k', alpha=0.01)
            axs[1].plot(np.mean(all_lengths,axis=0))

    def plot_outside_curvature3(self):
        """
        use the nagel approach of curvature = angular/planer velocity
        """
        # load preprocessed data
        preprocessed = self.load_trajectories()
        # go through each fly and all its trials
        for log_name in list(preprocessed.keys()):
            all_angles = []
            all_lengths = []
            temp = preprocessed[log_name]
            do = temp['do']
            # d = temp['d']
            for bout in list(do.keys()):
                df_bout = do[bout]
                if fn.return_to_edge(df_bout):
                    fig, axs = plt.subplots(1,1)
                    _,df_bout = fn.find_stops(df_bout)
                    axs.plot(df_bout.ft_posx, df_bout.ft_posy)

                    df_bout.drop(df_bout.loc[df_bout['stop'].isnull()].index, inplace=True)
                    x = df_bout.ft_posx.to_numpy()
                    y = df_bout.ft_posy.to_numpy()
                    curvature = df_bout.curvature.to_numpy()
                    axs.plot(x,y)
                    pl.colorline(axs,x,y,curvature, norm=[0.4, 1])
                    print(np.mean(curvature), np.max(curvature))
            return

    def find_outbound_inbound_metrics(self):
        """
        find the inbound and outbound metrics for all trajectories
        """
        def find_single_outbound_inbound(df, returns=True):
            """
            for a 90 degree plume log file, calculate the outbound and inbound metrics
            """
            from scipy import interpolate

            delta_t = np.mean(np.gradient(df.seconds.to_numpy()))
            _,df = fn.find_stops(df)
            d,di,do = fn.inside_outside(df)
            tout = []
            tmid = []
            tin = []
            sout = []
            smid = []
            sin = []
            xvin = []
            xvmid = []
            xvout = []
            yvin = []
            yvmid = []
            yvout = []
            angout = []
            angmid = []
            angin = []
            plot=True
            for key in list(do.keys())[1:]:
                temp = do[key]
                temp = fn.find_cutoff(temp)
                x = temp.ft_posx.to_numpy()
                y = temp.ft_posy.to_numpy()
                t = temp.seconds.to_numpy()
                stop = temp.stop.to_numpy()
                x0 = x[0]
                y0 = y[0]
                x = x-x0
                y = y-y0
                if np.mean(x)>0:
                    x=-x
                if np.abs(x[-1]-x[0])<1: # returns to edge
                    max_ix = np.argmin(x) # maximum index

                    xy0 = np.concatenate((x[:,None],y[:,None]),axis=-1)
                    try:
                        simplified = fn.rdp_pts(xy0, epsilon=1)
                        out_ix = np.where(x==simplified[1,0])[0][0]
                        in_ix = np.where(x==simplified[-2,0])[0][0]
                        xv = np.gradient(x)*stop/delta_t # x speed
                        yv = np.gradient(y)*stop/delta_t # y speed

                        #first deal with the inbound and outbound
                        t1 = t[out_ix]-t[0]
                        t3 = t[-1]-t[in_ix]
                        # path lengths
                        _,s1 = fn.path_length(x[0:out_ix], y[0:out_ix])
                        _,s3 = fn.path_length(x[in_ix:], y[in_ix:])
                        # speeds
                        tout.append(t1)
                        tin.append(t3)
                        sout.append(s1/t1)
                        sin.append(s3/t3)
                        # velocities
                        xvout.append(np.nanmean(xv[0:out_ix]))
                        yvout.append(np.nanmean(yv[0:out_ix]))
                        xvin.append(np.nanmean(xv[in_ix:]))
                        yvin.append(np.nanmean(yv[in_ix:]))
                        
                        # angles
                        angout = np.array([np.arctan2(y[out_ix]-y[0], x[out_ix]-x[0])])
                        angin = np.array([np.arctan2(y[-1]-y[in_ix], x[-1]-x[in_ix])])
                        angout = fn.conv_cart_upwind(angout)
                        angin = fn.conv_cart_upwind(angin)

                        # now deal with the mid segment
                        if len(simplified)==4:
                            t2 = t[in_ix]-t[out_ix]
                            _,s2 = fn.path_length(x[out_ix:in_ix], y[out_ix:in_ix])
                            tmid.append(t2)
                            smid.append(s2/t2)
                            xvmid.append(np.nanmean(xv[out_ix:in_ix]))
                            yvmid.append(np.nanmean(yv[out_ix:in_ix]))
                            angmid = np.array([np.arctan2(y[in_ix]-y[out_ix], x[in_ix]-x[out_ix])])
                            angmid = fn.conv_cart_upwind(angmid)

                    except:
                        print('Recursion limit: could not find dp simplification')
                        continue
            d = {
                'tout' : np.nanmean(tout),
                'tmid' : np.nanmean(tmid),
                'tin' : np.nanmean(tin),
                'sout' : np.nanmean(sout),
                'smid' : np.nanmean(smid),
                'sin' : np.nanmean(sin),
                'xvout' : np.nanmean(xvout),
                'yvout' : np.nanmean(yvout),
                'xvmid' : np.nanmean(xvmid),
                'yvmid' : np.nanmean(yvmid),
                'xvin' : np.nanmean(xvin),
                'yvin' : np.nanmean(yvin),
                'angout' : fn.circmean(angout),
                'angmid' : fn.circmean(angmid),
                'angin' : fn.circmean(angin)
            }
            # fn.save_obj(d,os.path.join(self.picklefol, 'inbound_outbound_metrics.p'))
            return d

        all_data = self.load_trajectories()
        
        d_all = {
        'all_animals_out_xv' : [],
        'all_animals_mid_xv' : [],
        'all_animals_in_xv' : [],
        'all_animals_out_yv' : [],
        'all_animals_mid_yv' : [],
        'all_animals_in_yv' : [],
        'all_animals_out_ang' : [],
        'all_animals_mid_ang' : [],
        'all_animals_in_ang' : [],
        }
        
        for log in all_data.keys():
            # deal with rotated plume

            df = all_data[log]['data']
            do = all_data[log]['do']
            if len(do)>6:
                d = find_single_outbound_inbound(df)
                d_all['all_animals_out_xv'].append(d['xvout'])
                d_all['all_animals_mid_xv'].append(d['xvmid'])
                d_all['all_animals_in_xv'].append(d['xvin'])
                d_all['all_animals_out_yv'].append(d['yvout'])
                d_all['all_animals_mid_yv'].append(d['yvmid'])
                d_all['all_animals_in_yv'].append(d['yvin'])
                d_all['all_animals_out_ang'].append(d['angout'])
                d_all['all_animals_mid_ang'].append(d['angmid'])
                d_all['all_animals_in_ang'].append(d['angin'])
        fn.save_obj(d_all, os.path.join(self.picklefol, 'inbound_outbound_metrics.p'))
        return d_all

    def inbound_outbound_speed_paired(self, overwrite=True):
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        fig, axs = plt.subplots(1,1, figsize=(2,2))
        # axs.plot([0,0], [-20,0], 'k', linewidth=1)

        filename = os.path.join(self.picklefol, 'inbound_outbound_metrics.p')
        if not os.path.exists(filename) or overwrite:
            d = self.find_outbound_inbound_metrics()
        else:
            d = fn.load_obj(filename)

        axs.plot(d['all_animals_out_xv'], d['all_animals_out_yv'], 'o', color='#ADD8E6')
        axs.plot(d['all_animals_mid_xv'], d['all_animals_mid_yv'], 'o', color='g')
        axs.plot(d['all_animals_in_xv'], d['all_animals_in_yv'], 'o', color='#00008B')
        xavg = [np.nanmean(d['all_animals_out_xv']), np.nanmean(d['all_animals_mid_xv']), np.nanmean(d['all_animals_in_xv'])]
        yavg = [np.nanmean(d['all_animals_out_yv']), np.nanmean(d['all_animals_mid_yv']), np.nanmean(d['all_animals_in_yv'])]
        axs.plot(xavg, yavg, '.-', color='k')
        axs.set_xlim(-10,10)
        axs.set_ylim(-10,10)
        axs.set_xticks([-10,0,10])
        axs.set_yticks([-10,0,10])
        axs.plot([-10, 10], [0,0], 'k', alpha=0.3)
        axs.plot([0,0], [-10, 10], 'k', alpha=0.3)
        axs.set_xlabel('x velocity (mm/s)')
        axs.set_ylabel('y velocity (mm/s)')
        # pl.paired_plot(axs, all_animals_out, all_animals_in, color1='#ADD8E6', color2='#00008B', mean_line=False, scatter_scale=0, print_stat=True)
        # axs.set_ylabel('tortuosity (mm/mm)')
        # axs.set_xticks([0,1])
        # axs.set_xticklabels(['outbound', 'inbound'])
        #axs.set_yticks([45, 90, 135])
        fig.tight_layout()
        sns.despine()
        fig.savefig(os.path.join(self.figurefol, '0_outbound_inbound_velocities.pdf'))

    def inbound_outbound_angles_paired(self, overwrite=False):
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')

        # axs.plot([0,0], [-20,0], 'k', linewidth=1)

        filename = os.path.join(self.picklefol, 'inbound_outbound_metrics.p')
        if not os.path.exists(filename) or overwrite:
            d = self.find_outbound_inbound_metrics()
        else:
            d = fn.load_obj(filename)

        angout = d['all_animals_out_ang']
        angmid = d['all_animals_mid_ang']
        angin = d['all_animals_in_ang']

        fig, axs = plt.subplots(1,1, figsize=(2,2), subplot_kw={'projection': 'polar'})
        segs = [angout, angmid, angin]
        colors = ['#ADD8E6', 'g', '#00008B']
        for i, seg in enumerate(segs):
            seg = np.array(seg)
            seg = (seg+np.pi) % (2*np.pi) - np.pi
            seg=-seg
            for ang in seg:
                axs.plot([ang,ang], [0,1.8] ,color=colors[i], linewidth=0.5, alpha=0.5)
            ang_avg = fn.circmean(seg)
            axs.plot([ang_avg,ang_avg], [0,2] ,color=colors[i], linewidth=2)
        axs.set_theta_offset(np.pi/2)
        axs.set_yticks([])
        axs.set_xticks([])
        axs.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])

        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, '90_outbound_inbound_angles.pdf'))

    def plot_average_trajectories(self, plot_individuals, pts = 10000):
        """
        for plotting average trajectories for the 0 degree plume.  Also can
        plot the average trajectories for each individual trial
        """
        # load preprocessed data
        all_data = self.load_trajectories()
        n=0

        # plotting parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        fig,axs = plt.subplots(1,1, figsize=(2.5,2.5))

        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = [],[],[],[]
        for i, log in enumerate(list(all_data.keys())):
            avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
            temp = all_data[log]
            do = temp['do']
            di = temp['di']
            df = temp['data']

            for key in list(do.keys())[1:]:
                temp = do[key]
                if len(temp)>10:
                    temp = fn.find_cutoff(temp)
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()
                    x0 = x[0]
                    y0 = y[0]
                    x = x-x0
                    y = y-y0

                    if np.abs(x[-1]-x[0])<1:

                        if np.mean(x)<0: # align insides to the right and outsides to the left
                            x = -x
                        t = np.arange(len(x))
                        t_common = np.linspace(t[0], t[-1], pts)
                        fx = interpolate.interp1d(t, x)
                        fy = interpolate.interp1d(t, y)

                        avg_x_out.append(fx(t_common))
                        avg_y_out.append(fy(t_common))
            for key in list(di.keys())[1:]:
                temp = di[key]
                if len(temp)>10:
                    temp = fn.find_cutoff(temp)
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()
                    x0 = x[0]
                    y0 = y[0]
                    x = x-x0
                    y = y-y0

                    if np.abs(x[-1]-x[0])<1:
                        if np.mean(x)>0: # align insides to the right and outsides to the left
                            x = -x

                        t = np.arange(len(x))
                        t_common = np.linspace(t[0], t[-1], pts)
                        fx = interpolate.interp1d(t, x)
                        fy = interpolate.interp1d(t, y)
                        #axs.plot(fx(t_common), fy(t_common))
                        avg_x_in.append(fx(t_common))
                        avg_y_in.append(fy(t_common))

            # make individual animal plots
            if plot_individuals:
                individual_plots_fol = os.path.join(self.figurefol, 'individual_average_trajectories')
                if not os.path.exists(individual_plots_fol):
                    os.makedirs(individual_plots_fol)
                fig2, axs2 = plt.subplots(1,1, figsize=(2.5,2.5))

                # inside trajectories
                for i in np.arange(len(avg_x_in)):
                    axs2.plot(avg_x_in[i], avg_y_in[i], color=pl.inside_color, alpha=0.2)
                axs2.plot(np.mean(avg_x_in, axis=0), np.mean(avg_y_in, axis=0), pl.inside_color, linewidth=2)
                exit_x = np.mean(avg_x_in, axis=0)[-1]
                exit_y = np.mean(avg_y_in, axis=0)[-1]

                # outside trajectories
                for i in np.arange(len(avg_x_out)):
                    axs2.plot(avg_x_out[i]+exit_x, avg_y_out[i]+exit_y, pl.outside_color, alpha=0.2)
                axs2.plot(np.mean(avg_x_out+exit_x, axis=0), np.mean(avg_y_out+exit_y, axis=0), pl.outside_color, linewidth=2)

                # scale bar
                axs2.plot([-5,5], [-5,-5], 'k')
                axs2.text(-7, -12, '10 mm')

                # plume boundary
                axs2.plot([0,0], [0,np.max(avg_y_in)], 'k', linewidth=0.5, linestyle='--')

                # axis parameters
                axs2.axis('equal')
                axs2.axis('off')
                axs2.set_title(log.replace('.log', ''))

                fig2.savefig(os.path.join(individual_plots_fol, log.replace('.log', '.pdf')))


            animal_avg_x_out.append(np.mean(avg_x_out, axis=0))
            animal_avg_y_out.append(np.mean(avg_y_out, axis=0))
            animal_avg_x_in.append(np.mean(avg_x_in, axis=0))
            animal_avg_y_in.append(np.mean(avg_y_in, axis=0))

        # plot the inside trajectories
        for i in np.arange(len(animal_avg_x_in)):
            axs.plot(animal_avg_x_in[i], animal_avg_y_in[i], color=pl.inside_color, alpha=0.2)
        axs.plot(np.mean(animal_avg_x_in, axis=0), np.mean(animal_avg_y_in, axis=0), pl.inside_color, linewidth=2)
        exit_x = np.mean(animal_avg_x_in, axis=0)[-1]
        exit_y = np.mean(animal_avg_y_in, axis=0)[-1]

        # plot the outside trajectories with the exit point offset
        for i in np.arange(len(animal_avg_x_out)):
            # fig2,axs2 = plt.subplots(1,1)
            # axs2.plot(animal_avg_x_out[i]+exit_x, animal_avg_y_out[i]+exit_y, pl.outside_color)
            # axs2.axis('equal')
            axs.plot(animal_avg_x_out[i]+exit_x, animal_avg_y_out[i]+exit_y, pl.outside_color, alpha=0.2)
        axs.plot(np.mean(animal_avg_x_out+exit_x, axis=0), np.mean(animal_avg_y_out+exit_y, axis=0), pl.outside_color, linewidth=2)

        # scale bar
        axs.plot([-5,5], [-5,-5], 'k')
        axs.text(-7, -12, '10 mm')

        # plume boundary
        axs.plot([0,0], [0,120], 'k', linewidth=0.5, linestyle='--')

        # axis parameters
        axs.axis('equal')
        axs.axis('off')

        # save
        fig.savefig(os.path.join(self.figurefol, 'average_trajectories_all.pdf'))

        print('Data from '+str(i+1)+' animals')

    def improvement_over_time(self, normalize=True, plot_individual=False, plot_pts=False, set_log=True, tt='constant odor'):
        """
        function for looking at improvement over time for the vertical plume
        Really only want to look at this for the constant plume
        """


        all_data = self.load_trajectories()
        all_results = []

        for log in list(all_data.keys()):
            params = {
                'log': log,
                'o_t':[],
                'o_d':[],
                'o_e':[],
                'mean_x':[]
            }
            temp = all_data[log]
            do = temp['do']
            data = temp['data']
            trial_type = temp['trial_type']
            if (len(do.keys())>=10) and (trial_type==tt):
                for key in list(do.keys())[0:]:
                    if not fn.return_to_edge(do[key]):
                        continue
                    else:
                        t = do[key].seconds.to_numpy()
                        del_t = t[-1]-t[0]
                        params['o_t'].append(del_t)
                        x = do[key].ft_posx.to_numpy()
                        y = do[key].ft_posy.to_numpy()
                        _,dis = fn.path_length(x,y)
                        dis_away = np.max(np.abs(x-x[0]))
                        params['o_d'].append(dis)
                        params['o_e'].append(dis_away/dis)
                        params['mean_x'].append(np.mean(x)-x[0]) # average x position
                params['o_d'] = params['o_d']
                params['o_t'] = params['o_t']
                all_results.append(params)
                if plot_individual:
                    fig, axs = plt.subplots(1,2, figsize=(6,3))
                    pl.plot_trajectory(data, axs[0])
                    pl.plot_trajectory_odor(data, axs[0])
                    axs[1].plot(params['o_t'], 'o')


        o_t = []
        o_d = []
        o_e = []
        for params in all_results:
            o_t.append(params['o_t'])
            o_d.append(params['o_d'])
            o_e.append(params['o_e'])
        o_t = fn.list_matrix_nan_fill(o_t)
        o_d = fn.list_matrix_nan_fill(o_d)
        o_e = fn.list_matrix_nan_fill(o_e)
        if normalize:
            norm = np.expand_dims(np.nanmax(o_t, axis=1), axis=1)
            o_t = o_t/norm
            norm = np.expand_dims(np.nanmax(o_d, axis=1), axis=1)
            o_d = o_d/norm
            norm = np.expand_dims(np.nanmax(o_e, axis=1), axis=1)
            o_e = o_e/norm

        # plot the results
        distance_color = pl.lighten_color(pl.outside_color,1.0)
        time_color = pl.lighten_color(pl.outside_color,2.5)
        efficiency_color = pl.lighten_color(pl.outside_color,3.0)

        fig,axs = plt.subplots(1,1, figsize=(2.5,2))
        fig2,axs2 = plt.subplots(1,1, figsize=(2.5,2))
        # concise version for with groups
        fig3,axs3 = plt.subplots(1,1, figsize=(1.1,1.5))
        fig4,axs4 = plt.subplots(1,1, figsize=(1.1,1.5))
        fig5,axs5 = plt.subplots(1,1, figsize=(1.1,1.5))


        axs.set_xlabel('outside trajectory (n)')
        axs.set_ylabel('time (s)', color=time_color)
        axs.set_xticks([1,10,20])
        axs.set_yticks([20,100])
        if plot_pts:
            for b,row in enumerate(o_t[:, 0:21]):
                axs.plot(row, 'o', color = time_color, markersize=3, alpha=0.2)
        # axs.set_yscale("log")
        axs.plot(np.nanmean(o_t[:, 0:21],axis=0), '-o', color = time_color)
        if set_log:
            axs.set_yscale('log')
        sns.despine(ax=axs)

        axs2.set_ylabel('time (s)', color=time_color)
        axs2.set_yticks([100,500])
        axs2.set_xticks([1,10,20])
        axs2.set_ylabel('distance (mm)', color=distance_color)
        axs2.plot(np.nanmean(o_d[:, 0:21],axis=0), '-o', color = distance_color)
        if plot_pts:
            for b,row in enumerate(o_d[:, 0:21]):
                axs2.plot(row, 'o', color = distance_color, markersize=3, alpha=0.2)
        sns.despine(ax=axs2)


        # plot condensed version for time
        axs3.set_xlabel('outside trajectory (n)', fontsize=6)
        axs3.set_ylabel('time (s)', color=time_color, fontsize=6)
        axs3.set_xticks([0,1,2])
        axs3.set_xticklabels(['1', '2-5', '5-10'], fontsize=6)
        axs3.set_yticks([20,100])
        axs3.set_xlim(-0.5,2.5)
        axs3.set_title('vertical plume', fontsize=6)
        for axis in ['top','bottom','left','right']:
            axs3.spines[axis].set_linewidth(0.75)
        axs3.tick_params(width=0.75, length=4)
        axs3.tick_params(axis='y',width=0.75, length=4, pad=-1, labelsize=6)
        axs3.tick_params(axis='x', width=0.0, length=0)
        axs3.xaxis.labelpad = 0
        axs3.yaxis.labelpad = 0
        if plot_pts:
            for b,row in enumerate(o_t[:, 0:21]):
                axs3.plot(0, row[0], 'o', color = time_color, markersize=3, alpha=0.2)
                axs3.plot(1, np.mean(row[1:5]), 'o', color = time_color, markersize=3, alpha=0.2)
                axs3.plot(2, np.mean(row[6:11]), 'o', color = time_color, markersize=3, alpha=0.2)
        # axs.set_yscale("log")
        ts = [np.nanmean(o_t[:, 0],axis=0), np.nanmean(o_t[:, 1:5]), np.nanmean(o_t[:, 6:11])]
        axs3.plot([0,1,2], ts, '-o', color = time_color)
        if set_log:
            axs3.set_yscale('log')
        sns.despine(ax=axs3)
        axs3.minorticks_off()

        # plot condensed version for distance
        axs4.set_xlabel('outside trajectory (n)', fontsize=6)
        axs4.set_ylabel('distance (mm)', color=distance_color, fontsize=6)
        axs4.set_xticks([0,1,2])
        axs4.set_xticklabels(['1', '2-5', '5-10'], fontsize=6)
        axs4.set_xlim(-0.5,2.5)
        axs4.set_title('vertical plume', fontsize=6)
        for axis in ['top','bottom','left','right']:
            axs4.spines[axis].set_linewidth(0.75)
        axs4.tick_params(width=0.75, length=4)
        axs4.tick_params(axis='y',width=0.75, length=4, pad=-1, labelsize=6)
        axs4.tick_params(axis='x', width=0.0, length=0)
        axs4.xaxis.labelpad = 0
        axs4.yaxis.labelpad = 0
        if plot_pts:
            for b,row in enumerate(o_d[:, 0:21]):
                axs4.plot(0, row[0], 'o', color = distance_color, markersize=3, alpha=0.2)
                axs4.plot(1, np.mean(row[1:5]), 'o', color = distance_color, markersize=3, alpha=0.2)
                axs4.plot(2, np.mean(row[6:11]), 'o', color = distance_color, markersize=3, alpha=0.2)
        # axs.set_yscale("log")
        ds = [np.nanmean(o_d[:, 0],axis=0), np.nanmean(o_d[:, 1:5]), np.nanmean(o_d[:, 6:11])]
        axs4.plot([0,1,2], ds, '-o', color = distance_color)
        if set_log:
            axs4.set_yscale('log')
        sns.despine(ax=axs4)
        axs4.minorticks_off()

        # plot condensed version for efficiency
        axs5.set_xlabel('outside trajectory (n)', fontsize=6)
        axs5.set_ylabel('efficiency (mm/mm)', color=efficiency_color, fontsize=6)
        axs5.set_xticks([0,1,2])
        axs5.set_xticklabels(['1', '2-5', '5-10'], fontsize=6)
        axs5.set_xlim(-0.5,2.5)
        axs5.set_title('vertical plume', fontsize=6)
        for axis in ['top','bottom','left','right']:
            axs5.spines[axis].set_linewidth(0.75)
        axs5.tick_params(width=0.75, length=4)
        axs5.tick_params(axis='y',width=0.75, length=4, pad=-1, labelsize=6)
        axs5.tick_params(axis='x', width=0.0, length=0)
        axs5.xaxis.labelpad = 0
        axs5.yaxis.labelpad = 0
        if plot_pts:
            for b,row in enumerate(o_e[:, 0:21]):
                axs5.plot(0, row[0], 'o', color = efficiency_color, markersize=3, alpha=0.2)
                axs5.plot(1, np.mean(row[1:5]), 'o', color = efficiency_color, markersize=3, alpha=0.2)
                axs5.plot(2, np.mean(row[6:11]), 'o', color = efficiency_color, markersize=3, alpha=0.2)
        # axs.set_yscale("log")
        ds = [np.nanmean(o_e[:, 0],axis=0), np.nanmean(o_e[:, 1:5]), np.nanmean(o_e[:, 6:11])]
        axs5.plot([0,1,2], ds, '-o', color = efficiency_color)
        # if set_log:
        #     axs5.set_yscale('log')
        sns.despine(ax=axs5)
        axs5.minorticks_off()


        if set_log:
            axs2.set_yscale('log')
        fig.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()
        fig5.tight_layout()
        if plot_pts:
            c='_with_pts_'
        else:
            c='_without_pts_'
        if set_log:
            d='_with_log_'
        else:
            d='_without_log_'
        if normalize:
            e='_normalized_'
        else:
            e='_not_normalized_'
        f = '_'+str(tt).replace(' ','_')
        fig.savefig(os.path.join(self.figurefol, 'improvement_time_'+c+d+e+f+'.pdf'))
        fig2.savefig(os.path.join(self.figurefol, 'improvement_distance_'+c+d+e+f+'.pdf'))
        fig3.savefig(os.path.join(self.figurefol, 'improvement_time_concise_'+c+d+e+f+'.pdf'))
        fig4.savefig(os.path.join(self.figurefol, 'improvement_distance_concise_'+c+d+e+f+'.pdf'))
        fig5.savefig(os.path.join(self.figurefol, 'improvement_efficiency_concise_'+c+d+e+f+'.pdf'))

        # statistics
        pre = o_d[:,0]
        post = np.nanmean(o_d[:,1:5], axis=1)
        print(stats.ttest_rel(pre, post, nan_policy='omit'))
        return axs4

    def paired_comparison(self, metric = 'dist_up_plume'):
        """
        for any plume, calculate the number of returns per total pathlength.
        For multiple trials per fly, average the number of returns per trial
        """
        
        df = self.sheet

        # collect all calculated data in one place
        all_data = []
        all_data = self.compare_upwind_tracking(all_data, df)
        df = pd.DataFrame(all_data)
        # store average return per m for experimental and control animals
        pairs = []
        # go through each animal and calculate upwind for all genotypes
        for fly in df.fly.unique():
            df_fly = df[df.fly==fly]
            df_group = df_fly.groupby('trial_type')
            gradient = df_group.get_group('gradient')[metric].mean()
            constant = df_group.get_group('constant')[metric].mean()
            y=[gradient, constant]
            pairs.append(y)
        pairs = np.array(pairs)
        fig, axs = plt.subplots(1,1)
        pl.paired_plot(axs,pairs[:,0], pairs[:,1])
        #fig.savefig(os.path.join(self.figurefol, metric+'.pdf'))

        return pairs

    def high_v_low_inside(self):
        """
        look at the distance tracked up the plume for high and low odor concentrations inside the plume
        """
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')

        # load preprocessed data
        preprocessed = self.load_trajectories()

        lows = []
        highs = []

        # go through each fly and all its trials
        for log_name in list(preprocessed.keys()):
            temp = preprocessed[log_name]
            trial_type = temp['trial_type']
            
            di = temp['di']
            
            if trial_type == 'increasing gradient':
                for key in list(di.keys()):
                    df = di[key]
                    lo = np.min(df.mfc3_stpt)
                    hi = np.max(df.mfc3_stpt)

                    if (lo>0.2) & (hi<0.4):
                        t1 = df.seconds.iloc[-1]
                        t0 = df.seconds.iloc[0]
                        dt = t1-t0
                        s = np.mean(df.speed)
                        lows.append(s)
                    
                    elif (lo>0.6) & (hi<0.8):
                        t1 = df.seconds.iloc[-1]
                        t0 = df.seconds.iloc[0]
                        dt = t1-t0
                        s = np.mean(df.speed)
                        highs.append(s)

        print(stats.mannwhitneyu(lows, highs))

        fig, axs = plt.subplots(1,1, figsize = (1.5,2))
        x1 = np.random.normal(loc=0.0, scale=0.05, size=len(lows))
        x2 = np.random.normal(loc=0.5, scale=0.05, size=len(highs))
        axs.plot(x1, lows, '.', color = 'silver')
        axs.plot(x2, highs, '.', color = 'dimgrey')
        axs.set_xticks([0,0.5])
        axs.set_xticklabels(['20-40%', '60-80%'])
        axs.set_ylabel('speed (mm/s)')
        fig.tight_layout()
        sns.despine()

        fig.savefig(os.path.join(self.figurefol, 'low_v_high_concen_speed.pdf'))

    def heatmap(self,cmap='Blues', overwrite = False, plot_individual=False, res = 5):
        @jit(nopython=True)
        def populate_heatmap(xi, yi, x_bounds, y_bounds, fly_density):
            x_previous,y_previous = 1000,1000
            for i in np.arange(len(xi)):
                if min(x_bounds)<=xi[i]<=max(x_bounds) and min(y_bounds)<=yi[i]<=max(y_bounds):
                    x_index = np.argmin((xi[i]-x_bounds)**2)
                    y_index = np.argmin((yi[i]-y_bounds)**2)
                    if x_index != x_previous or y_index != y_previous:
                        fly_density[x_index, y_index] += 1
                        x_previous = x_index
                        y_previous = y_index
            return fly_density

        filename = os.path.join(self.picklefol, 'heatmaps.p')
        d = {}
        examples = ['08212020-212557_45degOdorRight.log','07102020-201510_Odor.log','08212020-171137_15degOdorRight.log', '03042022-183431_T_Plume_Fly4.log']
        end_xy = []


        # set up arrays for heatmap
        x_bounds = np.arange(-200, 200, res)
        y_bounds = np.arange(0, 1100, res)

        x_previous = [1000, 1000]
        y_previous = [1000, 1000]
        x_previous_rotate = [1000, 1000]
        y_previous_rotate = [1000, 1000]
        x_bounds_rotate = np.arange(-300, 300, res)
        y_bounds_rotate = np.arange(0, 1000, res)
        d['x_bounds'] = x_bounds
        d['y_bounds'] = y_bounds
        d['x_bounds_rotate'] = x_bounds_rotate
        d['y_bounds_rotate'] = y_bounds_rotate
        fly_density = np.zeros((len(x_bounds), len(y_bounds)))
        fly_density_rotate = np.zeros((len(x_bounds_rotate), len(y_bounds_rotate)))

        all_data = self.load_trajectories()

        for i, log in enumerate(list(all_data.keys())):
            print(log)
            df = all_data[log]['data']
            # crop trajectories so that they start when the odor turns on
            idx_start = df.index[df.instrip==True].tolist()[0]
            idx_end = df.index[df.instrip==True].tolist()[-1]
            df = df.iloc[idx_start:idx_end]
            xi = df.ft_posx.to_numpy()
            yi = df.ft_posy.to_numpy()
            xi=xi-xi[0]
            yi=yi-yi[0]

            # flip trajectories
            if np.mean(df.ft_posx)>0:
                print('trajectory flipped')
                xi=-xi

            # save and example trajectory to plot on top of the heatmap
            if log in examples:
                x_ex = xi
                y_ex = yi
            
            # end points
            end_xy.append([xi[-1], yi[-1]])

            #individual plots
            if plot_individual == True:
                fig, axs = plt.subplots(1,1)
                axs = pl.plot_trajectory(df,axs)
                axs = pl.plot_trajectory_odor(df,axs)
                axs.title.set_text(log)

            # rotate the tilted plume to make them vertical.
            angle = 0
            rot_angle = -angle*np.pi/180
            if self.experiment == '90':
                xir,yir = fn.coordinate_rotation(xi,yi-275,rot_angle)
            else:
                xir,yir = fn.coordinate_rotation(xi,yi,rot_angle)

            fly_density = populate_heatmap(xi, yi, x_bounds, y_bounds, fly_density)
            fly_density_rotate = populate_heatmap(xir, yir, x_bounds_rotate, y_bounds_rotate, fly_density_rotate)

        fly_density = np.rot90(fly_density, k=1, axes=(0,1))
        fly_density = fly_density/np.sum(fly_density)
        fly_density_rotate = np.rot90(fly_density_rotate, k=1, axes=(0,1))
        d['fly_density'] = fly_density
        d['fly_density_rotate'] = fly_density_rotate
        fn.save_obj(d, filename)

        fig, axs = plt.subplots(1,1,figsize=(2,2))
        vmin = np.percentile(fly_density[fly_density>0],0)
        vmax = np.percentile(fly_density[fly_density>0],95)
        im = axs.imshow(fly_density, cmap=cmap, vmin=vmin,vmax = vmax, rasterized=True, extent=(min(x_bounds), max(x_bounds), min(y_bounds), max(y_bounds)))

        axs.plot([-25,-25], [np.min(y_bounds), np.max(y_bounds)], color='k', linewidth=0.5)
        axs.plot([25,25], [np.min(y_bounds), np.max(y_bounds)], color='k', linewidth=0.5)

        # plot end points
        for xy in end_xy:
            if xy[1]<1100:
                xx = [-25, 25]
                yy = [xy[1], xy[1]]
                axs.plot(xx,yy,color='grey', linewidth=0.5)
        # axs.plot(x_ex, y_ex, 'red', linewidth=1)#, alpha = 0.8, linewidth=0.5)
        axs.axis('off')

        # colorbar
        cb = fig.colorbar(im, shrink=0.25, drawedges=False)
        cb.outline.set_visible(False)

        # scalebar
        x1=200
        x2=300
        axs.plot([x1,x2], [300,300], 'k', linewidth=0.5)
        axs.text(x1-100,200, str(x2-x1)+'mm')

        # clean up plot
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'heatmap.pdf'))

        # make rotated plot
        fig, axs = plt.subplots(1,1)
        # vmin = np.percentile(fly_density_rotate[fly_density_rotate>0],10)
        # vmax = np.percentile(fly_density_rotate[fly_density_rotate>0],90)
        axs.imshow(fly_density_rotate, cmap=cmap, vmin=1,vmax = 6, extent=(min(x_bounds_rotate), max(x_bounds_rotate), min(y_bounds_rotate), max(y_bounds_rotate)))


        # projection
        boundary=25
        fly_density_projection = fly_density_rotate[0:-40, :]/np.sum(fly_density_rotate[0:-40, :])
        x_mean = np.sum(fly_density_projection, axis = 0)
        fig, axs = plt.subplots(1,1, figsize=(2,2))
        axs.plot([-boundary, -boundary], [min(x_mean), max(x_mean)], 'k', alpha=0.5)
        axs.plot([boundary, boundary], [min(x_mean), max(x_mean)],'k', alpha=0.5)
        axs.plot(x_bounds_rotate, x_mean, color='k')
        axs.set_xlabel('x position (mm)')
        axs.set_ylabel('occupancy')
        # axs.set_ylim(0,0.06)
        fig.tight_layout()
        sns.despine()
        fig.savefig(os.path.join(self.figurefol, self.experiment+'_density.pdf'))
        return fly_density, fly_density_rotate
    
    def assess_returns(self):
        """
        Control for jumping plume experiment
        conditions:
        -flies return to new edge
        -flies go at least 10 mm away from old edge (30mm from new edge)
        """
        def pathlength_grid(df, lines):
            pathlengths = []
            for i in np.arange(len(lines)-1):
                l = 0
                lower = lines[i]
                upper = lines[i+1]
                df['cond'] = np.where((df['ft_posx']<upper) & (df['ft_posx']>lower), 1, 0)
                a = dict([*df.groupby(df['cond'].ne(df['cond'].shift()).cumsum())])
                for key in list(a.keys()):
                    df_temp = a[key]
                    avg_x = np.mean(df_temp.ft_posx)
                    if (avg_x>lower) and (avg_x<upper):
                        _,L = fn.path_length(df_temp.ft_posx.to_numpy(), df_temp.ft_posy.to_numpy())
                        l+=L
                pathlengths.append(l)
            return pathlengths
        
        def stop_positions(d_stop, t_thresh=0.5):
            stops = []
            for key in list(d_stop.keys()):
                df = d_stop[key]
                t = df.seconds.iloc[-1]-df.seconds.iloc[0]
                # look at stops longer than 0.5 second
                if t>t_thresh:
                    stops.append(
                        {
                            'stop_x': np.mean(df.ft_posx),
                            'stop_y': np.mean(df.ft_posy),
                            'stop_time': t
                        }
                    )
            return stops
        
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        fig, axs = plt.subplots(1,1, figsize=(2,6))
        fig2, axs2 = plt.subplots(1,1, figsize=(3,2))
        path_color = '#1bbbff'
        stop_time_color = '#080098'
        stop_num_color = '#ffb0fd'

        # load data
        all_data = self.load_trajectories()

        # storage variables
        all_animals_x = []
        all_animals_y = []
        all_anumals_speed = []
        all_pathlengths = []
        all_stops = []

        #only look at thresholds that went this far away
        thresh=-40
        lines = np.linspace(thresh,0,11)
        lines_mid = (lines[1:]+lines[:-1])/2



        for log in list(all_data.keys()):
            do = all_data[log]['do']
            di = all_data[log]['di']
            for key in list(do.keys())[1:]:
                temp = do[key]
                temp = fn.find_cutoff(temp)
                x = temp.ft_posx.to_numpy()
                y = temp.ft_posy.to_numpy()
                s = temp.speed.to_numpy()
                x0 = x[0]
                y0 = y[0]
                x = x-x0
                y = y-y0
                if np.mean(x)>0:
                    x=-x 
                if np.abs(x[-1]-x[0])<1: # flies that returned
                    max_ix = np.argmin(x)
                    if x[max_ix]<thresh: # flies that went at least 40mm
                        # max_ix = np.argwhere(x<thresh)[0][0]
                        # all_animals_x.extend(x[max_ix:])

                        stops, df = fn.find_stops(temp)

                        df['ft_posx']=x
                        df['ft_posy']=y
                        df=df.iloc[max_ix:]
                        df = df.reset_index()
                        # find the pathlengths across the grid
                        all_pathlengths.append(pathlength_grid(df, lines))
                        try:
                            _,_,d_stop = fn.dict_stops(df)
                        except:
                            return df
                        all_stops += stop_positions(d_stop)
                        if False: # plot showing return trajectories and where they start
                            axs.plot(x[max_ix:], y[max_ix:], 'k', linewidth=0.5, alpha=0.1)
                            axs.plot(x[max_ix], y[max_ix], 'o')
                        axs.plot(df.ft_posx, df.ft_posy, color='k', alpha=1, linewidth=0.1)
                        
                        
                        # plot individual stops
                        # for ks in list(d_stop.keys()):
                        #     df_temp = d_stop[ks]
                        #     stop_x = np.mean(df_temp.ft_posx)
                        #     stop_y = np.mean(df_temp.ft_posy)
                        #     axs.plot(stop_x, stop_y, '.', color='r', alpha=0.1)
        
        # return pathlengths broken into segments
        all_pathlengths = np.array(all_pathlengths)
        all_pathlengths = np.sum(all_pathlengths, axis=0)
        
        # return stops broken into segments
        df_all_stops = pd.DataFrame(all_stops)
        num_stops = df_all_stops['stop_time'].groupby(pd.cut(df_all_stops['stop_x'], lines)).count().to_numpy()
        stop_time = df_all_stops['stop_time'].groupby(pd.cut(df_all_stops['stop_x'], lines)).sum().to_numpy()
        stop_x = df_all_stops.stop_x.to_numpy()
        stop_y = df_all_stops.stop_y.to_numpy()

        # return trajectories
        for l in lines:
            axs.plot([l,l], [-300,500], 'grey', linewidth=0.5)
        axs.plot(stop_x, stop_y, '.',markersize=5, color = path_color,markeredgecolor='none', alpha=1)
        axs.plot([0,0], [-300,500],color='k',linestyle='dashed', linewidth=2)
        axs.plot([20,20], [-300,500],color='grey',linestyle='dashed', linewidth=2)
        axs.text(-10, -320, 'old edge', color='k')
        axs.set_xlim(-40,0)
        axs.set_ylabel('y postion(mm)')
        axs.set_xlabel('x postion(mm)')
        sns.despine(ax=axs)
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol,'all_return_pathlengths.pdf'))
        

        # total pathlengths + stopped time + number of stops

        for l in lines:
            axs2.plot([l,l], [np.min(all_pathlengths),np.max(all_pathlengths)], 'grey', linewidth=0.5)
        axs2.plot(lines_mid,all_pathlengths, '-o', color = path_color)
        axs2.plot([0,0], [np.min(all_pathlengths),np.max(all_pathlengths)],color='k',linestyle='dashed', linewidth=2)
        axs2.set_ylabel('sum total pathlength (mm)', color = path_color)
        axs2.set_xlabel('x position (mm)')
        axs2.text(-10, np.max(all_pathlengths)+50, 'edge', color='k')
        twin1 = axs2.twinx()
        twin2 = axs2.twinx()
        twin1.plot(lines_mid,num_stops, '-o', color = stop_num_color)
        twin2.plot(lines_mid, stop_time, '-o', color = stop_time_color)
        twin2.spines.right.set_position(("axes", 1.4))

        twin1.set_ylabel('number of stops', color = stop_num_color)
        twin2.set_ylabel('total stop time (s)', color = stop_time_color)
        sns.despine(ax=axs2)
        sns.despine(ax=twin1)
        sns.despine(ax=twin2)
        fig2.tight_layout()
        fig2.savefig(os.path.join(self.figurefol, 'total_pathlengths_return.pdf'))
        return df_all_stops

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

        fig, axs = plt.subplots(1,1,figsize=(3,3))

        for log in list(all_data.keys()):

            heading_entry = []
            heading_outside = []
            displacement_outside = []
            
            temp = all_data[log]
            df = temp['data']
            # crop trajectory
            idx_start = df.index[df.instrip==True].tolist()[0]
            idx_end = df.index[df.instrip==True].tolist()[-1]
            df = df.iloc[idx_start:idx_end]
            df['ft_posx'] = df['ft_posx']-df['ft_posx'].iloc[0]
            df['ft_posy'] = df['ft_posy']-df['ft_posy'].iloc[0]
            df['ft_posx'] = np.abs(df['ft_posx'])
            fig1, axs1 = plt.subplots(1,1)
            pl.plot_trajectory(df, axs=axs1)
            pl.plot_trajectory_odor(df, axs=axs1)

            _,_,do = fn.inside_outside(df)
            for key in list(do.keys())[:-1]:
                
                df1 = do[key]
                df2 = do[key+2]

                heading_entry.append(calc_entry_heading(df1, t0=0.5))
                heading_outside.append(calc_outside_heading(df2))
                displacement_outside.append(calc_displacement(df2))

            savebase = log.replace('.log', '')
            
            
            # fig, axs = plt.subplots(1,1)
            # all_colors = np.array(all_colors)
            # axs.plot(np.arange(len(heading_entry)), heading_entry)
            # axs.plot(np.arange(len(heading_outside)), displacement_outside)
            # num_live = len(all_colors[all_colors=='r'])
            # axs.plot([num_live+0.5, num_live+0.5], [-np.pi, np.pi], 'k')
            # axs.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
            # axs.set_ylabel('angle')
            # axs.set_xlabel('bout number')
            # fig.savefig(os.path.join(self.figurefol, savebase+'_vs_bout.pdf'))

            axs.scatter(heading_entry, heading_outside,color='k', alpha=0.2, edgecolors='none')
            axs.set_ylim(-np.pi, np.pi)
            axs.set_xlim(-np.pi, np.pi)
            axs.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
            axs.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
            axs.set_xlabel('entry heading (prev 1 sec.)')
            axs.set_ylabel('outside heading')
            fig.tight_layout()
            num_fly+=1

        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'scatter_heading_entry_outside.pdf'))

    def plot_example_three_pt_simp(self):

        """
        need to make some examples showing the three point simplification method

        """
        preprocessed = self.load_trajectories()

        i=0
        ex3 = 0
        ex2 = 0
        max_plt = 5
        fig, axs = plt.subplots(max_plt, 2, figsize = (5,5))

        # go through each fly and all its trials
        for log_name in list(preprocessed.keys()):
            temp = preprocessed[log_name]
            do = temp['do']
            # work through the outside bouts
            for bout in list(do.keys()):
                df_bout = do[bout]
                if fn.return_to_edge(df_bout): # only calculating information for bouts that return to the edge
                    x = df_bout.ft_posx.to_numpy()
                    y = df_bout.ft_posy.to_numpy()
                    x = x-x[0]
                    y = y-y[0]
                    xy0 = np.concatenate((x[:,None],y[:,None]),axis=-1)
                    simplified = fn.rdp_pts(xy0,epsilon=1, nodes=4, run_lim=1000)
                    if len(simplified)==3:
                        if ex2<max_plt:
                            axs[ex2,0].plot(x,y, pl.outside_color)
                            axs[ex2,0].plot(simplified[:,0], simplified[:,1], 'k')
                            axs[ex2,0].plot([0,0], [min(y)-2, max(y)+2], color='grey', linestyle='dashed')
                            axs[ex2,0].axis('equal')
                            ex2+=1
                    if len(simplified)==4:
                        if ex3<max_plt:
                            axs[ex3,1].plot(x,y, pl.outside_color)
                            axs[ex3,1].plot(simplified[:,0], simplified[:,1], 'k')
                            axs[ex3,1].plot([0,0], [min(y)-2, max(y)+2], color='grey', linestyle='dashed')
                            axs[ex3,1].axis('equal')
                            ex3+=1
                    if (ex3>=max_plt) & (ex2>=max_plt):
                        fig.tight_layout()
                        print(self.figurefol)
                        fig.savefig(os.path.join(self.figurefol, '2_3_pt_example.pdf'))
                        return



