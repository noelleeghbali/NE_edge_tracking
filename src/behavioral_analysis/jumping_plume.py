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

sns.set(font="Arial")
sns.set(font_scale=0.6)
sns.set_style("white")
sns.set_style("ticks")

class jumping_plume():
    """

    """
    def __init__(self, directory='M1', experiment = 'jump'):
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
        self.sheet_id = '1bviAb5C9EiNtWHL3n8lMOaIJZMMmo6t1GmnzOstOTRI'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df

        self.logfol = '/Volumes/Andy/logs'
        self.logfiles = os.listdir(self.logfol)

        # ensure that every entry in sheet has a corresponding log file on drive
        self.sheet = self.match_sheet_log()

        # specify pickle folder and pickle name
        self.picklefol = os.path.join(self.cwd, 'data/jump/pickles')
        if not os.path.exists(self.picklefol):
            os.makedirs(self.picklefol)
        self.picklesname = os.path.join(self.picklefol, 'et_manuscript_jump.p')

        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/jump')
        if not os.path.exists(self.figurefol):
            os.makedirs(self.figurefol)



        d = dr.drive_hookup()
        #d.download_logs_to_local('/Volumes/Andy/logs')

    def match_sheet_log(self):
        """
        for each row in google sheet dataframe, matches datetime with a datetime
        of a log file.  updates the logfile name in the dataframe to match that
        of the .log name in drive.  prints all entries where there is not match
        for resolution.
        """
        df = self.sheet
        log_file_names = self.logfiles
        # log_file_names = []
        # for log in self.logfiles:
        #     log_file_names.append(log['name'])
        for i, log in enumerate(df.logfile):
            datetime = log.replace('-','')[0:14]
            match_found = False
            for name in log_file_names:
                if datetime == name.replace('-','')[0:14]:
                    df.iloc[i].logfile = name
                    if name == '04132022-114338_UAS-GtACR1_lights _on.log':
                        print('error')
                    match_found = True
                    break
            if not match_found:
                print('NO LOG FILE FOR ENTRY: ', df.iloc[i])
                print(datetime)
        return df

    def split_trajectories(self):
        # dict where all log files are stored
        all_data = {}

        for i, log in enumerate(self.sheet.logfile):
            # specify trial type
            trial_type = self.sheet.condition.iloc[i]
            # jump, only look at 50
            jump = self.sheet.jump.iloc[i]
            if jump == '50':
                continue
            # read in each log file
            try:
                data = fn.open_log_edit(os.path.join(self.logfol, log))
            except:
                return os.path.join(self.logfol, log)
            # if the tracking was lost, select correct segment
            data = fn.exclude_lost_tracking(data, thresh=10)
            # specificy when the fly is in the strip for old mass flow controllers
            mfcs = self.sheet.mfcs.iloc[i]
            if mfcs == '0':
                data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
            # consolidate short in and short out periods
            #data = fn.consolidate_in_out(data) # I think this becomes a problem with the jumping plume, so we will not consolidate ins and outs
            # append speeds to dataframe
            data = fn.calculate_speeds(data)
            # split trajectories into inside and outside component
            d, di, do = fn.inside_outside(data)


            if len(do)>6:
                dict_temp = {"data": data,
                            "d": d,
                            "di": di,
                            "do": do,
                            "trial_type":trial_type
                            }
                all_data[log] = dict_temp
                print('SUCCESS: ', log)
            else:
                print('FAILURE: ', log)
        # pickle everything
        fn.save_obj(all_data, self.picklesname)

    def load_trajectories(self):
        """
        open the pickled data stored from split_trajectories()
        """
        all_data = fn.load_obj(self.picklesname)
        return all_data

    def plot_individual_trajectories(self):
        """
        make a plot and save each individual trajectory
        """
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

    def improvement_over_time(self, plot_individual=False, plot_pts=True, set_log=True):

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
            for key in list(do.keys())[1:-1]:
                t = do[key].seconds.to_numpy()
                del_t = t[-1]-t[0]
                params['o_t'].append(del_t)
                x = do[key].ft_posx.to_numpy()
                y = do[key].ft_posy.to_numpy()
                x = x-x[0]
                y = y-y[0]
                if 19<np.abs(x[0]-x[-1])<21:
                    if x[-1]<x[0]:
                        x=-x
                    _,dis = fn.path_length(x,y)
                    dis_away = np.abs(np.min(x))+20 # distance to new edge (add 20)
                    params['o_d'].append(dis)
                    params['o_e'].append(dis_away/dis)
                    params['mean_x'].append(np.mean(x)-x[0]) # average x position
            all_results.append(params)
            if plot_individual:
                fig, axs = plt.subplots(1,2, figsize=(6,3))
                pl.plot_trajectory(data, axs[0])
                pl.plot_trajectory_odor(data, axs[0])
                axs[1].plot(params['mean_x'], 'o')


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
        axs4.set_xlim(-0.5,2.5)
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
        if set_log:
            axs2.set_yscale('log')
        
       # plot condensed version for time
        axs3.set_xlabel('outside trajectory (n)', fontsize=6)
        axs3.set_ylabel('time (s)', color=time_color, fontsize=6)
        axs3.set_xticks([0,1,2])
        axs3.set_xticklabels(['1', '2-5', '5-10'], fontsize=6)
        axs3.set_yticks([20,100])
        axs3.set_xlim(-0.5,2.5)
        axs3.set_title('jumping plume', fontsize=6)
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
        axs4.set_title('jumping plume', fontsize=6)
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
        axs5.set_title('jumping plume', fontsize=6)
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

        fig.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()
        fig5.tight_layout()
        if plot_pts:
            c='with'
        else:
            c='without'
        if set_log:
            d='with'
        else:
            d='without'
        fig.savefig(os.path.join(self.figurefol, 'improvement_time_'+c+'_pts_'+d+'log.pdf'))
        fig2.savefig(os.path.join(self.figurefol, 'improvement_distance_'+c+'_pts_'+d+'log.pdf'))
        fig3.savefig(os.path.join(self.figurefol, 'improvement_time_concise'+c+'_pts_'+d+'log.pdf'))
        fig4.savefig(os.path.join(self.figurefol, 'improvement_distance_concise'+c+'_pts_'+d+'log.pdf'))
        fig5.savefig(os.path.join(self.figurefol, 'improvement_efficiency_concise'+c+'_pts_'+d+'log.pdf'))

        # statistics
        pre = o_d[:,0]
        post = np.nanmean(o_d[:,1:15], axis=1)
        print(stats.ttest_rel(pre, post, nan_policy='omit'))
        return pre, post

    def interp_outside_trajectories(self, df, avg_x=[], avg_y=[], returns=True, pts=10000):

        """
        for a jumping plume log file, calculates the average trajectory.
        """
        from scipy import interpolate
        d,di,do = fn.inside_outside(df)
        for key in list(do.keys())[1:]:
            temp = do[key]
            temp = fn.find_cutoff(temp)
            x = temp.ft_posx.to_numpy()
            y = temp.ft_posy.to_numpy()
            x0 = x[0]
            y0 = y[0]
            x = x-x0
            y = y-y0
            if x0>di[key-1].ft_posx.iloc[-1]:
                x=-x
            t = np.arange(len(x))
            t_common = np.linspace(t[0], t[-1], pts)
            fx = interpolate.interp1d(t, x)
            fy = interpolate.interp1d(t, y)
            if returns:
                if 19<np.abs(x[0]-x[-1])<21:
                    avg_x.append(fx(t_common))
                    avg_y.append(fy(t_common))
            else:
                avg_x.append(fx(t_common))
                avg_y.append(fy(t_common))
        return avg_x, avg_y

    def interp_inside_trajectories(self, df, avg_x=[], avg_y=[], returns=True, pts=10000):

        """
        for a jumping plume log file, calculates the average trajectory inside the odor
        """
        from scipy import interpolate
        _,di,_ = fn.inside_outside(df)
        for key in list(di.keys())[1:]:
            temp = di[key]
            if len(temp)>3: #with this plume configuration, can get very short inside trajectories.
                x = temp.ft_posx.to_numpy()
                y = temp.ft_posy.to_numpy()
                x0 = x[0]
                y0 = y[0]
                x = x-x0
                y = y-y0
                if np.mean(x)<0:
                    x=-x
                t = np.arange(len(x))
                t_common = np.linspace(t[0], t[-1], pts)
                fx = interpolate.interp1d(t, x)
                fy = interpolate.interp1d(t, y)
                if fn.return_to_edge(temp):
                    avg_x.append(fx(t_common))
                    avg_y.append(fy(t_common))
        return avg_x, avg_y

    def find_outbound_inbound_metrics(self, df, returns=True):
        """
        for a jumping plume log file, calculate the outbound and inbound times
        """
        from scipy import interpolate
        delta_t = np.mean(np.gradient(df.seconds.to_numpy()))
        _,df = fn.find_stops(df)
        d,di,do = fn.inside_outside(df)
        tout = []
        tin = []
        sout = []
        sin = []
        xvin = []
        xvout = []
        yvin = []
        yvout = []
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
            if x0>di[key-1].ft_posx.iloc[-1]:
                x=-x
            if returns:
                if 19<np.abs(x[0]-x[-1])<21:
                    max_ix = np.argmin(x) # maximum index
                    xv = np.gradient(x)*stop/delta_t # x speed
                    yv = np.gradient(y)*stop/delta_t # y speed
                    t1 = t[max_ix]-t[0]
                    t2 = t[-1]-t[max_ix]
                    # path lengths
                    _,s1 = fn.path_length(x[0:max_ix], y[0:max_ix])
                    _,s1d = fn.path_length(np.array([x[0], x[max_ix]]), np.array([y[0], y[max_ix]])) #delta
                    _,s2 = fn.path_length(x[max_ix:], y[max_ix:])
                    _,s2d = fn.path_length(np.array([x[max_ix], x[-1]]), np.array([y[max_ix], y[-1]])) #delta
                    # speeds
                    tout.append(t1)
                    tin.append(t2)
                    sout.append(s1/t1)
                    sin.append(s2/t2)
                    # velocities
                    xvout.append(np.nanmean(xv[0:max_ix]))
                    yvout.append(np.nanmean(yv[0:max_ix]))
                    xvin.append(np.nanmean(xv[max_ix:]))
                    yvin.append(np.nanmean(yv[max_ix:]))
        tout = np.mean(tout)
        tin = np.mean(tin)
        sout = np.nanmean(sout)
        sin = np.nanmean(sin)
        xvout = np.mean(xvout)
        yvout = np.mean(yvout)
        xvin = np.mean(xvin)
        yvin = np.mean(yvin)

        return tout, tin, sout, sin, xvout, yvout, xvin, yvin

    def average_trajectory(self):
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        fig, axs = plt.subplots(1,1, figsize=(2.5,2.5))
        # axs.plot([0,0], [-20,0], 'k', linewidth=1)
        axs.plot([20,20], [-20,150], 'k', linewidth=1)
        axs.plot([0,0], [-20,150], 'k',linestyle='dotted', linewidth=1)

        all_data = self.load_trajectories()
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = [],[],[],[]

        for log in list(all_data.keys()):
            df = all_data[log]['data']
            avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
            avg_x_out, avg_y_out = self.interp_outside_trajectories(df, avg_x=avg_x_out, avg_y=avg_y_out)
            avg_x_in, avg_y_in = self.interp_inside_trajectories(df, avg_x=avg_x_in, avg_y=avg_y_in)
            for single, all in zip([avg_x_out, avg_y_out, avg_x_in, avg_y_in], [animal_avg_x_out, animal_avg_y_out, animal_avg_x_in, animal_avg_y_in]):
                all.append(np.nanmean(np.array(single), axis=0))
        # find transition point
        x_end = np.nanmean(np.array(animal_avg_x_in), axis=0)[-1]
        y_end = np.nanmean(np.array(animal_avg_y_in), axis=0)[-1]
        for x,y in zip(animal_avg_x_out, animal_avg_y_out):
            axs.plot(x+x_end,y+y_end, color=pl.outside_color, alpha=0.2)
        for x,y in zip(animal_avg_x_in, animal_avg_y_in):
            axs.plot(x,y, color=pl.inside_color, alpha=0.2)
        axs.plot(np.nanmean(np.array(animal_avg_x_out), axis=0)+x_end, np.nanmean(np.array(animal_avg_y_out), axis=0)+y_end, color=pl.outside_color, linewidth=2.5)
        axs.plot(np.nanmean(np.array(animal_avg_x_in), axis=0), np.nanmean(np.array(animal_avg_y_in), axis=0), color=pl.inside_color, linewidth=2.5)
        axs.plot([-35,-15], [-10,-10], color='k')
        axs.text(-40,-20,'10 mm')
        # axs.plot(0,0,'o',color='red', markersize=3)
        axs.arrow(0+x_end,0+y_end,10,0, color='k', head_width=3)
        axs.axis('equal')
        axs.axis('off')

        fig.savefig(os.path.join(self.figurefol, 'average_trajectory.pdf'))

    def inbound_outbound_angle_outside(self, x, y):
        """
        calculate the inbound and outbound angle relative to the edge for a given outside trajectory
        modified to work with the jumping plume geometry

        """
        def calculate_angle(vector_1, vector_2):
            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(dot_product)
            return angle

        max_ix = np.argmin(x) # point farthest away from the edge
        start_x = x[0]
        start_y = y[0]
        max_x = x[max_ix]
        max_y = y[max_ix]
        return_x = x[-1]
        return_y = y[-1]

        vec_out = [max_x-start_x, max_y-start_y]
        edge_vec_out = [0, 1]
        vec_in = [max_x-return_x, max_y-return_y]
        edge_vec_in = [0,-1]

        # fig, axs = plt.subplots(1,1)
        # axs.plot(x, y)
        # axs.axis('equal')
        # axs.plot(max_x, max_y, 'o', color='yellow')
        # axs.plot(start_x, start_y, 'o', color='green')
        # axs.plot(return_x, return_y, 'o', color='red')
        # axs.plot([start_x, max_x, return_x, start_x], [start_y, max_y, return_y, start_y], color='black')

        angle_out = np.rad2deg(calculate_angle(vec_out, edge_vec_out))
        angle_in = np.rad2deg(calculate_angle(vec_in, edge_vec_in))
        return [angle_out, angle_in]

    def inbound_outbound_angle_outside_absolute(self, x, y):
        """
        calculate the inbound and outbound angle relative to the edge for a given outside trajectory
        modified to work with the jumping plume geometry

        """
        def calculate_angle(vector_1, vector_2):
            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(dot_product)
            return angle

        max_ix = np.argmin(x) # point farthest away from the edge
        start_x = x[0]
        start_y = y[0]
        max_x = x[max_ix]
        max_y = y[max_ix]
        return_x = x[-1]
        return_y = y[-1]

        # fig, axs = plt.subplots(1,1)
        # axs.plot(x, y)
        # axs.axis('equal')
        # axs.plot(max_x, max_y, 'o', color='yellow')
        # axs.plot(start_x, start_y, 'o', color='green')
        # axs.plot(return_x, return_y, 'o', color='red')
        # axs.plot([start_x, max_x, return_x, start_x], [start_y, max_y, return_y, start_y], color='black')
        angle_out = np.array([np.arctan2(max_y-start_y, max_x-start_x)])
        angle_in = np.array([np.arctan2(return_y-max_y, return_x-max_x)])
        angle_out = fn.conv_cart_upwind(angle_out)
        angle_in = fn.conv_cart_upwind(angle_in)
        return [angle_out, angle_in]

    def inbound_outbound_angle_paired(self):
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        fig, axs = plt.subplots(1,1, figsize=(1.5,2))
        # axs.plot([0,0], [-20,0], 'k', linewidth=1)

        all_data = self.load_trajectories()
        all_animals_out = []
        all_animals_in = []

        for log in list(all_data.keys()):
            df = all_data[log]['data']
            avg_x = []
            avg_y = []
            avg_x, avg_y = self.interp_outside_trajectories(df, avg_x=avg_x, avg_y=avg_y)
            avg_x = np.nanmean(np.array(avg_x), axis=0)
            avg_y = np.nanmean(np.array(avg_y), axis=0)
            [angle_out, angle_in] = self.inbound_outbound_angle_outside(avg_x,avg_y)
            all_animals_out.append(angle_out)
            all_animals_in.append(angle_in)
        pl.paired_plot(axs, all_animals_out, all_animals_in, color1='#ADD8E6', color2='#00008B', mean_line=False, scatter_scale=0, print_stat=True)
        axs.set_ylabel('angle (\N{DEGREE SIGN})')
        axs.set_xticks([0,1])
        axs.set_xticklabels(['outbound', 'inbound'])
        axs.set_yticks([45, 90, 135])
        fig.tight_layout()
        sns.despine()
        fig.savefig(os.path.join(self.figurefol, 'jumping_outbound_inbound_angle.pdf'))
        return all_animals_out, all_animals_in

    def inbound_outbound_angle_circ_hist(self):
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        fig, axs = plt.subplots(1,1, subplot_kw={'projection':'polar'}, figsize=(2,2))
        # axs.plot([0,0], [-20,0], 'k', linewidth=1)

        all_data = self.load_trajectories()
        all_animals_out = []
        all_animals_in = []

        for log in list(all_data.keys()):
            df = all_data[log]['data']
            avg_x = []
            avg_y = []
            avg_x, avg_y = self.interp_outside_trajectories(df, avg_x=avg_x, avg_y=avg_y)
            avg_x = np.nanmean(np.array(avg_x), axis=0)
            avg_y = np.nanmean(np.array(avg_y), axis=0)
            [angle_out, angle_in] = self.inbound_outbound_angle_outside_absolute(avg_x,avg_y)
            all_animals_out.append(angle_out)
            all_animals_in.append(angle_in)
        pl.circular_hist(axs,np.array(all_animals_out), offset=np.pi/2, color='#ADD8E6')
        pl.circular_hist(axs,np.array(all_animals_in), offset=np.pi/2, color='#00008B')
        # pl.circular_hist2(axs,np.deg2rad(all_animals_out), facecolor='#ADD8E6')
        # pl.circular_hist2(axs,np.deg2rad(all_animals_in), facecolor='#00008B')
        # pl.paired_plot(axs, all_animals_out, all_animals_in, color1='#ADD8E6', color2='#00008B', mean_line=False, scatter_scale=0, print_stat=True)
        # axs.set_ylabel('angle (\N{DEGREE SIGN})')
        # axs.set_xticks([0,1])
        # axs.set_xticklabels(['outbound', 'inbound'])
        # axs.set_yticks([45, 90, 135])
        axs.tick_params(pad=-5)
        fig.tight_layout()

        fig.savefig(os.path.join(self.figurefol, 'jumping_outbound_inbound_angle_polar.pdf'))
        return all_animals_out, all_animals_in

    def inbound_outbound_time_paired(self):
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        fig, axs = plt.subplots(1,1, figsize=(1.5,2))
        # axs.plot([0,0], [-20,0], 'k', linewidth=1)

        all_data = self.load_trajectories()
        all_animals_out = []
        all_animals_in = []

        for log in list(all_data.keys()):
            df = all_data[log]['data']
            avg_x = []
            avg_y = []
            tout, tin,_,_ = self.find_outbound_inbound_time_tortuosity(df)
            all_animals_out.append(tout)
            all_animals_in.append(tin)
        pl.paired_plot(axs, all_animals_out, all_animals_in, color1='#ADD8E6', color2='#00008B', mean_line=False, scatter_scale=0, print_stat=True)
        axs.set_ylabel('time (s)')
        axs.set_xticks([0,1])
        axs.set_xticklabels(['outbound', 'inbound'])
        #axs.set_yticks([45, 90, 135])
        fig.tight_layout()
        sns.despine()
        fig.savefig(os.path.join(self.figurefol, 'jumping_outbound_inbound_time.pdf'))
        return all_animals_out, all_animals_in

    def inbound_outbound_speed_paired(self):
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        fig, axs = plt.subplots(1,1, figsize=(2,2))
        # axs.plot([0,0], [-20,0], 'k', linewidth=1)

        all_data = self.load_trajectories()
        all_animals_out_xv = []
        all_animals_in_xv = []
        all_animals_out_yv = []
        all_animals_in_yv = []

        for log in list(all_data.keys()):
            df = all_data[log]['data']
            tout, tin, sout, sin, xvout, yvout, xvin, yvin = self.find_outbound_inbound_metrics(df)
            all_animals_out_xv.append(xvout)
            all_animals_in_xv.append(xvin)
            all_animals_out_yv.append(yvout)
            all_animals_in_yv.append(yvin)
        axs.plot(all_animals_out_xv, all_animals_out_yv, 'o', color='#ADD8E6')
        axs.plot(all_animals_in_xv, all_animals_in_yv, 'o', color='#00008B')
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
        fig.savefig(os.path.join(self.figurefol, '90_outbound_inbound_velocities.pdf'))

        # fig, axs = plt.subplots(1,1)
        # axs.plot(all_animals_out, all_animals_in, 'o')

    def assess_returns(self):
        """
        do the flies perseverate at the prior plume location
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
        thresh=-20
        lines = np.linspace(thresh,20,11)
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
                if x0>di[key-1].ft_posx.iloc[-1]:
                    x=-x
                if 19<np.abs(x[0]-x[-1])<21: # flies that returned
                    max_ix = np.argmin(x)
                    if x[max_ix]<thresh: # flies that went at least 10mm away
                        # max_ix = np.argwhere(x<thresh)[0][0]
                        # all_animals_x.extend(x[max_ix:])

                        stops, df = fn.find_stops(temp)

                        df['ft_posx']=x
                        df['ft_posy']=y
                        df=df.iloc[max_ix:]
                        df = df.reset_index()
                        # find the pathlengths across the grid
                        all_pathlengths.append(pathlength_grid(df, lines))
                        _,_,d_stop = fn.dict_stops(df)
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
        axs.text(10, -320, 'new edge', color = 'grey')
        axs.set_xlim(-30,25)
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
        axs2.plot([20,20], [np.min(all_pathlengths),np.max(all_pathlengths)],color='grey',linestyle='dashed', linewidth=2)
        axs2.set_ylabel('sum total pathlength (mm)', color = path_color)
        axs2.set_xlabel('x position (mm)')
        axs2.text(-10, np.max(all_pathlengths)+50, 'old edge', color='k')
        axs2.text(10, np.max(all_pathlengths)+50, 'new edge', color = 'grey')
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



