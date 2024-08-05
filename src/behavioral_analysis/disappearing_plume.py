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
#from numba import jit
import seaborn as sns
import time
importlib.reload(dr)
importlib.reload(im)
importlib.reload(pl)
importlib.reload(fn)

class disappearing_plume():
    """
    class for comparing constant concentration and gradient plumes
    """
    def __init__(self,directory='M1'):
        d = dr.drive_hookup()
        
        # your working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'
        elif directory == 'Andy':
            self.cwd = '/Volumes/Andy/GitHub/edge-tracking'

        self.logfol = '/Volumes/Andy/logs'
        self.logfiles = os.listdir(self.logfol)

        # specify which Google sheet to pull log files from
        self.sheet_id = '1mCPLBuKWVoEPvfMR7uYBGjuG0gLyqrRprGvwzs1D0Rc'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        df.drop(df[df.include=='FALSE'].index, inplace=True)
        self.sheet = df

        # specify pickle folder and pickle name
        self.picklefol = os.path.join(self.cwd, 'data/disappearing_plume/pickles')
        if not os.path.exists(self.picklefol):
            os.makedirs(self.picklefol)
        self.picklesname = os.path.join(self.picklefol, 'et_manuscript_disappearing_plume.p')

        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/disappearing_plume')
        if not os.path.exists(self.figurefol):
            os.makedirs(self.figurefol)
    
    def split_trajectories(self):
        # dict where all log files are stored
        all_data = {}
        accept = []
        for i, log in enumerate(self.sheet.log):
            # read in each log file
            data = fn.read_log(os.path.join(self.logfol, log))

            # specificy when the fly is in the strip for old mass flow controllers
            mfcs = self.sheet.mfcs.iloc[i]
            if mfcs == '1':
                data['instrip'] = np.where(np.abs(data.mfc2_stpt)>0.01, True, False)
            elif mfcs == '0':
                data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
            
            # some of these contain replay components and need to be cropped
            if 'mode' in data.columns:
                if 'replay' in data['mode'].values:
                    ix0 = data.index[(data['mode'] == 'replay') & (data['instrip'])].to_list()[0]
                    data = data.iloc[:ix0]
            
            experiment = self.sheet.experiment.iloc[i]
            direction = self.sheet.direction.iloc[i]

            # consolidate short in and short out periods
            data = fn.consolidate_in_out(data)

            data = fn.calculate_speeds(data)
            # split trajectories into inside and outside component
            d, di, do = fn.inside_outside(data)

            dict_temp = {"data": data,
                        "d": d,
                        "di": di,
                        "do": do,
                        "experiment": experiment,
                        "direction": direction 
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

            fig, axs = plt.subplots(1,1)
            
            temp = all_data[log]
            df = temp['data']
            angle = temp['experiment']
            angle = int(angle)*np.pi/180

            # crop to the start point
            idx_start = df.index[df.instrip==True].tolist()[0]
            df = df.iloc[idx_start:]

            # center on (0,0)
            df['ft_posx'] = df['ft_posx']-df['ft_posx'].iloc[0]
            df['ft_posy'] = df['ft_posy']-df['ft_posy'].iloc[0]
            if temp['direction']=='left':
                df['ft_posx'] = -df['ft_posx']
            _,_,do = fn.inside_outside(df)
            end_key = list(do.keys())[-1]
            do_end = do[end_key]

            # look at the outside segment after the plume disappears and find where it 'crosses over the old boundary
            x_end = do_end.ft_posx.to_numpy()
            y_end = do_end.ft_posy.to_numpy()
            x_end_0 = x_end[0]
            y_end_0 = y_end[0]

            x_end, y_end = fn.coordinate_rotation(x_end-x_end_0, y_end-y_end_0, angle)
            ix = np.where(x_end<-0.1)[0]
            if len(ix)>0:
                ix = ix[0]
                do_end = do_end.iloc[ix:]
                do_end['ft_posx'] = do_end['ft_posx']-do_end['ft_posx'].iloc[0]
                do_end['ft_posy'] = do_end['ft_posy']-do_end['ft_posy'].iloc[0]
                do_end['seconds'] = do_end['seconds']-do_end['seconds'].iloc[0]
                for time in [1,5,10,30,60]:
                    # look at a bunch of different times
                    do_temp = do_end[do_end.seconds<time]
                    # end point
                    x_end = do_temp['ft_posx'].iloc[-1]
                    y_end = do_temp['ft_posy'].iloc[-1]
                    axs.plot(x_end, y_end, 'o')

            pl.plot_trajectory(df, axs)
            pl.plot_trajectory_odor(df, axs)
            pl.plot_trajectory(do_end,axs, color='g')
                

            axs.axis('equal')
            fig.suptitle(log)
            fig.savefig(os.path.join(self.figurefol, 'trajectory_'+log.replace('.log', '.pdf')), transparent=True)
    
    def average_trajectories(self):
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        all_data = self.load_trajectories()
        keys_45 = list(self.sheet[self.sheet.experiment=='45'].log)
        keys_0 = list(self.sheet[self.sheet.experiment=='0'].log)

        def constant_path(a,x,y):
            _,L = fn.path_length(a*x, a*y)
            return np.abs(1-L)

        pts=10000

        fig,axs = plt.subplots(1,2, figsize=(6,3))
        for i, keys in enumerate([keys_0, keys_45]):
            
            average_out_bout_time = []          
            for key in keys:
                # calculate the average time for outside bouts that return to the edge
                
                temp = all_data[key]
                df = temp['data']
                _,_,do = fn.inside_outside(df)
                for bout in list(do.keys()):
                    if fn.return_to_edge(do[bout]):
                        out_t = do[bout].seconds.iloc[-1]-do[bout].seconds.iloc[0]
                        average_out_bout_time.append(out_t)
                        if average_out_bout_time == np.nan:
                            return do[bout]
            print(average_out_bout_time)
            average_out_bout_time = np.nanmedian(average_out_bout_time)
            print(average_out_bout_time)

            avg_x_out = []
            avg_y_out = []
            for key in keys:
                temp = all_data[key]
                df = temp['data']
                idx_start = df.index[df.instrip==True].tolist()[0]
                df = df.iloc[idx_start:]
                df['ft_posx'] = df.ft_posx-df.ft_posx.iloc[0]
                df['ft_posy'] = df.ft_posy-df.ft_posy.iloc[0]
                if temp['direction']=='left':
                    df['ft_posx'] = -df.ft_posx

                _,di,do = fn.inside_outside(df)
                
                if len(di)>5:
                    end_key = list(do.keys())[-1]
                    do_end = do[end_key]
                    di_end = di[end_key-1]
                    
                    # crop the last outside trajectory and interolate
                    do_end['seconds'] = do_end['seconds']-do_end['seconds'].iloc[0]
                    do_end=do_end[do_end.seconds<45]
                    x = do_end.ft_posx-do_end.ft_posx.iloc[0]
                    y = do_end.ft_posy-do_end.ft_posy.iloc[0]
                    
                    _,pathl = fn.path_length(x.to_numpy(),y.to_numpy())

                    # axs[i].plot(x,y, pl.outside_color, alpha=0.5, linewidth=0.5)
                    t = np.arange(len(x))
                    t_common = np.linspace(t[0], t[-1], pts)
                    fx = interpolate.interp1d(t, x)
                    fy = interpolate.interp1d(t, y)
                    #axs.plot(fx(t_common), fy(t_common))
                    xnorm = fx(t_common)
                    ynorm = fy(t_common)
                    a = 1/pathl
                    res = minimize(constant_path, a, method='nelder-mead',
                                args=(xnorm,ynorm), options={'xatol': 0.01})
                    a = res.x
                    avg_x_out.append(a*xnorm)
                    avg_y_out.append(a*ynorm)
                    axs[i].plot(a*x,a*y, color='grey', alpha=0.5, linewidth=0.5)                  
                    # avg_x_out.append(fx(t_common))
                    # avg_y_out.append(fy(t_common))

                    # # plot the last inside trajectory
                    # x = di_end.ft_posx-di_end.ft_posx.iloc[-1]
                    # y = di_end.ft_posy-di_end.ft_posy.iloc[-1]
                    # axs[i].plot(x,y, pl.inside_color, alpha=0.5, linewidth=0.5)
                    # t = np.arange(len(x))
                    # t_common = np.linspace(t[0], t[-1], pts)
                    # fx = interpolate.interp1d(t, x)
                    # fy = interpolate.interp1d(t, y)
                    # #axs.plot(fx(t_common), fy(t_common))
                    # avg_x_in.append(fx(t_common))
                    # avg_y_in.append(fy(t_common))

            
            # axs[i].plot(np.mean(avg_x_out,axis=0),np.mean(avg_y_out,axis=0), pl.outside_color)
            # axs[i].plot(np.mean(avg_x_in,axis=0),np.mean(avg_y_in,axis=0), pl.inside_color)
            pl.colorline(axs=axs[i], x=np.mean(avg_x_out,axis=0),y=np.mean(avg_y_out,axis=0),z=t_common, segmented_cmap=False, cmap=plt.get_cmap('plasma'), norm=plt.Normalize(0,t_common.max()))
            if i==0:
                axs[i].plot([0,0], [-0.2,0.2], 'k')
                axs[i].set_ylim(-0.2, 0.2)
                axs[i].set_xlim(-0.2, 0.2)
            else:
                axs[i].plot([-0.2,0.2], [0.2,-0.2], 'k')
                axs[i].set_ylim(-0.1, 0.1)
                axs[i].set_xlim(-0.1, 0.1)

            # axs[i].axis('equal')

        fig.savefig(os.path.join(self.figurefol, 'disappearing_plumes.pdf'))

    def cartesian_displacement(self):
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        all_data = self.load_trajectories()
        
        times = [1,5,10,30,60]
        fig, axs = plt.subplots(1,len(times),figsize = (2*len(times), 2))
        
        for i,time in enumerate(times):
            x_45 = []
            y_45 = []
            x_0 = []
            y_0 = []
            for log in list(all_data.keys()):
                temp = all_data[log]
                df = temp['data'].copy()
                angle = temp['experiment']
                if angle == '0':
                    color = 'r'
                else:
                    color = 'b'
                angle_rad = int(angle)*np.pi/180

                # crop to the start point
                idx_start = df.index[df.instrip==True].tolist()[0]
                df = df.iloc[idx_start:]

                # center on (0,0)
                df.loc[:,'ft_posx'] = df['ft_posx']-df['ft_posx'].iloc[0]
                df.loc[:,'ft_posy'] = df['ft_posy']-df['ft_posy'].iloc[0]
                if temp['direction']=='left':
                    df.loc[:,'ft_posx'] = -df['ft_posx']
                _,_,do = fn.inside_outside(df)
                end_key = list(do.keys())[-1]
                do_end = do[end_key]

                # look at the outside segment after the plume disappears and find where it crosses over the old boundary
                x_end = do_end.ft_posx.to_numpy()
                y_end = do_end.ft_posy.to_numpy()
                x_end_0 = x_end[0]
                y_end_0 = y_end[0]
                x_end, y_end = fn.coordinate_rotation(x_end-x_end_0, y_end-y_end_0, angle_rad)
                ix = np.where(x_end<-0.1)[0]

                if len(ix)>0: # only consider trajectories that make it back to the edge
                    ix = ix[0]
                    # crop to make new outside segment and recenter on (0,0)
                    do_end = do_end.iloc[ix:]
                    do_end.loc[:,'ft_posx'] = do_end['ft_posx']-do_end['ft_posx'].iloc[0]
                    do_end.loc[:,'ft_posy'] = do_end['ft_posy']-do_end['ft_posy'].iloc[0]
                    do_end.loc[:,'seconds'] = do_end['seconds']-do_end['seconds'].iloc[0]
                    # look at a bunch of different times
                    do_temp = do_end[do_end.seconds<time]

                    # # end point
                    # x_end = do_temp['ft_posx'].iloc[-1]
                    # y_end = do_temp['ft_posy'].iloc[-1]

                    # center of mass
                    x_end = np.mean(do_temp['ft_posx'])
                    y_end = np.mean(do_temp['ft_posy'])

                    if angle=='0':
                        x_0.append(x_end)
                        y_0.append(y_end)
                    else:
                        x_45.append(x_end)
                        y_45.append(y_end)
                    axs[i].plot(x_end, y_end, '.', color=color, alpha=0.5)
            axs[i].plot([0, np.mean(x_0)],[0, np.mean(y_0)], color='r', linewidth=3)
            axs[i].plot([0, np.mean(x_45)],[0, np.mean(y_45)], color='b', linewidth=3)
           
        for i,a in enumerate(axs.flatten()):
            xlims = a.get_xlim()
            ylims = a.get_ylim()
            
            # print plume boundaries
            max_val = np.max([np.abs(ylims), np.abs(xlims)])
            a.plot([0,0], [-max_val, max_val], color='r', linewidth=0.5, linestyle='dashed')
            a.plot([-max_val, max_val], [max_val, -max_val], color='b', linewidth=0.5, linestyle='dashed')

            # plot scale bar
            sb_y = max_val/2
            sb_x = max_val/3
            round_x = np.ceil(sb_x)

            a.plot([sb_x, sb_x+round_x],[sb_y, sb_y], color='k')
            a.text(sb_x, sb_y+0.1*sb_y, str(int(round_x))+ ' mm')

            # set axis limits
            a.set_ylim((-max_val, max_val))
            a.set_xlim((-max_val, max_val))
            a.axis('off')

            # label each subplot with correct times
            a.title.set_text(str(times[i])+ ' s')

        fig.savefig(os.path.join(self.figurefol, 'COM_0_45_plumes_mult_time_pts.pdf'))

    def traveling_directions(self):
        def calc_entry_direction(df, t0):
            x = df.ft_posx.to_numpy()
            y = df.ft_posy.to_numpy()
            t = df['seconds'].to_numpy()
            t = t-t[0]
            tmax = t[-1]
            if tmax>t0:
                ix = np.argmin(np.abs(t-(tmax-t0)))
                x = x[ix:]
                y = y[ix:]
            delx = x[-1]-x[0]
            dely = y[-1]-y[0]
            direction = np.arctan2(dely,delx)
            return direction
        
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        all_data = self.load_trajectories()
        
        times = [1,5,10,30,60]
        fig, axs = plt.subplots(1,len(times)+1,figsize = (2*(len(times)+1), 2), subplot_kw={'projection': 'polar'})
        for i,a in enumerate(axs.flatten()):
            a.plot(np.linspace(0,2*np.pi, num=50), np.ones(50), 'k', linewidth=0.5)

        all_entry_directions_0 = []
        all_entry_directions_45 = [] 
        
        for i,time in enumerate(times):
            angles_45 = []
            angles_0 = []
            for log in list(all_data.keys()):
                temp = all_data[log]
                df = temp['data'].copy()
                angle = temp['experiment']
                if angle == '0':
                    color = 'r'
                else:
                    color = 'b'
                angle_rad = int(angle)*np.pi/180

                # crop to the start point
                idx_start = df.index[df.instrip==True].tolist()[0]
                df = df.iloc[idx_start:]

                # center on (0,0)
                df.loc[:,'ft_posx'] = df['ft_posx']-df['ft_posx'].iloc[0]
                df.loc[:,'ft_posy'] = df['ft_posy']-df['ft_posy'].iloc[0]
                if temp['direction']=='left':
                    df.loc[:,'ft_posx'] = -df['ft_posx']
                _,di,do = fn.inside_outside(df)
                end_key = list(do.keys())[-1]
                do_end = do[end_key]

                # look at the outside segment after the plume disappears and find where it crosses over the old boundary
                x_end = do_end.ft_posx.to_numpy()
                y_end = do_end.ft_posy.to_numpy()
                x_end_0 = x_end[0]
                y_end_0 = y_end[0]
                x_end, y_end = fn.coordinate_rotation(x_end-x_end_0, y_end-y_end_0, angle_rad)
                ix = np.where(x_end<-0.1)[0]

                if len(ix)>0: # only consider trajectories that make it back to the edge
                    ix = ix[0]
                    # crop to make new outside segment and recenter on (0,0)
                    do_end = do_end.iloc[ix:]
                    do_end.loc[:,'ft_posx'] = do_end['ft_posx']-do_end['ft_posx'].iloc[0]
                    do_end.loc[:,'ft_posy'] = do_end['ft_posy']-do_end['ft_posy'].iloc[0]
                    do_end.loc[:,'seconds'] = do_end['seconds']-do_end['seconds'].iloc[0]
                    # look at a bunch of different times
                    do_temp = do_end[do_end.seconds<time]

                    # # end point
                    # x_end = do_temp['ft_posx'].iloc[-1]
                    # y_end = do_temp['ft_posy'].iloc[-1]

                    # center of mass
                    x_end = np.mean(do_temp['ft_posx'])
                    y_end = np.mean(do_temp['ft_posy'])
                    ang_temp = np.arctan2(y_end, x_end)

                    # calculate traveling direction
                    if angle=='0':
                        angles_0.append(ang_temp)
                    else:
                        angles_45.append(ang_temp)
                    axs[i+1].plot(ang_temp, 1, '.', color=color)
                    axs[i+1].plot([ang_temp,ang_temp], [0,1], alpha=0.5, linewidth=0.5, color=color)

                    # calculate the entry direction once
                    if time==1:
                        entry_direction = []
                        for key in list(do.keys())[:-1]:
                            do_t = do[key]
                            entry_direction.append(calc_entry_direction(do_t, t0=1))
                        avg_dir = stats.circmean(entry_direction)
                        axs[0].plot(avg_dir, 1, '.', color=color)
                        axs[0].plot([avg_dir,avg_dir], [0,1], alpha=0.5, linewidth=0.5, color=color)
                        if angle == '0':
                            all_entry_directions_0.append(avg_dir)
                        else:
                            all_entry_directions_45.append(avg_dir)
                        
            axs[i+1].plot([stats.circmean(angles_0), stats.circmean(angles_0)],[0, 1], color='r', linewidth=3)
            axs[i+1].plot([stats.circmean(angles_45), stats.circmean(angles_45)],[0, 1], color='b', linewidth=3)
        
        # plot the averges for the entry headings
        axs[0].plot([stats.circmean(all_entry_directions_0), stats.circmean(all_entry_directions_0)],[0, 1], color='r', linewidth=3)
        axs[0].plot([stats.circmean(all_entry_directions_45), stats.circmean(all_entry_directions_45)],[0, 1], color='b', linewidth=3)
           
        for i,a in enumerate(axs.flatten()):
            a.axis('off')


        fig.savefig(os.path.join(self.figurefol, 'trav_direc_0_45_plumes_mult_time_pts.pdf'))

    def plot_example_trajectories(self):
        """
        plot example trajectories of one vertical and one 45 degree disappearing plume for figure
        """
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        all_data = self.load_trajectories()
        for log in list(all_data.keys()):
            print(log)
            if log == '04072023-150529_45degree_replay_Fly6_2minbaseline.log' or log == '05162023-142916_disappearing_plume_vertical_Canton-s_Fly3.log':

                fig, axs = plt.subplots(1,1, figsize=(2,2))
                
                temp = all_data[log]
                df = temp['data']
                angle = temp['experiment']
                angle = int(angle)*np.pi/180

                # crop to the start point
                idx_start = df.index[df.instrip==True].tolist()[0]
                df = df.iloc[idx_start:]

                # center on (0,0)
                df.loc[:,'ft_posx'] = df['ft_posx']-df['ft_posx'].iloc[0]
                df.loc[:,'ft_posy'] = df['ft_posy']-df['ft_posy'].iloc[0]
                if temp['direction']=='left':
                    df.loc[:,'ft_posx'] = -df['ft_posx']
                _,_,do = fn.inside_outside(df)
                end_key = list(do.keys())[-1]
                do_end = do[end_key]

                # look at the outside segment after the plume disappears and find where it 'crosses over the old boundary
                x_end = do_end.ft_posx.to_numpy()
                y_end = do_end.ft_posy.to_numpy()
                x_end_0 = x_end[0]
                y_end_0 = y_end[0]

                x_end, y_end = fn.coordinate_rotation(x_end-x_end_0, y_end-y_end_0, angle)
                ix = np.where(x_end<-0.1)[0]
                if len(ix)>0:
                    ix = ix[0]
                    do_end = do_end.iloc[ix:]
                    do_end.loc[:,'seconds'] = do_end['seconds']-do_end['seconds'].iloc[0]
                    for time in [1,10,60]:
                        # look at a bunch of different times
                        do_temp = do_end[do_end.seconds<time]
                        # end point
                        x_end = do_temp['ft_posx'].iloc[-1]
                        y_end = do_temp['ft_posy'].iloc[-1]
                        axs.plot(x_end, y_end, 'o', color='k')

                pl.plot_trajectory(df, axs)
                pl.plot_trajectory_odor(df, axs)
                pl.plot_trajectory(do_end,axs, color='g')
                axs.plot([-50,50], [-50,-50], 'k')
                axs.text(-50,-50, '100 mm')

                axs.axis('equal')
                axs.axis('off')
                # fig.suptitle(log)
                fig.savefig(os.path.join(self.figurefol, 'example_trajectory_'+log.replace('.log', '.pdf')), transparent=True)

            

            





    
