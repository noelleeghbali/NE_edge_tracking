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
from numba import jit
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


class plume45():
    """
    class for comparing constant concentration and gradient plumes
    """
    def __init__(self, directory='M1', experiment=45):
        d = dr.drive_hookup()
        # your working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'
        elif directory == 'Andy':
            self.cwd = '/Volumes/Andy/GitHub/edge-tracking'

        # specify which Google sheet to pull log files from
        self.sheet_id = '15mE8k1Z9PN3_xhQH6mz1AEIyspjlfg5KPkd1aNLs9TM'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        df.drop(df[df.include=='FALSE'].index, inplace=True)
        #df[df.include=='FALSE']=False
        #df[df.include=='TRUE']=True
        self.sheet = df
        self.experiment=45
        self.angle = 45
        

        # specify pickle folder and pickle name
        self.picklefol = os.path.join(self.cwd, 'data/plume_45/pickles')
        if not os.path.exists(self.picklefol):
            os.makedirs(self.picklefol)
        self.picklesname = os.path.join(self.picklefol, 'et_manuscript_45.p')

        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/plume_45')
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
        for i, log in enumerate(self.sheet.log):
            # read in each log file
            data = fn.read_log(os.path.join(self.logfol, log))
            # if the tracking was lost, select correct segment
            data = fn.exclude_lost_tracking(data, thresh=10)
            # specificy when the fly is in the strip for old mass flow controllers
            mfcs = self.sheet.mfcs.iloc[i]
            #direction
            direction = self.sheet.direction.iloc[i]
            if mfcs == '0':
                data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
            # consolidate short in and short out periods
            data = fn.consolidate_in_out(data)
            # do not select for edge tracking trials for 45

            # append speeds to dataframe
            data = fn.calculate_speeds(data)
            # split trajectories into inside and outside component
            d, di, do = fn.inside_outside(data)

            dict_temp = {"data": data,
                        "d": d,
                        "di": di,
                        "do": do,
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

            return [length_out, length_in]

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
        for j, log_name in enumerate(self.sheet.log):
            fig, axs = plt.subplots(1,1)
            # deal with rotated plume
            direction = self.sheet.direction.iloc[j]
            angle = 45
            if direction == 'right':
                rot_angle = -angle*np.pi/180
            elif direction == 'left':
                rot_angle = +angle*np.pi/180
            
            plot=False
            temp = preprocessed[log_name]
            fly = 1 # to be filled in
            # log = temp['data']
            di = temp['di']
            do = temp['do']
            # d = temp['d']

            # calculate all inside parameters
            p = {
            'fly':fly,
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
                    x = df_bout.ft_posx.to_numpy()
                    y = df_bout.ft_posy.to_numpy()
                    x = x-x[0]
                    y = y-y[0]
                    # condition: fly must make it back to the edge. rotate trajectory to check
                    x,y = fn.coordinate_rotation(x,y,rot_angle)
                    if np.abs(x[-1]-x[0])<1:
                        if np.mean(x)>0: # align insides to the right and outsides to the left
                            x = -x
                        # rotate the trajectories back, for leftward plumes rotate in the same direction
                        if direction == 'left':
                            x,y = fn.coordinate_rotation(x, y, rot_angle)
                        elif direction == 'right':
                            x,y = fn.coordinate_rotation(x,y, -rot_angle)
                        axs.plot(x,y)

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
                        x_dist = np.min(x)
                        y_dist = y[-1]-y[0]
                        params['x_dist'].append(x_dist)
                        params['y_dist'].append(y_dist)

                        # speeds
                        xv = np.gradient(x)/del_t
                        yv = np.gradient(y)/del_t
                        params['speed'].append(np.mean(df_bout.speed))
                        params['up_vel'].append(np.mean(yv))
                        params['cross_speed'].append(np.mean(np.abs(xv)))
                        
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
                [params['out_dist'], params['in_dist']] = inbound_outbound_lengths(x_interp,y_interp, simplified)
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

    def plot_individual_trajectories(self):
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

    def heatmap(self,cmap='Blues', overwrite = True, plot_individual=False, res = 5):
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
        examples = ['08282023-182447_Canton-s_45_naive_Fly1']
        end_xy = []

        if not os.path.exists(filename) or overwrite:
            # set up arrays for heatmap
            if self.experiment == 0:
                x_bounds = np.arange(-200, 200, res)
                y_bounds = np.arange(0, 1000, res)
            if self.experiment == 45:
                x_bounds = np.arange(-50, 1000, res)
                y_bounds = np.arange(0, 1000, res)
            if self.experiment == 90:
                x_bounds = np.arange(-100, 700, res)
                y_bounds = np.arange(0, 1000, res)
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

            for i, log in enumerate(self.sheet.log):
                print(log)
                df = all_data[log]['data']
                # crop trajectories so that they start when the odor turns on
                idx_start = df.index[df.instrip==True].tolist()[0]
                idx_end = df.index[df.instrip==True].tolist()[-1]
                print(idx_start, idx_end, df.index.tolist()[-1])
                df = df.iloc[idx_start:idx_end]
                xi = df.ft_posx.to_numpy()
                yi = df.ft_posy.to_numpy()
                xi=xi-xi[0]
                yi=yi-yi[0]

                # reflect tilted and plumes so that they are facing the right way
                if self.sheet.iloc[i].direction=='left':
                    xi = -xi
                if self.experiment == 90:
                    # halfway point
                    ix1_2 = int(len(xi)/2)
                    if np.mean(xi[ix1_2:])<0:
                        xi = -xi

                # save an example trajectory to plot on top of the heatmap
                if log in examples:
                    x_ex = xi
                    y_ex = yi
                
                #end points
                end_xy.append([xi[-1], yi[-1]])

                # individual plots
                if plot_individual == True:
                    print('enter')
                    df['ft_posx'] = xi
                    df['ft_posy'] = yi
                    fig, axs = plt.subplots(1,1)
                    axs = pl.plot_trajectory(df, axs)
                    axs = pl.plot_trajectory_odor(df, axs)
                    axs.title.set_text(log)

                # rotate the tilted plume to make them vertical used to calculate density
                angle = int(self.experiment)
                rot_angle = -angle*np.pi/180
                if self.experiment == 90:
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
        
        elif os.path.exists(filename):
            print('enter')
            d = fn.load_obj(filename)
            print(d.keys())
            fly_density = d['fly_density']
            fly_density_rotate = d['fly_density_rotate']
            x_bounds = d['x_bounds']
            y_bounds = d['y_bounds']
            x_bounds_rotate = d['x_bounds_rotate']
            y_bounds_rotate = d['y_bounds_rotate']

       # heatmap figure 
        fig, axs = plt.subplots(1,1, figsize=(2,2))
        vmin = np.percentile(fly_density[fly_density>0],0)
        vmax = np.percentile(fly_density[fly_density>0],95)
        im = axs.imshow(fly_density, cmap=cmap, vmin=vmin,vmax = vmax, rasterized=True, extent=(min(x_bounds), max(x_bounds), min(y_bounds), max(y_bounds)))
        
        # plot boundaries

        axs.plot([-25,975],[0,1000], color='k', linewidth=0.5)
        axs.plot([25,1025],[0,1000], color='k', linewidth=0.5)

        
        # plot end points
        for xy in end_xy:
            xx = [xy[0]-10, xy[0]+10]
            yy = [xy[0]+10, xy[0]-10]
            axs.plot(xx,yy,color='grey', linewidth=0.5)
        # axs.plot(x_ex, y_ex, 'red', linewidth=1)#, alpha = 0.8, linewidth=0.5)
        axs.axis('off')

        # colorbar
        cb = fig.colorbar(im, shrink=0.25, drawedges=False)
        cb.outline.set_visible(False)

        # scalebar
        x1=500
        x2=600
        axs.plot([x1,x2], [300,300], 'k', linewidth=0.5)
        axs.text(x1-100,200, str(x2-x1)+'mm')

        # clean up plot
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, str(self.experiment)+'_heatmap.pdf'))

        # rotated plot
        fig, axs = plt.subplots(1,1)
        # vmin = np.percentile(fly_density_rotate[fly_density_rotate>0],10)
        # vmax = np.percentile(fly_density_rotate[fly_density_rotate>0],90)
        axs.imshow(fly_density_rotate, cmap=cmap, vmin=1,vmax = 6, extent=(min(x_bounds_rotate), max(x_bounds_rotate), min(y_bounds_rotate), max(y_bounds_rotate)))


        # boundaries
        if self.experiment == 0:
            boundary=25
        if self.experiment == 45:
            boundary = 25*np.cos(np.deg2rad(90-int(self.experiment)))
            #boundary = 25
        if self.experiment == 90:
            boundary=25
        fly_density_projection = fly_density_rotate[0:-1, :]/np.sum(fly_density_rotate[0:-1, :])
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
        fig.savefig(os.path.join(self.figurefol, str(self.experiment)+'_density.pdf'))
        return fly_density, fly_density_rotate
    
    def plot_trajectories(self, pts = 10000):
        """
        plot the average trajectories
        """
        all_data = self.load_trajectories()
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = [],[],[],[]
        for i, log in enumerate(self.sheet.log):

            # deal with rotated plume
            direction = self.sheet.direction.iloc[i]
            angle = 45
            if direction == 'right':
                rot_angle = -angle*np.pi/180
            elif direction == 'left':
                rot_angle = +angle*np.pi/180

            # create empy x,y variables and load data
            avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
            temp = all_data[log]
            df = temp['data']

            # crop trajectory
            idx_start = df.index[df.instrip==True].tolist()[0]
            idx_end = df.index[df.instrip==True].tolist()[-1]
            df = df.iloc[idx_start:idx_end]
            df_instrip = df.where(df.instrip==True)
            _,di,do = fn.inside_outside(df)

            count = 0
            for key in list(do.keys()):
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
                    x,y = fn.coordinate_rotation(x,y,rot_angle)
                    if np.abs(x[-1]-x[0])<1:
                        count+=1
                        if np.mean(x)>0: # align insides to the right and outsides to the left
                            x = -x
                        # rotate the trajectories back, for leftward plumes rotate in the same direction
                        if direction == 'left':
                            x,y = fn.coordinate_rotation(x, y, rot_angle)
                        elif direction == 'right':
                            x,y = fn.coordinate_rotation(x,y, -rot_angle)
                        t = np.arange(len(x))
                        t_common = np.linspace(t[0], t[-1], pts)
                        fx = interpolate.interp1d(t, x)
                        fy = interpolate.interp1d(t, y)
                        #axs.plot(fx(t_common), fy(t_common))
                        avg_x_out.append(fx(t_common))
                        avg_y_out.append(fy(t_common))
            for key in list(di.keys()):
                temp = di[key]
                if len(temp)>10:
                    temp = fn.find_cutoff(temp)
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()
                    x0 = x[0]
                    y0 = y[0]
                    x = x-x0
                    y = y-y0
                    x,y = fn.coordinate_rotation(x,y,rot_angle)
                    if np.abs(x[-1]-x[0])<1:
                        if np.mean(x)<0: # align insides to the right and outsides to the left
                            x = -x
                        # rotate the trajectories back, for leftward plumes rotate in the same direction
                        if direction == 'left':
                            x,y = fn.coordinate_rotation(x, y, rot_angle)
                        elif direction == 'right':
                            x,y = fn.coordinate_rotation(x,y, -rot_angle)

                        t = np.arange(len(x))
                        t_common = np.linspace(t[0], t[-1], pts)
                        fx = interpolate.interp1d(t, x)
                        fy = interpolate.interp1d(t, y)
                        #axs.plot(fx(t_common), fy(t_common))
                        avg_x_in.append(fx(t_common))
                        avg_y_in.append(fy(t_common))

            if count>6: # condition: each trajectory needs more than three outside trajectories
                print(log)
                x_traj = df.ft_posx.to_numpy()
                y_traj = df.ft_posy.to_numpy()
                x_traj_in = df_instrip.ft_posx.to_numpy()
                y_traj_in = df_instrip.ft_posy.to_numpy()
                fig, axs = plt.subplots(1,2)
                axs[0].plot(x_traj, y_traj)
                axs[0].plot(x_traj_in, y_traj_in, 'r')

                for i in np.arange(len(avg_x_out)):
                    axs[1].plot(avg_x_out[i], avg_y_out[i], 'k', alpha=0.1)
                axs[1].plot(np.mean(avg_x_out, axis=0),np.mean(avg_y_out, axis=0), color='k')
                animal_avg_x_out.append(np.mean(avg_x_out, axis=0))
                animal_avg_y_out.append(np.mean(avg_y_out, axis=0))

                for i in np.arange(len(avg_x_in)):
                    axs[1].plot(avg_x_in[i], avg_y_in[i], 'r', alpha=0.1)
                axs[1].plot(np.mean(avg_x_in, axis=0),np.mean(avg_y_in, axis=0), color='r')
                animal_avg_x_in.append(np.mean(avg_x_in, axis=0))
                animal_avg_y_in.append(np.mean(avg_y_in, axis=0))
                fig.suptitle(log)
                fig.savefig(os.path.join(self.figurefol, log.replace('.log', '.pdf')), transparent=True)


        # save the average trajectories
        fn.save_obj([animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out], os.path.join(self.picklefol, 'average_trajectories.p'))


        # make an average of the averages for ech fly
        fig, axs = plt.subplots(1,1, figsize=(2.5,2.5))

        # plot the inside trajectories with the exit point offset
        for i in np.arange(len(animal_avg_x_in)):
            axs.plot(animal_avg_x_in[i], animal_avg_y_in[i], color=pl.inside_color, alpha=0.2)
        axs.plot(np.mean(animal_avg_x_in, axis=0), np.mean(animal_avg_y_in, axis=0), pl.inside_color, linewidth=2)
        exit_x = np.mean(animal_avg_x_in, axis=0)[-1]
        exit_y = np.mean(animal_avg_y_in, axis=0)[-1]

        for i in np.arange(len(animal_avg_x_out)):
            axs.plot(animal_avg_x_out[i]+exit_x, animal_avg_y_out[i]+exit_y, pl.outside_color, alpha=0.2)
        axs.plot(np.mean(animal_avg_x_out+exit_x, axis=0), np.mean(animal_avg_y_out+exit_y, axis=0), pl.outside_color, linewidth=2)

        # draw the plume boundary line at the appropriate angle
        max_y_out = np.max(animal_avg_y_out)+exit_y
        max_y_in = np.max(animal_avg_y_in)
        max_y = np.max((max_y_out,max_y_in))

        # scale bar
        axs.plot([-5,5], [-5,-5], 'k')
        axs.text(-7, -12, '10 mm')

        # plume boundary
        axs.plot([0,max_y_out], [0,max_y_out], 'k', linewidth=0.5, linestyle='--')

        # axis parameters
        axs.axis('equal')
        axs.axis('off')

        fig.savefig(os.path.join(self.figurefol, 'all_averages.pdf'), transparent = True)
        return axs
    
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
        sns.set_style('white')

        df = df[df['io']=='out']
        params = ['angle', 'dist', 't', 's']
        param_labels = ['angle (o)', 'distance (mm)', 'time (s)', 'speed (mm/s)']

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
            pl.paired_plot(axs[i], outbound,inbound,color1=pl.inside_color, color2=pl.outside_color,
                indiv_pts=True, scatter_scale=0, alpha=1, indiv_markers=False, mean_line=False, log=False)
            
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].set_xticks([0,1])
            axs[i].set_xticklabels(labels = ['outbound', 'inbound'], rotation = 45)
            axs[i].set_ylabel(param_labels[i])
            
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, '90_inbound_outbound.pdf'))
        
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
        fig.savefig(os.path.join(self.figurefol, '45_inbound_outbound_scatter.pdf'))

    def find_outbound_inbound_metrics(self):
        """
        find the inbound and outbound metrics for all trajectories
        """
        def find_single_outbound_inbound(df, direction, returns=True):
            """
            find the inbound and outbound metrics for a single individual trajectory
            """
            from scipy import interpolate

            angle = 45
            if direction == 'right':
                rot_angle = -angle*np.pi/180
            elif direction == 'left':
                rot_angle = +angle*np.pi/180

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
                x,y = fn.coordinate_rotation(x,y,rot_angle)
                if np.mean(x)>0:
                    x=-x
                if np.abs(x[-1]-x[0])<1: # returns to edge
                    max_ix = np.argmin(x) # maximum index
                    
                    # rotate the trajectories back, for leftward plumes rotate in the same direction
                    if direction == 'left':
                        x,y = fn.coordinate_rotation(x, y, rot_angle)
                    elif direction == 'right':
                        x,y = fn.coordinate_rotation(x,y, -rot_angle)

                    xy0 = np.concatenate((x[:,None],y[:,None]),axis=-1)
                    try:
                        simplified = fn.rdp_pts(xy0, epsilon=np.max(y))
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

        for j, log in enumerate(self.sheet.log):
            # deal with rotated plume
            direction = self.sheet.direction.iloc[j]

            df = all_data[log]['data']
            do = all_data[log]['do']
            if len(do)>6:
                d = find_single_outbound_inbound(df, direction)
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
        
    def inbound_outbound_speed_paired(self, overwrite=False):
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
        fig.savefig(os.path.join(self.figurefol, '45_outbound_inbound_velocities.pdf'))

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
        fig.savefig(os.path.join(self.figurefol, '45_outbound_inbound_angles.pdf'))

    def improvement_over_time(self, plot_individual=False, plot_pts=True, set_log=True):
        # plot parameters

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
            direction = temp['direction']
            ff, aa = plt.subplots(1,1)
            for key in list(do.keys())[1:-1]:
                t = do[key].seconds.to_numpy()
                del_t = t[-1]-t[0]
                params['o_t'].append(del_t)
                x = do[key].ft_posx.to_numpy()
                y = do[key].ft_posy.to_numpy()

                # deal with the 45 degree rotated plume
                angle = 45
                if direction == 'right':
                    rot_angle = -angle*np.pi/180
                elif direction == 'left':
                    rot_angle = +angle*np.pi/180
                x,y = fn.coordinate_rotation(x,y,rot_angle)
                x = np.abs(x-x[0])

                aa.plot(x,y)
                _,dis = fn.path_length(x,y)
                dis_away = np.max(np.abs(x-x[0]))
                params['o_d'].append(dis)
                params['mean_x'].append(np.mean(x)-x[0]) # average x position
                params['o_e'].append(dis_away/dis)
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
        
        # concise version with groups
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
        if set_log:
            axs2.set_yscale('log')
        
       # plot condensed version for time
        axs3.set_xlabel('outside trajectory (n)', fontsize=6)
        axs3.set_ylabel('time (s)', color=time_color, fontsize=6)
        axs3.set_xticks([0,1,2])
        axs3.set_xticklabels(['1', '2-5', '5-10'], fontsize=6)
        axs3.set_yticks([20,100])
        axs3.set_xlim(-0.5,2.5)
        axs3.set_title('45 plume', fontsize=6)
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
        axs4.set_title('45 plume', fontsize=6)
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
        axs5.set_title('45 plume', fontsize=6)
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
        fig5.savefig(os.path.join(self.figurefol, 'improvement_efficienct_concise'+c+'_pts_'+d+'log.pdf'))

        # statistics
        pre = o_d[:,0]
        post = np.nanmean(o_d[:,1:15], axis=1)
        print(stats.ttest_rel(pre, post, nan_policy='omit'))
        return pre, post
    
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

        for i, log in enumerate(self.sheet.log):

            heading_entry = []
            heading_outside = []
            displacement_outside = []
            
            temp = all_data[log]
            df = temp['data']
            # crop trajectory
            idx_start = df.index[df.instrip==True].tolist()[0]
            idx_end = df.index[df.instrip==True].tolist()[-1]
            df = df.iloc[idx_start:idx_end]

            direction = self.sheet.direction.iloc[i]
            if direction == 'left':
                df['ft_posx'] = -df['ft_posx']

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