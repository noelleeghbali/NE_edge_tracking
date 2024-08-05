import pandas as pd
import importlib
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pdb
from src.drive import drive as dr
from src.utilities import funcs as fn
from src.utilities import plotting as pl
from src.utilities import imaging as im
from scipy import interpolate, stats
from scipy.stats import sem
#from numba import jit
import seaborn as sns
import time
import random
importlib.reload(dr)
importlib.reload(im)
importlib.reload(pl)
importlib.reload(fn)


class average_trajectory():
    """
    1) specify experiment
    2) split up log files based on inside/outside
    3) pickle results
    """
    def __init__(self, directory='M1', experiment = 'T', download_logs = False):

        # connect to the google drive
        d = dr.drive_hookup()

        # specify working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'
        elif directory == 'Andy':
            self.cwd = '/Volumes/Andy/GitHub/edge-tracking'
            self.logfol = '/Volumes/Andy/logs'
        self.experiment = experiment

        # for experiment specified, pull log file names from corresponding sheet
        if experiment == '90':
            self.sheet_id = '14r0TgRUhohZtw2GQgirUseBWXK8NPbyqPzPvAtND7Gs'
        elif experiment == '45':
            self.sheet_id = '15mE8k1Z9PN3_xhQH6mz1AEIyspjlfg5KPkd1aNLs9TM'
        elif experiment == '15':
            self.sheet_id = '1qCrV96jUo24lpZ7-k2-B9RWG5RSQDSgnSn-sFgjS7ys'
        elif experiment == '0':
            self.sheet_id = '1K_SkaT3JUA2Ik8uiwB6kJMwHnd935bZvZB4If0zh8rY'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df

        # download new log folders
        self.logfol = '/Volumes/Andy/logs'
        d = dr.drive_hookup()
        # d.download_logs_to_local('/Volumes/Andy/logs')

        # specify pickle folder and pickle name
        self.picklefol = os.path.join(self.cwd, 'data/trajectory_analysis', self.experiment, 'pickles')
        if not os.path.exists(self.picklefol):
            os.makedirs(self.picklefol)
        self.picklesname = os.path.join(self.picklefol, self.experiment+'_'+'plume.p')

        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/average_trajectories', self.experiment)
        if not os.path.exists(self.figurefol):
            os.makedirs(self.figurefol)

    def split_trajectories(self):
        """
        For a given experiment (0,45,90 degree plume), read in all log files,
        split them into inside and outside trajectories, and pickle the results.
        """

        # dict where all log files are stored
        all_data = {}

        for i, log in enumerate(self.sheet.log):
            # read in each log file
            data = fn.read_log(os.path.join(self.logfol, log))
            # if the tracking was lost, select correct segment
            data = fn.exclude_lost_tracking(data, thresh=10)
            # specificy when the fly is in the strip for old mass flow controllers
            mfc = self.sheet.mfc.iloc[i]
            if mfc == 'old':
                data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
            # consolidate short in and short out periods
            data = fn.consolidate_in_out(data)
            # append speeds to dataframe
            data = fn.calculate_speeds(data)
            # split trajectories into inside and outside components
            d, di, do = fn.inside_outside(data)
            # find direction
            direction = self.sheet.iloc[i].direction

            dict_temp = {"data": data,
                        "d": d,
                        "di": di,
                        "do": do,
                        "direction": direction}
            all_data[log] = dict_temp
        # pickle everything
        fn.save_obj(all_data, self.picklesname)

    def load_single_trajectory(self, log):
        data = fn.read_log(os.path.join(self.logfol, log))
        ix = self.sheet[self.sheet.log==log].index.to_numpy()

        mfc = self.sheet['mfc'][ix[0]]
        # data['instrip'] = np.where(np.abs(data.ft_posx)<25, True, False)
        data = fn.exclude_lost_tracking(data, thresh=10)
        if mfc == 'old': # all experiments performed with old MFCs
            data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
        return data

    def load_trajectories(self):
        """
        open the pickled data stored from split_trajectories()
        """
        all_data = fn.load_obj(self.picklesname)
        return all_data

    def find_rotation_angle(self, d):
        """
        find angle for rotation based on the experiment
        d: dict for each trial created in split_trajectories()
        """
        direction = d['direction']
        angle = int(self.experiment)
        angle_from_horizontal = np.pi/2 - angle*np.pi/180
        if direction == 'right':
            rot_angle = -angle*np.pi/180
        elif direction == 'left':
            rot_angle = +angle*np.pi/180
        elif direction == 'NA':
            rot_angle = -angle*np.pi/180
        return rot_angle

    def consecutive_outside_trajectories(self, flip=True):
        """
        calculate how many consecutive outside trajectories
        """
        all_data = self.load_trajectories()
        num_cons=[]
        for key in list(all_data.keys()):
            rot_angle = self.find_rotation_angle(all_data[key])
            # select outside trajectories
            do = all_data[key]['do']
            data = all_data[key]['data']

            cons = 0
            for key_o in list(do.keys()):
                df = do[key_o]
                # center all outside trajecories
                x0 = df['ft_posx']-df['ft_posx'].iloc[0]
                y0 = df['ft_posy']-df['ft_posy'].iloc[0]

                # appropriately rotate the trajectory so that it is vertical
                x,y = fn.coordinate_rotation(x0.to_numpy(),y0.to_numpy(),rot_angle)

                # only include outside segments where fly returns to the edge
                # return to the edge is defined as final x position being within 1mm of starting x position
                if np.abs(x[-1]-x[0])<1: # fly makes it back
                    cons+=1
            num_cons.append(cons)
            if False:
                fig, axs = plt.subplots(1,1)
                pl.plot_trajectory(data, axs)
                pl.plot_trajectory_odor(data, axs)
                axs.text(0,0,str(cons))
        return num_cons

    def create_outside_dp_segments(self, flip=True):
        """
        load outside trajectories, and run DP algorithm to simplify them.  Store
        results in DF and pickle
        """
        # load all trajectories
        all_data = self.load_trajectories()

        # iterate through all trial
        d = []
        exit_heading = []
        entry_heading = []
        trajectories = {}
        num_plot=0
        for key in list(all_data.keys()):

            # find angle for rotation
            # direction = all_data[key]['direction']
            # angle = int(self.experiment)
            # angle_from_horizontal = np.pi/2 - angle*np.pi/180
            # if direction == 'right':
            #     rot_angle = -angle*np.pi/180
            # elif direction == 'left':
            #     rot_angle = +angle*np.pi/180
            # elif direction == 'NA':
            #     rot_angle = -angle*np.pi/180
            rot_angle = self.find_rotation_angle(all_data[key])

            # select outside trajectories
            do = all_data[key]['do']
            for key_o in list(do.keys()):
                df = do[key_o]

                # center all outside trajecories
                x0 = df['ft_posx']-df['ft_posx'].iloc[0]
                y0 = df['ft_posy']-df['ft_posy'].iloc[0]

                # appropriately rotate the trajectory so that it is vertical
                x,y = fn.coordinate_rotation(x0.to_numpy(),y0.to_numpy(),rot_angle)

                # flip the outside trajectories so they are all on the left side
                if flip:
                    x = -np.abs(x)
                else:
                    x = (2*random.randint(0,1)-1)*x

                # only include outside segments where fly returns to the edge
                # return to the edge is defined as final x position being within 1mm of starting x position
                if np.abs(x[-1]-x[0])>1:
                    continue

                # rotate the trajectories back, for leftward plumes rotate in the same direction
                if direction == 'left':
                    x,y = fn.coordinate_rotation(x, y, rot_angle)
                elif direction == 'right':
                    x,y = fn.coordinate_rotation(x,y, -rot_angle)
                else:
                    x,y = fn.coordinate_rotation(x,y, -rot_angle)

                # break up trajectories using rdp algorithm
                simplified, heading, angles, L = fn.rdp_simp_heading_angles_len(x,y)

                # ignore outside segments composed of a single line
                if len(angles)==0:
                    continue

                # dictionary with turn angles and run lengths
                d.append({
                    'lengths': L[0:-1],
                    'angles': angles,
                    'heading': heading[0:-1]
                })

                # distribution of exit and entry headings
                exit_heading.append(heading[0])
                entry_heading.append(heading[-1])

                # coordinates of simplified trajectories
                trajectories[str(key)+'_'+str(key_o)] = simplified

        df = pd.DataFrame(d)
        df = df.explode(['lengths', 'angles', 'heading'])

        results = {
            'df':df,
            'exit_heading':exit_heading,
            'entry_heading':entry_heading,
            'trajectories':trajectories
        }

        # what does the heading look like
        #sns.scatterplot(np.rad2deg(fn.conv_cart_upwind(df.heading.to_numpy(dtype='float64'))), df.lengths, alpha=0.1)
        heading = fn.conv_cart_upwind(np.rad2deg(df.heading.to_numpy(dtype='float64')))
        lengths = df.lengths.to_numpy()
        g = sns.JointGrid(x=heading[0:-1], y=lengths[1:])
        g.plot_joint(sns.kdeplot,
                     fill=True, clip=((-180, 180), (-180, 180)), levels=25,cmap="rocket")

        # save results
        savename = os.path.join(self.picklefol, self.experiment+'_'+'outside_DP_segments.p')
        fn.save_obj(results, savename)
        return results

    def load_outside_dp_segments(self):
        """
        load the outside DP segments pickled in create_outside_dp_segments
        """
        savename = os.path.join(self.picklefol, self.experiment+'_'+'outside_DP_segments.p')
        results = fn.load_obj(savename)
        df = results['df']
        exit_heading = results['exit_heading']
        entry_heading = results['entry_heading']
        trajectories = results['trajectories']
        return df, exit_heading, entry_heading, trajectories

    def orthogonal_distance(self, x,y,plume_angle):
        theta = np.arctan2(x,y)
        alpha = fn.wrap(np.array([plume_angle-theta]))
        c = np.sqrt(x**2+y**2)
        d = c*np.sin(alpha)
        return d

    def make_random_segment_library(self, a = 0.5, min_dist=1, color='r'):
        """
        make fictive trajectories from DP segments
        a is upwind parameter
        method = sin or sample
        """
        def next_point(r,theta, xy):
            """
            given current x,y position, heading and run length,
            calculate the next x,y position.
            """
            x = xy[0]+r*np.sin(theta)
            y = xy[1]+r*np.cos(theta)
            return np.array([x,y])

        def biased_turn(angle, a):
            """
            a is the upwind parameter
            a=0 animal turns randomly left or right
            a=1 animal always turns to the upwind direction.
            angle is the angle in the upwind reference frame.
            Use fn.conv_cart_upwind() for conversion into upwind coordinates
            """
            prob = random.random()
            thresh = 0.5+a*0.5*np.sin(angle)
            if prob>thresh:
                mult=1
            else:
                mult=-1
            return mult

        # convert plume direction to cartesian coordinates
        plume_angle = np.deg2rad(int(self.experiment))
        # plume_angle = np.array([plume_angle])
        # plume_angle = fn.conv_cart_upwind(plume_angle)
        # ang1 = plume_angle
        # ang2 = fn.wrap(plume_angle + np.pi)
        # print('plume angle is '+ str(plume_angle)+' ang1 is '+str(ang1)+' ang2 is '+str(ang2))

        # success metrics
        failures = 0

        # load DP segments
        df, exit_heading, entry_heading, trajectories = self.load_outside_dp_segments()

        # convert headings from wind-centered to wind-centered coordinates
        exit_heading = -np.abs(fn.conv_cart_upwind(np.array(exit_heading)))
        entry_heading = fn.conv_cart_upwind(np.array(entry_heading))
        df['angles'] = fn.conv_cart_upwind(df.angles.to_numpy())

        # set number of trajectories equal to the number of real trajectories
        # n_traj = len(exit_heading)
        n_traj = len(trajectories)

        # figure to plot efficiency
        fig, axs = plt.subplots(1,1)

        # save trajectories in a dictionary
        d_traj={}
        orthogonal_distances = []
        pathlengths = []

        # create trajectories
        n=0
        while n<n_traj:
            xy = np.array([[0,0]]) # starting point
            heading = random.choice(exit_heading) # select initial heading

            # fly needs to go a minimum orthogonal distance away from the edge
            min_len = min_dist/np.cos(plume_angle-heading)
            #min_len = min_len[0]
            if len(df[df.lengths>min_len])<1:
                continue
            df_init = df[df.lengths>min_len]
            L = df_init.sample().lengths.iloc[0]

            # make the first point
            new = next_point(L,heading,xy[-1])
            xy = np.vstack((xy,new))

            # while animal is outside plume, orthogonal distance is greater than 0
            while self.orthogonal_distance(xy[-1][0], xy[-1][1], plume_angle)>0:
                L = df.sample().lengths.iloc[0]
                new = next_point(L,heading,xy[-1])
                xy = np.vstack((xy,new))

                # distance cutoff -- if fly goes more than 200 away from edge, terminate trajectory
                if self.orthogonal_distance(xy[-1,0], xy[-1,1],plume_angle)>200:
                    print('enter')
                    failures+=1
                    break
                # mult = 2*random.randint(0, 1)-1
                mult = biased_turn(heading, a)
                #print(heading, mult)
                heading = heading + mult*np.abs(df.sample().angles.iloc[0])


            if self.orthogonal_distance(xy[-1,0], xy[-1,1],plume_angle)<0: #save trajectories that make it back
                # plot dist away and pathlength
                dist_away = self.orthogonal_distance(xy[:,0], xy[:,1],plume_angle)
                dist_away = np.max(dist_away)
                _, pathlen = fn.path_length(xy[:,0], xy[:,1])
                axs.plot(dist_away, pathlen, 'o', color=color)

                # save efficiency
                orthogonal_distances.append(dist_away)
                pathlengths.append(pathlen)

                if True:
                    # correct the final point so that it ends on the border and doesn't overshoot it
                    p1 = [xy[-2, 0], xy[-2, 1]]
                    p2 = [xy[-1, 0], xy[-1, 1]]
                    p3 = [-1E06*np.sin(plume_angle), -1E06*np.cos(plume_angle)]
                    p4 = [1E06*np.sin(plume_angle), 1E06*np.cos(plume_angle)]
                    xy[-1] = fn.line_intersection(p1,p2,p3,p4)
                    # coefficients = np.polyfit(x, y, 1)
                    # polynomial = np.poly1d(coefficients)
                    # x_end = 0
                    # y_end = polynomial[x_end]
                    # xy[-1] = [x_end, y_end]


                # save traj
                d_traj[n] = xy
                n+=1

        print('failures = ' + str(failures)+ '. success = '+str(n/(n+failures)))

        # make a plot of some real examples and some random trajectories
        num_examples = 10
        fig, axs = plt.subplots(2, num_examples)
        # real trajectories
        for i, key in enumerate(random.choices(list(trajectories.keys()), k=num_examples)):
            traj = trajectories[key]
            x = traj[:,0]
            y = traj[:,1]
            axs[0,i].plot(x, y,'green')
            axs[0,i].axis('equal')
            axs[0,i].axis('off')

        # fictive trajectories
        for i, key in enumerate(random.choices(list(d_traj.keys()), k=num_examples)):
            traj = d_traj[key]
            x = traj[:,0]
            y = traj[:,1]
            axs[1,i].plot(x, y, color=color)
            axs[1,i].axis('equal')
            axs[1,i].axis('off')
        fig.savefig(os.path.join(self.figurefol+'_'+'comparison'+'_'+str(a)+'.pdf'))

        # save the trajectories
        results = {
            'n_traj': n_traj,
            'failures': failures,
            'orthogonal_distances': orthogonal_distances,
            'pathlengths': pathlengths,
            'd_traj': d_traj
        }

        savename = os.path.join(self.picklefol, self.experiment+'_'+'outside_fictive_trajectories'+'_'+str(a)+'.p')
        fn.save_obj(results, savename)

    def make_random_segment_library_sample(self, delta = 0.1, min_dist=1, color='r'):
        """
        make fictive trajectories from DP segments
        a is upwind parameter
        method = sin or sample
        """
        def next_point(r,theta, xy):
            """
            given current x,y position, heading and run length,
            calculate the next x,y position.
            """
            x = xy[0]+r*np.sin(theta)
            y = xy[1]+r*np.cos(theta)
            return np.array([x,y])

        def sample_heading(df, heading, delta):
            """
            sample joint heading-turn distrubtion at a heading +/- delta
            """
            a1 = np.array([heading+delta])
            a2 = np.array([heading-delta])
            if a1>np.pi or a2<-np.pi:
                a1=fn.wrap(a1)
                a2=fn.wrap(a2)
                a1,a2=a2,a1
            a1=a1[0]
            a2=a2[0]
            df_sample = df[np.logical_and(df.heading<a1, df.heading>a2)]
            return df_sample

        # convert plume direction to cartesian coordinates
        plume_angle = np.deg2rad(int(self.experiment))

        # success metrics
        failures = 0

        # load DP segments
        df, exit_heading, entry_heading, trajectories = self.load_outside_dp_segments()

        # use vertical plume library for 0, 45, and 90 degree plumes
        savename = os.path.join(self.cwd, 'data/trajectory_analysis', '0', 'pickles','0_outside_DP_segments.p')
        results = fn.load_obj(savename)
        df = results['df']

        # convert headings from wind-centered to wind-centered coordinates
        exit_heading = -np.abs(fn.conv_cart_upwind(np.array(exit_heading)))
        entry_heading = fn.conv_cart_upwind(np.array(entry_heading))
        #df['angles'] = fn.conv_cart_upwind(df.angles.to_numpy())
        df['heading'] = fn.conv_cart_upwind(df.heading.to_numpy())

        # set number of trajectories equal to the number of real trajectories
        # n_traj = len(exit_heading)
        n_traj = 200

        # figure to plot efficiency
        fig, axs = plt.subplots(1,1)

        # save trajectories in a dictionary
        d_traj={}
        orthogonal_distances = []
        pathlengths = []

        # create trajectories
        n=0
        while n<n_traj:
            xy = np.array([[0,0]]) # starting point
            heading = random.choice(exit_heading) # select initial heading
            # fly needs to go a minimum orthogonal distance away from the edge
            min_len = min_dist/np.cos(plume_angle-heading)
            #min_len = min_len[0]
            if len(df[df.lengths>min_len])<1:
                continue
            df_init = df[df.lengths>min_len]
            L = df_init.sample().lengths.iloc[0]

            # make the first point
            new = next_point(L,heading,xy[-1])
            xy = np.vstack((xy,new))

            # while animal is outside plume, orthogonal distance is greater than 0
            while self.orthogonal_distance(xy[-1][0], xy[-1][1], plume_angle)>0:
                # select run length randomly
                L = df.sample().lengths.iloc[0]

                # select the turn based on current heading
                df_sample = sample_heading(df, heading, delta)
                heading = heading + df_sample.sample().angles.iloc[0]
                heading = fn.wrap(np.array([heading]))[0]

                # find the next point
                new = next_point(L,heading,xy[-1])
                xy = np.vstack((xy,new))

                # distance cutoff -- if fly goes more than 200 away from edge, terminate trajectory
                if self.orthogonal_distance(xy[-1,0], xy[-1,1],plume_angle)>200:
                    print('enter')
                    failures+=1
                    break


            if self.orthogonal_distance(xy[-1,0], xy[-1,1],plume_angle)<0: #save trajectories that make it back
                # plot dist away and pathlength
                dist_away = self.orthogonal_distance(xy[:,0], xy[:,1],plume_angle)
                dist_away = np.max(dist_away)
                _, pathlen = fn.path_length(xy[:,0], xy[:,1])
                axs.plot(dist_away, pathlen, 'o', color=color)

                # save efficiency
                orthogonal_distances.append(dist_away)
                pathlengths.append(pathlen)

                if True:
                    # correct the final point so that it ends on the border and doesn't overshoot it
                    p1 = [xy[-2, 0], xy[-2, 1]]
                    p2 = [xy[-1, 0], xy[-1, 1]]
                    p3 = [-1E06*np.sin(plume_angle), -1E06*np.cos(plume_angle)]
                    p4 = [1E06*np.sin(plume_angle), 1E06*np.cos(plume_angle)]
                    xy[-1] = fn.line_intersection(p1,p2,p3,p4)
                    # coefficients = np.polyfit(x, y, 1)
                    # polynomial = np.poly1d(coefficients)
                    # x_end = 0
                    # y_end = polynomial[x_end]
                    # xy[-1] = [x_end, y_end]


                # save traj
                d_traj[n] = xy
                n+=1

        print('failures = ' + str(failures)+ '. success = '+str(n/(n+failures)))

        # make a plot of some real examples and some random trajectories
        num_examples = 10
        fig, axs = plt.subplots(2, num_examples)
        # real trajectories
        for i, key in enumerate(random.choices(list(trajectories.keys()), k=num_examples)):
            traj = trajectories[key]
            x = traj[:,0]
            y = traj[:,1]
            axs[0,i].plot(x, y,'green')
            axs[0,i].axis('equal')
            axs[0,i].axis('off')

        # fictive trajectories
        for i, key in enumerate(random.choices(list(d_traj.keys()), k=num_examples)):
            traj = d_traj[key]
            x = traj[:,0]
            y = traj[:,1]
            axs[1,i].plot(x[0], y[0],'o', color=color)
            axs[1,i].plot(x, y, color=color)
            axs[1,i].axis('equal')
            axs[1,i].axis('off')
        fig.savefig(os.path.join(self.figurefol+'_'+'comparison'+'_'+str(a)+'.pdf'))

        # save the trajectories
        results = {
            'n_traj': n_traj,
            'failures': failures,
            'orthogonal_distances': orthogonal_distances,
            'pathlengths': pathlengths,
            'd_traj': d_traj
        }

        savename = os.path.join(self.picklefol, self.experiment+'_'+'outside_fictive_trajectories_sample'+'_'+str(delta)+'.p')
        fn.save_obj(results, savename)

    def fictive_traj_upwind_bias(self):
        """
        create fictive trajectories with different upwind biases using the sine
        method for upwind bias
        """
        upwind_a = np.round(np.linspace(0,1,11),1)
        colors = sns.color_palette("rocket",len(upwind_a))
        for i,a in enumerate(upwind_a):
            color = colors[i]
            self.make_random_segment_library(a=a, min_dist=1,color=color)

    def make_biased_turn_figure(self):
        """
        figure for making the sine-based biased turn algorithm
        """
        fig, axs = plt.subplots(1,1)
        upwind_a = np.round(np.linspace(0,1,11),1)
        colors = sns.color_palette("rocket",len(upwind_a))
        x = np.linspace(-180,180,100)
        for i,a in enumerate(upwind_a):
            y = 0.5+a*0.5*np.sin(np.deg2rad(x))
            axs.plot(x,y, color=colors[i])
        axs.set_yticks([0,0.5,1])
        axs.set_xticks([-180,-90,0,90,180])
        axs.set_xlabel('current heading')
        axs.set_ylabel('P(left turn)')
        fig.savefig(os.path.join(self.figurefol, 'biased_turn_figure_cartoon.pdf'))

    def fictive_traj_upwind_bias_sample(self):
        """
        create fictive trajectories with different upwind biases using the
        sampling method for upwind bias
        """
        delta = [0.05, 0.1, 0.2, 0.5, 1, 2]
        colors = sns.color_palette("rocket",len(delta))
        for i,a in enumerate(delta):
            color = colors[i]
            self.make_random_segment_library_sample(delta=a, min_dist=1,color=color)

    def plot_dist_turns(self):
        """
        make a plot of the distribution of turn angles
        """
        fig, axs = plt.subplots(1,1)
        df, exit_heading, entry_heading, trajectories = self.load_outside_dp_segments()
        angles = np.rad2deg(df.angles.to_list())
        sns.histplot(angles, ax = axs, element = 'step')
        axs.set_xlabel('turn angle (degrees)')
        axs.set_xticks([-90,0,90])
        fig.savefig(os.path.join(self.figurefol, 'turn_angle_distribution.pdf'))

    def compare_fictive_real_efficiency(self, axs=None, axs1=None):
        """
        load fictive trajectories and compare them to the DP simplification of real trajectories
        """
        # figure for efficiency
        if axs is None:
            fig, axs = plt.subplots(1,1)
        else:
            fig = axs.figure

        # figure for consecutive trajectories
        if axs1 is None:
            fig1, axs1 = plt.subplots(1,1)
        else:
            fig1 = axs1.figure

        # load real segments
        df, exit_heading, entry_heading, trajectories  = self.load_outside_dp_segments()

        # what is the plume angle?
        plume_angle = np.deg2rad(int(self.experiment))

        # need to calculate how many consecutive returns the animal makes
        returns = self.consecutive_outside_trajectories()

        # plot the efficiency
        for key in list(trajectories.keys()):
            xy = trajectories[key]
            dist_away = self.orthogonal_distance(xy[:,0], xy[:,1],plume_angle)
            dist_away = np.max(dist_away)
            _, pathlen = fn.path_length(xy[:,0], xy[:,1])
            axs.plot(dist_away, pathlen, '*', color='grey')

        # load all the fictive trajectories
        upwind_a_all = np.round(np.linspace(0,1,11),1)
        upwind_a = [0.,0.5,1.]
        colors = sns.color_palette("rocket",len(upwind_a_all))
        for i, a in enumerate(upwind_a):
            savename = os.path.join(self.picklefol, self.experiment+'_'+'outside_fictive_trajectories'+'_'+str(a)+'.p')

            # load the data
            results = fn.load_obj(savename)
            n_traj = results['n_traj']
            failures = results['failures']
            orthogonal_distances = results['orthogonal_distances']
            pathlengths = results['pathlengths']

            # color for the selected value of a
            color_ix = np.where(upwind_a_all==a)[0][0]
            color = colors[color_ix]

            # plot efficiencies
            axs.plot(orthogonal_distances, pathlengths, '.',linestyle="None", color=color, alpha=1)
            axs.set_xlabel('distance away from plume (mm)')
            axs.set_ylabel('total path length (mm)')

            # plot the consecutive outside trajectories
            n = np.arange(1,np.max(returns))
            success = n_traj/(failures+n_traj)
            probability = success**n
            axs1.plot(n, probability, color=color)
            axs1.set_ylim(-0.05,1.05)
            axs1.set_xlabel('number of consecutive outside trajectories')
            axs1.set_ylabel('P(n returns)')

            # plot the actual number of consecutive outside trajectories
            axs1r = axs1.twinx()
            sns.histplot(x=returns, fill=False, element='step',color='grey')

            # find max bin height
            max_bin = axs1r.lines[0].properties()['data'][1].max()
            axs1r.set_ylim(-0.05*max_bin, 1.05*max_bin)
            axs1r.set_ylabel('returns (n)')


        fig.savefig(os.path.join(self.figurefol, 'compare_fictive_real_efficiency.pdf'))
        fig1.savefig(os.path.join(self.figurefol, 'compare_fictive_real_efficiency_consecutive.pdf'))
        return axs1r

    def find_upwind_bias(self):
        """
        calculate how the upwind bias in the real trajectories compares to the
        upwind bias in the fictive trajectories.  Upwind bias is a function of
        parameter 'a' in the biased turn function used to create the fictive
        fictive trajectories.  Upwind bias is calculated as the change in y
        position divided by the pathlength for any given outside trajectory.
        """
        #figure for plotting upwind bias vs parameter a
        fig, axs = plt.subplots(1,1)
        # load the real data
        df, exit_heading, entry_heading, trajectories  = self.load_outside_dp_segments()

        d_upwind_bias = []

        delta_y = []
        pathlength = []

        # for each trajectory calculate delta y and pathlength
        for key in list(trajectories.keys()):
            xy = trajectories[key]
            _, L = fn.path_length(xy[:,0], xy[:,1])
            delta_y.append(xy[-1,1]-xy[0,1])
            pathlength.append(L)
            d_upwind_bias.append({
                'bias': (xy[-1,1]-xy[0,1])/L,
                'a': 'real'
            })
        delta_y = np.array(delta_y)
        pathlength = np.array(pathlength)

        # calculate the average upwind bias
        upwind_bias = np.mean(delta_y/pathlength)
        axs.plot([0,1], [upwind_bias, upwind_bias], 'k')

        # load all the fictive trajectories
        upwind_a = np.round(np.linspace(0,1,11),1)
        colors = sns.color_palette("rocket",len(upwind_a))
        for i, a in enumerate(upwind_a):
            # reset delta_y, pathlength
            delta_y = []
            pathlength = []

            # load fictive data
            savename = os.path.join(self.picklefol, self.experiment+'_'+'outside_fictive_trajectories'+'_'+str(a)+'.p')
            results = fn.load_obj(savename)
            trajectories = results['d_traj']
            # for each trajectory calculate delta y and pathlength
            for key in list(trajectories.keys()):
                xy = trajectories[key]
                _, L = fn.path_length(xy[:,0], xy[:,1])
                # store results
                delta_y.append(xy[-1,1]-xy[0,1])
                pathlength.append(L)
                # store in a dict as well
                d_upwind_bias.append({
                    'bias': (xy[-1,1]-xy[0,1])/L,
                    'a': a
                })
            pathlength = np.array(pathlength)
            delta_y = np.array(delta_y)

            # calculate the average upwind bias
            upwind_bias = np.median(delta_y/pathlength)

            # plot the bias vs a
            axs.plot(a, upwind_bias, 'o', color = colors[i])
            axs.set_xlabel('a (upwind turn bias)')
            axs.set_ylabel('upwind bias (mm/mm)')

        # df with upwind biases for each traj
        df = pd.DataFrame(d_upwind_bias)
        # sns.boxplot(data = df, x='a', y='bias', ax=axs)
        fig.savefig(os.path.join(self.figurefol,'upwind_bias.pdf'))
        return df

    def compare_fictive_real_avg_trajectory(self, remove_outliers=False, pts=10000):
        """
        plot average trajectories of fictive and real trajectories
        """
        from scipy.stats import sem

        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')


        # upwind bias values
        upwind_a_all = np.round(np.linspace(0,1,11),1)
        upwind_a = [0.,0.5,1.] # smaller subset of a
        colors = sns.color_palette("rocket",len(upwind_a_all))

        fig, axs = plt.subplots(1,1+len(upwind_a), figsize = (4,4))

        # load real segments
        df, exit_heading, entry_heading, trajectories  = self.load_outside_dp_segments()

        # load outside figtive segments
        savename = os.path.join(self.picklefol, self.experiment+'_'+'outside_fictive_trajectories.p')
        d_traj = fn.load_obj(savename)

        # will need to rotate trajectories to calculate farthest distances "up plume"
        angle = int(self.experiment)
        rot_angle = -angle*np.pi/180

        # average trajectory for real segments
        avg_x_out_real = []
        avg_y_out_real = []
        for key in list(trajectories.keys()):
            xy = trajectories[key]
            x = xy[:,0]
            y = xy[:,1]
            t = np.arange(len(x))
            t_common = np.linspace(t[0], t[-1], pts)
            fx = interpolate.interp1d(t, x)
            fy = interpolate.interp1d(t, y)
            #axs.plot(fx(t_common), fy(t_common))
            avg_x_out_real.append(fx(t_common))
            avg_y_out_real.append(fy(t_common))

        x = np.array(avg_x_out_real)
        y = np.array(avg_y_out_real)
        if remove_outliers: # filter outlier trajectories
            # rotate trajectories
            _,y = fn.coordinate_rotation(x,y,angle)
            y_enter = y[:,-1]

            # remove outlier trajectories
            q1 = np.percentile(y_enter, 25)
            q2 = np.percentile(y_enter, 75)
            iqr = q2-q1
            lower = q1-2.0*iqr
            upper = q2+2.0*iqr
            keep_ix = np.where((y_enter>lower)&(y_enter<upper))[0]
            y_enter = y_enter[keep_ix]
            x = x[keep_ix]
            y = y[keep_ix]

        # plot mean with error bar
        xa = np.mean(x, axis=0)
        ya = np.mean(y, axis=0)
        xerr=sem(x)
        yerr=sem(y)
        axs[0].errorbar(xa, ya, xerr=xerr, yerr=yerr, rasterized=True, color='gray', ecolor='k')
        axs[0].plot([0,0], [-10,100], 'k', linewidth=1)
        axs[0].axis('equal')
        axs[0].set_xlim(-20,0)
        axs[0].set_ylim(-10,100)
        axs[0].axis('off')
        axs[0].plot([-13,-3],[-4,-4], 'k')
        axs[0].text(-12.5,-6.5, '10 mm')

        if False:
            x = np.array(x)
            y = np.array(y)

            # rotate trajectories
            _,y = fn.coordinate_rotation(x,y,0)
            y_enter = y[:,-1]

            # remove outlier trajectories
            q1 = np.percentile(y_enter, 25)
            q2 = np.percentile(y_enter, 75)
            iqr = q2-q1
            lower = q1-1.5*iqr
            upper = q2+1.5*iqr
            keep_ix = np.where((y_enter>lower)&(y_enter<upper))[0]

            y_enter = y_enter[keep_ix]
            x = x[keep_ix]
            y = y[keep_ix]

            percentiles = [25, 50, 75]
            for i in np.arange(len(percentiles)-1):
                p = np.percentile(y_enter, [percentiles[i], percentiles[i+1]])
                ix = np.where((p[0]<y_enter)&(y_enter<p[1]))[0]
                x_mean = np.mean(x[ix], axis=0)
                y_mean = np.mean(y[ix], axis=0)
                axs.plot(x_mean, y_mean)
            x_mean = np.mean(x, axis=0)
            y_mean = np.mean(y, axis=0)
            axs.plot(x_mean,y_mean, color='k')
        if False:
            xerr=sem(avg_x_out_real)
            yerr=sem(avg_y_out_real)
            ang = np.arctan2(np.gradient(y),np.gradient(x))
            ang = ang-np.pi/2
            # df = pd.DataFrame({'angle': ang})
            # df = df.rolling(200).apply(scipy.stats.circmean, kwargs={'high':np.pi, 'low':-np.pi}) # smooth angle
            # ang = df.angle.to_numpy()
            err = np.maximum(xerr,yerr)
            x1=err*np.cos(ang)
            y1=err*np.sin(ang)
            x2,y2 = -x1,-y1

            for i in np.arange(len(x)):
                if i%2==0:
                    xl = [x1[i], x2[i]]+x[i]
                    yl = [y1[i], y2[i]]+y[i]
                    axs[0].plot(xl,yl,color = 'g')
            axs[0].plot(x,y,color='white')
            axs[0].set_xlim(-20,0)
            axs[0].set_ylim(-10,100)


        # average trajectory for fictive segments
        # load all the fictive trajectories
        for i, a in enumerate(upwind_a):
            # color
            color_ix = np.where(upwind_a_all==a)[0][0]
            color = colors[color_ix]
            # load fictive data
            savename = os.path.join(self.picklefol, self.experiment+'_'+'outside_fictive_trajectories'+'_'+str(a)+'.p')
            results = fn.load_obj(savename)
            d_traj = results['d_traj']
            avg_x_out_fictive = []
            avg_y_out_fictive = []
            for key in list(d_traj.keys()):
                xy = d_traj[key]
                x = xy[:,0]
                y = xy[:,1]

                t = np.arange(len(x))
                t_common = np.linspace(t[0], t[-1], pts)
                fx = interpolate.interp1d(t, x)
                fy = interpolate.interp1d(t, y)
                #axs.plot(fx(t_common), fy(t_common))
                avg_x_out_fictive.append(fx(t_common))
                avg_y_out_fictive.append(fy(t_common))
            x = np.array(avg_x_out_fictive)
            y = np.array(avg_y_out_fictive)
            if remove_outliers: # filter outlier trajectories
                # rotate trajectories
                _,y = fn.coordinate_rotation(x,y,angle)
                y_enter = y[:,-1]

                # remove outlier trajectories
                q1 = np.percentile(y_enter, 25)
                q2 = np.percentile(y_enter, 75)
                iqr = q2-q1
                lower = q1-2.0*iqr
                upper = q2+2.0*iqr
                keep_ix = np.where((y_enter>lower)&(y_enter<upper))[0]
                y_enter = y_enter[keep_ix]
                x = x[keep_ix]
                y = y[keep_ix]

            xa = np.mean(x, axis=0)
            ya = np.mean(y, axis=0)
            xerr=sem(x)
            yerr=sem(y)
            axs[i+1].errorbar(xa, ya, xerr=xerr, yerr=yerr, color='gray', ecolor=color, rasterized=True)
            axs[i+1].plot([0,0], [-10,100], 'k', linewidth=1)
            axs[i+1].axis('equal')
            axs[i+1].set_xlim(-20,0)
            axs[i+1].set_ylim(-10,100)
            axs[i+1].axis('off')

            if False:

                xerr=sem(avg_x_out_fictive)
                yerr=sem(avg_y_out_fictive)
                ang = np.arctan2(np.gradient(y),np.gradient(x))
                ang = ang-np.pi/2
                # df = pd.DataFrame({'angle': ang})
                # df = df.rolling(200).apply(scipy.stats.circmean, kwargs={'high':np.pi, 'low':-np.pi}) # smooth angle
                # ang = df.angle.to_numpy()
                err = np.maximum(xerr,yerr)
                x1=err*np.cos(ang)
                y1=err*np.sin(ang)
                x2,y2 = -x1,-y1

                for j in np.arange(len(x)):
                    if j%2==0:
                        xl = [x1[j], x2[j]]+x[j]
                        yl = [y1[j], y2[j]]+y[j]
                        axs[i+1].plot(xl,yl,color = colors[i])
                axs[i+1].plot(x,y,color='white')

                axs[i+1].set_xlim(-20,0)
                axs[i+1].set_ylim(-10,100)
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'average_trajectory.pdf'))
        return avg_x_out_real, avg_y_out_real

    def inbound_outbound_angle_outside(self, x, y):
        """
        calculate the inbound and outbound angle relative to the edge for a given outside trajectory
        modified to work with the jumping plume geometry

        """

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
        return [angle_out[0], angle_in[0]]

    def compare_fictive_real_inbound_outbound_angle(self, remove_outliers=False, pts=10000):
        """
        make a plot of the outbound and inbound distance for the real trajectories
        and for different random walk models with different parameters of a.
        """
        from scipy.stats import sem

        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')

        # store results
        metrics = []
        # upwind bias values
        upwind_a_all = np.round(np.linspace(0,1,11),1)
        upwind_a = [0.,0.5,1.] # smaller subset of a
        colors = sns.color_palette("rocket",len(upwind_a_all))

        fig, axs = plt.subplots(1,4, figsize = (2,1), sharey=True)
        fig2, axs2 = plt.subplots(1,1, figsize = (1,1), sharey=True)

        # load real segments
        df, exit_heading, entry_heading, trajectories  = self.load_outside_dp_segments()
        all_angles_in, all_angles_out = [],[]
        num_plot=0
        for key in list(trajectories.keys()):
            xy = trajectories[key]
            x,y = xy[:,0], xy[:,1]
            [angle_out, angle_in] = self.inbound_outbound_lengths(x,y)
            all_angles_in.append(angle_in)
            all_angles_out.append(angle_out)
            metrics.append({
            'a': 'real',
            'angle': angle_out,
            'dir': 'out'
            })
            metrics.append({
            'a': 'real',
            'angle': angle_in,
            'dir': 'in'
            })
        for aax in [axs[0], axs2]:
            sns.despine(ax=aax)
            aax.set_xticks([0,1])
            aax.set_xticklabels(['out', 'in'])
            aax.set_ylabel('distance (mm)')
            aax.spines['bottom'].set_visible(False)
            pl.paired_plot(aax,all_angles_out, all_angles_in)
            aax.set_title('flies', color='k')
            aax.text(x=0.3, y=35, s='***')

        #axs[0].set_yscale('log')

        # load fictive segments
        for i, a in enumerate(upwind_a):
            all_angles_in, all_angles_out = [],[]
            # color
            color_ix = np.where(upwind_a_all==a)[0][0]
            color = colors[color_ix]
            # load fictive data
            savename = os.path.join(self.picklefol, self.experiment+'_'+'outside_fictive_trajectories'+'_'+str(a)+'.p')
            results = fn.load_obj(savename)
            d_traj = results['d_traj']
            for key in list(d_traj.keys()):
                xy = d_traj[key]
                x = xy[:,0]
                y = xy[:,1]
                [angle_out, angle_in] = self.inbound_outbound_lengths(x,y)
                all_angles_in.append(angle_in)
                all_angles_out.append(angle_out)
                metrics.append({
                'a': str(a),
                'angle': angle_out,
                'dir': 'out'
                })
                metrics.append({
                'a': str(a),
                'angle': angle_in,
                'dir': 'in'
                })
            pl.paired_plot(axs[i+1],all_angles_out, all_angles_in, line_color=color)
            axs[i+1].set_xticks([0,1])
            axs[i+1].set_xticklabels(['out', 'in'])
            axs[i+1].set_title('a='+str(a), color=color)
            axs[i+1].text(x=0.2, y=240, s='n.s.')
            pl.despine(axs[i+1])
            # axs[i+1].axis('off')

        fig.tight_layout()
        fig2.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'outbound_inbound_distance_real_vs_model.pdf'))
        fig2.savefig(os.path.join(self.figurefol, 'outbound_inbound_distance_real.pdf'))
        return pd.DataFrame(metrics)

    # def outbound_inbound_angle(self):

    def run_length_turn_angle_distributions(self):
        """
        make plots of the distributions for the run lengths and turn angles
        """
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')


        # load data
        df, exit_heading, entry_heading, trajectories = self.load_outside_dp_segments()
        turns = np.abs(np.rad2deg(df.angles.to_list())) # make absolute value
        lengths = df.lengths

        # turn angle plot
        fig, axs = plt.subplots(1,1, figsize=(1,1))
        sns.histplot(turns, element='step', bins=10, fill=False, color='grey')
        sns.despine()
        axs.set_xticks([0,90,180])
        axs.tick_params(axis='x', pad=-3)
        axs.tick_params(axis='y', pad=-3)
        axs.set_xlabel('|turn angle ($^o$)|', fontsize=7, labelpad=-1)
        axs.set_ylabel('count', fontsize=7, labelpad=-1)
        fig.savefig(os.path.join(self.figurefol, 'turn_angle_distribution_small.pdf'))

        # run length plot
        fig, axs = plt.subplots(1,1, figsize=(1,1))
        sns.histplot(lengths, element='step', bins=20, fill=False, color='grey')
        sns.despine()
        axs.set_xlabel('run lengths (mm)', fontsize=7, labelpad=-1)
        axs.set_ylabel('count', fontsize=7, labelpad=-1)
        axs.tick_params(axis='x', pad=-3)
        axs.tick_params(axis='y', pad=-3)
        fig.savefig(os.path.join(self.figurefol, 'run_length_distribution_small.pdf'))

    def current_turn_v_next_turn(self):
        """
        make a plot showing the current turn vs the next turn for the DP-simplified
        outside trajectories
        """
        # load trajectories
        df, exit_heading, entry_heading, trajectories = self.load_outside_dp_segments()

        current_turn = []
        next_turn = []
        for i in df.index.unique():
            df_temp = df.loc[df.index==i]
            if df_temp.shape[0]>1: # need to have at least two turns
                for m in np.arange(df_temp.shape[0]-1):
                    current_turn.append(df_temp.angles.iloc[m])
                    next_turn.append(df_temp.angles.iloc[m+1])

        g = sns.JointGrid(x=np.rad2deg(current_turn), y=np.rad2deg(next_turn))
        g.plot_joint(sns.kdeplot,
                     fill=True, clip=((-180, 180), (-180, 180)), levels=50,cmap="rocket")

        g.ax_joint.set_xlabel('current_turn')
        g.ax_joint.set_ylabel('next_turn')
        g.ax_joint.set_xticks([-90,0,90])
        g.ax_joint.set_yticks([-90,0,90])

        g.savefig(os.path.join(self.figurefol, 'current_turn_v_next_turn.pdf'))

    def current_heading_v_next_turn(self):
        """
        make a plot showing the current heading vs the next turn for the DP-simplified
        outside trajectories.
        """
        # load trajectories
        df, exit_heading, entry_heading, trajectories = self.load_outside_dp_segments()

        # convert headings to wind centric coordinates
        current_heading = np.rad2deg(fn.conv_cart_upwind(df.heading.to_numpy(dtype='float64')))
        df["heading"] = np.rad2deg(fn.conv_cart_upwind(df.heading.to_numpy(dtype='float64')))
        next_turn = np.rad2deg(df.angles.to_list())

        if True:
            g = sns.JointGrid(x=current_heading, y=next_turn)
            g.plot_joint(sns.kdeplot,
                         fill=True, clip=((-180, 180), (-180, 180)), levels=25,cmap="Blues")

            g.ax_joint.set_xlabel('current_heading')
            g.ax_joint.set_ylabel('next_turn')
            g.ax_joint.set_xticks([-90,0,90])
            g.ax_joint.set_yticks([-90,0,90])
            g.savefig(os.path.join(self.figurefol, 'current_heading_v_next_turn.pdf'))
        else:
            fig, axs = plt.subplots(1,1)
            sns.kdeplot(data=df, x='heading', y='angles', levels=5)
        return trajectories

    def current_turn_v_next_run(self):
        """
        make a plot showing the current turn vs the next run length for the DP-simplified
        outside trajectories.  Turns out there is a negative correlation but it's not that informative.
        The trajectories were simplified with the DP algorithm, which means small turns
        are more likely to be followed by large run lengths.
        """
        # load trajectories
        df, exit_heading, entry_heading, trajectories = self.load_outside_dp_segments()

        # convert headings to wind centric coordinates
        heading = np.rad2deg(fn.conv_cart_upwind(df.heading.to_numpy(dtype='float64')))
        df["heading"] = np.rad2deg(fn.conv_cart_upwind(df.heading.to_numpy(dtype='float64')))
        turn = np.rad2deg(df.angles.to_list())
        run = df.lengths.to_numpy()
        if True:
            g = sns.JointGrid(x=heading, y=run)
            g.plot_joint(sns.kdeplot,
                         fill=True, clip=((-180, 180), (-180, 180)), levels=25,cmap="rocket")

            g.ax_joint.set_xlabel('current_heading')
            g.ax_joint.set_ylabel('next_turn')
            g.ax_joint.set_xticks([-90,0,90])
            #g.ax_joint.set_yticks([-90,0,90])
            g.savefig(os.path.join(self.figurefol, 'current_turn_v_next_run.pdf'))
        else:
            turn = np.abs(turn)
            fig, axs = plt.subplots(1,1)
            sns.scatterplot(turn, run, alpha=0.1)
            print(stats.pearsonr(turn, run))
        return trajectories

    def current_heading_v_run_length(self):
        """
        make a plot showing the current turn vs the next turn for the DP-simplified
        outside trajectories
        """
        # load trajectories
        df, exit_heading, entry_heading, trajectories = self.load_outside_dp_segments()

        # convert headings to wind centric coordinates
        current_heading = np.rad2deg(fn.conv_cart_upwind(df.heading.to_numpy(dtype='float64')))
        lengths = df.lengths.to_numpy()
        fig, axs = plt.subplots(1,1)
        sns.scatterplot(ax=axs)

    def compare_fictive_real_entry_directions(self):
        """
        compare the fictive and real entry headings
        """
        # load real segments
        df, exit_heading, entry_heading, trajectories  = self.load_outside_dp_segments()

        # load outside figtive segments
        savename = os.path.join(self.picklefol, self.experiment+'_'+'outside_fictive_trajectories.p')
        d_traj = fn.load_obj(savename)

        headings = []

        fig, axs = plt.subplots(1,1)

        # work through real trajectories
        for key in list(trajectories.keys()):
            xy = trajectories[key]
            del_y = xy[-1,1]-xy[-2,1]
            del_x = xy[-1,0]-xy[-2,0]
            heading = np.arctan2(del_y, del_x)
            headings.append({
                'heading':heading,
                'type':'real'
            })

        for key in list(d_traj.keys()):
            xy = d_traj[key]
            del_y = xy[-1,1]-xy[-2,1]
            del_x = xy[-1,0]-xy[-2,0]
            heading = np.arctan2(del_y, del_x)
            headings.append({
                'heading':heading,
                'type':'fictive'
            })
        df = pd.DataFrame(headings)
        df['heading'] = fn.wrap(-df.heading+np.pi)
        sns.histplot(data=df,x='heading', hue='type', ax=axs)

    def plot_random_inside_outside_trajectories(self):
                for key in list(trajectories.keys()):
                    xy = trajectories[key]
                    x = xy[:,0]
                    y = xy[:,1]

    def heatmap(self,cmap='gist_gray', overwrite = False, plot_individual=False, res = 5):
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

        if not os.path.exists(filename) or overwrite:
            # set up arrays for heatmap
            if self.experiment == '0':
                x_bounds = np.arange(-200, 200, res)
                y_bounds = np.arange(0, 1000, res)
            if self.experiment == '15':
                x_bounds = np.arange(-100, 300, res)
                y_bounds = np.arange(0, 1000, res)
            if self.experiment == '45':
                x_bounds = np.arange(-50, 1000, res)
                y_bounds = np.arange(0, 1000, res)
            if self.experiment == 'T':
                self.experiment = '90'
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
                if self.sheet.iloc[i].mfc=='new':
                    idx_start = df.index[df.mfc2_stpt>0.01].tolist()[0]
                    df = df.iloc[idx_start:]
                xi = df.ft_posx.to_numpy()
                yi = df.ft_posy.to_numpy()
                xi=xi-xi[0]
                yi=yi-yi[0]

                # reflect tilted and plumes so that they are facing the right way
                if self.sheet.iloc[i].direction=='left':
                    xi = -xi
                if self.experiment == '90':
                    # halfway point
                    ix1_2 = int(len(xi)/2)
                    if np.mean(xi[ix1_2:])<0:
                        xi = -xi
                # save and example trajectory to plot on top of the heatmap
                if log in examples:
                    x_ex = xi
                    y_ex = yi
                if plot_individual == True:
                    fig, axs = plt.subplots(1,1)
                    axs.plot(xi,yi)
                    axs.title.set_text(log)

                # rotate the tilted plume to make them vertical.
                angle = int(self.experiment)
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
        fig, axs = plt.subplots(1,1)
        vmin = np.percentile(fly_density[fly_density>0],0)
        vmax = np.percentile(fly_density[fly_density>0],90)
        im = axs.imshow(fly_density, cmap=cmap, vmin=vmin,vmax = vmax, rasterized=True, extent=(min(x_bounds), max(x_bounds), min(y_bounds), max(y_bounds)))
        if self.experiment == '0':
            axs.plot([-25,-25], [np.min(y_bounds), np.max(y_bounds)], 'w', alpha=0.2)
            axs.plot([25,25], [np.min(y_bounds), np.max(y_bounds)], 'w', alpha=0.2)
        if self.experiment == '90':
            axs.plot([np.min(x_bounds), np.max(x_bounds)], [300,300], 'w', alpha=0.2)
            axs.plot([np.min(x_bounds), -25], [250,250], 'w', alpha=0.2)
            axs.plot([25, np.max(x_bounds)], [250,250], 'w', alpha=0.2)
            axs.plot([25,25], [np.min(y_bounds), 250], 'w', alpha=0.2)
            axs.plot([-25,-25], [np.min(y_bounds), 250], 'w', alpha=0.2)
        axs.plot(x_ex, y_ex, 'red', linewidth=1)#, alpha = 0.8, linewidth=0.5)
        fig.colorbar(im)
        fig.savefig(os.path.join(self.figurefol, self.experiment+'_heatmap.pdf'))

        fig, axs = plt.subplots(1,1)
        # vmin = np.percentile(fly_density_rotate[fly_density_rotate>0],10)
        # vmax = np.percentile(fly_density_rotate[fly_density_rotate>0],90)
        axs.imshow(fly_density_rotate, cmap=cmap, vmin=1,vmax = 6, extent=(min(x_bounds_rotate), max(x_bounds_rotate), min(y_bounds_rotate), max(y_bounds_rotate)))


        # boundaries
        if self.experiment == '0':
            boundary=25
        if self.experiment == '15':
            boundary = 25/np.sin(np.deg2rad(90-int(self.experiment)))
            boundary = 25
        if self.experiment == '45':
            boundary = 25*np.cos(np.deg2rad(90-int(self.experiment)))
            #boundary = 25
        if self.experiment == '90':
            boundary=25
        fly_density_projection = fly_density_rotate[0:-40, :]/np.sum(fly_density_rotate[0:-40, :])
        x_mean = np.sum(fly_density_projection, axis = 0)
        fig, axs = plt.subplots(1,1)
        axs.plot([-boundary, -boundary], [min(x_mean), max(x_mean)], 'k', alpha=0.5)
        axs.plot([boundary, boundary], [min(x_mean), max(x_mean)],'k', alpha=0.5)
        axs.plot(x_bounds_rotate, x_mean)
        axs.set_ylim(0,0.06)
        fig.savefig(os.path.join(self.figurefol, self.experiment+'_density.pdf'))
        return fly_density, fly_density_rotate

    def plot_individual_trajectories(self):
        for i, log in enumerate(self.sheet.log):
            print(log)
            data = fn.read_log(os.path.join(self.logfol, log))
            mfc = self.sheet.mfc.iloc[i]
            # data['instrip'] = np.where(np.abs(data.ft_posx)<25, True, False)
            data = fn.exclude_lost_tracking(data, thresh=10)
            if mfc == 'old': # all experiments performed with old MFCs
                data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
            fig, axs = plt.subplots(1,1)
            pl.plot_trajectory(data, axs)
            pl.plot_trajectory_odor(data, axs)
            axs.axis('equal')
            fig.suptitle(log)

    def load_average_trajectories(self):
        pickle_name = os.path.join(self.picklefol, self.experiment+'_'+'average_trajectories.p')
        [animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out] = fn.load_obj(pickle_name)
        return animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out

    def plot_trajectories_T(self, pts=10000):
        all_data = self.load_trajectories()
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = [],[],[],[]
        for log in self.sheet.log:
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
                    if np.abs(y[-1]-y[0])<1: #condition for returning on horizontal plume, may need to add in a y condition here
                        count+=1
                        x0 = x[0]
                        y0 = y[0]
                        x = x-x0
                        y = y-y0
                        # if x[-1]<x[0]:
                        #     x=-x
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
                    if np.abs(y[-1]-y[0])<1: #condition for returning on horizontal plume, may need to add in a y condition here
                        x0 = x[0]
                        y0 = y[0]
                        x = x-x0
                        y = y-y0
                        # if x[-1]<x[0]:
                        #     x=-x
                        t = np.arange(len(x))
                        t_common = np.linspace(t[0], t[-1], pts)
                        fx = interpolate.interp1d(t, x)
                        fy = interpolate.interp1d(t, y)
                        #axs.plot(fx(t_common), fy(t_common))
                        avg_x_in.append(fx(t_common))
                        avg_y_in.append(fy(t_common))
            if count>3: # condition: each trajectory needs more than three outside trajectories
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
                fig.savefig(os.path.join(self.figurefol, log.replace('.log', '.pdf')), transparent=True)

        # make an average of the averages for ech fly
        fig, axs = plt.subplots(1,1)


        for i in np.arange(len(animal_avg_x_in)):
            if animal_avg_x_in[i][-1]<0:
                animal_avg_x_in[i] = -animal_avg_x_in[i]
            axs.plot(animal_avg_x_in[i], animal_avg_y_in[i], 'r', alpha=0.1)
        axs.plot(np.mean(animal_avg_x_in, axis=0), np.mean(animal_avg_y_in, axis=0), 'r')
        exit_x = np.mean(animal_avg_x_in, axis=0)[-1]

        for i in np.arange(len(animal_avg_x_out)):
            if animal_avg_x_out[i][-1]<0:
                animal_avg_x_out[i] = -animal_avg_x_out[i]
            axs.plot(animal_avg_x_out[i]+exit_x, animal_avg_y_out[i], 'k', alpha=0.1)
        axs.plot(np.mean(animal_avg_x_out+exit_x, axis=0), np.mean(animal_avg_y_out, axis=0), 'k')

        # save the average trajectories
        fn.save_obj([animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out], os.path.join(self.picklefol, self.experiment+'_'+'average_trajectories.p'))

        fig.savefig(os.path.join(self.figurefol, 'all_averges.pdf'), transparent = True)
        return axs

    def plot_trajectories_angle(self,angle,pts = 10000):
        """
        for plotting average trajectories at a given angle
        """
        n=0
        all_data = self.load_trajectories()
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = [],[],[],[]
        for i, log in enumerate(self.sheet.log):
            avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
            temp = all_data[log]
            direction = self.sheet.direction.iloc[i]
            angle_from_horizontal = np.pi/2 - angle*np.pi/180
            if direction == 'right':
                rot_angle = -angle*np.pi/180
            elif direction == 'left':
                rot_angle = +angle*np.pi/180
            elif direction == 'NA':
                rot_angle = 0
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

            if count>3: # condition: each trajectory needs more than three outside trajectories
                n+=1
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
        fn.save_obj([animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out], os.path.join(self.picklefol, self.experiment+'_'+'average_trajectories.p'))


        # make an average of the averages for ech fly
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
        axs.plot([0,max_y/np.tan(angle_from_horizontal)], [0, max_y], color='k', linestyle='dashed')

        axs.axis('equal')
        fig.savefig(os.path.join(self.figurefol, 'all_averages.pdf'), transparent = True)
        print('number of animals = ', n)
        return axs

    def plot_speed_overlay(self, angle, pts=10000):
        """
        for plotting average trajectories at a given angle
        """
        n=0
        all_data = self.load_trajectories()
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out, animal_avg_speed_in, animal_avg_speed_out = [],[],[],[],[],[]
        for i, log in enumerate(self.sheet.log):
            avg_x_in, avg_y_in, avg_x_out, avg_y_out, avg_speed_in, avg_speed_out=[],[],[],[],[],[]
            temp = all_data[log]
            direction = self.sheet.direction.iloc[i]
            angle_from_horizontal = np.pi/2 - angle*np.pi/180
            if direction == 'right':
                rot_angle = -angle*np.pi/180
            elif direction == 'left':
                rot_angle = +angle*np.pi/180
            elif direction == 'NA':
                rot_angle = 0
            do = temp['do']
            di = temp['di']
            df = temp['data']
            df_instrip = df.where(df.instrip==True)
            del_t = np.mean(np.diff(df.seconds))
            effective_rate = 1/del_t
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
                    vx = np.gradient(x)*effective_rate
                    vy = np.gradient(y)*effective_rate
                    speed = np.sqrt(vx**2+vy**2)
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
                        fs = interpolate.interp1d(t,speed)
                        #axs.plot(fx(t_common), fy(t_common))
                        avg_x_out.append(fx(t_common))
                        avg_y_out.append(fy(t_common))
                        avg_speed_out.append(fs(t_common))
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
                    vx = np.gradient(x)*effective_rate
                    vy = np.gradient(y)*effective_rate
                    speed = np.sqrt(vx**2+vy**2)
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
                        fs = interpolate.interp1d(t,speed)
                        #axs.plot(fx(t_common), fy(t_common))
                        avg_x_in.append(fx(t_common))
                        avg_y_in.append(fy(t_common))
                        avg_speed_in.append(fs(t_common))

            if count>3: # condition: each trajectory needs more than three outside trajectories
                n+=1
                animal_avg_x_out.append(np.mean(avg_x_out, axis=0))
                animal_avg_y_out.append(np.mean(avg_y_out, axis=0))
                animal_avg_speed_out.append(np.mean(avg_speed_out, axis=0))

                animal_avg_x_in.append(np.mean(avg_x_in, axis=0))
                animal_avg_y_in.append(np.mean(avg_y_in, axis=0))
                animal_avg_speed_in.append(np.mean(avg_speed_in, axis=0))

        # make an average of the averages for ech fly
        fig, axs = plt.subplots(1,1)

        # for i in np.arange(len(animal_avg_x_in)):
        #     axs.plot(animal_avg_x_in[i], animal_avg_y_in[i], 'k', alpha=0.1)
        #axs.plot(np.mean(animal_avg_x_in, axis=0), np.mean(animal_avg_y_in, axis=0), 'r')
        exit_x = np.mean(animal_avg_x_in, axis=0)[-1]
        exit_y = np.mean(animal_avg_y_in, axis=0)[-1]
        xi = np.mean(animal_avg_x_in, axis=0)
        yi = np.mean(animal_avg_y_in, axis=0)
        si = np.mean(animal_avg_speed_in, axis=0)
        so = np.mean(animal_avg_speed_in, axis=0)
        cmin = np.min([np.min(so)])
        cmax = np.max([np.max(so)])
        axs = pl.colorline(axs, xi, yi, z=si, segmented_cmap=False, cmap=plt.get_cmap('viridis'), norm = plt.Normalize(cmin, cmax))


        # for i in np.arange(len(animal_avg_x_out)):
        #     axs.plot(animal_avg_x_out[i]+exit_x, animal_avg_y_out[i]+exit_y, 'k', alpha=0.1)
        #axs.plot(np.mean(animal_avg_x_out+exit_x, axis=0), np.mean(animal_avg_y_out+exit_y, axis=0), 'k')
        xo = np.mean(animal_avg_x_out+exit_x, axis=0)
        yo = np.mean(animal_avg_y_out+exit_y, axis=0)
        so = np.mean(animal_avg_speed_out, axis=0)
        axs = pl.colorline(axs, xo, yo, z=so, segmented_cmap=False, cmap=plt.get_cmap('viridis'), norm = plt.Normalize(cmin, cmax))
        axs.autoscale()
        axs.axis('equal')

        # draw the plume boundary line at the appropriate angle
        max_y_out = np.max(np.mean(animal_avg_y_out,axis=0))+exit_y
        max_y_in = np.max(np.mean(animal_avg_y_in, axis=0))
        max_y = np.max((max_y_out,max_y_in))
        axs.plot([0,max_y/np.tan(angle_from_horizontal)], [0, max_y], color='k', linestyle='dashed')

        # plot a colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=plt.Normalize(cmin, cmax))
        plt.colorbar(sm)

        axs.axis('equal')
        fig.savefig(os.path.join(self.figurefol, 'all_averages_speed_heatmap.pdf'), transparent = True)
        print('number of animals = ', n)
        return axs

                #fig.suptitle(log)
                #fig.savefig(os.path.join(self.figurefol, log.replace('.log', '.pdf')), transparent=True)

    def inbound_outbound_angle(self, x, y):
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
        max_ix = np.argmax(np.abs(x))
        start_x = x[0]
        start_y = y[0]
        max_x = x[max_ix]
        max_y = y[max_ix]
        return_x = x[-1]
        return_y = y[-1]

        ## old way of calculating vectors, calculates the inside.
        vec_out = [max_x-start_x, max_y-start_y]
        edge_vec_out = [return_x-start_x, return_y-start_y]
        vec_in = [max_x-return_x, max_y-return_y]
        edge_vec_in = [start_x-return_x, start_y-return_y]

        # vec_out = [max_x-start_x, max_y-start_y]
        # edge_vec_out = [return_x-start_x, return_y-start_y]
        # vec_in = [max_x-return_x, max_y-return_y]
        # edge_vec_in = [return_x-start_x, return_y-start_y]


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

    def inbound_outbound_lengths(self,x,y):
        """
        inbound and outbound path length
        """
        x = x-x[0]
        y = y-y[0]
        max_ix = np.argmax(np.abs(x))
        _,length_out = fn.path_length(x[0:max_ix+1], y[0:max_ix+1])
        _,length_in = fn.path_length(x[max_ix:], y[max_ix:])


        # fig, axs = plt.subplots(1,1)
        # axs.plot(x[0:max_ix], y[0:max_ix])
        # axs.plot(x[max_ix:], y[max_ix:])
        # axs.text(x[max_ix],y[max_ix],str([length_out, length_in]))

        return [length_out, length_in]

    def inbound_outbound_tortuosity(self,x,y):
        """
        inbound and outbound tortuosity calculated as path length divided by euclidean distance
        """
        max_ix = np.argmax(np.abs(x))
        start_x = x[0]
        start_y = y[0]
        max_x = x[max_ix]
        max_y = y[max_ix]
        return_x = x[-1]
        return_y = y[-1]
        euc_out = np.sqrt((start_x-max_x)**2+(start_y-max_y)**2)
        euc_in = np.sqrt((return_x-max_x)**2+(return_y-max_y)**2)
        _,length_out = fn.path_length(x[0:max_ix], y[0:max_ix])
        _,length_in = fn.path_length(x[max_ix:], y[max_ix:])
        tor_outbound = length_out/euc_out
        tor_inbound = length_in/euc_in
        return [tor_outbound, tor_inbound]

    def plot_outside_pathlengths(self):
        """
        take the average path length for the first n outside trajectories
        and compare it to the average path length for the rest of the
        outside trajectories.
        """
        #load data
        all_data = self.load_trajectories()

        # average over this number of outside bouts
        num_outside_avg = [1,2,3,4,5]
        fig, axs = plt.subplots(2,len(num_outside_avg), figsize = (2*len(num_outside_avg), 4), )
        for n in num_outside_avg:
            pairs = []
            pairs_t = []
            for i, log in enumerate(self.sheet.log):
                temp = all_data[log]
                do = temp['do']
                lens = []
                ts = []
                if len(do)>2*n:
                    for key in list(do.keys())[:-1]:
                        x = do[key].ft_posx.to_numpy()
                        y = do[key].ft_posy.to_numpy()
                        _, l = fn.path_length(x,y)
                        t = do[key].seconds.to_numpy()
                        t = t[-1]-t[0]

                        lens.append(l)
                        ts.append(t)
                    first = np.mean(lens[0:n])
                    rest = np.mean(lens[n:])
                    first_t = np.mean(ts[0:n])
                    rest_t = np.mean(ts[n:])
                    pairs.append([first, rest])
                    pairs_t.append([first_t, rest_t])
                    #axs.plot([first_2, rest])
            pairs = np.array(pairs)
            pairs_t = np.array(pairs_t)
            axs[0,n-1] = pl.paired_plot(axs[0,n-1], pairs[:,0], pairs[:,1])
            axs[0,n-1].set_xticks([0,1])
            axs[0,n-1].set_ylabel('average path length (mm)')
            axs[1,n-1] = pl.paired_plot(axs[1,n-1], pairs_t[:,0], pairs_t[:,1])
            axs[1,n-1].set_xticks([0,1])
            axs[1,n-1].set_ylabel('average outside time (s)')
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'first_rest_pathlengths.pdf'))
        return pairs

    def inbound_outbound_angle_distribution(self,angle, pts = 10000):
        all_data = self.load_trajectories()
        angles_outside, angles_inside=[],[]
        for i, log in enumerate(self.sheet.log):
            avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
            temp = all_data[log]
            direction = self.sheet.direction.iloc[i]
            angle_from_horizontal = np.pi/2 - angle*np.pi/180
            if direction == 'right':
                rot_angle = -angle*np.pi/180
            elif direction == 'left':
                rot_angle = +angle*np.pi/180
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

            if count>3: # condition: each trajectory needs more than three outside trajectories
                print(log)
                for i in np.arange(len(avg_x_in)):
                    x = avg_x_in[i]
                    y = avg_y_in[i]
                    x,y = fn.coordinate_rotation(x, y, rot_angle)
                    angles = self.inbound_outbound_angle(x, y)
                    angles_inside.append(angles)
                for i in np.arange(len(avg_x_out)):
                    x = avg_x_out[i]
                    y = avg_y_out[i]
                    x,y = fn.coordinate_rotation(x, y, rot_angle)
                    angles = self.inbound_outbound_angle(x, y)
                    angles_outside.append(angles)
        angles_outside = np.array(angles_outside)
        angles_inside = np.array(angles_inside)
        df_inside = pd.DataFrame({'outbound': angles_inside[:,0], 'inbound': angles_inside[:,1]})
        df_outside = pd.DataFrame({'outbound': angles_outside[:,0], 'inbound': angles_outside[:,1]})

        colors = sns.color_palette()

        fig, axs = plt.subplots(1,2, figsize=(6,3))

        sns.histplot(data=df_inside, x='inbound', element='step',stat='density', cumulative=True, fill=False, kde=False, color = colors[0], bins=32, ax=axs[0])
        sns.histplot(data=df_inside, x='outbound', element='step',stat='density',cumulative=True, fill=False, kde=False, color = colors[1], bins=32, ax=axs[0])
        axs[0].set_xlabel('angle (degrees)')
        axs[0].set_xticks([0,45,90,135,180])
        axs[0].title.set_text('inside_trajectories')
        #fig.savefig(os.path.join(self.figurefol, 'inside_angle_distibution.pdf'), transparent = True)


        sns.histplot(data=df_outside, x='inbound',stat='density', element='step',cumulative=True, fill=False, kde=False, color = colors[0], bins=32, ax=axs[1])
        sns.histplot(data=df_outside, x='outbound',stat='density', element='step',cumulative=True, fill=False, kde=False, color = colors[1], bins=32, ax=axs[1])
        axs[1].set_xlabel('angle (degrees)')
        axs[1].set_xticks([0,45,90,135,180])
        axs[1].title.set_text('outside_trajectories')
        fig.savefig(os.path.join(self.figurefol, 'angle_distribution.pdf'), transparent = True)

    def average_inbound_outbound_angle(self, angle):
        # calculate the outbound and inbound angles for the average trajectories
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = self.load_average_trajectories()
        rot_angle = -angle*np.pi/180
        angles_outside = []
        for i in np.arange(len(animal_avg_x_out)):
            x = animal_avg_x_out[i]
            y = animal_avg_y_out[i]
            x,y = fn.coordinate_rotation(x, y, rot_angle)
            angles = self.inbound_outbound_angle(x, y)
            angles_outside.append(angles)
        angles_outside = np.array(angles_outside)

        return angles_outside

    def average_inbound_outbound_path_length(self, angle):
        # calculate the outbound and inbound angles for the average trajectories
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = self.load_average_trajectories()
        rot_angle = -angle*np.pi/180
        lengths_outside = []
        for i in np.arange(len(animal_avg_x_out)):
            x = animal_avg_x_out[i]
            y = animal_avg_y_out[i]
            x,y = fn.coordinate_rotation(x, y, rot_angle)
            lengths = self.inbound_outbound_lengths(x, y)
            lengths_outside.append(lengths)
        lengths_outside = np.array(lengths_outside)

        return lengths_outside

    def average_inbound_outbound_tortuosity(self, angle):
        # calculate the outbound and inbound angles for the average trajectories
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = self.load_average_trajectories()
        rot_angle = -angle*np.pi/180
        tor_outside = []
        for i in np.arange(len(animal_avg_x_out)):
            x = animal_avg_x_out[i]
            y = animal_avg_y_out[i]
            x,y = fn.coordinate_rotation(x, y, rot_angle)
            tor = self.inbound_outbound_tortuosity(x, y)
            tor_outside.append(tor)
        tor_outside = np.array(tor_outside)

        return tor_outside

    def inbound_outbound_path_length(self,angle, pts = 10000):
        all_data = self.load_trajectories()
        lengths_outside, lengths_inside=[],[]
        for i, log in enumerate(self.sheet.log):
            avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
            temp = all_data[log]
            direction = self.sheet.direction.iloc[i]
            angle_from_horizontal = np.pi/2 - angle*np.pi/180
            if direction == 'right':
                rot_angle = -angle*np.pi/180
            elif direction == 'left':
                rot_angle = +angle*np.pi/180
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

            if count>3: # condition: each trajectory needs more than three outside trajectories
                print(log)
                for i in np.arange(len(avg_x_in)):
                    x = avg_x_in[i]
                    y = avg_y_in[i]
                    x,y = fn.coordinate_rotation(x, y, rot_angle)
                    lengths = self.inbound_outbound_tortuosity(x, y)
                    lengths_inside.append(lengths)
                for i in np.arange(len(avg_x_out)):
                    x = avg_x_out[i]
                    y = avg_y_out[i]
                    x,y = fn.coordinate_rotation(x, y, rot_angle)
                    lengths = self.inbound_outbound_tortuosity(x, y)
                    lengths_outside.append(lengths)
        lengths_outside = np.array(lengths_outside)
        lengths_inside = np.array(lengths_inside)
        # df_inside = pd.DataFrame({'outbound': lengths_inside[:,0], 'inbound': lengths_inside[:,1]})
        # df_outside = pd.DataFrame({'outbound': lengths_outside[:,0], 'inbound': lengths_outside[:,1]})
        #
        # colors = sns.color_palette()
        #
        # fig, axs = plt.subplots(1,1)
        #
        # sns.histplot(data=df_inside, x='inbound', element='step', fill=False, kde=False, color = colors[0], bins=16)
        # sns.histplot(data=df_inside, x='outbound', element='step', fill=False, kde=False, color = colors[1], bins=16)
        # axs.set_xlabel('angle (degrees)')
        # axs.set_xticks([0,45,90,135,180])
        # fig.savefig(os.path.join(self.figurefol, 'inside_angle_distibution.pdf'), transparent = True)

        # fig, axs = plt.subplots(1,1)
        # sns.histplot(data=df_outside, x='inbound', element='step', fill=False, kde=False, color = colors[0], bins=16)
        # sns.histplot(data=df_outside, x='outbound', element='step', fill=False, kde=False, color = colors[1], bins=16)
        # axs.set_xlabel('angle (degrees)')
        # axs.set_xticks([0,45,90,135,180])
        # fig.savefig(os.path.join(self.figurefol, 'outside_angle_distibution.pdf'), transparent = True)
        return lengths_inside, lengths_outside

    def time_in_v_time_out(self):
        """
        plot the current inside time with the next outside time.  Doesn't seem
        to be a clear correlation.  Won't do much more with this analysis, but
        it might be useful at some point
        """
        all_data = self.load_trajectories()
        sheet = self.sheet.log
        inside_time_all = []
        outside_time_all = []
        fig, axs = plt.subplots(1,1,figsize=(4,4))
        for i, log in enumerate(sheet):

            print(log)
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']
            if len(do)>10:
                for key in list(di.keys()):
                    if key+1 in list(do.keys()):
                        inside_time = di[key].seconds.to_numpy()
                        outside_time = do[key+1].seconds.to_numpy()
                        inside_time = inside_time[-1]-inside_time[0]
                        outside_time = outside_time[-1]-outside_time[0]
                        inside_time_all.append(inside_time)
                        outside_time_all.append(outside_time)
                        axs.plot(inside_time, outside_time, '.')
        axs.set_xscale('log')
        axs.set_yscale('log')
        cor = np.corrcoef(np.log10(inside_time_all), np.log10(outside_time_all))
        print(cor)

    def improvement_over_time(self, plot_individual=False, plot_pts=False, set_log=True):
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')

        all_data = self.load_trajectories()
        all_results = []

        for log in list(all_data.keys()):
            params = {
                'log': log,
                'o_t':[],
                'o_d':[],
                'mean_x':[]
            }
            temp = all_data[log]
            do = temp['do']
            data = temp['data']
            for key in list(do.keys())[:-1]:
                t = do[key].seconds.to_numpy()
                del_t = t[-1]-t[0]
                params['o_t'].append(del_t)
                x = do[key].ft_posx.to_numpy()
                y = do[key].ft_posy.to_numpy()
                _,dis = fn.path_length(x,y)
                params['o_d'].append(dis)
                params['mean_x'].append(np.mean(x)-x[0]) # average x position
            all_results.append(params)
            if plot_individual:
                fig, axs = plt.subplots(1,2, figsize=(6,3))
                pl.plot_trajectory(data, axs[0])
                pl.plot_trajectory_odor(data, axs[0])
                axs[1].plot(params['mean_x'], 'o')


        o_t = []
        o_d = []
        for params in all_results:
            o_t.append(params['o_t'])
            o_d.append(params['o_d'])
        o_t = fn.list_matrix_nan_fill(o_t)
        o_d = fn.list_matrix_nan_fill(o_d)

        # plot the results
        distance_color = pl.lighten_color(pl.outside_color,1.0)
        time_color = pl.lighten_color(pl.outside_color,2.5)
        fig,axs = plt.subplots(1,1, figsize=(2.5,2))
        fig2,axs2 = plt.subplots(1,1, figsize=(2.5,2))

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
        fig.tight_layout()
        fig2.tight_layout()
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

        # statistics
        pre = o_d[:,0]
        post = np.nanmean(o_d[:,1:15], axis=1)
        print(stats.ttest_rel(pre, post, nan_policy='omit'))
        return pre, post

    def time_distance_out_consecutive_trajectories(self):
        """
        make three histograms showing the time and distance outside for outside
        trajectories and the number of consecutive outside trajectories.
        """
        all_data = self.load_trajectories()
        sheet = self.sheet.log
        outside_time_all = []
        outside_dist_all = []
        cons_outside = []
        for i, log in enumerate(sheet):
            print(log)
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']
            count = 0
            if len(do)>1:
                for key in list(do.keys())[1:]:
                    temp = do[key]
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()
                    t = temp.seconds.to_numpy()
                    if np.abs(x[-1]-x[0])<2:
                        count+=1
                        t = t-t[0]
                        _,l = fn.path_length(x,y)
                        outside_time_all.append(t[-1])
                        outside_dist_all.append(l)
                cons_outside.append(count)
        print('number of trajectories = ', len(cons_outside))
        print('number of outside trajectories = ', len(outside_time_all))
        fig, axs = plt.subplots(1,3, figsize = (15, 5))
        sns.histplot(outside_time_all,ax=axs[0], element="step",log_scale=True, fill=False)
        axs[0].set_xlabel('time outside plume (s)')
        sns.histplot(outside_dist_all,ax=axs[1], element="step",log_scale=True, fill=False)
        axs[1].set_xlabel('distance traveled (mm)')
        sns.histplot(cons_outside,ax=axs[2],bins=20, element="step", fill=False)
        axs[2].set_xlabel('# of consectutive returns')

        fig.savefig(os.path.join(self.figurefol, 'time_distance_outside_consectuive.pdf'))

    def plot_example_trajectory(self, inside=False, savefig=False, pts = 10000):
        all_data = self.load_trajectories()
        if self.experiment == '15':
            if inside:
                ilog = 3
                ibout = 4
                log = self.sheet.iloc[ilog].log
                temp = all_data[log]
                di = temp['di'][ibout]
                df = temp['data']
                fig_example, axs = plt.subplots(1, 1)
                axs.plot(di.ft_posx, di.ft_posy)
                axs.axis('equal')
                savename = 'example_'+log.replace('.log','')+'_inside_'+str(ibout)+'.pdf'
            else:
                ilog=3
                ibout = 12
                log = self.sheet.iloc[ilog].log
                temp = all_data[log]
                do = temp['do'][ibout]
                df = temp['data']
                fig_example, axs = plt.subplots(1, 1)
                axs.plot(do.ft_posx, do.ft_posy)
                axs.axis('equal')
                savename = 'example_'+log.replace('.log','')+'_outside_'+str(ibout)+'.pdf'

        fig, axs = plt.subplots(1,1)
        axs.plot(df.ft_posx, df.ft_posy)

        if savefig:
            fig_example.savefig(os.path.join(self.figurefol, savename))

    def behavioral_metrics_in_out_time(self, plot_individual=False):
        """
        function for plotting the inside and outside times as a histogram.
        For this analysis we will just look at the inside and outside
        trajectories that returned to the edge. Look at the aggregate of all
        trajectories, pooled across animals.

        possible modifications to script:
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        t_in = []
        t_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # inside calculations: time
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    t_in_temp = (df_i.seconds.iloc[-1]-df_i.seconds.iloc[0])
                    t_in.append(t_in_temp)

            # outside calculations: time
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    t_out_temp = (df_o.seconds.iloc[-1]-df_o.seconds.iloc[0])
                    t_out.append(t_out_temp)

            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))

        metrics = {
        't_in': t_in,
        't_out': t_out,
        }

        max_x = 500
        # plot the inside and outside time for individual animals log scale on x
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        sns.histplot(x=metrics['t_in'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['t_out'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[0].axvline(np.mean(metrics['t_in']), color=pl.inside_color, linestyle='--')
        axs[0].axvline(np.mean(metrics['t_out']), color=pl.outside_color, linestyle='--')
        axs[0].set_xlabel('time (s)')

        # plot the inside and outside time for individual animals linear scale on x
        sns.histplot(x=metrics['t_in'], element="step", fill=False, ax=axs[1], bins=20, binrange=(0,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['t_out'], element="step", fill=False, ax=axs[1], bins=20, binrange=(0,max_x), color=pl.outside_color)
        axs[1].axvline(np.mean(metrics['t_in']), color=pl.inside_color, linestyle='--')
        axs[1].axvline(np.mean(metrics['t_out']), color=pl.outside_color, linestyle='--')
        axs[1].set_xlabel('time (s)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_time.pdf'), bbox_inches='tight', transparent=True)

        return metrics

    def behavioral_metrics_in_out_time_paired(self, plot_individual=False):
        """
        function for plotting the average inside and outside times for each
        animal as a paired plot

        duture modifications
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        times_in = []
        times_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # add time for inside trajectories that return to the edge
            t_in = []
            returns = 0
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    returns+=1
                    t_in_temp = (df_i.seconds.iloc[-1]-df_i.seconds.iloc[0])
                    t_in.append(t_in_temp)

            # add time for outside trajectories that return to the edge
            t_out = []
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    t_out_temp = (df_o.seconds.iloc[-1]-df_o.seconds.iloc[0])
                    t_out.append(t_out_temp)

            # only count trial is it returned to the edge at least twice (n=3 for inside)
            if returns>=2:
                times_in.append(np.mean(t_in))
                times_out.append(np.mean(t_out))

            # plot individual traces
            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        fig, axs = plt.subplots(1,1, figsize = (1,2))
        axs = pl.paired_plot(axs, times_in, times_out, color1=pl.inside_color, color2=pl.outside_color, log = True, alpha=0.1)
        axs.set_xticks([0, 1])
        axs.set_xticklabels(['in', 'out'])

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_time_paired.pdf'), bbox_inches='tight', transparent=True)

    def behavioral_metrics_in_out_upwind_dist(self, plot_individual=False):
        """
        function for plotting the inside and outside upwind distances as a histogram.
        For this analysis we will just look at the inside and outside
        trajectories that returned to the edge. Look at the aggregate of all
        trajectories, pooled across animals.

        possible modifications to script:
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        y_in = []
        y_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # inside calculations: time
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    y_in_temp = (df_i.ft_posy.iloc[-1]-df_i.ft_posy.iloc[0])
                    y_in.append(y_in_temp)

            # outside calculations: time
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    y_out_temp = (df_o.ft_posy.iloc[-1]-df_o.ft_posy.iloc[0])
                    y_out.append(y_out_temp)

            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))

        metrics = {
        'y_in': y_in,
        'y_out': y_out,
        }

        max_x = 30
        # plot the inside and outside time for individual animals log scale on x
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        sns.histplot(x=metrics['y_in'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['y_out'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[0].axvline(np.mean(metrics['y_in']), color=pl.inside_color, linestyle='--')
        axs[0].axvline(np.mean(metrics['y_out']), color=pl.outside_color, linestyle='--')
        axs[0].set_xlabel('upwind distance (mm)')

        # plot the inside and outside time for individual animals linear scale on x
        sns.histplot(x=metrics['y_in'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['y_out'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[1].axvline(np.mean(metrics['y_in']), color=pl.inside_color, linestyle='--')
        axs[1].axvline(np.mean(metrics['y_out']), color=pl.outside_color, linestyle='--')
        axs[1].set_xlabel('upwind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_y_dist.pdf'), bbox_inches='tight', transparent=True)

        return metrics

    def behavioral_metrics_in_out_upwind_dist_paired(self, plot_individual=False):
        """
        function for plotting the average inside and outside times for each
        animal as a paired plot

        duture modifications
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        ydist_in = []
        ydist_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # add time for inside trajectories that return to the edge
            y_in = []
            returns = 0
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    returns+=1
                    y_in_temp = (df_i.ft_posy.iloc[-1]-df_i.ft_posy.iloc[0])
                    y_in.append(y_in_temp)

            # add time for outside trajectories that return to the edge
            y_out = []
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    y_out_temp = (df_o.ft_posy.iloc[-1]-df_o.ft_posy.iloc[0])
                    y_out.append(y_out_temp)

            # only count trial is it returned to the edge at least twice (n=3 for inside)
            if returns>=2:
                ydist_in.append(np.mean(y_in))
                ydist_out.append(np.mean(y_out))

            # plot individual traces
            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        fig, axs = plt.subplots(1,1, figsize = (1,2))
        axs = pl.paired_plot(axs, ydist_in, ydist_out, color1=pl.inside_color, color2=pl.outside_color, alpha=0.1)
        axs.set_xticks([0, 1])
        axs.set_xticklabels(['in', 'out'])
        axs.set_ylabel('average upwind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_y_dist_paired.pdf'), bbox_inches='tight', transparent=True)
        return ydist_in, ydist_out

    def behavioral_metrics_in_out_crosswind_dist(self, plot_individual=False):
        """
        function for plotting the inside and outside crosswind as a histogram.
        For this analysis we will just look at the inside and outside
        trajectories that returned to the edge. Look at the aggregate of all
        trajectories, pooled across animals.

        possible modifications to script:
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        x_in = []
        x_out = []
        x_del_in = []
        x_del_out = []

        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # inside calculations: x distance
            for key in list(di.keys()):
                df_i = di[key]
                df_i.ft_posx = np.abs(df_i.ft_posx-df_i.ft_posx.iloc[0])
                df_i.ft_posy = df_i.ft_posy-df.ft_posy.iloc[0]
                if return_to_edge(df_i):
                    x_steps = np.abs(np.gradient(df_i.ft_posx))
                    x_in_temp = np.sum(x_steps)
                    x_in.append(x_in_temp)
                    x_del_in.append(np.max(df_i.ft_posx))


            # outside calculations: x distance
            for key in list(do.keys()):
                df_o = do[key]
                df_o.ft_posx = np.abs(df_o.ft_posx-df_o.ft_posx.iloc[0])
                df_o.ft_posy = df_o.ft_posy-df_o.ft_posy.iloc[0]
                if return_to_edge(df_o):
                    x_steps = np.abs(np.gradient(df_o.ft_posx))
                    x_out_temp = np.sum(x_steps)
                    x_out.append(x_out_temp)
                    x_del_out.append(np.max(df_o.ft_posx))

            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        metrics = {
        'x_in': x_in,
        'x_out': x_out,
        'x_del_in': x_del_in,
        'x_del_out': x_del_out
        }

        max_x = 1000
        # plot the inside and outside cumulative x distance with a log scale
        fig, axs = plt.subplots(1,4, figsize=(16,4))
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[0].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[0].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[0].set_xlabel('crosswind pathlen. (mm)')
        print('x_in mean = ', np.mean(metrics['x_in']), 'x_out mean = ', np.mean(metrics['x_out']))
        # plot the inside and outside cumulative x distance with a linear scale
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[1].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[1].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[1].set_xlabel('crosswind pathlen. (mm)')

        max_x = 500
        # plot the inside and outside maximum x distance with a log scale
        sns.histplot(x=metrics['x_del_in'], element="step", fill=False, ax=axs[2], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['x_del_out'], element="step", fill=False, ax=axs[2], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[2].axvline(np.mean(metrics['x_del_in']), color=pl.inside_color, linestyle='--')
        axs[2].axvline(np.mean(metrics['x_del_out']), color=pl.outside_color, linestyle='--')
        axs[2].set_xlabel('max crosswind dist (mm)')

        # plot the inside and outside maximum x distance with a linear scale
        sns.histplot(x=metrics['x_del_in'], element="step", fill=False, ax=axs[3], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['x_del_out'], element="step", fill=False, ax=axs[3], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[3].axvline(np.mean(metrics['x_del_in']), color=pl.inside_color, linestyle='--')
        axs[3].axvline(np.mean(metrics['x_del_out']), color=pl.outside_color, linestyle='--')
        axs[3].set_xlabel('max crosswind dist (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_x_dist.pdf'), bbox_inches='tight', transparent=True)

        return metrics

    def behavioral_metrics_in_out_crosswind_dist_nc(self, plot_individual=False):
        """
        function for plotting the inside and outside crosswind as a histogram.
        For this analysis we will just look at the inside and outside
        trajectories that returned to the edge. Look at the aggregate of all
        trajectories, pooled across animals.

        possible modifications to script:
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        x_in = []
        x_out = []
        x_del_in = []
        x_del_out = []

        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # inside calculations: x distance
            for key in list(di.keys()):
                df_i = di[key]
                df_i.ft_posx = np.abs(df_i.ft_posx-df_i.ft_posx.iloc[0])
                df_i.ft_posy = df_i.ft_posy-df.ft_posy.iloc[0]
                if return_to_edge(df_i):
                    x_in_temp = np.mean(df_i.ft_posx)
                    x_in.append(x_in_temp)
                    x_del_in.append(np.max(df_i.ft_posx))


            # outside calculations: x distance
            for key in list(do.keys()):
                df_o = do[key]
                df_o.ft_posx = np.abs(df_o.ft_posx-df_o.ft_posx.iloc[0])
                df_o.ft_posy = df_o.ft_posy-df_o.ft_posy.iloc[0]
                if return_to_edge(df_o):
                    x_out_temp = np.mean(df_o.ft_posx)
                    x_out.append(x_out_temp)
                    x_del_out.append(np.max(df_o.ft_posx))

            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        metrics = {
        'x_in': x_in,
        'x_out': x_out,
        'x_del_in': x_del_in,
        'x_del_out': x_del_out
        }

        max_x = 1000
        # plot the inside and outside cumulative x distance with a log scale
        fig, axs = plt.subplots(1,4, figsize=(16,4))
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[0].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[0].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[0].set_xlabel('crosswind pathlen. (mm)')
        print('x_in mean = ', np.mean(metrics['x_in']), 'x_out mean = ', np.mean(metrics['x_out']))
        # plot the inside and outside cumulative x distance with a linear scale
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[1].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[1].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[1].set_xlabel('crosswind pathlen. (mm)')

        max_x = 500
        # plot the inside and outside maximum x distance with a log scale
        sns.histplot(x=metrics['x_del_in'], element="step", fill=False, ax=axs[2], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['x_del_out'], element="step", fill=False, ax=axs[2], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[2].axvline(np.mean(metrics['x_del_in']), color=pl.inside_color, linestyle='--')
        axs[2].axvline(np.mean(metrics['x_del_out']), color=pl.outside_color, linestyle='--')
        axs[2].set_xlabel('max crosswind dist (mm)')
        print('x_in del = ', np.mean(metrics['x_del_in']), 'x_out del = ', np.mean(metrics['x_del_out']))

        # plot the inside and outside maximum x distance with a linear scale
        sns.histplot(x=metrics['x_del_in'], element="step", fill=False, ax=axs[3], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['x_del_out'], element="step", fill=False, ax=axs[3], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[3].axvline(np.mean(metrics['x_del_in']), color=pl.inside_color, linestyle='--')
        axs[3].axvline(np.mean(metrics['x_del_out']), color=pl.outside_color, linestyle='--')
        axs[3].set_xlabel('max crosswind dist (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_x_dist_nc.pdf'), bbox_inches='tight', transparent=True)

        return metrics

    def behavioral_metrics_in_out_crosswind_dist_paired(self, plot_individual=False):
        """
        function for plotting the average inside and outside times for each
        animal as a paired plot

        duture modifications
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        xdist_in = []
        xdist_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # add time for inside trajectories that return to the edge
            x_in = []
            returns = 0
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    returns+=1
                    x_steps = np.abs(np.gradient(df_i.ft_posx))
                    x_in_temp = np.sum(x_steps)
                    x_in.append(x_in_temp)

            # add time for outside trajectories that return to the edge
            x_out = []
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    x_steps = np.abs(np.gradient(df_o.ft_posx))
                    x_out_temp = np.sum(x_steps)
                    x_out.append(x_out_temp)

            # only count trial if it returned to the edge at least twice (n=3 for inside)
            if returns>=2:
                xdist_in.append(np.mean(x_in))
                xdist_out.append(np.mean(x_out))

            # plot individual traces
            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        fig, axs = plt.subplots(1,1, figsize = (1,2))
        axs = pl.paired_plot(axs, xdist_in, xdist_out, color1=pl.inside_color, color2=pl.outside_color, alpha=0.1)
        axs.set_xticks([0, 1])
        axs.set_xticklabels(['in', 'out'])
        axs.set_ylabel('average crosswind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_x_dist_paired.pdf'), bbox_inches='tight', transparent=True)
        return ydist_in, ydist_out

    def behavioral_metrics_in_out_crosswind_dist_paired_nc(self, plot_individual=False):
        """
        function for plotting the average inside and outside times for each
        animal as a paired plot
        nc = non-cumulative

        duture modifications
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        xdist_in = []
        xdist_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # add time for inside trajectories that return to the edge
            x_in = []
            returns = 0
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    returns+=1
                    x_steps = np.abs(df_i.ft_posx-df_i.ft_posx.iloc[0])
                    x_in_temp = np.mean(x_steps)
                    x_in.append(x_in_temp)

            # add time for outside trajectories that return to the edge
            x_out = []
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    x_steps = np.abs(df_o.ft_posx-df_o.ft_posx.iloc[0])
                    x_out_temp = np.mean(x_steps)
                    x_out.append(x_out_temp)

            # only count trial if it returned to the edge at least twice (n=3 for inside)
            if returns>=2:
                xdist_in.append(np.mean(x_in))
                xdist_out.append(np.mean(x_out))

            # plot individual traces
            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        fig, axs = plt.subplots(1,1, figsize = (1,2))
        axs = pl.paired_plot(axs, xdist_in, xdist_out, color1=pl.inside_color, color2=pl.outside_color, alpha=0.1)
        axs.set_xticks([0, 1])
        axs.set_xticklabels(['in', 'out'])
        axs.set_ylabel('average crosswind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_x_dist_paired_nc.pdf'), bbox_inches='tight', transparent=True)

    def behavioral_metrics_in_out_pathlength(self, plot_individual=False):
        """
        function for plotting the inside and outside pathlength.
        For this analysis we will just look at the inside and outside
        trajectories that returned to the edge. Look at the aggregate of all
        trajectories, pooled across animals.

        possible modifications to script:
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        p_in = []
        p_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # inside calculations: x distance
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    p_in_temp = fn.path_length(df_i.ft_posx.to_numpy(), df_i.ft_posy.to_numpy())
                    p_in.append(p_in_temp)

            # outside calculations: x distance
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    p_out_temp = fn.path_length(df_i.ft_posx.to_numpy(), df_i.ft_posy.to_numpy())
                    p_out.append(p_out_temp)

            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        metrics = {
        'p_in': p_in,
        'p_out': p_out,
        }

        max_x = 1000
        # plot the inside and outside time for individual animals log scale on x
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[0].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[0].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[0].set_xlabel('cumulative crosswind distance (mm)')

        # plot the inside and outside time for individual animals linear scale on x
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[1].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[1].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[1].set_xlabel('cumulative crosswind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_path.pdf'), bbox_inches='tight', transparent=True)

        return metrics

    def behavioral_metrics_in_out_pathlength_paired(self, plot_individual=False):
        """
        function for plotting the average inside and outside times for each
        animal as a paired plot

        future modifications
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        xdist_in = []
        xdist_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # add time for inside trajectories that return to the edge
            x_in = []
            returns = 0
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    returns+=1
                    x_steps = np.abs(np.gradient(df_i.ft_posx))
                    x_in_temp = np.sum(x_steps)
                    x_in.append(x_in_temp)

            # add time for outside trajectories that return to the edge
            x_out = []
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    x_steps = np.abs(np.gradient(df_o.ft_posx))
                    x_out_temp = np.sum(x_steps)
                    x_out.append(x_out_temp)

            # only count trial if it returned to the edge at least twice (n=3 for inside)
            if returns>=2:
                xdist_in.append(np.mean(x_in))
                xdist_out.append(np.mean(x_out))

            # plot individual traces
            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        fig, axs = plt.subplots(1,1, figsize = (1,2))
        axs = pl.paired_plot(axs, xdist_in, xdist_out, color1=pl.inside_color, color2=pl.outside_color, alpha=0.1)
        axs.set_xticks([0, 1])
        axs.set_xticklabels(['in', 'out'])
        axs.set_ylabel('average crosswind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_x_dist_paired.pdf'), bbox_inches='tight', transparent=True)
        return ydist_in, ydist_out


%matplotlib
a = average_trajectory(directory='Andy', experiment='45')
results = a.improvement_over_time(plot_individual=False, plot_pts=True, set_log=True)


# %%
%matplotlib
results
fig,axs = plt.subplots(1,1)
for i,j in zip(['angle_log'],['angle']):
    results[i] = np.log(results[j])
sns.violinplot(data=results, x='a',y='angle',hue='dir', ax=axs, cut=0.1)

# %%

# %%
group_figure_folder = os.path.join('/Volumes/Andy/GitHub/edge-tracking', 'figures/average_trajectories/group_figures')
#group_figure_folder = os.path.join('/Volumes/LACIE/edge-tracking', 'figures/average_trajectories/group_figures')

def plot_headings():
    """
    plot headings for 0,45,90
    """
    fig_cart, axs_cart = plt.subplots(1,2)
    fig_upwind, axs_upwind = plt.subplots(1,2)
    experiments = ['0','45','90']
    for j, exp in enumerate(experiments):
        a = average_trajectory(directory='Andy', experiment=exp)
        df, exit_heading, entry_heading, trajectories = a.load_outside_dp_segments()
        sns.histplot(exit_heading, stat='probability', ax=axs_cart[0], binrange=(-np.pi, np.pi), bins=20, fill=False, color = sns.color_palette()[j], element='step')
        sns.histplot(entry_heading, stat='probability', ax=axs_cart[1], binrange=(-np.pi, np.pi), bins=20, fill=False, color = sns.color_palette()[j], element='step')

        exit_heading = fn.conv_cart_upwind(np.array(exit_heading))
        entry_heading = fn.conv_cart_upwind(np.array(entry_heading))

        sns.histplot(exit_heading, stat='probability', ax=axs_upwind[0], binrange=(-np.pi, np.pi), bins=20, fill=False, color = sns.color_palette()[j], element='step')
        sns.histplot(entry_heading, stat='probability', ax=axs_upwind[1], binrange=(-np.pi, np.pi), bins=20, fill=False, color = sns.color_palette()[j], element='step')

def plot_efficiencies():
    """
    plot efficiencies for 0,45,90
    """
    sns.set(font="Arial")
    sns.set(font_scale=0.6)
    sns.set_style('white')
    fig, axs = plt.subplots(3,2, figsize = (5,5))
    experiments = ['0','45','90']
    for j, exp in enumerate(experiments):
        a = average_trajectory(directory='Andy', experiment=exp)
        a.compare_fictive_real_efficiency(axs[j,0], axs[j,1])
        axs[j,0].set_xlim(0,250)
        axs[j,0].set_ylim(-100,12500)
        axs[j,0].plot([0,250], [0,500], linestyle='dashed', color='k')
        if j in [0,1]:
            axs[j,0].set_ylabel('')
            axs[j,0].set_xlabel('')
            axs[j,1].set_ylabel('')
            axs[j,1].set_xlabel('')
            axs[j,1].get_shared_x_axes().get_siblings(axs[j,1])[3].set_ylabel('')
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(os.path.join(group_figure_folder,'efficiencies.pdf'))
    return axs[j,1]

def plot_average_trajectories():
    experiments = ['0','15','45','T']
    fig, axs = plt.subplots(1,4,figsize=(20,5), sharey=True, subplot_kw={'aspect':'equal'})
    for j, exp in enumerate(experiments):
        if exp == 'T':
            angle = 90
        else:
            angle = int(exp)
        angle_from_horizontal = np.pi/2 - angle*np.pi/180

        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = average_trajectory(exp).load_average_trajectories()

        for i in np.arange(len(animal_avg_x_in)):
            axs[j].plot(animal_avg_x_in[i], animal_avg_y_in[i], 'r', alpha=0.2)
        axs[j].plot(np.mean(animal_avg_x_in, axis=0), np.mean(animal_avg_y_in, axis=0), 'r')
        exit_x = np.mean(animal_avg_x_in, axis=0)[-1]
        exit_y = np.mean(animal_avg_y_in, axis=0)[-1]

        for i in np.arange(len(animal_avg_x_out)):
            axs[j].plot(animal_avg_x_out[i]+exit_x, animal_avg_y_out[i]+exit_y, 'k', alpha=0.2)
        axs[j].plot(np.mean(animal_avg_x_out+exit_x, axis=0), np.mean(animal_avg_y_out+exit_y, axis=0), 'k')

        # draw the plume boundary line at the appropriate angle
        max_y_out = np.max(animal_avg_y_out)+exit_y
        max_y_in = np.max(animal_avg_y_in)
        max_y = np.max((max_y_out,max_y_in))
        axs[j].plot([0,max_y/np.tan(angle_from_horizontal)], [0, max_y], color='k', linestyle='dashed')
        axs[j].axis('equal')
    fig.savefig(os.path.join(group_figure_folder, 'average_trajectories.pdf'))

def make_angle_distribution_plots():
    """
    for each experiment create an average trajectory object and run the script for angle distributions
    """
    experiments = ['0','15','45','T']
    for j, exp in enumerate(experiments):
        if exp=='T':
            ang=90
        else:
            ang=int(exp)
        average_trajectory(exp).inbound_outbound_angle_distribution(ang)

def paired_inbound_outbound_plots():
    """
    for each experiment create a paired plot for the outside inbound and outbound angle
    """
    experiments = ['0','15','45','T']
    colors = sns.color_palette()
    fig,axs = plt.subplots(1,4,figsize=(4,2), sharex=True, sharey=True)
    for j, exp in enumerate(experiments):
        if exp=='T':
            ang=90
        else:
            ang=int(exp)
        angles_outside = average_trajectory(directory='M1', experiment=exp).average_inbound_outbound_angle(ang)
        axs[j] = pl.paired_plot(axs[j],angles_outside[:,0], angles_outside[:,1], color1=colors[0], color2=colors[1])
        axs[j].set_yticks([45,90])
    fig.savefig(os.path.join(group_figure_folder,'inbound_outbound_angles.pdf'))
    fig,axs = plt.subplots(1,4,figsize=(4,2), sharex=True, sharey=True)
    for j, exp in enumerate(experiments):
        if exp=='T':
            ang=90
        else:
            ang=int(exp)
        lengths_outside = average_trajectory(directory='M1', experiment=exp).average_inbound_outbound_path_length(ang)
        axs[j] = pl.paired_plot(axs[j],lengths_outside[:,0], lengths_outside[:,1], color1=colors[0], color2=colors[1], log=True)
    fig.savefig(os.path.join(group_figure_folder,'inbound_outbound_path_length.pdf'))
    fig,axs = plt.subplots(1,4,figsize=(4,2), sharex=True, sharey=True)
    for j, exp in enumerate(experiments):
        if exp=='T':
            ang=90
        else:
            ang=int(exp)
        tor_outside = average_trajectory(directory='M1', experiment=exp).average_inbound_outbound_tortuosity(ang)
        axs[j] = pl.paired_plot(axs[j],tor_outside[:,0], tor_outside[:,1], color1=colors[0], color2=colors[1], log=False)
    fig.savefig(os.path.join(group_figure_folder,'inbound_outbound_tortuosity.pdf'))

def example_trajectory():
    import matplotlib.patches as patches
    """
    used to make a supplemental figure for showing how the trajectories are broken up

    """

    log = '09082020-132608_Fly9_CantonS_001.log'
    df = average_trajectory(directory='M1', experiment = '0').load_single_trajectory(log)

    df = fn.consolidate_in_out(df)
    df = fn.calculate_speeds(df)
    _, df_stop = fn.find_stops(df)
    df['curvature'] = df['curvature']*df_stop['stop']
    df_rolling = df.rolling(10).mean()
    df_rolling.instrip = df.instrip
    df_odor = df.where(df.instrip==True)
    df_odor_rolling = df_rolling.where(df_rolling.instrip==True)
    fig_traj, axs_traj = plt.subplots(1,2)
    for i in np.arange(len(axs_traj)):
        pl.plot_vertical_edges(df,axs_traj[i])
        pl.plot_trajectory(df, axs_traj[i])
        pl.plot_trajectory_odor(df, axs_traj[i])
        if i==0:
            axs_traj[i].axis('equal')
            axs_traj[i].set_xticks([-100,0,100])



    fig, axs = plt.subplots(1,1)
    d, di, do = fn.inside_outside(df)
    bout1 = 29
    bout2 = 42
    for i in np.arange(bout1,bout2):
        if i%2==1:
            color = pl.inside_color
        else:
            color = pl.outside_color
        df_temp = d[i]
        x = df_temp.ft_posx
        y = df_temp.ft_posy
        axs.plot(x,y, color)
    axs.axis('equal')
    fig.savefig(os.path.join(group_figure_folder, 'example_trajectory_zoom.pdf'))

    ix1 = d[bout1].index[0]
    ix2 = d[bout2-1].index[-1]
    time = df.iloc[ix1:ix2].seconds.to_numpy()
    t0 = time[0]
    time = time-t0
    x_position = df.iloc[ix1:ix2].ft_posx.to_numpy()
    x_position_odor = df_odor.iloc[ix1:ix2].ft_posx.to_numpy()
    y_position = df.iloc[ix1:ix2].ft_posy.to_numpy()
    y_position_odor = df_odor.iloc[ix1:ix2].ft_posy.to_numpy()
    y_velocity = df_rolling.iloc[ix1:ix2].yv.to_numpy()
    y_velocity_odor = df_odor_rolling.iloc[ix1:ix2].yv.to_numpy()
    curvature = df_rolling.iloc[ix1:ix2].curvature
    curvature_odor = df_odor_rolling.iloc[ix1:ix2].curvature

    # draw an inset box
    x_box = np.max(x_position)-np.min(x_position)
    y_box = np.max(y_position)-np.min(y_position)
    rect = patches.Rectangle((np.min(x_position), np.min(y_position)), x_box, y_box, linewidth=1, edgecolor='r', facecolor='none')
    axs_traj[0].add_patch(rect)
    rect1 = patches.Rectangle((np.min(x_position), np.min(y_position)), x_box, y_box, linewidth=1, edgecolor='r', facecolor='none')
    axs_traj[1].add_patch(rect1)
    fig_traj.savefig(os.path.join(group_figure_folder, 'example_trajectory.pdf'))

    fig, axs = plt.subplots(3,1)
    axs[0].plot(time, x_position)
    axs[0].plot(time, x_position_odor)
    axs[1].plot(time, y_position)
    axs[1].plot(time, y_position_odor)
    axs[2].plot(time, y_velocity)
    axs[2].plot(time, y_velocity_odor)

    for i in np.arange(bout1,bout2):
        if i%2==1:
            df_temp = d[i]
            t1 = df_temp.iloc[0].seconds-t0
            t2 = df_temp.iloc[-1].seconds-t0
            axs[0].plot([t1,t1],[np.min(x_position), np.max(x_position)],'k')
            axs[0].plot([t2,t2],[np.min(x_position), np.max(x_position)],'k')

    fig.tight_layout()
    fig.savefig(os.path.join(group_figure_folder, 'example_trajectory_metrics.pdf'))

    # make a plot showing the douglas peucker simplification for an outside trajectory
    fig, axs = plt.subplots(1,1,figsize=(2,2))
    bout1 =32
    bout2 = 33
    x_offset = 16
    for i in np.arange(bout1,bout2):
        if i%2==1:
            color = pl.inside_color
            alpha=1
        else:
            color = pl.outside_color
            alpha=1
        df_temp = d[i]
        x = df_temp.ft_posx.to_numpy()
        y = df_temp.ft_posy.to_numpy()
        # find start points
        if i==bout1:
            x0=x[0]
            y0=y[0]
        x = x-x0 # move trajectories to origin
        y = y-y0
        axs.plot(x,y, color, alpha=alpha)

        # plot the RDP simplification offset to the right
        simplified,_,_,_=fn.rdp_simp_heading_angles_len(x,y, epsilon=0)
        axs.plot(simplified[:,0]+x_offset, simplified[:,1], 'k', alpha=alpha, zorder=0)
        axs.plot(simplified[:,0]+x_offset, simplified[:,1], '.',color=color, alpha=alpha, zorder=0)

    # find end points
    x1=x[-1]
    y1=y[-1]

    # plot some plume edge lines
    axs.plot([0,0],[0,y1],color='k', linestyle='dashed')
    axs.plot([x_offset, x_offset], [0,y1], color='k', linestyle='dashed')

    # make a scalebar
    l=5
    axs.plot([0,l], [-3,-3], color='k')
    axs.text(0,-5,str(l)+' mm', fontsize=7, font='Arial')

    # plot mods
    axs.axis('equal')
    axs.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(group_figure_folder, 'example_trajectory_zoom_RDP.pdf'))

def plot_crossovers():
    at = average_trajectory('0')
    all_data = at.load_trajectories()
    sheet = at.sheet.log
    fig, axs = plt.subplots(1,2)

    #@jit(nopython=True)

    avg_x_out_pre, avg_y_out_pre = [],[]
    avg_x_in, avg_y_in = [],[]
    avg_x_out_post, avg_y_out_post = [],[]
    pts = 10000
    exit_count = 0
    return_count = 0
    ix_exit, ix_return = [],[]
    j=0
    for i, log in enumerate(sheet):
        print(log)
        df = all_data[log]['data']
        di = all_data[log]['di']
        do = all_data[log]['do']
        # iterate through the inside trajectories to find ones that cross over
        for key in list(di.keys())[1:]: # skip the first
            temp_in = di[key]
            x = temp_in.ft_posx.to_numpy()
            if np.abs(x[-1]-x[0])>20:
                # cross over found
                if (key+1 in list(do.keys())) and (key-1 in list(do.keys())):
                    print('enter')
                    temp_out_pre = do[key-1]
                    temp_in = di[key]
                    temp_out_post = do[key+1]

                    x_out_pre, y_out_pre = fn.center_x_y(temp_out_pre)
                    x_in, y_in = fn.center_x_y(temp_in)
                    x_out_post, y_out_post = fn.center_x_y(temp_out_post)
                    # flip trajectories if necessary
                    if np.mean(x_out_post)<0:
                        x_out_pre = -x_out_pre
                        x_in = -x_in
                        x_out_post = -x_out_post
                    avg_x_out_pre, avg_y_out_pre = fn.interp_append_x_y(x_out_pre,y_out_pre, avg_x_out_pre, avg_y_out_pre)
                    avg_x_out_post, avg_y_out_post = fn.interp_append_x_y(x_out_post,y_out_post, avg_x_out_post, avg_y_out_post)
                    avg_x_in, avg_y_in = fn.interp_append_x_y(x_in,y_in, avg_x_in, avg_y_in)

                    if np.abs(x_out_post[-1]-x_out_post[0])<0.5: # post returns to the edge
                        return_count += 1
                        ix_return.append(j)

                    else:
                        exit_count += 1
                        ix_exit.append(j)

                    j+=1
    averages = [avg_x_out_pre, avg_y_out_pre, avg_x_out_post, avg_y_out_post, avg_x_in, avg_y_in]
    averages_select = [avg_x_out_pre, avg_y_out_pre, avg_x_out_post, avg_y_out_post, avg_x_in, avg_y_in]
    for ax, condition in enumerate(['return', 'exit']):
        if condition == 'return':
            selection_ix = ix_return
        elif condition == 'exit':
            selection_ix = ix_exit
        else:
            selection_ix = np.arange(len(avg_x_in))

        # make selection

        for i, avg in enumerate(averages):
            avg = np.array(avg)[selection_ix]
            averages_select[i]=avg
        [avg_x_out_pre, avg_y_out_pre, avg_x_out_post, avg_y_out_post, avg_x_in, avg_y_in] = averages_select
        # entry and exit points to align average plots
        x_enter, y_enter = np.mean(avg_x_in, axis=0)[0]-25, np.mean(avg_y_out_pre, axis=0)[-1]
        x_exit, y_exit = np.mean(avg_x_in, axis=0)[-1]-25, np.mean(avg_y_in, axis=0)[-1]

        # plot_averages
        axs[ax].plot(np.mean(avg_x_in, axis=0)+x_enter, np.mean(avg_y_in, axis=0), 'r')
        axs[ax].plot(np.mean(avg_x_out_post, axis=0)+x_exit, np.mean(avg_y_out_post, axis=0)+y_exit, 'k')
        axs[ax].plot(np.mean(avg_x_out_pre, axis=0)+x_enter, np.mean(avg_y_out_pre, axis=0)-y_enter, 'k')
        axs[ax].axis('equal')

        # plot individuals
        for i in np.arange(len(avg_x_in)):
            axs[ax].plot(avg_x_in[i]+x_enter, avg_y_in[i], 'r', alpha=0.1)
            axs[ax].plot(avg_x_out_post[i]+x_exit, avg_y_out_post[i]+y_exit, 'k', alpha = 0.1)
            axs[ax].plot(avg_x_out_pre[i]+x_enter, avg_y_out_pre[i]-y_enter, 'k', alpha = 0.1)

        # draw lines
        axs[ax].plot([-25,-25], [np.min(avg_y_out_pre), np.max(avg_y_out_post)+y_exit], 'k')
        axs[ax].plot([25,25], [np.min(avg_y_out_pre), np.max(avg_y_out_post)+y_exit], 'k')

        #set the exit axes to the return axes
        if condition == 'return':
            xlim = axs[ax].get_xlim()
            ylim = axs[ax].get_ylim()
        if condition == 'exit':
            axs[ax].set_ylim(ylim)
            axs[ax].set_xlim(xlim)

    # calculate fraction
    print('number of returns: '+str(return_count)+'. number of exits: '+str(exit_count)+'. fraction exiting = '+str(exit_count/(exit_count+return_count)))
    fig.savefig(os.path.join(group_figure_folder, 'crossovers.pdf'))

def plot_no_return():
    """
    not sure how useful this will be in the end.  Function for plotting whether
    a plume exit came after an inside bout that was a crossover or same side
    """
    at = average_trajectory('0')
    all_data = at.load_trajectories()
    sheet = at.sheet.log
    avg_x_out_cross, avg_y_out_cross = [],[]
    avg_x_out_same, avg_y_out_same = [],[]
    fig, axs = plt.subplots(1,1,figsize=(2,2))
    for i, log in enumerate(sheet):
        print(log)
        df = all_data[log]['data']
        di = all_data[log]['di']
        do = all_data[log]['do']
        key_out = list(do.keys())[-1]
        do_temp = do[key_out]
        di_temp = di[key_out-1]
        xo,yo = fn.center_x_y(do_temp)
        xi,yi = fn.center_x_y(di_temp)
        _, path_length = fn.path_length(xo,yo)
        if path_length>100:
            # outside trajectory left
            if np.abs(xo[-1]-xo[0])>2:
                # flip if necessary
                if np.mean(xo)<0:
                    print('enter')
                    xi=-xi
                    xo=-xo


                # inside trajectory returned to the same edge
                if np.abs(xi[-1]-xi[0])<2:
                    axs.plot(xo,yo, 'purple', alpha=0.2, linewidth=0.5)
                    avg_x_out_same, avg_y_out_same = fn.interp_append_x_y(xo,yo,avg_x_out_same, avg_y_out_same)

                else:
                    axs.plot(xo,yo, 'cyan', alpha=0.2, linewidth=0.5)
                    avg_x_out_cross, avg_y_out_cross = fn.interp_append_x_y(xo,yo,avg_x_out_cross, avg_y_out_cross)
    axs.plot(np.mean(avg_x_out_cross, axis=0), np.mean(avg_y_out_cross, axis=0), color='cyan', linewidth=1)
    axs.plot(np.mean(avg_x_out_same, axis=0), np.mean(avg_y_out_same, axis=0), color='purple', linewidth=1)
    axs.axis(equal)

def plot_triggered_average_transitions(d1_name, d2_name,experiment = '0', savename = 'transitions.pdf',  t_start = -1, t_end = 3, pts=100):
    savename = d1_name+'_'+d2_name+'_'+experiment+'_'+savename
    at = average_trajectory(directory='M1', experiment = experiment)
    all_data = at.load_trajectories()
    sheet = at.sheet.log
    xv_transition = []
    yv_transition = []
    fig, axs = plt.subplots(1,1,figsize=(4,4))
    for i, log in enumerate(sheet):
        print(log)
        df = all_data[log]['data']
        di = all_data[log]['di']
        do = all_data[log]['do']
        d1 = locals()[d1_name]
        d2 = locals()[d2_name]
        for key in list(d1.keys()):
            if key+1 in list(d2.keys()):
                df_1 = d1[key]
                t1 = df_1.seconds.to_numpy()
                del_t1 = t1[-1]-t1[0]
                upvel_1 = df_1.yv
                xvel_1 = df_1.xv
                x_1, y_1 = fn.center_x_y(df_1)

                df_2 = d2[key+1]
                t2 = df_2.seconds.to_numpy()
                del_t2 = t2[-1]-t2[0]
                upvel_2 = df_2.yv
                xvel_2 = df_2.xv
                x_2, y_2 = fn.center_x_y(df_2)

                # check for plume return
                if experiment == 'T':
                    angle = 90
                else:
                    angle = int(experiment)
                rot_angle = -angle*np.pi/180
                x, y = fn.coordinate_rotation(x_1, y_1, rot_angle)
                if np.abs(x[-1]-x[0])>2:
                    continue


                yv = np.concatenate([upvel_1, upvel_2])
                xv = np.concatenate([xvel_1, xvel_2])
                t_common = np.linspace(t_start, t_end, pts)
                if del_t2>t_end and del_t1>-t_start: # and np.max(np.abs(xv))<100 and np.max(np.abs(yv))<100:
                    t = np.concatenate([t1, t2])
                    t = t-t1[-1]
                    if np.mean(x_1<0):
                        xv=-xv
                    xv_transition.append(fn.interp_crop(t, xv, t_common, pts))
                    yv_transition.append(fn.interp_crop(t, yv, t_common, pts))

    fig, axs = plt.subplots(1,1,figsize=(4,4))
    print('number of transitions:', len(xv_transition))
    xv_mean = np.mean(xv_transition, axis=0)
    sem_xv = stats.sem(xv_transition, axis=0)
    axs.plot(t_common, xv_mean)
    axs.fill_between(t_common, xv_mean-sem_xv, xv_mean+sem_xv, alpha = 0.3)

    yv_mean = np.mean(yv_transition, axis=0)
    sem_yv = stats.sem(yv_transition, axis=0)
    axs.plot(t_common, yv_mean)
    axs.fill_between(t_common, yv_mean-sem_yv, yv_mean+sem_yv, alpha = 0.3)
    axs.set_xlabel('time (s)')
    axs.set_ylabel('velocity (mm/s)')
    axs.set_ylim(-8,8)
    axs.set_xlim(t_start, t_end)

    fig.savefig(os.path.join(group_figure_folder, savename))

def plot_individual_fuzziness():
    at = average_trajectory(directory='LACIE', experiment = '0')
    all_data = at.load_trajectories()
    sheet = at.sheet.log
    example = '09082020-132608_Fly9_CantonS_001.log'
    for i, log in enumerate(sheet):
        print(log)
        if log == example:
            fig, axs = plt.subplots(1,1)
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']
            idx_start = df.index[df.instrip].tolist()[0]
            x0 = df.ft_posx
            y0 = df.ft_posy
            axs = pl.plot_vertical_edges(df, axs)
            axs.plot(x0, y0, 'k')
            for key in list(di.keys())[1:]:
                df = di[key]
                if len(df)>10: # greater than 400ms
                    df['ft_posx'] = df['ft_posx']
                    df['ft_posy'] = df['ft_posy']
                    df['seconds'] = df['seconds']-df['seconds'].iloc[0]
                    df_sort = df.iloc[(df['seconds']-0.4).abs().argsort()[:1]]
                    xi = df_sort.ft_posx.to_numpy()[0]
                    yi = df_sort.ft_posy.to_numpy()[0]
                    axs.plot(xi, yi, 'o', color = pl.inside_color, linestyle='')
            for key in list(do.keys())[1:]:
                df = do[key]
                if len(df)>10: # greater than 400ms
                    df['ft_posx'] = df['ft_posx']
                    df['ft_posy'] = df['ft_posy']
                    df['seconds'] = df['seconds']-df['seconds'].iloc[0]
                    df_sort = df.iloc[(df['seconds']-0.4).abs().argsort()[:1]]
                    xi = df_sort.ft_posx.to_numpy()[0]
                    yi = df_sort.ft_posy.to_numpy()[0]
                    axs.plot(xi, yi, 'o', color = pl.outside_color,linestyle='')
            axs.axis('equal')
            fig.savefig(os.path.join(group_figure_folder, 'plume_fuzziness_example.pdf'))

def plume_fizziness():
    at = average_trajectory('0')
    all_data = at.load_trajectories()
    sheet = at.sheet.log
    xv_transition = []
    yv_transition = []
    fig, axs = plt.subplots(1,1)
    x_in = []
    x_out = []
    for i, log in enumerate(sheet):
        print(log)
        df = all_data[log]['data']
        di = all_data[log]['di']
        do = all_data[log]['do']
        idx_start = df.index[df.instrip].tolist()[0]
        x0 = df.iloc[idx_start].ft_posx
        y0 = df.iloc[idx_start].ft_posy
        for key in list(di.keys())[1:]:
            df = di[key]
            if len(df)>10: # greater than 400ms
                df['ft_posx'] = df['ft_posx']-df['ft_posx'].iloc[0]
                if np.mean(df['ft_posx'])<0:
                    df['ft_posx'] = -df['ft_posx']
                df['ft_posy'] = df['ft_posy']-y0
                df['seconds'] = df['seconds']-df['seconds'].iloc[0]
                df_sort = df.iloc[(df['seconds']-0.4).abs().argsort()[:1]]
                if df_sort.ft_posx.to_numpy()[0]>0:
                    x_in.append(df_sort.ft_posx.to_numpy()[0])
                    axs.plot(df_sort.ft_posx, df_sort.ft_posy, '.', color = 'k', alpha=0.1)
        for key in list(do.keys())[1:]:
            df = do[key]
            if len(df)>10: # greater than 400ms
                df['ft_posx'] = df['ft_posx']-df['ft_posx'].iloc[0]
                if np.mean(df['ft_posx'])>0:
                    df['ft_posx'] = -df['ft_posx']
                df['ft_posy'] = df['ft_posy']-y0
                df['seconds'] = df['seconds']-df['seconds'].iloc[0]
                df_sort = df.iloc[(df['seconds']-0.4).abs().argsort()[:1]]
                if df_sort.ft_posx.to_numpy()[0]<0:
                    x_out.append(df_sort.ft_posx.to_numpy()[0])
                    axs.plot(df_sort.ft_posx, df_sort.ft_posy, '.', color = 'r', alpha=0.1)
    fig.savefig(os.path.join(group_figure_folder, 'plume_fizziness.pdf'))
    fig, axs = plt.subplots(1,1)
    sns.histplot(x_in, ax=axs, kde=True, color='black', stat='density')
    sns.histplot(x_out, ax=axs, kde=True, color='red', stat='density')
    fig.savefig(os.path.join(group_figure_folder, 'plume_fizziness_hist.pdf'))

    return x_in, x_out

def rdp_runlength_angle_inout():
    """
    create histograms showing the run lengths and turn angles inside and outside using the RDP
    algorithm.
    """
    at = average_trajectory(directory = 'M1', experiment='0')
    all_data = at.load_trajectories()
    sheet = at.sheet.log
    runs_in, runs_out = [],[]
    angles_in, angles_out = [],[]

    for i, log in enumerate(sheet):
        print(log)
        di = all_data[log]['di']
        do = all_data[log]['do']
        for key in list(di.keys()):
            df_i = di[key]
            xi = df_i.ft_posx.to_numpy()
            yi = df_i.ft_posy.to_numpy()
            _, _, angles, lens = fn.rdp_simp_heading_angles_len(xi,yi, epsilon=1)
            runs_in.append(lens)
            angles_in.append(angles)

        for key in list(do.keys()):
            df_o = do[key]
            xo = df_o.ft_posx.to_numpy()
            yo = df_o.ft_posy.to_numpy()
            _, _,angles, lens = fn.rdp_simp_heading_angles_len(xo,yo, epsilon=1)
            runs_out.append(lens)
            angles_out.append(angles)

    runs_in = np.concatenate(runs_in)
    angles_in = np.concatenate(angles_in)
    runs_out = np.concatenate(runs_out)
    angles_out = np.concatenate(angles_out)

    fig, axs = plt.subplots(1,2)
    sns.histplot(runs_in, bins=np.linspace(0,30,60), element='step', fill=False, color=pl.inside_color, ax=axs[0],stat='probability')
    sns.histplot(runs_out, bins=np.linspace(0,30,60), element='step', fill=False, color=pl.outside_color, ax=axs[0],stat='probability')
    sns.histplot(np.abs(np.rad2deg(angles_in)),bins=np.linspace(0,180, 30), element='step', fill=False, color=pl.inside_color, ax=axs[1],stat='probability')
    sns.histplot(np.abs(np.rad2deg(angles_out)),bins=np.linspace(0,180, 30), element='step', fill=False, color=pl.outside_color, ax=axs[1],stat='probability')
    fig.tight_layout()
    axs[0].set_xlabel('run lengths(mm)')
    axs[1].set_xlabel('|turn angle')
    axs[1].set_xticks([0,45,90,135,180])
    fig.savefig(os.path.join(group_figure_folder, 'RDP_angles_runs_inout.pdf'))
    print('number of flies: ', str(i+1))
    print('number of inside lengths: ', str(runs_in.shape))
    print('number of outside lengths: ', str(runs_out.shape))
    print('number of inside angles: ', str(angles_in.shape))
    print('number of outside angles: ', str(angles_out.shape))

def plot_example_45s():
    """
    plot 45 degree example trajectories showing longer outside load_trajectories
    in the initial part of the trial.
    """
    examples = ['08242020-222507_45degOdorRight_reinforced.log',
                '08212020-185046_45degOdorRight_reinforced.log',
                ]
    at = average_trajectory(directory='M1', experiment = '45')
    all_data = at.load_trajectories()
    sheet = at.sheet.log
    for i, log in enumerate(sheet):
        if log in examples:
            print(log)
            df = all_data[log]['data']
            fig,axs = plt.subplots(1,1)
            pl.plot_trajectory(df, axs)
            pl.plot_trajectory_odor(df, axs)
            axs.axis('equal')
            fig.suptitle(log)
            fig.savefig(os.path.join(group_figure_folder, log+'.pdf'))

a = plot_efficiencies()
a.get_shared_x_axes().get_siblings(a)
# a.get_shared_x_axes().get_siblings(a)[0]
# %%

class average_trajectory_old():
    """
    1) specify experiment
    2) split up log files based on inside/outside
    3) pickle results
    """
    def __init__(self, directory='M1', experiment = 'T', download_logs = False):
        d = dr.drive_hookup()
        # specify working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'
        elif directory == 'Andy':
            self.cwd = '/Volumes/Andy/edge-tracking'
        self.experiment = experiment

        if experiment == 'T':
            self.sheet_id = '14r0TgRUhohZtw2GQgirUseBWXK8NPbyqPzPvAtND7Gs'
            df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
            self.sheet = df
            self.logfol = os.path.join(self.cwd, 'data/trajectory_analysis/t-plume/logs')
            self.picklefol = os.path.join(self.cwd, 'data/trajectory_analysis/t-plume/pickles')
            self.picklesname = os.path.join(self.picklefol, self.experiment+'_'+'plume.p')
            self.log_id = '1LzXApM3XFJIoP5zEtKMtSsLb8ESQOeiZ'
            self.figurefol = os.path.join(self.cwd, 'figures/average_trajectories/t-plume')
            if download_logs:
                d.download_folder(self.log_id, self.logfol)

        elif experiment == '15':
            self.sheet_id = '1qCrV96jUo24lpZ7-k2-B9RWG5RSQDSgnSn-sFgjS7ys'
            df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
            self.sheet = df
            self.logfol = os.path.join(self.cwd, 'data/trajectory_analysis/15/logs')
            self.picklefol = os.path.join(self.cwd, 'data/trajectory_analysis/15/pickles')
            self.picklesname = os.path.join(self.picklefol, self.experiment+'_'+'plume.p')
            self.log_id = '1Mru2d6JY42n4dNe97W7xX0fWmsZqNsto'
            self.figurefol = os.path.join(self.cwd, 'figures/average_trajectories/15')
            if download_logs:
                d.download_folder(self.log_id, self.logfol)

        elif experiment == '45':
            self.sheet_id = '15mE8k1Z9PN3_xhQH6mz1AEIyspjlfg5KPkd1aNLs9TM'
            df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
            self.sheet = df
            self.logfol = os.path.join(self.cwd, 'data/trajectory_analysis/45/logs')
            self.picklefol = os.path.join(self.cwd, 'data/trajectory_analysis/45/pickles')
            self.picklesname = os.path.join(self.picklefol, self.experiment+'_'+'plume.p')
            self.log_id = '1DGZwqOoI6nv88cKdl9oFUCFm6k7HJouF'
            self.figurefol = os.path.join(self.cwd, 'figures/average_trajectories/45')
            if download_logs:
                        d.download_folder(self.log_id, self.logfol)

        elif experiment == '0':
            self.sheet_id = '1K_SkaT3JUA2Ik8uiwB6kJMwHnd935bZvZB4If0zh8rY'
            df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
            self.sheet = df
            self.logfol = os.path.join(self.cwd, 'data/trajectory_analysis/0/logs')
            self.picklefol = os.path.join(self.cwd, 'data/trajectory_analysis/0/pickles')
            self.picklesname = os.path.join(self.picklefol, self.experiment+'_'+'plume.p')
            self.log_id = '1DGZwqOoI6nv88cKdl9oFUCFm6k7HJouF'
            self.figurefol = os.path.join(self.cwd, 'figures/average_trajectories/0')
            if download_logs:
                        d.download_folder(self.log_id, self.logfol)

    def split_trajectories(self):
        """
        For a given experiment (0,45,90 degree plume), read in all log files,
        split them into inside and outside trajectories, and pickle the results.
        """

        # dict where all log files are stored
        all_data = {}

        for i, log in enumerate(self.sheet.log):
            # read in each log file
            data = fn.read_log(os.path.join(self.logfol, log))
            # if the tracking was lost, select correct segment
            data = fn.exclude_lost_tracking(data, thresh=10)
            # specificy when the fly is in the strip for old mass flow controllers
            mfc = self.sheet.mfc.iloc[i]
            if mfc == 'old':
                data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
            # consolidate short in and short out periods
            data = fn.consolidate_in_out(data)
            # append speeds to dataframe
            data = fn.calculate_speeds(data)
            # split trajectories into inside and outside components
            d, di, do = fn.inside_outside(data)
            dict_temp = {"data": data,
                        "d": d,
                        "di": di,
                        "do": do}
            all_data[log] = dict_temp
        # pickle everything
        fn.save_obj(all_data, self.picklesname)

    def load_single_trajectory(self, log):
        data = fn.read_log(os.path.join(self.logfol, log))
        ix = self.sheet[self.sheet.log==log].index.to_numpy()

        mfc = self.sheet['mfc'][ix[0]]
        # data['instrip'] = np.where(np.abs(data.ft_posx)<25, True, False)
        data = fn.exclude_lost_tracking(data, thresh=10)
        if mfc == 'old': # all experiments performed with old MFCs
            data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
        return data

    def load_trajectories(self):
        """
        open the pickled data stored from split_trajectories()
        """
        all_data = fn.load_obj(self.picklesname)
        return all_data

    def heatmap(self,cmap='gist_gray', overwrite = False, plot_individual=False, res = 5):
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

        if not os.path.exists(filename) or overwrite:
            # set up arrays for heatmap
            if self.experiment == '0':
                x_bounds = np.arange(-200, 200, res)
                y_bounds = np.arange(0, 1000, res)
            if self.experiment == '15':
                x_bounds = np.arange(-100, 300, res)
                y_bounds = np.arange(0, 1000, res)
            if self.experiment == '45':
                x_bounds = np.arange(-50, 1000, res)
                y_bounds = np.arange(0, 1000, res)
            if self.experiment == 'T':
                self.experiment = '90'
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
                if self.sheet.iloc[i].mfc=='new':
                    idx_start = df.index[df.mfc2_stpt>0.01].tolist()[0]
                    df = df.iloc[idx_start:]
                xi = df.ft_posx.to_numpy()
                yi = df.ft_posy.to_numpy()
                xi=xi-xi[0]
                yi=yi-yi[0]

                # reflect tilted and plumes so that they are facing the right way
                if self.sheet.iloc[i].direction=='left':
                    xi = -xi
                if self.experiment == '90':
                    # halfway point
                    ix1_2 = int(len(xi)/2)
                    if np.mean(xi[ix1_2:])<0:
                        xi = -xi
                # save and example trajectory to plot on top of the heatmap
                if log in examples:
                    x_ex = xi
                    y_ex = yi
                if plot_individual == True:
                    fig, axs = plt.subplots(1,1)
                    axs.plot(xi,yi)
                    axs.title.set_text(log)

                # rotate the tilted plume to make them vertical.
                angle = int(self.experiment)
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
        fig, axs = plt.subplots(1,1)
        vmin = np.percentile(fly_density[fly_density>0],0)
        vmax = np.percentile(fly_density[fly_density>0],90)
        im = axs.imshow(fly_density, cmap=cmap, vmin=vmin,vmax = vmax, rasterized=True, extent=(min(x_bounds), max(x_bounds), min(y_bounds), max(y_bounds)))
        if self.experiment == '0':
            axs.plot([-25,-25], [np.min(y_bounds), np.max(y_bounds)], 'w', alpha=0.2)
            axs.plot([25,25], [np.min(y_bounds), np.max(y_bounds)], 'w', alpha=0.2)
        if self.experiment == '90':
            axs.plot([np.min(x_bounds), np.max(x_bounds)], [300,300], 'w', alpha=0.2)
            axs.plot([np.min(x_bounds), -25], [250,250], 'w', alpha=0.2)
            axs.plot([25, np.max(x_bounds)], [250,250], 'w', alpha=0.2)
            axs.plot([25,25], [np.min(y_bounds), 250], 'w', alpha=0.2)
            axs.plot([-25,-25], [np.min(y_bounds), 250], 'w', alpha=0.2)
        axs.plot(x_ex, y_ex, 'red', linewidth=1)#, alpha = 0.8, linewidth=0.5)
        fig.colorbar(im)
        fig.savefig(os.path.join(self.figurefol, self.experiment+'_heatmap.pdf'))

        fig, axs = plt.subplots(1,1)
        # vmin = np.percentile(fly_density_rotate[fly_density_rotate>0],10)
        # vmax = np.percentile(fly_density_rotate[fly_density_rotate>0],90)
        axs.imshow(fly_density_rotate, cmap=cmap, vmin=1,vmax = 6, extent=(min(x_bounds_rotate), max(x_bounds_rotate), min(y_bounds_rotate), max(y_bounds_rotate)))


        # boundaries
        if self.experiment == '0':
            boundary=25
        if self.experiment == '15':
            boundary = 25/np.sin(np.deg2rad(90-int(self.experiment)))
            boundary = 25
        if self.experiment == '45':
            boundary = 25*np.cos(np.deg2rad(90-int(self.experiment)))
            #boundary = 25
        if self.experiment == '90':
            boundary=25
        fly_density_projection = fly_density_rotate[0:-40, :]/np.sum(fly_density_rotate[0:-40, :])
        x_mean = np.sum(fly_density_projection, axis = 0)
        fig, axs = plt.subplots(1,1)
        axs.plot([-boundary, -boundary], [min(x_mean), max(x_mean)], 'k', alpha=0.5)
        axs.plot([boundary, boundary], [min(x_mean), max(x_mean)],'k', alpha=0.5)
        axs.plot(x_bounds_rotate, x_mean)
        axs.set_ylim(0,0.06)
        fig.savefig(os.path.join(self.figurefol, self.experiment+'_density.pdf'))
        return fly_density, fly_density_rotate

    def plot_individual_trajectories(self):
        for i, log in enumerate(self.sheet.log):
            print(log)
            data = fn.read_log(os.path.join(self.logfol, log))
            mfc = self.sheet.mfc.iloc[i]
            # data['instrip'] = np.where(np.abs(data.ft_posx)<25, True, False)
            data = fn.exclude_lost_tracking(data, thresh=10)
            if mfc == 'old': # all experiments performed with old MFCs
                data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
            fig, axs = plt.subplots(1,1)
            pl.plot_trajectory(data, axs)
            pl.plot_trajectory_odor(data, axs)
            axs.axis('equal')
            fig.suptitle(log)

    def load_average_trajectories(self):
        pickle_name = os.path.join(self.picklefol, self.experiment+'_'+'average_trajectories.p')
        [animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out] = fn.load_obj(pickle_name)
        return animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out

    def plot_trajectories_T(self, pts=10000):
        all_data = self.load_trajectories()
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = [],[],[],[]
        for log in self.sheet.log:
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
                    if np.abs(y[-1]-y[0])<1: #condition for returning on horizontal plume, may need to add in a y condition here
                        count+=1
                        x0 = x[0]
                        y0 = y[0]
                        x = x-x0
                        y = y-y0
                        # if x[-1]<x[0]:
                        #     x=-x
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
                    if np.abs(y[-1]-y[0])<1: #condition for returning on horizontal plume, may need to add in a y condition here
                        x0 = x[0]
                        y0 = y[0]
                        x = x-x0
                        y = y-y0
                        # if x[-1]<x[0]:
                        #     x=-x
                        t = np.arange(len(x))
                        t_common = np.linspace(t[0], t[-1], pts)
                        fx = interpolate.interp1d(t, x)
                        fy = interpolate.interp1d(t, y)
                        #axs.plot(fx(t_common), fy(t_common))
                        avg_x_in.append(fx(t_common))
                        avg_y_in.append(fy(t_common))
            if count>3: # condition: each trajectory needs more than three outside trajectories
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
                fig.savefig(os.path.join(self.figurefol, log.replace('.log', '.pdf')), transparent=True)

        # make an average of the averages for ech fly
        fig, axs = plt.subplots(1,1)


        for i in np.arange(len(animal_avg_x_in)):
            if animal_avg_x_in[i][-1]<0:
                animal_avg_x_in[i] = -animal_avg_x_in[i]
            axs.plot(animal_avg_x_in[i], animal_avg_y_in[i], 'r', alpha=0.1)
        axs.plot(np.mean(animal_avg_x_in, axis=0), np.mean(animal_avg_y_in, axis=0), 'r')
        exit_x = np.mean(animal_avg_x_in, axis=0)[-1]

        for i in np.arange(len(animal_avg_x_out)):
            if animal_avg_x_out[i][-1]<0:
                animal_avg_x_out[i] = -animal_avg_x_out[i]
            axs.plot(animal_avg_x_out[i]+exit_x, animal_avg_y_out[i], 'k', alpha=0.1)
        axs.plot(np.mean(animal_avg_x_out+exit_x, axis=0), np.mean(animal_avg_y_out, axis=0), 'k')

        # save the average trajectories
        fn.save_obj([animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out], os.path.join(self.picklefol, self.experiment+'_'+'average_trajectories.p'))

        fig.savefig(os.path.join(self.figurefol, 'all_averges.pdf'), transparent = True)
        return axs

    def plot_trajectories_angle(self,angle,pts = 10000):
        """
        for plotting average trajectories at a given angle
        """
        n=0
        all_data = self.load_trajectories()
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = [],[],[],[]
        for i, log in enumerate(self.sheet.log):
            avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
            temp = all_data[log]
            direction = self.sheet.direction.iloc[i]
            angle_from_horizontal = np.pi/2 - angle*np.pi/180
            if direction == 'right':
                rot_angle = -angle*np.pi/180
            elif direction == 'left':
                rot_angle = +angle*np.pi/180
            elif direction == 'NA':
                rot_angle = 0
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

            if count>3: # condition: each trajectory needs more than three outside trajectories
                n+=1
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
        fn.save_obj([animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out], os.path.join(self.picklefol, self.experiment+'_'+'average_trajectories.p'))


        # make an average of the averages for ech fly
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
        axs.plot([0,max_y/np.tan(angle_from_horizontal)], [0, max_y], color='k', linestyle='dashed')

        axs.axis('equal')
        fig.savefig(os.path.join(self.figurefol, 'all_averages.pdf'), transparent = True)
        print('number of animals = ', n)
        return axs

    def plot_speed_overlay(self, angle, pts=10000):
        """
        for plotting average trajectories at a given angle
        """
        n=0
        all_data = self.load_trajectories()
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out, animal_avg_speed_in, animal_avg_speed_out = [],[],[],[],[],[]
        for i, log in enumerate(self.sheet.log):
            avg_x_in, avg_y_in, avg_x_out, avg_y_out, avg_speed_in, avg_speed_out=[],[],[],[],[],[]
            temp = all_data[log]
            direction = self.sheet.direction.iloc[i]
            angle_from_horizontal = np.pi/2 - angle*np.pi/180
            if direction == 'right':
                rot_angle = -angle*np.pi/180
            elif direction == 'left':
                rot_angle = +angle*np.pi/180
            elif direction == 'NA':
                rot_angle = 0
            do = temp['do']
            di = temp['di']
            df = temp['data']
            df_instrip = df.where(df.instrip==True)
            del_t = np.mean(np.diff(df.seconds))
            effective_rate = 1/del_t
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
                    vx = np.gradient(x)*effective_rate
                    vy = np.gradient(y)*effective_rate
                    speed = np.sqrt(vx**2+vy**2)
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
                        fs = interpolate.interp1d(t,speed)
                        #axs.plot(fx(t_common), fy(t_common))
                        avg_x_out.append(fx(t_common))
                        avg_y_out.append(fy(t_common))
                        avg_speed_out.append(fs(t_common))
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
                    vx = np.gradient(x)*effective_rate
                    vy = np.gradient(y)*effective_rate
                    speed = np.sqrt(vx**2+vy**2)
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
                        fs = interpolate.interp1d(t,speed)
                        #axs.plot(fx(t_common), fy(t_common))
                        avg_x_in.append(fx(t_common))
                        avg_y_in.append(fy(t_common))
                        avg_speed_in.append(fs(t_common))

            if count>3: # condition: each trajectory needs more than three outside trajectories
                n+=1
                animal_avg_x_out.append(np.mean(avg_x_out, axis=0))
                animal_avg_y_out.append(np.mean(avg_y_out, axis=0))
                animal_avg_speed_out.append(np.mean(avg_speed_out, axis=0))

                animal_avg_x_in.append(np.mean(avg_x_in, axis=0))
                animal_avg_y_in.append(np.mean(avg_y_in, axis=0))
                animal_avg_speed_in.append(np.mean(avg_speed_in, axis=0))

        # make an average of the averages for ech fly
        fig, axs = plt.subplots(1,1)

        # for i in np.arange(len(animal_avg_x_in)):
        #     axs.plot(animal_avg_x_in[i], animal_avg_y_in[i], 'k', alpha=0.1)
        #axs.plot(np.mean(animal_avg_x_in, axis=0), np.mean(animal_avg_y_in, axis=0), 'r')
        exit_x = np.mean(animal_avg_x_in, axis=0)[-1]
        exit_y = np.mean(animal_avg_y_in, axis=0)[-1]
        xi = np.mean(animal_avg_x_in, axis=0)
        yi = np.mean(animal_avg_y_in, axis=0)
        si = np.mean(animal_avg_speed_in, axis=0)
        so = np.mean(animal_avg_speed_in, axis=0)
        cmin = np.min([np.min(so)])
        cmax = np.max([np.max(so)])
        axs = pl.colorline(axs, xi, yi, z=si, segmented_cmap=False, cmap=plt.get_cmap('viridis'), norm = plt.Normalize(cmin, cmax))


        # for i in np.arange(len(animal_avg_x_out)):
        #     axs.plot(animal_avg_x_out[i]+exit_x, animal_avg_y_out[i]+exit_y, 'k', alpha=0.1)
        #axs.plot(np.mean(animal_avg_x_out+exit_x, axis=0), np.mean(animal_avg_y_out+exit_y, axis=0), 'k')
        xo = np.mean(animal_avg_x_out+exit_x, axis=0)
        yo = np.mean(animal_avg_y_out+exit_y, axis=0)
        so = np.mean(animal_avg_speed_out, axis=0)
        axs = pl.colorline(axs, xo, yo, z=so, segmented_cmap=False, cmap=plt.get_cmap('viridis'), norm = plt.Normalize(cmin, cmax))
        axs.autoscale()
        axs.axis('equal')

        # draw the plume boundary line at the appropriate angle
        max_y_out = np.max(np.mean(animal_avg_y_out,axis=0))+exit_y
        max_y_in = np.max(np.mean(animal_avg_y_in, axis=0))
        max_y = np.max((max_y_out,max_y_in))
        axs.plot([0,max_y/np.tan(angle_from_horizontal)], [0, max_y], color='k', linestyle='dashed')

        # plot a colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=plt.Normalize(cmin, cmax))
        plt.colorbar(sm)

        axs.axis('equal')
        fig.savefig(os.path.join(self.figurefol, 'all_averages_speed_heatmap.pdf'), transparent = True)
        print('number of animals = ', n)
        return axs

                #fig.suptitle(log)
                #fig.savefig(os.path.join(self.figurefol, log.replace('.log', '.pdf')), transparent=True)

    def inbound_outbound_angle(self, x, y):
        """
        calculate the inbound and outbound angle for a given inside or outside trajectory
        """
        def calculate_angle(vector_1, vector_2):
            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(dot_product)
            return angle

        max_ix = np.argmax(np.abs(x))
        start_x = x[0]
        start_y = y[0]
        max_x = x[max_ix]
        max_y = y[max_ix]
        return_x = x[-1]
        return_y = y[-1]

        vec_out = [max_x-start_x, max_y-start_y]
        edge_vec_out = [return_x-start_x, return_y-start_y]
        vec_in = [max_x-return_x, max_y-return_y]
        edge_vec_in = [start_x-return_x, start_y-return_y]

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

    def inbound_outbound_lengths(self,x,y):
        """
        inbound and outbound path length
        """
        max_ix = np.argmax(np.abs(x))
        _,length_out = fn.path_length(x[0:max_ix], y[0:max_ix])
        _,length_in = fn.path_length(x[max_ix:], y[max_ix:])

        # fig, axs = plt.subplots(1,1)
        # axs.plot(x[0:max_ix], y[0:max_ix])
        # axs.plot(x[max_ix:], y[max_ix:])
        # axs.text(x[max_ix],y[max_ix],str([length_out, length_in]))

        return [length_out, length_in]

    def inbound_outbound_tortuosity(self,x,y):
        """
        inbound and outbound tortuosity calculated as path length divided by euclidean distance
        """
        max_ix = np.argmax(np.abs(x))
        start_x = x[0]
        start_y = y[0]
        max_x = x[max_ix]
        max_y = y[max_ix]
        return_x = x[-1]
        return_y = y[-1]
        euc_out = np.sqrt((start_x-max_x)**2+(start_y-max_y)**2)
        euc_in = np.sqrt((return_x-max_x)**2+(return_y-max_y)**2)
        _,length_out = fn.path_length(x[0:max_ix], y[0:max_ix])
        _,length_in = fn.path_length(x[max_ix:], y[max_ix:])
        tor_outbound = length_out/euc_out
        tor_inbound = length_in/euc_in
        return [tor_outbound, tor_inbound]

    def plot_outside_pathlengths(self):
        """
        take the average path length for the first n outside trajectories
        and compare it to the average path length for the rest of the
        outside trajectories.
        """
        #load data
        all_data = self.load_trajectories()

        # average over this number of outside bouts
        num_outside_avg = [1,2,3,4,5]
        fig, axs = plt.subplots(2,len(num_outside_avg), figsize = (2*len(num_outside_avg), 4), )
        for n in num_outside_avg:
            pairs = []
            pairs_t = []
            for i, log in enumerate(self.sheet.log):
                temp = all_data[log]
                do = temp['do']
                lens = []
                ts = []
                if len(do)>2*n:
                    for key in list(do.keys())[:-1]:
                        x = do[key].ft_posx.to_numpy()
                        y = do[key].ft_posy.to_numpy()
                        _, l = fn.path_length(x,y)
                        t = do[key].seconds.to_numpy()
                        t = t[-1]-t[0]

                        lens.append(l)
                        ts.append(t)
                    first = np.mean(lens[0:n])
                    rest = np.mean(lens[n:])
                    first_t = np.mean(ts[0:n])
                    rest_t = np.mean(ts[n:])
                    pairs.append([first, rest])
                    pairs_t.append([first_t, rest_t])
                    #axs.plot([first_2, rest])
            pairs = np.array(pairs)
            pairs_t = np.array(pairs_t)
            axs[0,n-1] = pl.paired_plot(axs[0,n-1], pairs[:,0], pairs[:,1])
            axs[0,n-1].set_xticks([0,1])
            axs[0,n-1].set_ylabel('average path length (mm)')
            axs[1,n-1] = pl.paired_plot(axs[1,n-1], pairs_t[:,0], pairs_t[:,1])
            axs[1,n-1].set_xticks([0,1])
            axs[1,n-1].set_ylabel('average outside time (s)')
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'first_rest_pathlengths.pdf'))
        return pairs

    def inbound_outbound_angle_distribution(self,angle, pts = 10000):
        all_data = self.load_trajectories()
        angles_outside, angles_inside=[],[]
        for i, log in enumerate(self.sheet.log):
            avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
            temp = all_data[log]
            direction = self.sheet.direction.iloc[i]
            angle_from_horizontal = np.pi/2 - angle*np.pi/180
            if direction == 'right':
                rot_angle = -angle*np.pi/180
            elif direction == 'left':
                rot_angle = +angle*np.pi/180
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

            if count>3: # condition: each trajectory needs more than three outside trajectories
                print(log)
                for i in np.arange(len(avg_x_in)):
                    x = avg_x_in[i]
                    y = avg_y_in[i]
                    x,y = fn.coordinate_rotation(x, y, rot_angle)
                    angles = self.inbound_outbound_angle(x, y)
                    angles_inside.append(angles)
                for i in np.arange(len(avg_x_out)):
                    x = avg_x_out[i]
                    y = avg_y_out[i]
                    x,y = fn.coordinate_rotation(x, y, rot_angle)
                    angles = self.inbound_outbound_angle(x, y)
                    angles_outside.append(angles)
        angles_outside = np.array(angles_outside)
        angles_inside = np.array(angles_inside)
        df_inside = pd.DataFrame({'outbound': angles_inside[:,0], 'inbound': angles_inside[:,1]})
        df_outside = pd.DataFrame({'outbound': angles_outside[:,0], 'inbound': angles_outside[:,1]})

        colors = sns.color_palette()

        fig, axs = plt.subplots(1,2, figsize=(6,3))

        sns.histplot(data=df_inside, x='inbound', element='step',stat='density', cumulative=True, fill=False, kde=False, color = colors[0], bins=32, ax=axs[0])
        sns.histplot(data=df_inside, x='outbound', element='step',stat='density',cumulative=True, fill=False, kde=False, color = colors[1], bins=32, ax=axs[0])
        axs[0].set_xlabel('angle (degrees)')
        axs[0].set_xticks([0,45,90,135,180])
        axs[0].title.set_text('inside_trajectories')
        #fig.savefig(os.path.join(self.figurefol, 'inside_angle_distibution.pdf'), transparent = True)


        sns.histplot(data=df_outside, x='inbound',stat='density', element='step',cumulative=True, fill=False, kde=False, color = colors[0], bins=32, ax=axs[1])
        sns.histplot(data=df_outside, x='outbound',stat='density', element='step',cumulative=True, fill=False, kde=False, color = colors[1], bins=32, ax=axs[1])
        axs[1].set_xlabel('angle (degrees)')
        axs[1].set_xticks([0,45,90,135,180])
        axs[1].title.set_text('outside_trajectories')
        fig.savefig(os.path.join(self.figurefol, 'angle_distribution.pdf'), transparent = True)

    def average_inbound_outbound_angle(self, angle):
        # calculate the outbound and inbound angles for the average trajectories
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = self.load_average_trajectories()
        rot_angle = -angle*np.pi/180
        angles_outside = []
        for i in np.arange(len(animal_avg_x_out)):
            x = animal_avg_x_out[i]
            y = animal_avg_y_out[i]
            x,y = fn.coordinate_rotation(x, y, rot_angle)
            angles = self.inbound_outbound_angle(x, y)
            angles_outside.append(angles)
        angles_outside = np.array(angles_outside)

        return angles_outside

    def average_inbound_outbound_path_length(self, angle):
        # calculate the outbound and inbound angles for the average trajectories
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = self.load_average_trajectories()
        rot_angle = -angle*np.pi/180
        lengths_outside = []
        for i in np.arange(len(animal_avg_x_out)):
            x = animal_avg_x_out[i]
            y = animal_avg_y_out[i]
            x,y = fn.coordinate_rotation(x, y, rot_angle)
            lengths = self.inbound_outbound_lengths(x, y)
            lengths_outside.append(lengths)
        lengths_outside = np.array(lengths_outside)

        return lengths_outside

    def average_inbound_outbound_tortuosity(self, angle):
        # calculate the outbound and inbound angles for the average trajectories
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = self.load_average_trajectories()
        rot_angle = -angle*np.pi/180
        tor_outside = []
        for i in np.arange(len(animal_avg_x_out)):
            x = animal_avg_x_out[i]
            y = animal_avg_y_out[i]
            x,y = fn.coordinate_rotation(x, y, rot_angle)
            tor = self.inbound_outbound_tortuosity(x, y)
            tor_outside.append(tor)
        tor_outside = np.array(tor_outside)

        return tor_outside

    def inbound_outbound_path_length(self,angle, pts = 10000):
        all_data = self.load_trajectories()
        lengths_outside, lengths_inside=[],[]
        for i, log in enumerate(self.sheet.log):
            avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
            temp = all_data[log]
            direction = self.sheet.direction.iloc[i]
            angle_from_horizontal = np.pi/2 - angle*np.pi/180
            if direction == 'right':
                rot_angle = -angle*np.pi/180
            elif direction == 'left':
                rot_angle = +angle*np.pi/180
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

            if count>3: # condition: each trajectory needs more than three outside trajectories
                print(log)
                for i in np.arange(len(avg_x_in)):
                    x = avg_x_in[i]
                    y = avg_y_in[i]
                    x,y = fn.coordinate_rotation(x, y, rot_angle)
                    lengths = self.inbound_outbound_tortuosity(x, y)
                    lengths_inside.append(lengths)
                for i in np.arange(len(avg_x_out)):
                    x = avg_x_out[i]
                    y = avg_y_out[i]
                    x,y = fn.coordinate_rotation(x, y, rot_angle)
                    lengths = self.inbound_outbound_tortuosity(x, y)
                    lengths_outside.append(lengths)
        lengths_outside = np.array(lengths_outside)
        lengths_inside = np.array(lengths_inside)
        # df_inside = pd.DataFrame({'outbound': lengths_inside[:,0], 'inbound': lengths_inside[:,1]})
        # df_outside = pd.DataFrame({'outbound': lengths_outside[:,0], 'inbound': lengths_outside[:,1]})
        #
        # colors = sns.color_palette()
        #
        # fig, axs = plt.subplots(1,1)
        #
        # sns.histplot(data=df_inside, x='inbound', element='step', fill=False, kde=False, color = colors[0], bins=16)
        # sns.histplot(data=df_inside, x='outbound', element='step', fill=False, kde=False, color = colors[1], bins=16)
        # axs.set_xlabel('angle (degrees)')
        # axs.set_xticks([0,45,90,135,180])
        # fig.savefig(os.path.join(self.figurefol, 'inside_angle_distibution.pdf'), transparent = True)

        # fig, axs = plt.subplots(1,1)
        # sns.histplot(data=df_outside, x='inbound', element='step', fill=False, kde=False, color = colors[0], bins=16)
        # sns.histplot(data=df_outside, x='outbound', element='step', fill=False, kde=False, color = colors[1], bins=16)
        # axs.set_xlabel('angle (degrees)')
        # axs.set_xticks([0,45,90,135,180])
        # fig.savefig(os.path.join(self.figurefol, 'outside_angle_distibution.pdf'), transparent = True)
        return lengths_inside, lengths_outside

    def time_in_v_time_out(self):
        """
        plot the current inside time with the next outside time.  Doesn't seem
        to be a clear correlation.  Won't do much more with this analysis, but
        it might be useful at some point
        """
        all_data = self.load_trajectories()
        sheet = self.sheet.log
        inside_time_all = []
        outside_time_all = []
        fig, axs = plt.subplots(1,1,figsize=(4,4))
        for i, log in enumerate(sheet):

            print(log)
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']
            if len(do)>10:
                for key in list(di.keys()):
                    if key+1 in list(do.keys()):
                        inside_time = di[key].seconds.to_numpy()
                        outside_time = do[key+1].seconds.to_numpy()
                        inside_time = inside_time[-1]-inside_time[0]
                        outside_time = outside_time[-1]-outside_time[0]
                        inside_time_all.append(inside_time)
                        outside_time_all.append(outside_time)
                        axs.plot(inside_time, outside_time, '.')
        axs.set_xscale('log')
        axs.set_yscale('log')
        cor = np.corrcoef(np.log10(inside_time_all), np.log10(outside_time_all))
        print(cor)

    def time_distance_out_consecutive_trajectories(self):
        """
        make three histograms showing the time and distance outside for outside
        trajectories and the number of consecutive outside trajectories.
        """
        all_data = self.load_trajectories()
        sheet = self.sheet.log
        outside_time_all = []
        outside_dist_all = []
        cons_outside = []
        for i, log in enumerate(sheet):
            print(log)
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']
            count = 0
            if len(do)>1:
                for key in list(do.keys())[1:]:
                    temp = do[key]
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()
                    t = temp.seconds.to_numpy()
                    if np.abs(x[-1]-x[0])<2:
                        count+=1
                        t = t-t[0]
                        _,l = fn.path_length(x,y)
                        outside_time_all.append(t[-1])
                        outside_dist_all.append(l)
                cons_outside.append(count)
        print('number of trajectories = ', len(cons_outside))
        print('number of outside trajectories = ', len(outside_time_all))
        fig, axs = plt.subplots(1,3, figsize = (15, 5))
        sns.histplot(outside_time_all,ax=axs[0], element="step",log_scale=True, fill=False)
        axs[0].set_xlabel('time outside plume (s)')
        sns.histplot(outside_dist_all,ax=axs[1], element="step",log_scale=True, fill=False)
        axs[1].set_xlabel('distance traveled (mm)')
        sns.histplot(cons_outside,ax=axs[2],bins=20, element="step", fill=False)
        axs[2].set_xlabel('# of consectutive returns')

        fig.savefig(os.path.join(self.figurefol, 'time_distance_outside_consectuive.pdf'))

    def plot_example_trajectory(self, inside=False, savefig=False, pts = 10000):
        all_data = self.load_trajectories()
        if self.experiment == '15':
            if inside:
                ilog = 3
                ibout = 4
                log = self.sheet.iloc[ilog].log
                temp = all_data[log]
                di = temp['di'][ibout]
                df = temp['data']
                fig_example, axs = plt.subplots(1, 1)
                axs.plot(di.ft_posx, di.ft_posy)
                axs.axis('equal')
                savename = 'example_'+log.replace('.log','')+'_inside_'+str(ibout)+'.pdf'
            else:
                ilog=3
                ibout = 12
                log = self.sheet.iloc[ilog].log
                temp = all_data[log]
                do = temp['do'][ibout]
                df = temp['data']
                fig_example, axs = plt.subplots(1, 1)
                axs.plot(do.ft_posx, do.ft_posy)
                axs.axis('equal')
                savename = 'example_'+log.replace('.log','')+'_outside_'+str(ibout)+'.pdf'

        fig, axs = plt.subplots(1,1)
        axs.plot(df.ft_posx, df.ft_posy)

        if savefig:
            fig_example.savefig(os.path.join(self.figurefol, savename))

    def behavioral_metrics_in_out_time(self, plot_individual=False):
        """
        function for plotting the inside and outside times as a histogram.
        For this analysis we will just look at the inside and outside
        trajectories that returned to the edge. Look at the aggregate of all
        trajectories, pooled across animals.

        possible modifications to script:
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        t_in = []
        t_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # inside calculations: time
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    t_in_temp = (df_i.seconds.iloc[-1]-df_i.seconds.iloc[0])
                    t_in.append(t_in_temp)

            # outside calculations: time
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    t_out_temp = (df_o.seconds.iloc[-1]-df_o.seconds.iloc[0])
                    t_out.append(t_out_temp)

            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))

        metrics = {
        't_in': t_in,
        't_out': t_out,
        }

        max_x = 500
        # plot the inside and outside time for individual animals log scale on x
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        sns.histplot(x=metrics['t_in'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['t_out'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[0].axvline(np.mean(metrics['t_in']), color=pl.inside_color, linestyle='--')
        axs[0].axvline(np.mean(metrics['t_out']), color=pl.outside_color, linestyle='--')
        axs[0].set_xlabel('time (s)')

        # plot the inside and outside time for individual animals linear scale on x
        sns.histplot(x=metrics['t_in'], element="step", fill=False, ax=axs[1], bins=20, binrange=(0,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['t_out'], element="step", fill=False, ax=axs[1], bins=20, binrange=(0,max_x), color=pl.outside_color)
        axs[1].axvline(np.mean(metrics['t_in']), color=pl.inside_color, linestyle='--')
        axs[1].axvline(np.mean(metrics['t_out']), color=pl.outside_color, linestyle='--')
        axs[1].set_xlabel('time (s)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_time.pdf'), bbox_inches='tight', transparent=True)

        return metrics

    def behavioral_metrics_in_out_time_paired(self, plot_individual=False):
        """
        function for plotting the average inside and outside times for each
        animal as a paired plot

        duture modifications
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        times_in = []
        times_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # add time for inside trajectories that return to the edge
            t_in = []
            returns = 0
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    returns+=1
                    t_in_temp = (df_i.seconds.iloc[-1]-df_i.seconds.iloc[0])
                    t_in.append(t_in_temp)

            # add time for outside trajectories that return to the edge
            t_out = []
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    t_out_temp = (df_o.seconds.iloc[-1]-df_o.seconds.iloc[0])
                    t_out.append(t_out_temp)

            # only count trial is it returned to the edge at least twice (n=3 for inside)
            if returns>=2:
                times_in.append(np.mean(t_in))
                times_out.append(np.mean(t_out))

            # plot individual traces
            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        fig, axs = plt.subplots(1,1, figsize = (1,2))
        axs = pl.paired_plot(axs, times_in, times_out, color1=pl.inside_color, color2=pl.outside_color, log = True, alpha=0.1)
        axs.set_xticks([0, 1])
        axs.set_xticklabels(['in', 'out'])

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_time_paired.pdf'), bbox_inches='tight', transparent=True)

    def behavioral_metrics_in_out_upwind_dist(self, plot_individual=False):
        """
        function for plotting the inside and outside upwind distances as a histogram.
        For this analysis we will just look at the inside and outside
        trajectories that returned to the edge. Look at the aggregate of all
        trajectories, pooled across animals.

        possible modifications to script:
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        y_in = []
        y_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # inside calculations: time
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    y_in_temp = (df_i.ft_posy.iloc[-1]-df_i.ft_posy.iloc[0])
                    y_in.append(y_in_temp)

            # outside calculations: time
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    y_out_temp = (df_o.ft_posy.iloc[-1]-df_o.ft_posy.iloc[0])
                    y_out.append(y_out_temp)

            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))

        metrics = {
        'y_in': y_in,
        'y_out': y_out,
        }

        max_x = 30
        # plot the inside and outside time for individual animals log scale on x
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        sns.histplot(x=metrics['y_in'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['y_out'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[0].axvline(np.mean(metrics['y_in']), color=pl.inside_color, linestyle='--')
        axs[0].axvline(np.mean(metrics['y_out']), color=pl.outside_color, linestyle='--')
        axs[0].set_xlabel('upwind distance (mm)')

        # plot the inside and outside time for individual animals linear scale on x
        sns.histplot(x=metrics['y_in'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['y_out'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[1].axvline(np.mean(metrics['y_in']), color=pl.inside_color, linestyle='--')
        axs[1].axvline(np.mean(metrics['y_out']), color=pl.outside_color, linestyle='--')
        axs[1].set_xlabel('upwind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_y_dist.pdf'), bbox_inches='tight', transparent=True)

        return metrics

    def behavioral_metrics_in_out_upwind_dist_paired(self, plot_individual=False):
        """
        function for plotting the average inside and outside times for each
        animal as a paired plot

        duture modifications
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        ydist_in = []
        ydist_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # add time for inside trajectories that return to the edge
            y_in = []
            returns = 0
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    returns+=1
                    y_in_temp = (df_i.ft_posy.iloc[-1]-df_i.ft_posy.iloc[0])
                    y_in.append(y_in_temp)

            # add time for outside trajectories that return to the edge
            y_out = []
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    y_out_temp = (df_o.ft_posy.iloc[-1]-df_o.ft_posy.iloc[0])
                    y_out.append(y_out_temp)

            # only count trial is it returned to the edge at least twice (n=3 for inside)
            if returns>=2:
                ydist_in.append(np.mean(y_in))
                ydist_out.append(np.mean(y_out))

            # plot individual traces
            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        fig, axs = plt.subplots(1,1, figsize = (1,2))
        axs = pl.paired_plot(axs, ydist_in, ydist_out, color1=pl.inside_color, color2=pl.outside_color, alpha=0.1)
        axs.set_xticks([0, 1])
        axs.set_xticklabels(['in', 'out'])
        axs.set_ylabel('average upwind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_y_dist_paired.pdf'), bbox_inches='tight', transparent=True)
        return ydist_in, ydist_out

    def behavioral_metrics_in_out_crosswind_dist(self, plot_individual=False):
        """
        function for plotting the inside and outside crosswind as a histogram.
        For this analysis we will just look at the inside and outside
        trajectories that returned to the edge. Look at the aggregate of all
        trajectories, pooled across animals.

        possible modifications to script:
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        x_in = []
        x_out = []
        x_del_in = []
        x_del_out = []

        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # inside calculations: x distance
            for key in list(di.keys()):
                df_i = di[key]
                df_i.ft_posx = np.abs(df_i.ft_posx-df_i.ft_posx.iloc[0])
                df_i.ft_posy = df_i.ft_posy-df.ft_posy.iloc[0]
                if return_to_edge(df_i):
                    x_steps = np.abs(np.gradient(df_i.ft_posx))
                    x_in_temp = np.sum(x_steps)
                    x_in.append(x_in_temp)
                    x_del_in.append(np.max(df_i.ft_posx))


            # outside calculations: x distance
            for key in list(do.keys()):
                df_o = do[key]
                df_o.ft_posx = np.abs(df_o.ft_posx-df_o.ft_posx.iloc[0])
                df_o.ft_posy = df_o.ft_posy-df_o.ft_posy.iloc[0]
                if return_to_edge(df_o):
                    x_steps = np.abs(np.gradient(df_o.ft_posx))
                    x_out_temp = np.sum(x_steps)
                    x_out.append(x_out_temp)
                    x_del_out.append(np.max(df_o.ft_posx))

            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        metrics = {
        'x_in': x_in,
        'x_out': x_out,
        'x_del_in': x_del_in,
        'x_del_out': x_del_out
        }

        max_x = 1000
        # plot the inside and outside cumulative x distance with a log scale
        fig, axs = plt.subplots(1,4, figsize=(16,4))
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[0].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[0].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[0].set_xlabel('crosswind pathlen. (mm)')
        print('x_in mean = ', np.mean(metrics['x_in']), 'x_out mean = ', np.mean(metrics['x_out']))
        # plot the inside and outside cumulative x distance with a linear scale
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[1].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[1].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[1].set_xlabel('crosswind pathlen. (mm)')

        max_x = 500
        # plot the inside and outside maximum x distance with a log scale
        sns.histplot(x=metrics['x_del_in'], element="step", fill=False, ax=axs[2], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['x_del_out'], element="step", fill=False, ax=axs[2], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[2].axvline(np.mean(metrics['x_del_in']), color=pl.inside_color, linestyle='--')
        axs[2].axvline(np.mean(metrics['x_del_out']), color=pl.outside_color, linestyle='--')
        axs[2].set_xlabel('max crosswind dist (mm)')

        # plot the inside and outside maximum x distance with a linear scale
        sns.histplot(x=metrics['x_del_in'], element="step", fill=False, ax=axs[3], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['x_del_out'], element="step", fill=False, ax=axs[3], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[3].axvline(np.mean(metrics['x_del_in']), color=pl.inside_color, linestyle='--')
        axs[3].axvline(np.mean(metrics['x_del_out']), color=pl.outside_color, linestyle='--')
        axs[3].set_xlabel('max crosswind dist (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_x_dist.pdf'), bbox_inches='tight', transparent=True)

        return metrics

    def behavioral_metrics_in_out_crosswind_dist_nc(self, plot_individual=False):
        """
        function for plotting the inside and outside crosswind as a histogram.
        For this analysis we will just look at the inside and outside
        trajectories that returned to the edge. Look at the aggregate of all
        trajectories, pooled across animals.

        possible modifications to script:
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        x_in = []
        x_out = []
        x_del_in = []
        x_del_out = []

        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # inside calculations: x distance
            for key in list(di.keys()):
                df_i = di[key]
                df_i.ft_posx = np.abs(df_i.ft_posx-df_i.ft_posx.iloc[0])
                df_i.ft_posy = df_i.ft_posy-df.ft_posy.iloc[0]
                if return_to_edge(df_i):
                    x_in_temp = np.mean(df_i.ft_posx)
                    x_in.append(x_in_temp)
                    x_del_in.append(np.max(df_i.ft_posx))


            # outside calculations: x distance
            for key in list(do.keys()):
                df_o = do[key]
                df_o.ft_posx = np.abs(df_o.ft_posx-df_o.ft_posx.iloc[0])
                df_o.ft_posy = df_o.ft_posy-df_o.ft_posy.iloc[0]
                if return_to_edge(df_o):
                    x_out_temp = np.mean(df_o.ft_posx)
                    x_out.append(x_out_temp)
                    x_del_out.append(np.max(df_o.ft_posx))

            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        metrics = {
        'x_in': x_in,
        'x_out': x_out,
        'x_del_in': x_del_in,
        'x_del_out': x_del_out
        }

        max_x = 1000
        # plot the inside and outside cumulative x distance with a log scale
        fig, axs = plt.subplots(1,4, figsize=(16,4))
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[0].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[0].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[0].set_xlabel('crosswind pathlen. (mm)')
        print('x_in mean = ', np.mean(metrics['x_in']), 'x_out mean = ', np.mean(metrics['x_out']))
        # plot the inside and outside cumulative x distance with a linear scale
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[1].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[1].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[1].set_xlabel('crosswind pathlen. (mm)')

        max_x = 500
        # plot the inside and outside maximum x distance with a log scale
        sns.histplot(x=metrics['x_del_in'], element="step", fill=False, ax=axs[2], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['x_del_out'], element="step", fill=False, ax=axs[2], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[2].axvline(np.mean(metrics['x_del_in']), color=pl.inside_color, linestyle='--')
        axs[2].axvline(np.mean(metrics['x_del_out']), color=pl.outside_color, linestyle='--')
        axs[2].set_xlabel('max crosswind dist (mm)')
        print('x_in del = ', np.mean(metrics['x_del_in']), 'x_out del = ', np.mean(metrics['x_del_out']))

        # plot the inside and outside maximum x distance with a linear scale
        sns.histplot(x=metrics['x_del_in'], element="step", fill=False, ax=axs[3], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['x_del_out'], element="step", fill=False, ax=axs[3], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[3].axvline(np.mean(metrics['x_del_in']), color=pl.inside_color, linestyle='--')
        axs[3].axvline(np.mean(metrics['x_del_out']), color=pl.outside_color, linestyle='--')
        axs[3].set_xlabel('max crosswind dist (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_x_dist_nc.pdf'), bbox_inches='tight', transparent=True)

        return metrics

    def behavioral_metrics_in_out_crosswind_dist_paired(self, plot_individual=False):
        """
        function for plotting the average inside and outside times for each
        animal as a paired plot

        duture modifications
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        xdist_in = []
        xdist_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # add time for inside trajectories that return to the edge
            x_in = []
            returns = 0
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    returns+=1
                    x_steps = np.abs(np.gradient(df_i.ft_posx))
                    x_in_temp = np.sum(x_steps)
                    x_in.append(x_in_temp)

            # add time for outside trajectories that return to the edge
            x_out = []
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    x_steps = np.abs(np.gradient(df_o.ft_posx))
                    x_out_temp = np.sum(x_steps)
                    x_out.append(x_out_temp)

            # only count trial if it returned to the edge at least twice (n=3 for inside)
            if returns>=2:
                xdist_in.append(np.mean(x_in))
                xdist_out.append(np.mean(x_out))

            # plot individual traces
            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        fig, axs = plt.subplots(1,1, figsize = (1,2))
        axs = pl.paired_plot(axs, xdist_in, xdist_out, color1=pl.inside_color, color2=pl.outside_color, alpha=0.1)
        axs.set_xticks([0, 1])
        axs.set_xticklabels(['in', 'out'])
        axs.set_ylabel('average crosswind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_x_dist_paired.pdf'), bbox_inches='tight', transparent=True)
        return ydist_in, ydist_out

    def behavioral_metrics_in_out_crosswind_dist_paired_nc(self, plot_individual=False):
        """
        function for plotting the average inside and outside times for each
        animal as a paired plot
        nc = non-cumulative

        duture modifications
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        xdist_in = []
        xdist_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # add time for inside trajectories that return to the edge
            x_in = []
            returns = 0
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    returns+=1
                    x_steps = np.abs(df_i.ft_posx-df_i.ft_posx.iloc[0])
                    x_in_temp = np.mean(x_steps)
                    x_in.append(x_in_temp)

            # add time for outside trajectories that return to the edge
            x_out = []
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    x_steps = np.abs(df_o.ft_posx-df_o.ft_posx.iloc[0])
                    x_out_temp = np.mean(x_steps)
                    x_out.append(x_out_temp)

            # only count trial if it returned to the edge at least twice (n=3 for inside)
            if returns>=2:
                xdist_in.append(np.mean(x_in))
                xdist_out.append(np.mean(x_out))

            # plot individual traces
            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        fig, axs = plt.subplots(1,1, figsize = (1,2))
        axs = pl.paired_plot(axs, xdist_in, xdist_out, color1=pl.inside_color, color2=pl.outside_color, alpha=0.1)
        axs.set_xticks([0, 1])
        axs.set_xticklabels(['in', 'out'])
        axs.set_ylabel('average crosswind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_x_dist_paired_nc.pdf'), bbox_inches='tight', transparent=True)

    def behavioral_metrics_in_out_pathlength(self, plot_individual=False):
        """
        function for plotting the inside and outside pathlength.
        For this analysis we will just look at the inside and outside
        trajectories that returned to the edge. Look at the aggregate of all
        trajectories, pooled across animals.

        possible modifications to script:
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        p_in = []
        p_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # inside calculations: x distance
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    p_in_temp = fn.path_length(df_i.ft_posx.to_numpy(), df_i.ft_posy.to_numpy())
                    p_in.append(p_in_temp)

            # outside calculations: x distance
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    p_out_temp = fn.path_length(df_i.ft_posx.to_numpy(), df_i.ft_posy.to_numpy())
                    p_out.append(p_out_temp)

            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        metrics = {
        'p_in': p_in,
        'p_out': p_out,
        }

        max_x = 1000
        # plot the inside and outside time for individual animals log scale on x
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.inside_color, log_scale=True)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[0], bins=20, binrange=(0,np.log10(max_x)), color=pl.outside_color, log_scale=True)
        axs[0].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[0].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[0].set_xlabel('cumulative crosswind distance (mm)')

        # plot the inside and outside time for individual animals linear scale on x
        sns.histplot(x=metrics['x_in'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.inside_color)
        sns.histplot(x=metrics['x_out'], element="step", fill=False, ax=axs[1], bins=20, binrange=(-30,max_x), color=pl.outside_color)
        axs[1].axvline(np.mean(metrics['x_in']), color=pl.inside_color, linestyle='--')
        axs[1].axvline(np.mean(metrics['x_out']), color=pl.outside_color, linestyle='--')
        axs[1].set_xlabel('cumulative crosswind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_path.pdf'), bbox_inches='tight', transparent=True)

        return metrics

    def behavioral_metrics_in_out_pathlength_paired(self, plot_individual=False):
        """
        function for plotting the average inside and outside times for each
        animal as a paired plot

        duture modifications
        - add edge-tracking cutoff

        """
        def return_to_edge(df):
            xpos = df.ft_posx.to_numpy()
            if np.abs(xpos[-1]-xpos[0])<1:
                return True
            else:
                return False

        all_data = self.load_trajectories()
        sheet = self.sheet.log
        xdist_in = []
        xdist_out = []
        # iterate through each animal
        for i, log in enumerate(sheet):
            # load saved data
            df = all_data[log]['data']
            di = all_data[log]['di']
            do = all_data[log]['do']

            # add time for inside trajectories that return to the edge
            x_in = []
            returns = 0
            for key in list(di.keys()):
                df_i = di[key]
                if return_to_edge(df_i):
                    returns+=1
                    x_steps = np.abs(np.gradient(df_i.ft_posx))
                    x_in_temp = np.sum(x_steps)
                    x_in.append(x_in_temp)

            # add time for outside trajectories that return to the edge
            x_out = []
            for key in list(do.keys()):
                df_o = do[key]
                if return_to_edge(df_o):
                    x_steps = np.abs(np.gradient(df_o.ft_posx))
                    x_out_temp = np.sum(x_steps)
                    x_out.append(x_out_temp)

            # only count trial if it returned to the edge at least twice (n=3 for inside)
            if returns>=2:
                xdist_in.append(np.mean(x_in))
                xdist_out.append(np.mean(x_out))

            # plot individual traces
            if plot_individual:
                fig, axs = plt.subplots(1,1)
                axs.plot(df.ft_posx.to_numpy(), df.ft_posy.to_numpy())
                fig.suptitle('in time: '+str(t_in_avg)+ ' out time: ' + str(t_out_avg))


        fig, axs = plt.subplots(1,1, figsize = (1,2))
        axs = pl.paired_plot(axs, xdist_in, xdist_out, color1=pl.inside_color, color2=pl.outside_color, alpha=0.1)
        axs.set_xticks([0, 1])
        axs.set_xticklabels(['in', 'out'])
        axs.set_ylabel('average crosswind distance (mm)')

        fig.savefig(os.path.join(self.figurefol, 'inside_outside_x_dist_paired.pdf'), bbox_inches='tight', transparent=True)
        return ydist_in, ydist_out
