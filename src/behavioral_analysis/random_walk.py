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
import random
importlib.reload(dr)
importlib.reload(im)
importlib.reload(pl)
importlib.reload(fn)


class rw_model():
    """
    class for making random walk model.  Will be using the the constant and gradient vertical plumes to generate the turn angle and run lengths

    functions adapted from average_trajectories.py
    """
    def __init__(self, directory='M1', experiment = 0):
        d = dr.drive_hookup()
        # your working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'
        elif directory == 'Andy':
            self.cwd = '/Volumes/Andy/GitHub/edge-tracking'
        
        # experimental specification
        self.experiment=experiment
        if self.experiment=='jump':
            self.angle=0
        else:
            self.angle = int(self.experiment)

        # This google sheet contains all logs from the vertical plume
        self.sheet_id = '1Is1t3UtMAycrvpSMvEf6j2Gpc4b5jkEdm7yTIEAxfw8'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df

        # specify pickle folder and pickle name
        self.picklefol = os.path.join(self.cwd, 'data/random_walk/pickles')
        if not os.path.exists(self.picklefol):
            os.makedirs(self.picklefol)
        
        # pickle file names for split trajectories. Will pull from other analyis for jump, 45, 90
        if experiment == 0:
            self.picklesname = os.path.join(self.picklefol, 'et_manuscript_random_walk.p')
        elif experiment == 45:
            self.picklesname = os.path.join(self.cwd, 'data/plume_45/pickles/et_manuscript_45.p')
        elif experiment == 90:
            self.picklesname = os.path.join(self.cwd, 'data/plume_90/pickles/et_manuscript_90.p')
        elif experiment == 'jump':
            self.picklesname = os.path.join(self.cwd, 'data/jump/pickles/et_manuscript_jump.p')

        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/random_walk')
        if not os.path.exists(self.figurefol):
            os.makedirs(self.figurefol)

    def load_trajectories(self):
        """
        open the pickled data stored from split_trajectories()
        """
        all_data = fn.load_obj(self.picklesname)
        return all_data
    
    def create_outside_dp_segments(self, flip=True):
        """
        load outside trajectories, and run DP algorithm to simplify them.  Store
        results in DF and pickle

        - adapted from average_trajectories.py
        - now adapted only for finding the run lengths and path lengths of the vertical plume
        -
        """
        # load all trajectories
        all_data = self.load_trajectories()

        # iterate through all trial
        d = []
        exit_heading = []
        entry_heading = []
        trajectories = {}
        max_distance = []

        for key in list(all_data.keys()):
            fig, axs = plt.subplots(1,1)

            # select outside trajectories
            do = all_data[key]['do']
            rot_angle, direction = self.find_rotation_angle(all_data[key])
            
            for key_o in list(do.keys()):
                df = do[key_o]
                
                # center all outside trajecories
                x = df['ft_posx']-df['ft_posx'].iloc[0]
                y = df['ft_posy']-df['ft_posy'].iloc[0]
                x = x.to_numpy()
                y = y.to_numpy()
                
                # rotate the trajectories
                x,y = fn.coordinate_rotation(x,y,rot_angle)

                # find the maximum orthogonal distance away
                max_d = np.min(-np.abs(x))

                # flip the outside trajectories so they are all on the left side
                if flip:
                    x = -np.abs(x)
                else:
                    x = (2*random.randint(0,1)-1)*x

                # only include outside segments where fly returns to the edge
                # return to the edge is defined as final x position being within 1mm of starting x position


                if self.experiment == 'jump': # return condition for jumping plume
                    if not 19<np.abs(x[0]-x[-1])<21:
                        continue
                else: # return condition for all other plumes
                    if np.abs(x[-1]-x[0])>1:
                        continue

                # save the maximum distance
                max_distance.append(max_d)

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

                axs.plot(simplified[:,0], simplified[:,1])

        df = pd.DataFrame(d)
        df = df.explode(['lengths', 'angles', 'heading'])
        max_distance = np.max(np.abs(max_distance))

        results = {
            'df':df,
            'exit_heading':exit_heading,
            'entry_heading':entry_heading,
            'trajectories':trajectories,
            'max_distance':max_distance
        }

        # save results
        savename = os.path.join(self.picklefol, str(self.experiment)+'_outside_DP_segments.p')
        fn.save_obj(results, savename)
        return results

    def load_outside_dp_segments(self):
        """
        load the outside DP segments pickled in create_outside_dp_segments
        """
        savename = os.path.join(self.picklefol, str(self.experiment)+'_outside_DP_segments.p')
        results = fn.load_obj(savename)
        df = results['df']
        exit_heading = results['exit_heading']
        entry_heading = results['entry_heading']
        trajectories = results['trajectories']
        max_distance = results['max_distance']
        return df, exit_heading, entry_heading, trajectories, max_distance
    
    def orthogonal_distance(self, x,y, plume_angle):
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

        if self.experiment=='jump':
            plume_angle = 0
            thresh = -20
        else:
            # convert plume direction to cartesian coordinates
            plume_angle = np.deg2rad(int(self.experiment))
            thresh=0
        # plume_angle = np.array([plume_angle])
        # plume_angle = fn.conv_cart_upwind(plume_angle)
        # ang1 = plume_angle
        # ang2 = fn.wrap(plume_angle + np.pi)
        # print('plume angle is '+ str(plume_angle)+' ang1 is '+str(ang1)+' ang2 is '+str(ang2))

        # success metrics
        failures = 0

        # load DP segments
        df, exit_heading, entry_heading, trajectories, max_distance = self.load_outside_dp_segments()

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
            while self.orthogonal_distance(xy[-1][0], xy[-1][1], plume_angle)>thresh:
                L = df.sample().lengths.iloc[0]
                new = next_point(L,heading,xy[-1])
                xy = np.vstack((xy,new))

                # distance cutoff -- if fly goes more than 200 away from edge, terminate trajectory
                if self.orthogonal_distance(xy[-1,0], xy[-1,1],plume_angle)>max_distance:
                    print('failure')
                    failures+=1
                    break
                # mult = 2*random.randint(0, 1)-1
                mult = biased_turn(heading, a)
                #print(heading, mult)
                heading = heading + mult*np.abs(df.sample().angles.iloc[0])


            if self.orthogonal_distance(xy[-1,0], xy[-1,1],plume_angle)<0: #save trajectories that make it back
                print('success')
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

        savename = os.path.join(self.picklefol, str(self.experiment)+'_'+'outside_fictive_trajectories'+'_'+str(a)+'.p')
        fn.save_obj(results, savename)

    def fictive_traj_upwind_bias(self):
        """
        create fictive trajectories with different upwind biases using the sine
        method for upwind bias
        """
        upwind_a = [0.,1.]
        for i,a in enumerate(upwind_a):
            self.make_random_segment_library(a=a, min_dist=1)

    def plot_dist_turns(self):
        """
        make a plot of the distribution of turn angles
        """
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        fig, axs = plt.subplots(1,1, figsize = (1.5,1.5))
        df, exit_heading, entry_heading, trajectories, max_distance = self.load_outside_dp_segments()
        angles = np.rad2deg(df.angles.to_list())
        sns.histplot(angles, ax = axs, element = 'step', color='k')
        axs.set_xlabel('turn angle (degrees)')
        axs.set_xticks([-90,0,90])
        fig.tight_layout()
        sns.despine()
        fig.savefig(os.path.join(self.figurefol, 'turn_angle_distribution.pdf'))

    def find_rotation_angle(self, d):
        """
        find angle for rotation based on the experiment
        d: dict for each trial created in split_trajectories()
        """
        if 'direction' in list(d.keys()):
            direction = d['direction']
        else:
            direction = 'NA'

        angle = self.angle
        angle_from_horizontal = np.pi/2 - angle*np.pi/180
        if direction == 'right':
            rot_angle = -angle*np.pi/180
        elif direction == 'left':
            rot_angle = +angle*np.pi/180
        elif direction == 'NA':
            rot_angle = -angle*np.pi/180
        return rot_angle, direction

    def consecutive_outside_trajectories(self, flip=True):
        """
        calculate how many consecutive outside trajectories
        """
        all_data = self.load_trajectories()
        num_cons=[]
        for key in list(all_data.keys()):
            rot_angle, direction = self.find_rotation_angle(all_data[key])
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
                # return to the edge is defined as final x position being within 1mm of starting x position for 0, 45, 90
                # return to the edge is defined as final x position 20 mm from starting x position
                if self.experiment == 'jump':
                    if 19<np.abs(x[0]-x[-1])<21:
                        cons+=1
                else:    
                    if np.abs(x[-1]-x[0])<1: # fly makes it back
                        cons+=1
            num_cons.append(cons)
            if False:
                fig, axs = plt.subplots(1,1)
                pl.plot_trajectory(data, axs)
                pl.plot_trajectory_odor(data, axs)
                axs.text(0,0,str(cons))
        return num_cons

    def compare_fictive_real_efficiency(self, axs=None, axs1=None, axs2=None):
        """
        load fictive trajectories and compare them to the DP simplification of real trajectories
        """
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('ticks')

        # figure for efficiency
        if axs is None:
            fig, axs = plt.subplots(1,1, figsize=(2,2))
        else:
            fig = axs.figure

        # figure for consecutive trajectories
        if axs1 is None:
            fig1, axs1 = plt.subplots(1,1, figsize=(1.75,2))
        else:
            fig1 = axs1.figure

        # figure for distributions of distance away
        if axs2 is None:
            fig2, axs2 = plt.subplots(1,1, figsize=(2,2))
        else:
            fig2 = axs2.figure

        # load real segments
        _,_,_, trajectories,_  = self.load_outside_dp_segments()

        # what is the plume angle?
        plume_angle = self.angle

        # need to calculate how many consecutive returns the animal makes
        returns = self.consecutive_outside_trajectories()

        # collected efficiency variables for model and real data
        D = []
        L = []
        types = []

        # plot the efficiency for the real data
        all_dist_away = []
        all_path_length = []
        f,a = plt.subplots(1,1)
        for key in list(trajectories.keys()):
            
            xy = trajectories[key]
            dist_away = self.orthogonal_distance(xy[:,0], xy[:,1],plume_angle)
            a.plot(xy[:,0], xy[:,1])
            dist_away = np.max(dist_away)
            all_dist_away.append(dist_away)
            _, pathlen = fn.path_length(xy[:,0], xy[:,1])
            pathlen = pathlen/1000 # convert to meters
            all_path_length.append(pathlen)
            axs.plot(dist_away, pathlen, '*', color='grey')
        axs.plot([0,350], [0, 0.7], 'k')

        # add distances and path lengths for eventual joint plot
        D+=all_dist_away
        L+= all_path_length
        types+=['real']*len(all_dist_away)
        
        # make a plt showing the distribution of distances away for the real trajectories
        bins=25
        binrange=[0,350]
        sns.histplot(x=all_dist_away, stat='density', bins=bins, binrange=binrange, kde=True, color='grey', ax=axs2)

        # load all the fictive trajectories
        upwind_a_all = np.round(np.linspace(0,1,11),1)
        #upwind_a = [0.,0.5,1.]
        upwind_a = [0., 1.] # settled on upwind a values of 0 and 1 for this figure
        colors = sns.color_palette("rocket",len(upwind_a_all))
        for i, a in enumerate(upwind_a):
            savename = os.path.join(self.picklefol, str(self.experiment)+'_'+'outside_fictive_trajectories'+'_'+str(a)+'.p')

            # load the data
            results = fn.load_obj(savename)
            n_traj = results['n_traj']
            failures = results['failures']
            orthogonal_distances = results['orthogonal_distances']
            pathlengths = results['pathlengths']
            pathlengths = np.array(pathlengths)/1000
            D+=orthogonal_distances
            L+= pathlengths.tolist()
            types+=[str(a)]*len(pathlengths)


            # color for the selected value of a
            # color_ix = np.where(upwind_a_all==a)[0][0]
            # color = colors[color_ix]
            color = ['k', 'darkred']

            # plot the efficiencies
            axs.plot(orthogonal_distances, pathlengths, '.',linestyle="None", color=color[i], alpha=1)
            axs.set_xlabel('distance away from plume (mm)')
            axs.set_ylabel('total path length (m)')
            sns.despine(ax=axs)

            # plot distribution of distances away
            sns.histplot(x=orthogonal_distances, stat='density', bins=bins, binrange=binrange, kde=True, ax=axs2, color=color[i])
            axs2.set_xlabel('distance away from plume (mm)')

            # plot the consecutive outside trajectories
            n = np.arange(1,np.max(returns))
            success = n_traj/(failures+n_traj)
            probability = success**n
            axs1.plot(n, probability, color=color[i])
            axs1.set_ylim(-0.05,1.05)
            axs1.set_xlabel('number of consecutive outside trajectories')
            axs1.set_ylabel('P(n returns)')

            # plot the actual number of consecutive outside trajectories
            axs1r = axs1.twinx()
            sns.histplot(x=returns,ax=axs1r, fill=False, element='step',color='grey')
            axs1.spines[['top']].set_visible(False)
            axs1r.spines[['top']].set_visible(False)

            # find max bin height
            max_bin = axs1r.lines[0].properties()['data'][1].max()
            axs1r.set_ylim(-0.05*max_bin, 1.05*max_bin)
            axs1r.set_ylabel('returns (n)')

        # save pathlength plot
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, str(self.experiment)+'_compare_fictive_real_efficiency.pdf'))

        # save consecutive plot
        fig1.tight_layout()
        fig1.savefig(os.path.join(self.figurefol, str(self.experiment)+'_compare_fictive_real_efficiency_consecutive.pdf'))

        # save the distribution plot
        sns.despine(ax=axs2)
        fig2.tight_layout()
        fig2.savefig(os.path.join(self.figurefol, str(self.experiment)+'_compare_fictive_real_orthogonal_distributions.pdf'))

        # joint plot
        dpl = {'type': types, 'distance':D, 'pathlength': L}
        df = pd.DataFrame(dpl)
        df = df[df.distance<100]
        fig = sns.jointplot(data=df, x='distance', y='pathlength',hue='type', height=4, ratio=1, ylim=(0,np.max(df.pathlength)), xlim=(0,np.max(df.distance)), palette = ['grey', 'k', 'darkred'])
        fig.savefig(os.path.join(self.figurefol, str(self.experiment)+'_compare_fictive_real_efficiency_consecutive_joint.pdf'))

        # violin plot distance
        fig, axs = plt.subplots(1,1, figsize=(2,2))
        df = pd.DataFrame(dpl)
        sns.violinplot(data=df, x='type', y='distance', cut=0, palette = ['grey', 'k', 'darkred'], linewidth=0, log_scale=True)
        fig.tight_layout
        fig.savefig(os.path.join(self.figurefol, str(self.experiment)+'_compare_fictive_real_efficiency_distance.pdf'))

        # violin plot path length
        fig, axs = plt.subplots(1,1, figsize=(2,2))
        df = pd.DataFrame(dpl)
        sns.violinplot(data=df, x='type', y='pathlength', cut=0, palette = ['grey', 'k', 'darkred'], linewidth=0, log_scale=True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, str(self.experiment)+'_compare_fictive_real_efficiency_pathlength.pdf'))

        return df
        
    def run_length_turn_angle_distributions(self):
        """
        make plots of the distributions for the run lengths and turn angles
        """
        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')


        # load data
        df, exit_heading, entry_heading, trajectories,_ = self.load_outside_dp_segments()
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
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'turn_angle_distribution_small.pdf'))

        # run length plot
        fig, axs = plt.subplots(1,1, figsize=(1,1))
        sns.histplot(lengths, element='step', bins=20, fill=False, color='grey')
        sns.despine()
        axs.set_xlabel('run lengths (mm)', fontsize=7, labelpad=-1)
        axs.set_ylabel('count', fontsize=7, labelpad=-1)
        axs.tick_params(axis='x', pad=-3)
        axs.tick_params(axis='y', pad=-3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'run_length_distribution_small.pdf'))

    def find_upwind_bias(self):
        """
        calculate how the upwind bias in the real trajectories compares to the
        upwind bias in the fictive trajectories.  Upwind bias is a function of
        parameter 'a' in the biased turn function used to create the fictive
        fictive trajectories.  Upwind bias is calculated as the change in y
        position divided by the pathlength for any given outside trajectory.
        """
        #figure for plotting upwind bias vs parameter a
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')
        fig, axs = plt.subplots(1,1, figsize=(2,2))
        # load the real data
        df, exit_heading, entry_heading, trajectories,_  = self.load_outside_dp_segments()

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
        upwind_bias_real = np.mean(delta_y/pathlength)
        #axs.plot([0,1], [upwind_bias, upwind_bias], 'k')

        # load all the fictive trajectories
        upwind_a = np.round(np.linspace(0,1,11),1)
        colors = sns.color_palette("rocket",len(upwind_a))
        for i, a in enumerate(upwind_a):
            # reset delta_y, pathlength
            delta_y = []
            pathlength = []

            # load fictive data
            savename = os.path.join(self.picklefol, str(self.experiment)+'_'+'outside_fictive_trajectories'+'_'+str(a)+'.p')
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

        axs.plot(-0.2, upwind_bias_real, '*', color='grey')
        axs.set_xticks([-0.2,0,0.5,1])
        axs.set_xticklabels(['real', '0', '0.5', '1'])
        sns.despine()
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol,'upwind_bias.pdf'))

        # make a figure showing violin plot
        fig2, axs2 = plt.subplots(1,1, figsize=(2,2))
        df = pd.DataFrame(d_upwind_bias)
        sns.violinplot(data = df, x='a', y='bias', ax=axs2)
        sns.despine()
        fig.tight_layout()
        return df
    