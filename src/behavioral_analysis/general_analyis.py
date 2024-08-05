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
from src.behavioral_analysis import random_walk as rw
from scipy import interpolate, stats
from numba import jit
import seaborn as sns
import time
from colormap.colors import hex2rgb
importlib.reload(dr)
importlib.reload(im)
importlib.reload(pl)
importlib.reload(fn)

sns.set(font="Arial")
sns.set(font_scale=0.6)
sns.set_style('white')

def example_trajectories():
    """
    function for displaying example trajectories for the 0,45,90 degree plumes
    """
    fig, axs = plt.subplots(1,1, figsize=(5,5))

    ex0 = '09082020-134131_Fly9_CantonS_002.log'
    ex45 = '08292023-112627_Canton-s_45_naive_Fly5.log'
    ex90 = '08282023-192013_Canton-s_horizontal_Fly2.log'

    logfol = '/Volumes/Andy/logs'

    for angle, log in zip([0,45,90], [ex0,ex45,ex90]):
        df = fn.read_log(os.path.join(logfol, log))
        if angle==0:
            df['instrip'] = np.where(np.abs(df.mfc3_stpt)>0, True, False)
        idx_start = df.index[df.instrip==True].tolist()[0]
        idx_end = df.index[df.instrip==True].tolist()[-1]
        df = df.iloc[idx_start:idx_end]
        x = df.ft_posx.to_numpy()
        y = df.ft_posy.to_numpy()
        x = x-x[0]
        y = y-y[0]

        x_space=1200
        plume_color = 'k'
        plume_alpha = 0.5
        if angle == 0:
            color = '#19344a'
            color = pl.inside_color
            x=-x
            axs.plot([-25,-25],[0,1000],color=plume_color, alpha=plume_alpha)
            axs.plot([25,25],[0,1000],color=plume_color, alpha=plume_alpha)
            axs.plot(x,y, color=color)
            axs.plot(x[0],y[0], 'o',  color=color)
        elif angle == 45:
            color = '#4b4467'
            color = pl.inside_color
            y_end = 800
            x_dist = 50/np.sin(np.pi/4)/2
            x+=x_space
            axs.plot([-x_dist+x_space,-x_dist+y_end+x_space],[0,y_end],color=plume_color, alpha=plume_alpha)
            axs.plot([x_dist+x_space,x_dist+y_end+x_space],[0,y_end],color=plume_color, alpha=plume_alpha)
            axs.plot(x,y, color=color)
            axs.plot(x[0],y[0], 'o', color=color)
        elif angle == 90:
            color = '#388ab5'
            color = pl.inside_color
            x+=2*x_space
            axs.plot([0+2*x_space,1000+2*x_space],[50,50],color=plume_color, alpha=plume_alpha)
            axs.plot([0+2*x_space,1000+2*x_space],[0,0],color=plume_color, alpha=plume_alpha)
            axs.plot(x,y, color=color)
            axs.plot(x[0],y[0], 'o', color=color)
        
        # scalebar
        axs.plot([200,300], [500,500], 'k')
        axs.text(100,450,'100 mm')
        fig.tight_layout()
        axs.axis('equal')

    savename = os.path.join(os.getcwd(), 'figures', '0_45_90_examples.pdf')
    fig.savefig(savename)

def example_heatmaps():
    fig, axs = plt.subplots(1,1, figsize=(5,5))
    cmap='Blues'

    ex0 = os.path.join(os.getcwd(), 'data', 'constant_gradient','pickles', 'heatmaps.p')
    ex45 = os.path.join(os.getcwd(), 'data', 'plume_45','pickles', 'heatmaps.p')
    ex90 = os.path.join(os.getcwd(), 'data', 'plume_90','pickles', 'heatmaps.p')

    for angle, log in zip([0,45,90], [ex0,ex45,ex90]):
        d = fn.load_obj(log)
        res=5
        fly_density = d['fly_density']
        vmin = np.percentile(fly_density[fly_density>0],0)
        vmax = np.percentile(fly_density[fly_density>0],95)
        x_space=1200
        plume_color = 'k'
        plume_alpha = 0.5
        if angle == 0:
            color='k'
            x_bounds = np.arange(-200, 200, res)
            y_bounds = np.arange(0, 1100, res)

            axs.plot([-25,-25],[0,1000],color=plume_color, alpha=plume_alpha)
            axs.plot([25,25],[0,1000],color=plume_color, alpha=plume_alpha)
            axs.plot(0,0, 'o',  color=color)

            cmap = pl.create_linear_cm(RGB=hex2rgb('#19344a'))
            cmap='binary'
        elif angle == 45:
            color='k'
            x_bounds = np.arange(-50, 1000, res)+x_space
            y_bounds = np.arange(0, 1000, res)
            
            y_end = 800
            x_dist = 50/np.sin(np.pi/4)/2
            
            axs.plot([-x_dist+x_space,-x_dist+y_end+x_space],[0,y_end],color=plume_color, alpha=plume_alpha)
            axs.plot([x_dist+x_space,x_dist+y_end+x_space],[0,y_end],color=plume_color, alpha=plume_alpha)
            axs.plot(x_space,0, 'o',  color=color)
            
            cmap = pl.create_linear_cm(RGB=hex2rgb('#4b4467'))
            cmap='binary'
        elif angle == 90:
            color='k'
            x_bounds = np.arange(-200, 1200, res)+2*x_space
            y_bounds = np.arange(-500, 500, res)#+50

            axs.plot([0+2*x_space,1000+2*x_space],[-50,-50],color=plume_color, alpha=plume_alpha)
            axs.plot([0+2*x_space,1000+2*x_space],[0,0],color=plume_color, alpha=plume_alpha)
            axs.plot(2*x_space,0, 'o', color=color)

            cmap = pl.create_linear_cm(RGB=hex2rgb('#388ab5'))
            cmap='binary'
        alphas=np.zeros(fly_density.shape)
        alphas[fly_density>0.0001]=1
        axs.imshow(fly_density, cmap=cmap, vmin=vmin,vmax = vmax, rasterized=True, extent=(min(x_bounds), max(x_bounds), min(y_bounds), max(y_bounds)), alpha=alphas)
        axs.set_ylim(-100,1000)
        axs.set_xlim(-200,1000)

        # scalebar
        axs.plot([200,300], [500,500], 'k')
        axs.text(100,450,'100 mm')

        fig.tight_layout()
        axs.axis('equal')
        #axs.axis('off')
        savename = os.path.join(os.getcwd(), 'figures', '0_45_90_heatmap.pdf')
        fig.savefig(savename)

def compare_jump_vertical_efficiency():
    sns.set(font="Arial")
    sns.set(font_scale=0.6)
    sns.set_style('white')

    vert = rw.rw_model(experiment=0)
    jump = rw.rw_model(experiment='jump')

    fig,axs = plt.subplots(1,1, figsize=(2,2))
    fig1,axs1 = plt.subplots(1,1, figsize=(3,2))

    for e, exp in zip(['vert', 'jump'],[vert, jump]):
        if e == 'vert':
            c_real = '#FFDB87'
            c_fictive = '#DEB85F'
        elif e == 'jump':
            c_real = '#214491'
            c_fictive = '#5F87DE'
            
        _,_,_,trajectories  = exp.load_outside_dp_segments()
        returns = exp.consecutive_outside_trajectories()
        plume_angle = exp.angle
        all_dist_away = []
        
        for key in list(trajectories.keys()):
            xy = trajectories[key]
            dist_away = exp.orthogonal_distance(xy[:,0], xy[:,1],plume_angle)
            dist_away = np.max(dist_away)
            all_dist_away.append(dist_away)
            _, pathlen = fn.path_length(xy[:,0], xy[:,1])
            pathlen = pathlen/1000 # convert to meters
            axs.plot(dist_away, pathlen, '*', color=c_real)
        axs.plot([0,350], [0, 0.7], 'k')

        upwind_a = [1.]
        for i, a in enumerate(upwind_a):
            savename = os.path.join(exp.picklefol, str(exp.experiment)+'_'+'outside_fictive_trajectories'+'_'+str(a)+'.p')

            # load the data
            results = fn.load_obj(savename)
            n_traj = results['n_traj']
            failures = results['failures']
            orthogonal_distances = results['orthogonal_distances']
            pathlengths = results['pathlengths']
            pathlengths = np.array(pathlengths)/1000

            # plot efficiencies
            axs.plot(orthogonal_distances, pathlengths, '.',linestyle="None", color=c_fictive, alpha=0.5)
            axs.set_xlabel('distance away from plume (mm)')
            axs.set_ylabel('total path length (m)')
            sns.despine(ax=axs)

            # plot the consecutive outside trajectories
            n = np.arange(1,np.max(returns))
            success = n_traj/(failures+n_traj)
            probability = success**n
            axs1.plot(n, probability, color=c_fictive)
            axs1.set_ylim(-0.05,1.05)
            axs1.set_xlabel('number of consecutive outside trajectories')
            axs1.set_ylabel('P(n returns)')

            # plot the actual number of consecutive outside trajectories
            axs1r = axs1.twinx()
            sns.histplot(x=returns,ax=axs1r, fill=False, element='step',color=c_real)

            # find max bin height
            max_bin = axs1r.lines[0].properties()['data'][1].max()
            axs1r.set_ylim(-0.05*max_bin, 1.05*max_bin)
            axs1r.set_ylabel('returns (n)', color=c_real)
            if e=='jump':
                axs1r.spines.right.set_position(("axes", 1.4))
    fig.tight_layout()
    savename = os.path.join(os.getcwd(), 'figures', 'compare_vertical_jump_efficiency.pdf')
    fig.savefig(savename)
    fig1.tight_layout()
    savename = os.path.join(os.getcwd(), 'figures', 'compare_vertical_jump_consecutive.pdf')
    fig.savefig(savename)


