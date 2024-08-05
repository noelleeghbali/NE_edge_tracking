trials = {}
trials['20210512_60D05GCaMP7f_Fly1-001'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20210512_60D05GCaMP7f_Fly1-001',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked':'20210512_60D05GCaMP7f_Fly1-002',
'z_slices':[1,2,3],
'crop':False
}

trials['20210512_60D05GCaMP7f_Fly1-002'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20210512_60D05GCaMP7f_Fly1-002',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked':'20210512_60D05GCaMP7f_Fly1-001',
'z_slices':[1,2,3],
'crop':False
}

trials['20210513_60D05jGCaMP7f_Fly1_001'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20210513_60D05jGCaMP7f_Fly1_001',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked':'20210513_60D05jGCaMP7f_Fly1_002',
'z_slices':[1,2,3],
'crop':False
}

trials['20210513_60D05jGCaMP7f_Fly1_002'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20210513_60D05jGCaMP7f_Fly1_002',
'cell_type': 'epg',
'trial_type': 'replay',
'linked':'20210513_60D05jGCaMP7f_Fly1_001',
'z_slices':[1,2,3],
'crop':False
}

trials['20220126_60D05jGCaMP7f_Fly1-001'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220126_60D05jGCaMP7f_Fly1-001',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked':'20220126_60D05jGCaMP7f_Fly1-003',
'z_slices':[1,2,3],
'crop':False
}

trials['20220126_60D05jGCaMP7f_Fly1-002'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220126_60D05jGCaMP7f_Fly1-002',
'cell_type': 'epg',
'trial_type': 'replay',
'linked':'20220126_60D05jGCaMP7f_Fly1-001',
'z_slices':[1,2,3],
'crop':False
}

trials['20220126_60D05jGCaMP7f_Fly1-003'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220126_60D05jGCaMP7f_Fly1-003',
'cell_type': 'epg',
'trial_type': 'reply',
'linked':'20220126_60D05jGCaMP7f_Fly1-001',
'z_slices':[1,2,3],
'crop':False
}

trials['20220127_60D05jGCaMP7f_Fly1-001'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220127_60D05jGCaMP7f_Fly1-001',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked':'20220127_60D05jGCaMP7f_Fly1-002',
'z_slices':[1,2,3],
'crop':False
}

trials['20220127_60D05jGCaMP7f_Fly1-002'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220127_60D05jGCaMP7f_Fly1-002',
'cell_type': 'epg',
'trial_type': 'replay',
'linked':'20220127_60D05jGCaMP7f_Fly1-001',
'z_slices':[1,2,3],
'crop':False
}

trials['20220213_60D05jGCaMP7f_Fly1-001'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220213_60D05jGCaMP7f_Fly1-001',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked':None,
'z_slices':[1,2,3],
'crop':False
#'delta_t':-7.7
}

trials['20220213_60D05jGCaMP7f_Fly2-001'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220213_60D05jGCaMP7f_Fly2-001',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked': None,
'z_slices':[1,2,3],
'crop':False
}

trials['20220213_60D05jGCaMP7f_Fly2-002'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220213_60D05jGCaMP7f_Fly2-002',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked': None,
'z_slices':[1,2,3],
'crop':False
}

trials['20220213_60D05jGCaMP7f_Fly3-001'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220213_60D05jGCaMP7f_Fly3-001',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked':None,
'z_slices':[1,2,3],
'crop':False
}

trials['20220215_60D05jGCaMP7f_Fly1-001'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220215_60D05jGCaMP7f_Fly1-001',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked':None,
'z_slices':[1,2,3],
'crop':False
}

trials['20220215_60D05jGCaMP7f_Fly1-002'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220215_60D05jGCaMP7f_Fly1-002',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked':None,
'z_slices':[1,2,3],
'crop':False
}

trials['20220216_60DD05jGCaMP7f_Fly1-001'] = {
'fileloc': '/Volumes/LACIE/Andy/epg/20220216_60DD05jGCaMP7f_Fly1-001',
'cell_type': 'epg',
'trial_type': 'closed-loop',
'linked': None,
'z_slices':[1,2,3],
'crop':False
}


import skimage
from skimage import io, data, registration, filters, measure
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utilities import imaging as im
from src.utilities import funcs as fn
from src.utilities import plotting as pl

import importlib
import glob
import pandas as pd
import seaborn as sns
import importlib
#from astropy.stats import circcorrcoef
importlib.reload(im)
importlib.reload(fn)


# %% process data
%matplotlib
importlib.reload(im)
from pystackreg import StackReg
for trial in trials.keys():
    # if trial == '20210513_60D05jGCaMP7f_Fly1_001':

        # pull trial from trials dict
        name = trial
        folder = trials[name]['fileloc']
        trial_type = trials[name]['trial_type']
        slices = trials[name]['z_slices']
        kwargs = trials[name]

        # create imaging object
        ex = im.fly(name, folder, **kwargs)
        pb = im.PB2(name, folder, slices)

        # register images
        #ex.register_all_images(overwrite=True)
        ex.save_preprocessing()
        ex.save_postprocessing()



        # load image slices, save z slices, save z projection
        #pb.save_slices()

        # save pre and post processing
        #ex.save_preprocessing()
        #ex.save_postprocessing()


# %% plot data
figure_folder = '/Volumes/LACIE/Andy/epg/summary_plots'
%matplotlib
importlib.reload(im)
importlib.reload(fn)
importlib.reload(pl)

class grouped_epg_analysis:
    def __init__(self, trials):
        self.trials = trials
    def plot_save_individual_trajectories(self):
        """
        for each trial, plot and save the individual trajectory in its figure folder
        """
        trials = self.trials
        for trial in trials.keys():
            name = trial
            folder = trials[name]['fileloc']
            trial_type = trials[name]['trial_type']
            kwargs = trials[name]
            epg_plot = im.fly_plots(name, folder, trial_type)
            fig, axs = plt.subplots(1,1)
            epg_plot.plot_trajectory(axs)
            epg_plot.save_plot(fig, 'trajectory.pdf', summary=False)

    def plot_heading_phase(self):
        """
        plot the animal's heading vs corrected epg phase
        """

        trials = self.trials
        for trial in trials.keys():
            name = trial
            folder = trials[name]['fileloc']
            trial_type = trials[name]['trial_type']
            kwargs = trials[name]
            pb = im.PB(name, folder, **kwargs)
            epg_plot = im.fly_plots(name, folder, trial_type)
            phase = pb.subtract_phase_offset()
            heading = pb.ft2.ft_heading
            odor = pb.ft2.mfc2_stpt

            # calculate rolling phase difference
            tmp = pd.DataFrame({'diff': phase-heading.to_numpy()})
            def test(df):
                return fn.circmean(df)
            rolling_phase = tmp.rolling(window=10,center=False).apply(lambda x: test(x))

            # plot everything
            fig, axs = plt.subplots(2,1)
            axs[0].plot(phase, '.')
            axs[0].plot(heading, '.')
            # axs[0].title.set_text(trial+' '+np.array2string(circcorrcoef(phase, heading)))
            axs[0].title.set_text(trial+' '+np.array2string(fn.corrcoef(phase, heading, deg=False)))
            axs[1].plot(odor)
            axs2 = axs[1].twinx()
            axs2.plot(rolling_phase, 'k')
            axs2.set_ylim(-np.pi, np.pi)


            # fig, axs = plt.subplots(1,1)
            # sns.heatmap(pb.get_gloms(), ax=axs)

    def plot_bumps_in_out(self):
        """
        plot the pb bumps in and out of odor
        """
        fig, axs = plt.subplots(1, 1)
        trials = self.trials
        gloms_in = np.zeros((len(trials), 18)) # array for bumps of all experiments
        gloms_out = np.zeros((len(trials), 18)) # array for bumps of all experiments
        for i, trial in enumerate(trials.keys()):
            name = trial
            folder = trials[name]['fileloc']
            trial_type = trials[name]['trial_type']
            kwargs = trials[name]
            pb = im.PB(name, folder, **kwargs)

            # get bumps
            bumps_in, bumps_out = pb.bumps_in_out()

            # plot individual bumps
            axs.plot(np.mean(bumps_in, axis=0), color=pl.odor_color, alpha=0.2)
            axs.plot(np.mean(bumps_out, axis=0), color=pl.air_color, alpha=0.2)

            gloms_in[i,:] = np.mean(bumps_in, axis=0)
            gloms_out[i,:] = np.mean(bumps_out, axis=0)

        # calculte error
        #sem_in = scipy.stats.sem(gloms_in, axis=0, nan_policy='propagate')
        #sem_out = scipy.stats.sem(gloms_out, axis=0, nan_policy='propagate')

        axs.plot(np.nanmean(gloms_in, axis=0),'-s', color=pl.odor_color)
        axs.plot(np.nanmean(gloms_out, axis=0),'-s', color=pl.air_color)
        fig.savefig(os.path.join(figure_folder, 'average_bumps.pdf'))
        return fig

    def calc_circ_corr_coeff(self):
        """
        work in progress here.  Can't figure out the circular correlation coefficient,
        so for right now I'm just making a scatter plot
        """
        h = []
        p = []
        for i, trial in enumerate(trials.keys()):
            name = trial
            folder = trials[name]['fileloc']
            trial_type = trials[name]['trial_type']
            kwargs = trials[name]
            slices = trials[name]['z_slices']
            pb = im.PB2(name, folder, slices)
            pb.load_processed()
            ft2 = pb.ft2
            heading = ft2.ft_heading.to_numpy()
            phase = pb.subtract_phase_offset()
            h.append(heading)
            p.append(phase)
            # fig, axs = plt.subplots(1,1, figsize=(3,3))
            # axs.plot(heading, phase,'.', alpha=0.01)
            # axs.set_ylim(-np.pi, np.pi)
            # axs.set_xlim(-np.pi, np.pi)
        fig, axs = plt.subplots(1,1, figsize=(3,3))
        #axs.plot(np.concatenate(h),np.concatenate(p),'.', alpha=0.01)
        df = pd.DataFrame({'heading': np.concatenate(h), 'phase': np.concatenate(p)})
        g = sns.displot(df,x='heading', y='phase', ax=axs, cbar=True, thresh=0.01, stat='density')
        g.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), xticks=[-np.pi/2,0, np.pi/2], yticks=[-np.pi/2,0, np.pi/2], xticklabels=['-90', '0', '90'], yticklabels=['-90', '0', '90'])
        # g.set_ylim(-np.pi, np.pi)
        # g.set_xlim(-np.pi, np.pi)
        # g.set_xticks([-np.pi/2,0, np.pi/2])
        # g.set_xticklabels(['-90', '0', '90'])
        # g.set_yticks([-np.pi/2,0, np.pi/2])
        # g.set_yticklabels(['-90', '0', '90'])
        g.savefig(os.path.join(figure_folder, 'heading_v_phase.pdf'))
            #fig.suptitle(name+str(fn.circ_corr_coeff(heading, phase)))

class individual_plots:
    def __init__(self, trials):
        self.trials = trials

    def plot_example_heatmap_051221(self, range=[100,2000]):
        ix = np.arange(range[0], range[1])
        name = '20210512_60D05GCaMP7f_Fly1-001'
        folder = trials[name]['fileloc']
        trial_type = trials[name]['trial_type']
        kwargs = trials[name]
        slices = trials[name]['z_slices']
        pb = im.PB2(name, folder, slices)
        pb.load_processed()
        gloms = pb.get_layer_gloms()
        #gloms = gloms[ix,:]

        # plot heatmap
        fig, axs = plt.subplots(1,1, figsize=(10,2))
        sns.heatmap(gloms[ix].T, cmap='Blues', vmin=0.0, vmax=1.0, ax=axs)
        fig.savefig(os.path.join(figure_folder, 'example_heatmap.pdf'))

        # plot trajectory
        fig, axs = plt.subplots(1,1, figsize=(2,10))
        pl.plot_trajectory(pb.ft2.iloc[ix], axs=axs)
        pl.plot_trajectory_odor(pb.ft2.iloc[ix], axs=axs)
        fig.savefig(os.path.join(figure_folder, 'example_trajectory.pdf'))

        # plot phase and heading
        fig, axs = plt.subplots(2,1, figsize=(10,4))
        phase = pb.subtract_phase_offset()[ix]
        heading = pb.ft2.ft_heading.to_numpy()[ix]
        time = pb.ft2.seconds.to_numpy()[ix]
        odor = pb.ft2.mfc2_stpt.to_numpy()[ix]
        axs[0].plot(time, phase)
        axs[0].plot(time, heading)
        axs[1].plot(time, odor)
        fig.savefig(os.path.join(figure_folder, 'example_heading_v_phase.pdf'))

    def example_movie_02132022(self, range = [200,2200]):
        global stack, x, y, dx, dy
        import matplotlib.patches as patches
        from skimage import io
        from matplotlib import animation
        from matplotlib import rc
        rc('animation', html='jshtml')
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.animation import FuncAnimation
        from matplotlib.patches import Rectangle
        import matplotlib.cm as cm
        from matplotlib import colors

        # need ffmpeg for this to work correctly

        # create pb and plotting objects for trial
        name = '20220213_60D05jGCaMP7f_Fly2-001'
        folder = trials[name]['fileloc']
        trial_type = trials[name]['trial_type']
        kwargs = trials[name]
        pb = im.PB(name, folder, **kwargs)
        pb_fig = im.fly_plots(name, folder, trial_type)


        # frames to look at
        ix = np.arange(range[0], range[1])

        # read in ft data
        df = pb_fig.ft2
        df = df.iloc[ix]
        all_x_borders, all_y_borders = fn.find_borders(df)
        x = df.ft_posx.to_numpy()
        y = df.ft_posy.to_numpy()
        fn.find_borders(df)
        # calculate heading information for plotting arrows
        heading = df.ft_heading.to_numpy()
        hi = -heading+np.pi/2
        dx = np.cos(hi)
        dy = np.sin(hi)


        # load pb imaging  movie
        movie = os.path.join(pb.regfol, 'PB_stack_max.tif')
        stack = io.imread(movie)
        stack = stack[ix,:,:]
        stack = np.rot90(stack, axes=(2, 1))

        # define sampling and playback speed for movie
        downsample = 1
        fps_mult = 10
        speed_factor = downsample*fps_mult
        print('Video is sped up by', speed_factor, 'x')
        total_number_of_frames = int(np.round(len(df)/downsample)) # downsample to this number of frames
        num_movie_frames = total_number_of_frames #number of frames in movie, to test out making a shorter movie.
        delta_t = np.mean(np.diff(df.seconds)) # average time between frames
        memory= 3 #time in s to display memory
        memory = int(np.round(3/delta_t)) # adapt memory to number of frames
        interval = delta_t*1000/fps_mult



        # create a plot
        plt.style.use('dark_background')
        fig, axs = plt.subplots(2,2, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 4]})
        gs = axs[1, 1].get_gridspec()
        # remove the underlying axes
        for ax in axs[:, -1]:
            ax.remove()
        axbig = fig.add_subplot(gs[:, -1])
        #fig.tight_layout()

        axs[0,0].axis('off')

        x_quiv, y_quiv = np.meshgrid(np.arange(-20,20,4), np.arange(-20,20,4))
        dx_quiv = np.zeros(x_quiv.shape)
        dy_quiv = -0.5*np.ones(x_quiv.shape)
        axs[1,0].quiver(x_quiv, y_quiv, dx_quiv, dy_quiv, color='lightskyblue', alpha=0.1)
        axs[1,0].axis('equal')
        axs[1,0].set_xlim(-20, 20)
        axs[1,0].set_ylim(-20, 20)
        axs[1,0].axis('off')

        axbig.axis('equal')
        # x_quiv, y_quiv = np.meshgrid(np.arange(min(x)-100,max(x)+100,4), np.arange(min(x)-100,max(x)+100,4))
        # dx_quiv = np.zeros(x_quiv.shape)
        # dy_quiv = -1*np.ones(x_quiv.shape)
        # axbig.quiver(x_quiv, y_quiv, dx_quiv, dy_quiv, color='lightskyblue', alpha=0.2)
        rect = patches.Rectangle((all_x_borders[0][0], all_y_borders[0][0]), 10, np.max(y)-np.min(y), linewidth=1, edgecolor=None, facecolor='grey')
        axbig.add_patch(rect)
        axbig.set_xlim(min(x)-100, max(x)+100)
        axbig.set_ylim(min(y), max(y))
        axbig.axis('off')

        # initialize image and lines
        image = axs[0,0].imshow(stack[0,:,:], vmin=28, vmax=78, cmap='inferno', animated=True)
        line1, = axbig.plot(x[0], y[0])
        line2, = axbig.plot(x[0], y[0])
        line3, = axs[1,0].plot(x[0], y[0])
        arrow = axs[1,0].arrow(0,0,0,0,color='red', animated=True)

        def animate(frame, image, line1, line2, line3):
            image.set_array(stack[frame,:,:])

            # trajectory
            current_x = x[: frame+1]
            current_y = y[: frame+1]
            current_x2 = x[(frame-memory):(frame + 1)]
            current_y2 = y[(frame-memory):(frame + 1)]
            if frame>memory:
                x_mean = np.mean(x[(frame-memory):(frame + 1)])
                y_mean = np.mean(y[(frame-memory):(frame + 1)])
                current_x3 = x[(frame-memory):(frame + 1)]-x_mean
                current_y3 = y[(frame-memory):(frame + 1)]-y_mean
                arrow.set_data(x=current_x3[-1], y=current_y3[-1], dx=dx[frame], dy=dy[frame], head_width=1)
            else:
                current_x3 = np.nan
                current_y3 = np.nan
            line1.set_xdata(current_x)
            line1.set_ydata(current_y)
            line1.set_color("red")
            line1.set_alpha(0.5)

            line2.set_xdata(current_x2)
            line2.set_ydata(current_y2)
            line2.set_color("red")

            line3.set_xdata(current_x3)
            line3.set_ydata(current_y3)
            line3.set_color("red")

            return [image, line1, line2, line3,]

        animation = FuncAnimation(fig,animate,total_number_of_frames, fargs=[image,line1,line2,line3,], interval=interval)
        plt.show()
        save_name = os.path.join(pb_fig.plotfol, 'movie.mp4')
        animation.save(save_name)
        return stack, image

%matplotlib
grouped_epg_analysis(trials).calc_circ_corr_coeff()

# %%
name = '20210512_60D05GCaMP7f_Fly1-001'
folder = trials[name]['fileloc']
trial_type = trials[name]['trial_type']
kwargs = trials[name]
slices = trials[name]['z_slices']
pb = im.PB2(name, folder, slices)
pb.load_processed()
gloms = pb.get_layer_gloms()

# %%
ft2 = pb.ft2
heading = np.rad2deg(ft2.ft_heading.to_numpy())
plt.plot(phase)
phase = np.rad2deg(pb.get_phase())


x=heading
y=phase
import cmath
def mean(angles, deg=True):
    '''Circular mean of angle data(default to degree)
    '''
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
    return round(np.rad2deg(mean) if deg else mean, 7)

deg = True
convert = np.pi / 180.0 if deg else 1
sx = np.frompyfunc(np.sin, 1, 1)((x - mean(x, deg)) * convert)
sy = np.frompyfunc(np.sin, 1, 1)((y - mean(y, deg)) * convert)
r = (sx * sy).sum() / np.sqrt((sx ** 2).sum() * (sy ** 2).sum())
r
