from src.utilities import imaging as im
from src.utilities import plotting as pl
import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import numpy as np
import seaborn as sns
from skimage import io
from src.drive import drive as dr
from src.utilities import funcs as fn
from src.utilities import plotting as pl
from scipy import stats
matplotlib.rcParams['pdf.fonttype'] = 42

importlib.reload(dr)
importlib.reload(pl)
importlib.reload(fn)
importlib.reload(im)

class hdeltac:
    def __init__(self, celltype='hdeltac'):
        d = dr.drive_hookup()

        # load google sheet with experiment information
        if celltype == 'hdeltac':
            self.sheet_id = '1CnFEOQ06Nc8qiyaxab7mUQuWu_UHiFBRQj2C6ZhFfs4'
            self.datafol = "/Volumes/LACIE/Andy/hdeltac"
        elif celltype == 'hdeltab':
            self.sheet_id = '1RhzNSO-wDWqFJ3RLa7zPCrLgd3Lj94UY_VKHTi9F_z0'
            self.datafol = "/Volumes/LACIE/Andy/hdb"
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')

        # df.analyze.loc[df.analyze == 'TRUE'] = True
        # df.analyze.loc[df.analyze == 'FALSE'] = False
        # df.DP_analysis.loc[df.DP_analysis == 'TRUE'] = True
        # df.DP_analysis.loc[df.DP_analysis == 'FALSE'] = False
        # df.crop.loc[df.crop == 'FALSE'] = False
        df = df.replace('TRUE',True, regex=True)
        df = df.replace('FALSE',False, regex=True)
        self.sheet = df


    def register(self):
        """
        register all images
        """
        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            if d['analyze']:
                name = d['folder']
                folderloc = os.path.join(self.datafol, name)
                print(name, folderloc)
                ex = im.fly(name, folderloc, **d)
                print(ex.name, ex.folderloc)
                try:
                    ex.register_all_images(overwrite=True)
                except:
                    return ex

    def processing(self):
        """
        pre and post process data. Create hdc_example object to create
        glomeruli/wedges for each example
        """
        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            if d['analyze']:
                print(d)
                hdc = hdc_example(d, self.datafol)

                # get the FB wedges for each layer
                if hdc.FB_masks is not None:
                    for i, mask in enumerate(hdc.FB_masks):
                        print(mask)
                        wedges = hdc.fb.get_wedge_rois(mask)

                # get the EB wedges
                if hasattr(hdc, 'eb'):
                    wedges = hdc.eb.get_wedge_rois()

                # get the PB glomeruli
                if hasattr(hdc, 'pb'):
                    gloms = hdc.pb.get_gloms()

                # do pre and post processing
                name = d['folder']
                folderloc = os.path.join(self.datafol, name)
                print('FOLDER LOCATION IS',folderloc)
                kwargs = d
                print('kwargs = ', kwargs)
                ex = im.fly(name, folderloc, **kwargs)
                print('CROP is', ex.crop)
                ex.save_preprocessing()
                ex.save_postprocessing()


class hdc_example:
    """
    class for plotting and analyzing individual examples.  Could eventually fold
    this into hdeltc class
    """
    def __init__(self, dict, datafol):
        for key in list(dict.keys()):
            setattr(self,key, dict[key])
        # convert these columns to arrays
        keys = ['FB_slices', 'PB_slices', 'EB_slices']
        for key in keys:
            string = dict[key]
            if string == 'None':
                setattr(self, key, None)
            else:
                setattr(self, key, [int(s) for s in string.split(',')])

        keys = ['FB_masks']
        for key in keys:
            string = dict[key]
            if string == 'None':
                setattr(self, key, None)
            else:
                setattr(self, key, [mask for mask in string.split(',')])

        # specify where the figures should be stored
        cwd = os.getcwd()
        if self.cell_type == 'hdeltab':
            save_folder = os.path.join(cwd, 'figures/hdb', self.folder)
        elif self.cell_type == 'hdeltac':
            save_folder = os.path.join(cwd, 'figures/hdc', self.folder)
        save_folder = os.path.join('/Volumes/LACIE/Andy/hdeltac/figures', self.folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        self.figure_folder = save_folder

        # specify where the data is stored
        self.datafol = os.path.join(datafol, self.folder)

        if self.FB_masks is not None:
            self.fb = im.FSB2(self.folder, self.datafol, self.FB_slices)
        if self.PB_slices is not None:
            self.pb = im.PB2(self.folder, self.datafol, self.PB_slices)
        if self.EB_slices is not None:
            self.eb = im.EB(self.folder, self.datafol, self.EB_slices)

        self.fly = im.fly(self.folder, self.datafol, **dict)

    def save_all_info_eb_fb(self):
        fb = self.fb
        eb = self.eb
        fb.load_processed()
        ft2 = fb.ft2

        # calculate eb phases
        wedges_eb = eb.get_layer_wedges()
        phase = eb.get_centroidphase(wedges_eb)
        offset = eb.continuous_offset(wedges_eb)
        offset_eb_phase = fn.wrap(phase-offset)
        fit_wedges, all_params = fb.wedges_to_cos(wedges_eb, phase_offset=offset)
        wedges_eb = np.rot90(wedges_eb, k=1, axes=(0,1))
        ft2['offset_eb_phase'] = offset_eb_phase

        # dictionary to store all variables
        d = {
        'wedges_eb': wedges_eb,
        'offset': offset,
        'offset_eb_phase': offset_eb_phase,
        'fit_wedges_eb': fit_wedges,
        'all_params_eb': all_params,
        }

        # get fb information
        for layer in self.FB_masks:
            wedges_fb = fb.get_layer_wedges(layer)
            phase_fb = fb.get_centroidphase(wedges_fb)
            offset_fb_phase = fn.wrap(phase_fb-offset)
            fit_wedges, all_params = fb.wedges_to_cos(wedges_fb, phase_offset=offset)
            wedges_fb = np.rot90(wedges_fb, k=1, axes=(0,1))
            ft2['offset_fb_phase_'+layer] = offset_fb_phase
            d.update({
            'wedges_fb_'+layer: wedges_fb,
            'offset_fb_phase_'+layer: offset_fb_phase,
            'fit_wedges_fb_'+layer: fit_wedges,
            'all_params_fb_'+layer: all_params,
            })
        all_segs,inside_segs,outside_segs = fn.inside_outside(ft2)
        d.update({
        'all_segs': all_segs,
        'inside_segs': inside_segs,
        'outside_segs': outside_segs,
        'ft2': ft2
        })
        save_folder = fb.processedfol
        save_name = os.path.join(save_folder, 'get_all_info_eb_fb.p')
        fn.save_obj(d,save_name)
        return d

    def fetch_all_info_eb_fb(self, overwrite=False):
        save_folder = self.fb.processedfol
        save_name = os.path.join(save_folder, 'get_all_info_eb_fb.p')
        if os.path.exists(save_name) and not overwrite:
            print(self.folder +': fetching previously calculated data')
            d = fn.load_obj(save_name)
        else:
            print(self.folder + ': no data found. calculating...')
            d = self.save_all_info_eb_fb()
        return d

    def make_segment_library(self):
        save_folder = self.fb.processedfol
        save_name = os.path.join(save_folder, 'segment_library.p')
        if os.path.exists(save_name):
            print(self.folder +': fetching previously calculated segment data')
            d = fn.load_obj(save_name)
            segs = d['segs']
            ixs = d['ixs']
        else:
            print(self.folder + ': no segment data found. calculating...')
            all_data = self.fetch_all_info_eb_fb()
            x_all = all_data['ft2'].ft_posx.to_numpy()
            do = all_data['outside_segs']
            segs = []
            ixs = []
            for key in list(do.keys()):
                temp = do[key]
                x = temp.ft_posx.to_numpy()
                y = temp.ft_posy.to_numpy()
                simplified = fn.rdp_simp(x,y, epsilon=1)
                segs.append(np.diff(simplified,axis=0))
                ix = np.where(np.in1d(x_all, simplified[:,0]))[0]
                ix_middle = ix[1:-1]
                callback_ix = np.zeros(2*len(ix_middle)).astype(int)
                callback_ix[0::2]=ix_middle
                callback_ix[1::2]=ix_middle
                callback_ix=np.concatenate([[ix[0]], callback_ix, [ix[-1]]])
                callback_ix = np.reshape(callback_ix,(len(ix)-1,2))
                ixs.append(callback_ix)
            segs = np.concatenate(segs)
            ixs = np.concatenate(ixs)
            d={
            'segs': segs,
            'ixs': ixs
            }
            save_folder = self.fb.processedfol
            save_name = os.path.join(save_folder, 'segment_library.p')
            fn.save_obj(d,save_name)
        return segs, ixs

    def plot_all_heatmaps(self, cmap = sns.cm.rocket):
        d = self.fetch_all_info_eb_fb()
        num_masks = len(self.FB_masks)
        fig, axs = plt.subplots(3, 1, sharex=True)
        wedges = d['wedges_eb']
        sns.heatmap(wedges, ax=axs[0], cmap=cmap, cbar=False, rasterized=True)
        wedges = d['wedges_fb_upper']
        sns.heatmap(wedges, ax=axs[1], cmap=cmap, cbar=False, rasterized=True)
        wedges = d['wedges_fb_lower']
        sns.heatmap(wedges, ax=axs[2], cmap=cmap, cbar=False, rasterized=True)
        # plot the odor
        ft2 = d['ft2']
        d, di, do = fn.inside_outside(ft2)
        for key in list(di.keys()):
            df_temp = di[key]
            ix = df_temp.index.to_numpy()
            width = ix[-1]-ix[0]
            height = 2*np.pi
            rect = Rectangle((ix[0], -np.pi), width, height, fill=False)
            axs_r = axs[1].twinx()
            #axs_r.plot(phase, '.')#, color='red', alpha=0.5)
            axs_r.set_ylim(-np.pi, np.pi)
            axs_r.add_patch(rect)
        fig.savefig(os.path.join(self.figure_folder, 'all_heatmaps.pdf'))

    def plot_FB_heatmap_heading(self):
        """
        plot the fan-shaped body heatmaps
        overlay the extracted phase
        not upsampled so cannot be compared directly with behavior
        """
        self.fb.load_processed()
        heading = self.fb.ft2.ft_heading
        if self.FB_masks is None:
            print('NO FB specified')
        else:
            num_masks = len(self.FB_masks)
            fig, axs = plt.subplots(num_masks+1, 1, sharex=True)
            for i, mask in enumerate(self.FB_masks):
                print(mask)
                # plot the wedges and the phase
                wedges = self.fb.get_layer_wedges(mask)
                phase = self.fb.get_centroidphase(wedges)
                offset = fn.circmean(phase-heading)
                offset_phase = fn.wrap(phase-offset)
                #axs[num_masks].plot(offset_phase, '.')
                axs[num_masks].plot(phase, '.')


                axs_r = axs[i].twinx()
                #axs_r.plot(phase, '.')#, color='red', alpha=0.5)
                axs_r.set_ylim(-np.pi, np.pi)
                wedges = np.rot90(wedges, k=1, axes=(0,1))
                sns.heatmap(wedges, ax=axs[i], cmap='Blues', cbar=False, rasterized=True)

                # plot the odor
                ft2 = self.fb.ft2
                d, di, do = fn.inside_outside(ft2)
                for key in list(di.keys()):
                    df_temp = di[key]
                    ix = df_temp.index.to_numpy()
                    width = ix[-1]-ix[0]
                    height = 2*np.pi
                    rect = Rectangle((ix[0], -np.pi), width, height, fill=False)
                    rect2 = Rectangle((ix[0], -np.pi), width, height, fill=False)
                    axs_r.add_patch(rect)
                    axs[num_masks].add_patch(rect2)

            axs[i+1].plot(heading)
        return fig, axs

    def plot_EB_heatmap_heading(self):
        if hasattr(self, 'eb'):
            fig, axs = plt.subplots(2,1, sharex=True)
            self.eb.load_processed()
            heading = self.eb.ft2.ft_heading
            # plot the wedges and the phase
            wedges = self.eb.get_layer_wedges()
            phase = self.eb.get_centroidphase(wedges)
            offset = fn.circmean(phase-heading)
            offset_phase = fn.wrap(phase-offset)
            axs[1].plot(offset_phase, '.', color='red', alpha=0.5)
            axs[1].plot(heading)

            axs_r = axs[0].twinx()
            #axs_r.plot(phase, '.')
            axs_r.set_ylim(-np.pi, np.pi)
            wedges = np.rot90(wedges, k=1, axes=(0,1))
            sns.heatmap(wedges, ax=axs[0], cmap='Blues', cbar=False)

            # plot the odor
            ft2 = self.eb.ft2
            d, di, do = fn.inside_outside(ft2)
            for key in list(di.keys()):
                df_temp = di[key]
                ix = df_temp.index.to_numpy()
                width = ix[-1]-ix[0]
                height = 2*np.pi
                rect = Rectangle((ix[0], -np.pi), width, height, fill=False)
                axs_r.add_patch(rect)

    def plot_EPG_FB_heading(self):
        num_plots = len(self.FB_masks)+1
        fig, axs = plt.subplots(num_plots, 1, sharex=True)
        if hasattr(self, 'eb'):
            self.eb.load_processed()
            ft2 = self.eb.ft2
            d, di, do = fn.inside_outside(ft2)
            heading = self.eb.ft2.ft_heading
            # plot the wedges and the phase
            wedges = self.eb.get_layer_wedges()
            phase = self.eb.get_centroidphase(wedges)
            #offset = fn.circmean(phase-heading)
            offset = self.eb.continuous_offset(wedges)
            offset_eb_phase = fn.wrap(phase-offset)
            axs[num_plots-1].plot(heading)
            axs[num_plots-1].plot(offset_eb_phase, '.')
            axs_r = axs[num_plots-1].twinx()
            axs_r.set_ylim(-np.pi, np.pi)
            for key in list(di.keys()):
                df_temp = di[key]
                ix = df_temp.index.to_numpy()
                width = ix[-1]-ix[0]
                height = 2*np.pi
                rect = Rectangle((ix[0], -np.pi), width, height, fill=False)
                axs_r.add_patch(rect)
        if self.FB_masks is None:
            print('NO FB specified')
        else:
            num_masks = len(self.FB_masks)
            for i, mask in enumerate(self.FB_masks):
                print(mask)
                # plot the wedges and the phase
                wedges = self.fb.get_layer_wedges(mask)
                phase = self.fb.get_centroidphase(wedges)
                offset_fb_phase = fn.wrap(phase-offset)
                axs[i].plot(heading)
                axs[i].plot(offset_fb_phase, '.')
                axs_r = axs[i].twinx()
                axs_r.set_ylim(-np.pi, np.pi)
                for key in list(di.keys()):
                    df_temp = di[key]
                    ix = df_temp.index.to_numpy()
                    width = ix[-1]-ix[0]
                    height = 2*np.pi
                    rect = Rectangle((ix[0], -np.pi), width, height, fill=False)
                    axs_r.add_patch(rect)
        return fig, axs

    def get_phase_diff_between_layers(self):
        phases = []
        for i, mask in enumerate(self.FB_masks):
            print(mask)
            # plot the wedges and the phase
            wedges = self.fb.get_layer_wedges(mask)
            phase = self.fb.get_centroidphase(wedges)
            phases.append(phase)
        phase_diff = np.rad2deg(fn.wrap(phases[0]-phases[1]))
        df = self.fb.ft2
        ix_in = df[df.instrip==1.0].index.to_list()
        ix_out = df[df.instrip==0.0].index.to_list()
        return phase_diff, phase_diff[ix_in], phase_diff[ix_out]

    def get_phase_diff_between_layer_eb(self, layer ='upper'):
        phases = []

        # fb phase
        wedges = self.fb.get_layer_wedges(layer)
        phase = self.fb.get_centroidphase(wedges)
        phases.append(phase)
        # eb phase
        wedges = self.eb.get_layer_wedges()
        phase = self.eb.get_centroidphase(wedges)
        phases.append(phase)

        phase_diff = np.rad2deg(fn.wrap(phases[0]-phases[1]))
        df = self.fb.ft2
        ix_in = df[df.instrip==1.0].index.to_list()
        ix_out = df[df.instrip==0.0].index.to_list()
        return phase_diff, phase_diff[ix_in], phase_diff[ix_out]

    def phase_diff_between_layers(self):
        """
        point by point subtraction of the phase between one FB layer and another
        optionally specify pre-odor time
        """
        num_masks = len(self.FB_masks)
        phases = []
        for i, mask in enumerate(self.FB_masks):
            print(mask)
            # plot the wedges and the phase
            wedges = self.fb.get_layer_wedges(mask)
            phase = self.fb.get_centroidphase(wedges)
            phases.append(phase)
        phase_diff = np.rad2deg(fn.wrap(phases[0]-phases[1]))
        # find where odor turns on
        df = self.fb.ft2
        ix0 = df.index[df.instrip==True].tolist()[0]

        fig, axs = plt.subplots(1,1)
        x=np.arange(len(phase_diff))
        axs.plot(x[0:ix0],phase_diff[0:ix0], '.', color=sns.color_palette()[0])
        axs.plot(x[ix0:-1],phase_diff[ix0:-1], '.', color=sns.color_palette()[1])
        axs.set_ylabel('upper-lower phase difference (degrees)')
        d, di, do = fn.inside_outside(df)
        for key in list(di.keys()):
            df_temp = di[key]
            ix = df_temp.index.to_numpy()
            width = ix[-1]-ix[0]
            height = 360
            rect = Rectangle((ix[0], -180), width, height, fill=False)
            axs.add_patch(rect)

        fig, axs = plt.subplots(1,1)
        sns.histplot(phase_diff[0:ix0], ax=axs, color=sns.color_palette()[0], element="step", fill=False)
        sns.histplot(phase_diff[ix0:-1], ax=axs, color=sns.color_palette()[1], element="step", fill=False)
        axs.set_xlabel('upper-lower phase difference (degrees)')

        return fig, axs

    def polar_hist_bout(self):
        """
        plot the polar histograms for each inside and outside bout
        """
        self.fb.load_processed()
        df = self.fb.ft2
        d, di, do = fn.inside_outside(df)

        eb_wedges = self.eb.get_layer_wedges()
        #offset = self.eb.phase_offset(eb_wedges)
        offset = self.eb.continuous_offset(eb_wedges)

        # epg phase
        eb_phase = self.eb.get_centroidphase(eb_wedges)
        eb_phase = fn.wrap(eb_phase-offset)

        # lower phase
        lower_wedges = self.fb.get_layer_wedges('lower')
        lower_phase = self.fb.get_centroidphase(lower_wedges)
        lower_phase = fn.wrap(lower_phase-offset)

        # upper phase
        upper_wedges = self.fb.get_layer_wedges('upper')
        upper_phase = self.fb.get_centroidphase(upper_wedges)
        upper_phase = fn.wrap(upper_phase-offset)

        num_rows = int(np.ceil(len(d)/2))
        # fig, axs = plt.subplots(num_rows,4, subplot_kw={'projection': 'polar'})
        fig = plt.figure(figsize=(7,20))
        for i, key in enumerate(list(d.keys())):
            ix = d[key].index.to_numpy()
            x = d[key].ft_posx
            y = d[key].ft_posy
            row = int(np.floor(i/2))
            if i%2==0: # outside
                ax = fig.add_subplot(num_rows,4,2*i+1)
                ax.plot(x,y, 'k')
                ax.axis('off')
                ax = fig.add_subplot(num_rows,4,2*i+2, projection='polar')
                pl.circular_hist(ax, lower_phase[ix], color=sns.color_palette()[0], offset=np.pi/2, remove_x_labels=True)
            else: # inside
                ax = fig.add_subplot(num_rows,4,2*i+1)
                ax.plot(x,y, 'k')
                ax.axis('off')
                ax = fig.add_subplot(num_rows,4,2*i+2, projection='polar')
                pl.circular_hist(ax, lower_phase[ix], color=sns.color_palette()[0], offset=np.pi/2, remove_x_labels=True)
        return fig

    def polar_hist_in_out(self):
        """
        plot the polar histogram before and after odor turns on for EPGs,
        upper layer and lower layer
        """
        self.fb.load_processed()
        df = self.fb.ft2
        ix = df.index.to_numpy()
        ix_in = df[df.instrip==1.0].index.to_numpy()
        ix_out = df[df.instrip==0.0].index.to_numpy()
        ix0 = df.index[df.instrip==True].tolist()[0]
        ix0_end = ix[ix0:-1]
        ix_out = np.intersect1d(ix0_end, ix_out)

        eb_wedges = self.eb.get_layer_wedges()
        #offset = self.eb.phase_offset(eb_wedges)
        offset = self.eb.continuous_offset(eb_wedges)

        # epg phase
        eb_phase = self.eb.get_centroidphase(eb_wedges)
        eb_phase = fn.wrap(eb_phase-offset)

        # lower phase
        lower_wedges = self.fb.get_layer_wedges('lower')
        lower_phase = self.fb.get_centroidphase(lower_wedges)
        lower_phase = fn.wrap(lower_phase-offset)

        # upper phase
        upper_wedges = self.fb.get_layer_wedges('upper')
        upper_phase = self.fb.get_centroidphase(upper_wedges)
        upper_phase = fn.wrap(upper_phase-offset)

        fig = plt.figure(figsize=(14,7))
        ax1 = fig.add_subplot(131, projection='polar')
        ax2 = fig.add_subplot(132, projection='polar')
        ax3 = fig.add_subplot(133, projection='polar')
        ax1.title.set_text('EPG phase')
        ax2.title.set_text('hDeltaC L2 phase')
        ax3.title.set_text('hDeltaC L6 phase')
        pl.circular_hist(ax1, eb_phase[ix_in], color=sns.color_palette()[0], offset=np.pi/2)
        pl.circular_hist(ax1, eb_phase[ix_out], color=sns.color_palette()[1], offset=np.pi/2)
        pl.circular_hist(ax2, lower_phase[ix_in], color=sns.color_palette()[0], offset=np.pi/2)
        pl.circular_hist(ax2, lower_phase[ix_out], color=sns.color_palette()[1], offset=np.pi/2)
        pl.circular_hist(ax3, upper_phase[ix_in], color=sns.color_palette()[0], offset=np.pi/2)
        pl.circular_hist(ax3, upper_phase[ix_out], color=sns.color_palette()[1], offset=np.pi/2)
        plt.show()
        return fig

    def polar_hist(self):
        """
        plot the polar histogram before and after odor turns on for EPGs,
        upper layer and lower layer
        """
        self.fb.load_processed()
        df = self.fb.ft2
        ix0 = df.index[df.instrip==True].tolist()[0]
        eb_wedges = self.eb.get_layer_wedges()
        #offset = self.eb.phase_offset(eb_wedges)
        offset = self.eb.continuous_offset(eb_wedges)

        # epg phase
        eb_phase = self.eb.get_centroidphase(eb_wedges)
        eb_phase = fn.wrap(eb_phase-offset)

        # lower phase
        lower_wedges = self.fb.get_layer_wedges('lower')
        lower_phase = self.fb.get_centroidphase(lower_wedges)
        lower_phase = fn.wrap(lower_phase-offset)

        # upper phase
        upper_wedges = self.fb.get_layer_wedges('upper')
        upper_phase = self.fb.get_centroidphase(upper_wedges)
        upper_phase = fn.wrap(upper_phase-offset)

        fig = plt.figure(figsize=(14,7))
        ax1 = fig.add_subplot(131, projection='polar')
        ax2 = fig.add_subplot(132, projection='polar')
        ax3 = fig.add_subplot(133, projection='polar')
        ax1.title.set_text('EPG phase')
        ax2.title.set_text('hDeltaC L2 phase')
        ax3.title.set_text('hDeltaC L6 phase')
        pl.circular_hist(ax1, eb_phase[0:ix0], color=sns.color_palette()[0], offset=np.pi/2)
        pl.circular_hist(ax1, eb_phase[ix0:-1], color=sns.color_palette()[1], offset=np.pi/2)
        pl.circular_hist(ax2, lower_phase[0:ix0], color=sns.color_palette()[0], offset=np.pi/2)
        pl.circular_hist(ax2, lower_phase[ix0:-1], color=sns.color_palette()[1], offset=np.pi/2)
        pl.circular_hist(ax3, upper_phase[0:ix0], color=sns.color_palette()[0], offset=np.pi/2)
        pl.circular_hist(ax3, upper_phase[ix0:-1], color=sns.color_palette()[1], offset=np.pi/2)
        plt.show()
        fig.savefig(os.path.join(self.figure_folder,'lower_layer_polar.pdf'))

        return fig, axs

    def upper_layer_exclude(self):
        wedges = self.fb.get_layer_wedges('upper')
        wedges = np.rot90(wedges, k=1, axes=(0,1))
        fig,axs = plt.subplots(2,1, sharex=True)
        average_bump = fn.lnorm(np.mean(wedges, axis=0))
        average_bump[average_bump<0.2]=np.nan
        axs[0].plot(average_bump)
        axs[0].set_xlim(0,len(average_bump))
        sns.heatmap(wedges, ax=axs[1], cmap='Blues', cbar=False)

    def plot_trajectory(self):
        self.fb.load_processed()
        fig, axs = plt.subplots(1,2)
        axs[0].axis('equal')
        ft2 = self.fb.ft2
        axs[0] = pl.plot_vertical_edges(ft2, axs[0], width=10)
        axs[0] = pl.plot_trajectory(ft2, axs[0])
        axs[0] = pl.plot_trajectory_odor(ft2, axs[0])
        axs[1] = pl.plot_vertical_edges(ft2, axs[1], width=10)
        axs[1] = pl.plot_trajectory(ft2, axs[1])
        axs[1] = pl.plot_trajectory_odor(ft2, axs[1])
        fig.savefig(os.path.join(self.figure_folder, 'trajectory.pdf'))
        return fig, axs

    def plot_EPG_heading_phase_offset(self):
        # DEAL WITH EPGs
        eb = self.eb
        fig, axs = plt.subplots(3,1, sharex=True)
        eb.load_processed()
        heading = eb.ft2.ft_heading
        time = eb.ft2.seconds.to_numpy()
        time = time - time[0]
        time = np.linspace(time, time[-1],10)
        print('time is:', time[-1])

        # plot the wedges, heading and offset phase
        wedges = eb.get_layer_wedges()
        phase = eb.get_centroidphase(wedges)
        offset = fn.circmean(phase-heading)
        print(np.rad2deg(offset))
        offset_phase = fn.wrap(phase-offset)
        axs[2].plot(phase, '.', color='black', alpha=0.5)
        axs[2].plot(offset_phase, '.', color='gray', alpha=0.5)
        axs[2].plot(heading, color='purple')

        axs_r = axs[1].twinx()
        #axs_r.plot(phase, '.')
        axs_r.set_ylim(-np.pi, np.pi)
        axs_r.plot(phase, '.', color='black', alpha=0.5)
        wedges = np.rot90(wedges, k=1, axes=(0,1))
        sns.heatmap(wedges, ax=axs[0], cmap='Blues', cbar=False, rasterized=True)
        sns.heatmap(wedges, ax=axs[1], cmap='Blues', cbar=False, rasterized=True)

        fig.savefig(os.path.join(self.figure_folder,'eb.pdf'))

    def plot_FB_heatmap(self):
        """
        plot the fan-shaped body heatmaps
        overlay the extracted phase
        not upsampled so cannot be compared directly with behavior
        """
        if self.FB_masks is None:
            print('NO FB specified')
        else:
            num_masks = len(self.FB_masks)
            fig, axs = plt.subplots(num_masks, 1, sharex=True, squeeze=False)
            for i, mask in enumerate(self.FB_masks):
                wedges = self.fb.get_wedge_rois(mask)
                phase = self.fb.get_centroidphase(wedges)
                axs_r = axs[i, 0].twinx()
                axs_r.plot(phase, 'r.', alpha=0.5)
                axs_r.set_ylim(-np.pi, np.pi)
                wedges = np.rot90(wedges, k=1, axes=(0,1))
                sns.heatmap(wedges, ax=axs[i, 0], cmap='Blues',vmin=0,vmax=1, cbar=True)

    def plot_PB_heatmap(self):
        """
        this is a plot of the heatmap, not upsampled in time, cannot be compared
        directly to behavior
        """
        if hasattr(ex, 'pb'):
            gloms = self.pb.get_gloms()
            phase = self.pb.get_phase()
            fig, axs = plt.subplots(2,1, sharex=True)
            axs[0].plot(phase, 'r.', alpha=0.5, edgecolor=None)
            sns.heatmap(np.rot90(gloms, k=1, axes=(0,1)), cmap='Blues', ax=axs[1], cbar=False)
        else:
            print('no protocerebral bridge')

    def plot_EB_heatmap(self):
        """
        this is a plot of the heatmap, not upsampled in time, cannot be compared
        directly to behavior
        """
        if hasattr(ex, 'eb'):
            wedges = self.eb.get_wedge_rois()
            phase = self.eb.get_centroidphase(wedges)
            fig, axs = plt.subplots(2,1, sharex=True)
            axs[0].plot(phase, '.')
            sns.heatmap(np.rot90(wedges, k=1, axes=(0,1)), cmap='Blues', ax=axs[1], cbar=False)
        else:
            print('no protocerebral bridge')

    def fb_bumps_phase_null(self):
        fb = self.fb
        fig, axs = plt.subplots(3,1, sharex=True)
        fb.load_processed()

        # plot wedges
        wedges = fb.get_layer_wedges()
        wedges_phase_null = fb.cancel_phase(wedges)
        wedges_phase_null = fb.continuous_to_glom(wedges_phase_null, nglom=16)
        wedges = np.rot90(wedges, k=1, axes=(0,1))
        wedges_phase_null = np.rot90(wedges_phase_null, k=1, axes=(0,1))
        sns.heatmap(wedges, ax=axs[0], cmap='Blues', cbar=False, rasterized=True)

        # phase null wedges
        sns.heatmap(wedges_phase_null, ax=axs[1], cmap='Blues', cbar=False, rasterized=True)

        #plot phase null wedges with odor overlayed
        sns.heatmap(wedges_phase_null, ax=axs[2], cmap='Blues', cbar=False, rasterized=True)
        ft2 = fb.ft2
        d, di, do = fn.inside_outside(ft2)
        axs_r = axs[2].twinx()
        axs_r.set_ylim(-np.pi, np.pi)
        for key in list(di.keys()):
            df_temp = di[key]
            ix = df_temp.index.to_numpy()
            width = ix[-1]-ix[0]
            height = 2*np.pi
            rect = Rectangle((ix[0], -np.pi), width, height, fill=False)
            axs_r.add_patch(rect)
        fig.savefig(os.path.join(self.figure_folder,'fb_bumps_phase_null.pdf'))

    def fb_rotate_heatmap(self, additional_offset=0):
        # plot shifted bumps
        fb = self.fb
        eb = self.eb
        offset = eb.phase_offset(eb.get_layer_wedges())
        wedges = fb.get_layer_wedges()
        x_interp, row_interp = fb.interpolate_wedges(wedges)
        period_inds = row_interp.shape[1]
        shift = int((-offset-additional_offset)/(2*np.pi)*period_inds)
        wedges_shift = np.roll(row_interp, shift)
        wedges_shift = fb.continuous_to_glom(wedges_shift, nglom=16)
        wedges_shift_rot = np.rot90(wedges_shift, k=1, axes=(0,1))
        wedges = fb.get_layer_wedges()
        phase = fb.get_centroidphase(wedges)
        wedges = np.rot90(wedges, k=1, axes=(0,1))
        offset_fb_phase = fn.wrap(phase-offset-additional_offset)

        fig, axs = plt.subplots(2,1)
        sns.heatmap(wedges, ax=axs[0],vmin=0.1, vmax=1.1, cmap='Blues', cbar=False, rasterized=True)
        sns.heatmap(wedges_shift_rot, vmin=0.1, vmax=1.1, ax=axs[1], cmap='Blues', cbar=False, rasterized=True)
        axs_r0 = axs[0].twinx()
        axs_r0.set_ylim(-np.pi, np.pi)
        axs_r0.plot(phase, '.', color = 'red', alpha=0.1)
        axs_r1 = axs[1].twinx()
        axs_r1.set_ylim(-np.pi, np.pi)
        axs_r1.plot(offset_fb_phase, '.', color = 'red', alpha=0.1)
        ft2 = self.fb.ft2
        d, di, do = fn.inside_outside(ft2)
        for key in list(di.keys()):
            df_temp = di[key]
            ix = df_temp.index.to_numpy()
            width = ix[-1]-ix[0]
            height = 2*np.pi
            rect = Rectangle((ix[0], -np.pi), width, height, fill=False)
            axs_r1.add_patch(rect)
        fig.savefig(os.path.join(self.figure_folder, 'fb_shifted_bumps.pdf'))

    def fb_bump_in_out(self):
        if hasattr(ex, 'fb'):
            # is the bump different in and out of odor
            for i, mask in enumerate(self.FB_masks):
                wedges = self.fb.get_layer_wedges(tag=mask)
                bumps_in, bumps_out = self.fb.bumps_in_out(wedges)
                fig, axs = plt.subplots(1, 1)
                axs.plot(np.mean(bumps_in, axis=0))
                axs.plot(np.mean(bumps_out, axis=0))
                fig.savefig(os.path.join(self.figure_folder, 'fb_bump_in_out.pdf'))
        else:
            print('no fb')

    def fb_bump_moving_still(self):
        if hasattr(ex, 'fb'):
            # is the bump different when the fly is moving and not moving -- larger bump when moving
            for i, mask in enumerate(self.FB_masks):
                wedges = self.fb.get_layer_wedges(tag=mask)
                bumps_still, bumps_moving = self.fb.bumps_moving_still(wedges)
                fig, axs = plt.subplots(1, 1)
                axs.plot(np.mean(bumps_still, axis=0))
                axs.plot(np.mean(bumps_moving, axis=0))
                fig.savefig(os.path.join(self.figure_folder, 'fb_bump_moving_still.pdf'))
        else:
            print('no fb')

    def eb_bump_in_out(self):
        # is there a difference between in odor and out of odor?
        eb = self.eb
        bumps_in, bumps_out = eb.bumps_in_out(eb.get_layer_wedges())
        fig, axs = plt.subplots(1,1)
        axs.plot(np.mean(bumps_in,axis=0))
        axs.plot(np.mean(bumps_out, axis=0))
        fig.savefig(os.path.join(self.figure_folder, 'eb_bumps_in_out.pdf'))

    def plot_phase_difference(self):

        fig, axs = plt.subplots(1, 1, figsize=(20,4,))
        self.eb.load_processed()
        self.fb.load_processed()
        ft2 = self.eb.ft2
        d, di, do = fn.inside_outside(ft2)
        heading = self.eb.ft2.ft_heading

        wedges = self.eb.get_layer_wedges()
        phase = self.eb.get_centroidphase(wedges)
        offset = fn.circmean(phase-heading)
        offset_eb_phase = fn.wrap(phase-offset)

        wedges = self.fb.get_layer_wedges('fbmask')
        phase = self.fb.get_centroidphase(wedges)
        offset_fb_phase = fn.wrap(phase-offset)

        phase_diff = fn.wrap(offset_eb_phase-offset_fb_phase)

        for key in list(di.keys()):
            df_temp = di[key]
            ix = df_temp.index.to_numpy()
            width = ix[-1]-ix[0]
            height = 2*np.pi
            rect = Rectangle((ix[0], -np.pi), width, height, fill=False)
            axs.add_patch(rect)

        axs.plot(phase_diff, '.')
        fig.savefig(os.path.join(self.figure_folder, 'phase_diff.pdf'))

    def plot_phase_diff_x_y(self):
        fig, axs = plt.subplots(1, 3, figsize=(15,10))
        self.eb.load_processed()
        self.fb.load_processed()
        ft2 = self.eb.ft2
        x = ft2.ft_posx.to_numpy()
        y = ft2.ft_posy.to_numpy()
        heading = self.eb.ft2.ft_heading
        for i in np.arange(3):
            pl.plot_vertical_edges(ft2,axs[i],width=10)

        wedges = self.eb.get_layer_wedges()
        phase = self.eb.get_centroidphase(wedges)
        #offset = fn.circmean(phase-heading)
        offset = self.eb.continuous_offset(wedges)
        offset_eb_phase = fn.wrap(phase-offset)

        wedges = self.fb.get_layer_wedges('upper')
        phase = self.fb.get_centroidphase(wedges)
        offset_fb_phase = fn.wrap(phase-offset)

        phase_diff = fn.wrap(offset_eb_phase-offset_fb_phase)

        # 4 color quadrants
        color_quadrant = []
        for phase in phase_diff:
            if np.abs(phase)<np.pi/6:
                color_quadrant.append([0.1,0.1,0.1])
            elif np.abs(phase)>5*np.pi/6:
                color_quadrant.append([0.9,0.9,0.9])
            elif phase<0:
                color_quadrant.append([1,0,0])
            elif phase>0:
                color_quadrant.append([0,0,1])

        pl.colorline(axs[0],x,y, z=phase_diff)
        #pl.colorline_specify_colors(axs[0], x, y, color_quadrant, linewidth=3, alpha=1.0)
        axs[0].title.set_text('EPG-hDeltaC phase')
        pl.colorline(axs[1],x,y, z=offset_eb_phase)
        axs[1].title.set_text('EPG phase')
        pl.colorline(axs[2],x,y, z=offset_fb_phase)
        axs[2].title.set_text('hDeltaC phase')

        fig.savefig(os.path.join(self.figure_folder, 'phase_diff_x_y.pdf'))
        return phase_diff, offset_eb_phase, offset_fb_phase

    def plot_allo_directions_x_y(self):
        fig, axs = plt.subplots(1, 3, figsize=(15,10))
        self.eb.load_processed()
        self.fb.load_processed()
        ft2 = self.eb.ft2
        x = ft2.ft_posx.to_numpy()
        y = ft2.ft_posy.to_numpy()
        heading = self.eb.ft2.ft_heading
        if self.experiment != 'clean_air':
            for i in np.arange(3):
                pl.plot_vertical_edges(ft2,axs[i],width=10)

        eb_wedges = self.eb.get_layer_wedges()
        phase = self.eb.get_centroidphase(eb_wedges)
        #offset = fn.circmean(phase-heading)
        offset = self.eb.continuous_offset(eb_wedges)

        # epg phase
        eb_phase = fn.wrap(phase-offset)

        # lower phase
        lower_wedges = self.fb.get_layer_wedges('lower')
        lower_phase = self.fb.get_centroidphase(lower_wedges)
        lower_phase = fn.wrap(lower_phase-offset)

        # upper phase
        upper_wedges = self.fb.get_layer_wedges('upper')
        upper_phase = self.fb.get_centroidphase(upper_wedges)
        upper_phase = fn.wrap(upper_phase-offset)

        pl.colorline(axs[0],x,y, z=eb_phase)
        #pl.colorline_specify_colors(axs[0], x, y, color_quadrant, linewidth=3, alpha=1.0)
        axs[0].title.set_text('EPG phase')
        pl.colorline(axs[1],x,y, z=lower_phase)
        axs[1].title.set_text('Lower layer phase')
        pl.colorline(axs[2],x,y, z=upper_phase)
        axs[2].title.set_text('Upper layer phase')
        # axs[0].axis('equal')
        # axs[1].axis('equal')
        # axs[2].axis('equal')
        #fig.savefig(os.path.join(self.figure_folder, 'phase_diff_x_y.pdf'))
        return fig

    def plot_outbound_inbound_phase_diff(self):
        phase_diff, offset_eb_phase, offset_fb_phase = self.plot_phase_diff_x_y()
        df = self.fb.ft2
        _,df = fn.find_stops(df)
        d, di, do = fn.inside_outside(df)
        fig2, axs2 = plt.subplots(1,3)
        #axs2.set_ylim(-np.pi, np.pi)
        for key in list(do.keys())[1:]:
            temp = do[key]
            time = temp.seconds.iloc[-1]-temp.seconds.iloc[0]
            if time>2: # only look at outside bouts longer than 2 seconds
                fig, axs = plt.subplots(1,2)

                x = do[key].ft_posx.to_numpy()
                y = do[key].ft_posy.to_numpy()

                ix = list(do[key].index) # outside bout index
                max_dist_ix = ix[np.argmax(np.abs(x-x[0]))] # return point index
                ix_moving = df[df.stop==1.0].index.to_list() # moving index
                out_ix = np.arange(ix[0],max_dist_ix) # outward path (exit to return point)
                out_ix = np.intersect1d(out_ix, ix_moving) # only select periods where moving

                in_ix = np.arange(max_dist_ix,ix[-1]) # inward path (return point to re-entry)
                in_ix = np.intersect1d(in_ix, ix_moving) # only select periods where moving

                titles = ['EPG-FB phase', 'EPG phase', 'FB phase']
                for i, ang in enumerate([phase_diff, offset_eb_phase, offset_fb_phase]):
                    out_phase = fn.circmean(ang[out_ix])
                    in_phase = fn.circmean(ang[in_ix])
                    axs2[i].plot([0,1],[out_phase,in_phase])
                    axs2[i].set_title(titles[i])
                    axs2[i].set_xticks([0,1])
                    axs2[i].set_xticklabels(['outbound', 'inbound'], rotation=45)



                hi_out = np.pi/2-out_phase
                dx_out = np.cos(hi_out)
                dy_out = np.sin(hi_out)

                hi_in = np.pi/2-in_phase
                dx_in = np.cos(hi_in)
                dy_in = np.sin(hi_in)

                print(np.rad2deg(out_phase), np.rad2deg(in_phase))
                pl.colorline(axs[0], x,y,phase_diff[ix])
                #axs[0].plot(x[max_dist_ix],y[max_dist_ix], 'o')
                axs[0].plot(x[0],y[0], 'o', color='green')
                axs[0].axis('equal')
                axs[1].arrow(0,0,dx_in,dy_in, color='green', width=0.1)
                axs[1].arrow(0,0,dx_out,dy_out, color='purple', width=0.1)
                axs[1].axis('square')
                axs[1].set_xlim(-2,2)
                axs[1].set_ylim(-2,2)

    def plot_RDP_segment_phase(self, layer = 'upper', plot_individal_bouts = False):
        eb = self.eb
        fb = self.fb
        fb.load_processed()
        ft2 = fb.ft2

        wedges = eb.get_layer_wedges()
        phase = eb.get_centroidphase(wedges)
        #offset = fn.circmean(phase-heading)
        offset = eb.continuous_offset(wedges)
        offset_eb_phase = fn.wrap(phase-offset)

        wedges = fb.get_layer_wedges(layer)
        phase = fb.get_centroidphase(wedges)
        offset_fb_phase = fn.wrap(phase-offset)

        ft2['offset_fb_phase'] = offset_fb_phase
        ft2['offset_eb_phase'] = offset_eb_phase
        d,di,do = fn.inside_outside(ft2)
        print('speed is :', np.mean(fn.calculate_speeds(ft2).speed))

        outbound_phase_fb = []
        inbound_phase_fb = []
        outbound_phase_eb = []
        inbound_phase_eb = []
        for key in list(do.keys())[1:]:
            temp = do[key]
            x = temp.ft_posx.to_numpy()
            y = temp.ft_posy.to_numpy()
            if self.experiment == '45':
                x,y = fn.coordinate_rotation(x, y, np.pi/4)

            _,pathlenth = fn.path_length(x,y)
            if np.abs(x[-1]-x[0])<1 and pathlenth>10: # returns to the edge
                if self.experiment == '45':
                    x,y = fn.coordinate_rotation(x, y, -np.pi/4)
                simplified = fn.rdp_simp(x,y, epsilon=1)
                phase_fb = temp.offset_fb_phase
                phase_eb = temp.offset_eb_phase
                # first segment
                ix1 = np.where(y==simplified[:,1][1])
                ix1 = ix1[0][0]
                # last segment
                ix2 = np.where(y==simplified[:,1][-2])
                ix2 = ix2[0][0]
                op_fb = fn.circmean(phase_fb[0:ix1])
                ip_fb = fn.circmean(phase_fb[ix2:])
                op_eb = fn.circmean(phase_eb[0:ix1])
                ip_eb = fn.circmean(phase_eb[ix2:])
                # outbound_phase.append(phase[0:max_ix].to_list())
                # inbound_phase.append(phase[max_ix:].to_list())
                outbound_phase_fb.append(op_fb)
                inbound_phase_fb.append(ip_fb)
                outbound_phase_eb.append(op_eb)
                inbound_phase_eb.append(ip_eb)

                if plot_individal_bouts:
                    fig, axs = plt.subplots(1,2)
                    pl.colorline(axs[0],x,y,phase_fb)
                    axs[0].axis('equal')
                    # axs[1].plot(simplified[:,0],simplified[:,1])
                    # #axs[1].plot(x[max_ix],y[max_ix], 'o')
                    # axs[1].axis('equal')
                    axs[1].plot(x, color='r')
                    axs_r = axs[1].twinx()
                    axs_r.plot(phase_fb.to_numpy(), color='k')

        fig = plt.figure()
        fig.suptitle(self.folder)
        axs1 = fig.add_subplot(121, projection='polar')
        axs2 = fig.add_subplot(122, projection='polar')
        d = {}
        ang_avg, r_avg = pl.circular_hist(axs1, np.array(outbound_phase_eb), color=sns.color_palette()[0], offset=np.pi/2, label='outbound')
        d['outbound_eb'] = [ang_avg, r_avg]
        ang_avg, r_avg = pl.circular_hist(axs1, np.array(inbound_phase_eb), color=sns.color_palette()[1], offset=np.pi/2, label='inbound')
        d['inbound_eb'] = [ang_avg, r_avg]
        ang_avg, r_avg = pl.circular_hist(axs2, np.array(outbound_phase_fb), color=sns.color_palette()[0], offset=np.pi/2, label='outbound')
        d['outbound_fb'] = [ang_avg, r_avg]
        ang_avg, r_avg = pl.circular_hist(axs2, np.array(inbound_phase_fb), color=sns.color_palette()[1], offset=np.pi/2, label='inbound')
        d['inbound_fb'] = [ang_avg, r_avg]
        axs1.legend(loc='upper left', bbox_to_anchor=(1,1))
        axs2.legend(loc='upper left', bbox_to_anchor=(1,1))
        #fig.tight_layout()
        return fig, d

    def average_fsb_bump(self):
        # before odor
        fb = self.fb
        fb.load_processed()
        df = fb.ft2
        ix_in = df[df.instrip==1.0].index.to_list()
        ix_out = df[df.instrip==0.0].index.to_list()
        fb_stack = self.load_z_projection(region='fb')
        ix0 = df.index[df.instrip==True].tolist()[0]
        pre = fb_stack[0:ix0,:,:]
        pre = np.mean(pre, axis=0)
        odor = fb_stack[ix_in,:,:]
        odor = np.mean(odor, axis=0)
        air = fb_stack[ix_out,:,:]
        air = np.mean(air, axis=0)
        fig, axs = plt.subplots(3,1)
        axs[0].imshow(pre, vmin=0, vmax=1)
        axs[1].imshow(odor, vmin=0, vmax=1)
        axs[2].imshow(air, vmin=0, vmax=1)

    def example_plots(self):

        #plot trajectory
        fig, axs = ex.plot_trajectory()

        # plot extracted phase vs heading for EPG and hDelta
        fig, axs = self.plot_EPG_FB_heading()
        fig.savefig(os.path.join(self.figure_folder, 'EPG_FB_phase.pdf'))

        # make a plot showing how the extracted EPG phase can be aligned to the heading
        self.plot_EPG_heading_phase_offset()

        # Does the EB bump change when the animal is moving?
        self.eb_bump_in_out()

        # plot the phase nulled FB bumps
        self.fb_bumps_phase_null()

        # plot the FB bumps in and out of odor
        self.fb_bump_in_out()

        # plot the FB bumps for periods of moving and not moving
        self.fb_bump_moving_still()

        # plot the rotated heatmap based on the eb phase offset
        self.fb_rotate_heatmap()

    def z_projection(self, region='eb'):
        if region == 'eb':
            r = self.eb
            r.load_processed()
            all_masks = r.open_mask()
            tif_name = os.path.join(r.regfol, r.name+'_eb_z_projection'+'.tif')
        elif region == 'fb_lower':
            r = self.fb
            r.load_processed()
            all_masks = r.open_mask('lower')
            tif_name = os.path.join(r.regfol, r.name+'_fb_z_projection_lower'+'.tif')
        elif region == 'fb_upper':
            r = self.fb
            r.load_processed()
            all_masks = r.open_mask('upper')
            tif_name = os.path.join(r.regfol, r.name+'_fb_z_projection_upper'+'.tif')
        if len(all_masks.shape) == 2:
            all_masks = np.reshape(all_masks, all_masks.shape + (1,))
        for m, slice in enumerate(r.zslices):
            # open slice
            slice = r.open_slice(slice)
            if m == 0:
                proj = np.zeros(slice.shape)
            mask = all_masks[:,:,m]
            mask = np.expand_dims(mask, axis=2)
            slice_mask = slice*mask
            slice_mask = np.nan_to_num(slice_mask)
            proj+=slice_mask
        proj[proj==0.0]=np.nan
        proj = np.moveaxis(proj,2,0)
        bottom_10 = np.quantile(proj, 0.1, axis=0)
        proj = (proj-bottom_10)/bottom_10

        # upsample with linear interpolation of each pixel to match pv2 timepoints
        pv2 = r.pv2
        frames = np.arange(0,proj.shape[0])
        frames_upsample = np.linspace(0,proj.shape[0],len(pv2))
        array_upsample = np.zeros((len(pv2), proj.shape[1], proj.shape[2]))

        for i in range(proj.shape[1]):
            for j in range(proj.shape[2]):
                pixel = proj[:,i,j]
                pixel_upsample = np.interp(frames_upsample, frames, pixel)
                array_upsample[:,i,j]=pixel_upsample

        io.imsave(tif_name, array_upsample, plugin='tifffile')
        return proj

    def load_z_projection(self, region='eb'):
        if region == 'eb':
            r = self.eb
            tif_name = os.path.join(r.regfol, r.name+'_eb_z_projection'+'.tif')
            if not os.path.exists(tif_name):
                self.z_projection(region='eb')
            stack = io.imread(tif_name)
        elif region == 'fb_lower':
            r = self.fb
            tif_name = os.path.join(r.regfol, r.name+'_fb_z_projection_lower'+'.tif')
            if not os.path.exists(tif_name):
                self.z_projection(region='fb_lower')
            stack = io.imread(tif_name)
        elif region == 'fb_upper':
            r = self.fb
            tif_name = os.path.join(r.regfol, r.name+'_fb_z_projection_upper'+'.tif')
            if not os.path.exists(tif_name):
                self.z_projection(region='fb_upper')
            stack = io.imread(tif_name)
        return stack

    def make_movie(self):
        global image_eb, image_fb, x, y, dx, dy, odor
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

        # load z projection movies
        fb_stack = self.load_z_projection(region='fb_lower')
        fb_stack_u = self.load_z_projection(region='fb_upper')
        eb_stack = self.load_z_projection(region='eb')

        # load the behavioral data, define odor and borders, x and y positon
        self.fb.load_processed()
        df = self.fb.ft2
        odor = df.mfc2_stpt.to_numpy()
        x = df.ft_posx.to_numpy()
        y = df.ft_posy.to_numpy()
        heading = df.ft_heading.to_numpy()

        # find the FB phase for plotting arrows
        wedges = self.eb.get_layer_wedges()
        eb_phase = self.eb.get_centroidphase(wedges)
#        offset = fn.circmean(eb_phase-heading)
        offset = self.eb.continuous_offset(wedges)
        offset_eb_phase = fn.wrap(eb_phase-offset)

        # lower fb layer phase
        wedges = self.fb.get_layer_wedges('lower')
        fb_phase = self.fb.get_centroidphase(wedges)
        offset_fb_phase = fn.wrap(fb_phase-offset)
        offset_fb_phase_mag = np.mean(wedges, axis=1)

        # upper phase
        wedges_u = self.fb.get_layer_wedges('upper')
        fb_phase_u = self.fb.get_centroidphase(wedges_u)
        offset_fb_phase_u = fn.wrap(fb_phase_u-offset)
        offset_fb_phase_mag_u = np.mean(wedges_u, axis=1)

        # angles for plotting eb arrows
        heading = offset_eb_phase
        hi = np.pi/2-heading
        dx = 5*np.cos(hi)
        dy = 5*np.sin(hi)

        # angles for plotting fb phase arrow
        hi_fb = np.pi/2-offset_fb_phase
        dx_fb = 10*np.cos(hi_fb)*(offset_fb_phase_mag-np.min(offset_fb_phase_mag))/(np.max(offset_fb_phase_mag)-np.min(offset_fb_phase_mag))
        dy_fb = 10*np.sin(hi_fb)*(offset_fb_phase_mag-np.min(offset_fb_phase_mag))/(np.max(offset_fb_phase_mag)-np.min(offset_fb_phase_mag))

        # angles for plotting upper fb phase arrow
        hi_fb_u = np.pi/2-offset_fb_phase_u
        dx_fb_u = 10*np.cos(hi_fb_u)*(offset_fb_phase_mag_u-np.min(offset_fb_phase_mag_u))/(np.max(offset_fb_phase_mag_u)-np.min(offset_fb_phase_mag_u))
        dy_fb_u = 10*np.sin(hi_fb_u)*(offset_fb_phase_mag_u-np.min(offset_fb_phase_mag_u))/(np.max(offset_fb_phase_mag_u)-np.min(offset_fb_phase_mag_u))

        # define sampling and playback speed for movie
        downsample = 1
        fps_mult = 10
        speed_factor = downsample*fps_mult
        total_number_of_frames = int(np.round(len(df)/downsample)) # downsample to this number of frames
        num_movie_frames = total_number_of_frames #number of frames in movie, to test out making a shorter movie.
        delta_t = np.mean(np.diff(df.seconds)) # average time between frames
        memory= 3 #time in s to display memory
        memory = int(np.round(3/delta_t)) # adapt memory to number of frames
        interval = delta_t*1000/fps_mult

        # create a plot
        plt.style.use('dark_background')
        #fig, axs = plt.subplots(2,3, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 4]})
        #fig, axs = plt.subplots(2,3, figsize=(10, 6))
        fig, axs = plt.subplots(3,3, figsize=(5, 3))
        gs = axs[1, 1].get_gridspec()

        # remove the underlying axes
        for ax in axs[:, -1]:
            ax.remove()
        #axs[0,1].remove()
        #axs[2,1].remove()
        axbig = fig.add_subplot(gs[:, -1])
        axs[0,1].axis('off')
        axs[2,1].axis('off')
        axs[0,0].axis('off')
        axs[1,0].axis('off')
        axs[2,0].axis('off')

        # set titles
        axs[0,0].title.set_text('EPG signal')
        axs[0,0].title.set_color('red')
        axs[1,0].title.set_text('upper layer')
        axs[1,0].title.set_color('purple')
        axs[2,0].title.set_text('lower layer')
        axs[2,0].title.set_color('yellow')
        x_quiv, y_quiv = np.meshgrid(np.arange(-30,30,4), np.arange(-30,30,4))
        dx_quiv = np.zeros(x_quiv.shape)
        dy_quiv = -0.5*np.ones(x_quiv.shape)
        axs[1,1].quiver(x_quiv, y_quiv, dx_quiv, dy_quiv, color='lightskyblue', alpha=0.1)
        axs[1,1].axis('equal')
        axs[1,1].set_xlim(-30, 30)
        axs[1,1].set_ylim(-30, 30)
        axs[1,1].axis('off')
        #axbig.axis('equal')
        if self.experiment == 'strip_grid':
            all_x_borders, all_y_borders = fn.find_borders(df)
            for i in np.arange(len(all_x_borders))[::2]:
                rect1 = patches.Rectangle((all_x_borders[i][0], all_y_borders[i][0]), 10, np.max(y)-np.min(y), linewidth=1, edgecolor=None, facecolor='grey')
                axbig.add_patch(rect1)
        #axbig.add_patch(rect2)
        axbig.set_xlim(min(x)-100, max(x)+100)
        axbig.set_ylim(min(y), max(y))
        axbig.axis('off')

        # initialize image and lines
        image_eb = axs[0,0].imshow(eb_stack[0,:,:], cmap='gray', animated=True)
        image_fb_u = axs[1,0].imshow(fb_stack_u[0,:,:],vmin=0.2, vmax=1.0, cmap='gray', animated=True)
        image_fb = axs[2,0].imshow(fb_stack[0,:,:],vmin=0.2, vmax=1.0, cmap='gray', animated=True)
        lower_temp = np.rot90(wedges[0:memory,:], k=1, axes=(0,1))
        heatmap_lower = axs[2,1].imshow(lower_temp,vmin=np.min(wedges), vmax=np.max(wedges), cmap='Blues', animated=True)
        upper_temp = np.rot90(wedges_u[0:memory,:], k=1, axes=(0,1))
        heatmap_upper = axs[0,1].imshow(upper_temp,vmin=np.min(wedges_u), vmax=np.max(wedges_u), cmap='Blues', animated=True)
        line1, = axbig.plot(x[0], y[0])
        line2, = axbig.plot(x[0], y[0])
        line3, = axs[1,1].plot(x[0], y[0])
        arrow = axs[1,1].arrow(0,0,0,0,color='red', animated=True)
        arrow_fb = axs[1,1].arrow(0,0,0,0,color='yellow', animated=True)
        arrow_fb_u = axs[1,1].arrow(0,0,0,0,color='purple', animated=True)
        #arrow_fb_u = axs[1,1].arrow(0,0,0,0,color='white', animated=True)


        def animate(frame, image_eb,image_fb, line1, line2, line3):
            image_eb.set_array(eb_stack[frame,:,:])
            image_fb.set_array(fb_stack[frame,:,:])
            image_fb_u.set_array(fb_stack_u[frame,:,:])


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
                arrow_fb.set_data(x=current_x3[-1], y=current_y3[-1], dx=dx_fb[frame], dy=dy_fb[frame], head_width=1)
                arrow_fb_u.set_data(x=current_x3[-1], y=current_y3[-1], dx=dx_fb_u[frame], dy=dy_fb_u[frame], head_width=1)
                if odor[frame]>0:
                    arrow.set(color='blue')
                else:
                    arrow.set(color='red')

                upper_temp = np.rot90(wedges_u[(frame-memory):(frame + 1),:], k=1, axes=(0,1))
                heatmap_upper.set_array(upper_temp)
                lower_temp = np.rot90(wedges[(frame-memory):(frame + 1),:], k=1, axes=(0,1))
                heatmap_lower.set_array(lower_temp)
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

            return [image_eb,image_fb, line1, line2, line3,]
        animation = FuncAnimation(fig,animate,total_number_of_frames, fargs=[image_eb,image_fb, line1, line2, line3,], interval=interval)
        plt.show()
        save_name = os.path.join(ex.figure_folder, 'movie.avi')
        animation.save(save_name, writer='ffmpeg', codec='mjpeg')
        plt.style.use('default')

%matplotlib
#plt.style.use('default')
hdc = hdeltac(celltype='hdeltac')
df = hdc.sheet
figfol = '/Volumes/LACIE/Andy/hdeltac/figures'

def all_directions_RDP_first_last_seg_edge_tracking(layer = 'upper'):
    plt.style.use('default')
    hdc = hdeltac(celltype='hdeltac')
    df = hdc.sheet
    from matplotlib.backends.backend_pdf import PdfPages
    figfol = '/Volumes/LACIE/Andy/hdeltac/figures'
    offset = np.pi/2
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='polar')
    ax2 = fig.add_subplot(122, projection='polar')
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
    ax2.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d['DP_analysis'] == True:
                print(d)
                border_edge = d['border_edge']
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                ex.plot_allo_directions_x_y()
                _, ang_dict = ex.plot_RDP_segment_phase(layer)
                for key in list(ang_dict.keys()):
                    [ang_avg, r_avg] = ang_dict[key]
                    if border_edge=='left':
                        ang_avg = -ang_avg
                    if key == 'outbound_eb':
                        color = sns.color_palette()[0]
                        axis = ax1
                        label = 'outbound'
                    if key == 'inbound_eb':
                        color = sns.color_palette()[1]
                        axis = ax1
                        label = 'inbound'
                    if key == 'outbound_fb':
                        color = sns.color_palette()[0]
                        axis = ax2
                        label = 'outbound'
                    if key == 'inbound_fb':
                        color = sns.color_palette()[1]
                        axis = ax2
                        label = 'inbound'
                    axis.plot([ang_avg, ang_avg], [0,r_avg], color=color, linewidth=2, label=label)
                    axis.plot(ang_avg, r_avg, 'o',color=color, linewidth=2, markersize=3)
    ax1.set_theta_offset(offset)
    ax2.set_theta_offset(offset)
    ax1.set_title('EPG phase')
    ax2.set_title('HDC phase')
    # ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
    # ax2.legend(loc='upper left', bbox_to_anchor=(1,1))
    fig.savefig(os.path.join(figfol, 'inbound_outbound_directions_'+layer+'.pdf'))

def bump_amp_clean_air_v_et():
    """
    calculate the bump amplitude in clean air vs edge tracking
    """
    figfol = '/Volumes/LACIE/Andy/hdeltac/figures'
    edge_tracking = []
    clean_air = []
    fig, axs = plt.subplots(1,1)
    plt.style.use('default')
    hdc = hdeltac(celltype='hdeltac')
    df = hdc.sheet
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d['DP_analysis'] == True:
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                upper_wedges = ex.fb.get_layer_wedges('upper')
                bumps_before, bumps_after = ex.fb.bumps_air_before_after(upper_wedges)
                bumps_after = np.mean(bumps_after, axis=0)
                axs.plot(bumps_after, color = 'orange', alpha=0.5)
                edge_tracking.append(bumps_after)
            if d['clean_air'] == True:
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                upper_wedges = ex.fb.get_layer_wedges('upper')
                bumps_in, bumps_out = ex.eb.bumps_in_out(upper_wedges)
                bumps_out = np.mean(bumps_out, axis=0)
                axs.plot(bumps_out, color='blue', alpha=0.5)
                clean_air.append(bumps_out)
    clean_air = np.mean(np.array(clean_air), axis=0)
    edge_tracking = np.mean(np.array(edge_tracking), axis=0)
    axs.plot(clean_air, color = 'blue', label='clean air')
    axs.plot(edge_tracking, color = 'orange', label = 'edge tracking')
    axs.legend()
    fig.savefig(os.path.join(figfol, 'clean_air_v_et_bump_amp.pdf'))

def phase_differences_clean_air(analysis_type = 'DP_analysis'):
    figfol = '/Volumes/LACIE/Andy/hdeltac/figures'
    edge_tracking = []
    clean_air = []
    plt.style.use('default')
    hdc = hdeltac(celltype='hdeltac')
    df = hdc.sheet

    # plot the phase difference between the upper and lower layers
    fig, axs = plt.subplots(1,1, figsize=(4,4))
    all_phase_diff_layer = {
    'air': [],
    'odor': []
    }
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d[analysis_type] == True:
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                phase_diff, _,_= ex.get_phase_diff_between_layers()
                if analysis_type=='clean_air_analysis':
                    all_phase_diff_layer['air'].append(phase_diff)
                    sns.histplot(phase_diff, ax=axs, color=sns.color_palette()[0],stat='probability', alpha=0.5, element="step", fill=False)
                if analysis_type=='DP_analysis':
                    ex.fb.load_processed()
                    ft2 = ex.fb.ft2
                    ft2['phase_diff']=phase_diff
                    d_all, di, do = fn.inside_outside(ft2)
                    phase_diff = {
                    'odor': [],
                    'air': []
                    }
                    for key in list(di.keys()):
                        pd = di[key].phase_diff.to_numpy()
                        phase_diff['odor'].append(pd)
                    phase_diff['odor'] = np.concatenate(phase_diff['odor'])
                    sns.histplot(phase_diff['odor'], ax=axs, color=sns.color_palette()[1],stat='probability', alpha=0.5, element="step", fill=False)
                    for key in list(do.keys()):
                        pd = do[key].phase_diff.to_numpy()
                        phase_diff['air'].append(pd)
                    phase_diff['air'] = np.concatenate(phase_diff['air'])
                    sns.histplot(phase_diff['air'], ax=axs, color=sns.color_palette()[0],stat='probability', alpha=0.5, element="step", fill=False)
                    all_phase_diff_layer['air'].append(phase_diff['air'])
                    all_phase_diff_layer['odor'].append(phase_diff['odor'])
    sns.histplot(np.concatenate(all_phase_diff_layer['air']), ax=axs, color=sns.color_palette()[0],stat='probability', alpha=1, element="step", fill=False)
    if analysis_type=='DP_analysis':
        sns.histplot(np.concatenate(all_phase_diff_layer['odor']), ax=axs, color=sns.color_palette()[1],stat='probability', alpha=1, element="step", fill=False)
    fig.savefig(os.path.join(figfol, analysis_type+'_upper_lower_phase_difference.pdf'))
    #sns.histplot(np.concatenate(all_phase_diff_layer['air']), ax=axs, color=sns.color_palette()[0],stat='density', element="step", fill=False)
    #fig.savefig(os.path.join(figfol, 'clean_air_upper_lower_phase_difference.pdf'))

    if analysis_type == 'clean_air_analysis':
        fig, axs = plt.subplots(1,1, figsize=(4,4))
    elif analysis_type == 'DP_analysis':
        fig, axs = plt.subplots(1,2, figsize=(8,4))

    all_phase_diff_layer = {
    'air_upper': [],
    'air_lower': [],
    'odor_upper': [],
    'odor_lower': []
    }
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d[analysis_type] == True:
                # fig, axs = plt.subplots(1,1, figsize=(4,4))
                print(d)
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                phase_diff_u, _,_= ex.get_phase_diff_between_layer_eb(layer='upper')
                phase_diff_l, _,_= ex.get_phase_diff_between_layer_eb(layer='lower')
                if analysis_type=='clean_air_analysis':
                    all_phase_diff_layer['air_upper'].append(phase_diff_u)
                    all_phase_diff_layer['air_lower'].append(phase_diff_l)
                    sns.histplot(phase_diff_u, ax=axs, color=sns.color_palette()[6],stat='probability', alpha=0.5, element="step", fill=False)
                    sns.histplot(phase_diff_l, ax=axs, color=sns.color_palette()[7],stat='probability', alpha=0.5, element="step", fill=False)
                if analysis_type=='DP_analysis':
                    ex.fb.load_processed()
                    ft2 = ex.fb.ft2
                    ft2['phase_diff_u']=phase_diff_u
                    ft2['phase_diff_l']=phase_diff_l
                    d_all, di, do = fn.inside_outside(ft2)
                    phase_diff = {
                    'odor_upper': [],
                    'odor_lower': [],
                    'air_upper': [],
                    'air_lower': [],
                    }
                    for key in list(di.keys()):
                        pd_upper = di[key].phase_diff_u.to_numpy()
                        pd_lower = di[key].phase_diff_l.to_numpy()
                        phase_diff['odor_upper'].append(pd_upper)
                        phase_diff['odor_lower'].append(pd_lower)
                    phase_diff['odor_upper'] = np.concatenate(phase_diff['odor_upper'])
                    phase_diff['odor_lower'] = np.concatenate(phase_diff['odor_lower'])
                    all_phase_diff_layer['odor_upper'].append(phase_diff['odor_upper'])
                    all_phase_diff_layer['odor_lower'].append(phase_diff['odor_lower'])
                    sns.histplot(phase_diff['odor_upper'], ax=axs[0], color=sns.color_palette()[6],stat='probability', alpha=0.5, element="step", fill=False)
                    sns.histplot(phase_diff['odor_lower'], ax=axs[0], color=sns.color_palette()[7],stat='probability', alpha=0.5, element="step", fill=False)
                    for key in list(do.keys())[1:]:
                        pd_upper = do[key].phase_diff_u.to_numpy()
                        pd_lower = do[key].phase_diff_l.to_numpy()
                        phase_diff['air_upper'].append(pd_upper)
                        phase_diff['air_lower'].append(pd_lower)
                    phase_diff['air_upper'] = np.concatenate(phase_diff['air_upper'])
                    phase_diff['air_lower'] = np.concatenate(phase_diff['air_lower'])
                    all_phase_diff_layer['air_upper'].append(phase_diff['air_upper'])
                    all_phase_diff_layer['air_lower'].append(phase_diff['air_lower'])
                    sns.histplot(phase_diff['air_upper'], ax=axs[1], color=sns.color_palette()[6],stat='probability', alpha=0.5, element="step", fill=False)
                    sns.histplot(phase_diff['air_lower'], ax=axs[1], color=sns.color_palette()[7],stat='probability', alpha=0.5, element="step", fill=False)
    if analysis_type=='clean_air_analysis':
        sns.histplot(np.concatenate(all_phase_diff_layer['air_upper']), ax=axs, color=sns.color_palette()[6], stat='probability', element="step", fill=False, label='upper')
        sns.histplot(np.concatenate(all_phase_diff_layer['air_lower']), ax=axs, color=sns.color_palette()[7], stat='probability', element="step", fill=False, label='lower')
    if analysis_type=='DP_analysis':
        sns.histplot(np.concatenate(all_phase_diff_layer['odor_upper']), ax=axs[0], color=sns.color_palette()[6], stat='probability', element="step", fill=False, label='upper')
        sns.histplot(np.concatenate(all_phase_diff_layer['odor_lower']), ax=axs[0], color=sns.color_palette()[7], stat='probability', element="step", fill=False, label='lower')
        sns.histplot(np.concatenate(all_phase_diff_layer['air_upper']), ax=axs[1], color=sns.color_palette()[6], stat='probability', element="step", fill=False, label='upper')
        sns.histplot(np.concatenate(all_phase_diff_layer['air_lower']), ax=axs[1], color=sns.color_palette()[7], stat='probability', element="step", fill=False, label='lower')
    fig.tight_layout()
    fig.savefig(os.path.join(figfol, analysis_type+'_upper_lower_phase_v_epg.pdf'))

def plot_clean_air_heatmaps():
    figfol = '/Volumes/LACIE/Andy/hdeltac/figures'
    edge_tracking = []
    clean_air = []
    plt.style.use('default')
    hdc = hdeltac(celltype='hdeltac')
    df = hdc.sheet

    # plot the phase difference between the layers
    all_phase_diff_layer = []
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d['DP_analysis'] == True:
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                ex.plot_allo_directions_x_y()
                ex.plot_all_heatmaps()
    return ex

def individual_fly_plot_eb_fb_inbound_vector():
    hdc = hdeltac(celltype='hdeltac')
    df = hdc.sheet
    df = df[df.DP_analysis==True]
    df.reset_index(inplace=True)
    num_flies_to_analyze = len(df)
    figfol = '/Volumes/LACIE/Andy/hdeltac/figures'
    datafol = '/Volumes/LACIE/Andy/hdeltac/'
    fig_all = plt.figure(figsize=(14,2))
    offset = np.pi/2
    for i,row in df.iterrows():
        d = row.to_dict()
        border_edge = d['border_edge']
        ex = hdc_example(d,datafol)
        ex.fb.load_processed()

        ax = fig_all.add_subplot(1,num_flies_to_analyze,i+1,projection='polar')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
        ax.set_theta_offset(np.pi/2)
        _, ang_dict = ex.plot_RDP_segment_phase()
        for key in list(ang_dict.keys()):
            [ang_avg, r_avg] = ang_dict[key]
            if border_edge=='left':
                ang_avg = -ang_avg
            if key == 'inbound_eb':
                ax.plot([ang_avg, ang_avg], [0,1], color='purple', linewidth=2)
            if key == 'inbound_fb':
                ax.plot([ang_avg, ang_avg], [0,1], color='orange', linewidth=2)
    fig_all.tight_layout()
    fig_all.savefig(os.path.join(figfol, 'eb_fb_inbound_vector.pdf'))

def plot_upvel_xvel_v_bump_amp(analysis_type = 'DP_analysis'):
    figfol = '/Volumes/LACIE/Andy/hdeltac/figures'
    edge_tracking = []
    clean_air = []
    hdc = hdeltac(celltype='hdeltac')
    df = hdc.sheet
    # plot the phase difference between the layers
    #fig, axs = plt.subplots(1,1)
    bump_amp_all = {
    'upper_air': [],
    'lower_air': [],
    'upper_odor': [],
    'lower_odor': []
    }
    upwind_velocity_all = {
    'air': [],
    'odor': []
    }
    crosswind_velocity_all = {
    'air': [],
    'odor': []
    }



    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d[analysis_type] == True:
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                ex.fb.load_processed()
                ft2 = ex.fb.ft2
                # upwind and crosswind speed/velocity
                del_t = np.mean(np.diff(ft2.seconds))
                y = ft2.ft_posy.to_numpy()
                x = ft2.ft_posx.to_numpy()
                upwind_velocity = np.gradient(y)/del_t
                crosswind_velocity = np.abs(np.gradient(x))/del_t
                upper_wedges = ex.fb.get_layer_wedges('upper')
                lower_wedges = ex.fb.get_layer_wedges('lower')

                stops, df_stop = fn.find_stops(ft2)
                moving_ix = df_stop[df_stop.stop==1].index.to_list()
                #moving_ix = np.argwhere(np.abs(upwind_velocity)>1)[:,0]
                ix_in = ft2[ft2.instrip==1.0].index.to_numpy()
                ix_out = ft2[ft2.instrip==0.0].index.to_numpy()
                if analysis_type == 'clean_air_analysis':
                    start_ix = np.array([0])
                else:
                    start_ix = ix_in[0]
                ix = np.argmin(np.abs(moving_ix-start_ix)) # index in ix_out where the pre odor time ends
                moving_ix = moving_ix[ix:] # index where fly is moving
                if analysis_type == 'clean_air_analysis':
                    air_moving_ix = np.intersect1d(moving_ix,ix_out)
                elif analysis_type == 'DP_analysis':
                    air_moving_ix = np.intersect1d(moving_ix,ix_out)
                    odor_moving_ix = np.intersect1d(moving_ix,ix_in)

                up_vel_air = upwind_velocity[air_moving_ix]
                upwind_velocity_all['air'].append(up_vel_air)
                cvel_air = crosswind_velocity[air_moving_ix]
                crosswind_velocity_all['air'].append(cvel_air)
                bump_amp = np.mean(upper_wedges[air_moving_ix,:],axis=1)
                bump_amp_all['upper_air'].append(bump_amp)
                bump_amp = np.mean(lower_wedges[air_moving_ix,:],axis=1)
                bump_amp_all['lower_air'].append(bump_amp)

                if analysis_type == 'DP_analysis':
                    up_vel_odor = upwind_velocity[odor_moving_ix]
                    upwind_velocity_all['odor'].append(up_vel_odor)
                    cvel_odor = crosswind_velocity[odor_moving_ix]
                    crosswind_velocity_all['odor'].append(cvel_odor)
                    bump_amp = np.mean(upper_wedges[odor_moving_ix,:],axis=1)
                    bump_amp_all['upper_odor'].append(bump_amp)
                    bump_amp = np.mean(lower_wedges[odor_moving_ix,:],axis=1)
                    bump_amp_all['lower_odor'].append(bump_amp)


    df_air = pd.DataFrame({'upwind_velocity':np.concatenate(upwind_velocity_all['air']), 'crosswind_velocity':np.concatenate(crosswind_velocity_all['air']), 'bump_amp_upper': np.concatenate(bump_amp_all['upper_air']), 'bump_amp_lower': np.concatenate(bump_amp_all['lower_air'])})
    if analysis_type == 'DP_analysis':
        df_odor = pd.DataFrame({'upwind_velocity':np.concatenate(upwind_velocity_all['odor']), 'crosswind_velocity':np.concatenate(crosswind_velocity_all['odor']), 'bump_amp_upper': np.concatenate(bump_amp_all['upper_odor']), 'bump_amp_lower': np.concatenate(bump_amp_all['lower_odor'])})
    #df = df.groupby(df.index // 10).mean()
    fig,axs = plt.subplots(2,2, figsize=(6,6))
    sns.regplot(df_air.upwind_velocity, df_air.bump_amp_upper, ax=axs[0,0], scatter_kws={'alpha':0.05,'color':pl.outside_color, 'edgecolors':'none', 'rasterized':True})
    sns.regplot(df_air.crosswind_velocity, df_air.bump_amp_upper, ax=axs[0,1], scatter_kws={'alpha':0.05,'color':pl.outside_color, 'edgecolors':'none','rasterized':True})
    sns.regplot(df_air.upwind_velocity, df_air.bump_amp_lower, ax=axs[1,0], scatter_kws={'alpha':0.05,'color':pl.outside_color, 'edgecolors':'none', 'rasterized':True})
    sns.regplot(df_air.crosswind_velocity, df_air.bump_amp_lower, ax=axs[1,1], scatter_kws={'alpha':0.05,'color':pl.outside_color, 'edgecolors':'none', 'rasterized':True})
    if analysis_type == 'DP_analysis':
        sns.regplot(df_odor.upwind_velocity, df_odor.bump_amp_upper, ax=axs[0,0], scatter_kws={'alpha':0.05,'color':pl.inside_color, 'edgecolors':'none', 'rasterized':True})
        sns.regplot(df_odor.crosswind_velocity, df_odor.bump_amp_upper, ax=axs[0,1], scatter_kws={'alpha':0.05,'color':pl.inside_color, 'edgecolors':'none', 'rasterized':True})
        sns.regplot(df_odor.upwind_velocity, df_odor.bump_amp_lower, ax=axs[1,0], scatter_kws={'alpha':0.05,'color':pl.inside_color, 'edgecolors':'none', 'rasterized':True})
        sns.regplot(df_odor.crosswind_velocity, df_odor.bump_amp_lower, ax=axs[1,1], scatter_kws={'alpha':0.05,'color':pl.inside_color, 'edgecolors':'none', 'rasterized':True})
    fig.tight_layout()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_air.upwind_velocity, df_air.bump_amp_upper)
    print('for upwind velocity, p=',p_value,' r-squared=', r_value**2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_odor.upwind_velocity, df_odor.bump_amp_upper)
    print('for crosswind velocity, p=',p_value,' r-squared=', r_value**2)
    axs[0,0].set_xlabel('upwind velocity')
    axs[0,0].set_ylabel('HDC upper layer bump amplitude')
    axs[0,1].set_xlabel('crosswind speed')
    axs[0,1].set_ylabel('HDC upper layer bump amplitude')
    axs[1,0].set_xlabel('upwind velocity')
    axs[1,0].set_ylabel('HDC lower layer bump amplitude')
    axs[1,1].set_xlabel('crosswind speed')
    axs[1,1].set_ylabel('HDC lower layer bump amplitude')

    fig.savefig(os.path.join(figfol, 'bump_amplitude_v_upwind_crosswind_'+analysis_type+'_.pdf'))

%matplotlib
def plot_all_bumps_in_out():
    upper_in, upper_out = [],[]
    lower_in, lower_out = [],[]
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d['DP_analysis'] ==True:
                print(d)
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                wedges = ex.fb.get_layer_wedges(tag='upper')
                bumps_in, bumps_out = ex.fb.bumps_in_out(wedges)
                upper_in.append(np.mean(bumps_in, axis=0))
                upper_out.append(np.mean(bumps_out, axis=0))

                wedges = ex.fb.get_layer_wedges(tag='lower')
                bumps_in, bumps_out = ex.fb.bumps_in_out(wedges)
                lower_in.append(np.mean(bumps_in, axis=0))
                lower_out.append(np.mean(bumps_out, axis=0))
    fig, axs = plt.subplots(1, 2, sharey=True)
    for i in np.arange(len(upper_in)):
        axs[0].plot(upper_in[i], pl.inside_color, alpha=0.2)
        axs[0].plot(upper_out[i], pl.outside_color, alpha=0.2)
        axs[1].plot(lower_in[i], pl.inside_color, alpha=0.2)
        axs[1].plot(lower_out[i], pl.outside_color, alpha=0.2)
    axs[0].plot(np.mean(upper_in, axis=0), pl.inside_color, alpha=1)
    axs[0].plot(np.mean(upper_out, axis=0), pl.outside_color, alpha=1)
    axs[1].plot(np.mean(lower_in, axis=0), pl.inside_color, alpha=1)
    axs[1].plot(np.mean(lower_out, axis=0), pl.outside_color, alpha=1)
    fig.savefig(os.path.join(figfol, 'hdc_bumps_in_out.pdf'), transparent=True)

#plot_upvel_xvel_v_bump_amp()

# %%
%matplotlib
hdc = hdeltac(celltype='hdeltac')
df = hdc.sheet
comparison_type = 'replay'
def best_dp_match(segs, xy):
    """
    function for finding the best DP line segment to match a line segment
    which starts from the origin and goes to xy=[x,y]
    """
    distances = []
    for seg in segs:
        delta = seg-xy
        delta = delta**2
        dist = np.sqrt(np.sum(delta))
        distances.append(dist)
    match_ix = np.argmin(np.array(distances))
    return match_ix

def get_segment_info_comp(xy, all_info_comp, segs, ixs):
    ft2_comp = all_info_comp['ft2']
    x_comp = ft2_comp.ft_posx.to_numpy()
    y_comp = ft2_comp.ft_posy.to_numpy()
    match_ix = best_dp_match(segs,xy)
    return_ix = ixs[match_ix]
    i1 = return_ix[0]
    i2 = return_ix[1]

    # trajectory
    x_comp_seg = x_comp[i1:i2+1]
    y_comp_seg = y_comp[i1:i2+1]
    x_comp_seg = x_comp_seg-x_comp_seg[0]
    y_comp_seg = y_comp_seg-y_comp_seg[0]

    # for heatmaps
    fit_wedges_eb = np.rot90(all_info_comp['fit_wedges_eb'][i1:i2+1], k=1, axes=(0,1))
    fit_wedges_fb_upper = np.rot90(all_info_comp['fit_wedges_fb_upper'][i1:i2+1], k=1, axes=(0,1))
    fit_wedges_fb_lower = np.rot90(all_info_comp['fit_wedges_fb_lower'][i1:i2+1], k=1, axes=(0,1))
    eb_mean = np.mean(all_info_comp['fit_wedges_eb'][i1:i2+1], axis=0)
    fb_upper_mean = np.mean(all_info_comp['fit_wedges_fb_upper'][i1:i2+1], axis=0)
    fb_lower_mean = np.mean(all_info_comp['fit_wedges_fb_lower'][i1:i2+1], axis=0)

    # phases extracted from heatmaps
    def get_ang_mag(average_cosine):
        fit_ib, params_ib = fn.fit_cos(average_cosine)
        ang = fn.correct_fit_cos_phase(params_ib[2])
        mag = params_ib[0]
        ang_mag = [ang, mag]
        return ang_mag

    ang_mag_eb = get_ang_mag(eb_mean)
    ang_mag_fb_upper = get_ang_mag(fb_upper_mean)
    ang_mag_fb_lower = get_ang_mag(fb_lower_mean)

    # average phases
    eb_phase_mean = fn.circmean(all_info_comp['offset_eb_phase'][i1:i2+1])
    fb_upper_phase_mean = fn.circmean(all_info_comp['offset_fb_phase_upper'][i1:i2+1])
    fb_lower_phase_mean = fn.circmean(all_info_comp['offset_fb_phase_lower'][i1:i2+1])

    info = {
    'x_comp_seg': x_comp_seg,
    'y_comp_seg': y_comp_seg,
    'x_comp_seg_0': x_comp_seg-x_comp_seg[0],
    'y_comp_seg_0': y_comp_seg-y_comp_seg[0],
    'fit_wedges_eb': fit_wedges_eb,
    'fit_wedges_fb_upper': fit_wedges_fb_upper,
    'fit_wedges_fb_lower': fit_wedges_fb_lower,
    'eb_mean': eb_mean,
    'fb_upper_mean': fb_upper_mean,
    'fb_lower_mean': fb_lower_mean,
    'eb_phase_mean': eb_phase_mean,
    'fb_upper_phase_mean': fb_upper_phase_mean,
    'fb_lower_phase_mean': fb_lower_phase_mean,
    'ang_mag_eb': ang_mag_eb,
    'ang_mag_fb_upper': ang_mag_fb_upper,
    'ang_mag_fb_lower': ang_mag_fb_lower
    }
    return info

def get_segment_info(all_info_et, start_pt, end_pt):
    ft2 = all_info_et['ft2']
    x = ft2.ft_posx.to_numpy()
    y = ft2.ft_posy.to_numpy()

    # for a given [x,y] start and end point of a segment,
    # find the corresponding indices

    # i1 = np.where(x==start_pt[0])[0][0]
    # i2 = np.where(x==end_pt[0])[0][0]

    i1 = np.argmin(np.abs(x-start_pt[0]))
    i2 = np.argmin(np.abs(x-end_pt[0]))

    # trajectory
    x_seg = x[i1:i2+1]
    y_seg = y[i1:i2+1]
    x_seg = x_seg-x_seg[0]
    y_seg = y_seg-y_seg[0]

    # for heatmaps
    fit_wedges_eb = np.rot90(all_info_et['fit_wedges_eb'][i1:i2+1], k=1, axes=(0,1))
    fit_wedges_fb_upper = np.rot90(all_info_et['fit_wedges_fb_upper'][i1:i2+1], k=1, axes=(0,1))
    fit_wedges_fb_lower = np.rot90(all_info_et['fit_wedges_fb_lower'][i1:i2+1], k=1, axes=(0,1))
    eb_mean = np.mean(all_info_et['fit_wedges_eb'][i1:i2+1], axis=0)
    fb_upper_mean = np.mean(all_info_et['fit_wedges_fb_upper'][i1:i2+1], axis=0)
    fb_lower_mean = np.mean(all_info_et['fit_wedges_fb_lower'][i1:i2+1], axis=0)

    # phases extracted from heatmaps
    def get_ang_mag(average_cosine):
        fit_ib, params_ib = fn.fit_cos(average_cosine)
        ang = fn.correct_fit_cos_phase(params_ib[2])
        mag = params_ib[0]
        ang_mag = [ang, mag]
        return ang_mag

    ang_mag_eb = get_ang_mag(eb_mean)
    ang_mag_fb_upper = get_ang_mag(fb_upper_mean)
    ang_mag_fb_lower = get_ang_mag(fb_lower_mean)

    # average phases
    eb_phase_mean = fn.circmean(all_info_et['offset_eb_phase'][i1:i2+1])
    fb_upper_phase_mean = fn.circmean(all_info_et['offset_fb_phase_upper'][i1:i2+1])
    fb_lower_phase_mean = fn.circmean(all_info_et['offset_fb_phase_lower'][i1:i2+1])

    info = {
    'x_seg': x_seg,
    'y_seg': y_seg,
    'x_seg_0': x_seg-x_seg[0],
    'y_seg_0': y_seg-y_seg[0],
    'fit_wedges_eb': fit_wedges_eb,
    'fit_wedges_fb_upper': fit_wedges_fb_upper,
    'fit_wedges_fb_lower': fit_wedges_fb_lower,
    'eb_mean': eb_mean,
    'fb_upper_mean': fb_upper_mean,
    'fb_lower_mean': fb_lower_mean,
    'eb_phase_mean': eb_phase_mean,
    'fb_upper_phase_mean': fb_upper_phase_mean,
    'fb_lower_phase_mean': fb_lower_phase_mean,
    'ang_mag_eb': ang_mag_eb,
    'ang_mag_fb_upper': ang_mag_fb_upper,
    'ang_mag_fb_lower': ang_mag_fb_lower
    }
    return info

def get_average_vector(all_vecs):
    def cosine_func(x, amp,phase):
        return amp * np.sin(x + phase)
    all_cos = []
    t=np.linspace(0,2*np.pi,16)
    for ang_mag in all_vecs:
        phase = ang_mag[0]
        amp = ang_mag[1]
        cos = cosine_func(t,amp,phase)
        all_cos.append(cos)
    all_cos = np.array(all_cos)
    all_cos = np.mean(all_cos, axis=0)
    _, params = fn.fit_cos(all_cos)
    ang = params[2]
    #ang = fn.correct_fit_cos_phase(ang)
    mag = params[0]
    return ang, mag

def final_segment_analysis():
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d['folder'] == '20220517_hdc_split_60d05_sytgcamp7f_Fly1-001':
                print(d)
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                figfol = ex.figure_folder
                border_edge = d['border_edge']

                # figure for mean phases
                fig_all=plt.figure()
                ax1_all = fig_all.add_subplot(3,1,1, projection='polar')
                ax2_all = fig_all.add_subplot(3,1,2, projection='polar')
                ax3_all = fig_all.add_subplot(3,1,3, projection='polar')
                ax1_all.set_theta_offset(np.pi/2)
                ax2_all.set_theta_offset(np.pi/2)
                ax3_all.set_theta_offset(np.pi/2)

                # variables for average
                all_vecs = {
                'eb': [],
                'eb_comp': [],
                'fb_upper': [],
                'fb_upper_comp': [],
                'fb_lower': [],
                'fb_lower_comp':[]
                }

                # find replay/clean_air trial
                fly = d['fly']
                df_fly = df.loc[df.fly==fly]
                d_comparison = df_fly.loc[df_fly.experiment==comparison_type].squeeze().to_dict()
                ex_comparison = hdc_example(d_comparison, datafol)

                # fetch info for edge-tracking and comparison trial
                all_info_et = ex.fetch_all_info_eb_fb()
                all_info_comp = ex_comparison.fetch_all_info_eb_fb()

                # fetch segment library for comparison trial
                segs,ixs = ex_comparison.make_segment_library()

                do = all_info_et['outside_segs']
                # iterate through the edge tracking outside segments
                for key in list(do.keys())[1:]:
                    temp = do[key]
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()
                    if ex.experiment == '45':
                        x,y = fn.coordinate_rotation(x, y, np.pi/4)
                    _,pathlenth = fn.path_length(x,y)
                    if np.abs(x[-1]-x[0])<1 and pathlenth>10: # returns to the edge and goes more than 10mm
                        if ex.experiment == '45':
                            x,y = fn.coordinate_rotation(x, y, -np.pi/4)

                        # get the RDP simplification of the selected outside trajectory
                        simplified = fn.rdp_simp(x,y, epsilon=1)
                        simp_x = simplified[:,0]
                        simp_y = simplified[:,1]

                        # first segment end index (outbound) in segement
                        ix1 = np.where(y==simplified[:,1][1])
                        ix1 = ix1[0][0]

                        # last segment index start (inbound) in segment
                        ix2 = np.where(y==simplified[:,1][-2])
                        ix2 = ix2[0][0]

                        # center the return segment, find closest match
                        xy = simplified[-1]-simplified[-2]
                        info_comp = get_segment_info_comp(xy, all_info_comp, segs, ixs)
                        info = get_segment_info(all_info_et, simplified[-2], simplified[-1])


                        if True:
                            # plot the outside trajectory, first/last RDP segments
                            fig,axs = plt.subplots(4,3, figsize = (8,8))
                            axs[0,0].plot(x,y)
                            axs[0,0].plot(x[0],y[0],'o', color='purple')
                            axs[0,0].plot(x[-1],y[-1],'o', color='green')
                            axs[0,0].plot(simp_x[:2], simp_y[:2], 'k')
                            axs[0,0].plot(simp_x[-2:], simp_y[-2:], 'k')
                            #axs[0,0].plot(x[:ix1+1],y[:ix1+1], 'r')
                            #axs[0,0].plot(x[ix2:], y[ix2:], 'r')
                            axs[0,0].axis('equal')
                            axs[0,0].set_title('outside_trajectory')

                            # plot the inbound RDP segment
                            axs[0,1].plot(info['x_seg_0'], info['y_seg_0'])
                            axs[0,1].plot(info_comp['x_comp_seg_0'], info_comp['y_comp_seg_0'])
                            axs[0,1].plot(simp_x[-2:]-simp_x[-2:][0], simp_y[-2:]-simp_y[-2:][0], 'k')
                            axs[0,1].plot(xy[0],xy[1],'o', color='green')
                            axs[0,1].axis('equal')
                            axs[0,1].set_title('inbound segment')

                            # plot net motion
                            fig.delaxes(axs[0,2])
                            #axs[0,2].plot(temp.net_motion)

                            # plot the heatmaps for the segment
                            sns.heatmap(info['fit_wedges_eb'], vmin=0, vmax=1, ax = axs[1,0], cbar=None)
                            sns.heatmap(info['fit_wedges_fb_upper'], vmin=0, vmax=2, ax = axs[2,0], cbar=None)
                            sns.heatmap(info['fit_wedges_fb_lower'], vmin=0, vmax=2, ax = axs[3,0], cbar=None)
                            axs[1,0].set_title('EPG signal')
                            axs[1,1].set_title('EPG (replay) signal')
                            axs[1,2].set_title('EPG phases')
                            axs[2,0].set_title('HDC upper signal')
                            axs[2,1].set_title('HDC upper (air) signal')
                            axs[2,2].set_title('HDC upper phases')
                            axs[3,0].set_title('HDC lower signal')
                            axs[3,1].set_title('HDC lower (air) signal')
                            axs[3,2].set_title('HDC lower phases')

                            # plot the heatmaps for the replay segment
                            sns.heatmap(info_comp['fit_wedges_eb'], vmin=0, vmax=1, ax = axs[1,1], cbar=None)
                            sns.heatmap(info_comp['fit_wedges_fb_upper'], vmin=0, vmax=2, ax = axs[2,1], cbar=None)
                            sns.heatmap(info_comp['fit_wedges_fb_lower'], vmin=0, vmax=2, ax = axs[3,1], cbar=None)

                            for j in [0,1]:
                                for i in [1,2,3]:
                                    axs[i,j].get_xaxis().set_visible(False)
                                    axs[i,j].set_yticks([-0.5,8,15.5])
                                    axs[i,j].set_yticklabels(['180', '0', '-180'])

                            # plot the average bump
                            axs[1,2].plot(info['eb_mean'])
                            axs[2,2].plot(info['fb_upper_mean'])
                            axs[3,2].plot(info['fb_lower_mean'])
                            axs[1,2].plot(info_comp['eb_mean'])
                            axs[2,2].plot(info_comp['fb_upper_mean'])
                            axs[3,2].plot(info_comp['fb_lower_mean'])
                            for i in [1,2,3]:
                                axs[i,2].set_ylim(-0.5,1.5)
                                axs[i,2].set_xticks([0,8,15])
                                axs[i,2].set_xticklabels(['-180', '0', '180'])
                            fig.tight_layout()
                            fig.savefig(os.path.join(figfol, str(key)+'_heatmap_bout.pdf'))

                            #####
                            #####
                            fig=plt.figure(figsize = (3,8))
                            # just plot cosine phases
                            ax1 = fig.add_subplot(3,1,1, projection='polar')
                            ax2 = fig.add_subplot(3,1,2, projection='polar')
                            ax3 = fig.add_subplot(3,1,3, projection='polar')
                            ax1.set_title('EPG phase')
                            ax2.set_title('upper layer phase')
                            ax3.set_title('lower layer phase')
                            ax1.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            ax2.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            ax3.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            ax1.set_title('EPG angles')
                            ax2.set_title('upper layer angles')
                            ax3.set_title('lower layer angles')
                            # cosine phases
                            # ax4 = fig.add_subplot(3,2,2, projection='polar')
                            # ax5 = fig.add_subplot(3,2,4, projection='polar')
                            # ax6 = fig.add_subplot(3,2,6, projection='polar')
                            # ax4.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            # ax5.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            # ax6.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            # ax4.set_title('EPG phase')
                            # ax5.set_title('upper layer phase')
                            # ax6.set_title('lower layer phase')

                            # ang = -info['eb_phase_mean']
                            # ax1.plot([ang,ang],[0,1])
                            # ang = -info_comp['eb_phase_mean']
                            # ax1.plot([ang,ang],[0,1])
                            # ang = -info['fb_upper_phase_mean']
                            # ax2.plot([ang,ang],[0,1])
                            # ang = -info_comp['fb_upper_phase_mean']
                            # ax2.plot([ang,ang],[0,1])
                            # ang = -info['fb_lower_phase_mean']
                            # ax3.plot([ang,ang],[0,1])
                            # ang = -info_comp['fb_lower_phase_mean']
                            # ax3.plot([ang,ang],[0,1])
                            ang_mag = info['ang_mag_eb']
                            ax1.plot([ang_mag[0],ang_mag[0]],[0,1])
                            ang_mag = info_comp['ang_mag_eb']
                            ax1.plot([ang_mag[0],ang_mag[0]],[0,1])
                            ang_mag = info['ang_mag_fb_upper']
                            ax2.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]])
                            ang_mag = info_comp['ang_mag_fb_upper']
                            ax2.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]])
                            ang_mag = info['ang_mag_fb_lower']
                            ax3.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]])
                            ang_mag = info_comp['ang_mag_fb_lower']
                            ax3.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]])

                            ax1.set_theta_offset(np.pi/2)
                            ax2.set_theta_offset(np.pi/2)
                            ax3.set_theta_offset(np.pi/2)
                            # ax4.set_theta_offset(np.pi/2)
                            # ax5.set_theta_offset(np.pi/2)
                            # ax6.set_theta_offset(np.pi/2)
                            fig.savefig(os.path.join(figfol, str(key)+'_vectors_bout.pdf'))
                            fig.tight_layout()

                        ang_mag = info['ang_mag_eb']
                        all_vecs['eb'].append(info['ang_mag_eb'])
                        ax1_all.plot([ang_mag[0],ang_mag[0]],[0,1], color=sns.color_palette()[0], alpha=0.2)

                        ang_mag = info_comp['ang_mag_eb']
                        all_vecs['eb_comp'].append(info_comp['ang_mag_eb'])
                        ax1_all.plot([ang_mag[0],ang_mag[0]],[0,1], color=sns.color_palette()[1], alpha=0.2)

                        ang_mag = info['ang_mag_fb_upper']
                        all_vecs['fb_upper'].append(info['ang_mag_fb_upper'])
                        ax2_all.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]], color=sns.color_palette()[0], alpha=0.2)

                        ang_mag = info_comp['ang_mag_fb_upper']
                        all_vecs['fb_upper_comp'].append(info_comp['ang_mag_fb_upper'])
                        ax2_all.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]], color=sns.color_palette()[1], alpha=0.2)

                        ang_mag = info['ang_mag_fb_lower']
                        all_vecs['fb_lower'].append(info['ang_mag_fb_lower'])
                        ax3_all.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]], color=sns.color_palette()[0], alpha=0.2)

                        ang_mag = info_comp['ang_mag_fb_lower']
                        all_vecs['fb_lower_comp'].append(info_comp['ang_mag_fb_lower'])
                        ax3_all.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]], color=sns.color_palette()[1], alpha=0.2)

                ang, mag = get_average_vector(all_vecs['eb'])
                ax1_all.plot([ang,ang],[0,1], color=sns.color_palette()[0])

                ang, mag = get_average_vector(all_vecs['eb_comp'])
                ax1_all.plot([ang,ang],[0,1], color=sns.color_palette()[1])

                ang, mag = get_average_vector(all_vecs['fb_upper'])
                ax2_all.plot([ang,ang],[0,mag], color=sns.color_palette()[0])

                ang, mag = get_average_vector(all_vecs['fb_upper_comp'])
                ax2_all.plot([ang,ang],[0,mag], color=sns.color_palette()[1])

                ang, mag = get_average_vector(all_vecs['fb_lower'])
                ax3_all.plot([ang,ang],[0,mag], color=sns.color_palette()[0])

                ang, mag = get_average_vector(all_vecs['fb_lower_comp'])
                ax3_all.plot([ang,ang],[0,mag], color=sns.color_palette()[1])

                fig_all.savefig(os.path.join(figfol, 'summary_phases.pdf'))

def plot_outside_trajectory():
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d['folder'] == '20220628_HDC_sytjGCaMP7f_Fly1_45-004':
                print(d)
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                figfol = ex.figure_folder
                border_edge = d['border_edge']

                with PdfPages(os.path.join(ex.figure_folder, d['folder']+'_outside_heatmaps.pdf')) as pdf:

                    all_info_et = ex.fetch_all_info_eb_fb()

                    do = all_info_et['outside_segs']
                    # iterate through the edge tracking outside segments
                    for key in list(do.keys())[1:]:
                        temp = do[key]
                        x = temp.ft_posx.to_numpy()
                        y = temp.ft_posy.to_numpy()

                        eb_phase = temp.offset_eb_phase.to_numpy()
                        lower_phase = temp.offset_fb_phase_lower.to_numpy()
                        upper_phase= temp.offset_fb_phase_upper.to_numpy()
                        time = temp.seconds.to_numpy()
                        time = time-time[0]


                        _,pathlenth = fn.path_length(x,y)
                        if ex.experiment == '45':
                            x,y = fn.coordinate_rotation(x, y, np.pi/4)
                        if np.abs(x[-1]-x[0])<1 and pathlenth>10: # returns to the edge and goes more than 10mm
                            if ex.experiment == '45':
                                x,y = fn.coordinate_rotation(x, y, -np.pi/4)

                            simplified = fn.rdp_simp(x,y, epsilon=1)

                            info = get_segment_info(all_info_et, simplified[0], simplified[-1])

                            x = x-x[0]
                            fig,axs = plt.subplots(5,1, figsize = (np.max(time),10))
                            axs[0].plot(x,y)
                            axs[0].plot(x[0],y[0],'o', color='purple')
                            axs[0].plot(x[-1],y[-1],'o', color='green')
                            axs[0].axis('equal')
                            axr_1 = axs[1].twinx()
                            axr_2 = axs[2].twinx()
                            axr_3 = axs[3].twinx()

                            sns.heatmap(info['fit_wedges_eb'], vmin=0, vmax=1, ax = axs[1], cbar=None)
                            sns.heatmap(info['fit_wedges_fb_upper'], vmin=0, vmax=2, ax = axs[2], cbar=None)
                            sns.heatmap(info['fit_wedges_fb_lower'], vmin=0, vmax=2, ax = axs[3], cbar=None)
                            axr_1.plot(eb_phase, '.', color='white', alpha=0.5)
                            axr_1.set_ylim(-np.pi,np.pi)
                            axr_1.set_yticks([-np.pi/2, np.pi/2])
                            axr_1.set_yticklabels(['left', 'right'])
                            axs[1].axis('off')
                            axs[1].title.set_text('EPGs')

                            axr_2.plot(upper_phase, '.', color='white', alpha=0.5)
                            axr_2.set_ylim(-np.pi,np.pi)
                            axr_2.set_yticks([-np.pi/2, np.pi/2])
                            axr_2.set_yticklabels(['left', 'right'])
                            axs[2].axis('off')
                            axs[2].title.set_text('upper layer hDC')

                            axr_3.plot(lower_phase, '.', color='white', alpha=0.5)
                            axr_3.set_ylim(-np.pi,np.pi)
                            axr_3.set_yticks([-np.pi/2, np.pi/2])
                            axr_3.set_yticklabels(['left', 'right'])
                            axs[3].axis('off')
                            axs[3].title.set_text('lower layer hDC')

                            axs[4].plot(time, x)
                            axs[4].set_xlim(time[0], time[-1])
                            axs[4].set_xlabel('time (s)')
                            axs[4].set_ylabel('x position (mm)')
                            fig.tight_layout()

                            pdf.savefig()  # saves the current figure into a pdf page
                            plt.close()

def project_heatmaps(all_heatmaps):
    lens = [i.shape[-1] for i in all_heatmaps]
    arr = np.ma.empty((16, np.max(lens),len(all_heatmaps)))
    arr.mask = True
    for idx, l in enumerate(all_heatmaps):
        arr[:, (np.max(lens)-l.shape[-1]):,idx] = l
    #return arr.mean(axis = -1)
    return np.nanmean(arr, axis=-1)

def tolerant_mean(arrs):

    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[(np.max(lens)-len(l)):,idx] = l
    return arr

def proxy_data(bins, grouped):
    proxy = []
    bin_width = np.mean(np.diff(bins))/2
    centers = bins[0:-1]+bin_width
    counts = np.sum(grouped,axis=0)*10000
    for i, count in enumerate(counts):
        vals = centers[i]*np.ones(int(count))
        proxy.append(vals)
    return(np.concatenate(proxy))

def last_points_analysis():
    """
    analyis which most clearly suggests an HDC vector pointing towards the edge
    """
    fig_proj, axs_proj = plt.subplots(1,3, figsize=(9,3))
    fig_proj.tight_layout()
    fig_hist, axs_hist = plt.subplots(1,3, figsize=(9,3))
    fig_hist.tight_layout()
    plot_individual=True
    grouped_phases_left = [[],[],[]]
    grouped_phases_right = [[],[],[]]
    grouped_heatmaps_left =[[],[],[]]
    grouped_heatmaps_right =[[],[],[]]
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d['DP_analysis'] ==True:
                print(d)
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                figfol = ex.figure_folder
                ex.plot_trajectory()

                border_edge = d['border_edge']
                if border_edge=='left':
                    color = 'blue'
                else:
                    color = 'red'

                all_info_et = ex.fetch_all_info_eb_fb()
                ft2 = all_info_et['ft2']
                fig,axs = plt.subplots(1,1)
                axs.plot(ft2.ft_posx, ft2.ft_posy)


                do = all_info_et['outside_segs']
                # iterate through the edge tracking outside segments
                all_ebs, all_fb_upper, all_fb_lower = [],[],[]
                all_eb_phase, all_fb_upper_phase, all_fb_lower_phase = [],[],[]
                all_net_motion = []
                for key in list(do.keys())[1:]:
                    temp = do[key]
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()

                    eb_phase = temp.offset_eb_phase.to_numpy()
                    lower_phase = temp.offset_fb_phase_lower.to_numpy()
                    upper_phase= temp.offset_fb_phase_upper.to_numpy()
                    net_motion = temp.net_motion.to_numpy()
                    time = temp.seconds.to_numpy()
                    time = time-time[0]


                    _,pathlenth = fn.path_length(x,y)
                    if ex.experiment == '45':
                        x,y = fn.coordinate_rotation(x, y, np.pi/4)
                    if np.abs(x[-1]-x[0])<1 and pathlenth>10: # returns to the edge and goes more than 10mm
                        if ex.experiment == '45':
                            x,y = fn.coordinate_rotation(x, y, -np.pi/4)

                        simplified = fn.rdp_simp(x,y, epsilon=1)

                        info = get_segment_info(all_info_et, simplified[0], simplified[-1])
                        all_eb_phase.append(eb_phase)
                        all_fb_upper_phase.append(upper_phase)
                        all_fb_lower_phase.append(lower_phase)

                        all_ebs.append(info['fit_wedges_eb'])
                        all_fb_lower.append(info['fit_wedges_fb_lower'])
                        all_fb_upper.append(info['fit_wedges_fb_upper'])

                        all_net_motion.append(net_motion)

                        if plot_individual:
                            fig,axs = plt.subplots(4,1)
                            axr_1 = axs[0].twinx()
                            axr_2 = axs[1].twinx()
                            axr_3 = axs[2].twinx()

                            sns.heatmap(info['fit_wedges_eb'], vmin=0, vmax=1, ax = axs[0], rasterized=True)
                            sns.heatmap(info['fit_wedges_fb_upper'], ax = axs[1], rasterized=True)
                            sns.heatmap(info['fit_wedges_fb_lower'], ax = axs[2],rasterized=True)
                            axr_1.plot(eb_phase, '.', color='white', alpha=0.5)
                            axr_1.set_ylim(-np.pi,np.pi)
                            axr_1.set_yticks([-np.pi/2, np.pi/2])
                            axr_1.set_yticklabels(['left', 'right'])
                            axs[0].axis('off')
                            axs[0].title.set_text('EPGs')

                            axr_2.plot(upper_phase, '.', color='white', alpha=0.5)
                            axr_2.set_ylim(-np.pi,np.pi)
                            axr_2.set_yticks([-np.pi/2, np.pi/2])
                            axr_2.set_yticklabels(['left', 'right'])
                            axs[1].axis('off')
                            axs[1].title.set_text('upper layer hDC')

                            axr_3.plot(lower_phase, '.', color='white', alpha=0.5)
                            axr_3.set_ylim(-np.pi,np.pi)
                            axr_3.set_yticks([-np.pi/2, np.pi/2])
                            axr_3.set_yticklabels(['left', 'right'])
                            axs[2].axis('off')
                            axs[2].title.set_text('lower layer hDC')

                            axs[3].plot(x)

                            fig.savefig(os.path.join(figfol, str(key)+'_heatmap.pdf'))
                            plt.close()

                # Make animal average
                fig,axs = plt.subplots(4,1)
                axr_1 = axs[0].twinx()
                axr_1.set_ylim(-np.pi,np.pi)
                axr_1.set_yticks([-np.pi/2, np.pi/2])
                axr_1.set_yticklabels(['left', 'right'])
                axr_2 = axs[1].twinx()
                axr_2.set_ylim(-np.pi,np.pi)
                axr_2.set_yticks([-np.pi/2, np.pi/2])
                axr_2.set_yticklabels(['left', 'right'])
                axr_3 = axs[2].twinx()
                axr_3.set_ylim(-np.pi,np.pi)
                axr_3.set_yticks([-np.pi/2, np.pi/2])
                axr_3.set_yticklabels(['left', 'right'])

                # plot heatmap averages for each individual animal
                sns.heatmap(project_heatmaps(all_ebs),ax=axs[0], cbar=False)
                sns.heatmap(project_heatmaps(all_fb_upper),ax=axs[1], cbar=False)
                sns.heatmap(project_heatmaps(all_fb_lower),ax=axs[2], cbar=False)

                # on the heatmaps, overlay the phases for the individual returns
                longest_t = project_heatmaps(all_ebs).shape[-1]
                x = np.arange(0, longest_t)
                axes = [axr_1,axr_2,axr_3,axs[3]]
                for i, measure in enumerate([all_eb_phase, all_fb_upper_phase, all_fb_lower_phase, all_net_motion]):
                    a = axes[i]
                    for m in measure:
                        #x = np.arange(int(axs[0].get_xlim()[0]), int(axs[0].get_xlim()[-1]))
                        a.plot(x[(len(x)-len(m)):], m, '.', color='white', alpha=0.1)
                    a.set_xlim(x[-1]-80, x[-1])
                fig.tight_layout()
                fig.savefig(os.path.join(figfol, 'average_heatmaps_phases.pdf'))

                # number of time points
                num_pts = 40

                # plot projection -- seems to be flipped
                for i, n in enumerate([all_ebs, all_fb_upper, all_fb_lower]):
                    avg = project_heatmaps(n).data
                    if border_edge=='left':
                        grouped_heatmaps_left[i].append(avg)
                    elif border_edge=='right':
                        grouped_heatmaps_right[i].append(avg)
                    ix1 = avg.shape[-1]-num_pts
                    axs_proj[i].plot(np.mean(avg[:, ix1:], axis=1), color=color)

                # plt the histograms for all the animals
                bins = np.linspace(-np.pi, np.pi, 17)
                fig, axs = plt.subplots(1,3, figsize=(9,3), sharex=True, sharey=True)
                fig.tight_layout()
                for i, n in enumerate([all_eb_phase, all_fb_upper_phase, all_fb_lower_phase]):
                    arr = tolerant_mean(n)
                    data = arr.data[-num_pts:,:]
                    mask = arr.mask[-num_pts:,:]
                    data = data[mask==False]
                    counts, _ =  np.histogram(data, bins=bins, density=True)
                    counts = counts/sum(counts)
                    if border_edge=='left':
                        grouped_phases_left[i].append(counts)
                    elif border_edge=='right':
                        grouped_phases_right[i].append(counts)
                    sns.histplot(data, ax=axs_hist[i],stat='probability', color=color, bins=bins, alpha=0.1, element='step', fill=False)
                    sns.histplot(data, ax=axs[i],stat='probability', color=color, bins=bins, alpha=1, element='step', fill=False)
                    axs[i].set_xticks([-np.pi/2, np.pi/2])
                    axs[i].set_xticklabels(['-90', '+90'])
                fig.savefig(os.path.join(figfol, str(num_pts)+'_histogram.pdf'))

    # plot the average histograms
    for i, n in enumerate(grouped_phases_left):
        sns.histplot(proxy_data(bins, n), ax=axs_hist[i],stat='probability', color='blue', bins=bins, alpha=1, element='step', fill=False)
        axs_hist[i].set_xticks([-np.pi/2, np.pi/2])
        axs_hist[i].set_xticklabels(['-90', '+90'])
    for i, n in enumerate(grouped_phases_right):
        sns.histplot(proxy_data(bins, n), ax=axs_hist[i],stat='probability', color='red', bins=bins, alpha=1, element='step', fill=False)
        axs_hist[i].set_xticks([-np.pi/2, np.pi/2])
        axs_hist[i].set_xticklabels(['-90', '+90'])
    fig_hist.savefig(os.path.join(datafol, 'figures', str(num_pts)+'_histogram.pdf'))
    return grouped_heatmaps_right, grouped_heatmaps_left

def aggregate_last_points_analysis():
    """
    plot the aggregate sinusoids for all the animals calculated in def last points analysis
    """
    grouped_heatmaps_right, grouped_heatmaps_left = last_points_analysis()
    num_pts=40
    fig_h,axs_h = plt.subplots(2,3)
    fig_h.tight_layout()
    fig_s,axs_s = plt.subplots(2,3)
    fig_s.tight_layout()
    for i in [0,1]:
        if i==0:
            group = grouped_heatmaps_right
            vmin=0.3
            vmax=0.7
        else:
            group = grouped_heatmaps_left
            vmin=0.2
            vmax=0.7
        for j in [0,1,2]:
            avg = group[j]
            ix1 = len(avg)-num_pts
            avg = project_heatmaps(avg)
            sns.heatmap(avg[:,ix1:],ax=axs_h[i,j], vmin=vmin, vmax=vmax)
            axs_s[i,j].plot(np.mean(avg[:,ix1:], axis=1))
            axs_s[i,j].plot(np.mean(avg[:,:ix1], axis=1))
    fig_h.savefig(os.path.join(datafol, 'figures', str(num_pts)+'_super_heatmap.pdf'))
    fig_s.savefig(os.path.join(datafol, 'figures', str(num_pts)+'_super_sin.pdf'))

last_points_analysis()
# %%
# def stopped_analysis():
"""
analyis which most clearly suggests an HDC vector pointing towards the edge
"""
fig_proj, axs_proj = plt.subplots(1,3, figsize=(9,3))
fig_proj.tight_layout()
fig_hist, axs_hist = plt.subplots(1,3, figsize=(9,3))
fig_hist.tight_layout()
grouped_phases_left = [[],[],[]]
grouped_phases_right = [[],[],[]]
grouped_heatmaps_left =[[],[],[]]
grouped_heatmaps_right =[[],[],[]]
for i in np.arange(len(df)):
    d = df.iloc[i].to_dict()
    if d['analyze'] == True:
        if d['DP_analysis'] ==True:
            print(d)
            datafol = '/Volumes/LACIE/Andy/hdeltac/'
            ex = hdc_example(d,datafol)
            figfol = ex.figure_folder
            #ex.plot_trajectory()

            border_edge = d['border_edge']
            if border_edge=='left':
                color = 'blue'
            else:
                color = 'red'

            all_info_et = ex.fetch_all_info_eb_fb()
            ft2 = all_info_et['ft2']
            _, ft2 = fn.find_stops(ft2)
            d, di, do = fn.inside_outside(ft2)
            #stops = df.stop.to_numpy()

            fig,axs = plt.subplots(1,1)
            axs.plot(ft2.ft_posx, ft2.ft_posy)


            #do = all_info_et['outside_segs']
            # iterate through the edge tracking outside segments
            all_ebs, all_fb_upper, all_fb_lower = [],[],[]
            all_eb_phase, all_fb_upper_phase, all_fb_lower_phase = [],[],[]
            all_net_motion = []
            for key in list(do.keys())[1:]:
                temp = do[key]
                stop1 = do[key].stop.to_numpy()
                stop = np.empty(len(stop1))
                stop[:] = np.nan
                stop[np.isnan(stop1)] = 1

                x = temp.ft_posx.to_numpy()
                y = temp.ft_posy.to_numpy()

                eb_phase = temp.offset_eb_phase.to_numpy()*stop
                lower_phase = temp.offset_fb_phase_lower.to_numpy()*stop
                upper_phase= temp.offset_fb_phase_upper.to_numpy()*stop
                net_motion = temp.net_motion.to_numpy()*stop
                time = temp.seconds.to_numpy()
                time = time-time[0]


                _,pathlenth = fn.path_length(x,y)
                if ex.experiment == '45':
                    x,y = fn.coordinate_rotation(x, y, np.pi/4)
                if np.abs(x[-1]-x[0])<1 and pathlenth>10: # returns to the edge and goes more than 10mm
                    if ex.experiment == '45':
                        x,y = fn.coordinate_rotation(x, y, -np.pi/4)

                    simplified = fn.rdp_simp(x,y, epsilon=1)

                    info = get_segment_info(all_info_et, simplified[0], simplified[-1])
                    # phases
                    all_eb_phase.append(eb_phase)
                    all_fb_upper_phase.append(upper_phase)
                    all_fb_lower_phase.append(lower_phase)

                    #heatmaps
                    all_ebs.append(info['fit_wedges_eb']*stop)
                    all_fb_lower.append(info['fit_wedges_fb_lower']*stop)
                    all_fb_upper.append(info['fit_wedges_fb_upper']*stop)

                    all_net_motion.append(net_motion)


            # Make animal average
            fig,axs = plt.subplots(4,1)
            axr_1 = axs[0].twinx()
            axr_1.set_ylim(-np.pi,np.pi)
            axr_1.set_yticks([-np.pi/2, np.pi/2])
            axr_1.set_yticklabels(['left', 'right'])
            axr_2 = axs[1].twinx()
            axr_2.set_ylim(-np.pi,np.pi)
            axr_2.set_yticks([-np.pi/2, np.pi/2])
            axr_2.set_yticklabels(['left', 'right'])
            axr_3 = axs[2].twinx()
            axr_3.set_ylim(-np.pi,np.pi)
            axr_3.set_yticks([-np.pi/2, np.pi/2])
            axr_3.set_yticklabels(['left', 'right'])

            # plot heatmap averages for each individual animal
            sns.heatmap(project_heatmaps(all_ebs),ax=axs[0], cbar=False)
            sns.heatmap(project_heatmaps(all_fb_upper),ax=axs[1], cbar=False)
            sns.heatmap(project_heatmaps(all_fb_lower),ax=axs[2], cbar=False)

            # on the heatmaps, overlay the phases for the individual returns
            longest_t = project_heatmaps(all_ebs).shape[-1]
            x = np.arange(0, longest_t)
            axes = [axr_1,axr_2,axr_3,axs[3]]
            for i, measure in enumerate([all_eb_phase, all_fb_upper_phase, all_fb_lower_phase, all_net_motion]):
                a = axes[i]
                for m in measure:
                    #x = np.arange(int(axs[0].get_xlim()[0]), int(axs[0].get_xlim()[-1]))
                    a.plot(x[(len(x)-len(m)):], m, '.', color='white', alpha=0.1)
                a.set_xlim(x[-1]-80, x[-1])
            fig.tight_layout()
            fig.savefig(os.path.join(figfol, 'average_heatmaps_phases.pdf'))

            # number of time points
            num_pts = 40

            # plot projection -- seems to be flipped
            for i, n in enumerate([all_ebs, all_fb_upper, all_fb_lower]):
                avg = project_heatmaps(n).data
                if border_edge=='left':
                    grouped_heatmaps_left[i].append(avg)
                elif border_edge=='right':
                    grouped_heatmaps_right[i].append(avg)
                ix1 = avg.shape[-1]-num_pts
                #axs_proj[i].plot(np.mean(avg[:, ix1:], axis=1), color=color)
                axs_proj[i].plot(np.nanmean(avg, axis=1), color=color)

            # plt the histograms for all the animals
            bins = np.linspace(-np.pi, np.pi, 17)
            fig, axs = plt.subplots(1,3, figsize=(9,3), sharex=True, sharey=True)
            fig.tight_layout()
            for i, n in enumerate([all_eb_phase, all_fb_upper_phase, all_fb_lower_phase]):
                arr = tolerant_mean(n)
                data = arr.data[-num_pts:,:]
                mask = arr.mask[-num_pts:,:]
                data = data[mask==False]
                counts, _ =  np.histogram(data, bins=bins, density=True)
                counts = counts/sum(counts)
                if border_edge=='left':
                    grouped_phases_left[i].append(counts)
                elif border_edge=='right':
                    grouped_phases_right[i].append(counts)
                sns.histplot(data, ax=axs_hist[i],stat='probability', color=color, bins=bins, alpha=0.1, element='step', fill=False)
                sns.histplot(data, ax=axs[i],stat='probability', color=color, bins=bins, alpha=1, element='step', fill=False)
                axs[i].set_xticks([-np.pi/2, np.pi/2])
                axs[i].set_xticklabels(['-90', '+90'])
            fig.savefig(os.path.join(figfol, str(num_pts)+'_histogram.pdf'))

# plot the average histograms
for i, n in enumerate(grouped_phases_left):
    sns.histplot(proxy_data(bins, n), ax=axs_hist[i],stat='probability', color='blue', bins=bins, alpha=1, element='step', fill=False)
    axs_hist[i].set_xticks([-np.pi/2, np.pi/2])
    axs_hist[i].set_xticklabels(['-90', '+90'])
for i, n in enumerate(grouped_phases_right):
    sns.histplot(proxy_data(bins, n), ax=axs_hist[i],stat='probability', color='red', bins=bins, alpha=1, element='step', fill=False)
    axs_hist[i].set_xticks([-np.pi/2, np.pi/2])
    axs_hist[i].set_xticklabels(['-90', '+90'])
fig_hist.savefig(os.path.join(datafol, 'figures', str(num_pts)+'_histogram.pdf'))

fig_h,axs_h = plt.subplots(2,3)
fig_h.tight_layout()
fig_s,axs_s = plt.subplots(2,3)
fig_s.tight_layout()
for i in [0,1]:
    if i==0:
        group = grouped_heatmaps_right
        vmin=0.3
        vmax=0.7
    else:
        group = grouped_heatmaps_left
        vmin=0.2
        vmax=0.7
    for j in [0,1,2]:
        avg = group[j]
        ix1 = len(avg)-num_pts
        avg = project_heatmaps(avg)
        sns.heatmap(avg,ax=axs_h[i,j], vmin=vmin, vmax=vmax)
        axs_s[i,j].plot(np.nanmean(avg, axis=1))
    #return grouped_heatmaps_right, grouped_heatmaps_left, stop1



# ix1 = avg.shape[-1]-num_pts
# axs_proj[i].plot(np.mean(avg[:, ix1:], axis=1), color=color)

# counts, _ =  np.histogram(np.concatenate(n), bins=bins, density=True)
#
# data = arr.data[-num_pts:,:]
# mask = arr.mask[-num_pts:,:]
# data = data[mask==False]
# counts, _ =  np.histogram(data, bins=bins, density=True)
# counts = counts/sum(counts)
# np.sum(grouped_phases_right[1], axis=0)
# plt.plot(np.sum(grouped_phases_right[1], axis=0))
#
#
# pd = proxy_data(bins, grouped_phases_right[1])
#
# sns.histplot(pd,bins=bins)



# %%
def final_segment_analysis():
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d['folder'] == '20220517_hdc_split_60d05_sytgcamp7f_Fly1-001':
                print(d)
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                figfol = ex.figure_folder
                border_edge = d['border_edge']

                # figure for mean phases
                fig_all=plt.figure()
                ax1_all = fig_all.add_subplot(3,1,1, projection='polar')
                ax2_all = fig_all.add_subplot(3,1,2, projection='polar')
                ax3_all = fig_all.add_subplot(3,1,3, projection='polar')
                ax1_all.set_theta_offset(np.pi/2)
                ax2_all.set_theta_offset(np.pi/2)
                ax3_all.set_theta_offset(np.pi/2)

                # variables for average
                all_vecs = {
                'eb': [],
                'eb_comp': [],
                'fb_upper': [],
                'fb_upper_comp': [],
                'fb_lower': [],
                'fb_lower_comp':[]
                }

                # find replay/clean_air trial
                fly = d['fly']
                df_fly = df.loc[df.fly==fly]
                d_comparison = df_fly.loc[df_fly.experiment==comparison_type].squeeze().to_dict()
                ex_comparison = hdc_example(d_comparison, datafol)

                # fetch info for edge-tracking and comparison trial
                all_info_et = ex.fetch_all_info_eb_fb()
                all_info_comp = ex_comparison.fetch_all_info_eb_fb()

                # fetch segment library for comparison trial
                segs,ixs = ex_comparison.make_segment_library()

                do = all_info_et['outside_segs']
                # iterate through the edge tracking outside segments
                for key in list(do.keys())[1:]:
                    temp = do[key]
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()
                    if ex.experiment == '45':
                        x,y = fn.coordinate_rotation(x, y, np.pi/4)
                    _,pathlenth = fn.path_length(x,y)
                    if np.abs(x[-1]-x[0])<1 and pathlenth>10: # returns to the edge and goes more than 10mm
                        if ex.experiment == '45':
                            x,y = fn.coordinate_rotation(x, y, -np.pi/4)

                        # get the RDP simplification of the selected outside trajectory
                        simplified = fn.rdp_simp(x,y, epsilon=1)
                        simp_x = simplified[:,0]
                        simp_y = simplified[:,1]

                        # first segment end index (outbound) in segement
                        ix1 = np.where(y==simplified[:,1][1])
                        ix1 = ix1[0][0]

                        # last segment index start (inbound) in segment
                        ix2 = np.where(y==simplified[:,1][-2])
                        ix2 = ix2[0][0]

                        # center the return segment, find closest match
                        xy = simplified[-1]-simplified[-2]
                        info_comp = get_segment_info_comp(xy, all_info_comp, segs, ixs)
                        info = get_segment_info(all_info_et, simplified[-2], simplified[-1])


                        if True:
                            # plot the outside trajectory, first/last RDP segments
                            fig,axs = plt.subplots(4,3, figsize = (8,8))
                            axs[0,0].plot(x,y)
                            axs[0,0].plot(x[0],y[0],'o', color='purple')
                            axs[0,0].plot(x[-1],y[-1],'o', color='green')
                            axs[0,0].plot(simp_x[:2], simp_y[:2], 'k')
                            axs[0,0].plot(simp_x[-2:], simp_y[-2:], 'k')
                            #axs[0,0].plot(x[:ix1+1],y[:ix1+1], 'r')
                            #axs[0,0].plot(x[ix2:], y[ix2:], 'r')
                            axs[0,0].axis('equal')
                            axs[0,0].set_title('outside_trajectory')

                            # plot the inbound RDP segment
                            axs[0,1].plot(info['x_seg_0'], info['y_seg_0'])
                            axs[0,1].plot(info_comp['x_comp_seg_0'], info_comp['y_comp_seg_0'])
                            axs[0,1].plot(simp_x[-2:]-simp_x[-2:][0], simp_y[-2:]-simp_y[-2:][0], 'k')
                            axs[0,1].plot(xy[0],xy[1],'o', color='green')
                            axs[0,1].axis('equal')
                            axs[0,1].set_title('inbound segment')

                            # plot net motion
                            fig.delaxes(axs[0,2])
                            #axs[0,2].plot(temp.net_motion)

                            # plot the heatmaps for the segment
                            sns.heatmap(info['fit_wedges_eb'], vmin=0, vmax=1, ax = axs[1,0], cbar=None)
                            sns.heatmap(info['fit_wedges_fb_upper'], vmin=0, vmax=2, ax = axs[2,0], cbar=None)
                            sns.heatmap(info['fit_wedges_fb_lower'], vmin=0, vmax=2, ax = axs[3,0], cbar=None)
                            axs[1,0].set_title('EPG signal')
                            axs[1,1].set_title('EPG (replay) signal')
                            axs[1,2].set_title('EPG phases')
                            axs[2,0].set_title('HDC upper signal')
                            axs[2,1].set_title('HDC upper (air) signal')
                            axs[2,2].set_title('HDC upper phases')
                            axs[3,0].set_title('HDC lower signal')
                            axs[3,1].set_title('HDC lower (air) signal')
                            axs[3,2].set_title('HDC lower phases')

                            # plot the heatmaps for the replay segment
                            sns.heatmap(info_comp['fit_wedges_eb'], vmin=0, vmax=1, ax = axs[1,1], cbar=None)
                            sns.heatmap(info_comp['fit_wedges_fb_upper'], vmin=0, vmax=2, ax = axs[2,1], cbar=None)
                            sns.heatmap(info_comp['fit_wedges_fb_lower'], vmin=0, vmax=2, ax = axs[3,1], cbar=None)

                            for j in [0,1]:
                                for i in [1,2,3]:
                                    axs[i,j].get_xaxis().set_visible(False)
                                    axs[i,j].set_yticks([-0.5,8,15.5])
                                    axs[i,j].set_yticklabels(['180', '0', '-180'])

                            # plot the average bump
                            axs[1,2].plot(info['eb_mean'])
                            axs[2,2].plot(info['fb_upper_mean'])
                            axs[3,2].plot(info['fb_lower_mean'])
                            axs[1,2].plot(info_comp['eb_mean'])
                            axs[2,2].plot(info_comp['fb_upper_mean'])
                            axs[3,2].plot(info_comp['fb_lower_mean'])
                            for i in [1,2,3]:
                                axs[i,2].set_ylim(-0.5,1.5)
                                axs[i,2].set_xticks([0,8,15])
                                axs[i,2].set_xticklabels(['-180', '0', '180'])
                            fig.tight_layout()
                            fig.savefig(os.path.join(figfol, str(key)+'_heatmap_bout.pdf'))

                            #####
                            #####
                            fig=plt.figure(figsize = (3,8))
                            # just plot cosine phases
                            ax1 = fig.add_subplot(3,1,1, projection='polar')
                            ax2 = fig.add_subplot(3,1,2, projection='polar')
                            ax3 = fig.add_subplot(3,1,3, projection='polar')
                            ax1.set_title('EPG phase')
                            ax2.set_title('upper layer phase')
                            ax3.set_title('lower layer phase')
                            ax1.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            ax2.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            ax3.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            ax1.set_title('EPG angles')
                            ax2.set_title('upper layer angles')
                            ax3.set_title('lower layer angles')
                            # cosine phases
                            # ax4 = fig.add_subplot(3,2,2, projection='polar')
                            # ax5 = fig.add_subplot(3,2,4, projection='polar')
                            # ax6 = fig.add_subplot(3,2,6, projection='polar')
                            # ax4.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            # ax5.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            # ax6.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])
                            # ax4.set_title('EPG phase')
                            # ax5.set_title('upper layer phase')
                            # ax6.set_title('lower layer phase')

                            # ang = -info['eb_phase_mean']
                            # ax1.plot([ang,ang],[0,1])
                            # ang = -info_comp['eb_phase_mean']
                            # ax1.plot([ang,ang],[0,1])
                            # ang = -info['fb_upper_phase_mean']
                            # ax2.plot([ang,ang],[0,1])
                            # ang = -info_comp['fb_upper_phase_mean']
                            # ax2.plot([ang,ang],[0,1])
                            # ang = -info['fb_lower_phase_mean']
                            # ax3.plot([ang,ang],[0,1])
                            # ang = -info_comp['fb_lower_phase_mean']
                            # ax3.plot([ang,ang],[0,1])
                            ang_mag = info['ang_mag_eb']
                            ax1.plot([ang_mag[0],ang_mag[0]],[0,1])
                            ang_mag = info_comp['ang_mag_eb']
                            ax1.plot([ang_mag[0],ang_mag[0]],[0,1])
                            ang_mag = info['ang_mag_fb_upper']
                            ax2.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]])
                            ang_mag = info_comp['ang_mag_fb_upper']
                            ax2.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]])
                            ang_mag = info['ang_mag_fb_lower']
                            ax3.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]])
                            ang_mag = info_comp['ang_mag_fb_lower']
                            ax3.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]])

                            ax1.set_theta_offset(np.pi/2)
                            ax2.set_theta_offset(np.pi/2)
                            ax3.set_theta_offset(np.pi/2)
                            # ax4.set_theta_offset(np.pi/2)
                            # ax5.set_theta_offset(np.pi/2)
                            # ax6.set_theta_offset(np.pi/2)
                            fig.savefig(os.path.join(figfol, str(key)+'_vectors_bout.pdf'))
                            fig.tight_layout()

                        ang_mag = info['ang_mag_eb']
                        all_vecs['eb'].append(info['ang_mag_eb'])
                        ax1_all.plot([ang_mag[0],ang_mag[0]],[0,1], color=sns.color_palette()[0], alpha=0.2)

                        ang_mag = info_comp['ang_mag_eb']
                        all_vecs['eb_comp'].append(info_comp['ang_mag_eb'])
                        ax1_all.plot([ang_mag[0],ang_mag[0]],[0,1], color=sns.color_palette()[1], alpha=0.2)

                        ang_mag = info['ang_mag_fb_upper']
                        all_vecs['fb_upper'].append(info['ang_mag_fb_upper'])
                        ax2_all.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]], color=sns.color_palette()[0], alpha=0.2)

                        ang_mag = info_comp['ang_mag_fb_upper']
                        all_vecs['fb_upper_comp'].append(info_comp['ang_mag_fb_upper'])
                        ax2_all.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]], color=sns.color_palette()[1], alpha=0.2)

                        ang_mag = info['ang_mag_fb_lower']
                        all_vecs['fb_lower'].append(info['ang_mag_fb_lower'])
                        ax3_all.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]], color=sns.color_palette()[0], alpha=0.2)

                        ang_mag = info_comp['ang_mag_fb_lower']
                        all_vecs['fb_lower_comp'].append(info_comp['ang_mag_fb_lower'])
                        ax3_all.plot([ang_mag[0],ang_mag[0]],[0,ang_mag[1]], color=sns.color_palette()[1], alpha=0.2)

                ang, mag = get_average_vector(all_vecs['eb'])
                ax1_all.plot([ang,ang],[0,1], color=sns.color_palette()[0])

                ang, mag = get_average_vector(all_vecs['eb_comp'])
                ax1_all.plot([ang,ang],[0,1], color=sns.color_palette()[1])

                ang, mag = get_average_vector(all_vecs['fb_upper'])
                ax2_all.plot([ang,ang],[0,mag], color=sns.color_palette()[0])

                ang, mag = get_average_vector(all_vecs['fb_upper_comp'])
                ax2_all.plot([ang,ang],[0,mag], color=sns.color_palette()[1])

                ang, mag = get_average_vector(all_vecs['fb_lower'])
                ax3_all.plot([ang,ang],[0,mag], color=sns.color_palette()[0])

                ang, mag = get_average_vector(all_vecs['fb_lower_comp'])
                ax3_all.plot([ang,ang],[0,mag], color=sns.color_palette()[1])

                fig_all.savefig(os.path.join(figfol, 'summary_phases.pdf'))




# %%


# %%
info = all_info_et
ft2 = info['ft2']
offset_eb_phase = info['offset_eb_phase']
fit_wedges_eb = info['fit_wedges_eb']
offset = info['offset']
x = ft2.ft_posx.to_numpy()
y = ft2.ft_posy.to_numpy()
phase = ft2.ft_heading.to_numpy()

fig, axs = plt.subplots(1,1)
axs.plot(phase, '.')
ix = np.where(phase>0)[0]
axs.plot(ix, phase[ix], '.')
axs.plot(offset_eb_phase)

fig, axs = plt.subplots(1,1)
axs.plot(x,y)
axs.axis('equal')

fig, axs = plt.subplots(1,1)
sns.heatmap(fit_wedges_eb, ax=axs)

fig, axs = plt.subplots(1,1)
sns.heatmap(np.rot90(fit_wedges_eb, k=1, axes=(0,1)), ax=axs)
axr = axs.twinx()
axr.plot(phase, '.')


# axr = axs.twinx()
# axr.plot(fn.wrap(offset+offset_eb_phase), '.')
# axr.set_ylim(-np.pi, np.pi)





# ft2 = all_info_et['ft2']
# x_all = ft2.ft_posx.to_numpy()
# y_all = ft2.ft_posy.to_numpy()
# fig,axs = plt.subplots(1,1)
# axs.plot(x_all,y_all)
# axs.plot(simplified[:,0], simplified[:,1])
# axs.plot(x, y)
# np.where(x==simplified[-2][0])[0][0]
# simplified[-2]

# %%
hdc = hdeltac(celltype='hdeltac')
df = hdc.sheet
fig, axs = plt.subplots(4,1, sharex=True,sharey=True)
for i in np.arange(len(df)):
    d = df.iloc[i].to_dict()
    if d['analyze'] == True:
        if d['folder'] == '20220629_HDC_split_sytjGCaMP7f_Fly1_strip_grid-002':
            print(d)
            border_edge = d['border_edge']
            datafol = '/Volumes/LACIE/Andy/hdeltac/'
            ex = hdc_example(d,datafol)
            fb = ex.fb
            eb = ex.eb
            fb.load_processed()
            ft2 = fb.ft2

            # calculate eb phases
            wedges_eb = eb.get_layer_wedges()
            phase = eb.get_centroidphase(wedges_eb)
            offset = eb.continuous_offset(wedges_eb)
            offset_eb_phase = fn.wrap(phase-offset)
            wedges_eb = np.rot90(wedges_eb, k=1, axes=(0,1))

            # #plot eb information
            # axsr = axs[0].twinx()
            # axsr.set_ylim(-np.pi,np.pi)
            # sns.heatmap(wedges_eb,ax=axs[0],cmap='Blues', cbar='False')
            # axsr.plot(ft2.ft_heading,'.', alpha=0.1)
            # axsr.plot(phase, '.', alpha=0.1)
            # axsr.plot(offset_eb_phase, '.',alpha=0.1)


            # get fb information
            wedges_fb = fb.get_layer_wedges('upper')
            phase_fb = fb.get_centroidphase(wedges_fb)
            offset_fb_phase = fn.wrap(phase_fb-offset)
            rotated_wedges_fb = np.rot90(wedges_fb, k=1, axes=(0,1))

            # # plot the fb
            # axsr = axs[1].twinx()
            # axsr.set_ylim(-np.pi,np.pi)
            # sns.heatmap(rotated_wedges_fb,ax=axs[1],cmap='Blues', cbar='False')
            # axsr.plot(ft2.ft_heading,'.', alpha=0.1)
            # axsr.plot(phase_fb, '.', alpha=0.1)
            # axsr.plot(offset_fb_phase, '.',alpha=0.1)

            # fit the fb to sinusoids
            # fit_wedges, all_params = fb.wedges_to_cos(wedges_fb)
            # rotated_fit_wedges = np.rot90(fit_wedges, k=1, axes=(0,1))
            # axsr = axs[2].twinx()
            # axsr.set_ylim(-np.pi,np.pi)
            # sns.heatmap(rotated_fit_wedges,ax=axs[2],cmap='Blues', cbar='False')
            # phase = fn.centroid_weightedring(fit_wedges)
            # offset_phase = fn.wrap(phase-offset)
            # axsr.plot(phase,'.', alpha=0.1, color='orange')
            # axsr.plot(offset_phase,'.', alpha=0.1, color='green')
            #
            fit_wedges, all_params = fb.wedges_to_cos(wedges_fb, phase_offset=offset)
            # rotated_fit_wedges = np.rot90(fit_wedges, k=1, axes=(0,1))
            # sns.heatmap(rotated_fit_wedges,ax=axs[3],cmap='Blues', cbar='False')

            ft2['offset_fb_phase'] = offset_fb_phase
            ft2['offset_eb_phase'] = offset_eb_phase
            d,di,do = fn.inside_outside(ft2)

            outbound_fb = []
            inbound_fb = []
            outbound_phase_eb = []
            inbound_phase_eb = []
            outbound_phase_fb = []
            inbound_phase_fb = []
            for key in list(do.keys())[1:]:
                temp = do[key]

                x = temp.ft_posx.to_numpy()
                y = temp.ft_posy.to_numpy()
                if ex.experiment == '45':
                    x,y = fn.coordinate_rotation(x, y, np.pi/4)

                _,pathlenth = fn.path_length(x,y)
                if np.abs(x[-1]-x[0])<1 and pathlenth>10: # returns to the edge
                    if ex.experiment == '45':
                        x,y = fn.coordinate_rotation(x, y, -np.pi/4)
                    simplified = fn.rdp_simp(x,y, epsilon=1)
                    temp_wedges = fit_wedges[temp.index.to_list()]
                    phase_fb = temp.offset_fb_phase
                    phase_eb = temp.offset_eb_phase

                    # first segment
                    ix1 = np.where(y==simplified[:,1][1])
                    ix1 = ix1[0][0]

                    # last segment
                    ix2 = np.where(y==simplified[:,1][-2])
                    ix2 = ix2[0][0]

                    # eb phases for inbound/outbound segment
                    op_eb = fn.circmean(phase_eb[0:ix1])
                    ip_eb = fn.circmean(phase_eb[ix2:])
                    outbound_phase_eb.append(op_eb)
                    inbound_phase_eb.append(ip_eb)

                    # eb phases for inbound/outbound segment
                    op_fb = fn.circmean(phase_fb[0:ix1])
                    ip_fb = fn.circmean(phase_fb[ix2:])
                    outbound_phase_fb.append(op_fb)
                    inbound_phase_fb.append(ip_fb)

                    # inbound bump
                    ib = temp_wedges[ix2:,:]
                    ib = np.mean(ib,axis=0)
                    fit_ib, params_ib = fn.fit_cos(ib)
                    ang = fn.correct_fit_cos_phase(params_ib[2])
                    mag = params_ib[0]
                    ang_mag = [ang, mag]
                    inbound_fb.append(ang_mag)

                    # outbound bump
                    ob = temp_wedges[0:ix1,:]
                    ob = np.mean(ob,axis=0)
                    fit_ob, params_ob = fn.fit_cos(ob)
                    ang = fn.correct_fit_cos_phase(params_ob[2])
                    mag = params_ib[0]
                    ang_mag = [ang, mag]
                    outbound_fb.append(ang_mag)


            fig = plt.figure()
            #fig.suptitle(self.folder)
            axs1 = fig.add_subplot(131, projection='polar')
            axs2 = fig.add_subplot(132, projection='polar')
            axs3 = fig.add_subplot(133, projection='polar')
            d = {}
            ang_avg, r_avg = pl.circular_hist(axs1, np.array(outbound_phase_eb), color=sns.color_palette()[0], offset=np.pi/2, label='outbound')
            d['outbound_eb'] = [ang_avg, r_avg]
            ang_avg, r_avg = pl.circular_hist(axs1, np.array(inbound_phase_eb), color=sns.color_palette()[1], offset=np.pi/2, label='inbound')
            d['inbound_eb'] = [ang_avg, r_avg]
            ang_avg, r_avg = pl.circular_hist(axs2, np.array(outbound_phase_fb), color=sns.color_palette()[0], offset=np.pi/2, label='outbound')
            d['outbound_eb'] = [ang_avg, r_avg]
            ang_avg, r_avg = pl.circular_hist(axs2, np.array(inbound_phase_fb), color=sns.color_palette()[1], offset=np.pi/2, label='inbound')
            d['inbound_eb'] = [ang_avg, r_avg]

            for am in outbound_fb:
                axs3.plot([am[0], am[0]], [0,am[1]], color=sns.color_palette()[0])
            for am in inbound_fb:
                axs3.plot([am[0], am[0]], [0,am[1]], color=sns.color_palette()[1])
            axs3.set_theta_offset(np.pi/2)

            # axs1.legend(loc='upper left', bbox_to_anchor=(1,1))
            # axs2.legend(loc='upper left', bbox_to_anchor=(1,1))
            #fig.tight_layout()


# %%



# %%




# %matplotlib
#
# hdc = hdeltac(celltype='hdeltac')
# df = hdc.sheet
# for i in np.arange(len(df)):
#     d = df.iloc[i].to_dict()
#     if d['analyze'] == True:
#         if d['DP_analysis'] == True:
#             print(d)
#             layer='upper'
#             border_edge = d['border_edge']
#             datafol = '/Volumes/LACIE/Andy/hdeltac/'
#             ex = hdc_example(d,datafol)
#             ex.fb.load_processed()
#             ex.plot_RDP_segment_phase()
#             ft2_et = ex.fb.ft2
#
#             # find replay trial
#             fly = d['fly']
#             df_fly = df.loc[df.fly==fly]
#             d_replay = df_fly.loc[df_fly.experiment=='replay'].squeeze().to_dict()
#             ex_replay = hdc_example(d_replay, datafol)
#
#             # replay trial data
#             eb = ex_replay.eb
#             fb = ex_replay.fb
#             fb.load_processed()
#             ft2_replay = fb.ft2
#
#             # eb information
#             wedges = eb.get_layer_wedges()
#             phase = eb.get_centroidphase(wedges)
#             #offset = fn.circmean(phase-heading)
#             offset = eb.continuous_offset(wedges)
#             offset_eb_phase = fn.wrap(phase-offset)
#
#             # fb information
#             wedges = fb.get_layer_wedges(layer)
#             phase = fb.get_centroidphase(wedges)
#             offset_fb_phase = fn.wrap(phase-offset)
#
#             # split up replay trajectory into inside and outside
#             ft2_replay['offset_fb_phase'] = offset_fb_phase
#             ft2_replay['offset_eb_phase'] = offset_eb_phase
#             d,di,do = fn.inside_outside(ft2_replay)
#
#             # create a library of outside DP line segments for replay experiment
#             segs = []
#             ixs = []
#             x_all = ft2_replay.ft_posx.to_numpy()
#             for key in list(do.keys()):
#                 temp = do[key]
#                 x = temp.ft_posx.to_numpy()
#                 y = temp.ft_posy.to_numpy()
#                 simplified = fn.rdp_simp(x,y, epsilon=1)
#                 segs.append(np.diff(simplified,axis=0))
#                 ix = np.where(np.in1d(x_all, simplified[:,0]))[0]
#                 ix_middle = ix[1:-1]
#                 callback_ix = np.zeros(2*len(ix_middle)).astype(int)
#                 callback_ix[0::2]=ix_middle
#                 callback_ix[1::2]=ix_middle
#                 callback_ix=np.concatenate([[ix[0]], callback_ix, [ix[-1]]])
#                 callback_ix = np.reshape(callback_ix,(len(ix)-1,2))
#                 ixs.append(callback_ix)
#             segs = np.concatenate(segs)
#             ixs = np.concatenate(ixs)
#
#             # need segs, ixs and ft2_replay for next analysis
#             # now find the return trajectories for the edge-tracking trial
#
#             def best_dp_match(segs, xy):
#                 """
#                 function for finding the best DP line segment to match a line segment
#                 which starts from the origin and goes to xy=[x,y]
#                 """
#                 distances = []
#                 for seg in segs:
#                     delta = seg-xy
#                     delta = delta**2
#                     dist = np.sqrt(np.sum(delta))
#                     distances.append(dist)
#                 match_ix = np.argmin(np.array(distances))
#                 return match_ix
#
#             d,di,do = fn.inside_outside(ft2_et)
#             outbound_phase_fb = []
#             inbound_phase_fb = []
#             outbound_phase_eb = []
#             inbound_phase_eb = []
#             for key in list(do.keys())[1:]:
#                 temp = do[key]
#                 x = temp.ft_posx.to_numpy()
#                 y = temp.ft_posy.to_numpy()
#                 if ex.experiment == '45':
#                     x,y = fn.coordinate_rotation(x, y, np.pi/4)
#
#                 _,pathlenth = fn.path_length(x,y)
#                 if np.abs(x[-1]-x[0])<1 and pathlenth>10: # returns to the edge
#                     success_df = temp
#                     if ex.experiment == '45':
#                         x,y = fn.coordinate_rotation(x, y, -np.pi/4)
#                     simplified = fn.rdp_simp(x,y, epsilon=1)
#
#                     # first segment
#                     ix1 = np.where(y==simplified[:,1][1])
#                     ix1 = ix1[0][0]
#                     # last segment
#                     ix2 = np.where(y==simplified[:,1][-2])
#                     ix2 = ix2[0][0]
#                     xy = simplified[-1]-simplified[-2]
#                     match_ix = best_dp_match(segs,xy)
#                     return_ix = ixs[match_ix]
#
#                     #remove segment if it's been used
#                     segs = np.delete(segs, match_ix, axis=0)
#                     ixs = np.delete(ixs, match_ix, axis=0)
#
#                     ip_fb = ft2_replay.iloc[return_ix[0]:return_ix[1]].offset_fb_phase.to_numpy()
#                     ip_fb = fn.circmean(ip_fb)
#                     ip_eb = ft2_replay.iloc[return_ix[0]:return_ix[1]].offset_eb_phase.to_numpy()
#                     ip_eb = fn.circmean(ip_eb)
#                     # outbound_phase_fb.append(op_fb)
#                     inbound_phase_fb.append(ip_fb)
#                     # outbound_phase_eb.append(op_eb)
#                     inbound_phase_eb.append(ip_eb)
#
#             fig = plt.figure()
#             axs1 = fig.add_subplot(121, projection='polar')
#             axs2 = fig.add_subplot(122, projection='polar')
#             ang_avg, r_avg = pl.circular_hist(axs1, np.array(inbound_phase_eb), color=sns.color_palette()[1], offset=np.pi/2, label='inbound')
#             ang_avg, r_avg = pl.circular_hist(axs2, np.array(inbound_phase_fb), color=sns.color_palette()[1], offset=np.pi/2, label='inbound')
#










# %%
def plot_upvel_xvel_v_bump_amp(analysis_type = 'DP_analysis'):
    figfol = '/Volumes/LACIE/Andy/hdeltac/figures'
    edge_tracking = []
    clean_air = []
    plt.style.use('default')
    hdc = hdeltac(celltype='hdeltac')
    df = hdc.sheet
    # plot the phase difference between the layers
    #fig, axs = plt.subplots(1,1)
    bump_amp_all = {
    'upper_air': [],
    'lower_air': [],
    'upper_odor': [],
    'lower_odor': []
    }
    upwind_velocity_all = {
    'air': [],
    'odor': []
    }
    crosswind_velocity_all = {
    'air': [],
    'odor': []
    }



    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            if d[analysis_type] == True:
                datafol = '/Volumes/LACIE/Andy/hdeltac/'
                ex = hdc_example(d,datafol)
                ex.fb.load_processed()
                ft2 = ex.fb.ft2
                # upwind and crosswind speed/velocity
                del_t = np.mean(np.diff(ft2.seconds))
                y = ft2.ft_posy.to_numpy()
                x = ft2.ft_posx.to_numpy()
                upwind_velocity = np.gradient(y)/del_t
                crosswind_velocity = np.abs(np.gradient(x))/del_t
                upper_wedges = ex.fb.get_layer_wedges('upper')
                lower_wedges = ex.fb.get_layer_wedges('lower')

                stops, df_stop = fn.find_stops(ft2)
                moving_ix = df_stop[df_stop.stop==1].index.to_list()
                #moving_ix = np.argwhere(np.abs(upwind_velocity)>1)[:,0]
                ix_in = ft2[ft2.instrip==1.0].index.to_numpy()
                ix_out = ft2[ft2.instrip==0.0].index.to_numpy()
                if analysis_type == 'clean_air_analysis':
                    start_ix = np.array([0])
                else:
                    start_ix = ix_in[0]
                ix = np.argmin(np.abs(moving_ix-start_ix)) # index in ix_out where the pre odor time ends
                moving_ix = moving_ix[ix:] # index where fly is moving
                if analysis_type == 'clean_air_analysis':
                    air_moving_ix = np.intersect1d(moving_ix,ix_out)
                elif analysis_type == 'DP_analysis':
                    air_moving_ix = np.intersect1d(moving_ix,ix_out)
                    odor_moving_ix = np.intersect1d(moving_ix,ix_in)

                up_vel_air = upwind_velocity[air_moving_ix]
                upwind_velocity_all['air'].append(up_vel_air)
                cvel_air = crosswind_velocity[air_moving_ix]
                crosswind_velocity_all['air'].append(cvel_air)
                bump_amp = np.mean(upper_wedges[air_moving_ix,:],axis=1)
                bump_amp_all['upper_air'].append(bump_amp)
                bump_amp = np.mean(lower_wedges[air_moving_ix,:],axis=1)
                bump_amp_all['lower_air'].append(bump_amp)

                if analysis_type == 'DP_analysis':
                    up_vel_odor = upwind_velocity[odor_moving_ix]
                    upwind_velocity_all['odor'].append(up_vel_odor)
                    cvel_odor = crosswind_velocity[odor_moving_ix]
                    crosswind_velocity_all['odor'].append(cvel_odor)
                    bump_amp = np.mean(upper_wedges[odor_moving_ix,:],axis=1)
                    bump_amp_all['upper_odor'].append(bump_amp)
                    bump_amp = np.mean(lower_wedges[odor_moving_ix,:],axis=1)
                    bump_amp_all['lower_odor'].append(bump_amp)


    df_air = pd.DataFrame({'upwind_velocity':np.concatenate(upwind_velocity_all['air']), 'crosswind_velocity':np.concatenate(crosswind_velocity_all['air']), 'bump_amp_upper': np.concatenate(bump_amp_all['upper_air']), 'bump_amp_lower': np.concatenate(bump_amp_all['lower_air'])})
    if analysis_type == 'DP_analysis':
        df_odor = pd.DataFrame({'upwind_velocity':np.concatenate(upwind_velocity_all['odor']), 'crosswind_velocity':np.concatenate(crosswind_velocity_all['odor']), 'bump_amp_upper': np.concatenate(bump_amp_all['upper_odor']), 'bump_amp_lower': np.concatenate(bump_amp_all['lower_odor'])})
    #df = df.groupby(df.index // 10).mean()
    fig,axs = plt.subplots(2,2, figsize=(6,6))
    sns.regplot(df_air.upwind_velocity, df_air.bump_amp_upper, ax=axs[0,0], scatter_kws={'alpha':0.05, 'edgecolors':'none'})
    sns.regplot(df_air.crosswind_velocity, df_air.bump_amp_upper, ax=axs[0,1], scatter_kws={'alpha':0.05, 'edgecolors':'none'})
    sns.regplot(df_air.upwind_velocity, df_air.bump_amp_lower, ax=axs[1,0], scatter_kws={'alpha':0.05, 'edgecolors':'none'})
    sns.regplot(df_air.crosswind_velocity, df_air.bump_amp_lower, ax=axs[1,1], scatter_kws={'alpha':0.05, 'edgecolors':'none'})
    if analysis_type == 'DP_analysis':
        sns.regplot(df_odor.upwind_velocity, df_odor.bump_amp_upper, ax=axs[0,0], scatter_kws={'alpha':0.05, 'edgecolors':'none'})
        sns.regplot(df_odor.crosswind_velocity, df_odor.bump_amp_upper, ax=axs[0,1], scatter_kws={'alpha':0.05, 'edgecolors':'none'})
        sns.regplot(df_odor.upwind_velocity, df_odor.bump_amp_lower, ax=axs[1,0], scatter_kws={'alpha':0.05, 'edgecolors':'none'})
        sns.regplot(df_odor.crosswind_velocity, df_odor.bump_amp_lower, ax=axs[1,1], scatter_kws={'alpha':0.05, 'edgecolors':'none'})
    fig.tight_layout()
    # slope, intercept, r_value, p_value, std_err = stats.linregress(np.concatenate(upwind_velocity_all), np.concatenate(bump_amp_all))
    # print('for upwind velocity, p=',p_value,' r-squared=', r_value**2)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(np.concatenate(crosswind_velocity_all), np.concatenate(bump_amp_all))
    # print('for crosswind velocity, p=',p_value,' r-squared=', r_value**2)
    axs[0,0].set_xlabel('upwind velocity')
    axs[0,0].set_ylabel('HDC upper layer bump amplitude')
    axs[0,1].set_xlabel('crosswind speed')
    axs[0,1].set_ylabel('HDC upper layer bump amplitude')
    axs[1,0].set_xlabel('upwind velocity')
    axs[1,0].set_ylabel('HDC lower layer bump amplitude')
    axs[1,1].set_xlabel('crosswind speed')
    axs[1,1].set_ylabel('HDC lower layer bump amplitude')

    fig.savefig(os.path.join(figfol, 'bump_amplitude_v_upwind_crosswind_'+analysis_type+'_.pdf'))









# %%
%matplotlib
hdc = hdeltac(celltype='hdeltac')
df = hdc.sheet
for i in np.arange(len(df)):
    d = df.iloc[i].to_dict()
    if d['analyze'] == True:
        if d['DP_analysis'] == True:
            print(d)
            datafol = '/Volumes/LACIE/Andy/hdeltac/'
            ex = hdc_example(d,datafol)
            ex.plot_allo_directions_x_y()
            ex.plot_all_heatmaps()








# %%
eb = ex.eb
fb = ex.fb
wedges = eb.get_layer_wedges()
phase = eb.get_centroidphase(wedges)
fig, axs = plt.subplots(1,1)
sns.heatmap(fb.continuous_to_glom(fb.cancel_phase(wedges), nglom=16), ax=axs)
ft2 = eb.ft2
heading = ft2.ft_heading

fb_wedges = fb.get_layer_wedges('lower')

fig, axs = plt.subplots(1,1)
sns.heatmap(fb.continuous_to_glom(fb.cancel_phase(fb_wedges), nglom=16), ax=axs)
fig, axs = plt.subplots(1,1)
phase_diff2 = fn.unwrap(heading)-fn.unwrap(phase)
phase_diff2 = pd.Series(phase_diff2)
phase_diff2 = fn.wrap(phase_diff2.rolling(50, min_periods=1).mean())

axs.plot(phase_diff2, '.')

ex.make_movie()

# %%
#ex.make_movie()

#
# from matplotlib.backends.backend_pdf import PdfPages
# with PdfPages(os.path.join(ex.figure_folder, 'figures.pdf')) as pdf:
#     fig, axs = ex.plot_trajectory()
#     fig.suptitle('trajectory')
#     pdf.savefig()  # saves the current figure into a pdf page
#     plt.close()
#
#     fig, axs = ex.plot_FB_heatmap_heading()
#     fig.suptitle('FB heatmaps (no phase offset)')
#     pdf.savefig()
#     plt.close()
#
#     fig, axs = ex.plot_EPG_FB_heading()
#     fig.suptitle('FB EPG phase (with offset)')
#     pdf.savefig()
#     plt.close()
#
#     fig, axs = ex.phase_diff_between_layers()
#     fig.suptitle('phase difference between FB layers')
#     pdf.savefig()
#     plt.close()
#
#     fig, axs = ex.polar_hist()
#     fig.suptitle('bump phase polar histogram. ORANGE = edge-tracking. BLUE = pre edge-tracking')
#     pdf.savefig()
#     plt.close()
#
#     fig = ex.polar_hist_in_out()
#     fig.suptitle('bump phase in and out. ORANGE = in odor. BLUE = out of odor')
#     pdf.savefig()
#     plt.close()
#
#     fig = ex.polar_hist_bout()
#     fig.suptitle('left 2 columns = outside bouts. right 2 columns = inside bouts')
#     pdf.savefig()
#     plt.close()
#
#     fig = ex.plot_allo_directions_x_y()
#     fig.suptitle('allocentric directions. WHITE = upwind. RED = left. BLACK = downwind. BLUE = right.')
#     pdf.savefig()
#     plt.close()





# ex.z_projection(region='fb')
# ex.make_movie()



# %%
plt.style.use('default')
importlib.reload(im)
from matplotlib import cm
%matplotlib
hdb = hdeltac(celltype='hdeltac')
df = hdb.sheet
df
datafol = hdb.datafol
d = df.iloc[17].to_dict()
#d = df.iloc[35].to_dict()
#d = df.iloc[47].to_dict()
#d = df.iloc[48].to_dict()

ex = hdc_example(d, datafol)
phase_diff, offset_eb_phase, offset_fb_phase = ex.plot_phase_diff_x_y()
df = ex.fb.ft2
_,df = fn.find_stops(df)
d, di, do = fn.inside_outside(df)
out_phase_all = []
in_phase_all = []
fig2, axs2 = plt.subplots(1,3)
#axs2.set_ylim(-np.pi, np.pi)
for key in list(do.keys())[1:]:
    temp = do[key]
    time = temp.seconds.iloc[-1]-temp.seconds.iloc[0]
    if time>2: # only look at outside bouts longer than 2 seconds
        fig, axs = plt.subplots(1,2)

        x = do[key].ft_posx.to_numpy()
        y = do[key].ft_posy.to_numpy()

        ix = list(do[key].index) # outside bout index
        max_dist_ix = ix[np.argmax(np.abs(x-x[0]))] # return point index
        ix_moving = df[df.stop==1.0].index.to_list() # moving index
        out_ix = np.arange(ix[0],max_dist_ix) # outward path (exit to return point)
        out_ix = np.intersect1d(out_ix, ix_moving) # only select periods where moving

        in_ix = np.arange(max_dist_ix,ix[-1]) # inward path (return point to re-entry)
        in_ix = np.intersect1d(in_ix, ix_moving) # only select periods where moving

        titles = ['EPG-FB phase', 'EPG phase', 'FB phase']
        for i, ang in enumerate([phase_diff, offset_eb_phase, offset_fb_phase]):
            out_phase = fn.circmean(ang[out_ix])
            in_phase = fn.circmean(ang[in_ix])
            axs2[i].plot([0,1],[out_phase,in_phase])
            axs2[i].set_title(titles[i])
            axs2[i].set_xticks([0,1])
            axs2[i].set_xticklabels(['outbound', 'inbound'], rotation=45)



        hi_out = np.pi/2-out_phase
        dx_out = np.cos(hi_out)
        dy_out = np.sin(hi_out)

        hi_in = np.pi/2-in_phase
        dx_in = np.cos(hi_in)
        dy_in = np.sin(hi_in)

        print(np.rad2deg(out_phase), np.rad2deg(in_phase))
        pl.colorline(axs[0], x,y,phase_diff[ix])
        #axs[0].plot(x[max_dist_ix],y[max_dist_ix], 'o')
        axs[0].plot(x[0],y[0], 'o', color='green')
        axs[0].axis('equal')
        axs[1].arrow(0,0,dx_in,dy_in, color='green', width=0.1)
        axs[1].arrow(0,0,dx_out,dy_out, color='purple', width=0.1)
        axs[1].axis('square')
        axs[1].set_xlim(-2,2)
        axs[1].set_ylim(-2,2)
