# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:57:01 2024

@author: dowel
"""
#%% 
# Aim is to analyse fan shaped body data using a consistent pipeline
# Pipeline is basaed upon Andy's hdeltac class
# Aim is to simplify the code so that it can be universal to imaging data from 
# the fan-shaped body


#%%
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

#%%

class hdeltac:
    def __init__(self, celltype='hdeltac'):
        #d = dr.drive_hookup()

        # load spreadsheet information from folder - changed from Google sheet in Andy's script
        if celltype == 'hdeltac':
            self.datafol = "/Volumes/LACIE/Andy/hdeltac"
        elif celltype == 'hdeltab':
            self.datafol = "Y:\Data\FCI\AndyData\hdb"
            sheetname = 'hdeltab.xlsx'
        #df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        df = pd.read_excel(os.path.join(self.datafol,sheetname))
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
                ex.register_all_images(overwrite=True)
                # try:
                #     ex.register_all_images(overwrite=True)
                # except:
                #     print('Failed')
                #     return ex
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
            
            if string == 'None' or isinstance(string,float):
                setattr(self, key, None)
            
            else:
                print(string)
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
            save_folder = os.path.join("Y:\Data\FCI\AndyData", '\hdb\figures', self.folder)
        elif self.cell_type == 'hdeltac':
            save_folder = os.path.join("Y:\Data\FCI\AndyData", '\hdc\figures', self.folder)
        save_folder = os.path.join('Y:\Data\FCI\AndyData\hdb\\figures', self.folder)
        print(save_folder)
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
            if np.shape(wedges_fb)[1]==0:
                wedges_fb = fb.get_layer_wedges('fb')
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
        try :
            wedges = d['wedges_fb_upper']
            sns.heatmap(wedges, ax=axs[1], cmap=cmap, cbar=False, rasterized=True)
            wedges = d['wedges_fb_lower']
            sns.heatmap(wedges, ax=axs[2], cmap=cmap, cbar=False, rasterized=True)
        except :
            wedges = d['wedges_fb_fbmask']
            sns.heatmap(wedges, ax=axs[1], cmap=cmap, cbar=False, rasterized=True)
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
                if np.shape(wedges)[1]==0:
                    wedges = self.fb.get_layer_wedges('fb')
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
       

