# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:31:21 2024

@author: dowel

Aim of this script is to consolidate  image processing for the central complex into
a single entity. This will take many functions from Andy's code but simplify
it a bit since he has bespoke functions related to his project.

Main components:
    1. Aligning imaging and behavioural data: should be largely the same as Andy's
    2. Extracting fluor from ROIs. Make simpler than Andy's since ROI processing
    is done via the MatLab GUI. Could make a python GUI if there is time...
    
    This should be as generic as possible. Should go from 1-max(mask). These should
    be the column names. Extra processing can be done via bespoke scripts outside of this
    class
    3. Bump processing - these should be taken from Andy's data
"""

#%% Aim
import glob
import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET
import datetime
from analysis_funs.utilities import funcs as fn
import matplotlib.pyplot as plt
import fnmatch
import pickle
from scipy import signal as sg
from scipy import stats
from skimage import io
#%%
class CX:
    def __init__(self,name,roi_names,folderloc):
        self.name = name
        self.folderloc = folderloc
        self.datafol = os.path.join(self.folderloc, 'data') # data folder contains imaging folder, .log, .dat,
        self.regfol = os.path.join(self.folderloc, 'registered') # register folder contains registered .tiffs
        self.processedfol = os.path.join(self.folderloc, 'processed')
        self.roi_names = roi_names # these should be the same names as the tiff files
        for item in os.scandir(self.datafol):
            if item.is_dir():
                for file in os.scandir(item):
                    if file.is_file() and file.name.endswith('.xml'):
                        self.imagefol = os.path.join(self.datafol, item.name)
# %% Executive functions
    def save_preprocessing(self, overwrite=True):
        # Taken directly from Andy
        
        pre_processing_file = os.path.join(self.processedfol, 'preprocessing.h5')
        if os.path.exists(pre_processing_file) or overwrite:
            self.read_image_xml().to_hdf(pre_processing_file, key='timing', mode='w')
            self.read_dat().to_hdf(pre_processing_file, key='ft_raw')
            self.read_log().to_hdf(pre_processing_file, key='voe_log')
            self.merge_voe_ft().to_hdf(pre_processing_file, key='ft')
            if self.read_voltage_recording() is not None:
                self.read_voltage_recording().to_hdf(pre_processing_file, key='spikes')
                
    def process_rois(self):
        # Aim is to process rois, save as csvs. These can then be loaded in save post-processing
        # This is essentially the same as how Andy extacts glomeruli for the PB
        # It can be applied for any region provided the tiffs are made correctly,
        # which they can with the matlab gui
        rois = self.roi_names
        for r in rois:
            print(r)
            # Open mask
            t_path = os.path.join(self.folderloc,'registered',r + 'mask.tiff')
            mask = self.open_mask(t_path)
            
            r_num = np.max(mask[:])
            mrange = np.arange(1,r_num+1,dtype='int')
            # Mask number specifies slice number. No reason to change this
            slice_num = np.shape(mask)[2]# may need to change for single plane imaging
            
            for s in range(slice_num):
                # Load imaging data
                t_slice = self.open_slice(s+1)
                num_frames = t_slice.shape[-1]
                if s==0:
                    tseries = np.zeros((t_slice.shape[2],r_num,mask.shape[2]))
                    
                t_mask = mask[:,:,s]
                
                mrange = np.unique(t_mask[:])
                mrange = mrange[mrange>0]
                
                for i, i_n in enumerate(mrange):
                    
                    # Process each ROI
                    mskdx = t_mask ==i_n
                    projected = t_slice * mskdx[:,:,None]
                    active_pixels = projected[:,:,0].size-np.count_nonzero(projected[:,:,0]==0)
                    temp = []
                    for frame in range(num_frames):
                        temp.append(np.nansum(projected[:,:,frame])/active_pixels)
                    temp = np.array(temp)
                    
                    tseries[:,i_n-1,s] = temp
                    
                tseries_condensed = np.nansum(tseries,2) # For now is taking the  sum of means... don't know whether I would change
                tseries_df = pd.DataFrame(tseries_condensed)
                tseries_df = tseries_df.apply(fn.lnorm).to_numpy()
                # May want to add another function that interpolates a dynamic baseline as was done for my PhD
                pd.DataFrame(tseries_df).to_csv(os.path.join(self.regfol,r +'.csv'))
                
    def save_postprocessing(self, overwrite=True):
        post_processing_file = os.path.join(self.processedfol, 'postprocessing.h5')
        if not os.path.exists(post_processing_file) or overwrite:
            pv2, ft2, ft, ix = self.behavior_imaging_align()
            pv2.to_hdf(post_processing_file, key='pv2', mode='w')
            ft.to_hdf(post_processing_file, key='ft')
            ft2.to_hdf(post_processing_file, key='ft2')
            pd.DataFrame(ix).to_hdf(post_processing_file, key='ix')
    def load_postprocessing(self):
        post_processing_file = os.path.join(self.processedfol,'postprocessing.h5')
        pv2 = pd.read_hdf(post_processing_file, 'pv2')
        ft = pd.read_hdf(post_processing_file, 'ft')
        ft2 = pd.read_hdf(post_processing_file, 'ft2')
        ix = pd.read_hdf(post_processing_file, 'ix')
        return pv2, ft, ft2, ix 
    def unyoked_phase(self,roi_name):
        pv2, ft, ft2, ix  = self.load_postprocessing()
        wedges = pv2.filter(regex=roi_name)
        wedges.fillna(method='ffill', inplace=True)
        wedges = wedges.to_numpy()
        phase,amp = self.get_centroidphase(wedges)
        offset = self.continuous_offset(wedges,ft2)
        phase_offset = fn.wrap(phase-offset)
        
        return phase, phase_offset, amp
    def phase_yoke(self,yoke_roi,tether_roi,ft2,pv2):
        # Function will output phase and amplitude of columnar regions.
        # Specify a yoke and tether ROI to do the determination
        self.yoke_roi = yoke_roi
        self.tether_roi = tether_roi
        #pv2, ft, ft2, ix  = self.load_postprocessing()
        self.pv2 = pv2
        #self.ft = ft
        self.ft2 = ft2
        #self.ix = ix
        print('Yoking to: ', yoke_roi)        
        yoke_wedges = pv2.filter(regex=yoke_roi)
        yoke_wedges.fillna(method='ffill', inplace=True)
        yoke_wedges = yoke_wedges.to_numpy()
        phase,amp = self.get_centroidphase(yoke_wedges)
        offset = self.continuous_offset(yoke_wedges,ft2)
        phase_yoke_offset = fn.wrap(phase-offset)
        fit_wedges, all_params = self.wedges_to_cos(yoke_wedges,phase_offset = offset)
        rot_wedges = self.rotate_wedges(yoke_wedges,phase_offset =offset)
        d = {
        'wedges_' + yoke_roi: yoke_wedges,
        'wedges_offset_' + yoke_roi: rot_wedges,
        'phase_' + yoke_roi: phase,
        'offset': offset,
        'offset_' + yoke_roi +'_phase': phase_yoke_offset,
        'fit_wedges_' + yoke_roi: fit_wedges,
        'all_params_' + yoke_roi: all_params,
        'amp_' + yoke_roi: amp
        }
        for roi in tether_roi:
            print(roi)
            teth_wedges = pv2.filter(regex=roi)
            
            teth_wedges.fillna(method='ffill',inplace=True)
            teth_wedges  = teth_wedges.to_numpy()
            
            phase_teth,amp = self.get_centroidphase(teth_wedges)
            phase_teth_off = fn.wrap(phase_teth-offset)
            
            fit_wedges,all_params = self.wedges_to_cos(teth_wedges,phase_offset = offset)
            rot_wedges = self.rotate_wedges(teth_wedges,phase_offset =offset)
            d.update({
            'wedges_' + roi: teth_wedges,
            'wedges_offset_' + roi: rot_wedges,
            'phase_' + roi: phase_teth,
            'offset_' + roi +'_phase': phase_teth_off,
            'fit_wedges_' + roi: fit_wedges,
            'all_params_' + roi: all_params,
            'amp_' + roi: amp,
            })
        return d
# %% Workhorse functions

    def wedges_to_cos(self, wedges, phase_offset=None):
        if phase_offset is None:
            phase_offset = np.zeros(len(wedges))
        fit_wedges = np.zeros(wedges.shape)
        all_params = np.zeros((len(wedges),3))
        for i,fb_at_t in enumerate(wedges):
            offset = phase_offset[i]
            fit,params = fn.fit_cos(fb_at_t, offset=offset)
            fit_wedges[i,:] = fit
            all_params[i,:] = params
        return fit_wedges, all_params
    def rotate_wedges(self,wedges,phase_offset=None):
        pi = np.pi
        if phase_offset is None:
            rot_wedges = wedges
        else:
            rot_wedges = np.zeros_like(wedges)
            offset_idx = np.round(8*phase_offset/pi).astype(int)
            for i,o in enumerate(offset_idx):
                tw = wedges[i,:]
                rot_wedges[i,:] = np.append(tw[o:],tw[:o])
            
        return rot_wedges
            
    def continuous_offset(self, data,ft2):
        """
        calculate the phase offset between tube and epg bumps
        """
        phase,amp = self.get_centroidphase(data)
        
        tube = ft2['ft_heading'].to_numpy()
        offset = fn.unwrap(phase) - fn.unwrap(tube)
        offset = pd.Series(offset)
        offset = fn.wrap(offset.rolling(20, min_periods=1).mean())
        return offset
    def get_centroidphase(self, data):
        """
        project eip intensity to eb, then average to get centroid position
        """
        phase, amp = fn.centroid_weightedring(data)
        return phase, amp 
    def calculate_del_t(self):
        # prarieview starts when fictrac computer sends its first pulse
        ft = self.load_preprocessing()
        spikes = self.spikes
        timing = self.timing
        try:
            # timing of prarieview voltage recording
            pv_spike_time = timing.total_time[0] + spikes['Time(ms)'] / 1000
            # find index values when ft computer sent pulses (pulses
            ft_idx, _ = sg.find_peaks(ft.sig_status, height=0.8)
            
            
            # CD edit - sometimes first spike is not registered as peak
            sp = spikes[' Input 0'].to_numpy()
            #sp = np.append(np.median(sp),sp)
            
            # find index values when pv computer received pulses (pulses are 3V)
            pv_idx, _ = sg.find_peaks(sp, height=2.5)
            #pv_idx = pv_idx-1
            # if the first peak occurs at the beginning of the pv voltage trace, discard
            # if abs(pv_spike_time[pv_idx[0]]-timing.total_time[0])<0.5:
            #     pv_idx = pv_idx[1:]
            # start of prarieview recording
            A = timing.total_time[0]
            # first pulse from fictrac
            B = ft.seconds[ft_idx].iloc[0]
            delta_t = B-A # fictrac-prarieview

            # ensure that spikes align. if they don't revert to old delta t
            A = pv_spike_time[pv_idx].to_numpy()
            B = ft.seconds[ft_idx].to_numpy()-delta_t
            # find indices where fictrac pulses are closest to prarieview received pulses
            a = fn.closest_argmin(A, B)
            gap = np.mean(B[a]-A)
            if np.abs(gap)>0.1:
                delta_t = self.calculate_del_t_old()
                B = ft.seconds[ft_idx].to_numpy()-delta_t
                # find indices where fictrac pulses are closest to prarieview received pulses
                a = fn.closest_argmin(A, B)
                gap = np.mean(B[a]-A)

            fig, axs = plt.subplots(1,1)
            line1, = axs.plot(pv_spike_time, spikes[' Input 0'], label='signal_received')
            axs.plot(pv_spike_time[pv_idx], spikes[' Input 0'][pv_idx], 'ro')
            line2, = axs.plot(ft.seconds, ft.sig_status, label='signal_sent')
            axs.plot(ft.seconds[ft_idx], ft.sig_status[ft_idx], 'ro')
            line3, = axs.plot(ft.seconds-delta_t, ft.sig_status, label='aligned_signal_sent')
            axs.plot(ft.seconds[ft_idx]-delta_t, ft.sig_status[ft_idx], 'go')
            #axs.set_xlim(ft.seconds[0]-1, max(B[a]+1))
            axs.set_xlabel('time(s)')
            axs.set_ylabel('signal')
            plt.text(ft.seconds[0], 2, 'delta_t is: '+str(delta_t)+'s'+ ' gap:'+str(gap))
            plt.legend()
        except:
            ft_idx, _ = sg.find_peaks(ft.sig_status, height=0.8)
            A = timing.total_time[0]
            # first pulse from fictrac
            B = ft.seconds[ft_idx].iloc[0]
            delta_t = B-A # fictrac-prarieview

        return delta_t
    def split_files(self, ignore_Ch1=False):
        """
        split imaging files depending on whether
            1) there are two colors
            2) imaging is volumetric or simple t series
            3) if it's volumetric, see if it's bidirectional (sometimes used for faster imaging)
        """

        # read in xml file and imaging file names
        xml_file = self.read_image_xml()
        for i in np.arange(len(self.xmls)):
            strings = ['._','VoltageRecording']
            if not any(string in self.xmls[i].name for string in strings):
                tree=ET.parse(os.path.join(self.imagefol, self.xmls[i].name))
                root=tree.getroot()

        # read in .tif files
        if ignore_Ch1:
            fls = glob.glob(os.path.join(self.imagefol, '*Ch2*.tif'))
        else:
            fls = glob.glob(os.path.join(self.imagefol, '*.tif'))
        fls.sort()

        # identify number of channels
        red, green, dual_color = False, False, False
        for f in fls:
            if 'Ch1' in f:
                red = True
                break
        for f in fls:
            if 'Ch2' in f:
                green = True
                break
        if red and green:
            dual_color = True

        slice_stacks = {}

        # simple t-series
        if self.sequence_type == 'TSeries Timed Element':
            slice_stacks[1] = fls

        # volumetric imaging
        else:
            # first check if z imaging is bidirectional
            for elem in tree.iter(tag='Sequence'):
                if elem.attrib['bidirectionalZ']=="True":
                    bidirectional = False
                else:
                    bidirectional = False
                break

            # if bidirectional:
            #     zvals = []
            #     for elem in tree.iter(tag='Frame'):
            #         for subelem in elem.iter(tag='SubindexedValue'):
            #             if subelem.attrib['subindex']=="1":
            #                 zvals.append(float(subelem.attrib['value']))
            #     zvals = np.array(zvals)
            #     print(zvals)
            #     zval_unique = np.unique(zvals)
            #     print(zval_unique)
            #     fls = np.array(fls)
            #     for i, z in enumerate(zval_unique):
            #         slice_stacks[i+1] = fls[zvals==z].tolist()

            if bidirectional:
                slices = xml_file.idx.unique()
                num_slices = len(slices)
                num_cycles = int(np.floor(len(fls)/len(slices)))
                frames = np.arange(len(fls))
                frames = np.reshape(frames, (num_cycles, num_slices))
                frames[1::2,:]=frames[1::2,::-1]
                fls = np.array(fls)
                for i, slice in enumerate(slices):
                    slice_stacks[slice] = fls[frames[:,i]].tolist()

            # sequential
            else:
                slices = xml_file.idx.unique()
                for slice in slices:
                    stack = []
                    identifier = str(slice) + '.ome.tif' #filename identifier
                    for f in fls:
                        if identifier in f:
                            stack.append(f)
                    slice_stacks[slice] = stack
            #setattr(self, 'slices', slices)

            #ensure that all slices have the same number of frames
            min_frame = []
            for slice in list(slice_stacks.keys()):
                min_frame.append(len(slice_stacks[slice]))
            min_frame = min(min_frame)
            for slice in list(slice_stacks.keys()):
                slice_stacks[slice] = slice_stacks[slice][0:min_frame]

            # if crop is specified in google sheet, crop each slice
            if hasattr(self, 'crop'):
                if self.crop is not False:
                    crop = int(self.crop)
                    for slice in list(slice_stacks.keys()):
                        slice_stacks[slice] = slice_stacks[slice][0:crop]

        setattr(self, 'dual_color', dual_color)
        setattr(self, 'dual_color_old', False)
        setattr(self, 'num_slices', len(slice_stacks))
        return slice_stacks    
    def load_rois(self):
        tag = "*.csv"
        csv_files = glob.glob1(self.regfol,tag)
        df_list = []
        for file in csv_files:
            name = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(self.regfol, file))
            if df.columns.str.contains('Mean').any():
                df = df.loc[:, df.columns.str.contains('Mean')]
                df.columns = df.columns.str.strip('Mean(')
                df.columns = df.columns.str.strip(')')
                df_dff = df.apply(fn.dff).add_suffix('_dff')
                df = df.join(df_dff, how = 'right')
            else:
                df.columns = df.columns + '_' + name
                for column in df.columns:
                    if 'index' in column:
                        df.drop(labels=column, axis=1, inplace=True)
                    elif 'Unnamed' in column:
                        df.drop(labels=column, axis=1, inplace=True)
                df_list.append(df)
        if len(df_list)>1:
            rois = pd.concat(df_list, axis=1)
        else:
            rois = df
        # rois.drop(labels='index', axis=1, inplace=True)
        setattr(self, 'rois', rois)
        return rois
    def behavior_imaging_align(self, upsample=True):
        self.load_rois()
        self.split_files() #get number of slices
        delta_t = self.calculate_del_t()

        # before concatenating imaging and timing, need to deal with downsampling from z projections
        # proj_frames = int(np.round(len(self.timing)/len(self.rois)))
        proj_frames = self.num_slices
        #print('number of projected frames is:', proj_frames)
        if proj_frames>1:
            df = self.timing
            dfs = []
            for i in np.arange(1,proj_frames+1):
                dfs.append(df[df.idx==i].reset_index())
            dfs = pd.concat(dfs)
            timing_new = dfs.groupby(dfs.index).mean().drop(columns=['index', 'idx'])
            setattr(self, 'timing', timing_new)
        # if the images are to be cropped, crop the timing
        if self.crop is not False:
            crop = int(self.crop)
            timing_new = timing_new.iloc[0:crop]
            setattr(self, 'timing', timing_new)
            print('timing length =', len(self.timing))


        # combine imaging and timing
        pv2 = pd.concat([self.rois, self.timing], axis=1)
        pv2.rename(columns={"total_time": "seconds"}, inplace=True)

        # upsample imaging to 10Hz for consistency across animals/trials
        if upsample:
            seconds = pv2.seconds
            upsampled_seconds = np.arange(seconds[0], seconds.iloc[-1], 0.1)
            dropt_df = pv2.drop(columns='seconds')
            upsampled_dict = {}
            upsampled_dict['seconds'] = upsampled_seconds
            for column in dropt_df.columns:
                upsampled_dict[column] = np.interp(upsampled_seconds, seconds, dropt_df[column])
            pv2 = pd.DataFrame(upsampled_dict)
        print(pv2)
       
        # bin fictrac data based on closest timepoint to imaging data
        ft = self.load_preprocessing()
        ft['seconds'] = ft['seconds']-delta_t
        ix = fn.closest_argmin(pv2.seconds.to_numpy(),ft.seconds.to_numpy()) #FT index closest to PV index

        # if there are duplicates remove those rows from FT and PV.
        dup = pd.DataFrame(ix)
        if dup.duplicated().any():
            pv2 = pv2.drop(pv2.index[dup.duplicated()])
            ix = dup.drop(dup.index[dup.duplicated()]).to_numpy().flatten()

        # ix is the center of the bins, define left and right edges for grouping, make intervals
        right=[]
        for i in np.arange(len(ix)-1):
            right.append([np.floor((ix[i]+ix[i+1])/2)])
        right = np.array(right)
        left = right+1
        left = np.concatenate(([ix[0]],left.flatten()))
        right = np.concatenate((right.flatten(), [ix[-1]]))
        intervals = pd.IntervalIndex.from_arrays(left=left, right=right)

        # before downsampling unwrap headings
        ft['motor_heading'] = fn.unwrap(ft.motor_heading)
        ft['ft_heading'] = fn.unwrap(ft.ft_heading)

        # make bins according to intervals, take average to downsample
        ft2 = ft.groupby(pd.cut(ft.index, intervals, duplicates='drop')).mean()
        ft2 = ft2.reset_index()
        ft2 = ft2.drop(labels = 'index', axis = 1)
        #drop intervals, because cannot be saved to .h5
        ft2 = ft2.drop(columns=['level_0'])
        ft2['instrip'] = np.round(ft2['instrip'])
        # index where ft2 is missing data because of single time point bins (common with very fast imaging)
        ix_missing = ft2[ft2.isna().any(axis=1)].index.to_list()
        # corresponding ft points
        ix_fill = ix[ix_missing]
        # dataframe for filling missing rows
        df_fill = ft[ft2.columns]
        ft2.iloc[ix_missing] = df_fill.loc[ix_fill]

        # wrap up headings
        ft2['motor_heading'] = fn.wrap(ft2.motor_heading)
        ft2['ft_heading'] = fn.wrap(ft2.ft_heading)
        ft['motor_heading'] = fn.wrap(ft.motor_heading)
        ft['ft_heading'] = fn.wrap(ft.ft_heading)

        # crop fictrac to be same amount of time as ft and ft2.
        ft = ft.iloc[ix[0]:ix[-1]]

        # lowercase all column titles
        ft.columns= ft.columns.str.lower()
        pv2.columns= pv2.columns.str.lower()
        ft2.columns= ft2.columns.str.lower()

        # if last columns of pv2 is nans, it means there as an incomplete z slice, crop both pv2 and ft2 by one row
        n=1
        pv2.drop(pv2.tail(n).index,inplace=True)
        ft2.drop(ft2.tail(n).index,inplace=True)

        return pv2, ft2, ft, ix

    def open_slice(self, slice):
        for file in os.scandir(self.regfol):
            if file.name.endswith('slice'+str(slice)+'.tif') and not file.name.startswith('._'):
                registered_file = file
        slice = io.imread(registered_file.path)

        #transpose and flip to have (x,y,t) and EB in correct orientation
        slice = np.moveaxis(slice, [0,1], [-1,-2])
        slice = np.fliplr(slice)
        return slice
    
    def open_mask(self,maskname):
        fb_mask = io.imread(maskname)
        # little hacky, sometimes axis are read in differently my io.imread, correct is number of frames is axis 0
        num_frames = min(fb_mask.shape)
        if fb_mask.shape[0]==num_frames:
            fb_mask = np.moveaxis(fb_mask, 0, -1)
        fb_mask = np.rot90(fb_mask, axes=(1,0))
        return fb_mask
    
    def read_image_xml(self):
        self.find_xml()
        for i in np.arange(len(self.xmls)):
            strings = ['._','VoltageRecording','VoltageOutput']
            if not any(string in self.xmls[i].name for string in strings):
                # will probably need update for volumentric imaging
                #print('running read_xml')
                tree=ET.parse(os.path.join(self.imagefol, self.xmls[i].name))
                root=tree.getroot()
                #convert starting time to seconds
                for elem in tree.iter(tag='Sequence'):
                    if elem.attrib['cycle'] == '1':
                        time_string=elem.attrib['time']
                time_string=time_string[:-1] #chop off seventh decimal. necessary for datetime parsing
                date_time = datetime.datetime.strptime(time_string, "%H:%M:%S.%f")
                a_timedelta = date_time - datetime.datetime(1900, 1, 1)
                start_seconds = a_timedelta.total_seconds()
                start_seconds = float(start_seconds)
                absolute_time = []
                relative_time = []
                #time_total = []
                idx = []
                for elem in tree.iter(tag='Frame'):
                    relT = elem.attrib['relativeTime']
                    absT = elem.attrib['absoluteTime']
                    #totT = absT + start_seconds
                    ix = elem.attrib['index']
                    absolute_time.append(absT)
                    relative_time.append(relT)
                    #time_total.append(totT)
                    idx.append(ix)

                df = pd.DataFrame({'idx': idx,'absolute_time': absolute_time, 'relative_time': relative_time})
                del absolute_time, relative_time, idx
                df = df.apply( pd.to_numeric, errors='coerce' )
                df['total_time']=df['absolute_time']+start_seconds;
                setattr(self, 'image_xml', df)
                setattr(self, 'sequence_type', tree.find('Sequence').attrib['type'])

                return df
    def find_xml(self):
        xmls = []
        for file in os.scandir(self.imagefol):
            if file.name.endswith('.xml'):
                xmls.append(file)
        setattr(self,'xmls',xmls)
        
    def read_dat(self):
        self.find_dat()
        for file in os.scandir(self.datafol):
            if file.name.endswith('.dat') and not file.name.startswith('._'):
                file_path = os.path.join(self.datafol, file.name)
                names = [
                      'frame',
                      'del_rot_cam_x',
                      'del_rot_cam_y',
                      'del_rot_cam_z',
                      'del_rot_error',
                      'df_pitch',
                      'df_roll',
                      'df_yaw',
                      'abs_rot_cam_x',
                      'abs_rot_cam_y',
                      'abs_rot_cam_z',
                      'abs_rot_lab_x',
                      'abs_rot_lab_y',
                      'abs_rot_lab_z',
                      'ft_posx',
                      'ft_posy',
                      'ft_heading',
                      'ft_movement_dir',
                      'ft_speed',
                      'forward_motion',
                      'side_motion',
                      'timestamp',
                      'sequence_counter',
                      'delta_timestep',
                      'alt_timestep'
                ]
                df = pd.read_table(file_path, delimiter='[,]', names = names, engine='python')
                df.ft_posx = 3*df.ft_posx # CD edit: no longer flipping x flip x and y for mirror inversion
                df.ft_posy = -3*df.ft_posy # 
                df.ft_speed = 3*df.ft_speed
                df['seconds'] = (df.timestamp-df.timestamp.iloc[0])/1000
                return df

    def read_log(self):
        for file in os.scandir(self.datafol):
            if file.name.endswith('.log') and not file.name.startswith('._'):
                file_path = os.path.join(self.datafol, file.name)
        df = pd.read_table(file_path, delimiter='[,]', engine='python')

        #split timestamp and motor into separate columns
        new = df["timestamp -- motor_step_command"].str.split("--", n = 1, expand = True)
        df["timestamp"]= new[0]
        df["motor_step_command"]=new[1]
        df.drop(columns=["timestamp -- motor_step_command"], inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format="%m/%d/%Y-%H:%M:%S.%f ")
        df['seconds'] = 3600*df.timestamp.dt.hour+60*df.timestamp.dt.minute+df.timestamp.dt.second+10**-6*df.timestamp.dt.microsecond

        # motor command sent to arduino as string, need to convert to numeric
        df['motor_step_command'] = pd.to_numeric(df.motor_step_command)

        # CAUTION: invert x. Used to correct for mirror inversion, y already flipped in voe
        # CD edit: this sign flip has been reversed as is no longer an issue
        df['ft_posx'] = df['ft_posx']

        #calculate when fly is in strip
        df['instrip'] = np.where(df['mfc2_stpt']>0.0, True, False)

        return df

    def merge_voe_ft(self):
        # CD commented out references to .dat file.
        # As far as I can tell .dat information is only required if there is a 
        # large time interval between the log and .dat files being made
        # Removing .dat requirement saves data handling admin. Also the .dat
        # files don't appear to be saved all the time. This could be because of
        #how I have been terminating fictrac
        
        # load fictrac .dat and voe .log files
        df1 = self.read_dat()
        df2 = self.read_log()

        # drop unnecessary variables
        df1 = df1.drop(columns=['del_rot_cam_x', 'del_rot_cam_y', 'del_rot_cam_z','abs_rot_cam_x', 'abs_rot_cam_y', 'abs_rot_cam_z', 'abs_rot_lab_x', 'abs_rot_lab_y', 'abs_rot_lab_z', 'forward_motion', 'side_motion', 'alt_timestep'])
        try:
            df2 = df2.drop(columns=['ft_posx', 'ft_posy', 'ft_error', 'ft_roll', 'ft_pitch', 'ft_yaw', 'ft_heading', 'timestamp'])
        except:
            df2 = df2.drop(columns=['ft_posx', 'ft_posy', 'ft_error', 'ft_roll', 'ft_pitch', 'ft_yaw', 'timestamp'])
        if 'ft_frame' in df2.columns:
            df2 = df2.rename(columns={'ft_frame':'frame'})


        # voe starts on this fictrac frame
        start_ix = df2.frame.iloc[0]

        # combine the dataframes
        df_combine = df1.merge(df2, on='frame', how='left')
        
        #crop values where ft started, but voe hadn't
        df_combine = df_combine.iloc[start_ix:-1].reset_index()

        # interpolate nans, column dependent
        pad_nans = ['mfc1_stpt','mfc2_stpt', 'mfc3_stpt', 'led1_stpt', 'led2_stpt', 'instrip']
        linear_nans = ['motor_step_command', 'seconds_y']
        for column in pad_nans:
            df_combine[column] = df_combine[column].interpolate(method='pad')
        for column in linear_nans:
            df_combine[column] = df_combine[column].interpolate(method='linear')

        # interpolating timing results in extra pulses, fill nans with signal minimum
        df_combine['sig_status'] = df_combine['sig_status'].fillna(min(df_combine['sig_status']))

        # calculate motor heading
        heading = df_combine['motor_step_command']-800000
        heading = heading*2*np.pi/800
        heading = fn.wrap(heading)
        df_combine['motor_heading'] = heading

        # correct for points that fall in 3pi/4 to -3pi/4 motor no-go zone
        df_combine.motor_heading[df_combine.motor_heading<-3*np.pi/4]=-3*np.pi/4
        df_combine.motor_heading[df_combine.motor_heading>3*np.pi/4]=3*np.pi/4

        # remap heading to (-pi, pi), invert because of mirror flip
        # CD edit: remove sign flip because this is no longer an issue
        df_combine['ft_heading'] = fn.wrap(fn.unwrap(df_combine.ft_heading))

        # make interpolated voe time the time used for alignment with ft
        if 'seconds_y' in df_combine.columns:
            df_combine = df_combine.rename(columns={'seconds_y':'seconds'})

        # calculate x velocity, y velocity, net motion, all in mm/sec for x and y, rad/s for 'ang'
        df_combine['x_velocity'] = np.gradient(df_combine.ft_posx)/np.mean(np.gradient(df_combine.seconds))
        df_combine['y_velocity'] = np.gradient(df_combine.ft_posy)/np.mean(np.gradient(df_combine.seconds))
        df_combine['ang_velocity'] = df_combine.df_yaw/np.mean(np.gradient(df_combine.seconds))

        # calculate netmotion of ball in rad/frame
        df_combine['net_motion'] = np.sqrt(df_combine.df_pitch**2+df_combine.df_roll**2+df_combine.df_yaw**2)

        # remove very short outside bouts
        d, di, do = fn.inside_outside(df_combine)
        df_combine = fn.consolidate_out(df_combine, do, t_cutoff=0.5)
        return df_combine
    
        
    def read_voltage_recording(self):
        for file in os.scandir(self.imagefol):
            if fnmatch.fnmatch(file, '*VoltageRecording*.csv'):
                file_path = os.path.join(self.imagefol, file.name)
                df = pd.read_csv(file_path)
                return df
            
    def find_dat(self):
        import shutil
        for f in os.scandir(self.datafol):
            if f.name.endswith('.dat'):
                #print('found log file in folder', self.datafol)
                break
        else:
            for f in os.scandir(self.datafol):
                if f.name.endswith('.log') and not f.name.startswith('._'):
                    file = f.name
                    print('logfile:',file)
                    dt = file.split('_')[0]
                    date = dt.split('-')[0]
                    time = dt.split('-')[1]
                    date = date[4:8]+date[0:4] # year in front to match fictrac dat files
                    sec_log = 3600*int(time[0:2])+60*int(time[2:4])+int(time[4:6]) # seconds since midnight
                    print('sec_log:', sec_log)
            dat_files, secs_dat = [], []
            for f in os.scandir(self.dat_file_folder):
                if f.name.endswith('.dat') and not f.name.startswith('._'):
                    if (date in f.name):
                        dat_files.append(f)
                        file = f.name
                        time = file.split('-')[1]
                        time = time.split('_')[1]
                        s = 3600*int(time[0:2])+60*int(time[2:4])+int(time[4:6])
                        secs_dat.append(s)
                        #print(f.name, s)
            print('secs_dat:', secs_dat)
            print('dat_files:',dat_files)
            dat_file_ix = fn.closest_val_ix(secs_dat, sec_log)
            print(dat_file_ix)
            dat_file = dat_files[dat_file_ix]
            print('datfile:',dat_file.name)
            source_path = dat_file.path
            dest_path = os.path.join(self.datafol, dat_file.name)
            shutil.copy(source_path, dest_path)
            print('successfully copied .dat file')
        
    def load_preprocessing(self):
        try:
            spikes = pd.read_hdf(os.path.join(self.processedfol,'preprocessing.h5'), 'spikes')
        except:
            spikes = None
        timing = pd.read_hdf(os.path.join(self.processedfol,'preprocessing.h5'), 'timing')
        ft = pd.read_hdf(os.path.join(self.processedfol,'preprocessing.h5'), 'ft')
        setattr(self, 'spikes', spikes)
        setattr(self, 'timing', timing)
        return ft
    
    def bumpstraighten(self,ft,ft2):
        x = ft2['ft_posx'].to_numpy()
        y = ft2['ft_posy'].to_numpy()
        heading = ft2['ft_heading']
        obumps = ft['bump'].to_numpy()
        obumps_u = obumps[np.abs(obumps)>0]
        obumpsfr = ft['frame'][np.abs(obumps)>0]
        bumps = ft2['bump']
        frames = ft2['frame']
        bumps_new = np.zeros_like(bumps)
        for i,f in enumerate(obumpsfr):
            
            frd = frames-f
            w = np.argmin(np.abs(frd))
            
            bumps_new[w] = obumps_u[i]
        
        
        bumps = bumps_new
        binst = np.where(np.abs(bumps)>0)[0]
        xnew = x.copy()
        ynew = y.copy()
        headingnew = heading.copy()
        tbold = 0
        for b in range(len(binst)-1):
            bi = binst[b]
            tb = bumps[bi]+tbold
            tbold = tb
            bdx = np.arange(bi,binst[b+1],step=1,dtype=int)
            bc =np.cos(-tb)
            bs = np.sin(-tb)
            tx = x[bdx]
            ty = y[bdx]
            tx = tx-tx[0]
            ty = ty-ty[0]
            tx2 = tx*bc-ty*bs
            ty2 = tx*bs+ty*bc
            dx = tx2[0]-xnew[bdx[0]-1]
            dy = tx2[0]-ynew[bdx[0]-1]
            tx2 = tx2-dx
            ty2 = ty2-dy
            xnew[bdx] = tx2
            ynew[bdx] = ty2
            
            th = heading[bdx]+tb
            tc = np.cos(th)
            ts = np.sin(th)
            th = np.arctan2(ts,tc)
            headingnew[bdx] = th
            
            
        dx = xnew[(bdx[-1]+1)]-xnew[bdx[-1]]
        xnew[(bdx[-1]+1):] = xnew[(bdx[-1]+1):]-dx
        dy = ynew[(bdx[-1]+1)]-ynew[bdx[-1]]
        ynew[(bdx[-1]+1):] = ynew[(bdx[-1]+1):]-dy
        
        return xnew,ynew,headingnew
        
        
        
        
        