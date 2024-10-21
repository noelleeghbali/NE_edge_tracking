import glob
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import fnmatch
import pickle
from scipy import signal as sg
from scipy import stats
import os, sys
import xml.etree.ElementTree as ET
import pandas as pd
import datetime
import numpy as np
import skimage
import importlib
from skimage import io, data, registration, filters, measure
from scipy.ndimage import shift
import datetime
import importlib
#from numba import jit
from analysis_funs.utilities import funcs as fn
import cv2
importlib.reload(fn)
pickle.HIGHEST_PROTOCOL = 4

# %%
class fly:
    def __init__(self, name, folderloc, **kwargs):
        self.name = name
        self.folderloc = folderloc # folder of fly
        self.datafol = os.path.join(self.folderloc, 'data') # data folder contains imaging folder, .log, .dat,
        self.regfol = os.path.join(self.folderloc, 'registered') # register folder contains registered .tiffs
        self.processedfol = os.path.join(self.folderloc, 'processed')
        self.dat_file_folder = '/Volumes/LACIE/dat_files'# folder to look for .dat files
        if not os.path.exists(self.datafol):
            os.makedirs(self.datafol)
        if not os.path.exists(self.regfol):
            os.makedirs(self.regfol)
        if not os.path.exists(self.processedfol):
            os.makedirs(self.processedfol)
        # find folder containing .tiff images and .xmls
        for item in os.scandir(self.datafol):
            if item.is_dir():
                for file in os.scandir(item):
                    if file.is_file() and file.name.endswith('.xml'):
                        self.imagefol = os.path.join(self.datafol, item.name)
        # if 'split' in kwargs.keys():
        #     self.split = kwargs['split']
        # if 'delta_t' in kwargs.keys():
        #     self.delta_t = kwargs['delta_t']
        # if 'cell_type' in kwargs.keys():
        #     self.cell_type = kwargs['cell_type']

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])

        # specify which z slices to use for masks. Used primarily for hdelta/epg imaging
        mask_slice = {}
        for key in list(kwargs.keys()):
            if 'mask_slice' in key:
                mask_slice[key] = kwargs[key]
        self.mask_slice = mask_slice



    def find_xml(self):
        xmls = []
        for file in os.scandir(self.imagefol):
            if file.name.endswith('.xml'):
                xmls.append(file)
        setattr(self,'xmls',xmls)

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
                df.ft_posx = -3*df.ft_posx # flip x and y for mirror inversion
                df.ft_posy = -3*df.ft_posy
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
        df['ft_posx'] = -df['ft_posx']

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
        df_combine['ft_heading'] = -fn.wrap(fn.unwrap(df_combine.ft_heading))

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
                    
                
                # array approach if the data are not saved with plane number
                # this happens for non identical z spacings
                if not slice_stacks[2]:
                    fls = np.array(fls)
                    lf = len(fls)
                    for s in slices:
                        sdx = np.arange(s-1,lf,step=len(slices),dtype='int')
                        slice_stacks[s] = fls[sdx].tolist()
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


    def register_image_block(self, files, ini_reg_frames=600):
        from pystackreg import StackReg
        
        f = files
        f1 = f[0]
        f1s = f1.split('.')
        chn = int(f1s[0][-1])-1
        #images = io.imread_collection(f)
        
        # CD edit 08/10/2024 
        # Simplifying and speeding up image loading
        for i,file in enumerate(f):
            im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            if i ==0:
                original = np.zeros((len(f),np.shape(im)[0],np.shape(im)[1]),dtype = 'uint16')
            idx = int(file.split('Cycle')[1][0:5]) #More robust than file order
            #print(idx-1,i,idx-1-i)
            original[idx-1,:,:] = im
         
        
        # CD edit: io.concatenate does not work because tiffs are loaded in 
        # a strange way. This is a work around
        # try :
        #     print(chn)
        #     all_images = [image[np.newaxis, ...] for image in images]
        #     all_images2 = all_images[1:]
        #     print(np.shape(all_images[0]))
        #     if len(np.shape(all_images[0]))==4:
        #         extra_im = all_images[0][0,chn,:,:]
        #     elif len(np.shape(all_images[0]))==5:
        #         extra_im = all_images[0][0,0,chn,:,:]
        #     extra_im = extra_im[np.newaxis,...]
        #     all_images2.append(extra_im)
        #     original = np.concatenate(all_images2)
        # except: # CD edit -reordeded this since below actually takes some time
        #     print(chn)
        #     original = io.concatenate_images(images)
        print('Original shape',np.shape(original))
        registered = np.zeros(original.shape)
        registered_blurred = np.zeros(original.shape)
        sr = StackReg(StackReg.RIGID_BODY)
        #out_previous = sr.register_transform_stack(original, reference=reference)
        registered = sr.register_transform_stack(original, reference='mean')#, n_frames=20, moving_average = 20)
        for i, frame in enumerate(registered):
            registered_blurred[i] = filters.gaussian(frame, 1)
        reg_results = {}
        return reg_results, registered_blurred

    def register_all_images(self, overwrite=True):
        if overwrite or len(os.listdir(self.regfol))==0:
            slice_stacks = self.split_files()
            for i in np.arange(1, len(slice_stacks)+1):
                print('stack:', i)
                files = slice_stacks[i]
                files.sort()
                if hasattr(self, 'split'):
                    for j, range in enumerate(self.split):
                        tif_name = os.path.join(self.regfol, self.name+'_slice'+str(i)+'_split'+str(j)+'.tif')
                        pickle_name = os.path.join(self.regfol, self.name+'_slice'+str(i)+'_split'+str(j)+'.pickle')
                        files_sub = files[(range[0]-1):(range[1]-1)]
                        reg_results, registered_blurred = self.register_image_block(files_sub, ini_reg_frames=100)
                        io.imsave(tif_name, registered_blurred, plugin='tifffile')
                        fn.save_obj(reg_results, pickle_name)
                elif self.dual_color:
                    # need to create options for registering one color based on another, or registering both separately
                    Ch1 = [f for f in files if 'Ch1' in f]
                    Ch2 = [f for f in files if 'Ch2' in f]
                    Chns = [Ch1, Ch2]
                    for ch in [1,2]:
                        tif_name = os.path.join(self.regfol, self.name+'_Ch'+str(ch)+'_slice'+str(i)+'.tif')
                        pickle_name = os.path.join(self.regfol, self.name+'_Ch'+str(ch)+'_slice'+str(i)+'.pickle')
                        reg_results, registered_blurred = self.register_image_block(Chns[ch-1], ini_reg_frames=100)
                        io.imsave(tif_name, registered_blurred, plugin='tifffile')
                        fn.save_obj(reg_results, pickle_name)
                elif self.dual_color_old:
                    # for imaging on Lyndon where the channel names are 2 and 3 instead of 1 and 2
                    Ch2 = [f for f in files if 'Ch2' in f]
                    Ch3 = [f for f in files if 'Ch3' in f]
                    Chns = [Ch2, Ch3]
                    for ch in [2,3]:
                        tif_name = os.path.join(self.regfol, self.name+'_Ch'+str(ch)+'_slice'+str(i)+'.tif')
                        pickle_name = os.path.join(self.regfol, self.name+'_Ch'+str(ch)+'_slice'+str(i)+'.pickle')
                        reg_results, registered_blurred = self.register_image_block(Chns[ch-2], ini_reg_frames=100)
                        io.imsave(tif_name, registered_blurred, plugin='tifffile')
                        fn.save_obj(reg_results, pickle_name)
                else:
                    tif_name = os.path.join(self.regfol, self.name+'_slice'+str(i)+'.tif')
                    pickle_name = os.path.join(self.regfol, self.name+'_slice'+str(i)+'.pickle')
                    reg_results, registered_blurred = self.register_image_block(files, ini_reg_frames=100)
                    io.imsave(tif_name, registered_blurred, plugin='tifffile')
                    fn.save_obj(reg_results, pickle_name)

    def z_projection(self, overwrite=True):
        if overwrite:
            images = []
            for file in os.scandir(self.regfol):
                if file.name.endswith('.tif') and 'slice' in file.name and '_z_projection' not in file.name and 'mask' not in file.name and not file.name.startswith('._'):
                    image = io.imread(file.path, plugin='tifffile')
                    images.append(image)
            images = np.array(images)
            projection = np.mean(images, axis=0)
            psize = np.shape(projection)
            
            tif_name = os.path.join(self.regfol, self.name+'_z_projection'+'.tif')
            io.imsave(tif_name, projection, plugin='tifffile')

    def t_projection_mask_slice(self):
        """
        for specified slices, save a stack which consists of the projection of each slice.  Used to make masks
        """
        slist = os.listdir(self.regfol)
        slices = np.array([],int)
        for s in slist:
            spl = s.split("slice")
            if len(spl)>1:
                if spl[1][2] == 'p':
                    slices = np.append(slices,int(spl[1][0]))
        # for key in list(self.mask_slice.keys()):
        #     slices = self.mask_slice[key]
        key = 'All'    
        stack = []
        stackm = []
        for slice in slices:
            for file in os.scandir(self.regfol):
                if file.name.endswith('slice'+str(slice)+'.tif') and not file.name.startswith('._'):
                    image = io.imread(file.path, plugin='tifffile')                     
                    proj = np.mean(image, axis=0)
                    stack.append(proj)
                    projm = np.max(image, axis =0)
                    stackm.append(projm)
        stack = np.array(stack)
        stackm = np.array(stackm)
        tif_name = os.path.join(self.regfol, self.name+'_'+key+'.tif')
        io.imsave(tif_name, stack)
        tif_name_m = os.path.join(self.regfol, self.name+'_'+key+'_max.tif')
        io.imsave(tif_name_m, stackm)

    def save_preprocessing(self, overwrite=True):
        pre_processing_file = os.path.join(self.processedfol, 'preprocessing.h5')
        if os.path.exists(pre_processing_file) or overwrite:
            self.read_image_xml().to_hdf(pre_processing_file, key='timing', mode='w')
            self.read_dat().to_hdf(pre_processing_file, key='ft_raw')
            self.read_log().to_hdf(pre_processing_file, key='voe_log')
            self.merge_voe_ft().to_hdf(pre_processing_file, key='ft')
            if self.read_voltage_recording() is not None:
                self.read_voltage_recording().to_hdf(pre_processing_file, key='spikes')

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

    def load_rois_old(self):
        tag = "*.csv"
        csv_files = glob.glob1(self.regfol,tag)

        if 'epg' in self.cell_type:
            csv_files = glob.glob1(self.regfol,"*.csv")
            csv_list = []
            for file in csv_files:
                if 'pb.csv' in file:
                    pb = pd.read_csv(os.path.join(self.regfol, file))
                    pb = pb.loc[:, pb.columns.str.contains('Mean')]
                    pb.columns = pb.columns.str.strip('Mean(')
                    pb.columns = pb.columns.str.strip(')')
                    # sort to map each PB glomerulus to it's corresponting EB position
                    pb = pb[['2','4','6','8','10','12','14','16','1','3','5','7','9','11','13','15']]
                    pb.columns = pb.columns + '_pb'
                    for column in pb.columns:
                        if 'index' in column:
                            pb.drop(labels=column, axis=1, inplace=True)
                rois = pb

        elif 'hdelta' in self.cell_type:
            csv_files = glob.glob1(self.regfol,"*.csv")
            csv_list = []
            for file in csv_files:
                # first load PB which has ROIs manually drawn in Fiji. column labels need to be correct: L-2,4,6,8,10,12,14,16,1,3,5,7,9,11,13,15-R
                if 'pb.csv' in file:
                    pb = pd.read_csv(os.path.join(self.regfol, file))
                    pb = pb.loc[:, pb.columns.str.contains('Mean')]
                    pb.columns = pb.columns.str.strip('Mean(')
                    pb.columns = pb.columns.str.strip(')')
                    # sort to map each PB glomerulus to it's corresponting EB position
                    pb = pb[['2','4','6','8','10','12','14','16','1','3','5','7','9','11','13','15']]
                    pb.columns = pb.columns + '_pb'
                    for column in pb.columns:
                        if 'index' in column:
                            pb.drop(labels=column, axis=1, inplace=True)
                # now load FB wedges, which have been automatically created and enumerated left to right in class FB
                if 'fb.csv' in file:
                    fb = pd.read_csv(os.path.join(self.regfol, file))
                    fb.columns = fb.columns + '_fb'
                    for column in fb.columns:
                        if 'index' in column:
                            fb.drop(labels=column, axis=1, inplace=True)
                        elif 'Unnamed' in column:
                            fb.drop(labels=column, axis=1, inplace=True)
            rois = pd.concat([pb.reset_index(), fb.reset_index()], axis=1)
            rois.drop(labels='index', axis=1, inplace=True)

        elif 'hdb' in self.cell_type:
            csv_files = glob.glob1(self.regfol,"*.csv")
            csv_list = []
            for file in csv_files:
                # first load PB which has ROIs manually drawn in Fiji. column labels need to be correct: L-2,4,6,8,10,12,14,16,1,3,5,7,9,11,13,15-R
                if 'eb.csv' in file:
                    eb = pd.read_csv(os.path.join(self.regfol, file))
                    eb.columns = eb.columns + '_eb'
                    for column in eb.columns:
                        if 'index' in column:
                            eb.drop(labels=column, axis=1, inplace=True)
                        elif 'Unnamed' in column:
                            eb.drop(labels=column, axis=1, inplace=True)
                # now load FB wedges, which have been automatically created and enumerated left to right in class FB
                if 'fb.csv' in file:
                    fb = pd.read_csv(os.path.join(self.regfol, file))
                    fb.columns = fb.columns + '_fb'
                    for column in fb.columns:
                        if 'index' in column:
                            fb.drop(labels=column, axis=1, inplace=True)
                        elif 'Unnamed' in column:
                            fb.drop(labels=column, axis=1, inplace=True)
            if 'eb' in locals():
                rois = pd.concat([eb.reset_index(), fb.reset_index()], axis=1)
            else:
                rois = pd.concat([fb.reset_index()], axis=1)
            rois.drop(labels='index', axis=1, inplace=True)

        # experimenting with looking at upper and lower part of hdc
        elif 'hd' in self.cell_type:
            csv_files = glob.glob1(self.regfol,"*fb*.csv")
            csv_list = []
            for file in csv_files:
                fb = pd.read_csv(os.path.join(self.regfol, file))

                # for looking at upper and lower FB layers
                if 'upper' in file:
                    tag = 'upper'
                elif 'lower' in file:
                    tag = 'lower'
                else:
                    tag = 'fb'
                fb.columns = fb.columns + '_'+tag

                for column in fb.columns:
                    if 'index' in column:
                        fb.drop(labels=column, axis=1, inplace=True)
                    elif 'Unnamed' in column:
                        fb.drop(labels=column, axis=1, inplace=True)
                csv_list.append(fb)
            rois = pd.concat(csv_list, axis=1)
            #rois.drop(labels='index', axis=1, inplace=True)

        elif len(csv_files) == 1: # one z slice no splits
            csv = pd.read_csv(os.path.join(self.regfol, csv_files[0]))
            csv = csv.loc[:, csv.columns.str.contains('Mean')]
            csv.columns = csv.columns.str.strip('Mean(')
            csv.columns = csv.columns.str.strip(')')
            csv_dff = csv.apply(fn.dff).add_suffix('_dff')
            rois = csv.join(csv_dff, how = 'right')

        elif self.split is not None: # one slice, multiple splits
            csv_list = []
            for file in csv_files:
                csv = pd.read_csv(os.path.join(self.regfol, file))
                csv = csv.loc[:, csv.columns.str.contains('Mean')]
                csv.columns = csv.columns.str.strip('Mean(')
                csv.columns = csv.columns.str.strip(')')
                csv_dff = csv.apply(fn.dff).add_suffix('_dff')
                csv = csv.join(csv_dff, how = 'right')
                csv_list.append(csv)
            rois = pd.concat(csv_list, ignore_index = True)
        setattr(self, 'rois', rois)
        return rois

    def calculate_del_t_old(self):
        ft = self.load_preprocessing()
        spikes = self.spikes
        if hasattr(self, 'delta_t'):
            delta_t = self.delta_t
        elif (spikes is None) or ' Input 0' not in spikes.columns:
            delta_t = 0
        else:
            timing = self.timing
            # timing of prarieview voltage recording
            pv_spike_time = timing.total_time[0] + spikes['Time(ms)'] / 1000
            # find index values when ft computer sent pulses (pulses
            ft_idx, _ = sg.find_peaks(ft.sig_status, height=0.8)
            # find index values when pv computer received pulses (pulses are 3V)
            pv_idx, _ = sg.find_peaks(spikes[' Input 0'], height=2.5)
            # if the first peak occurs at the beginning of the pv voltage trace, discard
            if abs(pv_spike_time[pv_idx[0]]-timing.total_time[0])<0.5:
                pv_idx = pv_idx[1:]
            A = pv_spike_time[pv_idx].to_numpy()
            B = ft.seconds[ft_idx].to_numpy()
            # find indices where fictrac pulses are closest to prarieview received pulses
            a = fn.closest_argmin(A, B)

            delta_t = np.min(B[a]-A)

            fig, axs = plt.subplots(1,1)
            line1, = axs.plot(pv_spike_time, spikes[' Input 0'], label='signal_received')
            axs.plot(pv_spike_time[pv_idx], spikes[' Input 0'][pv_idx], 'ro')
            line2, = axs.plot(ft.seconds, ft.sig_status, label='signal_sent')
            axs.plot(ft.seconds[ft_idx], ft.sig_status[ft_idx], 'ro')
            #axs.set_xlim(ft.seconds[0]-1, max(B[a]+1))
            axs.set_xlabel('time(s)')
            axs.set_ylabel('signal')
            plt.text(ft.seconds[0], 2, 'delta_t is: '+str(delta_t)+'s')
            plt.legend()
        return delta_t

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
            # find index values when pv computer received pulses (pulses are 3V)
            pv_idx, _ = sg.find_peaks(spikes[' Input 0'], height=2.5)
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

class FSB:
    def __init__(self, name, folderloc):
        self.name = name
        self.folderloc = folderloc # folder of fly
        self.datafol = os.path.join(self.folderloc, 'data') # data folder contains imaging folder, .log, .dat,
        self.regfol = os.path.join(self.folderloc, 'registered') # register folder contains registered .tiffs
        self.processedfol = os.path.join(self.folderloc, 'processed')
        #self.zslices = z_slices #index of slices for FB, not currently using
        self.reg_tif = self.open_registered()
        self.len = self.reg_tif.shape[-1]

    def process_channel(self):
        self.rois, self.theta_list, self.a = self.get_wedges()

        self.al = self.norm_over_time(self.a, mode='linear')
        self.az = self.norm_over_time(self.a, mode='zscore')
        self.an = self.norm_over_time(self.a)
        self.phase_dff0 = self.get_centroidphase(self.an)


    def open_mask(self, name = None):
        for file in os.scandir(self.regfol):
            if file.name.endswith('fbmask.tif'):
                mask_file = file
        # open mask tif
        fb_mask = io.imread(mask_file.path)
        fb_mask = fb_mask.astype('float')
        fb_mask[fb_mask==0.0]=np.nan
        fb_mask = np.rot90(fb_mask, axes=(1,0))
        return fb_mask

    def open_boundary(self):
        for file in os.scandir(self.regfol):
            if file.name.endswith('boundary.tif'):
                boundary_file = file
        fb_boundary = io.imread(boundary_file.path)
        fb_boundary = np.rot90(fb_boundary, axes=(1,0))
        return fb_boundary

    def open_registered(self):
        for file in os.scandir(self.regfol):
            if file.name.endswith('FB_stack_max.tif'):
                registered_file = file
        reg_tif = io.imread(registered_file.path)

        #transpose to have (x,y,z,t) and select slices for FB
        reg_tif = np.moveaxis(reg_tif, [0, 1], [-1, -2])
        #reg_tif = reg_tif[:,:,:]
        return reg_tif

    def get_wedges(self, wedge_num=16):
        """
        open a mask, boundary and registered stack
        """
        fbmask = self.open_mask()
        self.fbmask = fbmask
        fb_boundary = self.open_boundary()
        self.fb_boundary =fb_boundary
        reg_tif = self.open_registered()
        projected = np.zeros(reg_tif.shape)


        def lines_intersection(a1, b1, a2, b2):
            return (b2-b1)/(a1-a2), (a1*b2-a2*b1)/(a1-a2)

        def wrap_rad_simple(angle):
            if angle < 0:
                return angle + np.pi
            else:
                return angle

        def larger_than(points, a, b):
            return points[1] >= (a*points[0] + b)

        def smaller_than(points, a, b):
            return points[1] < (a*points[0] + b)

        mat_con, num_con = measure.label(fb_boundary, return_num=True)
        total_points_list = [(mat_con==i).sum() for i in range(num_con)]
        background_num = total_points_list.index(max(total_points_list))
        ite_list = [0, 1, 2]
        del ite_list[background_num]

        # find xs, ys for the left and right boundaries
        if np.where(mat_con == ite_list[0])[1].mean() < np.where(mat_con == ite_list[1])[1].mean():
            left_y, left_x = np.where(mat_con == ite_list[0])
            right_y, right_x = np.where(mat_con == ite_list[1])
        else:
            left_y, left_x = np.where(mat_con == ite_list[1])
            right_y, right_x = np.where(mat_con == ite_list[0])
        a1, b1, r, p, e = stats.linregress(left_x, left_y)
        a2, b2, r, p, e = stats.linregress(right_x, right_y)

        # calculate the paras of two boundaries and their intersection
        x, y = lines_intersection(a1, b1, a2, b2)
        center = np.array([x, y]) + 0.5
        self.center = center
        self.a1 = a1
        self.a2 = a2

        # multiply masks through imaging data and z project slices

        for i in range(reg_tif.shape[-1]):
            projected[:,:,i] = reg_tif[:,:,i]*fbmask


        # calculate the iteration slopes for each ROI
        #dims = tif.shape[-2:]
        #points = np.array([(i, j) for j in np.arange(dims[0]) for i in np.arange(dims[1])]).T + .5
        #rois16 = np.zeros((wedge_num, dims[0], dims[1])).astype(bool)
        theta_left = wrap_rad_simple(np.arctan(self.a1))
        theta_right = wrap_rad_simple(np.arctan(self.a2))
        theta_list = np.arange(theta_left, theta_right-0.001, (theta_right-theta_left)/1.0/wedge_num)
        theta_list = np.concatenate((theta_list, [theta_right]))
        a_list = np.tan(theta_list)
        b_list = self.center[1] - a_list*self.center[0]

        dims = self.fb_boundary.shape
        points = np.array([(i, j) for j in np.arange(dims[0]) for i in np.arange(dims[1])]).T + .5
        rois16 = np.zeros((wedge_num, dims[0], dims[1])).astype(bool)

        pts = points-np.array([[self.center[0]],[self.center[1]]])
        pts = np.arctan(pts[1]/pts[0])
        for i in range(len(pts)):
            pts[i] = wrap_rad_simple(pts[i])
        for i in range(wedge_num):
            lower = theta_list[i]
            upper = theta_list[i+1]
            between = np.where((pts>lower) & (pts<upper), 1, 0)
            rois16[i, :, :] = between.reshape(dims[0], dims[1])

        # extract imaging values in each ROI
        num_frames = projected.shape[-1]
        wedges = np.zeros((num_frames, wedge_num))
        for i, roi in enumerate(rois16):
            wedge = projected * roi[:, :, None]
            active_pixels = wedge[:,:,0].size-np.count_nonzero(wedge[:,:,0]==0)
            temp = []
            for frame in range(num_frames):
                temp.append(np.nansum(wedge[:,:,frame])/active_pixels)
            wedges[:,i] = temp
        pd.DataFrame(wedges).to_csv(os.path.join(self.regfol, 'fb.csv'))

        return rois16, theta_list, wedges

    def norm_over_time(self, signal, mode='dF/F0'):
        if mode == 'zscore':
            newsignal = signal - signal.mean(axis=0)
            div = newsignal.std(axis=0)
            div[div == 0] = 1
            newsignal /= div
        elif mode == 'dF/F0':
            signal_sorted = np.sort(signal, axis=0)
            f0 = signal_sorted[:int(len(signal)*.05)].mean(axis=0)
            # std = signal.std(axis=0)
            # std[std==0] = np.nan
            if f0.ndim > 1:
                f0[f0 == 0] = np.nan
            newsignal = (signal - f0) / f0
        elif mode == 'linear':
            signal_sorted = np.sort(signal, axis=0)
            f0 = signal_sorted[:int(len(signal)*.05)].mean(axis=0)
            fm = signal_sorted[int(len(signal)*.97):].mean(axis=0)
            if f0.ndim > 1:
                f0[f0 == 0] = np.nan
                fm[fm == 0] = np.nan
            newsignal = (signal - f0) / (fm - f0)
        return newsignal

    def load_processed(self):
        post_processing_file = os.path.join(self.processedfol, 'postprocessing.h5')
        self.post_processing_file = post_processing_file

        pv2 = pd.read_hdf(post_processing_file, 'pv2').fillna(method='ffill')
        ft = pd.read_hdf(post_processing_file, 'ft').fillna(method='ffill')
        ft2 = pd.read_hdf(post_processing_file, 'ft2').fillna(method='ffill')
        ix = pd.read_hdf(post_processing_file, 'ix').fillna(method='ffill')
        self.pv2 = pv2
        self.ft = ft
        self.ft2 = ft2

    def get_layer_wedges(self, tag='fb'):
        if not hasattr(self, 'pv2'):
            self.load_processed()
        wedges = self.pv2.filter(regex=tag)
        wedges.fillna(method='ffill', inplace=True)
        wedges = wedges.apply(fn.lnorm).to_numpy()
        return wedges

    def phase_offset(self, data, heading = 'ft_heading'):
        """
        calculate the phase offset between tube and epg bumps
        """
        phase = self.get_centroidphase(data)
        if not hasattr(self, 'ft2'):
            self.load_processed()
        tube = self.ft2[heading].to_numpy()
        offset = fn.circmean(phase - tube)
        return offset

    def subtract_phase_offset(self, data):
        """
        subtract the phase offset from epg phase
        """
        phase = self.get_centroidphase(data)
        offset = self.phase_offset(data)
        offset_phase = fn.wrap(phase-offset)
        return offset_phase

    def get_centroidphase(self, data):
        """
        project eip intensity to eb, then average to get centroid position
        """
        phase = fn.centroid_weightedring(data)
        return phase

    def interpolate_wedges(self, fb, kind='cubic', dinterp=.1):
        from scipy.interpolate import interp1d
        period_inds = fb.shape[1]
        tlen, glomlen = fb.shape[:2]
        wrap = np.zeros((fb.shape[0], fb.shape[1]+2))
        wrap[:, 1:period_inds+1] = fb[:, :period_inds]
        wrap[:, [0, period_inds+1]] = fb[:, [period_inds-1, 0]]
        x = np.arange(0, period_inds+2, 1)
        f = interp1d(x, wrap, kind, axis=-1)
        x_interp = np.arange(.5, period_inds+.5, dinterp)
        row_interp = f(x_interp)

        x_interp -= .5
        return x_interp, row_interp

    def cancel_phase(self, fb, ppg=None, celltype=None, offset=np.pi):
        _, gc = self.interpolate_wedges(fb)
        phase = self.get_centroidphase(fb)
        period_inds = gc.shape[1]
        offset = int(offset * period_inds / (2*np.pi))
        gc_nophase = np.zeros_like(gc)
        for i in range(len(gc)):
            shift = int(np.round((-phase[i] + (np.pi)) * period_inds / (2*np.pi))) + offset
            row = np.roll(gc[i], shift)
            # row = np.zeros(gc.shape[1])
            # row = np.array([gc[i][(j - shift) % period_inds] for j in range(period_inds)])
            gc_nophase[i] = row
        return gc_nophase

    def continuous_to_glom(self, array, nglom):
        """
        takes a continuous fb and bins it into discreet glomeruli
        """
        grouped = np.zeros((len(array), nglom))
        for i, array in enumerate(np.split(array, nglom, axis=1)):
            grouped[:,i] = np.mean(array, axis=1)
        return grouped

    def bumps_in_out(self, fb_wedges):
        """
        cancel phase and return FB bumps in and out of odor
        """
        if not hasattr(self, 'pv2'):
            self.load_processed()
        df = self.ft2
        ix_in = df[df.instrip==1.0].index.to_list()
        print(len(ix_in))
        ix_out = df[df.instrip==0.0].index.to_list()
        print(len(ix_out))
        gc_nophase = self.cancel_phase(fb_wedges)

        bumps_in = gc_nophase[ix_in]
        bumps_out = gc_nophase[ix_out]

        bumps_in = self.continuous_to_glom(bumps_in, 16)
        bumps_out = self.continuous_to_glom(bumps_out, 16)

        return bumps_in, bumps_out

    def bumps_moving_still(self, fb_wedges):
        """
        cancel phase and return FB bumps when the fly is moving vs still
        """
        if not hasattr(self, 'pv2'):
            self.load_processed()
        _,df = fn.find_stops(self.ft2)
        ix_still = df[df.stop.isna()].index.to_list()
        #print(len(ix_still))
        ix_moving = df[df.stop==1.0].index.to_list()
        #print(len(ix_moving))
        gc_nophase = self.cancel_phase(fb_wedges)

        bumps_still = gc_nophase[ix_still]
        bumps_moving = gc_nophase[ix_moving]

        bumps_still = self.continuous_to_glom(bumps_still, 16)
        bumps_moving = self.continuous_to_glom(bumps_moving, 16)

        return bumps_still, bumps_moving

class FSB2:
    """
    FSB object for creating rois from individual slices, not from a projection
    """
    def __init__(self, name, folderloc, z_slices):
        self.name = name
        self.folderloc = folderloc # folder of fly
        self.datafol = os.path.join(self.folderloc, 'data') # data folder contains imaging folder, .log, .dat,
        self.regfol = os.path.join(self.folderloc, 'registered') # register folder contains registered .tiffs
        self.processedfol = os.path.join(self.folderloc, 'processed')
        self.zslices = z_slices #index of slices for FB
        self.nwedges = 16

    def open_mask(self, mask_name):
        for file in os.scandir(self.regfol):
            if file.name.endswith(mask_name+'.tif'):
                mask_file = file
        # open mask tif
        fb_mask = io.imread(mask_file.path, plugin='tifffile')
        # little hacky, sometimes axis are read in differently my io.imread, correct is number of frames is axis 0
        num_frames = min(fb_mask.shape)
        if fb_mask.shape[0]==num_frames:
            fb_mask = np.moveaxis(fb_mask, 0, -1)
        fb_mask = np.rot90(fb_mask, axes=(1,0))
        return fb_mask

    def open_boundary(self):
        for file in os.scandir(self.regfol):
            if file.name.endswith('boundary.tif'):
                boundary_file = file
        fb_boundary = io.imread(boundary_file.path, plugin='tifffile')
        fb_boundary = np.rot90(fb_boundary, axes=(1,0))
        return fb_boundary

    def open_slice(self, slice):
        for file in os.scandir(self.regfol):
            if file.name.endswith('slice'+str(slice)+'.tif') and not file.name.startswith('._'):
                registered_file = file
        slice = io.imread(registered_file.path)

        #transpose and flip to have (x,y,t) and EB in correct orientation
        slice = np.moveaxis(slice, [0,1], [-1,-2])
        slice = np.fliplr(slice)
        return slice

    def get_wedges(self, show_wedges=True):
        # load the boundary lines that define the FB
        self.fb_boundary = self.open_boundary()
        wedge_num = 16


        def lines_intersection(a1, b1, a2, b2):
            return (b2-b1)/(a1-a2), (a1*b2-a2*b1)/(a1-a2)

        def wrap_rad_simple(angle):
            if angle < 0:
                return angle + np.pi
            else:
                return angle

        # find connected pixels
        mat_con, num_con = measure.label(self.fb_boundary, return_num=True)
        total_points_list = [(mat_con==i).sum() for i in range(num_con)]
        background_num = total_points_list.index(max(total_points_list))
        ite_list = [0, 1, 2]
        del ite_list[background_num]

        # find xs, ys for the left and right boundaries
        if np.where(mat_con == ite_list[0])[1].mean() < np.where(mat_con == ite_list[1])[1].mean():
            left_y, left_x = np.where(mat_con == ite_list[0])
            right_y, right_x = np.where(mat_con == ite_list[1])
        else:
            left_y, left_x = np.where(mat_con == ite_list[1])
            right_y, right_x = np.where(mat_con == ite_list[0])
        a1, b1, r, p, e = stats.linregress(left_x, left_y)
        a2, b2, r, p, e = stats.linregress(right_x, right_y)

        # calculate the paras of two boundaries and their intersection
        x, y = lines_intersection(a1, b1, a2, b2)
        center = np.array([x, y]) + 0.5
        self.center = center
        self.a1 = a1
        self.a2 = a2

        # find the angles for each wedge
        print(a1, a2)
        theta_left = wrap_rad_simple(np.arctan(self.a1))
        theta_right = wrap_rad_simple(np.arctan(self.a2))
        theta_list = np.arange(theta_left, theta_right-0.001, (theta_right-theta_left)/1.0/wedge_num)
        theta_list = np.concatenate((theta_list, [theta_right]))

        # set up arrays to figure out which pixel belongs to each wedge
        dims = self.fb_boundary.shape
        points = np.array([(i, j) for j in np.arange(dims[0]) for i in np.arange(dims[1])]).T + .5
        rois16 = np.zeros((wedge_num, dims[0], dims[1])).astype(bool)

        #
        pts = points-np.array([[self.center[0]],[self.center[1]]])
        pts = np.arctan(pts[1]/pts[0])
        for i in range(len(pts)):
            pts[i] = wrap_rad_simple(pts[i])
        for i in range(wedge_num):
            lower = theta_list[i]
            upper = theta_list[i+1]
            between = np.where((pts>lower) & (pts<upper), 1, 0)
            rois16[i, :, :] = between.reshape(dims[0], dims[1])

        show_wedges = True
        if show_wedges:
            fig, axs = plt.subplots(4,4)
            for i, ax in enumerate(axs.flat):
                ax.imshow(rois16[i])

        return rois16

    def get_wedge_rois(self, mask_name):
        import seaborn as sns
        rois16 = self.get_wedges()
        wedge_num = 16

        # open masks
        all_masks = self.open_mask(mask_name)
        if len(all_masks.shape) == 2:
            all_masks = np.reshape(all_masks, all_masks.shape + (1,))

        for m, slice in enumerate(self.zslices):
            # open slice
            slice = self.open_slice(slice)
            projected = np.zeros(slice.shape)

            # array to hold wedge ROIs from different slices

            if m==0:
                wedges_all = np.zeros((slice.shape[2], wedge_num, all_masks.shape[2]))

            # select corresponding mask
            mask = all_masks[:,:,m]

            # project roi through slice stack
            for i in range(slice.shape[-1]):
                projected[:,:,i] = slice[:,:,i]*mask
            fig, axs = plt.subplots(1,1)
            axs.imshow(np.mean(projected, axis=2))

            # extract imaging values in each wedge ROI
            num_frames = projected.shape[-1]
            wedges = np.zeros((num_frames, wedge_num))
            for i, roi in enumerate(rois16):
                wedge = projected * roi[:, :, None]
                active_pixels = wedge[:,:,0].size-np.count_nonzero(wedge[:,:,0]==0)
                temp = []
                for frame in range(num_frames):
                    temp.append(np.nansum(wedge[:,:,frame])/active_pixels)
                wedges[:,i] = temp
            # if there is an extra frame in this slice, remove it
            wedges = wedges[range(wedges_all.shape[0]),:]
            wedges_all[:,:,m] = wedges
        wedges_all = np.nansum(wedges_all, axis=2)
        wedges_all = pd.DataFrame(wedges_all)
        #wedges_all = wedges_all.apply(fn.dff).to_numpy()
        fig, axs = plt.subplots(1,1)
        sns.heatmap(wedges_all.apply(fn.dff).to_numpy(), cmap='Blues', ax=axs)

        pd.DataFrame(wedges_all).to_csv(os.path.join(self.regfol, 'fb'+mask_name+'.csv'))

        return wedges_all

    def load_processed(self):
        post_processing_file = os.path.join(self.processedfol, 'postprocessing.h5')
        self.post_processing_file = post_processing_file

        pv2 = pd.read_hdf(post_processing_file, 'pv2').fillna(method='ffill')
        ft = pd.read_hdf(post_processing_file, 'ft').fillna(method='ffill')
        ft2 = pd.read_hdf(post_processing_file, 'ft2').fillna(method='ffill')
        ix = pd.read_hdf(post_processing_file, 'ix').fillna(method='ffill')
        self.pv2 = pv2
        self.ft = ft
        self.ft2 = ft2

    def get_layer_wedges(self, tag='lower'):
        #print("getting FB layer wedges")
        tag = tag.lower()
        if not hasattr(self, 'pv2'):
            self.load_processed()
        wedges = self.pv2.filter(regex=tag.lower())
        wedges.fillna(method='ffill', inplace=True)
        wedges = wedges.apply(fn.lnorm).to_numpy()
        return wedges

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


    def phase_offset(self, data, heading = 'ft_heading'):
        """
        calculate the phase offset between tube and epg bumps
        """
        phase = self.get_centroidphase(data)
        if not hasattr(self, 'ft2'):
            self.load_processed()
        tube = self.ft2[heading].to_numpy()
        offset = fn.circmean(phase - tube)
        return offset

    def subtract_phase_offset(self, data):
        """
        subtract the phase offset from epg phase
        """
        phase = self.get_centroidphase(data)
        offset = self.phase_offset(data)
        offset_phase = fn.wrap(phase-offset)
        return offset_phase

    def get_centroidphase(self, data):
        """
        project eip intensity to eb, then average to get centroid position
        """
        phase = fn.centroid_weightedring(data)
        phase = fn.circ_moving_average(phase)
        return phase

    def interpolate_wedges(self, fb, kind='cubic', dinterp=.1):
        from scipy.interpolate import interp1d
        period_inds = fb.shape[1]
        tlen, glomlen = fb.shape[:2]
        wrap = np.zeros((fb.shape[0], fb.shape[1]+2))
        wrap[:, 1:period_inds+1] = fb[:, :period_inds]
        wrap[:, [0, period_inds+1]] = fb[:, [period_inds-1, 0]]
        x = np.arange(0, period_inds+2, 1)
        f = interp1d(x, wrap, kind, axis=-1)
        x_interp = np.arange(.5, period_inds+.5, dinterp)
        row_interp = f(x_interp)

        x_interp -= .5
        return x_interp, row_interp

    def cancel_phase(self, fb, ppg=None, celltype=None, offset=np.pi):
        _, gc = self.interpolate_wedges(fb)
        phase = self.get_centroidphase(fb)
        period_inds = gc.shape[1]
        offset = int(offset * period_inds / (2*np.pi))
        gc_nophase = np.zeros_like(gc)
        for i in range(len(gc)):
            shift = int(np.round((-phase[i] + (np.pi)) * period_inds / (2*np.pi))) + offset
            row = np.roll(gc[i], shift)
            # row = np.zeros(gc.shape[1])
            # row = np.array([gc[i][(j - shift) % period_inds] for j in range(period_inds)])
            gc_nophase[i] = row
        return gc_nophase

    def continuous_to_glom(self, array, nglom):
        """
        takes a continuous fb and bins it into discreet glomeruli
        """
        grouped = np.zeros((len(array), nglom))
        for i, array in enumerate(np.split(array, nglom, axis=1)):
            grouped[:,i] = np.mean(array, axis=1)
        return grouped

    def bumps_in_out(self, fb_wedges):
        """
        cancel phase and return FB bumps in and out of odor
        """
        if not hasattr(self, 'pv2'):
            self.load_processed()
        df = self.ft2
        ix_in = df[df.instrip==1.0].index.to_list()
        ix_out = df[df.instrip==0.0].index.to_list()

        gc_nophase = self.cancel_phase(fb_wedges)

        bumps_in = gc_nophase[ix_in]
        bumps_out = gc_nophase[ix_out]

        bumps_in = self.continuous_to_glom(bumps_in, 16)
        bumps_out = self.continuous_to_glom(bumps_out, 16)

        return bumps_in, bumps_out

    def bumps_air_before_after(self, fb_wedges):
        """
        find the bumps in clean air before the animal starts edge tracking
        and after the animal starts edge tracking
        """
        if not hasattr(self, 'pv2'):
            self.load_processed()
        df = self.ft2

        ix_in = df[df.instrip==1.0].index.to_numpy()
        odor_on_ix = ix_in[0] # index where the odor turns on
        ix_out = df[df.instrip==0.0].index.to_numpy()
        ix = np.argmin(np.abs(ix_out-odor_on_ix)) # index in ix_out where the pre odor time ends


        gc_nophase = self.cancel_phase(fb_wedges)

        bumps_before = gc_nophase[ix_out[0:ix]]
        bumps_after = gc_nophase[ix_out[ix:]]

        bumps_before = self.continuous_to_glom(bumps_before, 16)
        bumps_after = self.continuous_to_glom(bumps_after, 16)

        return bumps_before, bumps_after

    def bumps_moving_still(self, fb_wedges):
        """
        cancel phase and return FB bumps when the fly is moving vs still
        """
        if not hasattr(self, 'pv2'):
            self.load_processed()
        df = self.ft2
        _,df = fn.find_stops(df)
        #print(len(ix_still))
        ix_moving = df[df.stop==1.0].index.to_list()
        ix_still = df[df.stop!=1.0].index.to_list()
        #print(len(ix_moving))
        gc_nophase = self.cancel_phase(fb_wedges)

        bumps_still = gc_nophase[ix_still]
        bumps_moving = gc_nophase[ix_moving]

        bumps_still = self.continuous_to_glom(bumps_still, 16)
        bumps_moving = self.continuous_to_glom(bumps_moving, 16)

        return bumps_still, bumps_moving

class EB:
    def __init__(self, name, folderloc, z_slices):
        self.name = name
        self.folderloc = folderloc # folder of fly
        self.datafol = os.path.join(self.folderloc, 'data') # data folder contains imaging folder, .log, .dat,
        self.regfol = os.path.join(self.folderloc, 'registered') # register folder contains registered .tiffs
        self.processedfol = os.path.join(self.folderloc, 'processed')
        self.zslices = z_slices #index of slices for FB
        self.nwedges = 16

    def load_processed(self):
        post_processing_file = os.path.join(self.processedfol, 'postprocessing.h5')
        self.post_processing_file = post_processing_file

        pv2 = pd.read_hdf(post_processing_file, 'pv2').fillna(method='ffill')
        ft = pd.read_hdf(post_processing_file, 'ft').fillna(method='ffill')
        ft2 = pd.read_hdf(post_processing_file, 'ft2').fillna(method='ffill')
        ix = pd.read_hdf(post_processing_file, 'ix').fillna(method='ffill')
        self.pv2 = pv2
        self.ft = ft
        self.ft2 = ft2

    def open_mask(self):
        for file in os.scandir(self.regfol):
            if file.name.endswith('ebmask.tif'):
                mask_file = file
        # open mask tif
        eb_mask = io.imread(mask_file.path)
        eb_mask = eb_mask.astype('float')
        eb_mask[eb_mask==0.0]=np.nan
        eb_mask = np.moveaxis(eb_mask, np.argmin(eb_mask.shape), -1)
        eb_mask = np.rot90(eb_mask, axes=(1,0))
        return eb_mask

    def open_center(self):
        for file in os.scandir(self.regfol):
            if file.name.endswith('center.tif'):
                center_file = file
        eb_center = io.imread(center_file.path)
        eb_center = np.rot90(eb_center, axes=(1,0))
        return eb_center

    def open_slice(self, slice):
        for file in os.scandir(self.regfol):
            if file.name.endswith('slice'+str(slice)+'.tif') and not file.name.startswith('._'):
                registered_file = file
        slice = io.imread(registered_file.path)

        #transpose and flip to have (x,y,t) and EB in correct orientation
        slice = np.moveaxis(slice, [0,1], [-1,-2])
        slice = np.fliplr(slice)
        return slice

    def get_wedges(self, show_wedges=False):

        eb_center = self.open_center()

        # define angles of wedges
        wedge_num = 16
        theta_left = -np.pi
        theta_right = np.pi
        theta_list = np.arange(theta_left, theta_right-0.001, (theta_right-theta_left)/1.0/wedge_num)
        theta_list = np.concatenate((theta_list, [theta_right]))

        # find center
        dims = eb_center.shape
        y, x = np.where(eb_center)
        center = (x[0], y[0])

        # create wedges
        points = np.array([(i, j) for j in np.arange(dims[0]) for i in np.arange(dims[1])]).T + .5
        rois16 = np.zeros((wedge_num, dims[0], dims[1])).astype(bool)

        pts = points-np.array([[center[0]],[center[1]]])
        pts = np.arctan2(pts[1],pts[0])
        pts = fn.wrap(pts+np.pi/2) # add pi/2 so wedge 1 is left of lower centerline and wedge 2 is right of lower centerline

        for i in range(wedge_num):
            lower = theta_list[i]
            upper = theta_list[i+1]
            between = np.where((pts>lower) & (pts<upper), 1, 0)
            rois16[i, :, :] = between.reshape(dims[0], dims[1])

        if show_wedges:
            fig, axs = plt.subplots(4,4)
            for i, ax in enumerate(axs.flat):
                ax.imshow(rois16[i])

        return rois16

    def get_wedge_rois(self):
        import seaborn as sns
        rois16 = self.get_wedges()
        wedge_num = 16

        # open masks
        all_masks = self.open_mask()

        for m, slice in enumerate(self.zslices):
            # open slice
            slice = self.open_slice(slice)
            projected = np.zeros(slice.shape)

            # array to hold wedge ROIs from different slices
            if m==0:
                wedges_all = np.zeros((slice.shape[2], wedge_num, all_masks.shape[2]))

            # select corresponding mask
            mask = all_masks[:,:,m]

            # project roi through slice stack
            for i in range(slice.shape[-1]):
                projected[:,:,i] = slice[:,:,i]*mask

            # extract imaging values in each wedge ROI
            num_frames = projected.shape[-1]
            wedges = np.zeros((num_frames, wedge_num))
            for i, roi in enumerate(rois16):
                wedge = projected * roi[:, :, None]
                active_pixels = wedge[:,:,0].size-np.count_nonzero(wedge[:,:,0]==0)
                temp = []
                for frame in range(num_frames):
                    temp.append(np.nansum(wedge[:,:,frame])/active_pixels)
                wedges[:,i] = temp
            wedges_all[:,:,m] = wedges
        wedges_all = np.nansum(wedges_all, axis=2)
        wedges_all = pd.DataFrame(wedges_all)
        wedges_all = wedges_all.apply(fn.lnorm).to_numpy()
        fig, axs = plt.subplots(1,1)
        sns.heatmap(wedges_all, cmap='Blues',vmax = 1.0, ax=axs)

        pd.DataFrame(wedges_all).to_csv(os.path.join(self.regfol, 'eb.csv'))

        return wedges_all

    def load_processed(self):
        post_processing_file = os.path.join(self.processedfol, 'postprocessing.h5')
        self.post_processing_file = post_processing_file

        pv2 = pd.read_hdf(post_processing_file, 'pv2').fillna(method='ffill')
        ft = pd.read_hdf(post_processing_file, 'ft').fillna(method='ffill')
        ft2 = pd.read_hdf(post_processing_file, 'ft2').fillna(method='ffill')
        ix = pd.read_hdf(post_processing_file, 'ix').fillna(method='ffill')
        self.pv2 = pv2
        self.ft = ft
        self.ft2 = ft2

    def get_layer_wedges(self, tag='eb'):
        #print("getting EB wedges")
        if not hasattr(self, 'pv2'):
            self.load_processed()
        wedges = self.pv2.filter(regex=tag)
        wedges.fillna(method='ffill', inplace=True)
        wedges = wedges.to_numpy()
        return wedges

    def phase_offset_old(self, data, heading = 'ft_heading'):
        """
        calculate the phase offset between tube and epg bumps
        """
        phase = self.get_centroidphase(data)
        if not hasattr(self, 'ft2'):
            self.load_processed()
        tube = self.ft2[heading].to_numpy()
        offset = fn.circmean(phase - tube)
        return offset

    def phase_offset(self, data, heading = 'ft_heading'):
        """
        calculate the phase offset between tube and epg bumps
        """
        phase = self.get_centroidphase(data)
        phase = fn.unwrap(phase)
        if not hasattr(self, 'ft2'):
            self.load_processed()
        tube = fn.unwrap(self.ft2[heading].to_numpy())
        offset = fn.circmean(fn.wrap(phase - tube))
        return offset

    def continuous_offset(self, data, heading = 'ft_heading'):
        """
        calculate the phase offset between tube and epg bumps
        """
        phase = self.get_centroidphase(data)
        if not hasattr(self, 'ft2'):
            self.load_processed()
        tube = self.ft2[heading].to_numpy()
        offset = fn.unwrap(phase) - fn.unwrap(tube)
        offset = pd.Series(offset)
        offset = fn.wrap(offset.rolling(20, min_periods=1).mean())
        return offset

    def subtract_phase_offset(self, data, heading='ft_heading'):
        """
        subtract the phase offset from epg phase
        """
        phase = self.get_centroidphase(data)
        offset = self.phase_offset(data, heading=heading)
        offset_phase = fn.wrap(phase-offset)
        return offset_phase

    def get_centroidphase(self, data):
        """
        project eip intensity to eb, then average to get centroid position
        """
        phase = fn.centroid_weightedring(data)
        return phase

    def interpolate_wedges(self, fb, kind='cubic', dinterp=.1):
        from scipy.interpolate import interp1d
        period_inds = fb.shape[1]
        tlen, glomlen = fb.shape[:2]
        wrap = np.zeros((fb.shape[0], fb.shape[1]+2))
        wrap[:, 1:period_inds+1] = fb[:, :period_inds]
        wrap[:, [0, period_inds+1]] = fb[:, [period_inds-1, 0]]
        x = np.arange(0, period_inds+2, 1)
        f = interp1d(x, wrap, kind, axis=-1)
        x_interp = np.arange(.5, period_inds+.5, dinterp)
        row_interp = f(x_interp)

        x_interp -= .5
        return x_interp, row_interp

    def cancel_phase(self, fb, ppg=None, celltype=None, offset=np.pi):
        _, gc = self.interpolate_wedges(fb)
        phase = self.get_centroidphase(fb)
        period_inds = gc.shape[1]
        offset = int(offset * period_inds / (2*np.pi))
        gc_nophase = np.zeros_like(gc)
        for i in range(len(gc)):
            shift = int(np.round((-phase[i] + (np.pi)) * period_inds / (2*np.pi))) + offset
            row = np.roll(gc[i], shift)
            # row = np.zeros(gc.shape[1])
            # row = np.array([gc[i][(j - shift) % period_inds] for j in range(period_inds)])
            gc_nophase[i] = row
        return gc_nophase

    def continuous_to_glom(self, array, nglom=16):
        """
        takes a continuous fb and bins it into discreet glomeruli
        """
        grouped = np.zeros((len(array), nglom))
        for i, array in enumerate(np.split(array, nglom, axis=1)):
            grouped[:,i] = np.mean(array, axis=1)
        return grouped

    def bumps_in_out(self, fb_wedges):
        """
        cancel phase and return FB bumps in and out of odor
        """
        if not hasattr(self, 'pv2'):
            self.load_processed()
        df = self.ft2
        ix_in = df[df.instrip==1.0].index.to_list()
        #print(len(ix_in))
        ix_out = df[df.instrip==0.0].index.to_list()
        #print(len(ix_out))
        gc_nophase = self.cancel_phase(fb_wedges)

        bumps_in = gc_nophase[ix_in]
        bumps_out = gc_nophase[ix_out]

        bumps_in = self.continuous_to_glom(bumps_in, nglom=16)
        bumps_out = self.continuous_to_glom(bumps_out, nglom=16)

        return bumps_in, bumps_out











    # def get_wedge_rois(self, center, nwedges, offset=0):
    #     thetas = np.arange(0, np.pi, np.pi/8)
    #     ms = np.tan(thetas)
    #     bs = center[1] - ms*center[0]
    #     dims = self.info['dimensions_pixels']
    #
    #     # use the center of each pixel as a threshold for segmenting into each wedge
    #     points = np.array([(i, j) for j in np.arange(dims[0]) for i in np.arange(dims[1])]).T + .5
    #
    #     def greater_than(points, m, b):
    #         return points[1] >= (m*points[0] + b)
    #
    #     def less_than(points, m, b):
    #         return points[1] < (m*points[0] + b)
    #
    #     rois16 = np.zeros((16, points.shape[1])).astype(bool)
    #     for i in range(16):
    #         j = i%8
    #         if i < 4:
    #             roi = greater_than(points, ms[j], bs[j]) & less_than(points, ms[j+1], bs[j+1])
    #         elif i == 4:
    #             roi = ((points[0]) < center[0]) & greater_than(points, ms[j+1], bs[j+1])
    #         elif i > 4 and i < 7:
    #             roi = less_than(points, ms[j], bs[j]) & greater_than(points, ms[j+1], bs[j+1])
    #         elif i == 7:
    #             roi = less_than(points, ms[j], bs[j]) & ((points[1]) >= center[1])
    #         elif i > 7 and i < 12:
    #             roi = less_than(points, ms[j], bs[j]) & greater_than(points, ms[j+1], bs[j+1])
    #         elif i == 12:
    #             roi = ( points[0] >= center[0] ) & less_than(points, ms[j+1], bs[j+1])
    #         elif i > 12:
    #             roi = greater_than(points, ms[j], bs[j]) & less_than(points, ms[(i+1)%8], bs[(i+1)%8])
    #         rois16[i] = roi
    #
    #     rois16 = rois16.reshape((16, dims[0], dims[1]))
    #     rois16 = np.flipud(rois16) # for some reason need this to match the tiffile orientation
    #     if nwedges == 8:
    #         rois8 = np.zeros((8, dims[0], dims[1]))
    #         for i in range(8):
    #             rois8[i] = rois16[2*i] | rois16[2*i-1]
    #         rois = np.roll(rois8[::-1], -1, axis=0)
    #         xedges = np.arange(0, 17, 2)
    #     elif nwedges == 16:
    #         rois = np.roll(rois16[::-1], -3, axis=0)
    #         xedges = np.arange(0, 17)
    #
    #     return xedges, rois

    # def get_wedges(self):
    #
    #     # load ellipsoid body mask, center position, and registered tif slices
    #     ebmask = self.open_mask()
    #     self.ebmask = ebmask
    #     eb_center = self.open_center()
    #     self.eb_center = eb_center
    #     reg_tif = self.open_registered()
    #     projected = np.zeros((reg_tif.shape[0], reg_tif.shape[1], reg_tif.shape[3]))
    #
    #     # find center of ellipsoid body
    #     x, y = np.where(self.eb_center.T)
    #     self.center = np.array((x[0], y[0]))+0.5
    #
    #     # multiply imaging stack by mask and z project
    #     for i in range(reg_tif.shape[-1]):
    #         projected[:,:,i] = np.nanmean(reg_tif[:,:,:,i]*ebmask, axis=2)
    #     return projected

class PB:
    def __init__(self, name, folderloc, **kwargs):
        self.name = name
        self.folderloc = folderloc # folder of fly
        self.datafol = os.path.join(self.folderloc, 'data') # data folder contains imaging folder, .log, .dat,
        self.regfol = os.path.join(self.folderloc, 'registered') # register folder contains registered .tiffs
        self.processedfol = os.path.join(self.folderloc, 'processed')

    def load_processed(self):
        post_processing_file = os.path.join(self.processedfol, 'postprocessing.h5')
        self.post_processing_file = post_processing_file

        pv2 = pd.read_hdf(post_processing_file, 'pv2').fillna(method='ffill')
        ft = pd.read_hdf(post_processing_file, 'ft').fillna(method='ffill')
        ft2 = pd.read_hdf(post_processing_file, 'ft2').fillna(method='ffill')
        ix = pd.read_hdf(post_processing_file, 'ix').fillna(method='ffill')
        self.pv2 = pv2
        self.ft = ft
        self.ft2 = ft2

    def get_gloms(self):
        if not hasattr(self, 'pv2'):
            self.load_processed()
        gloms = self.pv2.filter(regex='pb')
        gloms.fillna(method='ffill', inplace=True)
        gloms = gloms.apply(fn.lnorm).to_numpy()
        return gloms

    def get_phase(self):
        data = self.get_gloms()
        phase = fn.get_fftphase(data)
        phase = fn.circ_moving_average(phase, n=3)
        return -phase

    def interpolate_glomeruli(self):
        from scipy.interpolate import interp1d
        # load glomeruli, and add glomeruli 1 and 18 (not expressed in EPGs)
        g = self.get_gloms()
        gloms = np.zeros((len(g), 18))
        gloms.fill(np.nan)
        gloms[:,1:17] = g

        # interplolate glomeruli
        kind='cubic'
        dinterp = 0.1
        tlen, glomlen = gloms.shape[:2]
        row_interp = np.zeros((tlen, int(glomlen/dinterp)))
        nans, x = fn.nan_helper(np.nanmean(gloms, axis=0))
        wrap = np.zeros_like(gloms)
        wrap[:, 1:17] = gloms[:, 1:17]
        wrap[:, [0, 17]] = gloms[:, [8, 9]]
        x = np.arange(0, 18, 1)
        f = interp1d(x, wrap, kind, axis=-1)
        # f = np.interp(x, wrap, kind, axis=-1)
        # x_interp = np.arange(1, 17, dinterp)
        x_interp = np.arange(.5, 16.5, dinterp)     # modified by Cheng, 2020 Apr
        row_interp[:, int(1./dinterp) : int(17./dinterp)] = f(x_interp)

        fn.zero2nan(row_interp)
        x_interp = np.arange(0, 18, dinterp)

        return x_interp, row_interp

    def cancel_phase(self):
        _,gc = self.interpolate_glomeruli()
        phase = -self.get_phase()


        ppg=10
        period = 8 #if celltype=='PEN' else 9
        # period_inds = int(self.period * ppg)
        period_inds = int(period*ppg)
        #offset = int(offset * period_inds / 360)
        gc_nophase = np.zeros_like(gc)
        x = np.arange(0, 18, 1./ppg)

        for i in range(len(gc)):
            shift = int(np.round((phase[i] + np.pi) * period_inds / (2*np.pi)))
            row = np.zeros(len(x))
            row[:] = np.nan
            left_ind = (x < 9) & (x >= 1)
            right_ind = (x >= 9) & (x < 17)
            row[left_ind] = np.roll(gc[i, left_ind], shift)
            row[right_ind] = np.roll(gc[i, right_ind], shift)
            gc_nophase[i] = row
        return gc_nophase

    def bumps_in_out(self):
        """
        cancel phase and return PB bumps in and out of odor
        """
        if not hasattr(self, 'pv2'):
            self.load_processed()
        df = self.ft2
        ix_in = df[df.instrip==0.0].index.to_list()
        ix_out = df[df.instrip==1.0].index.to_list()
        gc_nophase = self.cancel_phase()

        bumps_in = gc_nophase[ix_in]
        bumps_out = gc_nophase[ix_out]

        def continuous_to_glom(array, nglom):
            """
            takes a continuous bridge and bins it into discreet glomeruli
            """
            grouped = np.zeros((len(array), nglom))
            for i, array in enumerate(np.split(array, nglom, axis=1)):
                grouped[:,i] = np.mean(array, axis=1)
            return grouped

        bumps_in = continuous_to_glom(bumps_in, 18)
        bumps_out = continuous_to_glom(bumps_out, 18)

        return bumps_in, bumps_out

    def phase_offset(self, heading = 'ft_heading'):
        """
        calculate the phase offset between tube and epg bumps
        """
        phase = self.get_phase()
        if not hasattr(self, 'ft2'):
            self.load_processed()
        tube = self.ft2[heading].to_numpy()
        offset = fn.circmean(phase - tube)
        return offset

    def subtract_phase_offset(self):
        """
        subtract the phase offset from epg phase
        """
        phase = self.get_phase()
        offset = self.phase_offset()
        offset_phase = fn.wrap(phase-offset)
        return offset_phase

    def save_slices(self, overwrite = True):
        """
        after images have been registered (through class fly), save them as a stack and a z projection
        """
        if overwrite or not os.path.exists(os.path.join(self.regfol, 'PB_stack.tif')):
            ch = []
            fls1 = glob.glob(os.path.join(self.regfol, '*slice*.tif'))
            for f in fls1:
                ch.append(io.imread(f))
            ch = np.array(ch, dtype='f')
            io.imsave(os.path.join(self.regfol,'PB_stack.tif'), ch, imagej=True)
            io.imsave(os.path.join(self.regfol,'PB_stack_max.tif'), np.max(ch, axis=0), imagej=True)

class PB2:
    def __init__(self, name, folderloc, z_slices):
        self.name = name
        self.folderloc = folderloc # folder of fly
        self.datafol = os.path.join(self.folderloc, 'data') # data folder contains imaging folder, .log, .dat,
        self.regfol = os.path.join(self.folderloc, 'registered') # register folder contains registered .tiffs
        self.processedfol = os.path.join(self.folderloc, 'processed')
        self.zslices = z_slices

    def open_mask(self):
        for file in os.scandir(self.regfol):
            if file.name.endswith('pbmask.tiff'):
                mask_file = file
        # open mask tif
        fb_mask = io.imread(mask_file.path, plugin='tifffile')
        # little hacky, sometimes axis are read in differently my io.imread, correct is number of frames is axis 0
        num_frames = min(fb_mask.shape)
        if fb_mask.shape[0]==num_frames:
            fb_mask = np.moveaxis(fb_mask, 0, -1)
        fb_mask = np.rot90(fb_mask, axes=(1,0))
        return fb_mask

    def open_slice(self, slice):
        for file in os.scandir(self.regfol):
            if file.name.endswith('slice'+str(slice)+'.tif') and not file.name.startswith('._'):
                registered_file = file
        slice = io.imread(registered_file.path)

        #transpose and flip to have (x,y,t) and EB in correct orientation
        slice = np.moveaxis(slice, [0,1], [-1,-2])
        slice = np.fliplr(slice)
        return slice

    def load_processed(self):
        post_processing_file = os.path.join(self.processedfol, 'postprocessing.h5')
        self.post_processing_file = post_processing_file

        pv2 = pd.read_hdf(post_processing_file, 'pv2').fillna(method='ffill')
        ft = pd.read_hdf(post_processing_file, 'ft').fillna(method='ffill')
        ft2 = pd.read_hdf(post_processing_file, 'ft2').fillna(method='ffill')
        ix = pd.read_hdf(post_processing_file, 'ix').fillna(method='ffill')
        self.pv2 = pv2
        self.ft = ft
        self.ft2 = ft2

    def get_gloms(self):
        import seaborn as sns
        glom_num = 16
        #glom_order = [2, 4, 6, 8, 10, 12, 14, 16, 1, 3, 5, 7, 9, 11, 13, 15]
        glom_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        # open masks
        all_masks = self.open_mask()
        if len(all_masks.shape) == 2:
            all_masks = np.reshape(all_masks, all_masks.shape + (1,))

        for m, slice in enumerate(self.zslices):
            # open slice
            slice = self.open_slice(slice)

            # array to hold wedge ROIs from different slices

            if m==0:
                gloms_all = np.zeros((slice.shape[2], glom_num, all_masks.shape[2]))

            # select corresponding mask
            mask = all_masks[:,:,m]

            # find glomeruli in that mask
            glom_id = np.unique(mask)
            glom_id = glom_id[glom_id!=0]
            print(glom_id)


            # extract imaging values in each glomerulus ROI
            num_frames = slice.shape[-1]
            gloms = np.zeros((num_frames, glom_num))
            for i, id in enumerate(glom_id):
                glom = (mask == id)
                # CD comment -  I don't understand why there is this subtraction of 1 in glom order
               #glom_ix = np.where(glom_order==(id-1)) 
                glom_ix = np.where(glom_order==id)
                print(id, glom_ix)
                projected = slice * glom[:, :, None]
                active_pixels = projected[:,:,0].size-np.count_nonzero(projected[:,:,0]==0)
                temp = []
                for frame in range(num_frames):
                    temp.append(np.nansum(projected[:,:,frame])/active_pixels)
                temp = np.array(temp)
                gloms[:,glom_ix[0]] = temp[:,None]
            # if there is an extra frame in this slice, remove it
            gloms = gloms[range(gloms_all.shape[0]),:]
            gloms_all[:,:,m] = gloms
        gloms_all = np.nansum(gloms_all, axis=2)
        gloms_all = pd.DataFrame(gloms_all)
        gloms_all = gloms_all.apply(fn.lnorm).to_numpy()
        fig, axs = plt.subplots(1,1)
        sns.heatmap(gloms_all, cmap='Blues',ax=axs,vmin=0.0,vmax = 1.1)
        fig.suptitle(self.name)

        pd.DataFrame(gloms_all).to_csv(os.path.join(self.regfol, 'pb.csv'))

        return gloms_all

    def get_layer_gloms(self, tag='pb'):
        #print("getting FB layer wedges")
        tag = tag.lower()
        if not hasattr(self, 'pv2'):
            self.load_processed()
        wedges = self.pv2.filter(regex=tag.lower())
        wedges.fillna(method='ffill', inplace=True)
        wedges = wedges.apply(fn.lnorm).to_numpy()
        return wedges

    def get_phase(self):
        data = self.get_layer_gloms()
        phase = fn.get_fftphase(data)
        #phase = fn.circ_moving_average(phase, n=3)
        return -phase

    def interpolate_glomeruli(self):
        from scipy.interpolate import interp1d
        # load glomeruli, and add glomeruli 1 and 18 (not expressed in EPGs)
        g = self.get_gloms()
        gloms = np.zeros((len(g), 18))
        gloms.fill(np.nan)
        gloms[:,1:17] = g

        # interplolate glomeruli
        kind='cubic'
        dinterp = 0.1
        tlen, glomlen = gloms.shape[:2]
        row_interp = np.zeros((tlen, int(glomlen/dinterp)))
        nans, x = fn.nan_helper(np.nanmean(gloms, axis=0))
        wrap = np.zeros_like(gloms)
        wrap[:, 1:17] = gloms[:, 1:17]
        wrap[:, [0, 17]] = gloms[:, [8, 9]]
        x = np.arange(0, 18, 1)
        f = interp1d(x, wrap, kind, axis=-1)
        # f = np.interp(x, wrap, kind, axis=-1)
        # x_interp = np.arange(1, 17, dinterp)
        x_interp = np.arange(.5, 16.5, dinterp)     # modified by Cheng, 2020 Apr
        row_interp[:, int(1./dinterp) : int(17./dinterp)] = f(x_interp)

        fn.zero2nan(row_interp)
        x_interp = np.arange(0, 18, dinterp)

        return x_interp, row_interp

    def cancel_phase(self):
        _,gc = self.interpolate_glomeruli()
        phase = -self.get_phase()


        ppg=10
        period = 8 #if celltype=='PEN' else 9
        # period_inds = int(self.period * ppg)
        period_inds = int(period*ppg)
        #offset = int(offset * period_inds / 360)
        gc_nophase = np.zeros_like(gc)
        x = np.arange(0, 18, 1./ppg)

        for i in range(len(gc)):
            shift = int(np.round((phase[i] + np.pi) * period_inds / (2*np.pi)))
            row = np.zeros(len(x))
            row[:] = np.nan
            left_ind = (x < 9) & (x >= 1)
            right_ind = (x >= 9) & (x < 17)
            row[left_ind] = np.roll(gc[i, left_ind], shift)
            row[right_ind] = np.roll(gc[i, right_ind], shift)
            gc_nophase[i] = row
        return gc_nophase

    def bumps_in_out(self):
        """
        cancel phase and return PB bumps in and out of odor
        """
        if not hasattr(self, 'pv2'):
            self.load_processed()
        df = self.ft2
        ix_in = df[df.instrip==0.0].index.to_list()
        ix_out = df[df.instrip==1.0].index.to_list()
        gc_nophase = self.cancel_phase()

        bumps_in = gc_nophase[ix_in]
        bumps_out = gc_nophase[ix_out]

        def continuous_to_glom(array, nglom):
            """
            takes a continuous bridge and bins it into discreet glomeruli
            """
            grouped = np.zeros((len(array), nglom))
            for i, array in enumerate(np.split(array, nglom, axis=1)):
                grouped[:,i] = np.mean(array, axis=1)
            return grouped

        bumps_in = continuous_to_glom(bumps_in, 18)
        bumps_out = continuous_to_glom(bumps_out, 18)

        return bumps_in, bumps_out

    def continuous_offset(self, heading = 'ft_heading'):
        """
        calculate the phase offset between tube and epg bumps
        """
        phase = self.get_phase()
        if not hasattr(self, 'ft2'):
            self.load_processed()
        tube = self.ft2[heading].to_numpy()
        offset = fn.unwrap(phase) - fn.unwrap(tube)
        offset = pd.Series(offset)
        offset = fn.wrap(offset.rolling(100, min_periods=1).mean())
        return offset

    def phase_offset(self, heading = 'ft_heading'):
        """
        calculate the phase offset between tube and epg bumps
        """
        phase = self.get_phase()
        if not hasattr(self, 'ft2'):
            self.load_processed()
        tube = self.ft2[heading].to_numpy()
        offset = fn.circmean(phase - tube)
        return offset

    def subtract_phase_offset(self):
        """
        subtract the phase offset from epg phase
        """
        phase = self.get_phase()
        offset = self.phase_offset()
        offset = self.continuous_offset()
        offset_phase = fn.wrap(phase-offset)
        return offset_phase

    def save_slices(self, overwrite = True):
        """
        after images have been registered (through class fly), save them as a stack and a z projection
        """
        if overwrite or not os.path.exists(os.path.join(self.regfol, 'PB_stack.tif')):
            ch = []
            fls1 = glob.glob(os.path.join(self.regfol, '*slice*.tif'))
            for f in fls1:
                ch.append(io.imread(f))
            ch = np.array(ch, dtype='f')
            io.imsave(os.path.join(self.regfol,'PB_stack.tif'), ch, imagej=True)
            io.imsave(os.path.join(self.regfol,'PB_stack_max.tif'), np.max(ch, axis=0), imagej=True)

class MB:
    def __init__(self, name, folderloc, z_slices):
        self.name = name
        self.folderloc = folderloc # folder of fly
        self.datafol = os.path.join(self.folderloc, 'data') # data folder contains imaging folder, .log, .dat,
        self.regfol = os.path.join(self.folderloc, 'registered') # register folder contains registered .tiffs
        self.processedfol = os.path.join(self.folderloc, 'processed')
        self.zslices = z_slices

    def open_mask(self):
        for file in os.scandir(self.regfol):
            if file.name.endswith('mbmask.tif'):
                mask_file = file
        # open mask tif
        mb_mask = io.imread(mask_file.path, plugin='tifffile')
        # little hacky, sometimes axis are read in differently my io.imread, correct is number of frames is axis 0
        num_frames = min(mb_mask.shape)
        if mb_mask.shape[0]==num_frames:
            mb_mask = np.moveaxis(fb_mask, 0, -1)
        mb_mask = np.rot90(mb_mask, axes=(1,0))
        return mb_mask

    def open_slice(self, slice):
        for file in os.scandir(self.regfol):
            if file.name.endswith('slice'+str(slice)+'.tif') and not file.name.startswith('._'):
                registered_file = file
        slice = io.imread(registered_file.path)

        #transpose and flip to have (x,y,t) and EB in correct orientation
        slice = np.moveaxis(slice, [0,1], [-1,-2])
        slice = np.fliplr(slice)
        return slice

    def load_processed(self):
        post_processing_file = os.path.join(self.processedfol, 'postprocessing.h5')
        self.post_processing_file = post_processing_file

        pv2 = pd.read_hdf(post_processing_file, 'pv2').fillna(method='ffill')
        ft = pd.read_hdf(post_processing_file, 'ft').fillna(method='ffill')
        ft2 = pd.read_hdf(post_processing_file, 'ft2').fillna(method='ffill')
        ix = pd.read_hdf(post_processing_file, 'ix').fillna(method='ffill')
        self.pv2 = pv2
        self.ft = ft
        self.ft2 = ft2

    def get_comps(self):

        @jit(nopython=True)
        def extract_values(num_frames, active_pixels, projected):
            temp = []
            for frame in np.arange(num_frames):
                temp.append(np.nansum(projected[:,:,frame])/active_pixels)
            return temp
        comp_ids = ['g2', 'g3', 'g4', 'g5', 'bp1', 'bp2', 'b1', 'b2']
        comp_num = len(comp_ids)
        all_masks = self.open_mask()
        if len(all_masks.shape) == 2:
            all_masks = np.reshape(all_masks, all_masks.shape + (1,))

        for m, slice in enumerate(self.zslices):
            # open slice
            slice = self.open_slice(slice)

            # array to hold wedge ROIs from different slices

            if m==0:
                comps_all = np.zeros((slice.shape[2], comp_num, all_masks.shape[2]))

            # select corresponding mask
            mask = all_masks[:,:,m]

            # find compartments in that mask
            comp_id = np.unique(mask)
            comp_id = comp_id[comp_id!=0]


            # extract imaging values in each compartment ROI
            num_frames = slice.shape[-1]
            comps = np.zeros((num_frames, comp_num))

            for i, id in enumerate(comp_id):
                comp = (mask == id)
                comp_ix = id-3
        #         print(id, glom_ix)
                projected = slice * comp[:, :, None]
                active_pixels = projected[:,:,0].size-np.count_nonzero(projected[:,:,0]==0)
                temp = extract_values(num_frames, active_pixels, projected)
                temp = np.array(temp)
                comps[:,comp_ix] = temp[:]

            # if there is an extra frame in this slice, remove it

            comps = comps[range(comps_all.shape[0]),:]
            comps_all[:,:,m] = comps
        comps_all = np.nansum(comps_all, axis=2)
        comps_all = pd.DataFrame(comps_all)
        comps_all.columns=comp_ids


        comps_all.to_csv(os.path.join(self.regfol, 'mb.csv'))

        return comps_all

    def load_comps(self):
        #print("getting FB layer wedges")
        tag='mb'
        if not hasattr(self, 'pv2'):
            self.load_processed()
        comps = self.pv2.filter(regex=tag.lower())
        comps.fillna(method='ffill', inplace=True)
        comps = comps.apply(fn.znorm)
        return comps
