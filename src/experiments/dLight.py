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
from scipy import interpolate, stats
#from numba import jit
import time
matplotlib.rcParams['pdf.fonttype'] = 42

importlib.reload(dr)
importlib.reload(pl)
importlib.reload(fn)
importlib.reload(im)

class dLight:
    def __init__(self, drive='Andy Backup'):
        d = dr.drive_hookup()

        # load google sheet with experiment information

        self.sheet_id = '17h8BCU_TN8d8pWge0FZdlTVb_4-Zto2L1oLkPqliZ2s'
        if drive == 'Andy Backup':
            self.datafol = "/Volumes/Andy Backup/2P/KCdLight/dlight"
            self.picklefol = "/Volumes/Andy Backup/2P/KCdLight/dlight/pickles"
            self.figurefol = "/Volumes/Andy Backup/2P/KCdLight/dlight/figures"
        elif drive == 'ANDY_2':
            self.datafol = "/Volumes/ANDY_2/dlight"
            self.figurefol = "/Volumes/ANDY_2/dlight/figures"
            self.processedfol = "/Volumes/ANDY_2/dlight/processed"

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
                # do pre and post processing
                name = d['folder']
                folderloc = os.path.join(self.datafol, name)
                kwargs = d
                ex = im.fly(name, folderloc, **kwargs)

                ex.save_preprocessing()
                ex.save_postprocessing()

    # def plot_trajectories(self):
    #     for i, row in self.sheet.iterrows():
    #         d = row.to_dict()
    #         #if d['analyze']:
    #         name = d['folder']
    #         folderloc = os.path.join(self.datafol, name)
    #         print(name, folderloc)
    #         ex = im.fly(name, folderloc, **d)
    #         df = ex.read_log()
    #         fig, axs = plt.subplots(1,1)
    #         axs.plot(df.ft_posx, df.ft_posy)
    #         fig.suptitle(name)

    def save_comps(self):
        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            name = d['folder']
            if d['analyze']:
                folderloc = os.path.join(self.datafol, name)
                print(name, folderloc)
                string = d['mb_slices']
                z_slices = [int(s) for s in string.split(',')]
                mb = im.MB(name, folderloc, z_slices)
                _ = mb.get_comps()

    def plot_all_traces(self):
        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            datafol = self.datafol
            name = d['folder']
            if d['analyze']:
                ex = dlight_ex(d,datafol)
                data = ex.plot_compartments()

    def plot_G4in_timeout(self):
        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            datafol = self.datafol
            name = d['folder']
            print(name)
            if d['analyze']:
                fig, axs = plt.subplots(1,1)
                axs.set_yscale('log')
                ex = dlight_ex(d,datafol)
                data = ex.load_split_data()
                di = data['di']
                do = data['do']
                times = []
                for key in list(do.keys())[1:]:
                    times.append(len(do[key]))
                times = stats.zscore(times)
                timesdict ={}
                for i,key in enumerate(list(do.keys())[1:]):
                    timesdict[key]=times[i]

                for key in list(di.keys())[1:]:
                    if key-2 in list(di.keys()) and key+1 in list(do.keys()):
                        df_in1 = di[key-2]
                        df_in2 = di[key]
                        df_out = do[key+1]
                        delta = np.mean([np.max(df_in2.g4_mb),np.max(df_in1.g4_mb)])
                        axs.plot(np.max(df_in1.g4_mb), timesdict[key+1], '.')

    def copy_processed_to_lacie(self):
        from distutils.dir_util import copy_tree
        datafol = self.datafol
        destination = '/Volumes/LACIE/edge-tracking/data/dlight'
        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            if d['analyze']:
                name = d['folder']
                source_folder = os.path.join(datafol, name)
                dest_folder = os.path.join(destination, name)
                dest_processedfol = os.path.join(dest_folder, 'processed')
                source_processedfol = os.path.join(source_folder, 'processed')
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
                if not os.path.exists(dest_processedfol):
                    os.makedirs(dest_processedfol)
                    copy_tree(source_processedfol, dest_processedfol)

    def example_trajectory(self):
        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            if d['folder'] == '20210224_KCdLight_Fly1-001':
                name = d['folder']
                folderloc = os.path.join(self.datafol, name)
                kwargs = d
                ex = im.fly(name, folderloc, **kwargs)
                pv2, _, ft2, _ = ex.load_postprocessing()

                # make a folder to store figures
                figure_folder = os.path.join(ex.folderloc, 'figures')
                if not os.path.exists(figure_folder):
                    os.makedirs(figure_folder)

                # plot trajectory
                fig, axs = plt.subplots(1,1)
                axs = pl.plot_vertical_edges(ft2,axs,width=10)
                axs = pl.plot_trajectory(ft2,axs)
                axs = pl.plot_trajectory_odor(ft2, axs)
                #axs.axis('equal')
                fig.savefig(os.path.join(figure_folder, 'trajectory.pdf'))

    def calculate_triggered_averages(self):

        # make a dictionary for storing all the transitions
        d_in = {
        'g2_mb':[],
        'g3_mb':[],
        'g4_mb':[],
        'g5_mb':[],
        'b2_mb':[],
        'b1_mb':[],
        }

        d_out = {
        'g2_mb':[],
        'g3_mb':[],
        'g4_mb':[],
        'g5_mb':[],
        'b2_mb':[],
        'b1_mb':[],
        }

        # number of points to plot on either side of the transition
        buf_pts = 20 # 2 seconds

        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            datafol = self.datafol
            name = d['folder']
            print(name)
            if d['analyze']:
                ex = dlight_ex(d,datafol)

                # get compartments, exclude ones for which no ROI was drawn
                comps = ex.mb.load_comps()
                comps = comps.loc[:, ~comps.isna().any()]
                num_comps = len(comps.columns)

                # load bouts and find entry and exit points
                bouts = ex.load_split_data(overwrite=False)
                di = bouts['di']
                do = bouts['do']
                ix_in = []
                ix_out = []
                for key in list(di.keys()):
                    ix_in.append(di[key].index[0])
                for key in list(do.keys())[1:]:
                    ix_out.append(do[key].index[0])

                # iterate through each compartment and calculate in and out transitions
                for comp in comps:
                    trace = comps[comp].to_numpy()

                    # hard to distinguish between b2 and bp2, so these will be grouped together
                    if ('b2' in comp) or ('bp2' in comp):
                        comp = 'b2_mb'
                    elif ('b1' in comp) or ('bp1' in comp):
                        comp = 'b1_mb'

                    temp = []
                    for ix in ix_in:
                        ixs = np.arange(ix-buf_pts, ix+buf_pts)
                        try:
                            temp.append(trace[ixs])
                        except:
                            print('transition too close to start/end trial')
                    d_in[comp].append(temp)

                    temp = []
                    for ix in ix_out:
                        ixs = np.arange(ix-buf_pts, ix+buf_pts)
                        try:
                            temp.append(trace[ixs])
                        except:
                            print('transition too close to start/end trial')
                    d_out[comp].append(temp)
        savename = os.path.join(self.processedfol, 'triggered.p')
        fn.save_obj([d_out, d_in], savename)

        return d_out, d_in

    def load_triggered_averages(self, overwrite=False):
        savename = os.path.join(self.processedfol, 'triggered.p')
        if not os.path.exists(savename) or overwrite:
            d_out,d_in = self.calculate_triggered_averages()
        else:
            [d_out, d_in] = fn.load_obj(savename)
        return d_out, d_in

    def plot_triggered_averages(self):
        d_out, d_in = self.load_triggered_averages()
        fig, axs = plt.subplots(2,6, sharey=True)
        t = (np.arange(len(ixs))-buf_pts)/10

        # iterate through in to out transitions
        for i, key in enumerate(list(d_out.keys())):
            all_outs = []
            d_temp = d_out[key]
            for j in np.arange(len(d_temp)):
                all_outs.append(np.mean(d_temp[j], axis=0))
                axs[0,i].plot(t, np.mean(d_temp[j], axis=0), color=pl.mb_color[key], alpha=0.2)
            m = np.mean(all_outs, axis=0)
            error = stats.sem(all_outs, axis=0)
            axs[0,i].plot(t, m, color = 'k', linewidth=2)
            axs[0,i].fill_between(t, m+error, m-error, color=pl.mb_color[key])
            axs[0,i].spines['top'].set_visible(False)
            axs[0,i].spines['right'].set_visible(False)
            axs[0,i].plot([0,0], [-0.5,8], '--', color='k')
            if i>0:
                axs[0,i].axis('off')

        # iterate through out to in transitions
        for i, key in enumerate(list(d_in.keys())):
            all_ins = []
            d_temp = d_in[key]
            for j in np.arange(len(d_temp)):
                all_ins.append(np.mean(d_temp[j], axis=0))
                axs[1,i].plot(t, np.mean(d_temp[j], axis=0), color=pl.mb_color[key], alpha=0.2)
            m = np.mean(all_ins, axis=0)
            error = stats.sem(all_ins, axis=0)
            axs[1,i].plot(t, m, color = 'k', linewidth=2)
            axs[1,i].fill_between(t, m+error, m-error, color=pl.mb_color[key])
            axs[1,i].spines['top'].set_visible(False)
            axs[1,i].spines['right'].set_visible(False)
            axs[1,i].plot([0,0], [-0.5,8], '--', color='k')
            if i>0:
                axs[1,i].axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'triggered_averages.pdf'))

    def plot_trajectories(self):
        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            datafol = self.datafol
            name = d['folder']
            print(name)
            if d['analyze']:
                ex = dlight_ex(d,datafol)
                data = ex.load_split_data(overwrite=False)
                df = data['data']
                fig, axs = plt.subplots(1,1)
                pl.plot_trajectory(df, axs)
                pl.plot_trajectory_odor(df, axs)
                fig.savefig(os.path.join(self.figurefol,name+'2.pdf'),transparent=True)

    def plot_averages(self, pts = 10000):
        same_side = 0
        cross_over = 0
        animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out = [],[],[],[]
        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            datafol = self.datafol
            name = d['folder']
            print(name)
            if d['analyze']:
                avg_x_in, avg_y_in, avg_x_out, avg_y_out=[],[],[],[]
                ex = dlight_ex(d,datafol)
                data = ex.load_split_data(overwrite=False)
                df = data['data']
                di = data['di']
                do = data['do']
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
                        if np.abs(x[-1]-x[0])<1:
                            count+=1
                            if np.mean(x)>0: # align insides to the right and outsides to the left
                                x = -x
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
                        if np.abs(x[-1]-x[0])<1:
                            same_side+=1
                            if np.mean(x)<0: # align insides to the right and outsides to the left
                                x = -x
                            t = np.arange(len(x))
                            t_common = np.linspace(t[0], t[-1], pts)
                            fx = interpolate.interp1d(t, x)
                            fy = interpolate.interp1d(t, y)
                            #axs.plot(fx(t_common), fy(t_common))
                            avg_x_in.append(fx(t_common))
                            avg_y_in.append(fy(t_common))
                        else:
                            cross_over+=1

                if count>6: # condition: each trajectory needs more than three outside trajectories
                    print(name)
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
                    fig.suptitle(name)
                    fig.savefig(os.path.join(self.figurefol, name+'.pdf'), transparent=True)


        # save the average trajectories
        fn.save_obj([animal_avg_x_in, animal_avg_y_in, animal_avg_x_out, animal_avg_y_out], os.path.join(self.picklefol, 'average_trajectories.p'))


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

        axs.axis('equal')
        fig.savefig(os.path.join(self.figurefol, 'all_averages.pdf'), transparent = True)
        print(same_side, cross_over, same_side/(same_side+cross_over))
        return axs




class dlight_ex():
    """
    class for manipulating data and plotting examples from individual dlight examples
    """
    def __init__(self, dict, datafol):
        for key in list(dict.keys()):
            setattr(self,key, dict[key])
        string = dict['mb_slices']
        setattr(self, 'mb_slices', [int(s) for s in string.split(',')])

        self.datafol = os.path.join(datafol, self.folder)
        self.figure_folder = os.path.join(self.datafol, 'figures')
        if not os.path.exists(self.figure_folder):
            os.makedirs(self.figure_folder)
        self.fly = im.fly(self.folder, self.datafol, **dict)
        self.mb = im.MB(self.folder, self.datafol, self.mb_slices)

    def split_data(self):
        """
        split data into inside and out
        """

        # get compartments, exclude ones for which no ROI was drawn
        comps = self.mb.load_comps()
        comps = comps.loc[:, ~comps.isna().any()]
        ft2 = self.mb.ft2

        data = pd.concat([ft2,comps], axis=1)
        # data['instrip'] = np.where(np.abs(data.ft_posx)<25, True, False)
        data = fn.exclude_lost_tracking(data, thresh=10)
        data = fn.consolidate_in_out(data)
        data = fn.calculate_speeds(data)
        d, di, do = fn.inside_outside(data)
        dict_temp = {"data": data,
                    "d": d,
                    "di": di,
                    "do": do}


        pickle_name = os.path.join(self.fly.processedfol, 'bouts.p')
        fn.save_obj(dict_temp, pickle_name)
        return dict_temp

    def load_split_data(self, overwrite=True):
        pickle_name = os.path.join(self.fly.processedfol, 'bouts.p')
        if (not overwrite) and os.path.exists(pickle_name):
            data = fn.load_obj(pickle_name)
        else:
            data = self.split_data()
        return data

    def plot_compartments(self):

        # get compartments, exclude ones for which no ROI was drawn
        comps = self.mb.load_comps()
        comps = comps.loc[:, ~comps.isna().any()]
        num_comps = len(comps.columns)

        bouts = self.load_split_data(overwrite=True)
        di = bouts['di']

        t = self.mb.ft2.seconds.to_numpy()
        t = t - t[0]
        fig, axs = plt.subplots(num_comps, figsize = (8, 2*num_comps), sharex=True)
        for i, comp in enumerate(comps.columns):
            axs[i].plot(t, comps[comp], color=pl.mb_color[comp])
            height = np.max(comps[comp])-np.min(comps[comp])
            y0=np.min(comps[comp])
            for key in list(di.keys()):
                df_temp = di[key]
                ix = df_temp.index.to_numpy()
                t0 = t[ix[0]]
                t1 = t[ix[-1]]
                width = t1-t0
                rect = Rectangle((t0, y0), width, height, fill='k', alpha=0.3)
                #rect = Rectangle((t0, -np.pi), width, height, fill=False)

                axs[i].add_patch(rect)


        fig.savefig(os.path.join(self.figure_folder, 'compartments.pdf'))

    # def plot_triggered_averages():


%matplotlib

dLight().plot_averages()

# %%
fig, axs = plt.subplots(2,6)
t = (np.arange(len(ixs))-buf_pts)/10

# iterate through in to out transitions
for i, key in enumerate(list(d_out.keys())):
    all_outs = []
    d_temp = d_out[key]
    for j in np.arange(len(d_temp)):
        all_outs.append(np.mean(d_temp[j], axis=0))
        axs[0,i].plot(t, np.mean(d_temp[j], axis=0), color=pl.mb_color[key], alpha=0.2)
    m = np.mean(all_outs, axis=0)
    error = stats.sem(all_outs, axis=0)
    axs[0,i].plot(t, m, color = 'k', linewidth=2)
    axs[0,i].fill_between(t, m+error, m-error, color=pl.mb_color[key])

# iterate through out to in transitions
for i, key in enumerate(list(d_in.keys())):
    all_ins = []
    d_temp = d_in[key]
    for j in np.arange(len(d_temp)):
        all_ins.append(np.mean(d_temp[j], axis=0))
        axs[1,i].plot(t, np.mean(d_temp[j], axis=0), color=pl.mb_color[key], alpha=0.2)
    m = np.mean(all_ins, axis=0)
    error = stats.sem(all_ins, axis=0)
    axs[1,i].plot(t, m, color = 'k', linewidth=2)
    axs[1,i].fill_between(t, m+error, m-error, color=pl.mb_color[key])

fig.tight_layout()


# %%
datafol = dl.datafol
d = dl.sheet.iloc[39].to_dict()
name = d['folder']
fileloc = os.path.join(datafol, name)
z_slices = d['mb_slices']

mb = im.MB(name, fileloc, z_slices)
for m, slice in enumerate(mb.zslices):
    print(slice)

# %%
# open masks

@jit(nopython=True)
def extract_values(num_frames, active_pixels, projected):
    temp = []
    for frame in np.arange(num_frames):
        temp.append(np.nansum(projected[:,:,frame])/active_pixels)
    return temp
comp_ids = ['g2', 'g3', 'g4', 'g5', 'bp1', 'bp2', 'b1', 'b2']
comp_num = len(comp_ids)
all_masks = mb.open_mask()
if len(all_masks.shape) == 2:
    all_masks = np.reshape(all_masks, all_masks.shape + (1,))

for m, slice in enumerate(mb.zslices):
    # open slice
    slice = mb.open_slice(slice)

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
        t1 = time.time()
        temp = extract_values(num_frames, active_pixels, projected)
        t2 = time.time()
        print(t2-t1)
        temp = np.array(temp)
        comps[:,comp_ix] = temp[:]

    # if there is an extra frame in this slice, remove it

    comps = comps[range(comps_all.shape[0]),:]
    comps_all[:,:,m] = comps
comps_all = np.nansum(comps_all, axis=2)
comps_all = pd.DataFrame(comps_all)
comps_all = comps_all.apply(fn.lnorm)
comps_all.columns=comp_ids


comps_all.to_csv(os.path.join(self.regfol, 'pb.csv'))

return gloms_all
