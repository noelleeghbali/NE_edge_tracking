from src.utilities import imaging as im
from src.utilities import plotting as pl
import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import numpy as np
from src.drive import drive as dr
from src.utilities import funcs as fn
from src.utilities import plotting as pl
importlib.reload(dr)
importlib.reload(pl)
importlib.reload(fn)

class fb4m_imaging:
    def __init__(self):
        d = dr.drive_hookup()

        # load google sheet with experiment information
        self.sheet_id = '1ynEdQPwIDcLiygxgWOEhpiBeImgtk_2i3ha3ZE_35tA'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df

        # define local folder for data
        self.datafol = "/Volumes/LACIE/Andy/FB4M"


    def register(self):
        """
        register all images
        """
        for folder in self.sheet.folder:
            name = folder
            folderloc = os.path.join(self.datafol, name)
            print(folderloc)
            ex = im.fly(name, folderloc)
            ex.register_all_images(overwrite=False)

    def processing(self):
        """
        pre and post process data
        """
        for i, folder in enumerate(self.sheet.folder):
            name = folder
            folderloc = os.path.join(self.datafol, name)
            kwargs = self.sheet.iloc[i].to_dict()
            ex = im.fly(name, folderloc, **kwargs)
            ex.save_preprocessing()
            ex.save_postprocessing()

    def make_plots(self):
        for i, folder in enumerate(self.sheet.folder):
            name = folder
            folderloc = os.path.join(self.datafol, name)
            kwargs = self.sheet.iloc[i].to_dict()
            ex = im.fly(name, folderloc, **kwargs)
            pv2, ft, ft2, ix = ex.load_postprocessing()

            # plot the trajectory
            fig, axs = plt.subplots(1,1, sharex=True)
            pl.plot_trajectory(ft2, axs)
            pl.plot_trajectory_odor(ft2, axs)


            # plot the trace
            fig, axs = plt.subplots(2,1, sharex=True)
            axs[0].plot(pv2['fb4m_dff'])

            axs2 = axs[0].twinx()
            axs2.set_ylim(0,1)
            d, di, do = fn.inside_outside(ft2)
            for key in list(di.keys()):
                df_temp = di[key]
                ix = df_temp.index.to_list()
                width = ix[-1]-ix[0]
                height = 1
                rect = Rectangle((ix[0], 0), width, height, fill=False)
                axs2.add_patch(rect)

            d, dmove, dstop = fn.dict_stops(ft2)
            for key in list(dstop.keys()):
                df_temp = dstop[key]
                ix = df_temp.index.to_list()
                width = ix[-1]-ix[0]
                height = 1
                rect = Rectangle((ix[0], 0), width, height, alpha=0.2, ec=None, color='red')
                axs2.add_patch(rect)

            # plot the trace and the planer speed
            axs[1].plot(pv2['fb4m_dff'])
            df_speed = fn.calculate_speeds(ft2)
            axs_r = axs[1].twinx()
            axs_r.plot(df_speed.speed, 'r')






%matplotlib
a = fb4m_imaging()
a.make_plots()
