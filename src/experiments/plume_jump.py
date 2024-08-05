import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.drive import drive as dr
from src.utilities import funcs
from src.utilities import plotting as pl
from src.utilities import imaging as im
importlib.reload(dr)
importlib.reload(pl)
importlib.reload(im)
importlib.reload(funcs)


class plume_jump:
    def __init__(self):
        d = dr.drive_hookup()

        # load google sheet with experiment information
        self.sheet_id = '16mP8FXDmy1ZEl_NL5TM7HV72Qngv-LvukE76vbp80Qc'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df

        # define local site for log files.
        self.logfol = os.path.join(os.getcwd(), 'data', 'plume_jump')

        # define local site for log files.
        self.figure_folder = os.path.join(os.getcwd(), 'figures', 'strip_jump')


    def plot_trajectories(self):
        from matplotlib.backends.backend_pdf import PdfPages
        for fly in self.sheet.fly.unique():
            with PdfPages(os.path.join(self.figure_folder, 'Fly_'+fly+'.pdf')) as pdf:
                fly_sheet = self.sheet[self.sheet.fly==fly]
                for log in fly_sheet.log:
                    print(log)
                    file_loc = os.path.join(self.logfol, log)
                    df = funcs.read_log(file_loc)
                    fig, axs = plt.subplots(1)
                    axs.axis('equal')
                    pl.plot_trajectory(df, axs)
                    pl.plot_trajectory_odor(df, axs)
                    #pl.plot_jumping_edges(df, axs)
                    fig.suptitle('Fly: '+fly+' log: '+log)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()
        return df

%matplotlib
df = plume_jump().plot_trajectories()
