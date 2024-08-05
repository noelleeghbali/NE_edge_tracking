import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.drive import drive as dr
from src.utilities import funcs as fn
from src.utilities import plotting as pl
from src.utilities import imaging as im
importlib.reload(dr)
importlib.reload(pl)
importlib.reload(im)
importlib.reload(funcs)



class plume_shift():
    """
    20230415 -- digging up old data on the shifting plume
    """
    def __init__(self):

        log = '09252020-171547_Alternating strip_period_400mm_width_50mm.log'
        self.logfol = '/Volumes/Andy/logs'
        self.data = fn.read_log(os.path.join(self.logfol, log))

a = plume_shift()
# %%
%matplotlib
df = a.data
df['instrip'] = np.where(np.abs(df.mfc3_stpt)>0, True, False)
fig, axs = plt.subplots(1,1)
period = 400
for i in np.arange(0,4):
    x1l = [-25,-25]
    x1r = [25, 25]
    y1 = [i*period, i*period+period/2]
    x2l = [25,25]
    x2r = [75,75]
    y2 = [period/2+i*period, i*period+period]
    axs.plot(x1l, y1, 'grey')
    axs.plot(x1r, y1, 'grey')
    axs.plot(x2l, y2, 'k')
    axs.plot(x2r, y2, 'k')
axs = pl.plot_trajectory(df, axs=axs)
axs = pl.plot_trajectory_odor(df, axs=axs)
