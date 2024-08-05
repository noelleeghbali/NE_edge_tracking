
from imaging_analysis import *
from behavior_analysis import *
from analysis_funs.regression import fci_regmodel
from analysis_funs.optogenetics import opto
from analysis_funs.CX_imaging import CX
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
import sys
import pickle

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/MBON21/picklez/trackerz'
neuron = 'mbon21'

# Fluorescence trace over experiment
trace_FF(figure_folder, neuron)

# Triggered average fluorescence at entry and exit
#triggered_FF(figure_folder, neuron, tbef=10, taf=10, event_type='entry', first=True)
#triggered_FF(figure_folder, neuron, tbef=30, taf=30, event_type='exit', first=False)

# Trajectory w/ fluorescence colormap
#traj_FF(figure_folder, neuron, cmin=0,cmax=1)

#heading_FF(figure_folder, neuron)
