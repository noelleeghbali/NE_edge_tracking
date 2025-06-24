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

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/KCdLight_replay1'
lobes = ['G2', 'G3', 'G4', 'G5']
colors = ['#7fffd4', '#08e8de','#ff55a3', '#ff1493']

et_replay_traces_bw(figure_folder, lobes, colors, size = (12,6))
et_replay_auc_comp(figure_folder, lobes, colors, size=(6,6))
et_replay_peak_comp(figure_folder, lobes, colors, size=(6,6))
et_replay_tuning(figure_folder, lobes, colors, size=(10,4))

