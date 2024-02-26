import os
import sys
import numpy as np
import traceback
sys.path.append("../")
sys.path.append("../ActivityAnalyses")
sys.path.append("../UtilsDynamicSimulations/OpenSimAD")
baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)
activityAnalysesDir = os.path.join(baseDir, 'ActivityAnalyses')
sys.path.append(activityAnalysesDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)
from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking
from utils import storage_to_numpy
import shutil

# %% Paths.
dataFolder = os.path.join(baseDir, 'Data', 'Benchmark')
i = 11
subjects = ['subject' + str(i) for i in range(2,12)]


trials = {
    'subject2': {'walking1': {'start':-1, 'end':1.85}, 'walking2': {'start':-0.9, 'end':1.76}, 'walking3': {'start':-1, 'end':1.76}, 'walkingTS1': {'start':-1, 'end':2.15}, 'walkingTS2': {'start':-1, 'end':1.97}, 'walkingTS4': {'start':-0.9, 'end':2.13}},
    'subject3': {'walking1': {'start':-1.8, 'end':1.41}, 'walking2': {'start':-1.8, 'end':1.46}, 'walking3': {'start':-1.7, 'end':1.48}, 'walkingTS2': {'start':-2.5, 'end':1.97}, 'walkingTS3': {'start':-2, 'end':1.79}, 'walkingTS4': {'start':-2.2, 'end':1.7}},
    'subject4': {'walking1': {'start':-0.7, 'end':1.6}, 'walking2': {'start':-0.7, 'end':1.87}, 'walking4': {'start':-0.7, 'end':1.7}, 'walkingTS1': {'start':-0.7, 'end':1.7}, 'walkingTS2': {'start':-0.7, 'end':1.6}, 'walkingTS3': {'start':-0.7, 'end':1.95}},
    'subject5': {'walking1': {'start':-0.7, 'end':1.83}, 'walking2': {'start':-0.7, 'end':1.8}, 'walking3': {'start':-0.7, 'end':1.8}, 'walkingTS1': {'start':-0.8, 'end':1.88}, 'walkingTS2': {'start':-0.7, 'end':1.75}, 'walkingTS3': {'start':-0.7, 'end':1.72}},
    'subject6': {'walking1': {'start':-1.2, 'end':1.63}, 'walking2': {'start':-1.2, 'end':1.6}, 'walking3': {'start':-1.2, 'end':2}, 'walkingTS1': {'start':-0.7, 'end':1.65}, 'walkingTS2': {'start':-0.8, 'end':1.72}, 'walkingTS3': {'start':-1.1, 'end':1.78}},
    'subject7': {'walking1': {'start':-0.8, 'end':1.79}, 'walking2': {'start':-0.7, 'end':1.82}, 'walking3': {'start':-0.7, 'end':1.87}, 'walkingTS1': {'start':-1.1, 'end':1.83}, 'walkingTS2': {'start':-1.1, 'end':1.9}, 'walkingTS3': {'start':-1.1, 'end':2.12}},
    'subject8': {'walking1': {'start':-1, 'end':1.83}, 'walking2': {'start':-0.7, 'end':1.89}, 'walking3': {'start':-0.7, 'end':1.92}, 'walkingTS1': {'start':-0.7, 'end':2.3}, 'walkingTS3': {'start':-0.7, 'end':1.9}}, # walkingTS2 excluded
    'subject9': {'walking1': {'start':-0.6, 'end':1.65}, 'walking2': {'start':-0.5, 'end':1.55}, 'walking3': {'start':-0.6, 'end':1.6}, 'walkingTS1': {'start':-0.7, 'end':1.68}, 'walkingTS2': {'start':-0.7, 'end':1.63}, 'walkingTS3': {'start':-0.7, 'end':1.56}},
    'subject10': {'walking1': {'start':-0.7, 'end':1.46}, 'walking2': {'start':-0.7, 'end':1.49}, 'walking3': {'start':-0.7, 'end':1.5}, 'walkingTS1': {'start':-0.8, 'end':1.78}, 'walkingTS2': {'start':-0.9, 'end':1.85}, 'walkingTS3': {'start':-0.8, 'end':1.66}},
    'subject11': {'walking2': {'start':-0.7, 'end':1.6}, 'walking3': {'start':-0.7, 'end':1.55}, 'walking4': {'start':-0.7, 'end':1.62}, 'walkingTS1': {'start':-0.7, 'end':1.9}, 'walkingTS2': {'start':-0.7, 'end':1.85}, 'walkingTS3': {'start':-0.7, 'end':1.9}}, # arms weird in beginning, if 0.7 does not work roll back to 0.5
    }

for subject in subjects:

    sessionDir = os.path.join(dataFolder, subject)
    session_id = ''

    for count, trial_name in enumerate(list(trials[subject].keys())):

        pathFolder = os.path.join(baseDir, 'Data', 'Benchmark', subject, 'OpenSimData_trimmed', 'Dynamics', trial_name + '_videoAndMocap')
        pathFolderEnd = os.path.join(baseDir, 'Data', 'Benchmark', subject, 'OpenSimData', 'Dynamics', trial_name + '_video')

        # Copy Setup_0.yaml, stats_0.npy, and w_opt_0.npy
        pathSetup = os.path.join(pathFolder, 'Setup_0.yaml')
        pathStats = os.path.join(pathFolder, 'stats_0.npy')
        pathWOpt = os.path.join(pathFolder, 'w_opt_0.npy')
        pathSetupEnd = os.path.join(pathFolderEnd, 'Setup_0.yaml')
        pathStatsEnd = os.path.join(pathFolderEnd, 'stats_0.npy')
        pathWOptEnd = os.path.join(pathFolderEnd, 'w_opt_0.npy')
        shutil.copyfile(pathSetup, pathSetupEnd)
        shutil.copyfile(pathStats, pathStatsEnd)
        shutil.copyfile(pathWOpt, pathWOptEnd)

