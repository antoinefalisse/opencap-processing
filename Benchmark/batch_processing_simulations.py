'''
    ---------------------------------------------------------------------------
    OpenCap processing: batch_processing_giat_simulations.py
    ---------------------------------------------------------------------------
    Copyright 2023 Stanford University and the Authors
    
    Author(s): Scott Uhlrich & Antoine Falisse
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.    
'''

# v0.63: latest - models 77 and 93 from marker-augmenter repository.
# path: G:\My Drive\Projects\mobilecap\Data

import os
import sys
import numpy as np
import traceback
import yaml
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

# %% Paths.
dataFolder = os.path.join(baseDir, 'Data', 'Benchmark')
i = 2
subjects = ['subject' + str(i) for i in range(i,i+2)]

# TODO: subject 10 might be 56.6 instead of 60kgs, check if that makes a diff.


trials = {
    'subject2': {'walking': {'walking1': {'start':-1, 'end':1.85}, 'walking2': {'start':-0.9, 'end':1.76}, 'walking3': {'start':-1, 'end':1.76},
                             'walkingTS1': {'start':-1, 'end':2.15}, 'walkingTS2': {'start':-1, 'end':1.97}, 'walkingTS4': {'start':-0.9, 'end':2.13}},
                 'STS':     {'STS1': {'start':None, 'end':None}}},
    'subject3': {'walking': {'walking1': {'start':-1.8, 'end':1.56}, 'walking2': {'start':-1.8, 'end':1.46}, 'walking3': {'start':-1.7, 'end':1.48}, 
                             'walkingTS2': {'start':-2.5, 'end':1.97}, 'walkingTS3': {'start':-2, 'end':1.79}, 'walkingTS4': {'start':-2.2, 'end':1.7}}},
    'subject4': {'walking': {'walking1': {'start':-0.7, 'end':1.6}, 'walking2': {'start':-0.7, 'end':1.87}, 'walking4': {'start':-0.7, 'end':1.7},
                             'walkingTS1': {'start':-0.7, 'end':1.7}, 'walkingTS2': {'start':-0.7, 'end':1.6}, 'walkingTS3': {'start':-0.7, 'end':1.95}}},
    'subject5': {'walking': {'walking1': {'start':-0.7, 'end':1.83}, 'walking2': {'start':-0.7, 'end':1.8}, 'walking3': {'start':-0.7, 'end':1.8},
                             'walkingTS1': {'start':-0.8, 'end':1.88}, 'walkingTS2': {'start':-0.7, 'end':1.75}, 'walkingTS3': {'start':-0.7, 'end':1.72}}},
    'subject6': {'walking': {'walking1': {'start':-1.2, 'end':1.63}, 'walking2': {'start':-1.2, 'end':1.6}, 'walking3': {'start':-1.2, 'end':2},
                             'walkingTS1': {'start':-0.7, 'end':1.65}, 'walkingTS2': {'start':-0.8, 'end':1.72}, 'walkingTS3': {'start':-1.1, 'end':1.78}}},
    'subject7': {'walking': {'walking1': {'start':-0.8, 'end':1.79}, 'walking2': {'start':-0.7, 'end':1.82}, 'walking3': {'start':-0.7, 'end':1.87},
                             'walkingTS1': {'start':-1.1, 'end':1.83}, 'walkingTS2': {'start':-1.1, 'end':1.9}, 'walkingTS3': {'start':-1.1, 'end':2.12}}},
    'subject8': {'walking': {'walking1': {'start':-1, 'end':1.83}, 'walking2': {'start':-0.7, 'end':1.89}, 'walking3': {'start':-0.7, 'end':1.92}, 
                             'walkingTS1': {'start':-0.7, 'end':2.3}, 'walkingTS3': {'start':-0.7, 'end':1.9}}},
    'subject9': {'walking': {'walking1': {'start':-0.6, 'end':1.65}, 'walking2': {'start':-0.5, 'end':1.55}, 'walking3': {'start':-0.6, 'end':1.6}, 
                             'walkingTS1': {'start':-0.7, 'end':1.68}, 'walkingTS2': {'start':-0.7, 'end':1.63}, 'walkingTS3': {'start':-0.7, 'end':1.56}}},
    'subject10': {'walking': {'walking1': {'start':-0.7, 'end':1.46}, 'walking2': {'start':-0.7, 'end':1.49}, 'walking3': {'start':-0.7, 'end':1.5}, 
                              'walkingTS1': {'start':-0.8, 'end':1.78}, 'walkingTS2': {'start':-0.9, 'end':1.85}, 'walkingTS3': {'start':-0.8, 'end':1.66}}},
    'subject11': {'walking': {'walking2': {'start':-0.7, 'end':1.6}, 'walking3': {'start':-0.7, 'end':1.55}, 'walking4': {'start':-0.7, 'end':1.62}, 
                              'walkingTS1': {'start':-0.7, 'end':1.9}, 'walkingTS2': {'start':-0.7, 'end':1.85}, 'walkingTS3': {'start':-0.7, 'end':1.9}}},
    }


# %% User-defined variables.
filter_frequency = 6

# Settings for dynamic simulation.
# motion_style = 'STS'
# motion_type = 'sit_to_stand_formulation2'
# repetition = 1

motion_style = 'walking'
motion_type = 'walking_formulation2'

case = '7'
runProblem = True
processInputs = True
runSimulation = True
solveProblem = True
analyzeResults = False
plotResults = False

if case == '0':
    buffer_start = 0
    buffer_end = 0
elif case == '1':
    buffer_start = 0.7
    buffer_end = 0.5
elif case == '2': # Did the same one to compare end times
    buffer_start = 0.7
    buffer_end = 0.5
elif case == '3':
    buffer_start = 0.7
    buffer_end = 0
elif case == '4':
    buffer_start = 0.7
    buffer_end = 0.5
    weight_activation = 5
elif case == '5':
    buffer_start = 0.7
    buffer_end = 0.5
    weight_activation = 1
elif case == '6':
    buffer_start = 0.5
    buffer_end = 0.5
elif case == '7':
    buffer_start = 0.7
    buffer_end = 0.5
    weight_activation = 20
    
# %% Gait segmentation and kinematic analysis.

no_results = []
for subject in subjects:

    sessionDir = os.path.join(dataFolder, subject)
    session_id = ''

    pathData = os.path.join(dataFolder, subject, 'OpenSimData', 'Video', 'mmpose_0.8', '2-cameras', 'v0.63', 'IK', 'LaiArnoldModified2017_poly_withArms_weldHand')
    for count, trial_name in enumerate(list(trials[subject][motion_style].keys())):
        
        # if count > 2:
        #     continue
        
        trial_name += '_video'
        
        if runProblem:        

                print('Processing {}-{} for dynamic simulation...'.format(subject, trial_name))
                if processInputs:
                    try:
                        if not 'repetition' in locals():
                            repetition = None
                        settings = processInputsOpenSimAD(
                            baseDir, sessionDir, session_id, trial_name, 
                            motion_type, repetition=repetition)
                        
                        pathMotionFile = os.path.join(dataFolder, subject, 'OpenSimData', 'Kinematics', trial_name + '.mot')
                        motion_file = storage_to_numpy(pathMotionFile)
                        full_time_window = [motion_file['time'][0], motion_file['time'][-1]]
                        
                        if 'walking' in trial_name:
                            # Get time interval from trimmed trial
                            pathTrimmedMotionFile = os.path.join(dataFolder, subject, 'OpenSimData_trimmed', 'Kinematics', trial_name.replace('_video', '_videoAndMocap') + '.mot')
                            trimmed_motion_file = storage_to_numpy(pathTrimmedMotionFile)
                            trimmed_time_window = [trimmed_motion_file['time'][0], trimmed_motion_file['time'][-1]]
                            
                            

                            # Update time window
                            time_start = np.round(max(trimmed_time_window[0] - buffer_start, full_time_window[0], trials[subject][motion_style][trial_name.replace('_video', '')]['start']),2)
                            time_end = np.round(min(trimmed_time_window[1] + buffer_end, full_time_window[1], trials[subject][motion_style][trial_name.replace('_video', '')]['end']),2)
                            buffer_start_applied = np.abs(np.round(time_start - trimmed_time_window[0], 2))
                            buffer_end_applied = np.abs(np.round(time_end - trimmed_time_window[1], 2))
                            settings['buffers'] = [round(float(buffer_start_applied),6),
                                                round(float(buffer_end_applied),6)]
                            time_window = [time_start, time_end]
                            settings['timeInterval'] = [round(float(i),6) for i in time_window]
                            settings['timeIntervalWithoutBuffers'] = [round(float(settings['timeInterval'][0] + settings['buffers'][0]),6),
                                                                    round(float(settings['timeInterval'][1] - settings['buffers'][1]),6)]     
                            
                            if case == '4' or case == '5' or case == '7':
                                settings['weights']['activationTerm'] = weight_activation
                                
                                
                        if 'STS' in trial_name:
                            
                            time_start = np.round(max(settings['timeInterval'][0] - buffer_start, full_time_window[0]),2)
                            time_end = np.round(min(settings['timeInterval'][1] + buffer_end, full_time_window[1]),2)
                            buffer_start_applied = np.abs(np.round(time_start - settings['timeInterval'][0], 2))
                            buffer_end_applied = np.abs(np.round(time_end - settings['timeInterval'][1], 2))
                            settings['buffers'] = [round(float(buffer_start_applied),6),
                                                   round(float(buffer_end_applied),6)]
                            time_window = [time_start, time_end]
                            settings['timeInterval'] = [round(float(i),6) for i in time_window]
                            settings['timeIntervalWithoutBuffers'] = [round(float(settings['timeInterval'][0] + settings['buffers'][0]),6),
                                                                      round(float(settings['timeInterval'][1] - settings['buffers'][1]),6)] 
                            
                        
                    except Exception as e:
                        print(f"Error setting up dynamic optimization for trial {trial_name}: {e}")
                        continue
            
                # Simulation.
                if runSimulation:
                    try:
                        # print('Running dynamic simulation...')
                        run_tracking(baseDir, sessionDir, settings, case=case, 
                                    solveProblem=solveProblem, analyzeResults=analyzeResults)
                        test=1
                    except Exception as e:
                        tb_info = traceback.format_exc()
                        print(f"Error during dynamic optimization for trial {trial_name}: {e}\nTraceback: {tb_info}")
                        no_results.append(subject + '_' + trial_name)
                        continue
            
        if plotResults:    

            # Load settings

            pathResults = os.path.join(dataFolder, subject, 'OpenSimData', 'Dynamics', trial_name)
            if 'repetition' in locals():
                pathResults = os.path.join(dataFolder, subject, 'OpenSimData', 'Dynamics', trial_name + '_rep' + str(repetition))     
            pathSettings = os.path.join(pathResults, 'Setup_{}.yaml'.format(case))
            with open(pathSettings, 'r') as file:
                settings = yaml.safe_load(file)


            plotResultsOpenSimAD(sessionDir, trial_name, settings, cases=['0', '6'], mainPlots=True, grfPlotOnly=False)
        
        test=1

print('No results for the following trials:')
print(no_results)
