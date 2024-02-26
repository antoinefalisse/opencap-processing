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
sys.path.append("../")
sys.path.append("../ActivityAnalyses")
sys.path.append("../UtilsDynamicSimulations/OpenSimAD")
baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)
activityAnalysesDir = os.path.join(baseDir, 'ActivityAnalyses')
sys.path.append(activityAnalysesDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)
from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD, getGRF
from mainOpenSimAD import run_tracking
from utils import storage_to_numpy, import_metadata

# %% Paths.
dataFolder = os.path.join(baseDir, 'Data', 'Benchmark')
i = 10
subjects = ['subject' + str(i) for i in range(i,i+1)]


trials = {
    'subject2': {'walking1': {'start':-1, 'end':1.85}, 'walking2': {'start':-0.9, 'end':1.76}, 'walking3': {'start':-1, 'end':1.76}, 'walkingTS1': {'start':-1, 'end':2.15}, 'walkingTS2': {'start':-1, 'end':1.97}, 'walkingTS4': {'start':-0.9, 'end':2.13}},
    # 'subject3': {'walking1': {'start':-1.8, 'end':1.41}, 'walking2': {'start':-1.8, 'end':1.46}, 'walking3': {'start':-1.7, 'end':1.48}, 'walkingTS2': {'start':-2.5, 'end':1.97}, 'walkingTS3': {'start':-2, 'end':1.79}, 'walkingTS4': {'start':-2.2, 'end':1.7}},
    'subject3': {'walking1': {'start':-1.8, 'end':1.56}, 'walking2': {'start':-1.8, 'end':1.56}, 'walking3': {'start':-1.7, 'end':1.58}, 'walkingTS2': {'start':-2.5, 'end':2.08}, 'walkingTS3': {'start':-2, 'end':1.96}, 'walkingTS4': {'start':-2.2, 'end':1.82}},
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

    pathData = os.path.join(dataFolder, subject, 'ForceData')
    pathFile = os.path.join(pathData, 'static1_forces.mot')
    
    GRF = {        
        'headers': {
            'forces': {
                'right': ['R_ground_force_vx', 'R_ground_force_vy', 
                          'R_ground_force_vz'],
                'left': ['L_ground_force_vx', 'L_ground_force_vy', 
                         'L_ground_force_vz'],
                'all': ['R_ground_force_vx', 'R_ground_force_vy', 
                        'R_ground_force_vz','L_ground_force_vx', 
                        'L_ground_force_vy', 'L_ground_force_vz']},
            'COP': {
                'right': ['R_ground_force_px', 'R_ground_force_py', 
                          'R_ground_force_pz'],
                'left': ['L_ground_force_px', 'L_ground_force_py', 
                         'L_ground_force_pz'],
                'all': ['R_ground_force_px', 'R_ground_force_py', 
                        'R_ground_force_pz','L_ground_force_px', 
                        'L_ground_force_py', 'L_ground_force_pz']},
            'torques': {
                'right': ['R_ground_torque_x', 'R_ground_torque_y', 
                          'R_ground_torque_z'],
                'left': ['L_ground_torque_x', 'L_ground_torque_y', 
                         'L_ground_torque_z'],
                'all': ['R_ground_torque_x', 'R_ground_torque_y', 
                        'R_ground_torque_z', 'L_ground_torque_x', 
                        'L_ground_torque_y', 'L_ground_torque_z']}}}
    
    GRF_right = getGRF(pathFile, GRF['headers']['forces']['right'])
    GRF_left = getGRF(pathFile, GRF['headers']['forces']['left'])
    
       
    mass = np.mean(GRF_left['L_ground_force_vy'].to_numpy()) / 9.8066499999999994

    metadata = import_metadata(os.path.join(dataFolder, subject, 'sessionMetadata.yaml'))
    subject_mass = metadata['mass_kg']

    print(f'Estimated mass: {mass} kg')
    print(f'Metadata mass: {subject_mass} kg')

    

    