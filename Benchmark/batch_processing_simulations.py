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
# dataFolder = os.path.join(baseDir, 'Data', 'Benchmark')
dataFolder = os.path.join(baseDir, 'Data', 'Benchmark_mocap_updated')
i = 2
# subjects = ['subject' + str(i) for i in range(i,i+1)]
subjects = ['subject' + str(i) for i in range(4,5)]

trials = {
    'subject2': {
        'walking': {
            'walking1': {'start':-1, 'end':1.85}, 'walking2': {'start':-0.9, 'end':1.76}, 'walking3': {'start':-1, 'end':1.90}, # case 2: 'walking3': {'start':-1, 'end':1.76},
            'walkingTS1': {'start':-1, 'end':2.15}, 'walkingTS2': {'start':-1, 'end':1.97}, 'walkingTS4': {'start':-0.9, 'end':2.13}}, # case 2: 'walkingTS1': {'start':-1, 'end':2.25}},
        'STS':     {
            'STS1': {'start':None, 'end':None}, 'STSweakLegs1': {'start':None, 'end':None}},
        'Squats':{
            'squats1': {'start':None, 'end':None}, 'squatsAsym1': {'start':None, 'end':None}},
        'DJ': {
            'DJ1': {'start':None, 'end':None}, 'DJ2': {'start':None, 'end':None}, 'DJ3': {'start':None, 'end':None}, 'DJAsym1': {'start':None, 'end':None}, 'DJAsym4': {'start':None, 'end':None}, 'DJAsym5': {'start':None, 'end':None}}
        },
    'subject3': {
        'walking': {
            # Case 1: Warning, I re-organised stuff such that the good cases are 1.
            'walking1': {'start':-1.8, 'end':1.67}, 'walking2': {'start':-1.8, 'end':1.61}, 'walking3': {'start':-1.7, 'end':1.63}, 
            'walkingTS2': {'start':-2.5, 'end':2.11}, 'walkingTS3': {'start':-2, 'end':1.79}, 'walkingTS4': {'start':-2.2, 'end':1.82}},
            # Case 2
            # 'walking1': {'start':-1.8, 'end':1.56}, 'walking2': {'start':-1.8, 'end':1.59}, 'walking3': {'start':-1.7, 'end':1.61}, 
            # 'walkingTS2': {'start':-2.5, 'end':1.97}, 'walkingTS3': {'start':-2, 'end':1.96}, 'walkingTS4': {'start':-2.2, 'end':1.7}},
             # Case 12
            # 'walking2': {'start':-1.8, 'end':1.46}, 'walking3': {'start':-1.7, 'end':1.48}, 
            # 'walkingTS3': {'start':-2, 'end':2.00}},
        'STS':     {'STS1': {'start':None, 'end':None}, 'STSweakLegs1': {'start':None, 'end':None}},
        'Squats':{
            'squats1': {'start':None, 'end':None}, 'squatsAsym1': {'start':None, 'end':None}},
        'DJ': {
            'DJ1': {'start':None, 'end':None}, 'DJ2': {'start':None, 'end':None}, 'DJ4': {'start':None, 'end':None}, 'DJAsym1': {'start':None, 'end':None}, 'DJAsym2': {'start':None, 'end':None}, 'DJAsym4': {'start':None, 'end':None}}
        },
    'subject4': {
        'walking': {
            'walking1': {'start':-0.7, 'end':1.6}, 'walking2': {'start':-0.7, 'end':1.87}, 'walking4': {'start':-0.7, 'end':1.7},
            'walkingTS1': {'start':-0.7, 'end':1.7}, 'walkingTS2': {'start':-0.7, 'end':1.6}, 'walkingTS3': {'start':-0.7, 'end':1.95}},
        'STS':     {
            'STS1': {'start':None, 'end':None}, 'STSweakLegs1': {'start':None, 'end':None}},
        'Squats':{
            'squats1': {'start':None, 'end':None}, 'squatsAsym1': {'start':None, 'end':None}},
        'DJ': {
            'DJ1': {'start':None, 'end':None}, 'DJ2': {'start':None, 'end':None}, 'DJ3': {'start':None, 'end':None}, 'DJAsym1': {'start':None, 'end':None}, 'DJAsym2': {'start':None, 'end':None}, 'DJAsym3': {'start':None, 'end':None}}
        },
    'subject5': {
        'walking': {
            'walking1': {'start':-0.7, 'end':1.83}, 'walking2': {'start':-0.7, 'end':1.8}, 'walking3': {'start':-0.7, 'end':1.8},
            'walkingTS1': {'start':-0.8, 'end':1.88}, 'walkingTS2': {'start':-0.7, 'end':1.75}, 'walkingTS3': {'start':-0.7, 'end':1.72}},
        'STS':     {'STS1': {'start':None, 'end':None}, 'STSweakLegs1': {'start':None, 'end':None}},
        'Squats':{
            'squats1': {'start':None, 'end':None}, 'squatsAsym1': {'start':None, 'end':None}},
        'DJ': {
            'DJ1': {'start':None, 'end':None}, 'DJ2': {'start':None, 'end':None}, 'DJ3': {'start':None, 'end':None}, 'DJAsym1': {'start':None, 'end':None}, 'DJAsym2': {'start':None, 'end':None}, 'DJAsym3': {'start':None, 'end':None}}
        },
    'subject6': {
        'walking': {
            'walking1': {'start':-1.2, 'end':1.63}, 'walking2': {'start':-1.2, 'end':1.6}, 'walking3': {'start':-1.2, 'end':2},
            'walkingTS1': {'start':-0.7, 'end':1.65}, 'walkingTS2': {'start':-0.8, 'end':1.72}, 'walkingTS3': {'start':-1.1, 'end':1.78}},
        'STS':     {
            'STS1': {'start':None, 'end':None}, 'STSweakLegs1': {'start':None, 'end':None}},
        'Squats':{
            'squats1': {'start':None, 'end':None}, 'squatsAsym1': {'start':None, 'end':None}},
        'DJ': {
            'DJ1': {'start':None, 'end':None}, 'DJ2': {'start':None, 'end':None}, 'DJ3': {'start':None, 'end':None}, 'DJAsym1': {'start':None, 'end':None}, 'DJAsym2': {'start':None, 'end':None}, 'DJAsym3': {'start':None, 'end':None}}
        },
    'subject7': {
        'walking': {
            'walking1': {'start':-0.8, 'end':1.79}, 'walking2': {'start':-0.7, 'end':1.82}, 'walking3': {'start':-0.7, 'end':1.87},
            'walkingTS1': {'start':-1.1, 'end':1.83}, 'walkingTS2': {'start':-1.1, 'end':1.9}, 'walkingTS3': {'start':-1.1, 'end':2.12}},
        'STS':     {
            'STS1': {'start':None, 'end':None}, 'STSweakLegs1': {'start':None, 'end':None}},
        'Squats':{
            'squats1': {'start':None, 'end':None}, 'squatsAsym1': {'start':None, 'end':None}},
        'DJ': {
            'DJ2': {'start':None, 'end':None}, 'DJ3': {'start':None, 'end':None}, 'DJ4': {'start':None, 'end':None}, 'DJAsym1': {'start':None, 'end':None}, 'DJAsym2': {'start':None, 'end':None}, 'DJAsym3': {'start':None, 'end':None}}
        },
    'subject8': {
        'walking': {
            'walking1': {'start':-1, 'end':1.83}, 'walking2': {'start':-0.7, 'end':1.89}, 'walking3': {'start':-0.7, 'end':1.92}, 
            'walkingTS1': {'start':-0.7, 'end':2.3}, 'walkingTS2': {'start':-1.0, 'end':2.06}, 'walkingTS3': {'start':-0.7, 'end':1.9}},
        'STS':     {
            'STS1': {'start':None, 'end':None}, 'STSweakLegs1': {'start':None, 'end':None}},
        'Squats':{
            'squats1': {'start':None, 'end':None}, 'squatsAsym1': {'start':None, 'end':None}},
        'DJ': {
            'DJ1': {'start':None, 'end':None}, 'DJ2': {'start':None, 'end':None}, 'DJ3': {'start':None, 'end':None}, 'DJAsym1': {'start':None, 'end':None}, 'DJAsym2': {'start':None, 'end':None}, 'DJAsym3': {'start':None, 'end':None}}
        },
    'subject9': {
        'walking': {
            'walking1': {'start':-0.6, 'end':1.65}, 'walking2': {'start':-0.5, 'end':1.55}, 'walking3': {'start':-0.6, 'end':1.6}, 
            'walkingTS1': {'start':-0.7, 'end':1.68}, 'walkingTS2': {'start':-0.7, 'end':1.63}, 'walkingTS3': {'start':-0.7, 'end':1.56}},
        'STS':     {'STS1': {'start':None, 'end':None}, 'STSweakLegs1': {'start':None, 'end':None}},
        'Squats':{
            'squats1': {'start':None, 'end':None}, 'squatsAsym1': {'start':None, 'end':None}},
        'DJ': {
            'DJ1': {'start':None, 'end':None}, 'DJ2': {'start':None, 'end':None}, 'DJ3': {'start':None, 'end':None}, 'DJAsym1': {'start':None, 'end':None}, 'DJAsym2': {'start':None, 'end':None}, 'DJAsym3': {'start':None, 'end':None}}
        },
    'subject10': {
        'walking': {
            'walking1': {'start':-0.7, 'end':1.46}, 'walking2': {'start':-0.7, 'end':1.49}, 'walking3': {'start':-0.7, 'end':1.5}, 
            'walkingTS1': {'start':-0.8, 'end':1.78}, 'walkingTS2': {'start':-0.9, 'end':1.85}, 'walkingTS3': {'start':-0.8, 'end':1.66}},
        'STS':     {
            'STS1': {'start':None, 'end':None}, 'STSweakLegs1': {'start':None, 'end':None}},
        'Squats':{
            'squats1': {'start':None, 'end':None}, 'squatsAsym1': {'start':None, 'end':None}},
        'DJ': {
            'DJ1': {'start':None, 'end':None}, 'DJ2': {'start':None, 'end':None}, 'DJ3': {'start':None, 'end':None}, 'DJAsym1': {'start':None, 'end':None}, 'DJAsym2': {'start':None, 'end':None}, 'DJAsym3': {'start':None, 'end':None}}
        },
    'subject11': {
        'walking': {
            'walking2': {'start':-0.7, 'end':1.6}, 'walking3': {'start':-0.7, 'end':1.55}, 'walking4': {'start':-0.7, 'end':1.62}, 
            'walkingTS1': {'start':-0.7, 'end':1.9}, 'walkingTS2': {'start':-0.7, 'end':1.85}, 'walkingTS3': {'start':-0.7, 'end':1.9}},
        'STS':     {
            'STS1': {'start':None, 'end':None}, 'STSweakLegs1': {'start':None, 'end':None}},
        'Squats':{
            'squats1': {'start':None, 'end':None}, 'squatsAsym1': {'start':None, 'end':None}},
        'DJ': {
            'DJ1': {'start':None, 'end':None}, 'DJ4': {'start':None, 'end':None}, 'DJ5': {'start':None, 'end':None}, 'DJAsym3': {'start':None, 'end':None}, 'DJAsym4': {'start':None, 'end':None}, 'DJAsym5': {'start':None, 'end':None}}
        },
    }

# %% User-defined variables.
mocap_simulation = True
v1_simulation = False

# Settings for dynamic simulation.
# motion_style = 'DJ'
# cases = ['14']

# motion_style = 'Squats'
# repetitions = [1, 2, 3]
# # repetitions = [4] # for subject9-mocap
# cases = ['18']

motion_style = 'STS'
repetitions = [1, 2, 3]
cases = ['75']

# motion_style = 'walking'
# motion_type = 'walking_formulation2'
# cases = ['1']

runProblem = True
processInputs = True
runSimulation = True
solveProblem = False
analyzeResults = True
plotResults = False

# runProblem = False
# processInputs = True
# runSimulation = True
# solveProblem = True
# analyzeResults = False
# plotResults = True

if mocap_simulation:
    os_folder_name = 'Mocap'
elif v1_simulation:
    os_folder_name = 'OpenSimData_v1'
else:
    os_folder_name = 'OpenSimData'

for case in cases:
    if motion_style == 'DJ':
        if case == '0':
            motion_type = 'drop_jump'
        elif case == '1':
            motion_type = 'drop_jump'
            buffer_start = 0.1
            buffer_end = 0.3
        elif case == '2':
            motion_type = 'drop_jump'
            buffer_start = 0.1
            buffer_end = 1.0
        elif case == '3':
            motion_type = 'drop_jump'
            scaleIsometricMuscleForce = 2
        elif case == '4':
            motion_type = 'drop_jump'
            torque_driven_model = True
        elif case == '5':
            motion_type = 'drop_jump'
            ignoreFLV = True
        elif case == '6':
            motion_type = 'drop_jump'
            weight_activation = 10
        elif case == '7':
            motion_type = 'drop_jump'
            weight_activation = 100
            scaleIsometricMuscleForce = 2
        elif case == '8':
            motion_type = 'drop_jump'
            reserve_hip_adduction = 30
        elif case == '9':
            motion_type = 'drop_jump'
            reserve_hip_rotation = 30
        elif case == '10':
            motion_type = 'drop_jump'
            reserve_hip_rotation = 30
            weight_activation = 10
        elif case == '11':
            motion_type = 'drop_jump_formulation1'
            reserve_hip_rotation = 30
            weight_activation = 10
        elif case == '12':
            motion_type = 'drop_jump_formulation1'
            reserve_hip_rotation = 30
            weight_activation = 10
            weight_pelvis_tilt = 100
        elif case == '13':
            motion_type = 'drop_jump_formulation1'
            reserve_hip_rotation = 30
            weight_activation = 10
            weight_pelvis_tilt = 50
        elif case == '14':
            motion_type = 'drop_jump_formulation2'
            reserve_hip_rotation = 30
            weight_activation = 10
        

    elif motion_style == 'Squats':
        if case == '0':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats'
        elif case == '1':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_formulation0'
        elif case == '2':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_formulation0'
            weight_activation = 10
        elif case == '3':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats'
            ignoreFLV = True
        elif case == '4':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats'
            ignoreFLV = True
            weight_activation = 1
            scaleIsometricMuscleForce = 3
        elif case == '5':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_formulation0'
            ignoreFLV = True
        elif case == '6':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_formulation0'
            weight_activation = 200
        elif case == '7':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_formulation0'
            ignoreFLV = True
            weight_activation = 200
        elif case == '8':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats'
            reserve_hip_flexion = 30
        elif case == '9':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats'
            reserve_hip_adduction = 30
        elif case == '10':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_torque_driven'
        elif case == '11':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_torque_driven'
            meshDensity = 50
        elif case == '12':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_formulation1'
        elif case == '13':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats'
            meshDensity = 50
        elif case == '14':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_formulation1'
            meshDensity = 50
        elif case == '15':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_formulation2'
            meshDensity = 50
        elif case == '16':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_formulation3'
            meshDensity = 50
        elif case == '17':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_formulation4'
            meshDensity = 50
        elif case == '18':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'squats_formulation4'

    elif motion_style == 'walking':
        if case == '0':
            buffer_start = 0
            buffer_end = 0
        elif case == '1':
            buffer_start = 0.7
            buffer_end = 0.5
        elif case == '2': # Did the same one to compare start/end times
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
        elif case == '7':
            buffer_start = 0.7
            buffer_end = 0.5
            weight_activation = 20
        elif case == '8':
            buffer_start = 0.7
            buffer_end = 0.5
            weight_acceleration = 200
            # Not really useful.
        elif case == '9':
            buffer_start = 0.7
            buffer_end = 0.5
            weight_velocity = 10
            # Not really useful.
        elif case == '10':
            buffer_start = 0.7
            buffer_end = 0.5
            motion_type = 'walking_formulation3'
            # Add reserves for the mtp, bad.
        elif case == '11':
            buffer_start = 0.7
            buffer_end = 0.5
            use_same_weight_individual_coordinate_value_acceleration = True
            use_same_weight_individual_coordinate_value_speed = True
        elif case == '12': # Did the same one to compare start/end times
            buffer_start = 0.7
            buffer_end = 0.5

    elif motion_style == 'STS':
        if case == '0':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand'
            periodicSTS = True
        elif case == '1':
            buffer_start = 0.5
            buffer_end = 0.5
            motion_type = 'sit_to_stand_formulation2'
            periodicSTS = False
        elif case == '2':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation3'
            periodicSTS = False
        elif case == '3':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation4'
            periodicSTS = True
        elif case == '4':
            buffer_start = 0.5
            buffer_end = 0.5
            motion_type = 'sit_to_stand_formulation3'
            periodicSTS = False
        elif case == '5':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation5'
            periodicSTS = True
        elif case == '6':
            buffer_start = 0.5
            buffer_end = 0.5
            motion_type = 'sit_to_stand_formulation6'
            periodicSTS = False
        elif case == '7':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation7'
            periodicSTS = True
        elif case == '8':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
        elif case == '9':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
        elif case == '10':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
        elif case == '11':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
        elif case == '12':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            vGRFRatioTerm = 0.05
        elif case == '13':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            vGRFRatioTerm = 0.05
        elif case == '14':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            weight_activation = 1
        elif case == '15':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            weight_activation = 1
        elif case == '16':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 100
        elif case == '17':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 200
        elif case == '18':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
        elif case == '19':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
        elif case == '20':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 200
        elif case == '21':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
        elif case == '22':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            min_ratio_vGRF = False
        elif case == '23':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
        elif case == '24':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            periodic_mtp_angle = True
            reserve_mtp_angle = 100
        elif case == '25':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 100
        elif case == '26':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 50
        elif case == '27':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 10
        elif case == '28':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
        elif case == '29':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            vGRFRatioTerm = 0.2
        elif case == '30':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            vGRFRatioTerm = 0.5
        elif case == '31':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            vGRFRatioTerm = 1
        elif case == '32':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            vGRFRatioTerm = 10
        elif case == '33':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 10
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            vGRFRatioTerm = 0.5
        elif case == '34':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 20
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            vGRFRatioTerm = 0.5
        elif case == '35':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 10
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            vGRFRatioTerm = 1
        elif case == '36':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 20
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            vGRFRatioTerm = 1
        elif case == '37':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
        elif case == '38':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 200
        elif case == '39':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
        elif case == '40':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            stand_to_stand = True
            weight_acceleration = 500
        elif case == '41':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 1000
        elif case == '42':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            endSTS_to_endSTS = True
            weight_acceleration = 500
        elif case == '43':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            endSTS_to_endSTS = True
            weight_acceleration = 500
            weight_activation = 1
        elif case == '44':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            startSTSnoDelay_to_Stand = True
        elif case == '45':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500      
        elif case == '46':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation10'
            periodicSTS = False
        elif case == '47':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            weight_subtalar = 50
            weight_hip_rotation = 50
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
        elif case == '48':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 200
            weight_lumbar_extension = 100
            weight_subtalar = 50
            weight_hip_rotation = 50
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
        elif case == '49':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            weight_subtalar = 50
            weight_hip_rotation = 50
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
            torque_driven_model = True
        elif case == '50':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 200
            weight_lumbar_extension = 100
            weight_subtalar = 50
            weight_hip_rotation = 50
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
            torque_driven_model = True
        elif case == '51':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            weight_subtalar = 50
            weight_hip_rotation = 50
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
            torque_driven_model = True
            meshDensity = 50
        elif case == '52':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            weight_subtalar = 50
            weight_hip_rotation = 50
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
            torque_driven_model = True
            meshDensity = 50
            weight_hip_adduction = 100
        elif case == '53':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            weight_pelvis_ty = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500 
            torque_driven_model = True 
        elif case == '54':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            weight_pelvis_ty = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500 
            torque_driven_model = True 
            weight_velocity = 100
        elif case == '55':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
            scaleIsometricMuscleForce = 2
        elif case == '56':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
            scaleIsometricMuscleForce = 3
        elif case == '57':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
            ignoreFLV = True
        elif case == '58':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
            weight_velocity = 100
            ignoreFLV = True
        elif case == '59':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation8'
            periodicSTS = False
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
            ignoreFLV = True
            scaleIsometricMuscleForce = 2
        elif case == '60':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 5
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            weight_acceleration = 500
            ignoreFLV = True
        elif case == '61':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 20
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            vGRFRatioTerm = 0.5
            ignoreFLV = True
        elif case == '62':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation9'
            periodicSTS = True
            meshDensity = 100
            weight_pelvis_tilt = 500
            weight_position_tracking = 100
            weight_lumbar_extension = 100
            heel_vGRF_threshold = 20
            trackValueCoordinates = {'mtp_angle_r': 0, 'mtp_angle_l': 0}
            weight_trackValueTerm = 1
            vGRFRatioTerm = 0.5
            weight_activation = 100
        elif case == '63':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11' # without min_ratio_vGRF = True, todo remove in settings
            periodicSTS = False
        elif case == '64': # Same as 63, because min_ratio_vGRF is False by default
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = False
            vGRFRatioTerm = 0.1
        elif case == '65': # Same as 63, because min_ratio_vGRF is False by default
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = False
            vGRFRatioTerm = 0.2
        elif case == '66': # Same as 63, because min_ratio_vGRF is False by default
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = False
            vGRFRatioTerm = 0.5
        elif case == '67':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = False
            min_ratio_vGRF = True
            vGRFRatioTerm = 0.1
        elif case == '68':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = False
            min_ratio_vGRF = True
            vGRFRatioTerm = 0.2
        elif case == '69':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = False
            min_ratio_vGRF = True
            vGRFRatioTerm = 0.5
        elif case == '70':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = False
            min_ratio_vGRF = True
            vGRFRatioTerm = 1
        elif case == '71':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = True
            min_ratio_vGRF = True
            vGRFRatioTerm = 0.5
        elif case == '72':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = False
            min_ratio_vGRF = True
            vGRFRatioTerm = 0.5
            startSTSnoDelay_to_Stand = True
        elif case == '73':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = False
            min_ratio_vGRF = True
            vGRFRatioTerm = 0.5
            buffer_start = 0.3
        elif case == '74':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = False
            min_ratio_vGRF = True
            vGRFRatioTerm = 0.5
            startSTSnoDelay_to_endSTS = True
        elif case == '75':
            buffer_start = 0
            buffer_end = 0
            motion_type = 'sit_to_stand_formulation11'
            periodicSTS = False
            min_ratio_vGRF = True
            vGRFRatioTerm = 0.5
            startSTSnoDelay_to_endSTS = True
            heel_vGRF_threshold = 0
        
    # %% Gait segmentation and kinematic analysis.

    no_results = []
    for subject in subjects:

        sessionDir = os.path.join(dataFolder, subject)
        session_id = ''

        # pathData = os.path.join(dataFolder, subject, 'OpenSimData', 'Video', 'mmpose_0.8', '2-cameras', 'v0.63', 'IK', 'LaiArnoldModified2017_poly_withArms_weldHand')
        for count, trial_name in enumerate(list(trials[subject][motion_style].keys())):
            
            # if count != 3:
            #     continue
        
            # if count < 2:
            #     continue

            if not 'repetitions' in locals():
                nReps = 1
            else: 
                nReps = len(repetitions)

            for rep in range(nReps):
                
                # if rep != 0:
                    # continue

                if 'repetitions' in locals():
                    repetition = repetitions[rep]   
            
                if not '_video' in trial_name and not mocap_simulation:
                    trial_name += '_video'
                    
                # if trial_name == 'STS1_video' and repetition == 1:
                #     continue
                
                if runProblem:        

                        if 'repetitions' in locals():
                            print('Processing {}-{}-rep{} for dynamic simulation...'.format(subject, trial_name, repetition))
                        else:
                            print('Processing {}-{} for dynamic simulation...'.format(subject, trial_name))
                        if processInputs:
                            try:
                                if not 'repetition' in locals():
                                    repetition = None
                                if not 'periodicSTS' in locals():
                                    periodicSTS = False
                                if not 'stand_to_stand' in locals():
                                    stand_to_stand = False
                                if not 'endSTS_to_endSTS' in locals():
                                    endSTS_to_endSTS = False                                    
                                if not 'startSTSnoDelay_to_Stand' in locals():
                                    startSTSnoDelay_to_Stand = False
                                if not 'startSTSnoDelay_to_endSTS' in locals():
                                    startSTSnoDelay_to_endSTS = False
                                    
                                settings = processInputsOpenSimAD(
                                    baseDir, sessionDir, session_id, trial_name, 
                                    motion_type, os_folder_name=os_folder_name, repetition=repetition, periodicSTS=periodicSTS, stand_to_stand=stand_to_stand, endSTS_to_endSTS=endSTS_to_endSTS, startSTSnoDelay_to_Stand=startSTSnoDelay_to_Stand, startSTSnoDelay_to_endSTS=startSTSnoDelay_to_endSTS)
                                
                                pathMotionFile = os.path.join(dataFolder, subject, os_folder_name, 'Kinematics', trial_name + '.mot')
                                motion_file = storage_to_numpy(pathMotionFile)
                                full_time_window = [motion_file['time'][0], motion_file['time'][-1]]
                                
                                if 'DJ' in trial_name:
                                    # Get time interval from trimmed trial
                                    if mocap_simulation:
                                        pathTrimmedMotionFile = os.path.join(dataFolder, subject, 'OpenSimData', 'Kinematics_trimmed', trial_name + '_videoAndMocap.mot')
                                    else:
                                        pathTrimmedMotionFile = os.path.join(dataFolder, subject, os_folder_name, 'Kinematics_trimmed', trial_name.replace('_video', '_videoAndMocap') + '.mot')
                                    trimmed_motion_file = storage_to_numpy(pathTrimmedMotionFile)
                                    trimmed_time_window = [trimmed_motion_file['time'][0], trimmed_motion_file['time'][-1]]

                                    if 'buffer_start' not in locals():
                                        buffer_start = 0

                                    if 'buffer_end' not in locals():
                                        buffer_end = 0
        
                                    # Update time window
                                    time_start = np.round(max(trimmed_time_window[0] - buffer_start, full_time_window[0]),2)
                                    time_end = np.round(min(trimmed_time_window[1] + buffer_end, full_time_window[1]),2)
                                    buffer_start_applied = np.abs(np.round(time_start - trimmed_time_window[0], 2))
                                    buffer_end_applied = np.abs(np.round(time_end - trimmed_time_window[1], 2))
                                    settings['buffers'] = [round(float(buffer_start_applied),6),
                                                        round(float(buffer_end_applied),6)]
                                    time_window = [time_start, time_end]
                                    settings['timeInterval'] = [round(float(i),6) for i in time_window]
                                    settings['timeIntervalWithoutBuffers'] = [round(float(settings['timeInterval'][0] + settings['buffers'][0]),6),
                                                                            round(float(settings['timeInterval'][1] - settings['buffers'][1]),6)]  
                                
                                elif 'walking' in trial_name:
                                    # Get time interval from trimmed trial
                                    if mocap_simulation:
                                        pathTrimmedMotionFile = os.path.join(dataFolder, subject, 'OpenSimData', 'Kinematics_trimmed', trial_name + '_videoAndMocap.mot')
                                    else:
                                        pathTrimmedMotionFile = os.path.join(dataFolder, subject, os_folder_name, 'Kinematics_trimmed', trial_name.replace('_video', '_videoAndMocap') + '.mot')
                                    trimmed_motion_file = storage_to_numpy(pathTrimmedMotionFile)
                                    trimmed_time_window = [trimmed_motion_file['time'][0], trimmed_motion_file['time'][-1]]
        
                                    # Update time window
                                    if mocap_simulation:
                                        time_start = np.round(max(trimmed_time_window[0] - buffer_start, full_time_window[0], trials[subject][motion_style][trial_name]['start']),2)
                                        time_end = np.round(min(trimmed_time_window[1] + buffer_end, full_time_window[1], trials[subject][motion_style][trial_name]['end']),2)
                                    else:
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
                                        
                                elif 'STS' in trial_name:
                                    
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
                                        
                                elif 'squats' in trial_name.lower():
                                    
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
                                    
                                if 'meshDensity' in locals():
                                    settings['meshDensity'] = meshDensity
    
                                if 'vGRFRatioTerm' in locals():
                                    settings['weights']['vGRFRatioTerm'] = vGRFRatioTerm
    
                                if 'weight_activation' in locals():
                                    settings['weights']['activationTerm'] = weight_activation

                                if 'weight_pelvis_tilt' in locals():
                                    settings['coordinates_toTrack']['pelvis_tilt']['weight'] = weight_pelvis_tilt
                                    
                                if 'weight_position_tracking' in locals():
                                    settings['weights']['positionTrackingTerm'] = weight_position_tracking

                                if 'weight_lumbar_extension' in locals():
                                    settings['coordinates_toTrack']['lumbar_extension']['weight'] = weight_lumbar_extension

                                if 'min_ratio_vGRF' in locals():
                                    settings['min_ratio_vGRF'] = min_ratio_vGRF

                                if 'heel_vGRF_threshold' in locals():
                                    settings['heel_vGRF_threshold'] = heel_vGRF_threshold

                                if 'periodic_mtp_angle' in locals():
                                    if 'periodicConstraints' in settings:
                                        periodicCoordinates = settings['periodicConstraints']['coordinateValues']
                                        periodicCoordinates.append('mtp_angle_r')
                                        periodicCoordinates.append('mtp_angle_l')
                                        settings['periodicConstraints']['coordinateValues'] = periodicCoordinates
                                    else:
                                        settings['periodicConstraints'] = {}
                                        settings['periodicConstraints']['coordinateValues'] = ['mtp_angle_r', 'mtp_angle_r']
                                        
                                if 'reserve_mtp_angle' in locals():
                                    if 'reserveActuatorCoordinates' in settings:
                                        reserveActuatorCoordinates = settings['reserveActuatorCoordinates']
                                        reserveActuatorCoordinates['mtp_angle_r'] = reserve_mtp_angle
                                        reserveActuatorCoordinates['mtp_angle_l'] = reserve_mtp_angle
                                        settings['reserveActuatorCoordinates'] = reserveActuatorCoordinates
                                    else:
                                        reserveActuatorCoordinates = {}                                        
                                        reserveActuatorCoordinates['mtp_angle_r'] = reserve_mtp_angle
                                        reserveActuatorCoordinates['mtp_angle_l'] = reserve_mtp_angle
                                        settings['reserveActuatorCoordinates'] = reserveActuatorCoordinates
                                        settings['withReserveActuators'] = True
                                if 'reserve_hip_flexion' in locals():
                                    if 'reserveActuatorCoordinates' in settings:
                                        reserveActuatorCoordinates = settings['reserveActuatorCoordinates']
                                        reserveActuatorCoordinates['hip_flexion_r'] = reserve_hip_flexion
                                        reserveActuatorCoordinates['hip_flexion_l'] = reserve_hip_flexion
                                        settings['reserveActuatorCoordinates'] = reserveActuatorCoordinates
                                    else:
                                        reserveActuatorCoordinates = {}                                        
                                        reserveActuatorCoordinates['hip_flexion_r'] = reserve_hip_flexion
                                        reserveActuatorCoordinates['hip_flexion_l'] = reserve_hip_flexion
                                        settings['reserveActuatorCoordinates'] = reserveActuatorCoordinates
                                        settings['withReserveActuators'] = True
                                if 'reserve_hip_adduction' in locals():
                                    if 'reserveActuatorCoordinates' in settings:
                                        reserveActuatorCoordinates = settings['reserveActuatorCoordinates']
                                        reserveActuatorCoordinates['hip_adduction_r'] = reserve_hip_adduction
                                        reserveActuatorCoordinates['hip_adduction_l'] = reserve_hip_adduction
                                        settings['reserveActuatorCoordinates'] = reserveActuatorCoordinates
                                    else:
                                        reserveActuatorCoordinates = {}                                        
                                        reserveActuatorCoordinates['hip_adduction_r'] = reserve_hip_adduction
                                        reserveActuatorCoordinates['hip_adduction_l'] = reserve_hip_adduction
                                        settings['reserveActuatorCoordinates'] = reserveActuatorCoordinates
                                        settings['withReserveActuators'] = True
                                if 'reserve_hip_rotation' in locals():
                                    if 'reserveActuatorCoordinates' in settings:
                                        reserveActuatorCoordinates = settings['reserveActuatorCoordinates']
                                        reserveActuatorCoordinates['hip_rotation_r'] = reserve_hip_rotation
                                        reserveActuatorCoordinates['hip_rotation_l'] = reserve_hip_rotation
                                        settings['reserveActuatorCoordinates'] = reserveActuatorCoordinates
                                    else:
                                        reserveActuatorCoordinates = {}                                        
                                        reserveActuatorCoordinates['hip_rotation_r'] = reserve_hip_rotation
                                        reserveActuatorCoordinates['hip_rotation_l'] = reserve_hip_rotation
                                        settings['reserveActuatorCoordinates'] = reserveActuatorCoordinates
                                        settings['withReserveActuators'] = True
                                

                                # TODO
                                if 'trackValueCoordinates' in locals():
                                    settings['trackValueCoordinates'] = trackValueCoordinates

                                if 'weight_trackValueTerm' in locals():
                                    settings['weights']['trackValueTerm'] = weight_trackValueTerm
                                    
                                if 'weight_acceleration' in locals():
                                    settings['weights']['accelerationTrackingTerm'] = weight_acceleration

                                if 'weight_velocity' in locals():
                                    settings['weights']['velocityTrackingTerm'] = weight_velocity

                                if 'weight_subtalar' in locals():
                                    settings['coordinates_toTrack']['subtalar_angle_r']['weight'] = weight_subtalar
                                    settings['coordinates_toTrack']['subtalar_angle_l']['weight'] = weight_subtalar

                                if 'weight_hip_rotation' in locals():
                                    settings['coordinates_toTrack']['hip_rotation_r']['weight'] = weight_hip_rotation
                                    settings['coordinates_toTrack']['hip_rotation_l']['weight'] = weight_hip_rotation

                                if 'use_same_weight_individual_coordinate_value_acceleration' in locals():
                                    settings['use_same_weight_individual_coordinate_value_acceleration'] = True

                                if 'use_same_weight_individual_coordinate_value_speed' in locals():
                                    settings['use_same_weight_individual_coordinate_value_speed'] = True

                                if 'torque_driven_model' in locals():
                                    settings['torque_driven_model'] = torque_driven_model

                                if 'weight_hip_adduction' in locals():
                                    settings['coordinates_toTrack']['hip_adduction_r']['weight'] = weight_hip_adduction
                                    settings['coordinates_toTrack']['hip_adduction_l']['weight'] = weight_hip_adduction

                                if 'weight_pelvis_ty' in locals():
                                    settings['coordinates_toTrack']['pelvis_ty']['weight'] = weight_pelvis_ty

                                if 'scaleIsometricMuscleForce' in locals():
                                    settings['scaleIsometricMuscleForce'] = scaleIsometricMuscleForce

                                if 'ignoreFLV' in locals():
                                    settings['ignoreFLV'] = ignoreFLV
                                
                            except Exception as e:
                                print(f"Error setting up dynamic optimization for trial {trial_name}: {e}")
                                continue
                    
                        # Simulation.
                        if runSimulation:
                            try:
                                # print('Running dynamic simulation...')
                                run_tracking(baseDir, sessionDir, settings, os_folder_name=os_folder_name, case=case, 
                                            solveProblem=solveProblem, analyzeResults=analyzeResults)
                                test=1
                            except Exception as e:
                                tb_info = traceback.format_exc()
                                print(f"Error during dynamic optimization for trial {trial_name}: {e}\nTraceback: {tb_info}")
                                no_results.append(subject + '_' + trial_name)
                                continue
                    
                if plotResults:    

                    # Load settings
                    # Overwritting
                    # trial_name = 'squats1_video'
                    # trial_name = 'squatsAsym1_video'
                    # repetition = 1

                    pathResults = os.path.join(dataFolder, subject, os_folder_name, 'Dynamics', trial_name)
                    if 'repetition' in locals():
                        pathResults = os.path.join(dataFolder, subject, os_folder_name, 'Dynamics', trial_name + '_rep' + str(repetition))     
                    pathSettings = os.path.join(pathResults, 'Setup_{}.yaml'.format(case))
                    with open(pathSettings, 'r') as file:
                        settings = yaml.safe_load(file)


                    # plotResultsOpenSimAD(sessionDir, trial_name, settings, cases=['37', '44'], mainPlots=False, grfPlotOnly=False)
                    plotResultsOpenSimAD(sessionDir, trial_name, os_folder_name, settings, cases=['1'], mainPlots=False, grfPlotOnly=False,
                                         momentArmsPlots=False, detailedMusclePlots=False)
                
                test=1

    print('No results for the following trials:')
    print(no_results)
