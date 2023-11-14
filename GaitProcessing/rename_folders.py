'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_gait_analysis.py
    ---------------------------------------------------------------------------
    Copyright 2023 Stanford University and the Authors
    
    Author(s): Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
                
    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to run a kinematic analysis of gait data. It works
    with either treadmill or overground gait. You can compute scalar metrics 
    as well as gait cycle-averaged kinematic curves.
    
'''

import os
import sys
sys.path.append("../")
baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)

import shutil

from utils import get_trial_id

from data_info import get_data_info

# %% Paths.
# driveDir = 'G:/.shortcut-targets-by-id/1PsjYe9HAdckqeTmAhxFd6F7Oad1qgZNy/ParkerStudy/'
driveDir = 'C:/MyDriveSym/Projects/ParkerStudy'
dataFolder = os.path.join(driveDir, 'Data')

# %% Gait segmentation and kinematic analysis.
ii = 1

trials_info = get_data_info(trial_indexes=[i for i in range(0,98)])

for trial in trials_info:
    # Get trial info.
    session_id = trials_info[trial]['sid']
    trial_name = trials_info[trial]['trial']
    print('Processing session {} - trial {}...'.format(session_id, trial_name))
    # Get trial id from name.
    trial_id = get_trial_id(session_id, trial_name)
    # Set session path.
    sessionDir = os.path.join(dataFolder, "{}_{}".format(trial, session_id))
    # Set kinematic folder path.
    pathOpenSimFolder = os.path.join(sessionDir, 'OpenSimData')
    pathOpenSimFolder_old = os.path.join(sessionDir, 'OpenSimData', 'Old')
    os.makedirs(pathOpenSimFolder_old, exist_ok=True)

    pathKinematicsFolder = os.path.join(pathOpenSimFolder, 'Kinematics')
    pathFeaturesFolder = os.path.join(pathOpenSimFolder, 'Features')
    pathDynamicsFolder = os.path.join(pathOpenSimFolder, 'Dynamics')
    pathModelFolder = os.path.join(pathOpenSimFolder, 'Model')

    pathKinematicsFolder_old = os.path.join(pathOpenSimFolder_old, 'Kinematics')
    pathFeaturesFolder_old = os.path.join(pathOpenSimFolder_old, 'Features')
    pathDynamicsFolder_old = os.path.join(pathOpenSimFolder_old, 'Dynamics')
    pathModelFolder_old = os.path.join(pathOpenSimFolder_old, 'Model')

    # if folders exist then rename them to <>_old
    if os.path.exists(pathKinematicsFolder):
        os.rename(pathKinematicsFolder, pathKinematicsFolder_old)
    if os.path.exists(pathFeaturesFolder):
        os.rename(pathFeaturesFolder, pathFeaturesFolder_old)
    if os.path.exists(pathDynamicsFolder):
        os.rename(pathDynamicsFolder, pathDynamicsFolder_old)
    if os.path.exists(pathModelFolder):
        os.rename(pathModelFolder, pathModelFolder_old)
    
    # Create new model folder
    pathExternalFunction = os.path.join(pathModelFolder, 'ExternalFunction')
    pathExternalFunction_old = os.path.join(pathModelFolder_old, 'ExternalFunction')
    if os.path.exists(pathExternalFunction_old):
        os.makedirs(pathExternalFunction)
        for file in os.listdir(pathExternalFunction_old):
            if file.endswith('.cpp') or file.endswith('.py') or file.endswith('.npy'):
                shutil.copy(os.path.join(pathExternalFunction_old, file), pathExternalFunction)
    for file in os.listdir(pathModelFolder_old):
        if file.endswith('.osim') or file.endswith('.log') or file.endswith('_mtParameters_l.npy') or file.endswith('_mtParameters_r.npy'):
            shutil.copy(os.path.join(pathModelFolder_old, file), pathModelFolder)


    