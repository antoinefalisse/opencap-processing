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


# %% Paths.
dataFolder = os.path.join(baseDir, 'Data', 'Benchmark')
subjects = ['subject' + str(i) for i in range(2, 3)]


trials = {'subject2': ['walking1', 'walking2', 'walking3', 'walking_TS1', 'walking_TS2', 'walking_TS4'],
          'subject3': ['walking1', 'walking2', 'walking3', 'walking_TS2', 'walking_TS3', 'walking_TS4'],
          'subject4': ['walking1', 'walking2', 'walking4', 'walking_TS1', 'walking_TS2', 'walking_TS3'],
          'subject5': ['walking1', 'walking2', 'walking3', 'walking_TS1', 'walking_TS2', 'walking_TS3'],
          'subject6': ['walking1', 'walking2', 'walking3', 'walking_TS1', 'walking_TS2', 'walking_TS3'],
          'subject7': ['walking1', 'walking2', 'walking3', 'walking_TS1', 'walking_TS2', 'walking_TS3'],
          'subject8': ['walking1', 'walking2', 'walking3', 'walking_TS1', 'walking_TS2', 'walking_TS3'],
          'subject9': ['walking1', 'walking2', 'walking3', 'walking_TS1', 'walking_TS2', 'walking_TS3'],
          'subject10': ['walking1', 'walking2', 'walking3', 'walking_TS1', 'walking_TS2', 'walking_TS3'],
          'subject11': ['walking1', 'walking2', 'walking4', 'walking_TS1', 'walking_TS2', 'walking_TS3'],
          }

# %% User-defined variables.
filter_frequency = 6

# Settings for dynamic simulation.
motion_type = 'walking_formulation2'
case = '0'
runProblem = True
processInputs = True
runSimulation = True
solveProblem = True
analyzeResults = True
plotResults = False
    
# %% Gait segmentation and kinematic analysis.

for subject in subjects:

    sessionDir = os.path.join(dataFolder, subject)
    session_id = ''

    pathData = os.path.join(dataFolder, subject, 'OpenSimData', 'Video', 'mmpose_0.8', '2-cameras', 'v0.63', 'IK', 'LaiArnoldModified2017_poly_withArms_weldHand')
    for trial_name in trials[subject]:
        
        trial_name += '_videoAndMocap'
        
        if runProblem:        

                print('Processing data for dynamic simulation...')
                if processInputs:
                    try:
                        settings = processInputsOpenSimAD(
                            baseDir, sessionDir, session_id, trial_name, 
                            motion_type)
                    except Exception as e:
                        print(f"Error setting up dynamic optimization for trial {trial_name}: {e}")
                        continue
            
                # Simulation.
                if runSimulation:
                    try:
                        run_tracking(baseDir, sessionDir, settings, case=case, 
                                    solveProblem=solveProblem, analyzeResults=analyzeResults)
                        test=1
                    except Exception as e:
                        print(f"Error during dynamic optimization for trial {trial_name}: {e}")
                        continue
            
        if plotResults:            
            plotResultsOpenSimAD(sessionDir, trial_name, cases=['0'], mainPlots=True)
        
test=1
