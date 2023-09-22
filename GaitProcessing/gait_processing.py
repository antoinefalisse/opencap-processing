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
sys.path.append("../ActivityAnalyses")
sys.path.append("../UtilsDynamicSimulations/OpenSimAD")
baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)
activityAnalysesDir = os.path.join(baseDir, 'ActivityAnalyses')
sys.path.append(activityAnalysesDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)

from gait_analysis import gait_analysis
from utils import get_trial_id, download_trial
from utilsPlotting import plot_dataframe

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking

from utilsKineticsOpenSimAD import kineticsOpenSimAD 

from utilsTRC import TRCFile

from scipy.spatial.transform import Rotation as R

# %% Paths.
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.
# session_id = 'bca0aad8-c129-4a62-bef3-b5de1659df5e'
# trial_name = '10mwt'

session_id = 'a08ec9d6-24f8-44f7-a59c-f603b7517e4d'
trial_name = '10mwrt_2'

scalar_names = {'gait_speed','stride_length','step_width','cadence',
                'single_support_time','double_support_time'}

# Select how many gait cycles you'd like to analyze. Select -1 for all gait
# cycles detected in the trial.
n_gait_cycles = 1 

# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

# Settings for dynamic simulation.
# motion_type = 'walking_periodic_torque_driven'
# case = '2'
# solveProblem = True
# analyzeResults = True
motion_type = 'walking_periodic'
case = '13'
solveProblem = True
analyzeResults = True


if case == '2' or case == '3':
    contact_configuration = 'dhondt2023'
else:
    contact_configuration = 'generic'

# %% Gait segmentation and kinematic analysis.
# Get trial id from name.
trial_id = get_trial_id(session_id,trial_name)    

# Set session path.
sessionDir = os.path.join(dataFolder, session_id)

# Download data.
trialName = download_trial(trial_id,sessionDir,session_id=session_id) 

# Data processing.
legs = ['r']
gait, gaitResults = {}, {}
for leg in legs:
    gaitResults[leg] = {}
    gait[leg] = gait_analysis(
        sessionDir, trialName, leg=leg,
        lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
        n_gait_cycles=n_gait_cycles)
    # Compute scalars.
    gaitResults[leg]['scalars'] = gait[leg].compute_scalars(scalar_names)
    # Get gait events.
    gaitResults[leg]['events'] = gait[leg].get_gait_events()

    pathTRCFile = os.path.join(sessionDir, 'MarkerData', trialName + '.trc')

    trc_file = TRCFile(pathTRCFile)
    
    C7 = trc_file.marker('C7_study')
    
    from scipy import signal
    import numpy as np
    from scipy.spatial.transform import Rotation
    peaks, _ = signal.find_peaks(C7[:,1], distance=10, width=10, prominence=0.05)
    diff_peaks = np.diff([peaks])
    # Select diff_peak from which the difference with the previous diff_peak is
    # lower than 10 percent. We take the last peak as reference.
    for i in range(len(diff_peaks[0])-2, 0, -1):
        if np.abs(diff_peaks[0][i]-diff_peaks[0][i+1]) > 0.2*diff_peaks[0][i+1]:
            break
        
    peak_start = peaks[i+1]
    peak_end = peaks[-1]
    
    # Extract marker data
    pos_start = C7[peak_start, :]
    pos_end = C7[peak_end, :]
    
    # Calculate the original vector
    original_vector = pos_end - pos_start
    
    vector_A = np.array([1, 0, 0])  # Replace with your vector's coordinates
    vector_B = original_vector # Replace with your vector's coordinates
    
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector_A, vector_B)
    
    # Calculate the magnitudes (lengths) of the vectors
    magnitude_A = np.linalg.norm(vector_A)
    magnitude_B = np.linalg.norm(vector_B)
    
    # Calculate the angle between the two vectors in radians
    angle_rad = np.arccos(dot_product / (magnitude_A * magnitude_B))
    
    # Convert the angle from radians to degrees if needed
    angle_deg = np.degrees(angle_rad)
    
    # Print the angle in radians and degrees
    print("Angle between vectors (radians):", angle_rad)
    print("Angle between vectors (degrees):", angle_deg)
    
    trc_file.rotate('z', angle_deg)
    pathTRCFile_out = os.path.join(sessionDir, 'MarkerData', trialName + '_rotated.trc')
    trc_file.write(pathTRCFile_out)
    
    
    C7_out = trc_file.marker('C7_study')
    pos_start = C7_out[peak_start, :]
    pos_end = C7_out[peak_end, :]
        
    
    # def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
    #     """get rotation matrix between two vectors using scipy"""
    #     vec1 = np.reshape(vec1, (1, -1))
    #     vec2 = np.reshape(vec2, (1, -1))
    #     r = R.align_vectors(vec2, vec1)
    #     return r[0]
    
    # # Faster version of rotateArraySphere3
    # def rotateArray(data, ref_vec, unit_vec=np.array([0,0,0])):    
    #     assert np.mod(data.shape[1],3) == 0, 'wrong dimension rotateArray'
        
    #     if not np.any(unit_vec):
    #         unit_vec = data[0, :3]
        
    #     r_align = get_rotation_matrix(vec1=unit_vec, vec2=ref_vec)
    #     data_out = np.zeros((data.shape[0], data.shape[1]))
    #     for i in range(int(data.shape[1]/3)):    
    #         unit_vec_align = r_align.apply(data[:, i*3:(i+1)*3])
    #         data_out[:,i*3:(i+1)*3] = unit_vec_align
            
    #     return data_out, unit_vec
    
    # markers = trc_file.marker_names
    # for marker in markers:    
    #     data_out, _ = rotateArray(trc_file.marker(marker), original_vector, [1,0,0])
    #     trc_file.set_value(marker, data_out)
        
    # pathTRCFile_out = os.path.join(sessionDir, 'MarkerData', trialName + '_rotated.trc')
    # trc_file.write(pathTRCFile_out)
        
    
    
    # # Calculate the angle of rotation (in radians) to align with Y-axis
    # angle_rad = np.arctan2(-original_vector[0], original_vector[2])
    
    # # Create a Rotation object for the rotation about the X-axis
    # rotation = Rotation.from_euler('y', angle_rad)
    
    # # Apply the rotation to the original vector
    # rotated_vector = rotation.apply(original_vector)
    
    # # Calculate the new position of pos_end (new_point2)
    # new_point2 = pos_start + rotated_vector
    
    # # Print the rotated vector and new coordinates of pos_end
    # print("pos_start:", pos_start)
    # print("new_point2:", new_point2)
    
    
    # from scipy.spatial.transform import Rotation
    # original_vector = pos_end - pos_start
    # angle_rad = np.arctan2(original_vector[0], original_vector[2])
    # rotation = Rotation.from_euler('x', angle_rad)
    # rotated_vector = rotation.apply(original_vector)
    # new_point2 = pos_start + rotated_vector
    
    # angle_deg = angle_rad * 180 / np.pi    
    
    
    
    

#     # Apply the rotation to the original vector
    
    
#     # Calculate the new position of point2
    
    
    
    
#     trc_file.rotate('x', angle_deg)
    
    
#     pathTRCFile_out = os.path.join(sessionDir, 'MarkerData', trialName + '_rotated.trc')
#     trc_file.write(pathTRCFile_out)
    
#     # rotated_time_series_data = rotation.apply(time_series_data)
    




    # # Setup dynamic optimization problem.
    # time_window = [float(gaitResults[leg]['events']['ipsilateralTime'][0, 0]),
    #                 float(gaitResults[leg]['events']['ipsilateralTime'][0, -1])]
    # test= [1.2, 2.4] 
    # settings = processInputsOpenSimAD(
    #     baseDir, dataFolder, session_id, trial_name, 
    #     motion_type, time_window=time_window, 
    #     contact_configuration=contact_configuration)
    
    # settings['contact_configuration'] = contact_configuration
    # if case == '4':    
    #     settings['tendon_compliances'] =  {'soleus_r': 17.5, 'gaslat_r': 17.5, 'gasmed_r': 17.5,
    #                                         'soleus_l': 17.5, 'gaslat_l': 17.5, 'gasmed_l': 17.5}
    # if case == '5' or case == '6' or case == '7' or case == '11' or case == '12' or case == '13':
    #     settings['weights']['activationTerm'] = 10 

    # if case == '7':
    #     settings['weights']['accelerationTrackingTerm'] = 10 

    # if case == '8' or case == '11':
    #     settings['weights']['positionTrackingTerm'] = 20
    # if case == '9' or case == '12':
    #     settings['weights']['positionTrackingTerm'] = 50
    # if case == '10' or case == '13':
    #     settings['weights']['positionTrackingTerm'] = 100
        
    # # Simulation.
    # run_tracking(baseDir, dataFolder, session_id, settings, case=case, 
    #               solveProblem=solveProblem, analyzeResults=analyzeResults)

# plotResultsOpenSimAD(dataFolder, session_id, trial_name, cases=['9', '10'])
    # test=1

# # %% Print scalar results.
# print('\nRight foot gait metrics:')
# print('(units: m and s)')
# print('-----------------')
# for key, value in gaitResults['scalars_r'].items():
#     rounded_value = round(value, 2)
#     print(f"{key}: {rounded_value}")
    
# print('\nLeft foot gait metrics:')
# print('(units: m and s)')
# print('-----------------')
# for key, value in gaitResults['scalars_l'].items():
#     rounded_value = round(value, 2)
#     print(f"{key}: {rounded_value}")

    
# # %% You can plot multiple curves, in this case we compare right and left legs.
# plot_dataframe_with_shading(
#     [gaitResults['curves_r']['mean'], gaitResults['curves_l']['mean']],
#     [gaitResults['curves_r']['sd'], gaitResults['curves_l']['sd']],
#     leg = ['r','l'],
#     xlabel = '% gait cycle',
#     title = 'kinematics (m or deg)',
#     legend_entries = ['right','left'])