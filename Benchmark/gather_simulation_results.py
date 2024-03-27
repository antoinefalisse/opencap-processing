"""
This script gathers simulation results across activities, in order to reproduce
the results and figures in the "OpenCap: 3D human movement dynamics
from smartphone videos" paper.

Before running the script, you need to download the data from simtk.org/opencap. 
See the README in this folder for more details.

Authors: Scott Uhlrich, Antoine Falisse
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
# import glob
import yaml

baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)
# dataDir = os.path.join(baseDir, 'Data', 'Benchmark')
dataDir = os.path.join(baseDir, 'Data', 'Benchmark_mocap_updated')

# repoDir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
# dataDir = os.path.join(repoDir, 'Data')
# sys.path.append(repoDir) # utilities from base repository directory

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
tempKeys = ['0']


# %% User inputs
fieldStudy = False # True to process field study results, false to process LabValidation results
saveResults = True

if fieldStudy:
    dataDir = dataDir.replace('LabValidation','FieldStudy')
    subjects = ['subject' + str(i) for i in range(100)]
    motion_types = ['squats','squatsAsym']
    
    # load trialnames
    trialFileName = os.path.join(os.path.abspath(os.path.dirname(__file__)),'fieldStudyTrialNames.yml')
    with open(trialFileName, "r") as stream:
        trialDict = yaml.safe_load(stream)
        
else:
    subjects = list(trials.keys())
    
    # motion_style = 'walking'
    # motion_types = ['walking', 'walkingTS']
    
    # motion_style = 'STS'
    # motion_types = ['STS','STSweakLegs']

    # motion_style = 'Squats'
    # motion_types = ['squats','squatsAsym']

    motion_style = 'DJ'
    motion_types = ['DJ','DJAsym']

    # motion_types = ['DJ', 'DJAsym', 'walking', 'walkingTS', 'squats','squatsAsym','STS','STSweakLegs']
    
                
# Fixed settings used in the paper
data_type = 'Video'
poseDetector = 'HRNet' # HRNet, OpenPose_default, or OpenPose_default
cameraSetup = '2-cameras' # 2-cameras, 3-cameras, 4-cameras
invert_left_right = False

for iSub, subject in enumerate(subjects):
    
    print('Processing subject {} of {}: '.format(iSub+1,len(subjects)) + subject)
    
    for motion_type in motion_types:
        
        # Get filenames corresponding to this motion type
        
        if fieldStudy:        
            osDir = os.path.join(dataDir,'FieldStudy', subject, 'OpenSimData')
            pathOSData = os.path.join(osDir,'Dynamics')
            
            # Check for data folder
            if not os.path.isdir(pathOSData):
                raise Exception('The data is not found in ' + dataDir + '. Download it from https://simtk.org/projects/opencap, and save to the the repository directory. E.g., Data/FieldStudy')
            
            trialNames = trialDict[subject][motion_type]
        else:            
            osDir = os.path.join(dataDir, subject, 'OpenSimData')
            pathOSData = os.path.join(osDir, 'Dynamics')
            
            # Check for data folder
            if not  os.path.isdir(pathOSData):
                raise Exception('The data is not found in ' + dataDir + '. Download it from https://simtk.org/projects/opencap, and save to the the repository directory. E.g., Data/LabValidation')

            trialNamesAll = [i + '_video' for i in trials[subject][motion_style]]

            for i in trialNamesAll:
                if motion_type == 'walking':
                    trialNames = [i for i in trialNamesAll if motion_type in i and not 'walkingTS' in i]
                elif motion_type == 'STS':
                    trialNames_temp = [i for i in trialNamesAll if motion_type in i and not 'STSweakLegs' in i]
                elif motion_type == 'STSweakLegs':
                    trialNames_temp = [i for i in trialNamesAll if motion_type in i]
                elif motion_type == 'squats':
                    trialNames_temp = [i for i in trialNamesAll if motion_type in i and not 'Asym' in i]
                elif motion_type == 'squatsAsym':
                    trialNames_temp = [i for i in trialNamesAll if motion_type in i]
                elif motion_type == 'DJ':
                    trialNames = [i for i in trialNamesAll if motion_type in i and not 'DJAsym' in i]
                else:
                    trialNames = [i for i in trialNamesAll if motion_type in i]

            if motion_type == 'STS' or motion_type == 'STSweakLegs' or motion_type == 'squats' or motion_type == 'squatsAsym':
                trialNames = [i + '_rep' + str(j) for i in trialNames_temp for j in range(1,4)]

            # Continue if trialNames is empty
            if not trialNames:
                continue
            print("{} trials found: {}".format(len(trialNames),trialNames))
                    
            # print(trialNames)
            
            
            # possiblePaths = glob.glob(os.path.join(pathOSData,motion_type+'*/'))
            # trialNames = []
            # for pP in possiblePaths:
            #     _,folderName = os.path.split(pP[:-1])
            #     if not folderName[len(motion_type)].isalpha(): # check if there aren't add'l letters in name
            #         trialNames.append(folderName)
            # trialNames.sort()
        
        
        # %% Load optimal trajectories
        optimaltrajectories = {}
        for tName in trialNames:
                       
            try:
                c_tr = np.load(os.path.join(pathOSData,tName,
                                            'optimaltrajectories.npy'),
                               allow_pickle=True).item()
                # TODO
                optimaltrajectories[tName] = c_tr[tempKeys[0]]
            except:
                print('No optimal trajectories found for {} - {}'.format(subject,tName))
                continue
            # TODO
            # tempKeys = list(c_tr.keys())
            
            
            # case_toPlot = list(c_tr[tempKeys[0]].keys())[0].replace('_videoAndMocap','')
            
            
        
        # %% process settings
        cases = list(optimaltrajectories.keys())
            
        # %% Visualize results
        plt.close('all')
        
        # %%
        jointsOpposite = ['pelvis_list', 'pelvis_rotation', 'pelvis_tz',
                          'lumbar_bending', 'lumbar_rotation',]
        
        jointsInverse = ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
                         'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                         'knee_angle_l', 'knee_angle_r',
                         'ankle_angle_l', 'ankle_angle_r',
                         'subtalar_angle_l', 'subtalar_angle_r',
                         'arm_flex_l', 'arm_add_l', 'arm_rot_l',
                         'arm_flex_r', 'arm_add_r', 'arm_rot_r',
                         'elbow_flex_l', 'elbow_flex_r',
                         'pro_sup_l', 'pro_sup_r']
        
        grfOpposite = ['ground_force_r_vz', 'ground_force_l_vz']
        
        grfInverse = ['ground_force_r_vx', 'ground_force_r_vy',
                      'ground_force_l_vx', 'ground_force_l_vy']
        
        grmInverse = ['ground_force_r_vx', 'ground_force_r_vy', 'ground_force_r_vz',
                      'ground_force_l_vx', 'ground_force_l_vy', 'ground_force_l_vz']
        
        JR_labels = ['KAM_r', 'KAM_l']
        MCF_labels = ['MCF_r', 'MCF_l']
        jrOpposite = []
        jrInverse = []
               
        # %% Joint coordinates
        joints = optimaltrajectories[cases[0]]['coordinates']
        NJoints = len(joints)
        rotationalJoints = optimaltrajectories[cases[0]]['rotationalCoordinates']
        # trials = optimaltrajectories[cases[0]]['time'].keys()
        ny = np.ceil(np.sqrt(NJoints))   
        fig, axs = plt.subplots(int(ny), int(ny), sharex=True)
        # for trial in trials:
            
        kinematics = {}
        positions = {}
        for case in cases:
            positions[case] = {}    
            # trial = list(optimaltrajectories[case]['time'].keys())[0]
            time = optimaltrajectories[case]['time'][0,:-1].T    
            positions[case]['toTrack'] = np.zeros((NJoints+1, time.shape[0]))
            positions[case]['ref'] = np.zeros((NJoints+1, time.shape[0]))
            positions[case]['sim'] = np.zeros((NJoints+1, time.shape[0]))
            positions[case]['headers'] = ['time']
            # Adding this to be able to segment STS when analyzing results 
            if 'timeIntervalRising' in optimaltrajectories[case]:
                positions[case]['timeIntervalRising'] = optimaltrajectories[case]['timeIntervalRising']                
            
        fig.suptitle('Joint positions: DC vs IK ')
        for i, ax in enumerate(axs.flat):
            if i < NJoints:
                color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                if joints[i] in rotationalJoints:
                    scale_angles = 180 / np.pi
                else:
                    scale_angles = 1
                for case in cases:
                    c_col = next(color)
                    # trial = list(optimaltrajectories[case]['time'].keys())[0]
                    time = optimaltrajectories[case]['time'][0,:-1].T
                    
                    positions[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    positions[case]['toTrack'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    positions[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    
                    if invert_left_right:
                        if joints[i] in jointsOpposite:
                            c_curve_ref = -1 * optimaltrajectories[case]['coordinate_values_ref'][i:i+1,:].T * scale_angles
                            c_curve_toTrack = -1 * optimaltrajectories[case]['coordinate_values_toTrack'][i:i+1,:].T * scale_angles
                            c_curve_sim = -1 * optimaltrajectories[case]['coordinate_values'][i:i+1,:-1].T * scale_angles                    
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                            ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                            ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case)                
                        elif joints[i] in jointsInverse:                    
                            side = joints[i][-1]
                            if side == 'r':
                                inv_side = 'l'
                            elif side == 'l':
                                inv_side = 'r'
                            else:
                                raise ValueError("Error side")                    
                            inv_joint = joints[i][:-1] + inv_side
                            inv_i = joints.index(inv_joint)                        
                            c_curve_ref = optimaltrajectories[case]['coordinate_values_ref'][inv_i:inv_i+1,:].T * scale_angles
                            c_curve_toTrack = optimaltrajectories[case]['coordinate_values_toTrack'][inv_i:inv_i+1,:].T * scale_angles
                            c_curve_sim = optimaltrajectories[case]['coordinate_values'][inv_i:inv_i+1,:-1].T * scale_angles
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                            ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                            ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case)    
                        else:
                            c_curve_ref = optimaltrajectories[case]['coordinate_values_ref'][i:i+1,:].T * scale_angles
                            c_curve_toTrack = optimaltrajectories[case]['coordinate_values_toTrack'][i:i+1,:].T * scale_angles
                            c_curve_sim = optimaltrajectories[case]['coordinate_values'][i:i+1,:-1].T * scale_angles                    
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                            ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                            ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case)                 
                    else:
                        if 'coordinate_values_ref' in optimaltrajectories[case]:
                            c_curve_ref = optimaltrajectories[case]['coordinate_values_ref'][i:i+1,:].T * scale_angles
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                        c_curve_toTrack = optimaltrajectories[case]['coordinate_values_toTrack'][i:i+1,:].T * scale_angles
                        c_curve_sim = optimaltrajectories[case]['coordinate_values'][i:i+1,:-1].T * scale_angles                                        
                        ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                        ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case)
                    if 'coordinate_values_ref' in optimaltrajectories[case]:
                        positions[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                    positions[case]['toTrack'][i+1:i+2, :] = c_curve_toTrack.T
                    positions[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                    positions[case]['headers'].append(joints[i])        
                
                ax.set_title(joints[i])
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
        plt.setp(axs[-1, :], xlabel='Time (s)')
        plt.setp(axs[:, 0], ylabel='(deg or m)')
        fig.align_ylabels()
            
        # %% Joint speeds
        velocities = {}
        for case in cases:
            velocities[case] = {}    
            # trial = list(optimaltrajectories[case]['time'].keys())[0]
            time = optimaltrajectories[case]['time'][0,:-1].T    
            velocities[case]['ref'] = np.zeros((NJoints+1, time.shape[0]))
            velocities[case]['toTrack'] = np.zeros((NJoints+1, time.shape[0]))
            velocities[case]['sim'] = np.zeros((NJoints+1, time.shape[0]))
            velocities[case]['headers'] = ['time']       
        
        fig, axs = plt.subplots(int(ny), int(ny), sharex=True)
        # for trial in trials:
        fig.suptitle('Joint speeds: DC vs IK') 
        for i, ax in enumerate(axs.flat):
            if i < NJoints:
                color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                if joints[i] in rotationalJoints:
                    scale_angles = 180 / np.pi
                else:
                    scale_angles = 1
                for case in cases:
                    c_col = next(color)
                    # trial = list(optimaltrajectories[case]['time'].keys())[0]
                    time = optimaltrajectories[case]['time'][0,:-1].T
                    
                    velocities[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    velocities[case]['toTrack'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    velocities[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    
                    if invert_left_right:
                        if joints[i] in jointsOpposite:                    
                            c_curve_ref = -1 * optimaltrajectories[case]['coordinate_speeds_ref'][i:i+1,:].T * scale_angles
                            c_curve_toTrack = -1 * optimaltrajectories[case]['coordinate_speeds_toTrack'][i:i+1,:].T * scale_angles
                            c_curve_sim = -1 * optimaltrajectories[case]['coordinate_speeds'][i:i+1,:-1].T * scale_angles                    
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                            ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                            ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case)                
                        elif joints[i] in jointsInverse:                    
                            side = joints[i][-1]
                            if side == 'r':
                                inv_side = 'l'
                            elif side == 'l':
                                inv_side = 'r'
                            else:
                                raise ValueError("Error side")                    
                            inv_joint = joints[i][:-1] + inv_side
                            inv_i = joints.index(inv_joint)                        
                            c_curve_ref = optimaltrajectories[case]['coordinate_speeds_ref'][inv_i:inv_i+1,:].T * scale_angles
                            c_curve_toTrack = optimaltrajectories[case]['coordinate_speeds_toTrack'][inv_i:inv_i+1,:].T * scale_angles
                            c_curve_sim = optimaltrajectories[case]['coordinate_speeds'][inv_i:inv_i+1,:-1].T * scale_angles
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                            ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                            ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case) 
                        else:
                            c_curve_ref = optimaltrajectories[case]['coordinate_speeds_ref'][i:i+1,:].T * scale_angles
                            c_curve_toTrack = optimaltrajectories[case]['coordinate_speeds_toTrack'][i:i+1,:].T * scale_angles
                            c_curve_sim = optimaltrajectories[case]['coordinate_speeds'][i:i+1,:-1].T * scale_angles
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                            ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                            ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case)              
                    else:
                        if 'coordinate_speeds_ref' in optimaltrajectories[case]:
                            c_curve_ref = optimaltrajectories[case]['coordinate_speeds_ref'][i:i+1,:].T * scale_angles
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                        c_curve_toTrack = optimaltrajectories[case]['coordinate_speeds_toTrack'][i:i+1,:].T * scale_angles
                        c_curve_sim = optimaltrajectories[case]['coordinate_speeds'][i:i+1,:-1].T * scale_angles
                        ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                        ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case)        
                    if 'coordinate_speeds_ref' in optimaltrajectories[case]:    
                        velocities[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                    velocities[case]['toTrack'][i+1:i+2, :] = c_curve_toTrack.T
                    velocities[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                    velocities[case]['headers'].append(joints[i])                
                
                ax.set_title(joints[i])
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
        plt.setp(axs[-1, :], xlabel='Time (s)')
        plt.setp(axs[:, 0], ylabel='(deg/s or m/s)')
        fig.align_ylabels()
            
        # %% Joint accelerations
        accelerations = {}
        for case in cases:
            accelerations[case] = {}    
            # trial = list(optimaltrajectories[case]['time'].keys())[0]
            time = optimaltrajectories[case]['time'][0,:-1].T    
            accelerations[case]['ref'] = np.zeros((NJoints+1, time.shape[0]))
            accelerations[case]['toTrack'] = np.zeros((NJoints+1, time.shape[0]))
            accelerations[case]['sim'] = np.zeros((NJoints+1, time.shape[0]))
            accelerations[case]['headers'] = ['time']    
        
        fig, axs = plt.subplots(int(ny), int(ny), sharex=True)
        # for trial in trials:
        fig.suptitle('Joint accelerations: DC vs IK') 
        for i, ax in enumerate(axs.flat):
            if i < NJoints:
                color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                if joints[i] in rotationalJoints:
                    scale_angles = 180 / np.pi
                else:
                    scale_angles = 1
                for case in cases:
                    c_col = next(color)
                    # trial = list(optimaltrajectories[case]['time'].keys())[0]
                    time = optimaltrajectories[case]['time'][0,:-1].T
                    accelerations[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    accelerations[case]['toTrack'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    accelerations[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    
                    if invert_left_right:
                        if joints[i] in jointsOpposite:                    
                            c_curve_ref = -1 * optimaltrajectories[case]['coordinate_accelerations_ref'][i:i+1,:].T * scale_angles
                            c_curve_toTrack = -1 * optimaltrajectories[case]['coordinate_accelerations_toTrack'][i:i+1,:].T * scale_angles
                            c_curve_sim = -1 * optimaltrajectories[case]['coordinate_accelerations'][i:i+1,:].T * scale_angles                    
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                            ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                            ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case)                 
                        elif joints[i] in jointsInverse:                    
                            side = joints[i][-1]
                            if side == 'r':
                                inv_side = 'l'
                            elif side == 'l':
                                inv_side = 'r'
                            else:
                                raise ValueError("Error side")                    
                            inv_joint = joints[i][:-1] + inv_side
                            inv_i = joints.index(inv_joint)                        
                            c_curve_ref = optimaltrajectories[case]['coordinate_accelerations_ref'][inv_i:inv_i+1,:].T * scale_angles
                            c_curve_toTrack = optimaltrajectories[case]['coordinate_accelerations_toTrack'][inv_i:inv_i+1,:].T * scale_angles
                            c_curve_sim = optimaltrajectories[case]['coordinate_accelerations'][inv_i:inv_i+1,:].T * scale_angles
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                            ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                            ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case) 
                        else:
                            c_curve_ref = optimaltrajectories[case]['coordinate_accelerations_ref'][i:i+1,:].T * scale_angles
                            c_curve_toTrack = optimaltrajectories[case]['coordinate_accelerations_toTrack'][i:i+1,:].T * scale_angles
                            c_curve_sim = optimaltrajectories[case]['coordinate_accelerations'][i:i+1,:].T * scale_angles                    
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                            ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                            ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case)               
                    else:
                        if 'coordinate_accelerations_ref' in optimaltrajectories[case]:
                            c_curve_ref = optimaltrajectories[case]['coordinate_accelerations_ref'][i:i+1,:].T * scale_angles
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based IK ' + case)
                        c_curve_toTrack = optimaltrajectories[case]['coordinate_accelerations_toTrack'][i:i+1,:].T * scale_angles
                        c_curve_sim = optimaltrajectories[case]['coordinate_accelerations'][i:i+1,:].T * scale_angles                        
                        ax.plot(time, c_curve_toTrack, c=c_col, label='video-based IK ' + case, linestyle='dashed')
                        ax.plot(time, c_curve_sim, c=c_col, label='video-based DC ' + case)        
                    if 'coordinate_accelerations_ref' in optimaltrajectories[case]:
                        accelerations[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                    accelerations[case]['toTrack'][i+1:i+2, :] = c_curve_toTrack.T
                    accelerations[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                    accelerations[case]['headers'].append(joints[i])
                    
                    ax.set_title(joints[i])
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
        plt.setp(axs[-1, :], xlabel='Time (s)')
        plt.setp(axs[:, 0], ylabel='(deg/s2 or m/s2)')
        fig.align_ylabels()
            
        # %% Joint torques
        torques = {}
        for case in cases:
            torques[case] = {}    
            # trial = list(optimaltrajectories[case]['time'].keys())[0]
            time = optimaltrajectories[case]['time'][0,:-1].T    
            torques[case]['ref'] = np.zeros((NJoints+1, time.shape[0]))
            torques[case]['sim'] = np.zeros((NJoints+1, time.shape[0]))
            torques[case]['headers'] = ['time']  
        
        fig, axs = plt.subplots(int(ny), int(ny), sharex=True)
        # for trial in trials:
        fig.suptitle('Joint torques: DC vs IK') 
        for i, ax in enumerate(axs.flat):
            if i < NJoints:
                color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                for case in cases:
                    
                    # trial = list(optimaltrajectories[case]['time'].keys())[0]
                    time = optimaltrajectories[case]['time'][0,:-1].T
                    
                    torques[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    torques[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    
                    if invert_left_right:
                        if joints[i] in jointsOpposite:
                            if 'torques_ref' in optimaltrajectories[case]:
                                c_curve_ref = -1 * optimaltrajectories[case]['torques_ref'][i:i+1,:].T
                            c_curve_sim = -1 * optimaltrajectories[case]['torques'][i:i+1,:].T  
                            if 'torques_ref' in optimaltrajectories[case]:
                                ax.plot(time, c_curve_ref, c='black', label='mocap-based ID ' + case)
                            ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)                
                        elif joints[i] in jointsInverse:                    
                            side = joints[i][-1]
                            if side == 'r':
                                inv_side = 'l'
                            elif side == 'l':
                                inv_side = 'r'
                            else:
                                raise ValueError("Error side")                    
                            inv_joint = joints[i][:-1] + inv_side
                            inv_i = joints.index(inv_joint)
                            if 'torques_ref' in optimaltrajectories[case]:
                                c_curve_ref = optimaltrajectories[case]['torques_ref'][inv_i:inv_i+1,:].T
                            c_curve_sim = optimaltrajectories[case]['torques'][inv_i:inv_i+1,:].T
                            if 'torques_ref' in optimaltrajectories[case]:
                                ax.plot(time, c_curve_ref, c='black', label='mocap-based ID ' + case)
                            ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                        else:
                            if 'torques_ref' in optimaltrajectories[case]:
                                c_curve_ref = optimaltrajectories[case]['torques_ref'][i:i+1,:].T
                            c_curve_sim = optimaltrajectories[case]['torques'][i:i+1,:].T   
                            if 'torques_ref' in optimaltrajectories[case]:
                                ax.plot(time, c_curve_ref, c='black', label='mocap-based ID ' + case)
                            ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)              
                    else:
                        if 'torques_ref' in optimaltrajectories[case]:
                            c_curve_ref = optimaltrajectories[case]['torques_ref'][i:i+1,:].T
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based ID ' + case)
                        c_curve_sim = optimaltrajectories[case]['torques'][i:i+1,:].T
                        ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                    if 'torques_ref' in optimaltrajectories[case]:
                        torques[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                    torques[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                    torques[case]['headers'].append(joints[i])
                    
                ax.set_title(joints[i])
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
        plt.setp(axs[-1, :], xlabel='Time (s)')
        plt.setp(axs[:, 0], ylabel='(Nm)')
        fig.align_ylabels()
        
        # %% Joint torques BWht
        torques_BWht = {}
        for case in cases:
            torques_BWht[case] = {}    
            # trial = list(optimaltrajectories[case]['time'].keys())[0]
            time = optimaltrajectories[case]['time'][0,:-1].T    
            torques_BWht[case]['ref'] = np.zeros((NJoints+1, time.shape[0]))
            torques_BWht[case]['sim'] = np.zeros((NJoints+1, time.shape[0]))
            torques_BWht[case]['headers'] = ['time']  
        
        fig, axs = plt.subplots(int(ny), int(ny), sharex=True)
        # for trial in trials:
        fig.suptitle('Joint torques_BWht: DC vs IK') 
        for i, ax in enumerate(axs.flat):
            if i < NJoints:
                color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                for case in cases:
                    
                    # trial = list(optimaltrajectories[case]['time'].keys())[0]
                    time = optimaltrajectories[case]['time'][0,:-1].T
                    
                    torques_BWht[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    torques_BWht[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    
                    if invert_left_right:
                        if joints[i] in jointsOpposite:
                            if 'torques_BWht_ref' in optimaltrajectories[case]:
                                c_curve_ref = -1 * optimaltrajectories[case]['torques_BWht_ref'][i:i+1,:].T
                            c_curve_sim = -1 * optimaltrajectories[case]['torques_BWht'][i:i+1,:].T  
                            if 'torques_BWht_ref' in optimaltrajectories[case]:
                                ax.plot(time, c_curve_ref, c='black', label='mocap-based ID ' + case)
                            ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)                
                        elif joints[i] in jointsInverse:                    
                            side = joints[i][-1]
                            if side == 'r':
                                inv_side = 'l'
                            elif side == 'l':
                                inv_side = 'r'
                            else:
                                raise ValueError("Error side")                    
                            inv_joint = joints[i][:-1] + inv_side
                            inv_i = joints.index(inv_joint)
                            if 'torques_BWht_ref' in optimaltrajectories[case]:
                                c_curve_ref = optimaltrajectories[case]['torques_BWht_ref'][inv_i:inv_i+1,:].T
                            c_curve_sim = optimaltrajectories[case]['torques_BWht'][inv_i:inv_i+1,:].T
                            if 'torques_BWht_ref' in optimaltrajectories[case]:
                                ax.plot(time, c_curve_ref, c='black', label='mocap-based ID ' + case)
                            ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                        else:
                            if 'torques_BWht_ref' in optimaltrajectories[case]:
                                c_curve_ref = optimaltrajectories[case]['torques_BWht_ref'][i:i+1,:].T
                            c_curve_sim = optimaltrajectories[case]['torques_BWht'][i:i+1,:].T   
                            if 'torques_BWht_ref' in optimaltrajectories[case]:
                                ax.plot(time, c_curve_ref, c='black', label='mocap-based ID ' + case)
                            ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)              
                    else:
                        if 'torques_BWht_ref' in optimaltrajectories[case]:
                            c_curve_ref = optimaltrajectories[case]['torques_BWht_ref'][i:i+1,:].T
                            ax.plot(time, c_curve_ref, c='black', label='mocap-based ID ' + case)
                        c_curve_sim = optimaltrajectories[case]['torques_BWht'][i:i+1,:].T
                        ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                    if 'torques_BWht_ref' in optimaltrajectories[case]:
                        torques_BWht[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                    torques_BWht[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                    torques_BWht[case]['headers'].append(joints[i])
                    
                ax.set_title(joints[i])
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
        plt.setp(axs[-1, :], xlabel='Time (s)')
        plt.setp(axs[:, 0], ylabel='(%BW_ht)')
        fig.align_ylabels()
            
        # %% GRFs
        GRFs = {}
        GRFs_peaks = {}
        GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
        NGRF = len(GRF_labels)
        for case in cases:
            GRFs[case] = {}
            GRFs_peaks[case] = {}
            # trial = list(optimaltrajectories[case]['time'].keys())[0]
            time = optimaltrajectories[case]['time'][0,:-1].T    
            GRFs[case]['ref'] = np.zeros((NGRF+1, time.shape[0]))
            GRFs[case]['sim'] = np.zeros((NGRF+1, time.shape[0]))
            GRFs[case]['headers'] = ['time'] 
        
        
        fig, axs = plt.subplots(2, 3, sharex=True)
        # for trial in trials:
        fig.suptitle('GRFs: DC vs IK') 
        for i, ax in enumerate(axs.flat):
            if i < NGRF:
                color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                plotedGRF = False
                for case in cases:
                    
                    # trial = list(optimaltrajectories[case]['time'].keys())[0]
                    time = optimaltrajectories[case]['time'][0,:-1].T
                    
                    GRFs[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    GRFs[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    
                    if invert_left_right:
                        if GRF_labels[i] in grfOpposite or GRF_labels[i] in grfInverse:
                            
                            side = GRF_labels[i][13]
                            if side == 'r':
                                inv_side = 'l'
                            elif side == 'l':
                                inv_side = 'r'
                            else:
                                raise ValueError("Error side")                    
                            inv_GRF = GRF_labels[i][:13] + inv_side + GRF_labels[i][14:]
                            inv_i = GRF_labels.index(inv_GRF)
                            
                            if GRF_labels[i] in grfOpposite:
                            
                                if 'GRF_ref' in optimaltrajectories[case]:
                                    plotedGRF = True
                                    c_curve_ref = -optimaltrajectories[case]['GRF_ref'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_ref, c='black', label='measured GRF ' + case)
                                c_curve_sim = -optimaltrajectories[case]['GRF_filt'][inv_i:inv_i+1,:].T
                                ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                                
                            elif GRF_labels[i] in grfInverse:
                                
                                if 'GRF_ref' in optimaltrajectories[case]:
                                    plotedGRF = True
                                    c_curve_ref = optimaltrajectories[case]['GRF_ref'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_ref, c='black', label='measured GRF ' + case)
                                c_curve_sim = optimaltrajectories[case]['GRF_filt'][inv_i:inv_i+1,:].T
                                ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                            
                        else:
                            raise ValueError("Error side")                
                    else:            
                        if 'GRF_ref' in optimaltrajectories[case]:
                            plotedGRF = True
                            c_curve_ref = optimaltrajectories[case]['GRF_ref'][i:i+1,:].T
                            ax.plot(time, c_curve_ref, c='black', label='measured GRF ' + case)
                        c_curve_sim = optimaltrajectories[case]['GRF_filt'][i:i+1,:].T
                        ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                        
                        if 'GRF_ref_peaks' in optimaltrajectories[case]:
                            c_peak_ref = optimaltrajectories[case]['GRF_ref_peaks']
                        
                    if 'GRF_ref' in optimaltrajectories[case]:
                        GRFs[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                    GRFs[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                    GRFs[case]['headers'].append(GRF_labels[i])
                    
                    if 'GRF_ref_peaks' in optimaltrajectories[case]:
                        GRFs_peaks[case]['ref'] = c_peak_ref   
                    
                ax.set_title(GRF_labels[i])
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
        plt.setp(axs[-1, :], xlabel='Time (s)')
        plt.setp(axs[:, 0], ylabel='(N)')
        fig.align_ylabels()
        
        # %% GRFs normalized
        GRFs_BW = {}
        GRFs_BW_peaks = {}
        for case in cases:
            GRFs_BW[case] = {}
            GRFs_BW_peaks[case] = {}
            # trial = list(optimaltrajectories[case]['time'].keys())[0]
            time = optimaltrajectories[case]['time'][0,:-1].T    
            GRFs_BW[case]['ref'] = np.zeros((NGRF+1, time.shape[0]))
            GRFs_BW[case]['sim'] = np.zeros((NGRF+1, time.shape[0]))
            GRFs_BW[case]['headers'] = ['time'] 
        
        GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
        NGRF = len(GRF_labels)
        fig, axs = plt.subplots(2, 3, sharex=True)
        # for trial in trials:
        fig.suptitle('GRFs_BW: DC vs IK') 
        for i, ax in enumerate(axs.flat):
            if i < NGRF:
                color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                plotedGRF = False
                for case in cases:
                    
                    # trial = list(optimaltrajectories[case]['time'].keys())[0]
                    time = optimaltrajectories[case]['time'][0,:-1].T
                    
                    GRFs_BW[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    GRFs_BW[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    
                    if invert_left_right:
                        if GRF_labels[i] in grfOpposite or GRF_labels[i] in grfInverse:
                            
                            side = GRF_labels[i][13]
                            if side == 'r':
                                inv_side = 'l'
                            elif side == 'l':
                                inv_side = 'r'
                            else:
                                raise ValueError("Error side")                    
                            inv_GRF = GRF_labels[i][:13] + inv_side + GRF_labels[i][14:]
                            inv_i = GRF_labels.index(inv_GRF)
                            
                            if GRF_labels[i] in grfOpposite:
                            
                                if 'GRF_BW_ref' in optimaltrajectories[case]:
                                    plotedGRF = True
                                    c_curve_ref = -optimaltrajectories[case]['GRF_BW_ref'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_ref, c='black', label='measured GRF ' + case)
                                c_curve_sim = -optimaltrajectories[case]['GRF_filt_BW'][inv_i:inv_i+1,:].T
                                ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                                
                            elif GRF_labels[i] in grfInverse:
                                
                                if 'GRF_BW_ref' in optimaltrajectories[case]:
                                    plotedGRF = True
                                    c_curve_ref = optimaltrajectories[case]['GRF_BW_ref'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_ref, c='black', label='measured GRF ' + case)
                                c_curve_sim = optimaltrajectories[case]['GRF_filt_BW'][inv_i:inv_i+1,:].T
                                ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                            
                        else:
                            raise ValueError("Error side")                
                    else:            
                        if 'GRF_BW_ref' in optimaltrajectories[case]:
                            plotedGRF = True
                            c_curve_ref = optimaltrajectories[case]['GRF_BW_ref'][i:i+1,:].T
                            ax.plot(time, c_curve_ref, c='black', label='measured GRF ' + case)
                        c_curve_sim = optimaltrajectories[case]['GRF_filt_BW'][i:i+1,:].T
                        ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                        
                        if 'GRF_BW_ref_peaks' in optimaltrajectories[case]:
                            c_peak_ref = optimaltrajectories[case]['GRF_BW_ref_peaks']
                        
                    if 'GRF_BW_ref' in optimaltrajectories[case]:
                        GRFs_BW[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                    GRFs_BW[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                    GRFs_BW[case]['headers'].append(GRF_labels[i])
                    
                    if 'GRF_BW_ref_peaks' in optimaltrajectories[case]:
                        GRFs_BW_peaks[case]['ref'] = c_peak_ref                        
                    
                ax.set_title(GRF_labels[i])
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
        plt.setp(axs[-1, :], xlabel='Time (s)')
        plt.setp(axs[:, 0], ylabel='(N)')
        fig.align_ylabels()
        
        # %% GRMs
        GRMs = {}
        for case in cases:
            GRMs[case] = {}    
            # trial = list(optimaltrajectories[case]['time'].keys())[0]
            time = optimaltrajectories[case]['time'][0,:-1].T    
            GRMs[case]['ref'] = np.zeros((NGRF+1, time.shape[0]))
            GRMs[case]['sim'] = np.zeros((NGRF+1, time.shape[0]))
            GRMs[case]['headers'] = ['time'] 
        
        GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
        NGRF = len(GRF_labels)
        fig, axs = plt.subplots(2, 3, sharex=True)
        # for trial in trials:
        fig.suptitle('GRMs: DC vs IK') 
        for i, ax in enumerate(axs.flat):
            if i < NGRF:
                color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                plotedGRM = False
                for case in cases:
                    
                    # trial = list(optimaltrajectories[case]['time'].keys())[0]
                    time = optimaltrajectories[case]['time'][0,:-1].T
                    
                    GRMs[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    GRMs[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    
                    if invert_left_right:
                        if GRF_labels[i] in grmInverse:
                            
                            side = GRF_labels[i][13]
                            if side == 'r':
                                inv_side = 'l'
                            elif side == 'l':
                                inv_side = 'r'
                            else:
                                raise ValueError("Error side")                    
                            inv_GRF = GRF_labels[i][:13] + inv_side + GRF_labels[i][14:]
                            inv_i = GRF_labels.index(inv_GRF)
                                
                            if GRF_labels[i] in grmInverse:
                                
                                if 'GRM_ref' in optimaltrajectories[case]:
                                    plotedGRM = True
                                    c_curve_ref = optimaltrajectories[case]['GRM_ref'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_ref, c='black', label='measured GRM ' + case)
                                c_curve_sim = optimaltrajectories[case]['GRM'][inv_i:inv_i+1,:].T
                                ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                            
                        else:
                            raise ValueError("Error side")                
                    else:            
                        if 'GRM_ref' in optimaltrajectories[case]:
                            plotedGRM = True
                            c_curve_ref = optimaltrajectories[case]['GRM_ref'][i:i+1,:].T
                            ax.plot(time, c_curve_ref, c='black', label='measured GRM ' + case)
                            GRMs[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                        c_curve_sim = optimaltrajectories[case]['GRM'][i:i+1,:].T
                        ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                    GRMs[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                    GRMs[case]['headers'].append(GRF_labels[i])
                        
                ax.set_title(GRF_labels[i])
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
        plt.setp(axs[-1, :], xlabel='Time (s)')
        plt.setp(axs[:, 0], ylabel='(%BW)')
        fig.align_ylabels()
        
        # %% GRMs_BWht
        GRMs_BWht = {}
        for case in cases:
            GRMs_BWht[case] = {}    
            # trial = list(optimaltrajectories[case]['time'].keys())[0]
            time = optimaltrajectories[case]['time'][0,:-1].T    
            GRMs_BWht[case]['ref'] = np.zeros((NGRF+1, time.shape[0]))
            GRMs_BWht[case]['sim'] = np.zeros((NGRF+1, time.shape[0]))
            GRMs_BWht[case]['headers'] = ['time'] 
        
        GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
        NGRF = len(GRF_labels)
        fig, axs = plt.subplots(2, 3, sharex=True)
        # for trial in trials:
        fig.suptitle('GRMs_BWht: DC vs IK') 
        for i, ax in enumerate(axs.flat):
            if i < NGRF:
                color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                plotedGRM = False
                for case in cases:
                    
                    # trial = list(optimaltrajectories[case]['time'].keys())[0]
                    time = optimaltrajectories[case]['time'][0,:-1].T
                    
                    GRMs_BWht[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    GRMs_BWht[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    
                    if invert_left_right:
                        if GRF_labels[i] in grmInverse:
                            
                            side = GRF_labels[i][13]
                            if side == 'r':
                                inv_side = 'l'
                            elif side == 'l':
                                inv_side = 'r'
                            else:
                                raise ValueError("Error side")                    
                            inv_GRF = GRF_labels[i][:13] + inv_side + GRF_labels[i][14:]
                            inv_i = GRF_labels.index(inv_GRF)
                                
                            if GRF_labels[i] in grmInverse:
                                
                                if 'GRM_BWht_ref' in optimaltrajectories[case]:
                                    plotedGRM = True
                                    c_curve_ref = optimaltrajectories[case]['GRM_BWht_ref'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_ref, c='black', label='measured GRM ' + case)
                                c_curve_sim = optimaltrajectories[case]['GRM_BWht'][inv_i:inv_i+1,:].T
                                ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                            
                        else:
                            raise ValueError("Error side")                
                    else:            
                        if 'GRM_BWht_ref' in optimaltrajectories[case]:
                            plotedGRM = True
                            c_curve_ref = optimaltrajectories[case]['GRM_BWht_ref'][i:i+1,:].T
                            ax.plot(time, c_curve_ref, c='black', label='measured GRM ' + case)
                            GRMs_BWht[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                        c_curve_sim = optimaltrajectories[case]['GRM_BWht'][i:i+1,:].T
                        ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)                    
                    GRMs_BWht[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                    GRMs_BWht[case]['headers'].append(GRF_labels[i])
                ax.set_title(GRF_labels[i])
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
        plt.setp(axs[-1, :], xlabel='Time (s)')
        plt.setp(axs[:, 0], ylabel='(%BW_ht)')
        fig.align_ylabels()
            
        # %% Muscle activations
        muscles = optimaltrajectories[cases[0]]['muscles']
        NMuscles = len(muscles)
        
        activations = {}
        for case in cases:
            activations[case] = {}    
            # trial = list(optimaltrajectories[case]['time'].keys())[0]
            time = optimaltrajectories[case]['time'][0,:-1].T
            if 'static_optimization_ref' in optimaltrajectories[case]:
                activations[case]['so'] = np.zeros((NMuscles+1, time.shape[0]))
            activations[case]['ref'] = np.zeros((NMuscles+1, time.shape[0]))
            activations[case]['sim'] = np.zeros((NMuscles+1, time.shape[0]))
            activations[case]['headers'] = ['time']
        
        
        ny = np.ceil(np.sqrt(NMuscles))   
        fig, axs = plt.subplots(int(ny), int(ny), sharex=True)
        fig.suptitle('Muscle activations: DC vs IK') 
        for i, ax in enumerate(axs.flat):
            if i < NMuscles:
                color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                for case in cases:
                    
                    # trial = list(optimaltrajectories[case]['time'].keys())[0]
                    time = optimaltrajectories[case]['time'][0,:-1].T
                    if 'static_optimization_ref' in optimaltrajectories[case]:
                        activations[case]['so'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    activations[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    activations[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                    
                    if invert_left_right:                    
                        side = muscles[i][-1]
                        if side == 'r':
                            inv_side = 'l'
                        elif side == 'l':
                            inv_side = 'r'
                        else:
                            raise ValueError("Error side")
                            
                        inv_muscle = muscles[i][:-1] + inv_side
                        inv_i = muscles.index(inv_muscle)                        
                        c_curve_sim = optimaltrajectories[case]['muscle_activations'][inv_i:inv_i+1,:-1].T                    
                        ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)                              
                    else:
                        if 'muscle_activations_ref' in optimaltrajectories[case]:
                            mWithoutEMG = np.isnan(optimaltrajectories[case]['muscle_activations_ref'][i,0])
                            if not mWithoutEMG:
                                c_curve_ref = optimaltrajectories[case]['muscle_activations_ref'][i:i+1,:].T
                                ax.plot(time, c_curve_ref, c='black', label='measured EMG ' + case)
                            else:
                                c_curve_ref = np.nan*np.ones((time.shape[0],1))
                            activations[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                        if 'static_optimization_ref' in optimaltrajectories[case]:
                            c_curve_so = optimaltrajectories[case]['static_optimization_ref'][i:i+1,:].T
                            ax.plot(time, c_curve_so, c='black', linestyle='dotted', label='SO ' + case)
                        c_curve_sim = optimaltrajectories[case]['muscle_activations'][i:i+1,:-1].T
                        ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                    
                    
                    activations[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                    if 'static_optimization_ref' in optimaltrajectories[case]:
                        activations[case]['so'][i+1:i+2, :] = c_curve_so.T
                    activations[case]['headers'].append(muscles[i])
                        
                ax.set_title(muscles[i])
                ax.set_ylim((0,1))
                handles, labels = ax.get_legend_handles_labels()
                # plt.legend(handles, labels, loc='upper right')
        plt.setp(axs[-1, :], xlabel='Time (s)')
        plt.setp(axs[:, 0], ylabel='()')
        fig.align_ylabels()
        
        # %% KAM
        if not fieldStudy and ('walking' in motion_type or 'DJ' in motion_type):
            KAMs = {}
            NKams = len(JR_labels)
            for case in cases:
                KAMs[case] = {}    
                # trial = list(optimaltrajectories[case]['time'].keys())[0]
                time = optimaltrajectories[case]['time'][0,:-1].T    
                KAMs[case]['ref'] = np.zeros((NKams+1, time.shape[0]))
                KAMs[case]['sim'] = np.zeros((NKams+1, time.shape[0]))
                KAMs[case]['headers'] = ['time'] 
            
            # GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
            
            fig, axs = plt.subplots(1, 2, sharex=True)
            # for trial in trials:
            fig.suptitle('KAMs: DC vs IK') 
            for i, ax in enumerate(axs.flat):
                if i < NKams:
                    color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                    plotedJR = False
                    for case in cases:
                        
                        # trial = list(optimaltrajectories[case]['time'].keys())[0]
                        time = optimaltrajectories[case]['time'][0,:-1].T
                        
                        KAMs[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                        KAMs[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                        
                        # invert_left_right = cases_toPlot[case]['invert_left_right']
                        if invert_left_right:
                            if JR_labels[i] in jrOpposite or JR_labels[i] in jrInverse:
                                
                                side = JR_labels[i][11]
                                if side == 'r':
                                    inv_side = 'l'
                                elif side == 'l':
                                    inv_side = 'r'
                                else:
                                    raise ValueError("Error side")                    
                                inv_JR = JR_labels[i][:11] + inv_side + JR_labels[i][12:]
                                inv_i = JR_labels.index(inv_JR)
                                
                                if JR_labels[i] in jrOpposite:
                                
                                    if 'jr_ref' in optimaltrajectories[case]:
                                        plotedJR = True
                                        c_curve_ref = -optimaltrajectories[case]['jr_ref'][inv_i:inv_i+1,:].T
                                        ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                                    c_curve_sim = -optimaltrajectories[case]['jr_opt'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                                    
                                elif JR_labels[i] in jrInverse:
                                    
                                    if 'jr_ref' in optimaltrajectories[case]:
                                        plotedJR = True
                                        c_curve_ref = optimaltrajectories[case]['jr_ref'][inv_i:inv_i+1,:].T
                                        ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                                    c_curve_sim = optimaltrajectories[case]['jr_opt'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                                
                            else:
                                raise ValueError("Error side")                
                        else:            
                            if 'KAM_ref' in optimaltrajectories[case]:
                                plotedJR = True
                                c_curve_ref = optimaltrajectories[case]['KAM_ref'][i:i+1,:].T
                                ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                            c_curve_sim = optimaltrajectories[case]['KAM'][i:i+1,:].T
                            ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                            
                        if 'KAM_ref' in optimaltrajectories[case]:
                            KAMs[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                        KAMs[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                        KAMs[case]['headers'].append(JR_labels[i])
                        
                    ax.set_xlabel('Time (s)')
                    ax.set_title(JR_labels[i])
                    handles, labels = ax.get_legend_handles_labels()
                    plt.legend(handles, labels, loc='upper right')
            # plt.setp(axs[0, :], xlabel='Time (s)')
            plt.setp(axs[0], ylabel='(N)')
            fig.align_ylabels()
        
        # %% KAM
        if not fieldStudy and ('walking' in motion_type or 'DJ' in motion_type):
            KAMs_BWht = {}
            NKams = len(JR_labels)
            for case in cases:
                KAMs_BWht[case] = {}    
                # trial = list(optimaltrajectories[case]['time'].keys())[0]
                time = optimaltrajectories[case]['time'][0,:-1].T    
                KAMs_BWht[case]['ref'] = np.zeros((NKams+1, time.shape[0]))
                KAMs_BWht[case]['sim'] = np.zeros((NKams+1, time.shape[0]))
                KAMs_BWht[case]['headers'] = ['time'] 
            
            # GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
            
            fig, axs = plt.subplots(1, 2, sharex=True)
            # for trial in trials:
            fig.suptitle('KAMs_BWht: DC vs IK') 
            for i, ax in enumerate(axs.flat):
                if i < NKams:
                    color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                    plotedJR = False
                    for case in cases:
                        
                        # trial = list(optimaltrajectories[case]['time'].keys())[0]
                        time = optimaltrajectories[case]['time'][0,:-1].T
                        
                        KAMs_BWht[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                        KAMs_BWht[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                        
                        # invert_left_right = cases_toPlot[case]['invert_left_right']
                        if invert_left_right:
                            if JR_labels[i] in jrOpposite or JR_labels[i] in jrInverse:
                                
                                side = JR_labels[i][11]
                                if side == 'r':
                                    inv_side = 'l'
                                elif side == 'l':
                                    inv_side = 'r'
                                else:
                                    raise ValueError("Error side")                    
                                inv_JR = JR_labels[i][:11] + inv_side + JR_labels[i][12:]
                                inv_i = JR_labels.index(inv_JR)
                                
                                if JR_labels[i] in jrOpposite:
                                
                                    if 'jr_ref' in optimaltrajectories[case]:
                                        plotedJR = True
                                        c_curve_ref = -optimaltrajectories[case]['jr_ref'][inv_i:inv_i+1,:].T
                                        ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                                    c_curve_sim = -optimaltrajectories[case]['jr_opt'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                                    
                                elif JR_labels[i] in jrInverse:
                                    
                                    if 'jr_ref' in optimaltrajectories[case]:
                                        plotedJR = True
                                        c_curve_ref = optimaltrajectories[case]['jr_ref'][inv_i:inv_i+1,:].T
                                        ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                                    c_curve_sim = optimaltrajectories[case]['jr_opt'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                                
                            else:
                                raise ValueError("Error side")                
                        else:            
                            if 'KAM_BWht_ref' in optimaltrajectories[case]:
                                plotedJR = True
                                c_curve_ref = optimaltrajectories[case]['KAM_BWht_ref'][i:i+1,:].T
                                ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                            c_curve_sim = optimaltrajectories[case]['KAM_BWht'][i:i+1,:].T
                            ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                            
                        if 'KAM_BWht_ref' in optimaltrajectories[case]:
                            KAMs_BWht[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                        KAMs_BWht[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                        KAMs_BWht[case]['headers'].append(JR_labels[i])
                        
                    ax.set_xlabel('Time (s)')
                    ax.set_title(JR_labels[i])
                    handles, labels = ax.get_legend_handles_labels()
                    plt.legend(handles, labels, loc='upper right')
            # plt.setp(axs[0, :], xlabel='Time (s)')
            plt.setp(axs[0], ylabel='(%BW_ht)')
            fig.align_ylabels()
        
        # %% MCF
        if not fieldStudy and 'walking' in motion_type:
            MCFs = {}
            NMCFs = len(MCF_labels)
            for case in cases:
                MCFs[case] = {}    
                # trial = list(optimaltrajectories[case]['time'].keys())[0]
                time = optimaltrajectories[case]['time'][0,:-1].T    
                MCFs[case]['ref'] = np.zeros((NMCFs+1, time.shape[0]))
                MCFs[case]['sim'] = np.zeros((NMCFs+1, time.shape[0]))
                MCFs[case]['headers'] = ['time'] 
            
            # GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
            
            fig, axs = plt.subplots(1, 2, sharex=True)
            # for trial in trials:
            fig.suptitle('MCFs: DC vs IK') 
            for i, ax in enumerate(axs.flat):
                if i < NMCFs:
                    color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                    plotedJR = False
                    for case in cases:
                        
                        # trial = list(optimaltrajectories[case]['time'].keys())[0]
                        time = optimaltrajectories[case]['time'][0,:-1].T
                        
                        MCFs[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                        MCFs[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                        
                        # invert_left_right = cases_toPlot[case]['invert_left_right']
                        if invert_left_right:
                            if MCF_labels[i] in jrOpposite or MCF_labels[i] in jrInverse:
                                
                                side = MCF_labels[i][11]
                                if side == 'r':
                                    inv_side = 'l'
                                elif side == 'l':
                                    inv_side = 'r'
                                else:
                                    raise ValueError("Error side")                    
                                inv_JR = MCF_labels[i][:11] + inv_side + MCF_labels[i][12:]
                                inv_i = MCF_labels.index(inv_JR)
                                
                                if MCF_labels[i] in jrOpposite:
                                
                                    if 'jr_ref' in optimaltrajectories[case]:
                                        plotedJR = True
                                        c_curve_ref = -optimaltrajectories[case]['jr_ref'][inv_i:inv_i+1,:].T
                                        ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                                    c_curve_sim = -optimaltrajectories[case]['jr_opt'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                                    
                                elif MCF_labels[i] in jrInverse:
                                    
                                    if 'jr_ref' in optimaltrajectories[case]:
                                        plotedJR = True
                                        c_curve_ref = optimaltrajectories[case]['jr_ref'][inv_i:inv_i+1,:].T
                                        ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                                    c_curve_sim = optimaltrajectories[case]['jr_opt'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                                
                            else:
                                raise ValueError("Error side")                
                        else:            
                            if 'MCF_ref' in optimaltrajectories[case]:
                                plotedJR = True
                                c_curve_ref = optimaltrajectories[case]['MCF_ref'][i:i+1,:].T
                                ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                            c_curve_sim = optimaltrajectories[case]['MCF'][i:i+1,:].T
                            ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                            
                        if 'MCF_ref' in optimaltrajectories[case]:
                            MCFs[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                        MCFs[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                        MCFs[case]['headers'].append(MCF_labels[i])
                        
                    ax.set_xlabel('Time (s)')
                    ax.set_title(MCF_labels[i])
                    handles, labels = ax.get_legend_handles_labels()
                    plt.legend(handles, labels, loc='upper right')
            # plt.setp(axs[0, :], xlabel='Time (s)')
            plt.setp(axs[0], ylabel='(N)')
            fig.align_ylabels()
        
        # %% MCF
        if not fieldStudy and 'walking' in motion_type:
            MCFs_BW = {}
            NMCFs = len(MCF_labels)
            for case in cases:
                MCFs_BW[case] = {}    
                # trial = list(optimaltrajectories[case]['time'].keys())[0]
                time = optimaltrajectories[case]['time'][0,:-1].T    
                MCFs_BW[case]['ref'] = np.zeros((NMCFs+1, time.shape[0]))
                MCFs_BW[case]['sim'] = np.zeros((NMCFs+1, time.shape[0]))
                MCFs_BW[case]['headers'] = ['time'] 
            
            # GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
            
            fig, axs = plt.subplots(1, 2, sharex=True)
            # for trial in trials:
            fig.suptitle('MCFs_BW: DC vs IK') 
            for i, ax in enumerate(axs.flat):
                if i < NMCFs:
                    color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
                    plotedJR = False
                    for case in cases:
                        
                        # trial = list(optimaltrajectories[case]['time'].keys())[0]
                        time = optimaltrajectories[case]['time'][0,:-1].T
                        
                        MCFs_BW[case]['ref'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                        MCFs_BW[case]['sim'][0:1, :] = np.reshape(time, (1, time.shape[0]))
                        
                        # invert_left_right = cases_toPlot[case]['invert_left_right']
                        if invert_left_right:
                            if MCF_labels[i] in jrOpposite or MCF_labels[i] in jrInverse:
                                
                                side = MCF_labels[i][11]
                                if side == 'r':
                                    inv_side = 'l'
                                elif side == 'l':
                                    inv_side = 'r'
                                else:
                                    raise ValueError("Error side")                    
                                inv_JR = MCF_labels[i][:11] + inv_side + MCF_labels[i][12:]
                                inv_i = MCF_labels.index(inv_JR)
                                
                                if MCF_labels[i] in jrOpposite:
                                
                                    if 'jr_ref' in optimaltrajectories[case]:
                                        plotedJR = True
                                        c_curve_ref = -optimaltrajectories[case]['jr_ref'][inv_i:inv_i+1,:].T
                                        ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                                    c_curve_sim = -optimaltrajectories[case]['jr_opt'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                                    
                                elif MCF_labels[i] in jrInverse:
                                    
                                    if 'jr_ref' in optimaltrajectories[case]:
                                        plotedJR = True
                                        c_curve_ref = optimaltrajectories[case]['jr_ref'][inv_i:inv_i+1,:].T
                                        ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                                    c_curve_sim = optimaltrajectories[case]['jr_opt'][inv_i:inv_i+1,:].T
                                    ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                                
                            else:
                                raise ValueError("Error side")                
                        else:            
                            if 'MCF_BW_ref' in optimaltrajectories[case]:
                                plotedJR = True
                                c_curve_ref = optimaltrajectories[case]['MCF_BW_ref'][i:i+1,:].T
                                ax.plot(time, c_curve_ref, c='black', label='measured JR ' + case)
                            c_curve_sim = optimaltrajectories[case]['MCF_BW'][i:i+1,:].T
                            ax.plot(time, c_curve_sim, c=next(color), label='video-based DC ' + case)
                            
                        if 'MCF_BW_ref' in optimaltrajectories[case]:
                            MCFs_BW[case]['ref'][i+1:i+2, :] = c_curve_ref.T
                        MCFs_BW[case]['sim'][i+1:i+2, :] = c_curve_sim.T
                        MCFs_BW[case]['headers'].append(MCF_labels[i])
                        
                    ax.set_xlabel('Time (s)')
                    ax.set_title(MCF_labels[i])
                    handles, labels = ax.get_legend_handles_labels()
                    plt.legend(handles, labels, loc='upper right')
            # plt.setp(axs[0, :], xlabel='Time (s)')
            plt.setp(axs[0], ylabel='(%BW)')
            fig.align_ylabels()
        
        # %% Gather results
        if fieldStudy:
            pathDCResults = os.path.join(pathOSData, 
                                         '{}_results.npy'.format(motion_type))
        else:
            pathDCResults = os.path.join(pathOSData,
                                         '{}_results.npy'.format(motion_type))
        
        if not os.path.exists(pathDCResults): 
                results = {}
        else:  
            results = np.load(pathDCResults, allow_pickle=True).item()
        
        results = {}
        results['positions'] = positions
        results['velocities'] = velocities
        results['accelerations'] = accelerations
        results['torques'] = torques
        results['torques_BWht'] = torques_BWht
        results['GRFs'] = GRFs
        results['GRFs_peaks'] = GRFs_peaks
        results['GRFs_BW'] = GRFs_BW
        results['GRFs_BW_peaks'] = GRFs_BW_peaks
        results['GRMs'] = GRMs
        results['GRMs_BWht'] = GRMs_BWht
        results['activations'] = activations
        if not fieldStudy and ('walking' in motion_type or 'DJ' in motion_type):
            results['KAMs'] = KAMs
            results['KAMs_BWht'] = KAMs_BWht
            if 'walking' in motion_type:
                results['MCFs'] = MCFs
                results['MCFs_BW'] = MCFs_BW
        
        # save results
        if saveResults:
            np.save(pathDCResults, results)
            