# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:36:07 2023

@author: antoi
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


baseDir = os.getcwd()
dataDir = os.path.join(baseDir, 'Data')
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(baseDir)
sys.path.append(opensimADDir)

from utilsOpenSimAD import getGRF

# session_ids = ['subject2_v2_mmpose'] #'subject2_v2_mmpose']#,'subject2_v54_mmpose']
# legends = ["mocap"]#['Deployed']#, 'Latest']

# session_ids = ['subject2_mocap'] #'subject2_v2_mmpose']#,'subject2_v54_mmpose']
# legends = ["mocap"]#['Deployed']#, 'Latest']

session_ids = ['subject2_v2_mmpose','subject2_v54_mmpose','subject2_mocap']
legends = ['Deployed', 'Latest',"mocap"]

colors = sns.color_palette('colorblind', len(session_ids))
case = '0'
trial_name = 'DJ1'

GRF_headers = ['R_ground_force_vx', 'R_ground_force_vy', 'R_ground_force_vz',
               'L_ground_force_vx', 'L_ground_force_vy', 'L_ground_force_vz']

if trial_name == 'DJ1':
    time_zoom = [1.5, 2.3]
elif trial_name == 'DJ2':
    time_zoom = [1.1,1.9]



GRFs, times = {}, {}
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
handles_dict = {}
labels_dict = {}
for idx_sess, session_id in enumerate(session_ids):

    pathOSData = os.path.join(dataDir, session_id, 'OpenSimData')
    c_pathResults = os.path.join(pathOSData, 'Dynamics', trial_name)    
    c_tr = np.load(os.path.join(c_pathResults, 'optimaltrajectories.npy'),
                    allow_pickle=True).item()   
    optimaltrajectory = c_tr[case]
    GRFs[session_id] = optimaltrajectory['GRF']
    GRF_labels = optimaltrajectory['GRF_labels']
    times[session_id] = optimaltrajectory['time'][:,:-1]

    pathForceData = os.path.join(dataDir, session_id, 'ForceData')
    c_pathGRF = os.path.join(pathForceData, trial_name + '.mot') 
    exp_GRF = getGRF(c_pathGRF, GRF_headers).to_numpy()
    exp_time = exp_GRF[:,0]
    exp_GRF_data = exp_GRF[:,1:]

    # Get indices of time window from 1.5 to 2.3 seconds
    idx = np.where((exp_time >= time_zoom[0]) & (exp_time <= time_zoom[1]))[0]
    # Get indices of time window from 1.5 to 2.3 seconds in times[session_id]
    idx2 = np.where((times[session_id][0,:] >= time_zoom[0]) & (times[session_id][0,:] <= time_zoom[1]))[0]


    # Create a 2x3 array of subplots and plot GRFs[session_id] with titles GRF_labels    
    for i in range(2):
        for j in range(3):
            # axs[i, j].plot(times[session_id][0,idx2].T, GRFs[session_id][i*3+j,idx2])
            # Plot this axs[i, j].plot(times[session_id][0,idx2].T, GRFs[session_id][i*3+j,idx2]) but keep handle for legend
            handle, = axs[i, j].plot(times[session_id][0,idx2].T, GRFs[session_id][i*3+j,idx2],color=colors[idx_sess])
            label = legends[session_ids.index(session_id)]
            if label in handles_dict:
                handles_dict[label].append(handle)
            else:
                handles_dict[label] = [handle]
                labels_dict[label] = label
            handle, = axs[i, j].plot(exp_time[idx], exp_GRF_data[idx,i*3+j],color='black', linewidth=2, label='Experimental')
            label = 'Experimental'
            if label in handles_dict:
                handles_dict[label].append(handle)
            else:
                handles_dict[label] = [handle]
                labels_dict[label] = label
            axs[i, j].set_title(GRF_labels[i*3+j], fontsize=14, fontweight='bold')

            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].tick_params(axis='both', which='major', labelsize=14)
            axs[i, j].tick_params(axis='both', which='minor', labelsize=14)

    # Set the x and y labels of the subplots
    axs[0, 0].set_ylabel('GRF (N)', fontsize=14, fontweight='bold')
    axs[1, 0].set_ylabel('GRF (N)', fontsize=14, fontweight='bold')
    axs[1, 0].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    axs[1, 1].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    axs[1, 2].set_xlabel('Time (s)', fontsize=14, fontweight='bold')

# Add legend to the plot
handles = []
for label in handles_dict:
    handles.append(handles_dict[label][0])
    labels_dict[label] = label
fig.legend(handles=handles, labels=list(labels_dict.values()), fontsize=14)
plt.show()
    
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# handles_dict = {}
# labels_dict = {}
# for session_id in session_ids:
#     # Create a 2x3 array of subplots and plot GRFs[session_id] with titles GRF_labels    
#     for i in range(2):
#         for j in range(3):
#             # axs[i, j].plot(times[session_id][0,idx2].T, GRFs[session_id][i*3+j,idx2])
#             # Plot this axs[i, j].plot(times[session_id][0,idx2].T, GRFs[session_id][i*3+j,idx2]) but keep handle for legend
#             handle, = axs[i, j].plot(times[session_id][0,idx2].T, GRFs[session_id][i*3+j,idx2])
#             label = legends[session_ids.index(session_id)]
#             if label in handles_dict:
#                 handles_dict[label].append(handle)
#             else:
#                 handles_dict[label] = [handle]
#                 labels_dict[label] = label
#             handle, = axs[i, j].plot(exp_time[idx], exp_GRF_data[idx,i*3+j],color='black', linewidth=2, label='Experimental')
#             label = 'Experimental'
#             if label in handles_dict:
#                 handles_dict[label].append(handle)
#             else:
#                 handles_dict[label] = [handle]
#                 labels_dict[label] = label
#             axs[i, j].set_title(GRF_labels[i*3+j])
#     # Set the x and y labels of the subplots
#     axs[0, 0].set_ylabel('GRF (N)')
#     axs[1, 0].set_ylabel('GRF (N)')
#     axs[1, 0].set_xlabel('Time (s)')
#     axs[1, 1].set_xlabel('Time (s)')
#     axs[1, 2].set_xlabel('Time (s)')

# # Add legend to the plot
# handles = []
# for label in handles_dict:
#     handles.append(handles_dict[label][0])
#     labels_dict[label] = label
# fig.legend(handles=handles, labels=list(labels_dict.values()))
# plt.show()
