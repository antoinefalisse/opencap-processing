# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:37:20 2023

@author: suhlr
"""

from data_info import get_data_info
import sys
import os
baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)

from utils import get_calibration_trial_id, get_trial_json

# Prints a dict relating child sessions to unique calibrations
# Also prints a list of unique calibration sessions

data_info = get_data_info()

calibration_trials_with_alignment = []
sessions_to_align = []
# sessions = [data_info[i]['sid'] for i in data_info.keys()]

for idx,info in data_info.items():
    sid = info['sid']
    
    calibID = get_calibration_trial_id(sid)
    sid_unique = get_trial_json(calibID)['session']
    if calibID in calibration_trials_with_alignment:
        pass
    else:
        calibration_trials_with_alignment.append(calibID)
        sessions_to_align.append(sid_unique)
        
    print(str(idx) + ": {'angle':alignment_unique['" + sid_unique + "']},")
    
# create the unique session dict
for sid in sessions_to_align:
    print("'" + sid + "': 0,")
    




    
    