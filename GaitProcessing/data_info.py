# # -*- coding: utf-8 -*-
# """
# Created on Wed Sep 27 16:15:43 2023

# @author: antoi
# """

# import pandas as pd

# # Load the Excel sheet
# excel_file_path = "trial_info.xlsx"  # Replace with the path to your Excel file
# df = pd.read_excel(excel_file_path, engine="openpyxl")

# # Initialize empty lists to store the results
# pid_list = []
# sid_list = []
# trial_list = []
# trial_clean_list = []

# # Iterate over the rows and filter based on the "trial_clean" column
# count = 0
# for index, row in df.iterrows():
#     if "10mwt" in str(row["trial_clean"]).lower():
#         if "mdf" in str(row["pid"]).lower():
#             pid_list.append(row["pid"])
#             sid_list.append(row["sid"])
#             trial_list.append(row["trial"])
#             trial_clean_list.append(row["trial_clean"])
            
#             print('{}: {{"pid": "{}", "sid": "{}", "trial": "{}", "trial_clean": "{}"}},'.format(100+count, str(row["pid"]), str(row["sid"]), str(row["trial"]), str(row["trial_clean"])))
#             count += 1
# Print the lists
# print("PID List:", pid_list)
# print("SID List:", sid_list)
# print("Trial List:", trial_list)

# %% Test

def get_data_case():

    data = {
        25: {"r": '3', "l": '3'},
        86: {"l": '5'},
        94: {"r": '5',"l": '5'},
        96: {"r": '5',"l": '5'}
    }

    return data

def get_data_select_window():
    
    data = {
        0: [-1.0, 7.80],
        1: [-1.0, 7.72],
        2: [-1.0, 7.48],
        3: [-1.0, 6.55],
        4: [-1.0, 10.91],
        5: [-1.0, 8.85],
        6: [-1.0, 6.17],
        7: [-1.0, 8.43],
        8: [-1.0, 7.59],
        9: [-1.0, 7.5],
        10: [-1.0, 7.46],
        11: [-1.0, 6.12],
        12: [2.4, 9.7],
        13: [-1.0, 7.30],
        14: [-1.0, 6.82],
        15: [-1.0, 7.82],
        16: [-1.0, 7.8],
        17: [-1.0, 7.73],
        18: [-1.0, 7.1],
        19: [-1.0, 13.74],
        20: [-1.0, 9.0],
        21: [-1.0, 13.0],
        22: [-1.0, 12.47],
        23: [-1.0, 5.9],
        24: [-1.0, 8.55],
        25: [-1.0, 8.54],
        # 26: [nan, nan],
        27: [-1.0, 6.8],
        28: [-1.0, 6.71],
        29: [-1.0, 7.25],
        30: [-1.0, 7.9],
        31: [-1.0, 8.8],
        32: [-1.0, 7.23],
        33: [-1.0, 6.46],
        # 34: [nan, nan],
        35: [-1.0, 5.60],
        36: [-1.0, 8.85],
        37: [-1.0, 9.47],
        38: [-1.0, 12.01],
        39: [-1.0, 7.85],
        40: [-1.0, 6.6],
        41: [-1.0, 5.28],
        42: [-1.0, 7.63],
        43: [-1.0, 11.82],
        44: [-1.0, 8.03],
        45: [-1.0, 6.25],
        46: [-1.0, 7.23],
        47: [-1.0, 9.93],
        48: [-1.0, 8.00],
        49: [-1.0, 7.77],
        50: [-1.0, 14.05],
        51: [-1.0, 7.44],
        52: [-1.0, 6.7],
        53: [-1.0, 11.02],
        54: [-1.0, 9.2],
        55: [-1.0, 8.6],
        56: [-1.0, 13.33],
        57: [-1.0, 5.74],
        58: [-1.0, 6.21],
        59: [-1.0, 7.16],
        60: [-1.0, 6.97],
        61: [-1.0, 8.1],
        62: [11, 18.49],
        63: [-1.0, 5.35],
        64: [-1.0, 5.75],
        65: [-1.0, 7.93],
        66: [-1.0, 13.70],
        67: [-1.0, 7.10],
        68: [-1.0, 7.62],
        69: [-1.0, 7.19],
        70: [-1.0, 8.60],
        71: [-1.0, 6.28],
        72: [-1.0, 4.92],
        # 73: [nan, nan],
        74: [-1.0, 5.99],
        75: [-1.0, 8.57],
        76: [-1.0, 7.0],
        77: [-1.0, 5.64],
        78: [-1.0, 11.98],
        79: [-1.0, 6.05],
        80: [-1.0, 8.42],
        81: [-1.0, 4.68],
        82: [-1.0, 8.52],
        83: [-1.0, 6.83],
        84: [-1.0, 11.05],
        85: [-1.0, 5.92],
        86: [-1.0, 7.20],
        87: [3.23, 8.9],
        88: [-1.0, 7.57],
        89: [-1.0, 11.9],
        90: [-1.0, 10.55],
        91: [-1.0, 7.22],
        92: [-1.0, 5.8],
        93: [-1.0, 7.67],
        94: [-1.0, 7.07],
        95: [-1.0, 5.85],
        96: [-1.0, 7.17],
        97: [-1.0, 5.75],

        100: [-1, 9.85],
        101: [-1, 9.05],
        102: [-1, 7.1],
        103: [-1, 8.2],
        104: [-1, 7.6],
        106: [-1, 8.5],
        107: [-1, 7.15],
        108: [-1, 4.7],
        109: [-1, 7.3],
        110: [-1, 7.2],
        111: [-1, 6.75],
        112: [-1, 6.9],
        113: [-1, 8.7],
        115: [-1, 9.65],
        116: [-1, 8.4],
        117: [-1, 8.9],
        118: [-1, 6.2],
        119: [-1, 7.05],
        120: [-1, 6.2],
        121: [-1, 6.25],
        122: [-1, 6.8],
        123: [-1, 7.23],
        124: [-1, 6.6],
        125: [-1, 5.4],
        126: [-1, 5.9],
        127: [-1, 6.8],
        132: [-1, 6.75],
        133: [-1, 6.8],
        134: [-1, 8.6],
        135: [-1, 7.5],
        136: [-1, 8.0],
        138: [-1, 6.8],
        139: [-1, 4.95],
        140: [-1, 9.8],
        141: [-1, 9.0],
        142: [-1, 5.7],
        143: [-1, 5.5],
        145: [-1, 6.3],
        146: [-1, 10.5],
        147: [-1, 7.9],
        148: [-1, 5.2],
        149: [-1, 4.05],
        }
    
    return data

def get_data_manual_alignment():
    
    # Define alignments for sessions with unique calibrations
    alignment_unique = {
        '9f19fa3b-9dd5-426f-bed2-7825148f2cf6': 0.5, # 0
        'd6b90c12-92a9-4e5b-9500-54655dde7e63': 0, # 1
        '057d10da-34c7-4fb7-a127-6040010dde06': 1, # 2
        '522fa7c2-6d12-4c67-b383-f3096aa3017f': -0.5, # 3-7-8-9-11-13-77
        '4911bbab-78ee-4b6e-8da6-52a189dea747': 0.75, # 5
        'c793b4dc-940e-4d2e-bfed-bbbb16443b01': 0.5, # 6
        'a118e69e-f652-42e0-9b4c-009ba60b4f86': -0.25, # 10-12-14-92-93
        '8f745d21-b747-4c3c-b6c6-b69c223e1f80': 1.75, # 15-20-21-22-27-30-31-32
        '725e0d28-d043-44bd-9f3e-0e58338a6bc4': 0.5, # 16
        'a4679c46-2f38-4cc5-86b4-bdd3f01791ab': 0, # 17
        'dc497821-4eac-49e0-87ac-c088d1245edb': 4.5, # 18
        '8b7b2366-0d1e-4336-a867-59d9b6e576de': 2, # 34
        '9c4666be-57f4-4a6b-9c51-7e394f4a53e1': 1, # 48        
        'c3e8706c-47ea-463c-b14e-0326553a10ed': 1, # 49
        'a7675b05-88da-4bb0-b27b-5357c3ec5807': 1, # 50
        'a08ec9d6-24f8-44f7-a59c-f603b7517e4d': 5.25, # 51
        'ba7d94c8-dccb-486d-90d5-b3dcc237bfce': 0.5, # 52
        '59f3fc4e-674f-43cc-9fdc-e9a2cf0a109e': 0, # bad trial (single)
        '5cddbc88-bf08-4408-80df-4248578ae373': 0.75, # 76
        'b26b073c-9528-4052-a117-cbc226de83d1': 0, # 78
        '18f57fa8-41a0-4d8d-b7a3-d0c838516f24': 0, # 80
        '3a681925-50b5-470e-a44a-c1cab743f58b': 2, # 85
        'f77e0358-c94e-47ee-8b23-a9bbcadafde2': 0.5, # 86
        'dac69d8a-0790-48ea-9296-66b5903d9ba8': 1, # 87
        'e91690d8-1a0d-4031-a7ad-3fb1e23d7c36': 2, # 88
        '6a0fbe94-23aa-4302-89bb-d878517f1cc8': 0.5, # 91
        "bf33bc40-d6e8-499e-b060-94b720133e3a": 1.75,
        "a384272c-bc90-4150-ab94-a2af8f5a9315": 1.25,
        "9e22db8f-6356-46c0-a118-d8f2741f97be": 0.5,
        "28811d60-1973-427c-8814-db521690c051": 2.5
    }
    
    data = {
        ## Initial manual alignments
        # 89: {"angle": 4},
        # 90: {"angle": 4},
        # 91: {"angle": 0}, # not re-aligment but cutting end
        # 21: {"angle": 2.5},
        # 24: {"angle": 2},
        # 31: {"angle": 2},
        # 43: {"angle": 2},
        # 51: {"angle": 5.25},
        # 53: {"angle": 1},
        # 57: {"angle": 1},
        # 4: {"angle": 1},
        # 66: {"angle": 0.75},
        # 85: {"angle": 2},
        # 86: {"angle": 0.5},
        # 70: {"angle": 1},
        # 62: {"angle": 1},
        # 48: {"angle": 1},
        # 35: {"angle": 0},
        # 13: {"angle": -0.25},
        # 3: {"angle": 0},
        # 0: {"angle": 0.5},
        
        0: {'angle':alignment_unique['9f19fa3b-9dd5-426f-bed2-7825148f2cf6']},
        1: {'angle':alignment_unique['d6b90c12-92a9-4e5b-9500-54655dde7e63']},
        2: {'angle':alignment_unique['057d10da-34c7-4fb7-a127-6040010dde06']},
        3: {'angle':alignment_unique['522fa7c2-6d12-4c67-b383-f3096aa3017f']},
        4: {'angle':alignment_unique['9f19fa3b-9dd5-426f-bed2-7825148f2cf6']},
        5: {'angle':alignment_unique['4911bbab-78ee-4b6e-8da6-52a189dea747']},
        6: {'angle':alignment_unique['c793b4dc-940e-4d2e-bfed-bbbb16443b01']},
        7: {'angle':alignment_unique['522fa7c2-6d12-4c67-b383-f3096aa3017f']},
        8: {'angle':alignment_unique['522fa7c2-6d12-4c67-b383-f3096aa3017f']},
        9: {'angle':alignment_unique['522fa7c2-6d12-4c67-b383-f3096aa3017f']},
        10: {'angle':alignment_unique['a118e69e-f652-42e0-9b4c-009ba60b4f86']},
        11: {'angle':alignment_unique['522fa7c2-6d12-4c67-b383-f3096aa3017f']},
        12: {'angle':alignment_unique['a118e69e-f652-42e0-9b4c-009ba60b4f86']},
        13: {'angle':alignment_unique['522fa7c2-6d12-4c67-b383-f3096aa3017f']},
        14: {'angle':alignment_unique['a118e69e-f652-42e0-9b4c-009ba60b4f86']},
        15: {'angle':alignment_unique['8f745d21-b747-4c3c-b6c6-b69c223e1f80']},
        16: {'angle':alignment_unique['725e0d28-d043-44bd-9f3e-0e58338a6bc4']},
        17: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        18: {'angle':alignment_unique['dc497821-4eac-49e0-87ac-c088d1245edb']},
        19: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        20: {'angle':alignment_unique['8f745d21-b747-4c3c-b6c6-b69c223e1f80']},
        21: {'angle':alignment_unique['8f745d21-b747-4c3c-b6c6-b69c223e1f80']},
        22: {'angle':alignment_unique['8f745d21-b747-4c3c-b6c6-b69c223e1f80']},
        23: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        24: {'angle':alignment_unique['8f745d21-b747-4c3c-b6c6-b69c223e1f80']},
        25: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        26: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        27: {'angle':alignment_unique['8f745d21-b747-4c3c-b6c6-b69c223e1f80']},
        28: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        29: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        30: {'angle':alignment_unique['8f745d21-b747-4c3c-b6c6-b69c223e1f80']},
        31: {'angle':alignment_unique['8f745d21-b747-4c3c-b6c6-b69c223e1f80']},
        32: {'angle':alignment_unique['8f745d21-b747-4c3c-b6c6-b69c223e1f80']},
        33: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        34: {'angle':alignment_unique['8b7b2366-0d1e-4336-a867-59d9b6e576de']},
        35: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        36: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        37: {'angle':alignment_unique['8b7b2366-0d1e-4336-a867-59d9b6e576de']},
        38: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        39: {'angle':alignment_unique['8b7b2366-0d1e-4336-a867-59d9b6e576de']},
        40: {'angle':alignment_unique['8b7b2366-0d1e-4336-a867-59d9b6e576de']},
        41: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        42: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        43: {'angle':alignment_unique['8b7b2366-0d1e-4336-a867-59d9b6e576de']},
        44: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        45: {'angle':alignment_unique['a4679c46-2f38-4cc5-86b4-bdd3f01791ab']},
        46: {'angle':alignment_unique['8b7b2366-0d1e-4336-a867-59d9b6e576de']},
        47: {'angle':alignment_unique['8b7b2366-0d1e-4336-a867-59d9b6e576de']},
        48: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        49: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        50: {'angle':alignment_unique['a7675b05-88da-4bb0-b27b-5357c3ec5807']},
        51: {'angle':alignment_unique['a08ec9d6-24f8-44f7-a59c-f603b7517e4d']},
        52: {'angle':alignment_unique['ba7d94c8-dccb-486d-90d5-b3dcc237bfce']},
        53: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        54: {'angle':alignment_unique['ba7d94c8-dccb-486d-90d5-b3dcc237bfce']},
        55: {'angle':alignment_unique['dc497821-4eac-49e0-87ac-c088d1245edb']},
        56: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        57: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        58: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        59: {'angle':alignment_unique['ba7d94c8-dccb-486d-90d5-b3dcc237bfce']},
        60: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        61: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        62: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        63: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        64: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        65: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        66: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        67: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        68: {'angle':alignment_unique['ba7d94c8-dccb-486d-90d5-b3dcc237bfce']},
        69: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        70: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        71: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        72: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        73: {'angle':alignment_unique['59f3fc4e-674f-43cc-9fdc-e9a2cf0a109e']},
        74: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        75: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        76: {'angle':alignment_unique['5cddbc88-bf08-4408-80df-4248578ae373']},
        77: {'angle':alignment_unique['522fa7c2-6d12-4c67-b383-f3096aa3017f']},
        78: {'angle':alignment_unique['b26b073c-9528-4052-a117-cbc226de83d1']},
        79: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        80: {'angle':alignment_unique['18f57fa8-41a0-4d8d-b7a3-d0c838516f24']},
        81: {'angle':alignment_unique['b26b073c-9528-4052-a117-cbc226de83d1']},
        82: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        83: {'angle':alignment_unique['c3e8706c-47ea-463c-b14e-0326553a10ed']},
        84: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        85: {'angle':alignment_unique['3a681925-50b5-470e-a44a-c1cab743f58b']},
        86: {'angle':alignment_unique['f77e0358-c94e-47ee-8b23-a9bbcadafde2']},
        87: {'angle':alignment_unique['dac69d8a-0790-48ea-9296-66b5903d9ba8']},
        88: {'angle':alignment_unique['e91690d8-1a0d-4031-a7ad-3fb1e23d7c36']},
        89: {'angle':alignment_unique['dc497821-4eac-49e0-87ac-c088d1245edb']},
        90: {'angle':alignment_unique['dc497821-4eac-49e0-87ac-c088d1245edb']},
        91: {'angle':alignment_unique['6a0fbe94-23aa-4302-89bb-d878517f1cc8']},
        92: {'angle':alignment_unique['a118e69e-f652-42e0-9b4c-009ba60b4f86']},
        93: {'angle':alignment_unique['a118e69e-f652-42e0-9b4c-009ba60b4f86']},
        94: {'angle':alignment_unique['8b7b2366-0d1e-4336-a867-59d9b6e576de']},
        95: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        96: {'angle':alignment_unique['9c4666be-57f4-4a6b-9c51-7e394f4a53e1']},
        97: {'angle':alignment_unique['b26b073c-9528-4052-a117-cbc226de83d1']},
        100: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        101: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        102: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        103: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        104: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        105: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        106: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        107: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        108: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        109: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        110: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        111: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        112: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        113: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        114: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        115: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        116: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        117: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        118: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        119: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        120: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        121: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        122: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        123: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        124: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        125: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        126: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        127: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        128: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        129: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        130: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        131: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        132: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        133: {'angle':alignment_unique['bf33bc40-d6e8-499e-b060-94b720133e3a']},
        134: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        135: {'angle':alignment_unique['9e22db8f-6356-46c0-a118-d8f2741f97be']},
        136: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        137: {'angle':alignment_unique['9e22db8f-6356-46c0-a118-d8f2741f97be']},
        138: {'angle':alignment_unique['9e22db8f-6356-46c0-a118-d8f2741f97be']},
        139: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        140: {'angle':alignment_unique['9e22db8f-6356-46c0-a118-d8f2741f97be']},
        141: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        142: {'angle':alignment_unique['9e22db8f-6356-46c0-a118-d8f2741f97be']},
        143: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        144: {'angle':alignment_unique['9e22db8f-6356-46c0-a118-d8f2741f97be']},
        145: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        146: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        147: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        148: {'angle':alignment_unique['a384272c-bc90-4150-ab94-a2af8f5a9315']},
        149: {'angle':alignment_unique['28811d60-1973-427c-8814-db521690c051']}
        }
    
    return data
    

def get_data_select_previous_cycle():
    
    data = {
        # 2: {"leg": ['r']},
        # 33: {"leg": ['r']},
        # 39: {"leg": ['r']},
        # 41: {"leg": ['r', 'l']},
        # 55: {"leg": ['l']},
        # 83: {"leg": ['l']},
        # 89: {"leg": ['l', 'r']},
        # 90: {"leg": ['r']},
        }
    
    return data
    

# def get_data_alignment():

#     # Trials that need re-aligment
#     data = [0, 2, 3, 4, 13, 15, 18, 20, 21, 22, 24, 26, 27, 30, 31, 32, 35, 37, 39, 40, 43, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57, 58,
#             62, 63, 64, 66, 67, 70, 71, 72, 73, 75, 76, 83, 85, 86, 89, 90, 91, 92,94,96,97,
            
#             100,101,103,105,107,109,111,113,115,116,117,118,120,122,124,126,129,131,133,            
#             102,104,106,108,110,112,114,119,121,123,125,127,128,130,132,134,136,139,141,143,145,146,147,148,
            
#             135,137,138,140,142,144,

#             149
            
#             ]

#     return data

def get_data_info_problems():

    data = {
        26: {"leg": ['r', 'l']},    # Bad kinematics - flying model
        34: {"leg": ['r', 'l']},    # Bad kinematics - problem feet
        73: {"leg": ['r', 'l']},    # jittery legs throughout trial

        # From old simulations, let's re-run and confirm
        # 55: {"leg": ['r', 'l'],"just_kinematics":['r','l']},    # No decent simulations / Glitch
        # 73: {"leg": ['r', 'l']},    # Glitch
        # 77: {"leg": ['r', 'l']},    # Glitch
        # 85: {"leg": ['r']},         # No decent simulations / Glitch
        # 86: {"leg": ['r'],"just_kinematics":['r','l']},         # No decent simulations / Glitch
        # 87: {"leg": ['r', 'l']},    # Glitch
        # 88: {"leg": ['r', 'l']},    # Glitch91: {"leg": ['l']},         # No decent simulations / Glitch
    }   

    return data

# Add optional argument to specify the trial index.
# If not specified, return all trials.
def get_data_info(trial_indexes=[]):

    data = {
        0: {"pid": "p011", "sid": "ee23fbb3-a991-4aa4-9a2f-a213ec9ec6c5", "trial": "10mwt", "trial_clean": "10mwt"},    # Okay kinematics, decent simulations
        1: {"pid": "p012", "sid": "d6b90c12-92a9-4e5b-9500-54655dde7e63", "trial": "10mwt", "trial_clean": "10mwt"},    # Okay kinematics, RERUN
        2: {"pid": "p013", "sid": "057d10da-34c7-4fb7-a127-6040010dde06", "trial": "10mwt_2", "trial_clean": "10mwt"},  # Not great kinematics, not great simulations
        3: {"pid": "p014", "sid": "a0d8b67a-89b0-45a7-a634-348ef2bcca5a", "trial": "10mwt", "trial_clean": "10mwt"},    # Bad right cycle, NO DATA
        4: {"pid": "p017", "sid": "3e7926e1-e6a7-4fe0-aafc-2ca7dc7aa8b5", "trial": "10mwt_1", "trial_clean": "10mwt"},  # Okay kinematics but huge pelvis, very large vGRF first peak
        5: {"pid": "p018", "sid": "e4d2328b-6f01-4f4e-bea3-64076357de64", "trial": "10mwt", "trial_clean": "10mwt"},    # Okay kinematics, decent simulations
        6: {"pid": "p019", "sid": "97e9730e-368d-4699-8da4-92bc4de5e182", "trial": "10mwt", "trial_clean": "10mwt"},    # Okay kinematics, decent simulations
        7: {"pid": "p022", "sid": "5041cd8c-1a83-43fe-949a-100001832aa4", "trial": "10mwt", "trial_clean": "10mwt"},    # Okay kinematics, decent simulations
        8: {"pid": "p023", "sid": "c8fd3b4b-4ed5-4d45-8c1a-a92f2ef82576", "trial": "10mwt", "trial_clean": "10mwt"},    # Okay kinematics, decent simulations
        9: {"pid": "p025", "sid": "25f84572-d054-49b1-b13e-81f1fec4b625", "trial": "10mwt", "trial_clean": "10mwt"},    # Okay kinematics
        10: {"pid": "p027", "sid": "92964f5a-40f1-413a-811f-b02bf2274db8", "trial": "10mwt", "trial_clean": "10mwt"},
        11: {"pid": "p028", "sid": "4cc6f9e7-7c02-4961-9867-e57da1e2fd11", "trial": "10mwt", "trial_clean": "10mwt"},
        12: {"pid": "p029", "sid": "5e12db52-2d0d-47e3-ba98-04e561dc2c2a", "trial": "10mwt", "trial_clean": "10mwt"},
        13: {"pid": "p030", "sid": "7290e122-5f10-4423-ac18-f853cd287c2c", "trial": "10mwt", "trial_clean": "10mwt"},
        14: {"pid": "p031", "sid": "642020a6-96fe-45a0-beae-a2dc8495f29b", "trial": "10mwt", "trial_clean": "10mwt"},
        15: {"pid": "p033", "sid": "8f745d21-b747-4c3c-b6c6-b69c223e1f80", "trial": "10mwt", "trial_clean": "10mwt"},
        16: {"pid": "p034", "sid": "725e0d28-d043-44bd-9f3e-0e58338a6bc4", "trial": "10mwt_1", "trial_clean": "10mwt"},
        17: {"pid": "p035", "sid": "41b87fed-9e9d-4af5-88fa-2ba7e30fe83b", "trial": "10mwt", "trial_clean": "10mwt"},
        18: {"pid": "p036", "sid": "9ce91ceb-2bf8-4544-ae22-3da5ad3cb9a1", "trial": "10mwt_1", "trial_clean": "10mwt"},
        19: {"pid": "p037", "sid": "a4679c46-2f38-4cc5-86b4-bdd3f01791ab", "trial": "10mwt", "trial_clean": "10mwt"},
        20: {"pid": "p038", "sid": "9bf3bb84-5dc8-453d-b271-438a967e3fca", "trial": "10mwt", "trial_clean": "10mwt"},
        21: {"pid": "p039", "sid": "3df39d34-e609-4dd7-ac27-dc9808e219b8", "trial": "10mwt_1", "trial_clean": "10mwt"},
        22: {"pid": "p040", "sid": "af3d2ecf-c872-4df6-80a4-4673acf68364", "trial": "10mwt", "trial_clean": "10mwt"},
        23: {"pid": "p041", "sid": "eb0a3f37-92a6-4a76-864a-678af4d636bf", "trial": "10mwt", "trial_clean": "10mwt"},
        24: {"pid": "p042", "sid": "1fc22b0c-1ef8-4b6d-b83b-85b12634905d", "trial": "10mwt", "trial_clean": "10mwt"},
        25: {"pid": "p043", "sid": "f329d908-eff4-4227-b52e-7e8fa2b07079", "trial": "10mwt", "trial_clean": "10mwt"},
        26: {"pid": "p044", "sid": "51003106-cdec-40f7-9204-0e21953bb4a7", "trial": "10mwt", "trial_clean": "10mwt"},
        27: {"pid": "p045", "sid": "c70f88f1-c189-40f9-8fa5-41d673a8708d", "trial": "10mwt", "trial_clean": "10mwt"},
        28: {"pid": "p046", "sid": "cdbd4609-53e0-473d-b893-e542090aece4", "trial": "10mwt", "trial_clean": "10mwt"},
        29: {"pid": "p047", "sid": "f0d502ef-b442-46e6-a55f-088303d325bc", "trial": "10mwt", "trial_clean": "10mwt"},
        30: {"pid": "p048", "sid": "e414a132-7d67-4f7b-b6d9-aca0b72297c5", "trial": "10mwt", "trial_clean": "10mwt"},
        31: {"pid": "p049", "sid": "71cd6204-a0d7-41fa-b6f1-715f28a5353d", "trial": "10mwt", "trial_clean": "10mwt"},
        32: {"pid": "p050", "sid": "11ca2bd1-e9db-49de-b754-a674a6f6181f", "trial": "10mwt", "trial_clean": "10mwt"},
        33: {"pid": "p051", "sid": "dc82b637-f582-4f3f-8e3b-2144431f1259", "trial": "10mwt", "trial_clean": "10mwt"},
        34: {"pid": "p052", "sid": "d9538893-e95c-4e28-960e-dd4aa0760942", "trial": "10mwt", "trial_clean": "10mwt"},
        35: {"pid": "p053", "sid": "df982665-8e59-4691-88f8-611ccbfe7f4a", "trial": "10mwt", "trial_clean": "10mwt"},
        36: {"pid": "p054", "sid": "2f6340cd-d4ad-4ef1-b8c3-d1296821c78f", "trial": "10mwt", "trial_clean": "10mwt"},
        37: {"pid": "p055", "sid": "8b7b2366-0d1e-4336-a867-59d9b6e576de", "trial": "10mwt", "trial_clean": "10mwt"},
        38: {"pid": "p056", "sid": "af3ff7d9-5ca1-426e-a831-7297a461a102", "trial": "10mwt", "trial_clean": "10mwt"},
        39: {"pid": "p057", "sid": "fca67864-be38-44ec-abb5-3ae8b8488573", "trial": "10mwt", "trial_clean": "10mwt"},
        40: {"pid": "p058", "sid": "4c46fbb5-4ab8-49ff-b90d-5e8a38c5bcff", "trial": "10mwt", "trial_clean": "10mwt"},
        41: {"pid": "p059", "sid": "1ac1337f-9c1d-412c-a514-5f9846cf05b5", "trial": "10mwt", "trial_clean": "10mwt"},
        42: {"pid": "p060", "sid": "2a4697a0-8950-4ad1-99cf-6e572b024920", "trial": "10mwt", "trial_clean": "10mwt"},
        43: {"pid": "p061", "sid": "c5d16067-3112-4a76-b551-05c3f4b0819a", "trial": "10mwt", "trial_clean": "10mwt"},
        44: {"pid": "p062", "sid": "9604d92d-a436-4ac9-9093-e0cb80761aaa", "trial": "10mwt", "trial_clean": "10mwt"},
        45: {"pid": "p066", "sid": "52f401c9-ac73-4a81-887c-56cf99aee545", "trial": "10mwt", "trial_clean": "10mwt"},
        46: {"pid": "p067", "sid": "7abe4c9e-abe2-4793-bfbe-18f5b817da95", "trial": "10mwt", "trial_clean": "10mwt"},
        47: {"pid": "p069", "sid": "ac50e718-7eb9-47c8-b711-a6f93569b9e1", "trial": "10mwt", "trial_clean": "10mwt"},
        48: {"pid": "p071", "sid": "715504d6-c823-4632-8c79-b7639bfdcf4a", "trial": "10mwt", "trial_clean": "10mwt"},
        49: {"pid": "p072", "sid": "4174f23a-0821-4a23-9971-4dd94ca05c21", "trial": "10mwt_2", "trial_clean": "10mwt"},
        50: {"pid": "p073", "sid": "a7675b05-88da-4bb0-b27b-5357c3ec5807", "trial": "10mwt", "trial_clean": "10mwt"},
        51: {"pid": "p073", "sid": "a08ec9d6-24f8-44f7-a59c-f603b7517e4d", "trial": "10mwt", "trial_clean": "10mwt"},
        52: {"pid": "p074", "sid": "bca0aad8-c129-4a62-bef3-b5de1659df5e", "trial": "10mwt", "trial_clean": "10mwt"},
        53: {"pid": "p075", "sid": "822926eb-1c7d-4298-84c5-7ead6909387b", "trial": "10mwt", "trial_clean": "10mwt"},
        54: {"pid": "p076", "sid": "ba7d94c8-dccb-486d-90d5-b3dcc237bfce", "trial": "10mwt", "trial_clean": "10mwt"},
        55: {"pid": "p076", "sid": "1d34aa40-98c5-4147-88eb-5021447ec3e7", "trial": "10mwt", "trial_clean": "10mwt"},
        56: {"pid": "p077", "sid": "fd5cad10-1562-4e74-af15-ed1363ba0684", "trial": "10mwt", "trial_clean": "10mwt"},
        57: {"pid": "p078", "sid": "f249a8d6-9965-4240-94c3-e42a6b4bccf6", "trial": "10mwt", "trial_clean": "10mwt"},
        58: {"pid": "p079", "sid": "8f2a3dc8-3c9d-467f-98ce-4f53299c3be3", "trial": "10mwt", "trial_clean": "10mwt"},
        59: {"pid": "p080", "sid": "ab14edea-6d5d-4904-af7d-25f392f296b0", "trial": "10mwt", "trial_clean": "10mwt"},
        60: {"pid": "p081", "sid": "b69e7b72-d754-41e7-a171-18d641a0ad56", "trial": "10wmt", "trial_clean": "10mwt"},
        61: {"pid": "p082", "sid": "002ea0b9-a2b6-4983-a876-275e3087ef1e", "trial": "10mwt", "trial_clean": "10mwt"},
        62: {"pid": "p083", "sid": "98fc0617-1f5e-44ba-92c9-a32f09899295", "trial": "10mwt", "trial_clean": "10mwt"},
        63: {"pid": "p084", "sid": "df3c99df-a9ef-484a-88f6-a561137d7a86", "trial": "10mwt", "trial_clean": "10mwt"},
        64: {"pid": "p085", "sid": "abc1a09c-6e05-4321-9a3b-ad656c4970d1", "trial": "10mwt_1", "trial_clean": "10mwt"},
        65: {"pid": "p087", "sid": "623a4fe2-bc1e-4248-920a-e35416d1350d", "trial": "10mwt", "trial_clean": "10mwt"},
        66: {"pid": "p088", "sid": "c55e70d2-c651-4643-9801-24b2e9e5e7b2", "trial": "10mwt", "trial_clean": "10mwt"},
        67: {"pid": "p089", "sid": "348015e8-af2b-485c-bfe9-77d825916c91", "trial": "10mwt", "trial_clean": "10mwt"},
        68: {"pid": "p090", "sid": "46bab048-9377-4316-ab22-9879c9a35f3f", "trial": "10mwt", "trial_clean": "10mwt"},
        69: {"pid": "p092", "sid": "10818bae-3254-4d5d-be05-a1853041b81d", "trial": "10mwt", "trial_clean": "10mwt"},
        70: {"pid": "p093", "sid": "a652d863-6375-4a81-a467-14de0658c1e0", "trial": "10mwt", "trial_clean": "10mwt"},
        71: {"pid": "p094", "sid": "8bf259f7-000f-4474-9ad4-255949bdaf71", "trial": "10mwt", "trial_clean": "10mwt"},
        72: {"pid": "p095", "sid": "8b7a734d-2874-42f7-94a2-c6026ec35d99", "trial": "10mwt_1", "trial_clean": "10mwt"},
        73: {"pid": "p096", "sid": "59f3fc4e-674f-43cc-9fdc-e9a2cf0a109e", "trial": "10mwt", "trial_clean": "10mwt"},
        74: {"pid": "p097", "sid": "46c3eff2-02f4-4b0c-a8a2-4f62ed5e148f", "trial": "10mwt", "trial_clean": "10mwt"},
        75: {"pid": "p099", "sid": "3ab5d28b-3dba-4c73-8c5e-937c6f889394", "trial": "10mwt", "trial_clean": "10mwt"},
        76: {"pid": "p100", "sid": "5cddbc88-bf08-4408-80df-4248578ae373", "trial": "10mwt", "trial_clean": "10mwt"},
        77: {"pid": "p101", "sid": "6de88f3d-b84a-4ee5-8003-a5fbb994278e", "trial": "10mwt", "trial_clean": "10mwt"},
        78: {"pid": "p102", "sid": "b6ea2727-ff50-42e5-92cd-1ac4dfc17f86", "trial": "10mwt", "trial_clean": "10mwt"},
        79: {"pid": "p103", "sid": "936589a3-69ee-4cf2-a74d-728531362252", "trial": "10mwt", "trial_clean": "10mwt"},
        80: {"pid": "p104", "sid": "18f57fa8-41a0-4d8d-b7a3-d0c838516f24", "trial": "10mwt", "trial_clean": "10mwt"},
        81: {"pid": "p105", "sid": "beb1ee5b-fbc0-4697-af4e-f3ab71aaa2e0", "trial": "10mwt_1", "trial_clean": "10mwt"},
        82: {"pid": "p106", "sid": "bb12568e-2303-4f87-95e4-baec67f2c023", "trial": "10mwt", "trial_clean": "10mwt"},
        83: {"pid": "p110", "sid": "f3da90e3-cc7c-4496-9488-662b2f8626b3", "trial": "10mwt", "trial_clean": "10mwt"},
        84: {"pid": "p111", "sid": "c1374bb6-55a1-46f5-b6fa-4c2b83d66b8e", "trial": "10mwt", "trial_clean": "10mwt"},
        85: {"pid": "p120", "sid": "3a681925-50b5-470e-a44a-c1cab743f58b", "trial": "10mwt", "trial_clean": "10mwt"},
        86: {"pid": "p121", "sid": "f77e0358-c94e-47ee-8b23-a9bbcadafde2", "trial": "10mwt", "trial_clean": "10mwt"},
        87: {"pid": "p122", "sid": "dac69d8a-0790-48ea-9296-66b5903d9ba8", "trial": "10mwt", "trial_clean": "10mwt"},
        88: {"pid": "p124", "sid": "e91690d8-1a0d-4031-a7ad-3fb1e23d7c36", "trial": "10mwt", "trial_clean": "10mwt"},
        89: {"pid": "p125", "sid": "dc497821-4eac-49e0-87ac-c088d1245edb", "trial": "10mwt_1", "trial_clean": "10mwt"},
        90: {"pid": "p126", "sid": "ca05bbca-dfab-4c19-8ceb-50173c57ff41", "trial": "10mwt", "trial_clean": "10mwt"},
        91: {"pid": "p128", "sid": "6a0fbe94-23aa-4302-89bb-d878517f1cc8", "trial": "10mwt", "trial_clean": "10mwt"},
        # second batch of dhd participants
        92: {"pid": "p024", "sid": "a118e69e-f652-42e0-9b4c-009ba60b4f86", "trial": "10mwt", "trial_clean": "10mwt"},
        93: {"pid": "p026", "sid": "2e0b9298-4036-4fe6-ad88-cdbccc4cca57", "trial": "10mwt", "trial_clean": "10mwt"},
        94: {"pid": "p068", "sid": "109eabdf-e50d-459e-8917-c856338a720b", "trial": "10mwt", "trial_clean": "10mwt"},
        95: {"pid": "p070", "sid": "9c4666be-57f4-4a6b-9c51-7e394f4a53e1", "trial": "10mwt", "trial_clean": "10mwt"},
        96: {"pid": "p091", "sid": "04cad9b0-796a-40ec-aa24-346c71e8ff37", "trial": "10mwt", "trial_clean": "10mwt"},
        97: {"pid": "p098", "sid": "84aef447-6982-4842-8f95-e63459993fcf", "trial": "10mwt", "trial_clean": "10mwt"},        


        # MDF
        # Sesssions w/ calibration trials:
        #   "bf33bc40-d6e8-499e-b060-94b720133e3a"
        #   "a384272c-bc90-4150-ab94-a2af8f5a9315"
        #   "9e22db8f-6356-46c0-a118-d8f2741f97be"
        #   "28811d60-1973-427c-8814-db521690c051"

        100: {"pid": "mdf_005", "sid": "dfa1c060-df8d-40e8-90e6-107078621c7c", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        101: {"pid": "mdf_006", "sid": "ce0fdd88-53b6-4974-92fb-a720b93d1623", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        102: {"pid": "mdf_006", "sid": "fa8c8348-4ceb-41ee-9c48-5f3701f84996", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        103: {"pid": "mdf_007", "sid": "9e5d7c34-d9fc-43f7-85e9-d3e5def7ef3a", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        104: {"pid": "mdf_007", "sid": "f5ec71e8-7898-4c51-993b-897014a3e8e3", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        105: {"pid": "mdf_008", "sid": "d939164d-341b-4b6b-bb1f-1ed775755046", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        106: {"pid": "mdf_008", "sid": "dddeeed1-1c25-4bf2-b642-ea46d0dc122b", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        107: {"pid": "mdf_009", "sid": "773c33cd-a12d-46d2-af17-35fa7b4e83bd", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        108: {"pid": "mdf_009", "sid": "ec004090-100c-4444-aa67-0613d3528b4e", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        109: {"pid": "mdf_010", "sid": "c3dd4efc-ecd9-486e-91df-69322608f070", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        110: {"pid": "mdf_010", "sid": "2928a9e3-db6a-44d1-8f97-de5a3a209403", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        111: {"pid": "mdf_011", "sid": "0cafccab-1003-4e78-aa51-b234acafa2ed", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        112: {"pid": "mdf_011", "sid": "b5624568-3c42-48e4-a2e3-b351418d4150", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        113: {"pid": "mdf_012", "sid": "62593b25-d05d-4880-ac71-d65a06e99a6d", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        114: {"pid": "mdf_012", "sid": "ee86be93-b1da-4080-b445-176f6071e734", "trial": "10mwt_1", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        115: {"pid": "mdf_013", "sid": "ce0d7ba4-2659-4324-a6ea-691a716f981f", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        116: {"pid": "mdf_014", "sid": "93636607-6e8c-4b04-9bab-e389bebc4430", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        117: {"pid": "mdf_015", "sid": "00faec9c-71e8-4051-8175-612c2488b0bb", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        118: {"pid": "mdf_016", "sid": "6d836009-5ef9-4c8d-9317-940258e18206", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        119: {"pid": "mdf_016", "sid": "009cf17d-298a-4e63-812b-0eaeac87eb95", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        120: {"pid": "mdf_017", "sid": "e2944d85-5737-4105-a317-b54115b80b6b", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        121: {"pid": "mdf_017", "sid": "4d0cfa49-baa9-4e49-8b34-2bc6727f6052", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        122: {"pid": "mdf_018", "sid": "8b2f6e69-d612-43ad-9824-15c1111b8b3c", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        123: {"pid": "mdf_018", "sid": "cda2db6e-b268-42ee-99d0-9cc358e893d1", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        124: {"pid": "mdf_019", "sid": "30ef567b-6485-447c-a5c8-b99ed84c17a4", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        125: {"pid": "mdf_019", "sid": "5a3dd4e7-7293-46e6-9c0c-b77e2511860d", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        126: {"pid": "mdf_020", "sid": "c7be4554-1925-4ec5-9372-072416d68ebf", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        127: {"pid": "mdf_020", "sid": "6c071107-3735-4c71-a03a-68cac9aa0546", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        128: {"pid": "mdf_021", "sid": "b9bb774e-2b9e-4755-928d-8e57eefe1ae7", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        129: {"pid": "mdf_022", "sid": "551a60a5-11cb-49a5-879b-477cf499af7a", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        130: {"pid": "mdf_022", "sid": "a384272c-bc90-4150-ab94-a2af8f5a9315", "trial": "10mwt_2", "trial_clean": "10mwt"}, #session with calibration: REFERENCE
        131: {"pid": "mdf_023", "sid": "d7793a0f-1451-4e21-a2d2-3d3f1bd9c7de", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        132: {"pid": "mdf_023", "sid": "b9578e78-d717-49de-a226-7f797dcc43a3", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        133: {"pid": "mdf_024", "sid": "5e16a747-e5ca-4853-98f6-a449452c494d", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 0 "bf33bc40-d6e8-499e-b060-94b720133e3a"
        134: {"pid": "mdf_026", "sid": "91ef0085-13a1-49fe-8bcf-91e37efc4d53", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        135: {"pid": "mdf_027", "sid": "9e22db8f-6356-46c0-a118-d8f2741f97be", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: REFERENCE
        136: {"pid": "mdf_027", "sid": "57e48655-deab-447a-9d0c-c292b124fdbd", "trial": "10mwt_1", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        137: {"pid": "mdf_028", "sid": "c0bf6608-e35d-40a8-9ca2-2abc4a9c3590", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 3 "9e22db8f-6356-46c0-a118-d8f2741f97be"
        138: {"pid": "mdf_029", "sid": "513ec455-04ba-4bc4-b502-5f91d34c4ce6", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 3 "9e22db8f-6356-46c0-a118-d8f2741f97be"
        139: {"pid": "mdf_029", "sid": "0207a325-f5cb-4b26-8544-a93d9831552b", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        140: {"pid": "mdf_030", "sid": "6ff20ae9-7f99-4837-95e7-82cedb242522", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 3 "9e22db8f-6356-46c0-a118-d8f2741f97be"
        141: {"pid": "mdf_030", "sid": "f9a5c172-288a-4886-b3bb-417ab3f48b55", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        142: {"pid": "mdf_031", "sid": "78d9fbfe-04e0-4766-ba15-198e246d5e9c", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 3 "9e22db8f-6356-46c0-a118-d8f2741f97be"
        143: {"pid": "mdf_031", "sid": "e5e2f6ea-aac2-4d5f-b985-609f7209b747", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        144: {"pid": "mdf_031_db235", "sid": "fa856382-bf8d-411a-b086-a2d74fab9d1b", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 3 "9e22db8f-6356-46c0-a118-d8f2741f97be"
        145: {"pid": "mdf_031_db235", "sid": "55566ced-e1be-4789-b265-b0168087a402", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        146: {"pid": "mdf_032", "sid": "36f5a719-b5fd-418f-a322-676a321510b3", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        147: {"pid": "mdf_033", "sid": "9871c398-0d02-450b-86a6-0d8c6b27b26d", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        148: {"pid": "mdf_034", "sid": "28ec73c6-025b-4ea4-b844-803e8205bf1a", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 1 "a384272c-bc90-4150-ab94-a2af8f5a9315"
        149: {"pid": "mdf_035", "sid": "aca92056-3f67-403c-8d6a-513055274ffe", "trial": "10mwt", "trial_clean": "10mwt"}, #session with calibration: 4 "28811d60-1973-427c-8814-db521690c051"
    }

    if trial_indexes:
        # Return data dict with only the specified trials.
        return {trial_index: data[trial_index] for trial_index in trial_indexes}
    else:
        return data


# %%
#