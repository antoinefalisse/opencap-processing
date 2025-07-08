import sys
import os
import csv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ACTIVITY_ANALYSES = os.path.join(PROJECT_ROOT, "ActivityAnalyses")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if ACTIVITY_ANALYSES not in sys.path:
    sys.path.append(ACTIVITY_ANALYSES)

from gait_analysis import gait_analysis
from utils import get_session_json, get_trial_id, download_trial
from utilsPlotting import plot_dataframe_with_shading
from umohada_data import get_data
from datetime import datetime

# %% Paths.
data_folder = os.path.join(PROJECT_ROOT, 'Data', 'Umohada')
os.makedirs(data_folder, exist_ok=True)

# %% Data analysis
data = get_data()

# Select scalars you'd like to analyze.
scalar_names = {
    'gait_speed', 'stride_length', 'step_width', 'cadence', 'step_length',
    'single_support_time', 'double_support_time', 'step_length_symmetry'}

# Select how many gait cycles you'd like to analyze. Select -1 for all gait
# cycles detected in the trial.
n_gait_cycles = -1 

# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

# Loop over sessions in data.
for session_id in data:
    print(f"\nProcessing session: {session_id}")
    session_json = get_session_json(session_id)
    session_dir = os.path.join(data_folder, session_id)
    # Extract session name or use first part of session ID
    try:
        session_name = session_json['meta']['sessionName']
    except KeyError:
        session_name = session_id[:session_id.find('-')]
    if session_name == '':
        session_name = session_id[:session_id.find('-')]

    gait, gait_events, gait_scalars, gait_curves = {}, {}, {}, {}
    trial_info = {}
    count = 0
    for trial in session_json['trials']:
        if count >= 5:
            break
        # Extract trial information.
        trial_name = trial['name']
        if trial_name == 'neutral' or trial_name == 'calibration':
            continue
        trial_status = trial['status']
        if not trial_status == 'done':
            print(f"Skipping trial {trial_name} with status {trial_status}.")
            continue
        # Print session and trial information.
        print(f"\nProcessing trial: {trial_name}")
        
        trial_id = trial['id']
        trial_info[trial_id] = {}
        trial_info[trial_id]['created_at'] = {}
        trial_created_at = datetime.fromisoformat(trial['created_at'].replace("Z", ""))
        trial_info[trial_id]['created_at']['date'] = trial_created_at.strftime("%Y-%m-%d")
        trial_info[trial_id]['created_at']['time'] = trial_created_at.strftime("%H:%M:%S")
        trial_info[trial_id]['name'] = trial_name        

        # Download data.
        _ = download_trial(trial_id, session_dir, session_id=session_id) 

        # Gait analysis.
        gait[trial_id], gait_events[trial_id], gait_scalars[trial_id], gait_curves[trial_id] = {}, {}, {}, {}
        for leg in ['r', 'l']:
            try:
                gait[trial_id][leg] = gait_analysis(
                    session_dir, trial_name, leg=leg,
                    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
                    n_gait_cycles=n_gait_cycles, gait_style='overground')
                gait_events[trial_id][leg] = gait[trial_id][leg].get_gait_events()
                gait_scalars[trial_id][leg] = gait[trial_id][leg].compute_scalars(scalar_names)
                gait_curves[trial_id][leg] = gait[trial_id][leg].get_coordinates_normalized_time()
            except Exception as e:
                print(f"Error processing gait data for trial {trial_id}, leg {leg}: {e}")
                gait_events[trial_id][leg] = None
                gait_scalars[trial_id][leg] = None
                gait_curves[trial_id][leg] = None
        count += 1

    # Export data in csv (one file per session).
    csv_file_path = os.path.join(session_dir, 'scalar_metrics_{}.csv'.format(session_id))
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        headers = ['Session ID', 'Session Name', 'Trial ID', 'Trial Name', 'Date', 'Time', 'Side', '# Gait Cycles']
        for scalar_name in sorted(list(scalar_names)):
            headers.append(scalar_name)
        writer.writerow(headers)

        for trial_id in list(gait.keys()):
            for leg in ['r', 'l']:
                if gait_events[trial_id][leg] is None:
                    n_cycles = 'NA'
                else:
                    n_cycles = gait_events[trial_id][leg]['ipsilateralIdx'].shape[0]
                row = [session_id, session_name, trial_id, trial_info[trial_id]['name'], trial_info[trial_id]['created_at']['date'], trial_info[trial_id]['created_at']['time'], leg, n_cycles]
                for scalar_name in sorted(list(scalar_names)):
                    if gait_scalars[trial_id][leg] is None or scalar_name not in gait_scalars[trial_id][leg]:
                        row.append('NA')
                    else:
                        if scalar_name == 'step_length':
                            row.append(gait_scalars[trial_id][leg][scalar_name]['value'][leg])
                        else:
                            row.append(gait_scalars[trial_id][leg][scalar_name]['value'])
                writer.writerow(row)