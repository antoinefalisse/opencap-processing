import os
import shutil
import yaml

import sys
sys.path.append("../")
baseDir = os.path.join(os.getcwd(), '..')

from utils import import_metadata

# %% Data re-organization
dataDir = 'C:/MyDriveSym/Projects/mobilecap/data'

driveDir = os.path.join(baseDir, 'data', 'Benchmark')
dataFolder = os.path.join(driveDir, 'Data')

subjects = ['subject' + str(i) for i in range(3, 12)]

for subject in subjects:
    # Local
    pathSubject = os.path.join(driveDir, subject)
    os.makedirs(pathSubject, exist_ok=True)
    pathOpenSimData = os.path.join(pathSubject, 'OpenSimData')
    os.makedirs(pathOpenSimData, exist_ok=True)
    pathKinematics = os.path.join(pathOpenSimData, 'Kinematics')
    os.makedirs(pathKinematics, exist_ok=True)
    pathModel = os.path.join(pathOpenSimData, 'Model')
    os.makedirs(pathModel, exist_ok=True)

    # Drive
    pathModelDrive = os.path.join(dataDir, subject, 'OpenSimData', 'Video', 'mmpose_0.8', '2-cameras', 'v0.63')
    pathKinematicsDrive = os.path.join(pathModelDrive, 'IK', 'LaiArnoldModified2017_poly_withArms_weldHand')
    pathModelDrive = os.path.join(pathModelDrive, 'Model', 'LaiArnoldModified2017_poly_withArms_weldHand')

    # Copy model
    for file in os.listdir(pathModelDrive):
        pathFile = os.path.join(pathModelDrive, file)
        if os.path.isfile(pathFile) and not 'desktop.ini' in file:
            shutil.copy2(pathFile, pathModel)

    # Change model name
    for file in os.listdir(pathModel):
        if 'LaiArnoldModified2017_poly_withArms_weldHand' in file:
            fileName = file.replace('LaiArnoldModified2017_poly_withArms_weldHand', 'LaiUhlrich2022')
            os.rename(os.path.join(pathModel, file), os.path.join(pathModel, fileName))

    # Copy kinematics
    for file in os.listdir(pathKinematicsDrive):
        pathFile = os.path.join(pathKinematicsDrive, file)
        if os.path.isfile(pathFile) and not 'desktop.ini' in file:
            shutil.copy2(pathFile, pathKinematics)

    # Copy metadata
    pathMetadata = os.path.join(dataDir, subject, 'sessionMetadata.yaml')
    shutil.copy2(pathMetadata, pathSubject)

    # Load metadata and adjust opensim model
    metadata = import_metadata(os.path.join(pathSubject, 'sessionMetadata.yaml'))
    metadata['openSimModel'] = 'LaiUhlrich2022'
    # Save metadata
    with open(os.path.join(pathSubject, 'sessionMetadata.yaml'), 'w') as file:
        documents = yaml.dump(metadata, file)