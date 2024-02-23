import os
import shutil
import yaml

import sys
sys.path.append("../")
baseDir = os.path.join(os.getcwd(), '..')

# from utils import import_metadata

# %% Data re-organization
dataDir = 'C:/MyDriveSym/Projects/mobilecap/data'
# dataDir = os.path.join(baseDir, 'Data', 'Benchmark')

driveDir = os.path.join(baseDir, 'data', 'Benchmark')
dataFolder = os.path.join(driveDir, 'Data')

subjects = ['subject' + str(i) for i in range(2, 12)]

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
    pathMocap = os.path.join(pathSubject, 'Mocap')
    os.makedirs(pathMocap, exist_ok=True)
    pathID = os.path.join(pathMocap, 'InverseDynamics')
    os.makedirs(pathID, exist_ok=True)
    pathIK = os.path.join(pathMocap, 'Kinematics')
    os.makedirs(pathIK, exist_ok=True)
    pathGRF = os.path.join(pathSubject, 'ForceData')
    os.makedirs(pathGRF, exist_ok=True)
    pathSO = os.path.join(pathMocap, 'StaticOptimization')
    os.makedirs(pathSO, exist_ok=True)
    pathJRA = os.path.join(pathMocap, 'JointReactionAnalysis')
    os.makedirs(pathJRA, exist_ok=True)
    pathEMG = os.path.join(pathSubject, 'EMGData')
    os.makedirs(pathEMG, exist_ok=True)
    pathModelMocap = os.path.join(pathMocap, 'Model')
    os.makedirs(pathModelMocap, exist_ok=True)

    # Drive
    pathDataDrive = os.path.join(dataDir, subject, 'OpenSimData', 'Video', 'mmpose_0.8', '2-cameras', 'v0.63_allVideoOnly')
    pathMocapDrive = os.path.join(dataDir, subject, 'OpenSimData', 'Mocap')
    pathKinematicsDrive = os.path.join(pathDataDrive, 'IK', 'LaiArnoldModified2017_poly_withArms_weldHand')
    pathModelDrive = os.path.join(pathDataDrive, 'Model', 'LaiArnoldModified2017_poly_withArms_weldHand')
    pathIDDrive = os.path.join(pathMocapDrive, 'ID', 'LaiArnoldModified2017_poly_withArms_weldHand')
    pathIKDrive = os.path.join(pathMocapDrive, 'IK', 'LaiArnoldModified2017_poly_withArms_weldHand')
    pathGRFDrive = os.path.join(dataDir, subject, 'ForceData')
    pathSODrive = os.path.join(pathMocapDrive, 'SO', 'LaiArnoldModified2017_poly_withArms_weldHand')
    pathJRADrive = os.path.join(pathMocapDrive, 'JRA', 'LaiArnoldModified2017_poly_withArms_weldHand')
    pathEMGDrive = os.path.join(dataDir, subject, 'EMGData')
    pathModelMocapDrive = os.path.join(pathMocapDrive, 'Model', 'LaiArnoldModified2017_poly_withArms_weldHand')

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

    # # Copy metadata
    # pathMetadata = os.path.join(dataDir, subject, 'sessionMetadata.yaml')
    # shutil.copy2(pathMetadata, pathSubject)

    # # Load metadata and adjust opensim model
    # metadata = import_metadata(os.path.join(pathSubject, 'sessionMetadata.yaml'))
    # metadata['openSimModel'] = 'LaiUhlrich2022'
    # # Save metadata
    # with open(os.path.join(pathSubject, 'sessionMetadata.yaml'), 'w') as file:
    #     documents = yaml.dump(metadata, file)

    # Copy ID
    # for file in os.listdir(pathIDDrive):
    #     pathFile = os.path.join(pathIDDrive, file)
    #     if os.path.isfile(pathFile) and not 'desktop.ini' in file:
    #         shutil.copy2(pathFile, pathID)

    # Copy GRF
    # for file in os.listdir(pathGRFDrive):
    #     pathFile = os.path.join(pathGRFDrive, file)
    #     if os.path.isfile(pathFile) and not 'desktop.ini' in file:
    #         shutil.copy2(pathFile, pathGRF)

    # Copy IK
    # for file in os.listdir(pathIKDrive):
    #     pathFile = os.path.join(pathIKDrive, file)
    #     if os.path.isfile(pathFile) and not 'desktop.ini' in file:
    #         shutil.copy2(pathFile, pathIK)

    # # Copy SO
    # for file in os.listdir(pathSODrive):
    #     pathFile = os.path.join(pathSODrive, file)
    #     if os.path.isfile(pathFile) and not 'desktop.ini' in file:
    #         shutil.copy2(pathFile, pathSO)

    # # Copy JRA
    # for file in os.listdir(pathJRADrive):
    #     pathFile = os.path.join(pathJRADrive, file)
    #     if os.path.isfile(pathFile) and not 'desktop.ini' in file:
    #         shutil.copy2(pathFile, pathJRA)

    # # Copy EMG
    # for file in os.listdir(pathEMGDrive):
    #     pathFile = os.path.join(pathEMGDrive, file)
    #     if os.path.isfile(pathFile) and not 'desktop.ini' in file:
    #         shutil.copy2(pathFile, pathEMG)

    # # Copy model mocap
    # for file in os.listdir(pathModelMocapDrive):
    #     files_to_copy = [
    #         'LaiArnoldModified2017_poly_withArms_weldHand_generic.osim',
    #         'LaiArnoldModified2017_poly_withArms_weldHand_scaled.mot',
    #         'LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim',
    #         'Setup_Scale_LaiArnoldModified2017_poly_withArms_weldHand_scaled.xml'
    #         ]
    #     if file in files_to_copy:
    #         pathFile = os.path.join(pathModelMocapDrive, file)
    #         if os.path.isfile(pathFile) and not 'desktop.ini' in file:
    #             shutil.copy2(pathFile, pathModelMocap)

    # # Change model name
    # for file in os.listdir(pathModelMocap):
    #     if 'LaiArnoldModified2017_poly_withArms_weldHand' in file:
    #         fileName = file.replace('LaiArnoldModified2017_poly_withArms_weldHand', 'LaiUhlrich2022')
    #         os.rename(os.path.join(pathModelMocap, file), os.path.join(pathModelMocap, fileName))

