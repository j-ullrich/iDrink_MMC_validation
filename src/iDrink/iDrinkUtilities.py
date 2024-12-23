import glob
import os
import shutil
from tqdm import tqdm
import pandas as pd

import toml


def edit_xml(xml_file, xml_keys, new_values, target_file=None, filename_appendix=None):
    from xml.etree import ElementTree

    """
    by default, this function edits an existing file. 
    If target Directory is given, a new xml_file is created.
    A new Appendix to the new filename can be given. e.g. The Session/Participant ID
    
    Input:
        xml_file: path to xml_file to edit
        xml_keys: list of keys in file to be edited
        new_values: the new values
            ith value corresponds to ith key
        
        Possible changes to function: 
        Change so Input changes to:
        
        table: pandas Dataframe containing:
            - the path to the default xml_files
            - the path for new and edited xml_files
            - key that will be changed
            - new values
    """

    tree = ElementTree.parse(xml_file)
    root = tree.getroot()

    # Find the element to be edited
    for i in range(len(xml_keys)):
        for key in root.iter(xml_keys[i]):
            # print(f"changing {key.text} to {new_value}")
            key.text = new_values[i]  # Change the value of the element

    # check create_new_file
    if target_file is not None:  # if target file is given, create new file
        # check whether destination_folder exists
        target_dir = os.path.dirname(target_file)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if filename_appendix is not None:
            basename, extension = os.path.splitext(os.path.basename(xml_file))
            filename = f"{basename}_{filename_appendix}{extension}"
            target_file = os.path.join(target_dir, filename)

        try:
            with open(target_file, "wb") as f:
                tree.write(f)
        except Exception as e:
            print(f"Error: {e}")

    else:  # edit existing file
        try:
            tree.write(xml_file)
        except Exception as e:
            print(f"Error: {e}")


def write_to_config(old_file, new_file, category, variable_dict=None):
    """
    Loads a toml file to a dictionary, adds/changes values and saves back to the toml.

    There are two ways this function can be used

    1. category as dictionary:
        Keys are names of the categories, values are dictionaries equal to variable_dict in second way.

    2. category as string and variable_dict as dictionary:
        Category tells the category to change while variable_dict all the variables of this category to be changed.

    input:
    old_file: Path to toml file to be changed
    new_file: Path to save the toml file
    category: category of changes, dict or str
    variable_dict: dictionary containing the keys and variables to be changed., None or dict
        keys have to be the same as in the toml file.
    """

    temp = toml.load(old_file)

    if type(category) == dict:
        for cat in category:
            for key in category[cat]:
                """if type(category[cat][key]) == dict:
                    for sub_key in category[cat][key]:
                        temp[cat][key][sub_key] = category[cat][key][sub_key]
                else:"""
                temp[cat][key] = category[cat][key]

    elif type(category) == str and type(variable_dict) == dict:
        for key in variable_dict:
            temp[category][key] = variable_dict[key]

    else:
        print("Input valid.\n"
              "category must be dict or str\n"
              "variable_dict must be None or dict")

    with open(new_file, 'w+') as f:
        toml.dump(temp, f)

    return temp


def get_reference_data(curr_trial):
    """
    This function returns the path of the healthy data corresponding to the given task.

    It first tries to retrieve the file based on the config-File.

    If the path in the config file doesn't exist, or there is no path given, it will look in the reference folder.

    If there is no file, the function returns None.

    Input:
        - config: dictionary
        - root_dir
        - trial_file
        - task
    """

    # TODO: In Gui, user might want to choose healthy data themself.

    try:
        reference_data = curr_trial.path_reference_data
    except Exception as e:
        print(f"Reference Data saved in current trial could not be loaded.\n"
              f"New reference File searched in Reference_Data Folder"
              f"{e}")
        reference_data = None
    if reference_data:
        if not os.path.exists(reference_data):
            print(f"Healthy-File does not exist. Please check path: {reference_data}")
            reference_data = None
    if not reference_data:
        pattern = os.path.join(curr_trial.dir_reference, f"{curr_trial.task}_*healthy*.csv")
        found_files = glob.glob(pattern)
        if found_files:
            curr_trial.path_reference_data = found_files[0]
            """reference_data = found_files[0]
            config = write_to_config(trial_file, trial_file, "other", {"reference_data": reference_data})
            print(f"Found Healthy-File in {reference_data}")
        else:
            config = write_to_config(trial_file, trial_file, "other", {"reference_data": []})
            print(f"No Healthy-File found in Reference_Data.")
            reference_data = None"""
    return reference_data


def choose_from_dict(dict, choice):
    # TODO: When in GUI. Let User choose from List. based on list(dict.keys())

    pick = dict[choice]
    # pick = input(f"Please choose one of the following: {dict}")
    possible_picks = list(dict.values())

    # create switch Dictionary, find the chosen key and return corresponding value.
    switch = {pick: "{:02d}".format(i + 1) for i, pick in enumerate(possible_picks)}
    switch_ID = switch.get(pick, f"Please choose one of the following: {list(dict.keys())}")

    return switch_ID, pick  # return ID and task name.


def find_video_recordings(directory, video_format=None):
    """
    This function takes a directory and returns all videofiles it can find in it and its subdirectories:

    If format is given,
    Currently, it accepts, .mp4, .avi, .mov, and .mkv files.

    input: directory to search through

    output: list of video paths
    """
    if video_format is not None:
        formats = []
        patterns = [os.path.join(rf"{directory}\**", f"*{video_format}")]
    else:
        formats = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        patterns = [os.path.join(rf"{directory}\**", f) for f in formats]

    video_files = []
    for pattern in patterns:
        video_files.extend(glob.glob(pattern))

    return video_files


def move_json_to_trial(trial, poseback, filt, root_val, json_dst='pose', verbose=1):
    """
    Find folder containing json files for the current trial.

    It first retrieves the json directories containing the json files from pose estimation.

    Then it creates the corresponding json directories in the trial folder and copies the json files into the new directories.
    """
    if poseback == 'metrabs_multi':
        poseback = 'metrabs'
        #json_dst = 'pose-associated'

    if filt == 'unfiltered':
        filt = '01_unfiltered'
    else:
        filt = '02_filtered'

    id_t = f"trial_{int(trial.id_t.split('T')[1])}"
    id_p = trial.id_p
    cams = [f'cam{cam}' for cam in trial.used_cams]

    # get the filter
    dir_p = os.path.realpath(os.path.join(root_val, '02_pose_estimation', filt, id_p))  # participant directory, containing camera folders
    dir_j = os.path.realpath(os.path.join(root_val, '02_pose_estimation', filt, id_p, f'{id_p}_cam' ))

    dir_pose = os.path.realpath(os.path.join(trial.dir_trial, json_dst))

    """for cam in cams:
        json_dirs = glob.glob(os.path.join(dir_p, f"{id_p}_{cam}", poseback, f"{id_t}_*_json"))
        for json_dir_src in json_dirs:
            basename
            json_dir_dst = os.path.join(dir_pose, os.path.basename(json_dir_src))
            if not os.path.exists(json_dir_dst):
                shutil.copytree(json_dir_src, json_dir_dst)
            else:
                print(f"Directory {json_dir_dst} already exists.")"""

    cam_json = [[cam, glob.glob(os.path.join(dir_p, f"{id_p}_{cam}", poseback, f"{id_t}_*_json"))[0]] for cam in cams]
    if verbose>=1:
        prog = tqdm(cam_json, desc="Copying json files",unit='folder', position=0, leave=True)
    for cam, json_dir_src in cam_json:
        basename = f"{trial.identifier}_{cam}_{os.path.basename(json_dir_src)}"
        json_dir_dst = os.path.join(dir_pose, basename)

        if not os.path.exists(json_dir_dst):
            shutil.copytree(json_dir_src, json_dir_dst)
        else:
            print(f"Directory {json_dir_dst} already exists.")
        if verbose >= 1:
            prog.update(1)

    if verbose >= 1:
        prog.close()


def del_json_from_trial(trial, pose_only=True, verbose=1):
    """
    Deletes all json files in the trial folder.
    """

    # TODO: Check for correctness
    # look for all Folders ending with _json and delete them
    if pose_only:
        json_dirs = glob.glob(os.path.join(trial.dir_trial, 'pose', '*_json'))
    else:
        json_dirs = glob.glob(os.path.join(trial.dir_trial, '**', '*_json'), recursive=True)

    if verbose >= 1:
        prog = tqdm(json_dirs, desc="Deleting json files", position=0, leave=True)

    for dir in json_dirs:
        shutil.rmtree(dir)
        if verbose >= 1:
            prog.update(1)
    if verbose >= 1:
        prog.close()

def del_json_from_root(root, verbose=1):
    """Deletes all json files that are in a folder structrue with root as root directory"""

    # Look for are .json files in tree and delete themd

    json_dirs = glob.glob(os.path.join(root, '**', '*_json.zip'), recursive=True)
    if json_dirs == []:
        json_dirs = glob.glob(os.path.join(root, '**', '*_json'), recursive=True)
    if json_dirs == []:
        json_files = glob.glob(os.path.join(root, '**', '*.json'), recursive=True)



    if json_dirs != []:
        if verbose >= 1:
            prog = tqdm(json_dirs, desc="Deleting json Directories", position=0, leave=True, unit='dir')
        for dir in json_dirs:
            shutil.rmtree(dir)
            if verbose >= 1:
                prog.update(1)
        if verbose >= 1:
            prog.close()
    else:
        if verbose >= 1:
            prog = tqdm(json_files, desc="Deleting json files", position=0, leave=True, unit='file')
        for file in json_files:
            os.remove(file)
            if verbose >= 1:
                prog.update(1)
        if verbose >= 1:
            prog.close()


def del_geometry_from_trial(trial, verbose=1):
    """
    Deletes Geometry folder from trial directory.
    :param trial:
    :param pose_only:
    :param verbose:
    :return:
    """

    geometry_dir = os.path.join(trial.dir_trial, 'Geometry')

    if os.path.exists(geometry_dir):
        shutil.rmtree(geometry_dir)
        if verbose >= 1:
            print(f"\n"
                  f"Deleted Geometry Folder in {trial.identifier}\n"
                  f"")


def pack_as_zip(directory, verbose=1):
    """
    Repacks a folders content as zip file.

    The zip file is saved in the directory, the files are deleted.

    :param directory:
    :param filetype:
    :param verbose:
    :return:
    """
    import zipfile

    zip_file = os.path.join(directory, os.path.basename(directory) + '.zip')

    with zipfile.ZipFile(zip_file, 'w') as zip_ref:
        for root, dirs, files in os.walk(directory):
            if verbose >= 1:
                progress = tqdm(files, desc=f"Packing Files in {root}", unit="file", position=0, leave=True)
            for file in files:
                file_path = os.path.join(root, file)
                if file_path != zip_file:
                    zip_ref.write(file_path, os.path.relpath(file_path, directory))
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Could not delete file: {file_path}.\n'
                              f'{e}')

                if verbose >= 1:
                    progress.update(1)
            if verbose >= 1:
                progress.close()
    return zip_file


def unpack_zip_into_directory(zip_file, directory, delete_zip=False, verbose=1):
    """
    Unpacks a zip file into a directory.

    if delete_zip is True, the zip file is deleted after unpacking.

    :param zip_file:
    :param directory:
    :param delete_zip:
    :param verbose:
    :return:
    """
    import zipfile

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(path=directory)

    if delete_zip:
        os.remove(zip_file)

def unpack_zip_to_trial(trial, poseback, filt, root_val, json_dst='pose', verbose=1):
    """
    Unpack zip folder containning json_files into the the trial folder for pose2sim.
    
    :param trial: 
    :param poseback: 
    :param filt: 
    :param root_val: 
    :param json_dst: 
    :param verbose: 
    :return: 
    """

    if poseback == 'metrabs_multi':
        poseback = 'metrabs'
        #json_dst = 'pose-associated'

    if filt == 'unfiltered':
        filt = '01_unfiltered'
    else:
        filt = '02_filtered'

    id_t = f"trial_{int(trial.id_t.split('T')[1])}"
    id_p = trial.id_p
    cams = [f'cam{cam}' for cam in trial.used_cams]

    # get the filter
    dir_p = os.path.realpath(os.path.join(root_val, '02_pose_estimation', filt, id_p))  # participant directory, containing camera folders
    dir_j = os.path.realpath(os.path.join(root_val, '02_pose_estimation', filt, id_p, f'{id_p}_cam' ))

    dir_pose = os.path.realpath(os.path.join(trial.dir_trial, json_dst))

    cam_json = [[cam, glob.glob(os.path.join(dir_p, f"{id_p}_{cam}", poseback, f"{id_t}_*_json", "*.zip"))[0]] for cam in cams]
    if verbose>=1:
        prog = tqdm(cam_json, desc="Copying json files",unit='folder', position=0, leave=True)


    for cam, json_dir_src in cam_json:
        basename = f"{trial.identifier}_{cam}_{os.path.basename(json_dir_src).split('.zip')[0]}"
        json_dir_dst = os.path.join(dir_pose, basename)



        if not os.path.exists(json_dir_dst):
            os.makedirs(json_dir_dst, exist_ok=True)
            unpack_zip_into_directory(json_dir_src, json_dir_dst)

        else:
            print(f"Directory {json_dir_dst} already exists.")
        if verbose >= 1:
            prog.update(1)

    if verbose >= 1:
        prog.close()

def copy_files_from_to_dir(dir_src, dir_dst, empty_dst=False, filetype=None, verbose=1):
    """
    Move files from one directory to another.

    if filetype is given, only files with this filetype are moved.

    if empty_dst is True, the destination directory is emptied before moving files.

    :param dir_src:
    :param dir_dst:
    :param filetype:
    :param verbose:
    :return:
    """

    if not os.path.exists(dir_dst):
        os.makedirs(dir_dst, exist_ok=True)

    if empty_dst:
        for file in glob.glob(os.path.join(dir_dst, "*")):
            os.remove(file)

    if filetype is not None:
        files = glob.glob(os.path.join(dir_src, f"*.{filetype}"))
    else:
        files = glob.glob(os.path.join(dir_src, "*"))

    if verbose >= 1:
        prog = tqdm(files, desc="Moving Files", unit="file", position=0, leave=True)

    for file in files:
        shutil.copy2(file, dir_dst, )

        if verbose >= 1:
            prog.update(1)

    if verbose >= 1:
        prog.close()

def get_valid_trials(csv_murphy):
    """
    returns list of valid trials in form of 'P*T*'
    :param csv_murphy:
    :return valid_trials:
    """
    df = pd.read_csv(csv_murphy, sep=';')
    df = df[df['valid'] == 1]
    valid_trials = []
    # get list of if_p and id_t
    list_id_p = df['id_p'].values.tolist()
    list_id_t = df['id_t'].values.tolist()


    for id_p, id_t in zip(list_id_p, list_id_t):

        valid_trials.append(f"{id_p}*T{id_t}")
    return valid_trials

def get_drivepath():
    """
    Returns the drive path based on the machine the code is run on.
    :return:
    """
    import platform
    drives = ['C:', 'D:', 'E:', 'I:']

    # Get drive based on machines validation is run on
    match platform.uname().node:
        case 'DESKTOP-N3R93K5':
            drive = drives[1] + '\\'
        case 'DESKTOP-0GLASVD':
            drive = drives[2] + '\\'
        case _:  # Default case
            drive = drives[3] + '\\'

    return drive

def get_title_measure_name(measure, add_unit = False):
    """returns a string based on the murphy measure for a figure_title"""
    match measure:
        case 'PeakVelocity_mms':
            title = 'Peak Endeffector Velocity'
        case 'elbowVelocity':
            title = 'Peak Elbow Velocity'
        case 'tTopeakV_s':
            title = 'Time to Peak Velocity'
        case 'tToFirstpeakV_s':
            title = 'Time to First Peak Velocity'
        case 'tTopeakV_rel':
            title = 'Relative time to Peak Velocity relative'
        case 'tToFirstpeakV_rel':
            title = 'Relative time to First Peak Velocity'
        case 'NumberMovementUnits':
            title = 'Number of Movement Units'
        case 'InterjointCoordination':
            title = 'Interjoint Coordination'
        case 'trunkDisplacementMM':
            title = 'Maximum Trunk Displacement'
        case 'trunkDisplacementDEG':
            title = 'Maximum Trunk Displacement'
        case 'ShoulderFlexionReaching':
            title = 'Shoulder Flexion Reaching'
        case 'ElbowExtension':
            title = 'Maximum Elbow Extension Reaching'
        case 'shoulderAbduction':
            title = 'Maximum Shoulder Abduction Reaching'
        case 'shoulderFlexionDrinking':
            title = 'Shoulder Flexion Drinking'
        case 'hand_vel' | 'Hand Velocity [mm/s]':
            title = 'Hand Velocity'
        case 'elbow_vel' | 'Elbow Velocity [deg/s]':
            title = 'Elbow Velocity'
        case 'trunk_disp' | 'Trunk Displacement [mm]':
            title = 'Trunk Displacement'
        case 'trunk_ang':
            title = 'Trunk Angle'
        case 'elbow_flex_pos' | 'Elbow Flexion [deg]':
            title = 'Elbow Flexion'
        case 'shoulder_flex_pos' | 'Shoulder Flexion [deg]':
            title = 'Shoulder Flexion'
        case 'shoulder_abduction_pos' | 'Shoulder Abduction [deg]':
            title = 'Shoulder Abduction'
        case _:
            title = measure

    if add_unit:
        unit = get_unit(measure)
        if unit != '':
            title = f"{title} [{unit}]"

    return title

def get_unit(kin):

    cases_deg = ['trunk_ang',
                 'elbow_flex_pos', 'Elbow Flexion', 'Elbow Flexion [deg]',
                 'shoulder_flex_pos', 'Shoulder Flexion', 'Shoulder Flexion [deg]',
                 'shoulder_abduction_pos', 'Shoulder Abduction', 'Shoulder Abduction [deg]',
                 'trunkDisplacementDEG', 'trunk_ang',
                 'ShoulderFlexionReaching', 'ElbowExtension',
                 'shoulderAbduction', 'shoulderFlexionDrinking',
                 ]

    match kin:
        case 'hand_vel' | 'PeakVelocity_mms' | 'Hand Velocity'  | 'Hand Velocity [mm/s]':
            unit = 'mm/s'
        case 'elbow_vel' | 'elbowVelocity' | 'Elbow Velocity' | 'Elbow Velocity [deg/s]':
            unit = 'deg/s'
        case 'trunk_disp' | 'trunkDisplacementMM' | 'Trunk Displacement [mm]':
            unit = 'mm'
        case k if k in cases_deg:
            unit = 'deg'
        case 'tTopeakV_s' | 'tToFirstpeakV_s' :
            unit = 's'
        case 'tTopeakV_rel' | 'tToFirstpeakV_rel':
            unit = '%'
        case _:
            unit = ''

    return unit

def get_cad(df, measure):
    match measure:
        case 'PeakVelocity_mms' | 'hand_vel':
            measure_name = 'peak_V'
        case 'elbowVelocity' | 'elbow_vel':
            measure_name = 'peak_V_elb'
        case 'tTopeakV_s':
            measure_name = 't_to_PV'
        case 'tToFirstpeakV_s':
            measure_name = 't_first_PV'
        case 'tTopeakV_rel':
            measure_name = 't_PV_rel'
        case 'tToFirstpeakV_rel':
            measure_name = 't_first_PV_rel'
        case 'NumberMovementUnits':
            measure_name = 'n_mov_units'
        case 'InterjointCoordination':
            measure_name = 'interj_coord'
        case 'trunkDisplacementMM' | 'trunk_disp':
            measure_name = 'trunk_disp'
        case 'trunkDisplacementDEG' | 'trunk_ang':
            return None
        case 'ShoulderFlexionReaching' | 'elbow_flex_pos':
            measure_name = 'arm_flex_reach'
        case 'ElbowExtension' | 'shoulder_flex_pos':
            measure_name = 'elb_ext'
        case 'shoulderAbduction' | 'shoulder_abduction_pos':
            measure_name = 'arm_abd'
        case 'shoulderFlexionDrinking' | 'shoulder_flex_pos':
            measure_name = 'arm_flex_drink'
        case _:
            return

    return df.loc[0, measure_name]

def get_setting_axis_name(id_s):
    match id_s:
        case 'S001':
            name = 'SimCC, Cams: 1,2,3,4,5'
        case 'S002':
            name = 'Metrabs, Cams: 1,2,3,4,5'
        case 'S003':
            name = 'SimCC, Cams: 6,7,8,9,10'
        case 'S004':
            name = 'Metrabs, Cams: 6,7,8,9,10'
        case 'S005':
            name = 'SimCC, Cams: 1,3,5'
        case 'S006':
            name = 'Metrabs, Cams: 1,3,5'
        case 'S007':
            name = 'SimCC, Cams: 2,3,4'
        case 'S008':
            name = 'Metrabs, Cams: 2,3,4'
        case 'S009':
            name = 'SimCC, Cams: 6,8,10'
        case 'S010':
            name = 'Metrabs, Cams: 6,8,10'
        case 'S011':
            name = 'SimCC, Cams: 7,8,9'
        case 'S012':
            name = 'Metrabs, Cams: 7,8,9'
        case 'S013':
            name = 'SimCC, Cams: 2,4'
        case 'S014':
            name = 'Metrabs, Cams: 2,4'
        case 'S015':
            name = 'SimCC, Cams: 7,9'
        case 'S016':
            name = 'Metrabs, Cams: 7,9'
        case 'S017':
            name = 'Single, Cam: 1, filt'
        case 'S018':
            name = 'Single, Cam: 1, unfilt'
        case 'S019':
            name = 'Single, Cam: 2, filt'
        case 'S020':
            name = 'Single, Cam: 2, unfilt'
        case 'S021':
            name = 'Single, Cam: 3, filt'
        case 'S022':
            name = 'Single, Cam: 3, unfilt'
        case 'S023':
            name = 'Single, Cam: 7, filt'
        case 'S024':
            name = 'Single, Cam: 7, unfilt'
        case 'S025':
            name = 'Single, Cam: 8, filt'
        case 'S026':
            name = 'Single, Cam: 8, unfilt'
        case 'S15133':
            name = 'OMC reference'
        case _:
            name = id_s

    return name


def get_measure_short_name(measure):
    """Returns the short versino of the kinematic measures name"""

    match measure:
        case 'PeakVelocity_mms':
            return 'peak_V'
        case 'elbowVelocity':
            return 'peak_V_elb'
        case 'tTopeakV_s':
            return 't_to_PV'
        case 'tToFirstpeakV_s':
            return 't_first_PV'
        case 'tTopeakV_rel':
            return 't_PV_rel'
        case 'tToFirstpeakV_rel':
            return 't_first_PV_rel'
        case 'NumberMovementUnits':
            return 'n_mov_units'
        case 'InterjointCoordination':
            return 'interj_coord'
        case 'trunkDisplacementMM':
            return 'trunk_disp'
        case 'trunkDisplacementDEG':
            return 'trunk_disp_deg'
        case 'ShoulderFlexionReaching':
            return 'arm_flex_reach'
        case 'ElbowExtension':
            return 'elb_ext'
        case 'shoulderAbduction':
            return 'arm_abd'
        case 'shoulderFlexionDrinking':
            return 'arm_flex_drink'
        case _:
            return measure


if __name__ == '__main__':
    csv = r"I:\iDrink\validation_root\04_statistics\02_categorical\murphy_measures.csv"

    get_valid_trials(csv)



