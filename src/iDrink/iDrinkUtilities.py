import glob
import os

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
