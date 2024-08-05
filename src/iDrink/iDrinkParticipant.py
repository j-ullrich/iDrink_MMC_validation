import os
import queue
import time
import pandas as pd

"""
Class for Participant Objects

This Class is meant to store all values necessary concerning a participants / Patients Measurments.

In here are all the functions needed for recording and trial cutting.

It must not contain any information necessary for running a trial analysis.
Any Information that is needed for a trial needs to be transfered to the trial object when creating it.

Participant objects should contain the following informations:
- id_s: Session ID
- id_p: PID (participant / patient ID)

Paths to directories:
- dir_root: Directory containing default, reference, session data and global settings file
- dir_default: directory containing all default files
- dir_ref: Directory containing reference files
- dir_calib: Directory containing calibration files and videos
- dir_video_raw: Directory containing all videos recorded in the current session.

"""

class Participant:
    def __init__(self, id_s, id_p, dir_root, dir_session, used_calib, assessement="", rec_resolution=(1920, 1080), frame_rate=60,
                 clean_video = True, record_duration=60 * 10, pose_model="Coco18_UpperBody", cams_chosen=None):
        self.dir_root = dir_root
        self.dir_default = os.path.realpath(os.path.join(self.dir_root, 'Default-Files'))
        self.dir_ref = os.path.realpath(os.path.join(self.dir_root, 'Reference_Data'))
        self.dir_session = dir_session

        self.id_s = id_s
        self.id_p = id_p
        self.identifier = f"{self.id_s}_{self.id_p}"
        self.dir_participant = os.path.realpath(os.path.join(self.dir_session, f'{self.id_s}_{self.id_p}'))

        self.dir_calib = os.path.realpath(os.path.join(self.dir_session, f'{self.id_p}_calibration'))
        self.dir_calib_videos = os.path.realpath(os.path.join(self.dir_calib, f'{self.id_p}calibration_videos'))
        self.dir_calib_files = os.path.realpath(os.path.join(self.dir_calib, f'{self.id_p}calibration_files'))
        self.used_calib = used_calib

        self.dir_recordings = os.path.realpath(os.path.join(self.dir_participant, f'{self.identifier}_recordings'))

        self.cams_chosen = cams_chosen
        self.rec_resolution = rec_resolution
        self.frame_rate = frame_rate
        self.clean_video = clean_video
        self.record_duration = record_duration

        self.video_files = None
        self.t_start_rec = None
        self.t_end_rec = None

        self.assessement = assessement
        self.pose_model = pose_model

        self.q_tstart = queue.LifoQueue()
        self.q_tend = queue.LifoQueue()
        self.q_affected = queue.LifoQueue()
        self.q_side = queue.LifoQueue()
        self.q_task = queue.LifoQueue()
        self.q_skip = queue.LifoQueue()
        self.q_skip_reason = queue.LifoQueue()

    def create_participant(self):
        """
        This function creates the participant folder and the corresponding subfolders.
        """
        # Create the participant folder
        os.makedirs(self.dir_participant, exist_ok=True)

        # Create the recordings folder
        os.makedirs(self.dir_recordings, exist_ok=True)

    def start_trial(self, is_affected, side, task, skip, skip_reason=''):
        """
        This function puts the timestamp into the queue for the start of the trial.

        Input:
            - affected: whether side is affected (True, False)
            - side: measured side (r/l)
            - task: task performed during trial
            - skip: whether trial is skipped (True, False)
        """

        self.q_tstart.put(time.time())
        self.q_affected.put(is_affected)
        self.q_side.put(side)
        self.q_task.put(task)
        self.q_skip.put(skip)
        self.q_skip_reason.put(skip_reason)

    def end_trial(self):
        """
        This function puts the timestamp into the queue for the end of the trial.
        """
        self.q_tend.put(time.time())


    def get_trials(self):
        """
        This function retrieves the trials from the queues and creates the corresponding trial folder and objects.

        The trial objects are put into a list and returned to caller.
        """
        from .iDrinkClassTrial import Trial
        from .iDrinkVisualInput import cut_videos_to_trials


        ntrials = self.q_tend.qsize()
        trial = 0
        list_trials = []
        while True:
            t_end = self.q_tend.get()

            #while t_start > t_end and skip is False:
            while True:
                trial += 1
                t_start = self.q_tstart.get()
                affected = self.q_affected.get()
                side = self.q_side.get()
                task = self.q_task.get()
                skip = self.q_skip.get()
                skip_reason = self.q_skip_reason.get()

                if t_start < t_end or skip is True:
                    break

            id_trial = f"T{trial:0{len(str(ntrials))}d}"
            trial_identifier = self.identifier + f"_{id_trial}"
            dir_trial = os.path.realpath(os.path.join(self.dir_participant, f'{trial_identifier}_{task}'))


            new_trial = Trial(identifier=trial_identifier, id_s=self.id_s, id_p=self.id_p, id_t=id_trial,
                              assessement=self.assessement, task=task, skip=skip, skip_reason=skip_reason,

                              dir_root=self.dir_root, dir_default=self.dir_default, dir_reference=self.dir_ref,
                              dir_session=self.dir_session, dir_calib=self.dir_calib,
                              dir_participant=self.dir_participant, dir_trial=dir_trial,

                              date_time=(t_start, t_end), measured_side=side, affected=affected, use_analyze_tool=True,

                              rec_resolution=self.rec_resolution, frame_rate=self.frame_rate,
                              clean_video=self.clean_video,

                              path_config=os.path.join(dir_trial, f'{trial_identifier}_Config.toml'), pose_model=self.pose_model)

            new_trial.create_trial()
            list_trials.append(new_trial)

            cut_videos_to_trials(self.video_files, new_trial, self.t_start_rec, self.t_end_rec, new_trial.dir_recordings,
                                 new_trial.clean_video)


            if self.q_tend.empty():
                break


        self.trial_list = list_trials
        return list_trials

    def run_recording(self, feed_size):
        """
        Starts video recording for the trial.

        The recording runs until prompted to end.
        During the recording two queues are created to give the start and end of each trial recording.
        The video is then cut into the individual trials. The cut videos are then stored in the corresponding folder.
        """
        from .iDrinkVisualInput import record_videos

        """
        We go backwards through recorded trials.
        We take the last tend and search for the last tstart before that.
        
        This way, therapists don't need to end a trial when something happens and trial needs to restart. 
        They only need to his start again.
        
        The amount of t_end is the amount of trials recorded
        """


        video_files, t_start_rec,t_end_rec = record_videos(cams_chosen=self.cams_chosen, resolution=self.rec_resolution, out_dir=self.dir_recordings, record_duration=self.record_duration,
                      filename_prefix=self.identifier, file_fps=self.frame_rate, feed_size=feed_size)

        self.video_files = video_files
        self.t_start_rec = t_start_rec
        self.t_end_rec = t_end_rec

        return video_files, t_start_rec
