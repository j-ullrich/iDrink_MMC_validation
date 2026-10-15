"""We compare each MMC-Trial Objects time series and Murphy Measures to the same metrics of the OMC-trials."""
import os
import sys
import pandas as pd
from iDrink import iDrinkUtilities
murphy_measures = ["PeakVelocity_mms",
                   "elbowVelocity",
                   "tTopeakV_s",
                   "tToFirstpeakV_s",
                   "tTopeakV_rel",
                   "tToFirstpeakV_rel",
                   "NumberMovementUnits",
                   "InterjointCoordination",
                   "ElbowExtension",
                   "ShoulderFlexionReaching",
                   "shoulderFlexionDrinking",
                   "shoulderAbduction",
                   "trunkDisplacementMM",
                   "trunkDisplacementDEG",
                   ]
ignore_id_p = ['P11', 'P19']
idx_s_singlecam = ['S017', 'S018', 'S019', 'S020', 'S021', 'S022', 'S023', 'S024', 'S025', 'S026']
idx_s_singlecam = ['S017', 'S018']
idx_s_multicam = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010', 'S011', 'S012', 'S013', 'S014', 'S015', 'S016']
idx_s_multicam_reduced = ['S001', 'S002']
idx_s_reduced = idx_s_multicam_reduced + idx_s_singlecam
idx_s_full = idx_s_multicam + idx_s_singlecam

# The functions live in iDrink.stats.*, re-exported so that existing imports keep working.
from iDrink.stats.preprocessing import delete_existing_files, get_offset_unfancy, normalize_data, preprocess_timeseries
from iDrink.stats.errors import get_error_mean_rmse, get_error_timeseries, get_omc_dirs, get_omc_mmc_error_old, get_rom_rmse, get_rom_rmse_old, omc_trial_present
from iDrink.stats.correlation import get_mmc_omc_difference, get_multiple_correlations, get_murphy_corrrelation, get_timeseries_correlations
from iDrink.stats.discrete import runs_statistics_discrete

if __name__ == '__main__':
    # this part is for Development and Debugging

    debug = False
    if sys.gettrace() is not None:
        print("Debug Mode is activated\n"
              "Starting debugging script.")
        debug=True


    root_iDrink = iDrinkUtilities.get_root_iDrink()
    root_val = os.path.join(root_iDrink, "validation_root")
    root_stat = os.path.join(root_val, '04_Statistics')
    root_omc = os.path.join(root_val, '03_data', 'OMC_new', 'S15133')
    root_data = os.path.join(root_val, "03_data")
    root_logs = os.path.join(root_val, "05_logs")

    # Prepare Logging Paths
    log_val_settings = os.path.join(root_logs, "validation_settings.csv")
    log_val_trials = os.path.join(root_logs, "validation_trials.csv")
    log_val_errors = os.path.join(root_logs, "validation_errors.csv")

    # prepare statistic paths
    path_csv_murphy_timestamps = os.path.join(root_stat, '02_categorical', 'murphy_timestamps.csv')
    path_csv_murphy_measures = os.path.join(root_stat, '02_categorical', 'murphy_measures.csv')

    df_settings = pd.read_csv(log_val_settings, sep=';')  # csv containing information for the various settings in use.

    test_timeseries = False
    corrections = ['fixed', 'dynamic']

    dir_processed = os.path.join(root_data, 'preprocessed_data')
    dir_results = os.path.join(root_stat, '01_continuous', '01_results')
    det_outliers = ['elbow', 'endeff']
    hand_vel_thresh = 3000
    thresh_elbowVelocity = 200

    if test_timeseries:

        for correct in corrections:
            debug = False

            preprocess_timeseries(root_val,
                                  downsample=True, drop_last_rows=False, detect_outliers= det_outliers,
                                  joint_vel_thresh=thresh_elbowVelocity, hand_vel_thresh=hand_vel_thresh, correct=correct, fancy_offset=False,
                                  verbose=1, plot_debug=False, print_able=False, empty_dst=True, debug=debug, debug_c=50)
            dir_src = '02_fully_preprocessed' if correct == 'fixed' else '03_fully_preprocessed_dynamic'
            dir_src = os.path.join(root_data, 'preprocessed_data', dir_src)
            normalize_data(dir_src=dir_src, dynamic = True if correct == 'dynamic' else False, verbose=1)
        
        

        get_error_timeseries(dir_processed = dir_processed, dir_results = dir_results, empty_dst=False, verbose=1, debug=debug)
        get_error_mean_rmse(dir_results, overwrite_csvs=False, verbose=1)
        get_rom_rmse_old(dir_results, overwrite_csvs=False, verbose=1)
        get_timeseries_correlations(dir_processed, dir_results, overwrite_csvs=False, verbose=1)
        get_rom_rmse(dir_results, overwrite_csvs=False, verbose=1)

        





    else:

        runs_statistics_discrete(path_csv_murphy_measures, root_stat, make_plots=True,
                                 thresh_PeakVelocity_mms=None, thresh_elbowVelocity=None, reduced_analysis=False, outlier_iqr=False)

        """runs_statistics_discrete(path_csv_murphy_measures, root_stat, make_plots=True,
                                 thresh_PeakVelocity_mms=hand_vel_thresh, thresh_elbowVelocity=thresh_elbowVelocity)"""
