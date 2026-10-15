import os
from tqdm import tqdm
from iDrink import iDrinkUtilities
ignore_id_p = []
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
rgba_mmc = '100, 149, 237'
rgba_omc = '255, 165, 0'
ignore_id_p = ['P11', 'P19']
idx_s_singlecam_full = ['S017', 'S018', 'S019', 'S020', 'S021', 'S022', 'S023', 'S024', 'S025', 'S026']
idx_s_singlecam = ['S017', 'S018']
idx_s_multicam = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010', 'S011', 'S012', 'S013', 'S014', 'S015', 'S016']
idx_s_multicam_reduced = ['S001', 'S002']
idx_s_reduced = idx_s_multicam_reduced + idx_s_singlecam
idx_s_full = idx_s_multicam + idx_s_singlecam

# The functions live in iDrink.plots.*, re-exported so that existing imports keep working.
from iDrink.plots.murphy import plot_measured_vs_errors, plot_murphy_blandaltman, plot_murphy_blandaltman_old, plot_murphy_error_rmse_box_bar_plot
from iDrink.plots.timeseries import generate_plots_for_timeseries, generate_plots_grouped_different_settings, get_plottable_timeseries_kinematics, plot_timeseries_RMSE, plot_timeseries_averaged, plot_timeseries_barplot_error_rmse, plot_timeseries_blandaltman_scale_location, plot_timeseries_boxplot_error_rmse, plot_two_timeseries, write_plottable_identifier
from iDrink.plots.calibration import calibration_boxplot

if __name__ == "__main__":
    cols_of_interest = ['']

    id_s = 'S001'

    """Set Root Paths for Processing"""

    root_iDrink = iDrinkUtilities.get_root_iDrink()
    root_val = os.path.join(root_iDrink, "validation_root")
    root_stat = os.path.join(root_val, '04_Statistics')
    root_omc = os.path.join(root_val, '03_data', 'OMC_new', 'S15133')
    root_data = os.path.join(root_val, "03_data")
    root_logs = os.path.join(root_val, "05_logs")


    dir_processed = os.path.join(root_val, '03_data', 'preprocessed_data', '02_fully_preprocessed')
    #dir_processed = os.path.join(root_val, '03_data', 'preprocessed_data', '03_fully_preprocessed_dynamic')

    if 'dynamic' in dir_processed:
        dynamic = True
        dynamic_str = 'dynamic'
    else:
        dynamic = False
        dynamic_str = 'fixed'

    verbose = 1

    csv_calib_errors = os.path.join(root_logs, 'calib_errors.csv')
    calib_plots_dst = os.path.join(root_stat, '05_calibration')
    calibration_boxplot(csv_calib_errors, calib_plots_dst, verbose=1, show_fig=False)

    plot_murphy_blandaltman(root_stat, write_html=False, write_svg=False, write_png=True, show_plots=False, verbose=1)

    for corr in [False]:
        plot_murphy_error_rmse_box_bar_plot(root_val, write_html=False, write_svg=False, write_png=True, outlier_corrected=corr)

    plot_timeseries_boxplot_error_rmse(root_val, showfig=False, write_html=False, write_svg=True, write_png=True, verbose=1)

    csv_plottable, set_sp_tuples = write_plottable_identifier(root_val, dir_processed,
                                               to_plot='preprocessed_timeseries', verbose=1)

    for tuple in set_sp_tuples:
        progbar_tuple = tqdm(total=1, desc=f'Plotting {tuple}', unit='Tuple', disable=verbose < 1)
        progbar_tuple.set_description(f'Plotting {tuple}')
        id_s = tuple[0]
        id_p = tuple[1]
        plot_timeseries_averaged(root_val, id_s, id_p, dynamic=dynamic)
        progbar_tuple.update(1)
    progbar_tuple.close()

    df_plottable = get_plottable_timeseries_kinematics(csv_plottable, 2, affected='unaffected', verbose=1)

    kinematics = ['hand_vel', 'elbow_vel', 'trunk_disp', 'trunk_ang', 'elbow_flex_pos', 'shoulder_flex_pos',
                  'shoulder_abduction_pos']

    # TODO: Make sure, MMC and OMC have always the same colour in Plots MMC - Blue, OMC - Orange check on averaged timeseries

    kinematic = kinematics[4]
    # iterate over all plottable trials and create plots

    for i in range(len(df_plottable)):
        #id_s = df_plottable['id_s'][i]
        id_p = df_plottable['id_p'][i]
        id_t = df_plottable['id_t'][i]

        generate_plots_for_timeseries(root_val, id_p_in = id_p, id_t_in = id_t, dynamic=dynamic,
                                      showfig = False, write_html=False, write_svg=True)


        for kinematic in kinematics:
            plot_timeseries_blandaltman_scale_location(root_val, kinematic=kinematic, idx_p=id_p, idx_t=id_t,
                                                       dynamic=dynamic_str,
                                                       write_html=False, write_svg=True, write_png=True,
                                                       show_plots=False)
            pass
