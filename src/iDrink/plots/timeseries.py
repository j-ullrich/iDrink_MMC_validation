"""timeseries part of iDrinkValPlots.py (moved without changes, see iDrinkValPlots.py for the public names)."""
import numpy as np
import pandas as pd
import os
import glob
from plotly.io import write_html
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from iDrink.iDrinkUtilities import get_title_measure_name, get_unit, get_cad, get_setting_axis_name, get_measure_plot_num
ignore_id_p = []
rgba_mmc = '100, 149, 237'
rgba_omc = '255, 165, 0'
ignore_id_p = ['P11', 'P19']
idx_s_singlecam = ['S017', 'S018']
idx_s_multicam_reduced = ['S001', 'S002']
idx_s_reduced = idx_s_multicam_reduced + idx_s_singlecam

# plot_timeseries_blandaltman_scale_location reads this global. It used to exist only when the module ran as script.
verbose = 1


def plot_timeseries_blandaltman_scale_location(root_val, kinematic, idx_p=None, idx_t=None, dynamic='static', write_html=True, write_svg=True, write_png=True, show_plots=True):
    """
    Prints all Bland altman and scale location plots and saves them into the correct folder.

    Normalized errors are used for the plots.

    #TODO: Implement settingwise if enough time is left.

    :param root_val:
    :return:
    """
    dir_src = os.path.join(root_val, '04_statistics', '01_continuous', '01_results', '01_ts_error')

    subdir_dyn = '02_dynamic' if dynamic else '01_fixed'
    dir_dst_bland = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '01_omc_mmc_error',
                                 '01_bland_altman', subdir_dyn)
    dir_dst_scale = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '01_omc_mmc_error',
                                 '02_scale_location', subdir_dyn)
    dir_dst_time = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '01_omc_mmc_error',
                                '03_time_error', subdir_dyn)

    unit = get_unit(kinematic)

    for dst in [dir_dst_bland, dir_dst_scale, dir_dst_time]:
        os.makedirs(dst, exist_ok=True)

    list_files_full = glob.glob(os.path.join(dir_src, '*norm.csv'))

    if idx_p is None:
        idx_p = list(set([os.path.basename(file).split('_')[1] for file in list_files_full]))
    elif type(idx_p) is str:
        idx_p = [idx_p]


    for id_p in idx_p:
        list_files_p = [file for file in list_files_full if id_p in file]



        if idx_t is None:
            idx_t = list(set([os.path.basename(file).split('_')[2].split('.')[0] for file in list_files_p]))
        elif type(idx_t) is str:
            idx_t = [idx_t]

        progbar = tqdm(idx_p, desc='Plotting Bland Altman and Scale Location Plots', unit='Patient',
                       disable=verbose < 1)
        for id_t in idx_t:
            files = [file for file in list_files_p if id_t in file]
            if files:
                file = files[0]
            else:
                continue
            progbar.set_description(f'Plotting Bland Altman and Scale Location Plots for {id_p}_{id_t} \t {kinematic}')

            df_pt = pd.read_csv(file, sep=';')
            idx_s = list(set(df_pt['id_s'].values))

            fig_bland = go.Figure()
            fig_scale = go.Figure()

            colours = px.colors.qualitative.Plotly

            list_colours = [colours[i % len(colours)] for i in range(len(idx_s))]

            for id_s in idx_s:
                df = df_pt[(df_pt['id_s'] == id_s) & (df_pt['dynamic'] == dynamic)]

                # Bland Altman Plot
                omc = df[f'{kinematic}_omc'].values
                mmc = df[f'{kinematic}_mmc'].values
                error = df[f'{kinematic}_error'].values

                mean = np.mean([omc, mmc], axis=0)
                time = df['time'].values

                # add data to plot
                fig_bland.add_trace(go.Scatter(x=mean, y=error, mode='markers', name=f'{id_s}', text=df['id_s'],
                                               hoverinfo='text',
                                               line=dict(color=list_colours[idx_s.index(id_s)])))

                # add smoother
                lowess = sm.nonparametric.lowess(error, mean, frac=0.6)
                fig_bland.add_trace(go.Scatter(x=lowess[:, 0], y=lowess[:, 1], mode='lines', name=f'{id_s}', line=dict(color='black')))

                # add line at 0
                fig_bland.add_hline(y=0, line_dash='dash', line_color='grey', name='Zero')

                # add limits of agreement
                std_diff = np.std(error)
                sd = 1.96
                upper_limit = mean + sd * std_diff
                lower_limit = mean - sd * std_diff

                fig_bland.add_hline(y=upper_limit, line_dash='dash', line_color='red', name=f'Upper Limit ({sd} SD)')
                fig_bland.add_hline(y=lower_limit, line_dash='dash', line_color='red', name=f'Lower Limit ({sd} SD)')

                # Scale Location Plot
                fig_scale.add_trace(go.Scatter(x=mmc, y=error, mode='markers', name=f'{id_s}', text=df['id_s'], hoverinfo='text'))
                fig_scale.add_trace(go.Scatter(x=mmc, y=error, mode='lines', name=f'{id_s}', line=dict(color='red')))
                # add line at 0
                fig_scale.add_hline(y=0, line_dash='dash', line_color='grey', name='Zero')
                # add limits of agreement
                fig_scale.add_hline(y=upper_limit, line_dash='dash', line_color='red', name=f'Upper Limit ({sd} SD)')
                fig_scale.add_hline(y=lower_limit, line_dash='dash', line_color='red', name=f'Lower Limit ({sd} SD)')

                # Error to time
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(x=time, y=error, mode='lines', name=f'{id_s}', line=dict(color='red')))
                # add line at 0
                fig_time.add_hline(y=0, line_dash='dash', line_color='grey', name='Zero')

                fig_time.add_trace(go.Scatter(x=time, y=mmc, mode='lines', name='MMC', line=dict(color='blue', dash='dash')))
                fig_time.add_trace(go.Scatter(x=time, y=omc, mode='lines', name='OMC', line=dict(color='green', dash='dash')))



                # update the layout
            fig_bland.update_layout(title=f'Bland Altman Plot for {kinematic} of {id_p}, {id_t}',
                                    xaxis_title=f'Mean of {kinematic}',
                                    yaxis_title=f'Error [{unit}]',
                                    legend=dict(
                                        orientation="h",
                                        x=0,
                                        y=-0.2  # Positionierung unterhalb der x-Achse
                                    )
                                    )


            fig_scale.update_layout(title=f'Bland Altman Plot for {kinematic} of {id_p}, {id_t}',
                                xaxis_title=f'MMC of {kinematic}',
                                yaxis_title=f'Error [{unit}]',
                                legend=dict(
                                    orientation="h",
                                    x=0,
                                    y=-0.2  # Positionierung unterhalb der x-Achse
                                )
                                )

            fig_time.update_layout(title=f'Error over trial {kinematic} of {id_p}, {id_t}',
                                xaxis_title='% of trial',
                                yaxis_title=f'Error [{unit}]')

            if show_plots:
                fig_bland.show()
                fig_scale.show()
                fig_time.show()

            if write_html:
                path_bland = os.path.join(dir_dst_bland, f'{id_p}_{id_t}_{kinematic}.html')
                fig_bland.write_html(path_bland)

                path_scale = os.path.join(dir_dst_scale, f'{id_p}_{id_t}_{kinematic}.html')
                fig_scale.write_html(path_scale)

                path_time = os.path.join(dir_dst_time, f'{id_p}_{id_t}_{kinematic}_time.html')
                fig_time.write_html(path_time)

            if write_svg:
                path_bland = os.path.join(dir_dst_bland, f'{id_p}_{id_t}_{kinematic}.svg')
                fig_bland.write_image(path_bland, scale=5)

                path_scale = os.path.join(dir_dst_scale, f'{id_p}_{id_t}_{kinematic}.svg')
                fig_scale.write_image(path_scale, scale=5)

                path_time = os.path.join(dir_dst_time, f'{id_p}_{id_t}_{kinematic}_time.svg')
                fig_time.write_image(path_time, scale=5)

            if write_png:
                path_bland = os.path.join(dir_dst_bland, f'{id_p}_{id_t}_{kinematic}.png')
                fig_bland.write_image(path_bland, scale=5)

                path_scale = os.path.join(dir_dst_scale, f'{id_p}_{id_t}_{kinematic}.png')
                fig_scale.write_image(path_scale, scale=5)

                path_time = os.path.join(dir_dst_time, f'{id_p}_{id_t}_{kinematic}_time.png')
                fig_time.write_image(path_time, scale=5)

            progbar.update(1)
    progbar.close()


def plot_timeseries_boxplot_error_rmse(root_val, showfig=False, write_html=False, write_svg=True, write_png=True, verbose=1):
    """
    Prints boxplots based on the mean errors and RMSE of the timeseries.

    Plots 2 plots (for now):
        For each paatient:
        - Boxplot of the mean errors of the timeseries x: error y: id_s
        - Boxplot of the RMSE of the timeseries x: RMSE y: id_s

    :param root_val:
    :return:
    """

    dir_src = os.path.join(root_val, '04_statistics', '01_continuous', '01_results')

    csv_in = os.path.join(root_val, '04_statistics', '01_continuous', '01_results', 'omc_mmc_error.csv')

    df_error = pd.read_csv(csv_in, sep=';')

    dir_dst_error_box = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '01_omc_to_mmc_error',
                                 '01_omc_mmc_error', '03_boxplot', '01_error')
    dir_dst_rmse_box = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '01_omc_to_mmc_error',
                                 '01_omc_mmc_error', '03_boxplot', '02_rmse')
    dir_dst_rmse_box_log = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '01_omc_to_mmc_error',
                                 '01_omc_mmc_error', '03_boxplot', '03_rmse_log')

    for dir in [dir_dst_error_box, dir_dst_rmse_box, dir_dst_rmse_box_log]:
        os.makedirs(dir, exist_ok=True)

    """subdir_dyn = '02_dynamic' if dynamic else '01_fixed'

    dir_dst_error_box = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '01_omc_to_mmc_error',
                                 '01_omc_mmc_error', '03_boxplot')
    dir_dst_rmse_box = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '01_omc_to_mmc_error',
                                 '01_omc_mmc_error', '03_boxplot')"""

    list_idx = list(df_error['id'].unique())

    idx_s = []
    idx_p = []

    list_idx_sp = [] # List of tuples containing setting and patient ids, that have been done.

    if verbose >= 1:
        prog = tqdm(list(df_error['id_p'].unique()), desc='Plotting Boxplots', unit='Patient')

    for identifier in list_idx:

        if len(identifier.split('_')) != 3:
            continue

        if verbose >= 1:
            prog.set_description(f'Plotting Boxplots for {identifier}')

        id_p = identifier.split('_')[1]
        if id_p in idx_p:
            continue

        idx_p.append(id_p)


        df = df_error[(df_error['id_p'] == id_p) & df_error['id'].str.contains('T')]

        df['id_s_name'] = df['id_s'].apply(lambda x: get_setting_axis_name(x))

        list_dynamic = list(df['dynamic'].unique())
        list_normalized = list(df['normalized'].unique())

        metrics = list(df['metric'].unique())

        for dynamic in list_dynamic:
            normal = 'original'

            for metric in metrics:

                df_to_plot = df[(df['dynamic'] == dynamic) & (df['normalized'] == normal) & (df['metric'] == metric) & (df['id_s'].isin(idx_s_reduced))]


                unit = get_unit(metric)
                metric_name = get_title_measure_name(metric)

                # Boxplot of the mean errors of the timeseries x: id_s y: mean

                fig_err = px.box(df_to_plot, x='id_s_name', y='mean', color = 'condition',
                             title=f'Mean Error for {metric_name} of {id_p} - {dynamic}',
                             labels={'mean': f'Mean Error [{unit}]', 'id_s': 'Setting'})

                fig_rmse = px.box(df_to_plot, x='id_s_name', y='rmse', color = 'condition',
                                      title=f'Log(RMSE) for {metric_name} of {id_p} - {dynamic}',
                                      labels={'rmse': f'RMSE [{unit}]', 'id_s': 'Setting'})

                fig_log_rmse = px.box(df_to_plot, x='id_s_name', y='rmse', color = 'condition', log_y=True,
                             title=f'Log(RMSE) for {metric_name} of {id_p} - {dynamic}',
                             labels={'rmse': f'Log(RMSE) [{unit}]', 'id_s': 'Setting'})



                if showfig:
                    fig_err.show()
                    fig_rmse.show()
                    fig_log_rmse.show()


                if write_html:
                    path = os.path.join(dir_dst_error_box, f'{id_p}_{metric}_{dynamic}_{normal}_error_box.html')
                    fig_err.write_html(path)

                    path = os.path.join(dir_dst_rmse_box, f'{id_p}_{metric}_{dynamic}_{normal}_rmse_box.html')
                    fig_rmse.write_html(path)

                    path = os.path.join(dir_dst_rmse_box_log, f'{id_p}_{metric}_{dynamic}_{normal}_log_rmse_box.html')
                    fig_log_rmse.write_html(path)

                if write_svg:
                    path = os.path.join(dir_dst_error_box, f'{id_p}_{metric}_{dynamic}_{normal}_error_box.svg')
                    fig_err.write_image(path, scale=5)

                    path = os.path.join(dir_dst_rmse_box, f'{id_p}_{metric}_{dynamic}_{normal}_rmse_box.svg')
                    fig_rmse.write_image(path, scale=5)

                    path = os.path.join(dir_dst_rmse_box_log, f'{id_p}_{metric}_{dynamic}_{normal}_log_rmse_box.svg')
                    fig_log_rmse.write_image(path, scale=5)

                if write_png:
                    path = os.path.join(dir_dst_error_box, f'{id_p}_{metric}_{dynamic}_{normal}_error_box.png')
                    fig_err.write_image(path, scale=5)

                    path = os.path.join(dir_dst_rmse_box, f'{id_p}_{metric}_{dynamic}_{normal}_rmse_box.png')
                    fig_rmse.write_image(path, scale=5)

                    path = os.path.join(dir_dst_rmse_box_log, f'{id_p}_{metric}_{dynamic}_{normal}_log_rmse_box.png')
                    fig_log_rmse.write_image(path, scale=5)

        if verbose >= 1:
            prog.update(1)


def plot_timeseries_barplot_error_rmse(root_val):
    """
    Creates barplot for each Patient with the mean error and RMSE of the timeseries.



    :param root_val:
    :return:
    """

    csv_in = os.path.join(root_val, '04_statistics', '01_continuous', '01_results', 'omc_mmc_error.csv')
    if not os.path.isfile(csv_in):
        print(f'File {csv_in} does not exist.')
        return

    df_error = pd.read_csv(csv_in, sep=';')

    list_idx = list(df_error['id'].unique())

    for identifier in list_idx:

        if len(identifier.split('_')) != 2:
            continue

        id_s = identifier.split('_')[0]
        id_p = identifier.split('_')[1]

        df = df_error[df_error['id'] == identifier]

        list_dynamic = list(df['dynamic'].unique())
        list_normalized = list(df['normalized'].unique())

        identifier


def plot_timeseries_averaged(root_val, id_s, id_p, dynamic=False, fig_show=False,
                             write_html=False, write_svg=False, write_png=True,
                             verbose = 1):
    """
    Plots averaged timeseries of all trials for a given setting and participant.

    I uses the normalized-data.

    On the plot are:
        - One line representing average of all mmc-trials
        - shaded area representing std of all mmc-trials

        - one line representing average of all omc-trials
        - shaded area representing std of all omc-trials

    On the figure are two plots.
    1. All unaffected trials
    2. All affected trials


    :param root_val:
    :param id_s:
    :return:
    """
    # List of kinematics used for averaged plots
    kinematics = ['hand_vel', 'elbow_vel', 'trunk_disp', 'elbow_flex_pos', 'shoulder_flex_pos', 'shoulder_abduction_pos']

    dir_out = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '02_omc_to_mmc_kinematics',
                           '04_averaged_timeseries')

    if dynamic:
        dir_in = os.path.join(root_val, '03_data', 'preprocessed_data', '03_fully_preprocessed_dynamic', '01_normalized')
        dir_out = os.path.join(dir_out, '02_dynamic', f'{id_s}_{id_p}')
    else:
        dir_in = os.path.join(root_val, '03_data', 'preprocessed_data', '02_fully_preprocessed', '01_normalized')
        dir_out = os.path.join(dir_out, '01_fixed', f'{id_s}_{id_p}')

    dir_out_aff = os.path.join(dir_out, '01_affected')
    dir_out_unaff = os.path.join(dir_out, '02_unaffected')

    for dir in [dir_out,dir_out_aff, dir_out_unaff]:
        os.makedirs(dir, exist_ok=True)

    files = [file for file in glob.glob(os.path.join(dir_in, '*.csv')) if any(kin in file for kin in kinematics)]


    progbar = tqdm(files, desc='Plotting Averaged Timeseries', unit='File', disable=verbose<1)

    for file in files:

        progbar.set_description(f'Plotting Averaged Timeseries for {id_s}_{id_p} - {os.path.basename(file)}')

        df_in_full = pd.read_csv(file, sep=';')

        # if Unnamed: 0 is in the columns, drop it
        if 'Unnamed: 0' in df_in_full.columns:
            df_in_full.drop(columns='Unnamed: 0', inplace=True)

        # Search for the kinematic in filename
        kinematic = [kin for kin in kinematics if kin in file][0]
        kinematic_name = get_title_measure_name(kinematic)
        unit = get_unit(kinematic)

        df_in_red = df_in_full[(df_in_full['id_s'] == id_s) & (df_in_full['id_p'] == id_p)]

        df_aff = df_in_red[df_in_red['condition'] == 'affected']
        df_unaff = df_in_red[df_in_red['condition'] == 'unaffected']


        side_aff = df_aff['side'].values[0] if len(df_aff) > 0 else None

        side_unaff = df_unaff['side'].values[0] if len(df_unaff) > 0 else None

        normalized_time = np.linspace(0, 1, 300)
        #get unaffected data

        def make_figure(mmc, omc, time, side, affected, dir_out, kinematic):
            mean_mmc = np.mean(mmc, axis=0)
            std_mmc = np.std(mmc, axis=0)
            mean_omc = np.mean(omc, axis=0)
            std_omc = np.std(omc, axis=0)

            unit = get_unit(kinematic)
            kinematic_name = get_title_measure_name(kinematic)

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=time, y=mean_mmc, mode='lines', name='MMC',
                                     line=dict(color=f'rgba({rgba_mmc}, 1)')))
            fig.add_trace(
                go.Scatter(x=time, y=mean_mmc + std_mmc, mode='lines', name='MMC + std',
                           line=dict(color=f'rgba({rgba_mmc}, 0.4)'), showlegend=False))
            fig.add_trace(
                go.Scatter(x=time, y=mean_mmc - std_mmc, mode='lines', name='MMC - std',
                           fill='tonexty', fillcolor=f'rgba({rgba_mmc}, 0.3)',
                           line=dict(color=f'rgba({rgba_mmc}, 0.4)'), showlegend=False))

            fig.add_trace(go.Scatter(x=time, y=mean_omc, mode='lines', name='OMC',
                                     line=dict(color=f'rgba({rgba_omc}, 1)')))
            fig.add_trace(
                go.Scatter(x=time, y=mean_omc + std_omc, mode='lines', name='OMC + std',
                           line=dict(color=f'rgba({rgba_omc}, 0.4)'), showlegend=False))
            fig.add_trace(
                go.Scatter(x=time, y=mean_omc - std_omc, mode='lines', name='OMC - std',
                           fill='tonexty', fillcolor=f'rgba({rgba_omc}, 0.3)',
                           line=dict(color=f'rgba({rgba_omc}, 0.4)'), showlegend=False))

            id_s_name = get_setting_axis_name(id_s)
            fig.update_layout(title=dict(text=f'<b>Averaged Timeseries for {kinematic_name} <br>of {id_s_name} for {id_p} - {side} - {affected}<b>', font = dict(size=20),
                                         y=0.95, yanchor='top'),
                              xaxis_title=dict(text='Normalized Time', font = dict(size=16)),
                              yaxis_title=dict(text=f'{kinematic_name} [{unit}]', font = dict(size=16)))

            if fig_show:
                fig.show()

            if write_html:
                path = os.path.join(dir_out, f'{id_s}_{id_p}_{kinematic}_{side}_{affected}_averaged.html')
                fig.write_html(path)

            if write_svg:
                path = os.path.join(dir_out, f'{id_s}_{id_p}_{kinematic}_{side}_{affected}_averaged.svg')
                fig.write_image(path, scale=5)

            if write_png:
                path = os.path.join(dir_out, f'{id_s}_{id_p}_{kinematic}_{side}_{affected}_averaged.png')
                fig.write_image(path, scale=5)

            return fig


        def interpolate_trial(df, idx_t, normalized_time, data):
            df_interp = []
            from scipy.interpolate import interp1d
            for id_t in idx_t:
                df_t = df[df['id_t'] == id_t]
                if df_t.shape[0] ==0:
                    print(id_t, df_t.shape)
                    continue
                f = interp1d(df_t["time_normalized"], df_t[data], kind="linear", bounds_error=False, fill_value="extrapolate")
                df_interp.append(f(normalized_time))

            return df_interp

        interpolated_mmc_unaff = None
        interpolated_omc_unaff = None
        interpolated_mmc_aff = None
        interpolated_omc_aff = None
        fig_unaff = None
        fig_aff = None

        if side_unaff is not None:
            idx_t_unaff = df_unaff['id_t'].unique()
            interpolated_mmc_unaff = interpolate_trial(df_unaff, idx_t_unaff, normalized_time, 'mmc')
            interpolated_omc_unaff = interpolate_trial(df_unaff, idx_t_unaff, normalized_time, 'omc')

        if side_aff is not None:
            idx_t_aff = df_aff['id_t'].unique()
            interpolated_mmc_aff = interpolate_trial(df_aff, idx_t_aff, normalized_time, 'mmc')
            interpolated_omc_aff = interpolate_trial(df_aff, idx_t_aff, normalized_time, 'omc')



        if interpolated_mmc_unaff and interpolated_omc_unaff:
            fig_unaff = make_figure(interpolated_mmc_unaff, interpolated_omc_unaff, normalized_time,
                                    side_unaff, 'unaffected', dir_out_unaff, kinematic)

        if interpolated_mmc_aff and interpolated_omc_aff:
            fig_aff = make_figure(interpolated_mmc_aff, interpolated_omc_aff, normalized_time,
                                  side_aff, 'affected', dir_out_aff, kinematic)

        if fig_unaff and fig_aff:
            # get both plots in one figure side by side
            fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Unaffected - {side_unaff}', f'Affected - {side_aff}'),
                                shared_yaxes=True)

            for i in range(len(fig_unaff.data)):
                fig.add_trace(fig_unaff.data[i], row=1, col=1)
                fig_aff.data[i].showlegend = False
                fig.add_trace(fig_aff.data[i], row=1, col=2)

            id_s_name = get_setting_axis_name(id_s)
            fig.update_layout(title=dict(text= f'<b>Averaged Timeseries for {kinematic_name} <br>of {id_s_name} and {id_p}<b>', font = dict(size=20),
                                         y=0.95, yanchor='top'),
                              xaxis_title=dict(text= 'Normalized Time', font = dict(size=18)),
                              yaxis_title=dict(text= f'{kinematic_name} [{unit}]'), font = dict(size=18))

            fig.update_xaxes(title=dict(text='Normalized Time', font = dict(size=18)), row=1, col=1)
            fig.update_xaxes(title=dict(text='Normalized Time', font = dict(size=18)), row=1, col=2)



            if fig_show:
                fig.show()

            if write_html:
                path = os.path.join(dir_out, f'{id_s}_{id_p}_{kinematic}_averaged.html')
                fig.write_html(path)

            if write_svg:
                path = os.path.join(dir_out, f'{id_s}_{id_p}_{kinematic}_averaged.svg')
                fig.write_image(path, scale=5)

            if write_png:
                path = os.path.join(dir_out, f'{id_s}_{id_p}_{kinematic}_averaged.png')
                fig.write_image(path, scale=5)

        progbar.update(1)
    progbar.close()


def plot_timeseries_RMSE(id_s, dir_dst, dir_data, joint_data, side, id_p=None,  verbose=1):
    """
    plot RMSE over time for all trials of a setting.

    If id_p is given, it only look at the data of the given id_p.

    Create one plot per participant and setting per value of interest.

    If joint_data is True:
        - plot position and velocity of elbow_flex, pro_sup, shoulderflexion, shoulderabduction
    if joint_data is False:
        - plot velocity of hand, trunk_displacement
    """

    if joint_data:
        cols_of_interest = [f'elbow_flex_{side.lower()}', f'pro_sup_{side.lower()}', f'arm_flex_{side.lower()}', f'arm_add_{side.lower()}']
    else:



        pass


def plot_two_timeseries(df_omc, df_mmc, xaxis_label, yaxis_label, title, path_dst, showfig=False, verbose=1):
    """
    Plot Waveform of OMC with MMC of same recording for given id_s, id_p, id_t.

    df_omc and df_mmc are DataFrames containing columns: time, value.

    Create one plot per column of interest.
    """

    value = df_omc.columns[1]


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_omc['time'], y=df_omc[value], mode='lines', name='OMC'))
    fig.add_trace(go.Scatter(x=df_mmc['time'], y=df_mmc[value], mode='lines', name='MMC'))


    """if title is None:
        fig = px.line(df, x='time', y=['omc', 'mmc'])
    else:
        fig = px.line(df, x='time', y=['omc', 'mmc'], title=title)"""
    fig.update_layout(xaxis_title=xaxis_label,
                      yaxis_title=yaxis_label,
                      title=title)

    if showfig:
        fig.show()

    if not os.path.isdir(os.path.dirname(path_dst)):
        os.makedirs(os.path.dirname(path_dst), exist_ok=True)

    if path_dst is not None:
        fig.write_image(path_dst, scale=5)


def generate_plots_grouped_different_settings(dir_src, dir_dst, df_omc, id_p, id_t, condition, value, y_label,
                                              write_html=True, write_svg=True,
                                              showfig = False, verbose=1):
    """
    Iterates over .csv files of processed data and creates plots for all files.
    Only the given values will be plotted
    """

    # get all idx_s for given id_p and id_t
    idx_s = [os.path.basename(file).split('_')[0] for file in os.listdir(dir_src) if id_p in file and id_t in file and 'S15133' not in file]

    fig = go.Figure()

    #time = pd.to_timedelta(df_omc['time']).apply(lambda x: x.total_seconds())
    time = df_omc['time']
    # Add line for omc
    fig.add_trace(go.Scatter(x=time, y=df_omc[value], mode='lines', name='OMC',
                             line=dict(color='red',
                                       width=5)))

    colours = px.colors.qualitative.Plotly

    for i, id_s in enumerate(idx_s):

        files = glob.glob(os.path.join(dir_src, f'{id_s}_{id_p}_{id_t}*.csv'))
        if files:
            df_mmc = pd.read_csv(files[0], sep=';')
        else:
            continue

        #df_mmc = pd.read_csv(os.path.join(dir_src, f'{id_s}_{id_p}_{id_t}_preprocessed.csv'), sep=';')

        #time = pd.to_timedelta(df_mmc['time']).apply(lambda x: x.total_seconds())
        time = df_omc['time']
        # Add line for mmc setting
        fig.add_trace(go.Scatter(x=time, y=df_mmc[value],
                                 mode='lines', name=f'MMC {id_s}', opacity=0.75,
                                 line=dict(color=colours[i % len(colours)])))

    fig.update_layout(xaxis_title='Time [s]',
                      yaxis_title=y_label,
                      title= f'{y_label} by setting of {id_p}, {id_t}')

    if write_html:
        path = os.path.join(dir_dst, f'{id_p}_{id_t}_{condition}_{value}.html')
        fig.write_html(path)
    if write_svg:
        path = os.path.join(dir_dst, f'{id_p}_{id_t}_{condition}_{value}.svg')
        fig.write_image(path, scale=5)


def generate_plots_for_timeseries(dir_root_val, values=None, id_s=None, id_p_in=None, id_t_in=None,
                                  write_html=True, write_svg=True, gen_plots_diff_settings=True, dynamic=False,
                                  showfig = True, verbose=1):
    """
    Iterates over .csv files of processed data and creates plots for all files.

    if id_p or id_t are given, only the data of the given id_s, id_p or id_t are used.

    if values are given, only those will be plotted. Otherwise all timeseries will be plotted.

    Following Plots can be generated:

    1.  Plots with single S, single P, single T
    2.  Plots with multiple S, single P, single T

    All Plots are generated for all t_ids. In 2 the line of OMC is highlighted and all MMC lines are slightly transparent.

    P_id and t_id of MMC and OMC need to be the same.

    Plots of 1:    os.path.join(root_val, 04_statistics, '01_continuous', '02_plots', '02_OMC_to_mmc_kinematics', '01_single')
    Plots of 2:    os.path.join(dir_root_val, '04_statistics', '01_continuous', '02_plots', '02_OMC_to_mmc_kinematics', '03_grouped_by_s')

    :param dir_processed:

    :param write_svg:
    :param write_html:
    :param values:
    :param id_s:
    :param id_p:
    :param id_t:
    :param verbose:
    :return:
    """
    if dynamic:
        dir_plots = os.path.join(dir_root_val, '04_statistics', '01_continuous', '02_plots', '03_OMC_to_mmc_kinematics_dynamic')
        dir_processed = os.path.join(dir_root_val, '03_data', 'preprocessed_data', '03_fully_preprocessed_dynamic')
        dir_dst_single = os.path.join(dir_plots, '01_single')
        dir_dst_grouped_by_s = os.path.join(dir_plots, '02_grouped_by_s')
    else:
        dir_plots = os.path.join(dir_root_val, '04_statistics', '01_continuous', '02_plots', '02_OMC_to_mmc_kinematics')
        dir_processed = os.path.join(dir_root_val, '03_data', 'preprocessed_data', '02_fully_preprocessed')
        dir_dst_single = os.path.join(dir_plots, '01_single')
        dir_dst_grouped_by_s = os.path.join(dir_plots, '02_grouped_by_s')

    df_timestamps = pd.read_csv(os.path.join(dir_root_val, '04_statistics', '02_categorical', 'murphy_timestamps.csv'), sep=';')


    if id_p_in is not None and id_t_in is not None:
        dir_dst_single = os.path.join(dir_plots, f'03_specific', '01_single')
        dir_dst_grouped_by_s = os.path.join(dir_plots, f'03_specific', '02_grouped_by_s')

    for dir in [dir_dst_single, dir_dst_grouped_by_s]:
        os.makedirs(dir, exist_ok=True)

    if values is None:
        values = ["hand_vel",
                  "elbow_vel",
                  "trunk_disp",
                  "trunk_ang",
                  "elbow_flex_pos",
                  "shoulder_flex_pos",
                  "shoulder_abduction_pos"]

    elif type(values) is str:
        values = [values]

    multi_id_s = True
    if id_s is None:
        idx_s = sorted(list(set([os.path.basename(file).split('_')[0] for file in os.listdir(dir_processed)])))
        idx_s.remove('S15133')
    elif type(id_s) is str:
        idx_s = [id_s]
        multi_id_s = False
    else:
        idx_s = id_s


    id_s_omc = 'S15133'

    # get all omc_trials
    omc_csvs = glob.glob(os.path.join(dir_processed, 'S15133_P*_T*.csv'))
    # retrieve all p_ids and t_ids present in omc data.
    idx_p_omc = list(set([os.path.basename(omc_csv).split('_')[1] for omc_csv in omc_csvs]))

    progress = tqdm(total=len(values), desc='Creating Plots', disable=verbose < 1)

    for value in values:
        print('start value')

        # get y_axis label from value
        match value:
            case 'hand_vel':
                y_label = 'Hand Velocity [mm/s]'
            case 'elbow_vel':
                y_label = 'Elbow Velocity [deg/s]'
            case 'trunk_disp':
                y_label = 'Trunk Displacement [mm]'
            case 'trunk_ang':
                y_label = 'Trunk Angle [deg]'
            case 'elbow_flex_pos':
                y_label = 'Elbow Flexion [deg]'
            case 'shoulder_flex_pos':
                y_label = 'Shoulder Flexion [deg]'
            case 'shoulder_abduction_pos':
                y_label = 'Shoulder Abduction [deg]'
            case _:
                y_label = value


        for id_s in idx_s:
            print('start s')


            if id_p_in is None:
                idx_p_mmc = [os.path.basename(file).split('_')[1] for file in os.listdir(dir_processed) if id_s in file]
            elif type(id_p_in) is str:
                idx_p_mmc = [id_p_in]
            else:
                idx_p_mmc = id_p_in

            # get all p_ids present for mmc and omc data
            idx_p = sorted(list(set(idx_p_omc) & set(idx_p_mmc)))

            for id_p in idx_p:

                print('start p')
                idx_t_omc = [os.path.basename(omc_csv).split('_')[2].split('.')[0] for omc_csv in omc_csvs if id_p in os.path.basename(omc_csv)]

                if id_t_in is None:
                    idx_t_mmc = [os.path.basename(file).split('_')[2].split('.')[0] for file in os.listdir(dir_processed) if id_s in file and id_p in file]
                elif type(id_t_in) is str:
                    idx_t_mmc = [id_t_in]
                else:
                    idx_t_mmc = id_t_in

                idx_t = sorted(list(set(idx_t_omc) & set(idx_t_mmc)))

                for id_t in idx_t:
                    print('start t')
                    progress.set_description(f'Creating Plots for {value}_{id_s}_{id_p}_{id_t}')

                    condition = df_timestamps[(df_timestamps['id_p'] == id_p) & (df_timestamps['id_t'] == id_t)]['condition'].values[0]

                    # read data
                    omc_files = glob.glob(os.path.join(dir_processed, f'{id_s_omc}_{id_p}_{id_t}*.csv'))
                    mmc_files = glob.glob(os.path.join(dir_processed, f'{id_s}_{id_p}_{id_t}*.csv'))
                    print("load csv")

                    if omc_files and mmc_files:
                        omc_file = omc_files[0]
                        df_omc = pd.read_csv(omc_file, sep=';')

                        mmc_file = mmc_files[0]
                        df_mmc = pd.read_csv(mmc_file, sep=';')
                    else:
                        continue

                    # plot t_plot

                    #time = pd.to_timedelta(df_omc['time']).apply(lambda x: x.total_seconds())

                    time = df_omc['time']
                    df_t = pd.DataFrame({'Time': time, 'OMC': df_omc[value], 'MMC': df_mmc[value]})
                    fig = px.line(df_t, x='Time', y=['OMC', 'MMC'], title=f'{y_label} for {id_s}, {id_p}, {id_t}, {condition}')
                    fig.update_layout(xaxis_title='Time [s]', yaxis_title=y_label)

                    if showfig:
                        fig.show()
                    
                    if write_html:
                        path = os.path.join(dir_dst_single, f'{id_s}_{id_p}_{id_t}_{condition}_{value}.html')
                        fig.write_html(path)

                    if write_svg:
                        path = os.path.join(dir_dst_single, f'{id_s}_{id_p}_{id_t}_{condition}_{value}.svg')
                        fig.write_image(path, scale=5)

                    if gen_plots_diff_settings and multi_id_s:
                        progress.set_description(f'Creating Plots for {value}_{id_s}_{id_p}_{id_t} - Grouped by S')

                        os.makedirs(dir_dst_grouped_by_s, exist_ok=True)
                        generate_plots_grouped_different_settings(dir_processed, dir_dst_grouped_by_s,
                                                                  df_omc, id_p, id_t, condition,
                                                                  value, y_label,
                                                                  write_html=write_html, write_svg=write_svg,
                                                                  showfig=showfig, verbose=1)

        progress.update(1)


def write_plottable_identifier(dir_root_val, dir_src, to_plot, verbose = 1):
    """
    Iterate over id_s, id_p, id_t and find all identifier, that can be plotted.

    To be plotted, id_p and id_t need to be present for both OMC and MMC data.

    to plot is the kind of plot that can be plotted and is used as suffix for the generated .csv file.

        - murphy_measures
        - preprocessed_timeseries

    :param dir_src:
    :return:
    """
    columns = ['id_s', 'id_p', 'id_t', 'condition']

    path_csv = os.path.join(dir_root_val, '05_logs', f'plottable_trials_{to_plot}.csv')

    df_timestamps = pd.read_csv(os.path.join(dir_root_val, '04_statistics', '02_categorical', 'murphy_timestamps.csv'),
                                sep=';')

    df = pd.DataFrame(columns=columns)

    id_s_omc = 'S15133'

    idx_s = sorted(list(set([os.path.basename(file).split('_')[0] for file in os.listdir(dir_src)])))
    idx_s.remove(id_s_omc)

    # get all omc_trials
    omc_csvs = glob.glob(os.path.join(dir_src, 'S15133_P*_T*.csv'))
    # retrieve all p_ids and t_ids present in omc data.
    idx_p_omc = list(set([os.path.basename(omc_csv).split('_')[1] for omc_csv in omc_csvs]))

    idx_p_mmc = sorted(list(set([os.path.basename(file).split('_')[1] for file in os.listdir(dir_src)])))
    # get all p_ids present for mmc and omc data
    idx_p = sorted(list(set(idx_p_omc) & set(idx_p_mmc)))

    total = len(idx_p)

    progbar = tqdm(total=total, desc='Searching plottable Trials', disable=verbose < 1)

    set_sp_tuples = set()

    for id_p in idx_p:
        if id_p in ignore_id_p:
            continue

        idx_t_omc = [os.path.basename(omc_csv).split('_')[2].split('.')[0] for omc_csv in omc_csvs
                     if id_p in os.path.basename(omc_csv)]
        idx_t_mmc = [os.path.basename(file).split('_')[2].split('.')[0] for file in os.listdir(dir_src) if id_p in file]
        idx_t = sorted(list(set(idx_t_omc) & set(idx_t_mmc)))

        total += len(idx_t)

        progbar.total = total
        progbar.refresh()

        df_temp=None
        for id_t in idx_t:
            if df_temp is None:
                df_temp = pd.DataFrame(columns=columns)
            progbar.set_description(f'Checking {id_p}_{id_t}')
            idx_s = [os.path.basename(file).split('_')[0] for file in os.listdir(dir_src) if id_p in file and id_t in file and 'S15133' not in file]
            condition = df_timestamps[(df_timestamps['id_p'] == id_p) & (df_timestamps['id_t'] == id_t)]['condition'].values[0]
            side = df_timestamps[(df_timestamps['id_p'] == id_p) & (df_timestamps['id_t'] == id_t)]['side'].values[0]
            # read data
            idx_s = [id_s for id_s in idx_s if id_s in idx_s_reduced]

            for id_s in idx_s:
                omc_files = glob.glob(os.path.join(dir_src, f'{id_s_omc}_{id_p}_{id_t}*.csv'))
                mmc_files = glob.glob(os.path.join(dir_src, f'{id_s}_{id_p}_{id_t}*.csv'))

                if omc_files and mmc_files:
                    set_sp_tuples.add((id_s, id_p))
                    df_new = pd.DataFrame({'id_s': id_s, 'id_p': id_p, 'id_t': id_t, 'condition': condition, 'side': side, 'to_plot': to_plot}, index = [0])
                    df_temp = pd.concat([df_temp, df_new], ignore_index=True)
                else:
                    continue

            progbar.update(1)

        if df_temp is not None:
            df = pd.concat([df, df_temp], ignore_index=True)

    df.to_csv(path_csv, sep=';')

    return path_csv, set_sp_tuples


def get_plottable_timeseries_kinematics(plottable_csv, min_n_ids, affected=None, verbose=1):
    """
    Read the csv file with the plottable trials and return the id_p and id_t that are used in the most settings.

    if n_s_ids is given, max(#sic) - n s_ids are accepted.
    e.g. if a trial has 5 settings and n_s_ids = 3, trials with down to 2 settings are accepted.

    If affected is given, only the affected or unaffected trials are returned.

    :param plottable_csv:
    :param n_s_ids:
    :return: dict containing the id_ps and id_ts in lists. keys are 'id_p' and 'id_t'
    """
    df = pd.read_csv(plottable_csv, sep=';')

    df_out = pd.DataFrame(columns=['id_p', 'id_t', 'n_s_ids'])


    if affected == 'affected':
        df = df[df['condition'] == 'affected']
    elif affected == 'unaffected':
        df = df[df['condition'] == 'unaffected']

    # iterate over id_p and id_t in df and count the number of settings
    for id_p in df['id_p'].unique():
        for id_t in df['id_t'].unique():
            n_s_ids = len(df[(df['id_p'] == id_p) & (df['id_t'] == id_t)])
            df_new = pd.DataFrame({'id_p': [id_p], 'id_t': [id_t], 'n_s_ids': [n_s_ids]})
            df_out = pd.concat([df_out, df_new], ignore_index=True)

            # If max amount of settings is lower than min_n_ids, only the trial with the most settings is kept Even if it is lower than min_n_ids
            max_n_s_ids = df_out['n_s_ids'].max()
            if max_n_s_ids <= min_n_ids:
                df_out = df_out[df_out['n_s_ids'] == max_n_s_ids]
            else:
                df_out = df_out[df_out['n_s_ids'] >= min_n_ids]

    return df_out
