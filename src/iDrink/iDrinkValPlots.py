import numpy as np
import pandas
import pandas as pd

import os
import ast
import glob

from tqdm import tqdm

import plotly as py
import plotly.express as px
import plotly.graph_objects as go

import statsmodels.api as sm


def plot_murphy_blandaltman(df_murphy, measured_value, id_s, id_p=None, plot_to_val=False, filename=None, filetype='.png', use_smoother=True,
                            colourcode = None, show_id_t=False, verbose=1, show_plots=True, customize_layout=False):
    """
    create bland altman plot.

    OMC Setting is Reference for Plots.

    if filename is given, the plot is saved to the given filename.

    If id_s is given, use only the data of the given id_s.

    If id_p is given, use only the data of the given id_p.
    The individual patients/participants are colour-coded in the plot

    The trial number is extracted from id_t and added to each datapoint in the plot.

    df_murphy['condition'] is shape coded for unaffected and affected

    :param dat_ref:
    :param dat_measured:
    :param show_plot:
    :return:
    """

    plot_id_p = False
    if id_p is None:
        idx_p = df_murphy[df_murphy['id_s'] == id_s]['id_p'].unique()
        if colourcode is None:
            colourcode = 'id_p'
    else:
        idx_p = [id_p]
        if colourcode is None:
            colourcode = 'condition'

        plot_id_p = True

    # Create DataFrame with all data of reference an measured data for all participants and trials that are in both datasets
    dat_ref_all = pd.DataFrame(columns=df_murphy.columns)
    dat_measured_all = pd.DataFrame(columns=df_murphy.columns)

    for id_p in idx_p:
        dat_ref = df_murphy[(df_murphy['id_s'] == 'S15133') & (df_murphy['id_p'] == id_p)].sort_values(by='id_t')
        dat_measured = df_murphy[(df_murphy['id_s'] == id_s) & (df_murphy['id_p'] == id_p)].sort_values(by='id_t')

        # Delete all Trials that are not in both datasets
        dat_ref = dat_ref[dat_ref['id_t'].isin(dat_measured['id_t'])]
        dat_measured = dat_measured[dat_measured['id_t'].isin(dat_ref['id_t'])]

        dat_ref_all = pd.concat([dat_ref_all, dat_ref])
        dat_measured_all = pd.concat([dat_measured_all, dat_measured])

    data = {
        'id_s': dat_measured_all['id_s'].values,
        'id_p': dat_measured_all['id_p'].values,
        'id_t': dat_measured_all['id_t'].values,
        'condition': dat_measured_all['condition'].values,
        'mmc': dat_measured_all[measured_value].values,
        'omc': dat_ref_all[measured_value].values,
        'difference': dat_measured_all[measured_value].values - dat_ref_all[measured_value].values,
        'mean': np.mean([dat_ref_all[measured_value].values, dat_measured_all[measured_value].values], axis=0)
    }
    df = pandas.DataFrame(data)
    idx_p = df['id_p'].unique()
    #create List with al id_t (index is the same as in the list of values)
    idx_t_all = df['id_t'].to_list()

    mean_omc = np.mean(df['omc'].values)
    mean_mmc = np.mean(df['mmc'].values)
    mean_all = np.mean([mean_omc, mean_mmc], axis=0)


    std_diff = np.std(df['difference'])
    sd = 1.96
    upper_limit = + sd * std_diff
    lower_limit = - sd * std_diff

    # creating plot

    if use_smoother:
        trendline = 'lowess'
    else:
        trendline = 'ewm'

    if show_id_t:
        text_annotation = 'id_t'
    else:
        text_annotation = None

    if plot_id_p:
        subject = id_p
    else:
        subject = id_s

    if plot_to_val:
        plot_type = 'Residuals vs. MMC'
        x_val = 'mmc'
    else:
        plot_type = 'Bland-Altman'
        x_val = 'mean'


    fig = px.scatter(df, x=x_val, y='difference', color=colourcode, symbol='condition',
                     hover_name='id_t', text=text_annotation, title=f'{plot_type} for {measured_value} of {subject}', trendline='lowess')
    # limits of agreement
    fig.add_trace(go.Scatter(x=[min(df['mean']), max(df['mean'])], y=[upper_limit, upper_limit], mode='lines', name=f'Upper Limit ({sd} SD)', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=[min(df['mean']), max(df['mean'])], y=[lower_limit, lower_limit], mode='lines', name=f'Lower Limit ({sd} SD)', line=dict(dash='dash')))

    # Add horicontal line at 0
    fig.add_trace(go.Scatter(x=[min(df['mean']), max(df['mean'])], y=[0, 0], mode='lines', name='Zero', line=dict(color='grey', dash='dash')))

    range_lim = max(abs(min(df['difference'])), abs(max(df['difference']))) * 1.5
    y_range = [-range_lim, range_lim]

    if customize_layout:
        # update the layout
        if plot_to_val:
            if id_p is not None:
                title = f'Residuals vs. MMC Plot for {measured_value} of {id_p}'
            else:
                title = f'Residuals vs. MMC  Plot for {measured_value} of {id_s}'

        else:
            if id_p is not None:
                title = f'Bland-Altman Plot for {measured_value} of {id_p}'
            else:
                title = f'Bland-Altman Plot for {measured_value} of {id_s}'




        fig.update_layout(title=title,
                          xaxis_title=f'mean of {measured_value}',
                          yaxis_title=f'Difference of MMC from OMC',
                          yaxis=dict(range=y_range),
                          legend=dict(
                              orientation="h",
                              x=0,
                              y=-0.2  # Positionierung unterhalb der x-Achse
                          )
                          )


    if verbose>=1 and show_plots:
        fig.show()

    if type(filetype) is str:
        filetype = [filetype]

    if filename is not None:
        for extension in filetype:
            path = filename + extension
            match extension:
                case '.html':
                    py.offline.plot(fig, filename=path, auto_open=False)
                case '.png':
                    fig.write_image(path, scale=5)
                case '.jpg':
                    fig.write_image(path, scale=5)
                case '.jpeg':
                    fig.write_image(path, scale=5)
                case _:
                    print(f'Filetype {extension} not supported. Please use .html, .png, .jpg or .jpeg')


def plot_timeseries_blandaltman_scale_location(root_val, kinematic, idx_p=None, idx_t=None, dynamic='dynamic', write_html=True, write_png=True, show_plots=True):
    """
    Prints all Bland altman and scale location plots and saves them into the correct folder.

    Normalized errors are used for the plots.

    #TODO: Implement settingwise if enough time is left.

    :param root_val:
    :return:
    """



    dir_src = os.path.join(root_val, '04_statistics', '01_continuous', '01_results', '01_ts_error')

    subdir_dyn = '02_dynamic' if dynamic else '01_fixed'
    dir_dst_bland = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '01_omc_to_mmc_error',
                                 '01_omc_mmc_error', '01_bland_altman', subdir_dyn)
    dir_dst_scale = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '01_omc_to_mmc_error',
                                 '01_omc_mmc_error', '02_scale_location', subdir_dyn)

    dir_dst_time = os.path.join(root_val, '04_statistics', '01_continuous', '02_plots', '01_omc_to_mmc_error',
                                    '01_omc_mmc_error', '03_time_error', subdir_dyn)

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


        for id_t in idx_t:
            files = [file for file in list_files_p if id_t in file]
            if files:
                file = files[0]
            else:
                continue


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
                fig_bland.add_trace(go.Scatter(x=lowess[:, 0], y=lowess[:, 1], mode='lines', name=f'{id_s}', line=dict(color='red')))

                # add line at 0
                fig_bland.add_trace(go.Scatter(x=[min(mean), max(mean)], y=[0, 0], mode='lines', name='Zero', line=dict(color='grey', dash='dash')))


                # add limits of agreement
                std_diff = np.std(error)
                sd = 1.96
                upper_limit = + sd * std_diff
                lower_limit = - sd * std_diff
                fig_bland.add_trace(go.Scatter(x=[min(mean), max(mean)], y=[upper_limit, upper_limit], mode='lines',
                                         name=f'Upper Limit ({sd} SD)', line=dict(dash='dash')))
                fig_bland.add_trace(go.Scatter(x=[min(mean), max(mean)], y=[lower_limit, lower_limit], mode='lines',
                                         name=f'Lower Limit ({sd} SD)', line=dict(dash='dash')))


                # Scale Location Plot
                fig_scale.add_trace(go.Scatter(x=mmc, y=error, mode='markers', name=f'{id_s}', text=df['id_s'], hoverinfo='text'))
                fig_scale.add_trace(go.Scatter(x=mmc, y=error, mode='lines', name=f'{id_s}', line=dict(color='red')))
                fig_scale.add_trace(go.Scatter(x=[min(mmc), max(mmc)], y=[0, 0], mode='lines', name='Zero',
                                               line=dict(color='grey', dash='dash')))
                fig_scale.add_trace(go.Scatter(x=[min(mmc), max(mmc)], y=[upper_limit, upper_limit], mode='lines',
                                            name=f'Upper Limit ({sd} SD)', line=dict(dash='dash')))
                fig_scale.add_trace(go.Scatter(x=[min(mmc), max(mmc)], y=[lower_limit, lower_limit], mode='lines',
                                            name=f'Lower Limit ({sd} SD)', line=dict(dash='dash')))

                # Error to time
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(x=time, y=error, mode='lines', name=f'{id_s}', line=dict(color='red')))
                fig_time.add_trace(go.Scatter(x=time, y=[0]*len(time), mode='lines', name='Zero', line=dict(color='grey', dash='dash')))
                fig_time.add_trace(go.Scatter(x=time, y=mmc, mode='lines', name='MMC', line=dict(color='blue', dash='dash')))
                fig_time.add_trace(go.Scatter(x=time, y=omc, mode='lines', name='OMC', line=dict(color='green', dash='dash')))


                match kinematic:
                    case 'hand_vel':
                        unit = 'mm/s'
                    case 'elbow_vel':
                        unit = 'deg/s'
                    case 'trunk_disp':
                        unit = 'mm'
                    case 'trunk_ang':
                        unit = 'deg'
                    case 'elbow_flex_pos':
                        unit = 'deg'
                    case 'shoulder_flex_pos':
                        unit = 'deg'
                    case 'shoulder_abduction_pos':
                        unit = 'deg'
                    case _:
                        unit = ''

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
                py.offline.plot(fig_bland, filename=path_bland, auto_open=False)

                path_scale = os.path.join(dir_dst_scale, f'{id_p}_{id_t}_{kinematic}.html')
                py.offline.plot(fig_scale, filename=path_scale, auto_open=False)

                path_time = os.path.join(dir_dst_time, f'{id_p}_{id_t}_{kinematic}_time.html')
                py.offline.plot(fig_time, filename=path_time, auto_open=False)

            if write_png:
                path_bland = os.path.join(dir_dst_bland, f'{id_p}_{id_t}_{kinematic}.png')
                fig_bland.write_image(path_bland, scale=5)

                path_scale = os.path.join(dir_dst_scale, f'{id_p}_{id_t}_{kinematic}.png')
                fig_scale.write_image(path_scale, scale=5)

                path_time = os.path.join(dir_dst_time, f'{id_p}_{id_t}_{kinematic}_time.png')
                fig_time.write_image(path_time, scale=5)


def plot_timeseries_boxplot_error_rmse(root_val, showfig=False, write_html=False, write_png=True, verbose=1):
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



        list_dynamic = list(df['dynamic'].unique())
        list_normalized = list(df['normalized'].unique())

        metrics = list(df['metric'].unique())

        for dynamic in list_dynamic:
            normal = 'original'

            for metric in metrics:

                df_to_plot = df[(df['dynamic'] == dynamic) & (df['normalized'] == normal) & (df['metric'] == metric)]

                match metric:
                    case 'hand_vel':
                        unit = 'mm/s'
                    case 'elbow_vel':
                        unit = 'deg/s'
                    case 'trunk_disp':
                        unit = 'mm'
                    case 'trunk_ang':
                        unit = 'deg'
                    case 'elbow_flex_pos':
                        unit = 'deg'
                    case 'shoulder_flex_pos':
                        unit = 'deg'
                    case 'shoulder_abduction_pos':
                        unit = 'deg'
                    case _:
                        unit = ''

                # Boxplot of the mean errors of the timeseries x: id_s y: mean

                fig_err = px.box(df_to_plot, x='id_s', y='mean', color = 'condition',
                             title=f'Mean Error for {metric} of {id_p} - {dynamic}',
                             labels={'mean': f'Mean Error [{unit}]', 'id_s': 'Setting ID'})

                fig_rmse = px.box(df_to_plot, x='id_s', y='rmse', color = 'condition',
                                      title=f'Log(RMSE) for {metric} of {id_p} - {dynamic}',
                                      labels={'rmse': f'RMSE [{unit}]', 'id_s': 'Setting ID'})

                fig_log_rmse = px.box(df_to_plot, x='id_s', y='rmse', color = 'condition', log_y=True,
                             title=f'Log(RMSE) for {metric} of {id_p} - {dynamic}',
                             labels={'rmse': f'Log(RMSE) [{unit}]', 'id_s': 'Setting ID'})



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



def plot_measured_vs_errors(dat_ref, dat_measured, measured_value, id_s, id_p=None, path=None, verbose=1, show_plots=True):
    """
    create bland altman plot.

    data1 is the reference data.

    if path is given, the plot is saved to the given path.

    :param dat_ref:
    :param dat_measured:
    :param show_plot:
    :return:
    """


    # calculate mean, difference, mean of differences, standard deviation of differences, upper and lower limits, smoother
    diff = dat_ref - dat_measured
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    sd = 1.96
    upper_limit = mean_diff + sd * std_diff
    lower_limit = mean_diff - sd * std_diff
    lowess = sm.nonparametric.lowess(diff, dat_measured, frac=0.6)

    # creating plot
    fig = go.Figure()

    # Add horicontal line at 0
    fig.add_trace(go.Scatter(x=[min(dat_measured), max(dat_measured)], y=[0, 0], mode='lines', name='Zero Difference',
                             line=dict(color='grey', dash='dash')))

    # Scatter-Plot of dat_measured against differences
    fig.add_trace(go.Scatter(x=dat_measured, y=diff, mode='markers', name='Differences'))

    # mean of differences
    fig.add_trace(go.Scatter(x=dat_measured, y=[mean_diff]*len(dat_measured), mode='lines', name='Mean of Differences'))

    # limits of agreement
    fig.add_trace(go.Scatter(x=dat_measured, y=[upper_limit]*len(dat_measured), mode='lines',
                             name=f'Upper Limit ({sd} SD)', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=dat_measured, y=[lower_limit]*len(dat_measured), mode='lines',
                             name=f'Lower Limit ({sd} SD)', line=dict(dash='dash')))

    # smoother
    fig.add_trace(
        go.Scatter(x=lowess[:, 0], y=lowess[:, 1], mode='lines', name='Smoother (LOWESS)', line=dict(color='red')))

    # update the layout
    if id_p is not None:
        title = f'Residuals vs. MMC Plot for {measured_value} of {id_p}'
    else:
        title = f'Residuals vs. MMC  Plot for {measured_value} of {id_s}'


    range_lim = max(abs(min(diff)), abs(max(diff))) * 1.5
    y_range = [-range_lim, range_lim]

    fig.update_layout(title=title,
                      xaxis_title=f'MMC-Value of {measured_value}',
                      yaxis_title=f'Difference of MMC and OMC',
                      yaxis=dict(range=y_range),
                      legend=dict(
                          orientation="h",
                          x=0,
                          y=-0.2  # Positionierung unterhalb der x-Achse
                      )
                      )


    if verbose>=1 and show_plots:
        fig.show()

    if path is not None:
        if path.endswith('.html'):
            py.offline.plot(fig, filename=path)
        elif path.endswith('.png') or path.endswith('.jpg') or path.endswith('.jpeg'):
            fig.write_image(path)


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


def calibration_boxplot(csv_calib_errors, dir_dst, verbose=1, show_fig=False):
    """
    Create a barplot of the calibration errors for each participant and camera setting.

    :param csv_calib_errors:
    :param dir_dst:
    :param verbose:
    :return:
    """
    df_calib_error = pd.read_csv(csv_calib_errors, sep=';')


    cam_settings = df_calib_error['cam_used'].unique()

    fig = px.box(df_calib_error, x='cam_used', y='error', color='cam_used',
                 title=f'Calibration Errors for camera setups',
                 hover_name='id_p')

    fig.update_layout(xaxis_title = 'Camera Setup',
                      yaxis_title = 'Reprojection Error')
    fig.update(layout_showlegend=False)

    if show_fig:
        fig.show()
    # save plot

    os.makedirs(dir_dst, exist_ok=True)
    path = os.path.join(dir_dst, f'CalibrationErrors.png')
    fig.write_image(path, scale=5)

def generate_plot_error_single_setting_all_patients_all_trials(dir_src, dir_dst, id_s, value, y_label,
                                      write_html=True, write_png=True,
                                      verbose=1):
    """
    Plots the error of the given value for all trials and patients of a single setting.
    """
    pass




def generate_plots_grouped_different_settings(dir_src, dir_dst, df_omc, id_p, id_t, condition, value, y_label,
                                              write_html=True, write_png=True,
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
    if write_png:
        path = os.path.join(dir_dst, f'{id_p}_{id_t}_{condition}_{value}.png')
        fig.write_image(path, scale=5)



def generate_plots_for_timeseries(dir_root_val, values=None, id_s=None, id_p_in=None, id_t_in=None,
                                  write_html=True, write_png=True, gen_plots_diff_settings=True, dynamic=False,
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

    :param write_png:
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


                    print('make plot 1')
                    # plot t_plot

                    #time = pd.to_timedelta(df_omc['time']).apply(lambda x: x.total_seconds())

                    time = df_omc['time']
                    df_t = pd.DataFrame({'Time': time, 'OMC': df_omc[value], 'MMC': df_mmc[value]})
                    fig = px.line(df_t, x='Time', y=['OMC', 'MMC'], title=f'{y_label} for {id_s}, {id_p}, {id_t}, {condition}')
                    fig.update_layout(xaxis_title='Time [s]', yaxis_title=y_label)

                    if showfig:
                        fig.show()
                    print('make plot 2')
                    if write_html:
                        path = os.path.join(dir_dst_single, f'{id_s}_{id_p}_{id_t}_{condition}_{value}.html')
                        fig.write_html(path)

                    if write_png:
                        path = os.path.join(dir_dst_single, f'{id_s}_{id_p}_{id_t}_{condition}_{value}.png')
                        fig.write_image(path, scale=5)

                    print('make plot 3')


                    print('make plot 4')
                    if gen_plots_diff_settings and multi_id_s:
                        progress.set_description(f'Creating Plots for {value}_{id_s}_{id_p}_{id_t} - Grouped by S')

                        os.makedirs(dir_dst_grouped_by_s, exist_ok=True)
                        generate_plots_grouped_different_settings(dir_processed, dir_dst_grouped_by_s,
                                                                  df_omc, id_p, id_t, condition,
                                                                  value, y_label,
                                                                  write_html=write_html, write_png=write_png,
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

    for id_p in idx_p:
        idx_t_omc = [os.path.basename(omc_csv).split('_')[2].split('.')[0] for omc_csv in omc_csvs
                     if id_p in os.path.basename(omc_csv)]
        idx_t_mmc = [os.path.basename(file).split('_')[2].split('.')[0] for file in os.listdir(dir_src) if id_p in file]
        idx_t = sorted(list(set(idx_t_omc) & set(idx_t_mmc)))

        total += len(idx_t)

        progbar.total = total
        progbar.refresh()

        for id_t in idx_t:
            idx_s = [os.path.basename(file).split('_')[0] for file in os.listdir(dir_src) if id_p in file and id_t in file and 'S15133' not in file]
            condition = df_timestamps[(df_timestamps['id_p'] == id_p) & (df_timestamps['id_t'] == id_t)]['condition'].values[0]
            side = df_timestamps[(df_timestamps['id_p'] == id_p) & (df_timestamps['id_t'] == id_t)]['side'].values[0]
            # read data
            for id_s in idx_s:
                omc_files = glob.glob(os.path.join(dir_src, f'{id_s_omc}_{id_p}_{id_t}*.csv'))
                mmc_files = glob.glob(os.path.join(dir_src, f'{id_s}_{id_p}_{id_t}*.csv'))

                if omc_files and mmc_files:
                    df_new = pd.DataFrame({'id_s': id_s, 'id_p': id_p, 'id_t': id_t, 'condition': condition, 'side': side, 'to_plot': to_plot}, index = [0])
                    df = pd.concat([df, df_new], ignore_index=True)
                else:
                    continue

            progbar.update(1)

    df.to_csv(path_csv, sep=';')

    return path_csv



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


if __name__ == "__main__":
    dir_data = r"I:\iDrink\validation_root\03_data"
    dir_dst = r"I:\iDrink\validation_root\04_statistics\01_continuous\02_plots"

    cols_of_interest = ['']

    id_s = 'S001'

    """Set Root Paths for Processing"""
    import iDrinkUtilities
    drive = iDrinkUtilities.get_drivepath()

    root_iDrink = os.path.join(drive, 'iDrink')
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


    plot_timeseries_boxplot_error_rmse(root_val, showfig=False, write_html=False, write_png=True, verbose=1)

    csv_plottable = write_plottable_identifier(root_val, dir_processed,
                                               to_plot='preprocessed_timeseries', verbose=1)

    df_plottable = get_plottable_timeseries_kinematics(csv_plottable, 2, affected='unaffected', verbose=1)

    kinematics =['hand_vel', 'elbow_vel', 'trunk_disp', 'trunk_ang', 'elbow_flex_pos', 'shoulder_flex_pos',
                 'shoulder_abduction_pos']

    kinematic = kinematics[4]
    # iterate over all plottable trials and create plots

    for i in range(len(df_plottable)):
        id_p = df_plottable['id_p'][i]
        id_t = df_plottable['id_t'][i]
        for kinematic in kinematics:
            plot_timeseries_blandaltman_scale_location(root_val, kinematic=kinematic, idx_p=id_p, idx_t=id_t, dynamic=dynamic_str, write_html=False, write_png=True, show_plots=False)
            pass

        generate_plots_for_timeseries(root_val, id_p_in = id_p, id_t_in = id_t, dynamic=dynamic,
                                      showfig = False, write_html=False, write_png=True)

    #plot_timeseries_RMSE(id_s, dir_dst, dir_data, joint_data=True, id_p=None,  verbose=1)
    #plot_measured_vs_errors(data1, data2, id_s='S000', measured_value='Test', path=path, show_plots=True)
