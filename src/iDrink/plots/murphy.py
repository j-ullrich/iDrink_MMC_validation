"""murphy part of iDrinkValPlots.py (moved without changes, see iDrinkValPlots.py for the public names)."""
import numpy as np
import pandas
import pandas as pd
import os
from plotly.io import write_html
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from scipy.stats import pearsonr, zscore
from iDrink.iDrinkUtilities import get_title_measure_name, get_unit, get_cad, get_setting_axis_name, get_measure_plot_num
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
ignore_id_p = ['P11', 'P19']
idx_s_singlecam = ['S017', 'S018']
idx_s_multicam = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010', 'S011', 'S012', 'S013', 'S014', 'S015', 'S016']
idx_s_multicam_reduced = ['S001', 'S002']
idx_s_reduced = idx_s_multicam_reduced + idx_s_singlecam
idx_s_full = idx_s_multicam + idx_s_singlecam


def plot_murphy_blandaltman(root_stat, write_html=False, write_svg=True, show_plots=False, write_png=True, verbose = 1):
    """Create Bland altman plot for Murphy measures.

    Plots are generated for:
        - {id_s}_{id_p}_blandaltman_{measure}
        - {id_s}_blandaltman_{measure}

        left side is unaffected, right side is affected
    """

    def make_DataFrame_for_plots(df_mmc, df_omc, measure):
        """Create the Dataframe that will be used for the figures
        columns are:
        - id_s
        - id_p
        - id_t
        - condition
        - side
        - mmc
        - omc
        - mean
        """
        df_mmc = df_mmc.sort_values(by='id_t')
        df_omc = df_omc.sort_values(by='id_t')



        data = {
            'id_s': df_mmc['id_s'].values,
            'id_p': df_mmc['id_p'].values,
            'id_t': df_mmc['id_t'].values,
            'condition': df_mmc['condition'].values,
            'side': df_mmc['side'].values,
            'mmc': df_mmc[measure].values,
            'omc': df_omc[measure].values,
            'mean': np.mean([df_mmc[measure].values, df_omc[measure].values], axis=0)
        }

        return pd.DataFrame(data)

    def make_figure(df, measure, affected, side, id_s, id_p, cad, unit):
        """
        Creates the Figure
        :param df:
        :param measure:
        :param id_s:
        :param id_p:
        :param cad:
        :return:
        """

        fig = go.Figure()
        suff = get_measure_plot_num(measure)

        if id_p is None:
            for i_p in df['id_p'].unique():
                df_p = df[df['id_p'] == i_p]
                text = [f'id_p: {string1}<br>id_t: {string2}'
                        for string1, string2 in zip(df_p['id_p'], df_p['id_t'])]
                fig.add_trace(go.Scatter(x=df_p['mean'], y=df_p['mmc'] - df_p['omc'], mode='markers', name=f'{i_p}',
                                         text=text, hoverinfo='text'))
        else:
            text = [f'id_p: {string1}<br>id_t: {string2}'
                    for string1, string2 in zip(df['id_p'], df['id_t'])]
            fig.add_trace(go.Scatter(x=df['mean'], y=df['mmc'] - df['omc'], mode='markers', name='Data',
                                     text=text, hoverinfo='text'))

        fig.add_hline(y=0, line_dash='dash', line_color='grey', name='Zero')

        # add smoother
        lowess = sm.nonparametric.lowess(df['mmc'] - df['omc'], df['mean'], frac=0.6)
        fig.add_trace(go.Scatter(x=lowess[:, 0], y=lowess[:, 1], mode='lines', name='Smoother', line=dict(color='black')))

        # add limits of agreement
        std_diff = np.std(df['mmc'] - df['omc'])
        mean_diff = np.mean(df['mmc'] - df['omc'])
        sd = 1.96
        upper_limit = mean_diff + sd * std_diff
        lower_limit = mean_diff - sd * std_diff

        fig.add_hline(y=upper_limit, line_dash='dash', line_color='orange', name=f'Upper Limit ({sd} SD)')
        fig.add_hline(y=lower_limit, line_dash='dash', line_color='orange', name=f'Lower Limit ({sd} SD)')

        # add horizontal line for cad
        if cad is not None:
            fig.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
            fig.add_hline(y=-cad, line_dash='dash', line_color='red', name='CAD')

        title_measure = get_title_measure_name(measure)
        ids_name = get_setting_axis_name(id_s)

        if id_p is None:
            title = f'Bland Altman Plot for {title_measure} of setting {ids_name} with a CAD of {cad} {unit}'
            dir_id = f'{id_s}'
            id_sp = f'{id_s}'
        else:
            title = f'Bland Altman Plot for {title_measure} of {id_p} and setting {ids_name}  with a CAD of {cad} {unit}'
            dir_id = f'{id_s}_{id_p}'
            id_sp = f'{id_s}_{id_p}'

        fig.update_layout(title=title,
                          xaxis_title=f'Mean of {title_measure} {unit}',
                          yaxis_title=f'Difference of MMC to OMC {unit}',
                          legend=dict(
                              orientation="h",
                              x=0,
                              y=-0.2  # Positionierung unterhalb der x-Achse
                          )
                          )

        d_aff = '01_affected' if affected == 'affected' else '02_unaffected'

        dir_dst = os.path.join(dir_out_bland, dir_id, d_aff)

        os.makedirs(dir_dst, exist_ok=True)

        if write_html:
            path = os.path.join(dir_dst, f'{id_sp}_{measure}_{side}_{affected}_blandaltman.html')
            fig.write_html(path)

        if write_svg:
            path = os.path.join(dir_dst, f'{id_sp}_{measure}_{side}_{affected}_blandaltman.svg')
            fig.write_image(path, scale=5)

        if show_plots:
            fig.show()

        return fig

    csv_murphy = os.path.join(root_stat, '02_categorical', 'murphy_measures.csv')
    df_murphy = pd.read_csv(csv_murphy, sep=';')

    csv_murph_diff = os.path.join(root_stat, '02_categorical', 'stat_murphy_diff.csv')
    df_murph_diff = pd.read_csv(csv_murph_diff, sep=';')

    csv_cad = os.path.join(root_stat, '02_categorical', 'clinically_acceptable_difference.csv')
    df_cad = pd.read_csv(csv_cad, sep=',')

    # TODO: Implement trials that need to be ignored.

    dir_out_bland = os.path.join(root_stat, '02_categorical', '02_plots', '01_bland_altman')

    for id_ignore in ignore_id_p:
        # Delete all rows with the given id_p
        df_murphy = df_murphy[df_murphy['id_p'] != id_ignore]

    id_s_omc = 'S15133'
    df_murphy_omc = df_murphy[df_murphy['id_s'] == id_s_omc]

    df_murphy_mmc = df_murphy[df_murphy['id_s'] != id_s_omc]

    cols_error = ['TotalMovementTime', 'PeakVelocity_mms', 'elbowVelocity', 'tTopeakV_s',
                  'tToFirstpeakV_s', 'tTopeakV_rel', 'tToFirstpeakV_rel', 'NumberMovementUnits',
                  'InterjointCoordination', 'trunkDisplacementMM', 'trunkDisplacementDEG', 'ShoulderFlexionReaching',
                  'ElbowExtension', 'shoulderAbduction', 'shoulderFlexionDrinking']

    df_murphy_mmc = df_murphy_mmc[(np.abs(zscore(df_murphy_mmc[cols_error]).fillna(0)) < 3).all(axis=1)].reindex()
    df_murphy_omc = df_murphy_omc[(np.abs(zscore(df_murphy_omc[cols_error])) < 3).all(axis=1)].reindex()

    idx_s_murph_mmc = sorted(df_murphy_mmc['id_s'].unique())

    # Use only id_s that are also in

    idx_s = [id_s for id_s in idx_s_murph_mmc if id_s in idx_s_full]



    # get total number of combinations of id_s and id_p for progbar
    total = 0
    for id_s in idx_s:
        total +=  df_murphy_mmc[df_murphy_mmc['id_s'] == id_s]['id_p'].nunique()

    total *= len(murphy_measures)

    progbar = tqdm(range(total), desc='Plotting Bland Altman', unit='Trial', disable=verbose<1)

    for measure in murphy_measures:
        suff = get_measure_plot_num(measure)
        for id_s in sorted(idx_s):
            df_murphy_mmc_s = df_murphy_mmc[df_murphy_mmc['id_s'] == id_s]

            idx_p = sorted(df_murphy_mmc_s['id_p'].unique())

            df_aff_s = None
            df_unaff_s = None

            id_s_name = get_setting_axis_name(id_s)


            for id_p in idx_p:

                progbar.set_description(f'Plotting Bland Altman for {measure} {id_s}_{id_p}')

                df_murphy_mmc_p = df_murphy_mmc_s[df_murphy_mmc_s['id_p'] == id_p]
                df_murphy_omc_p = df_murphy_omc[df_murphy_omc['id_p'] == id_p]

                idx_t_mmc = sorted(df_murphy_mmc_p['id_t'].unique())
                idx_t_omc = sorted(df_murphy_omc_p['id_t'].unique())


                idx_t = [id_t for id_t in idx_t_mmc if id_t in idx_t_omc]  # Get all id_ts that exist for omc and mmc

                df_mmc_t = df_murphy_mmc_p[df_murphy_mmc_p['id_t'].isin(idx_t)]
                df_omc_t = df_murphy_omc_p[df_murphy_omc_p['id_t'].isin(idx_t)]


                df_fig = make_DataFrame_for_plots(df_mmc_t, df_omc_t, measure)

                df_aff = df_fig[df_fig['condition'] == 'affected']
                df_unaff = df_fig[df_fig['condition'] == 'unaffected']
                fig_aff = None
                fig_unaff = None

                if id_p not in df_aff['id_p'].unique() or id_p not in df_unaff['id_p'].unique():
                    continue

                df_aff_s = df_aff if df_aff_s is None else pd.concat([df_aff_s, df_aff])
                df_unaff_s = df_unaff if df_unaff_s is None else pd.concat([df_unaff_s, df_unaff])

                cad = get_cad(df_cad, measure)


                side_aff = df_aff['side'].values[0] if len(df_aff) > 0 else None
                side_unaff = df_unaff['side'].values[0] if len(df_unaff) > 0 else None


                unit = get_unit(measure)
                unit = f'[{unit}]' if unit else ''

                if side_aff is not None:
                    fig_aff = make_figure(df_aff, measure, 'affected', side_aff, id_s, id_p, cad, unit)

                if side_unaff is not None:
                    fig_unaff = make_figure(df_unaff, measure, 'unaffected', side_unaff, id_s, id_p, cad, unit)

                if fig_aff is not None and fig_unaff is not None:
                    # get both plots in one figure side by side
                    fig = make_subplots(rows=1, cols=2,
                                        subplot_titles=(f'Unaffected - {side_unaff}', f'Affected - {side_aff}'),
                                        shared_yaxes=True)

                    title_measure = get_title_measure_name(measure)

                    for i in range(len(fig_unaff.data)):
                        if fig_unaff.data[i].mode == 'markers':
                            fig_unaff.data[i].name = f'error unaffected'
                            fig_aff.data[i].name = f'error affected'
                        elif fig_unaff.data[i].mode == 'lines':
                            fig_unaff.data[i].name = 'Smoother'
                            fig_aff.data[i].showlegend = False
                        fig.add_trace(fig_unaff.data[i], row=1, col=1)
                        fig.add_trace(fig_aff.data[i], row=1, col=2)


                    # add hline for cad in both plots
                    if cad is not None:
                        fig.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD', row=1, col=1)
                        fig.add_hline(y=-cad, line_dash='dash', line_color='red', name='CAD', row=1, col=1)

                        fig.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD', row=1, col=2)
                        fig.add_hline(y=-cad, line_dash='dash', line_color='red', name='CAD', row=1, col=2)

                    # add hlines for limits of agreement
                    std_unaff = np.std(df_unaff['mmc'] - df_unaff['omc'])
                    mean_diff_unaff = np.mean(df_unaff['mmc'] - df_unaff['omc'])
                    std_aff = np.std(df_aff['mmc'] - df_aff['omc'])
                    mean_diff_aff = np.mean(df_aff['mmc'] - df_aff['omc'])


                    sd = 1.96

                    upper_limit_unaff = mean_diff_unaff + sd * std_unaff
                    lower_limit_unaff = mean_diff_unaff - sd * std_unaff

                    upper_limit_aff = mean_diff_aff + sd * std_aff
                    lower_limit_aff = mean_diff_aff - sd * std_aff


                    fig.add_hline(y=upper_limit_unaff, line_dash='dash', line_color='orange', name=f'Upper Limit ({sd} SD)', row=1, col=1)
                    fig.add_hline(y=lower_limit_unaff, line_dash='dash', line_color='orange', name=f'Lower Limit ({sd} SD)', row=1, col=1)

                    fig.add_hline(y=upper_limit_aff, line_dash='dash', line_color='orange', name=f'Upper Limit ({sd} SD)', row=1, col=2)
                    fig.add_hline(y=lower_limit_aff, line_dash='dash', line_color='orange', name=f'Lower Limit ({sd} SD)', row=1, col=2)

                    # add limit of agreement to legend
                    mean_peak = np.mean([df_unaff['mmc'].values, df_unaff['omc'].values])
                    fig.add_trace(go.Scatter(x=[mean_peak], y=[0], mode='lines', name='Limits of Agreement', line=dict(color='orange', dash='dash')))
                    # add cad to legend
                    fig.add_trace(go.Scatter(x=[mean_peak], y=[0], mode='lines', name='CAD', line=dict(color='red', dash='dash')))

                    fig.update_xaxes(title=dict(text=f'<b>Mean of {title_measure} {unit}<b>', font=dict(size=16)), row=1, col=1)
                    fig.update_xaxes(title=dict(text=f'<b>Mean of {title_measure} {unit}<b>', font=dict(size=16)), row=1, col=2)
                    fig.update_yaxes(title=dict(text=f'<b>Difference of MMC to OMC {unit}<b>', font=dict(size=16)), row=1, col=1)

                    fig.update_layout(title=dict(
                        text=f'<b>Averaged Timeseries for {title_measure} of setting {id_s_name} for {id_p}<b>',
                        font=dict(size=20)),
                        width=1200, height=600)

                    if show_plots:
                        fig.show()

                    dir_out = os.path.join(dir_out_bland, f'{id_s}_{id_p}')
                    os.makedirs(dir_out, exist_ok=True)

                    if write_html:
                        path = os.path.join(dir_out, f'{id_s}_{id_p}_{measure}_blandaltman.html')
                        fig.write_html(path)
                    if write_svg:
                        path = os.path.join(dir_out, f'{id_s}_{id_p}_{measure}_blandaltman.svg')
                        fig.write_image(path, scale=5)
                    if write_png:
                        path = os.path.join(dir_out, f'{suff}{id_s}_{id_p}_{measure}_blandaltman.png')
                        fig.write_image(path, scale=5)
                progbar.update(1)

            progbar.set_description(f'Plotting Bland Altman for {measure} {id_s}')

            fig_s_aff = None
            fig_s_unaff = None

            if df_aff_s is not None:
                fig_s_aff = make_figure(df_aff_s, measure, 'affected', 'mean', id_s, None, cad, unit)

            if df_unaff_s is not None:
                fig_s_unaff = make_figure(df_unaff_s, measure, 'unaffected', 'mean', id_s, None, cad, unit)

            if fig_s_aff is not None and fig_s_unaff is not None:
                # get both plots in one figure side by side
                fig = make_subplots(rows=1, cols=2,
                                    subplot_titles=(f'Unaffected - {side_unaff}', f'Affected - {side_aff}'),
                                    shared_yaxes=True)

                title_measure = get_title_measure_name(measure)

                for i in range(len(fig_s_unaff.data)):
                    if fig_s_unaff.data[i].mode == 'markers':
                        p = fig_s_aff.data[i].name
                        fig_s_unaff.data[i].name = f'{p} error unaffected'
                    elif fig_s_unaff.data[i].mode == 'lines':
                        fig_s_unaff.data[i].name = 'Smoother'

                    fig.add_trace(fig_s_unaff.data[i], row=1, col=1)

                    if i < len(fig_s_aff.data):
                        if fig_s_aff.data[i].mode == 'markers':
                            p = fig_s_aff.data[i].name
                            fig_s_aff.data[i].name = f'{p} error affected'
                        elif fig_s_aff.data[i].mode == 'lines':
                            fig_s_aff.data[i].showlegend = False

                        fig.add_trace(fig_s_aff.data[i], row=1, col=2)

                # add hline for cad in both plots
                if cad is not None:
                    fig.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD', row=1, col=1)
                    fig.add_hline(y=-cad, line_dash='dash', line_color='red', name='CAD', row=1, col=1)

                    fig.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD', row=1, col=2)
                    fig.add_hline(y=-cad, line_dash='dash', line_color='red', name='CAD', row=1, col=2)

                # add hlines for limits of agreement
                std_unaff = np.std(df_unaff_s['mmc'] - df_unaff_s['omc'])
                std_aff = np.std(df_aff_s['mmc'] - df_aff_s['omc'])

                mean_unaff = np.mean(df_unaff_s['mmc'] - df_unaff_s['omc'])
                mean_aff = np.mean(df_aff_s['mmc'] - df_aff_s['omc'])

                sd = 1.96

                upper_limit_unaff = mean_unaff + sd * std_unaff
                lower_limit_unaff = mean_unaff - sd * std_unaff

                upper_limit_aff = mean_aff + sd * std_aff
                lower_limit_aff = mean_aff - sd * std_aff

                fig.add_hline(y=upper_limit_unaff, line_dash='dash', line_color='orange', name=f'Upper Limit ({sd} SD)',
                              row=1, col=1)
                fig.add_hline(y=lower_limit_unaff, line_dash='dash', line_color='orange', name=f'Lower Limit ({sd} SD)',
                              row=1, col=1)

                fig.add_hline(y=upper_limit_aff, line_dash='dash', line_color='orange', name=f'Upper Limit ({sd} SD)',
                              row=1, col=2)
                fig.add_hline(y=lower_limit_aff, line_dash='dash', line_color='orange', name=f'Lower Limit ({sd} SD)',
                              row=1, col=2)

                # add limit of agreement to legend
                mean_peak = np.mean([df_unaff_s['mmc'].values, df_unaff_s['omc'].values])
                fig.add_trace(go.Scatter(x=[mean_peak], y=[0], mode='lines', name='Limits of Agreement',
                                         line=dict(color='orange', dash='dash')))
                # add cad to legend
                fig.add_trace(
                    go.Scatter(x=[mean_peak], y=[0], mode='lines', name='CAD', line=dict(color='red', dash='dash')))

                fig.update_xaxes(title=dict(text=f'<b>Mean of {title_measure} {unit}<b>', font=dict(size=16)), row=1, col=1)
                fig.update_xaxes(title=dict(text=f'<b>Mean of {title_measure} {unit}<b>',font=dict(size=16)), row=1, col=2)
                fig.update_yaxes(title=dict(text=f'<b>Difference of MMC from OMC {unit}<b>', font=dict(size=16)), row=1, col=1)

                fig.update_layout(title=dict(text=f'<b>Bland Altman for {title_measure} of {id_s_name} <br>CAD = {cad} {unit}<b>', font=dict(size=20)),
                                  width=1200, height=600)

                if show_plots:
                    fig.show()

                dir_out = os.path.join(dir_out_bland, f'{id_s}')
                os.makedirs(dir_out, exist_ok=True)

                if write_html:
                    path = os.path.join(dir_out, f'{id_s}_{measure}_blandaltman.html')
                    fig.write_html(
                        path)
                if write_svg:
                    path = os.path.join(dir_out, f'{id_s}_{measure}_blandaltman.svg')
                    fig.write_image(path, scale=5)

                if write_png:
                    path = os.path.join(dir_out, f'{suff}{id_s}_{measure}_blandaltman.png')
                    fig.write_image(path, scale=5)
                    pass

                if measure == 'PeakVelocity_mms':
                    fig.update_yaxes(range=[-400, 400])
                    fig.update_xaxes(range=[-50, 1600])
                    path = os.path.join(dir_out, f'{suff}{id_s}_{measure}_blandaltman_zoomed.png')
                    fig.write_image(path, scale=5)

    progbar.close()


def plot_murphy_blandaltman_old(df_murphy, measured_value, id_s, id_p=None, plot_to_val=False, filename=None, filetype='.svg', use_smoother=True,
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
                    fig.write_html(path)
                case '.svg':
                    fig.write_image(path, scale=5)
                case '.jpg':
                    fig.write_image(path, scale=5)
                case '.jpeg':
                    fig.write_image(path, scale=5)
                case _:
                    print(f'Filetype {extension} not supported. Please use .html, .svg, .jpg or .jpeg')


def plot_murphy_error_rmse_box_bar_plot(dir_root, outlier_corrected = False, cap_yaxis=True, plot_patient=False, showfig=False, write_html=False, write_svg=True, write_png=True, verbose=1):
    """
    PLots boxplots for errors and rmse of Murphy measures.

    One plot per Measure and Patient.


    y-axis: error or rmse
    x-axis: id_s
    color: condition


    Figure_filenames have pattern: {id_p}_{kinematic}_{dynamic}_box_error.svg

    :return:
    """

    dir_in = os.path.join(dir_root, '04_statistics', '02_categorical')
    dir_out_box = os.path.join(dir_root, '04_statistics', '02_categorical', '02_plots', '02_box')
    dir_out_bar = os.path.join(dir_root, '04_statistics', '02_categorical', '02_plots', '03_bar')

    dir_out_box_error = os.path.join(dir_out_box, f'01_error')
    dir_out_box_rmse = os.path.join(dir_out_box, f'02_rmse')

    dir_out_bar_error = os.path.join(dir_out_bar, f'01_error')
    dir_out_bar_rmse = os.path.join(dir_out_bar, f'02_rmse')

    for dir in [dir_out_box_error, dir_out_box_rmse, dir_out_bar_error, dir_out_bar_rmse]:
        os.makedirs(dir, exist_ok=True)

    if outlier_corrected:
        csv_rmse = os.path.join(dir_in, 'stat_murphy_rmse_outlier_corrected.csv')
        csv_error = os.path.join(dir_in, 'stat_murphy_diff_outlier_corrected.csv')
        outlier_str = '_outlier_corrected'
    else:
        csv_rmse = os.path.join(dir_in, 'stat_murphy_rmse.csv')
        csv_error = os.path.join(dir_in, 'stat_murphy_diff.csv')
        outlier_str = ''
    csv_cad = os.path.join(dir_in, 'clinically_acceptable_difference.csv')

    def files_exist(files):
        for file in files:
            if not os.path.isfile(file):
                print(f'File {file} does not exist.')
                return False
        return True

    if not files_exist([csv_rmse, csv_error, csv_cad, os.path.join(dir_in, 'murphy_measures.csv')]):
        return

    df_rmse = pd.read_csv(csv_rmse, sep=';')
    df_error = pd.read_csv(csv_error, sep=';')
    df_murphy_measures = pd.read_csv(os.path.join(dir_in, 'murphy_measures.csv'), sep=';')
    df_cad = pd.read_csv(csv_cad, sep=',')

    for df in [df_rmse, df_error, df_cad]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns='Unnamed: 0', inplace=True)

    df_rmse['id_s_name'] = df_rmse['id_s'].apply(lambda x: get_setting_axis_name(x))
    df_rmse = df_rmse[~df_rmse['id_p'].isin(ignore_id_p)]
    df_rmse = df_rmse[df_rmse['id_s'].isin(idx_s_reduced)]

    df_error['id_s_name'] = df_error['id_s'].apply(lambda x: get_setting_axis_name(x))
    df_error = df_error[~df_error['id_p'].isin(ignore_id_p)]
    df_error = df_error[df_error['id_s'].isin(idx_s_reduced)]

    df_rmse_nonan = df_rmse.dropna()
    df_rmse_mean = df_rmse[df_rmse['id_p'].isna()]

    progbar = tqdm(murphy_measures, desc='Plotting for ', unit='Measure', disable=verbose<1)

    color_map = {"affected": "blue", "unaffected": "red"}
    category_order = ["affected", "unaffected"]

    for measure in murphy_measures:
        progbar.set_description(f'Plotting for {measure}')

        cad = get_cad(df_cad, measure)
        unit = get_unit(measure)
        measure_name = get_title_measure_name(measure)
        suff = get_measure_plot_num(measure)

        # fig_rmse = px.box(df_rmse_nonan, x='id_s', y = measure, color='condition')

        fig_box_error = go.Figure()
        fig_bar = go.Figure()


        fig_box = px.box(df_rmse_nonan.sort_values(by='id_s'), x='id_s_name', y=measure, color='condition',
                         color_discrete_map=color_map, category_orders={"condition": category_order},
                         title=f'RMSE for {measure_name} with CAD of {cad} {unit}',)

        fig_bar = px.bar(df_rmse_mean.sort_values(by='id_s'), x='id_s_name', y=measure, color='condition',
                         color_discrete_map=color_map, category_orders={"condition": category_order},
                         title=f'Mean RMSE for {measure_name} with CAD of {cad} {unit}',)

        # add horizontal line for cad
        if cad is not None:
            fig_box.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
            fig_bar.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')


        fig_box.update_layout(title=dict(text=f'<b>RMSE for {measure_name} <br>CAD = {cad} {unit}<b>', font=dict(size=20)),
                              xaxis_title=dict(text='<b>Setting<b>', font=dict(size=16)),
                              yaxis_title=dict(text=f'<b>RMSE {unit}<b>', font=dict(size=16)))
        fig_bar.update_layout(title=dict(text=f'<b>mean RMSE for {measure_name} <br>CAD = {cad} {unit}<b>', font=dict(size=20)),
                              xaxis_title=dict(text='<b>Setting<b>', font=dict(size=16)),
                              yaxis_title=dict(text=f'<b>RMSE {unit}<b>', font=dict(size=16)))

        if cap_yaxis:
            if cad is not None:
                fig_box.update_yaxes(range=[0, min(max(df_rmse_nonan[measure].max(), cad+(cad*0.2)), 5000)])
                fig_bar.update_yaxes(range=[0, min(max(df_rmse_mean[measure].max(), cad+(cad*0.2)), 5000)])
            else:
                fig_box.update_yaxes(range=[0, min(df_rmse_nonan[measure].max(), 5000)])
                fig_bar.update_yaxes(range=[0, min(df_rmse_mean[measure].max(), 5000)])

        if showfig:
            fig_box.show()
            fig_bar.show()

        if write_html:
            path = os.path.join(dir_out_box_rmse, f'murphy_box_{measure}_rmse{outlier_str}.html')
            fig_box.write_html(path)
            path = os.path.join(dir_out_bar_rmse, f'murphy_bar_{measure}_rmse{outlier_str}.html')
            fig_bar.write_html(path)

        if write_svg:
            path = os.path.join(dir_out_box_rmse, f'murphy_box_{measure}_rmse{outlier_str}.svg')
            fig_box.write_image(path, scale=5)
            path = os.path.join(dir_out_bar_rmse, f'murphy_bar_{measure}_rmse{outlier_str}.svg')
            fig_bar.write_image(path, scale=5)

        if write_png:
            path = os.path.join(dir_out_box_rmse, f'{suff}murphy_box_{measure}_rmse{outlier_str}.png')
            fig_box.write_image(path, scale=5)
            path = os.path.join(dir_out_bar_rmse, f'{suff}murphy_bar_{measure}_rmse{outlier_str}.png')
            fig_bar.write_image(path, scale=5)

        if plot_patient:

            for id_p in df_rmse_nonan['id_p'].unique():

                progbar.set_description(f'Plotting Boxplots for {measure} of {id_p}')
                fig_box = go.Figure()
                fig_bar = go.Figure()
                for condition, group in df_rmse_nonan[df_rmse_nonan['id_p'] == id_p].groupby('condition'):
                    offset  = 1 if condition == 'affected' else 0
                    fig_box.add_trace(go.Box(x=group['id_s'], y=group[measure], name=condition, offsetgroup=offset))

                for condition, group in df_rmse_nonan[df_rmse_nonan['id_p'] == id_p].groupby('condition'):
                    fig_bar.add_trace(go.Bar(x=group['id_s'], y=group[measure], name=condition))

                fig_box_error = px.box(df_error[df_error['id_p'] == id_p], x='id_s_name', y=measure, color='condition',
                                       color_discrete_map={'affected': 'blue', 'unaffected': 'red'},
                                       title=f'Error for {measure_name} of {id_p} with CAD of {cad}',
                                       labels={'condition': 'Condition', 'id_s': 'Setting', 'value': 'Error'})

                # add horicontal line for cad
                if cad is not None:
                    fig_box.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
                    fig_bar.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
                    fig_box_error.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
                    fig_box_error.add_hline(y=-cad, line_dash='dash', line_color='red', name='CAD')

                fig_box.update_layout(title=f'RMSE for {measure_name} of {id_p} with CAD of {cad}',
                                      xaxis_title='Setting',
                                      yaxis_title='RMSE')


                fig_bar.update_layout(title=f'RMSE for {measure_name} of {id_p} with CAD of {cad}', xaxis_title='Setting', yaxis_title='RMSE')
                fig_box_error.update_layout(title=f'Error for {measure_name} of {id_p} with CAD of {cad}', xaxis_title='Setting', yaxis_title='Error')


                if showfig:
                    fig_box.show()
                    fig_bar.show()
                    fig_box_error.show()

                if write_html:
                    path = os.path.join(dir_out_box_rmse, f'{id_p}_murphy_box_{measure}_rmse{outlier_str}.html')
                    fig_box.write_html(path)
                    path = os.path.join(dir_out_bar_rmse, f'{id_p}_murphy_bar_{measure}_rmse{outlier_str}.html')
                    fig_bar.write_html(path)
                    path = os.path.join(dir_out_box_error, f'{id_p}_murphy_box_{measure}_error{outlier_str}.html')
                    fig_box_error.write_html(path)

                if write_svg:
                    path = os.path.join(dir_out_box_rmse, f'{id_p}_murphy_box_{measure}_rmse{outlier_str}.svg')
                    fig_box.write_image(path, scale=5)
                    path = os.path.join(dir_out_bar_rmse, f'{id_p}_murphy_bar_{measure}_rmse{outlier_str}.svg')
                    fig_bar.write_image(path, scale=5)
                    path = os.path.join(dir_out_box_error, f'{id_p}_murphy_box_{measure}_error{outlier_str}.svg')
                    fig_box_error.write_image(path, scale=5)

                if write_png:
                    path = os.path.join(dir_out_box_rmse, f'{suff}{id_p}_murphy_box_{measure}_rmse{outlier_str}.png')
                    fig_box.write_image(path, scale=5)
                    path = os.path.join(dir_out_bar_rmse, f'{suff}{id_p}_murphy_bar_{measure}_rmse{outlier_str}.png')
                    fig_bar.write_image(path, scale=5)
                    path = os.path.join(dir_out_box_error, f'{suff}{id_p}_murphy_box_{measure}_error{outlier_str}.png')
                    fig_box_error.write_image(path, scale=5)

        # Error
        unit = get_unit(measure)
        fig_error_box = px.box(df_error.sort_values(by='id_s'), x='id_s_name', y=measure, color='condition',
                               color_discrete_map=color_map, category_orders={"condition": category_order},
                               title=f'<b>Error for {measure_name} with CAD of {cad}<b>',
                               labels={'condition': 'Condition', 'id_s_name': 'Setting', 'value': f'Error {unit}'})

        if cad is not None:
            fig_error_box.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
            fig_error_box.add_hline(y=-cad, line_dash='dash', line_color='red', name='CAD')

        fig_error_box.update_layout(title=dict(text=f'<b>Error for {measure_name} <br>CAD = {cad} {unit}<b>', font = dict(size=20)),
                                    xaxis_title=dict(text='<b>Setting<b>', font=dict(size=16)),
                                    yaxis_title=dict(text=f'<b>Error {unit}<b>', font=dict(size=16)))

        if cap_yaxis:
            if cad is not None:
                fig_error_box.update_yaxes(range=[min(0, min(df_error[measure].min(), -cad-(cad*0.2))), min(5000, max(df_error[measure].max(), cad+(cad*0.2)))])
            else:
                fig_error_box.update_yaxes(range=[min(0, df_error[measure].min()), min(5000, df_error[measure].max())])

        if showfig:
            fig_error_box.show()

        if write_html:
            path = os.path.join(dir_out_box_error, f'murphy_box_{measure}_error{outlier_str}.html')
            fig_error_box.write_html(path)

        if write_svg:
            path = os.path.join(dir_out_box_error, f'murphy_box_{measure}_error{outlier_str}.svg')
            fig_error_box.write_image(path, scale=5)

        if write_png:
            path = os.path.join(dir_out_box_error, f'{suff}murphy_box_{measure}_error{outlier_str}.png')
            fig_error_box.write_image(path, scale=5)

        progbar.update(1)


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



    # Scatter-Plot of dat_measured against differences
    fig.add_trace(go.Scatter(x=dat_measured, y=diff, mode='markers', name='Differences'))

    # mean of differences
    fig.add_trace(go.Scatter(x=dat_measured, y=[mean_diff]*len(dat_measured), mode='lines', name='Mean of Differences'))

    # Add horicontal line at 0
    fig.add_hline(y=0, line_dash='dash', line_color='grey', name='Zero')

    # limits of agreement
    fig.add_hline(y=upper_limit, line_dash='dash', line_color='red', name=f'Upper Limit ({sd} SD)')
    fig.add_hline(y=lower_limit, line_dash='dash', line_color='red', name=f'Lower Limit ({sd} SD)')

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
            fig.write_html(path)
        elif path.endswith('.svg') or path.endswith('.jpg') or path.endswith('.jpeg'):
            fig.write_image(path)
