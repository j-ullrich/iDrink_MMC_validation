from logging import disable

import numpy as np
import pandas
import pandas as pd

import os
import ast
import glob

from ansi2html.style import color
from plotly.io import write_html
from tqdm import tqdm

import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import statsmodels.api as sm

# idx_p to ignore for plotting
ignore_id_p = []

murphy_measures = ["PeakVelocity_mms",
                   "elbowVelocity",
                   "tTopeakV_s",
                   "tToFirstpeakV_s",
                   "tTopeakV_rel",
                   "tToFirstpeakV_rel",
                   "NumberMovementUnits",
                   "InterjointCoordination",
                   "trunkDisplacementMM",
                   "trunkDisplacementDEG",
                   "ShoulderFlexionReaching",
                   "ElbowExtension",
                   "shoulderAbduction"]

rgba_mmc = '100, 149, 237'
rgba_omc = '255, 165, 0'


def get_unit(kin):

    cases_deg = ['trunk_ang', 'elbow_flex_pos', 'shoulder_flex_pos', 'shoulder_abduction_pos',
                 'trunkDisplacementDEG', 'ShoulderFlexionReaching', 'ElbowExtension',
                 'shoulderAbduction', 'shoulderFlexionDrinking']

    match kin:
        case 'hand_vel' | 'PeakVelocity_mms':
            unit = 'mm/s'
        case 'elbow_vel' | 'elbowVelocity':
            unit = 'deg/s'
        case 'trunk_disp' | 'trunkDisplacementMM':
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
        case 'PeakVelocity_mms':
            measure_name = 'peak_V'
        case 'elbowVelocity':
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
        case 'trunkDisplacementMM':
            measure_name = 'trunk_disp'
        case 'trunkDisplacementDEG':
            return None
        case 'ShoulderFlexionReaching':
            measure_name = 'arm_flex_reach'
        case 'ElbowExtension':
            measure_name = 'elb_ext'
        case 'shoulderAbduction':
            measure_name = 'arm_abd'
        case 'shoulderFlexionDrinking':
            measure_name = 'arm_flex_drink'
        case _:
            return

    return df.loc[0, measure_name]

def get_title_measure_name(measure):
    """returns a string based on the murphy measure for a figure_title"""
    match measure:
        case 'PeakVelocity_mms':
            title = 'Peak Velocity'
        case 'elbowVelocity':
            title = 'Elbow Velocity'
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
            title = 'Trunk Displacement'
        case 'trunkDisplacementDEG':
            title = 'Trunk Displacement'
        case 'ShoulderFlexionReaching':
            title = 'Shoulder Flexion Reaching'
        case 'ElbowExtension':
            title = 'Elbow Extension'
        case 'shoulderAbduction':
            title = 'Shoulder Abduction'
        case 'shoulderFlexionDrinking':
            title = 'Shoulder Flexion Drinking'
        case _:
            title = ''
    return title

def plot_murphy_blandaltman(root_stat, write_html=False, write_svg=True, show_plots=False, verbose = 1):
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
        sd = 1.96
        upper_limit = + sd * std_diff
        lower_limit = - sd * std_diff

        fig.add_hline(y=upper_limit, line_dash='dash', line_color='orange', name=f'Upper Limit ({sd} SD)')
        fig.add_hline(y=lower_limit, line_dash='dash', line_color='orange', name=f'Lower Limit ({sd} SD)')

        # add horizontal line for cad
        if cad is not None:
            fig.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
            fig.add_hline(y=-cad, line_dash='dash', line_color='red', name='CAD')

        title_measure = get_title_measure_name(measure)

        if id_p is None:
            title = f'Bland Altman Plot for {title_measure} of {id_s} with CAD of {cad} {unit}'
            dir_id = f'{id_s}'
            id_sp = f'{id_s}'
        else:
            title = f'Bland Altman Plot for {title_measure} of {id_s}, {id_p} with CAD of {cad} {unit}'
            dir_id = f'{id_s}_{id_p}'
            id_sp = f'{id_s}_{id_p}'

        fig.update_layout(title=title,
                          xaxis_title=f'Mean of {title_measure} {unit}',
                          yaxis_title=f'Difference of MMC from OMC {unit}',
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
    idx_s = sorted(df_murphy_mmc['id_s'].unique())


    # get total number of combinations of id_s and id_p for progbar
    total = 0
    for id_s in idx_s:
        total +=  df_murphy_mmc[df_murphy_mmc['id_s'] == id_s]['id_p'].nunique()

    total *= len(murphy_measures)

    progbar = tqdm(range(total), desc='Plotting Bland Altman', unit='Trial', disable=verbose<1)

    for measure in murphy_measures:
        for id_s in idx_s:
            df_murphy_mmc_s = df_murphy_mmc[df_murphy_mmc['id_s'] == id_s]

            idx_p = sorted(df_murphy_mmc_s['id_p'].unique())

            df_aff_s = None
            df_unaff_s = None


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
                    std_aff = np.std(df_aff['mmc'] - df_aff['omc'])


                    sd = 1.96

                    upper_limit_unaff = + sd * std_unaff
                    lower_limit_unaff = - sd * std_unaff

                    upper_limit_aff = + sd * std_aff
                    lower_limit_aff = - sd * std_aff


                    fig.add_hline(y=upper_limit_unaff, line_dash='dash', line_color='orange', name=f'Upper Limit ({sd} SD)', row=1, col=1)
                    fig.add_hline(y=lower_limit_unaff, line_dash='dash', line_color='orange', name=f'Lower Limit ({sd} SD)', row=1, col=1)

                    fig.add_hline(y=upper_limit_aff, line_dash='dash', line_color='orange', name=f'Upper Limit ({sd} SD)', row=1, col=2)
                    fig.add_hline(y=lower_limit_aff, line_dash='dash', line_color='orange', name=f'Lower Limit ({sd} SD)', row=1, col=2)

                    # add limit of agreement to legend
                    mean_peak = np.mean([df_unaff['mmc'].values, df_unaff['omc'].values])
                    fig.add_trace(go.Scatter(x=[mean_peak], y=[0], mode='lines', name='Limits of Agreement', line=dict(color='orange', dash='dash')))
                    # add cad to legend
                    fig.add_trace(go.Scatter(x=[mean_peak], y=[0], mode='lines', name='CAD', line=dict(color='red', dash='dash')))

                    fig.update_xaxes(title_text=f'Mean of {title_measure} {unit}', row=1, col=1)
                    fig.update_xaxes(title_text=f'Mean of {title_measure} {unit}', row=1, col=2)
                    fig.update_yaxes(title_text=f'Difference of MMC from OMC {unit}', row=1, col=1)

                    fig.update_layout(title=f'Averaged Timeseries for {title_measure} of {id_s}_{id_p}')

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

                sd = 1.96

                upper_limit_unaff = + sd * std_unaff
                lower_limit_unaff = - sd * std_unaff

                upper_limit_aff = + sd * std_aff
                lower_limit_aff = - sd * std_aff

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

                fig.update_xaxes(title_text=f'Mean of {title_measure} {unit}', row=1, col=1)
                fig.update_xaxes(title_text=f'Mean of {title_measure} {unit}', row=1, col=2)
                fig.update_yaxes(title_text=f'Difference of MMC from OMC {unit}', row=1, col=1)

                fig.update_layout(title=f'Averaged Timeseries for {title_measure} of {id_s} with CAD of {cad} {unit}')

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


def plot_murphy_error_rmse_box_bar_plot(dir_root, outlier_corrected = False, showfig=False, write_html=False, write_svg=True, write_png=True, verbose=1):
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

    df_rmse_nonan = df_rmse.dropna()
    df_rmse_mean = df_rmse[df_rmse['id_p'].isna()]

    progbar = tqdm(murphy_measures, desc='Plotting for ', unit='Measure', disable=verbose<1)

    for measure in murphy_measures:
        progbar.set_description(f'Plotting for {measure}')

        cad = get_cad(df_cad, measure)

        # fig_rmse = px.box(df_rmse_nonan, x='id_s', y = measure, color='condition')

        fig_box = go.Figure()
        fig_box_error = go.Figure()
        fig_bar = go.Figure()
        for condition, group in df_rmse_nonan.groupby('condition'):
            offset = 1 if condition == 'affected' else 0
            fig_box.add_trace(go.Box(x=group['id_s'], y=group[measure], name=condition, offsetgroup=offset))

        for condition, group in df_rmse_mean.groupby('condition'):
            fig_bar.add_trace(go.Bar(x=group['id_s'], y=group[measure], name=condition))

        # add horicontal line for cad
        if cad is not None:
            fig_box.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
            fig_bar.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')


        fig_box.update_layout(title=f'RMSE for {measure} with CAD of {cad}', xaxis_title='Setting ID', yaxis_title='RMSE')
        fig_bar.update_layout(title=f'RMSE for {measure} with CAD of {cad}', xaxis_title='Setting ID', yaxis_title='RMSE')

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
            path = os.path.join(dir_out_box_rmse, f'murphy_box_{measure}_rmse{outlier_str}.png')
            fig_box.write_image(path, scale=5)
            path = os.path.join(dir_out_bar_rmse, f'murphy_bar_{measure}_rmse{outlier_str}.png')
            fig_bar.write_image(path, scale=5)

        for id_p in df_rmse_nonan['id_p'].unique():

            progbar.set_description(f'Plotting Boxplots for {measure} of {id_p}')
            fig_box = go.Figure()
            fig_bar = go.Figure()
            for condition, group in df_rmse_nonan[df_rmse_nonan['id_p'] == id_p].groupby('condition'):
                offset  = 1 if condition == 'affected' else 0
                fig_box.add_trace(go.Box(x=group['id_s'], y=group[measure], name=condition, offsetgroup=offset))

            for condition, group in df_rmse_nonan[df_rmse_nonan['id_p'] == id_p].groupby('condition'):
                fig_bar.add_trace(go.Bar(x=group['id_s'], y=group[measure], name=condition))

            fig_box_error = px.box(df_error[df_error['id_p'] == id_p], x='id_s', y=measure, color='condition',
                                   title=f'Error for {measure} of {id_p}',
                                   labels={'condition': 'Condition', 'id_s': 'Setting ID', 'value': 'Error'})

            # add horicontal line for cad
            if cad is not None:
                fig_box.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
                fig_bar.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
                fig_box_error.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
                fig_box_error.add_hline(y=-cad, line_dash='dash', line_color='red', name='CAD')

            fig_box.update_layout(title=f'RMSE for {measure} of {id_p} with CAD of {cad}', xaxis_title='Setting ID', yaxis_title='RMSE')
            fig_bar.update_layout(title=f'RMSE for {measure} of {id_p} with CAD of {cad}', xaxis_title='Setting ID', yaxis_title='RMSE')

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
                path = os.path.join(dir_out_box_rmse, f'{id_p}_murphy_box_{measure}_rmse{outlier_str}.png')
                fig_box.write_image(path, scale=5)
                path = os.path.join(dir_out_bar_rmse, f'{id_p}_murphy_bar_{measure}_rmse{outlier_str}.png')
                fig_bar.write_image(path, scale=5)
                path = os.path.join(dir_out_box_error, f'{id_p}_murphy_box_{measure}_error{outlier_str}.png')
                fig_box_error.write_image(path, scale=5)

        # Error
        fig_error_box = px.box(df_error.sort_values(by='id_s'), x='id_s', y=measure, color='condition', title=f'Error for {measure}',
                               labels={'condition': 'Condition', 'id_s': 'Setting ID', 'value': 'Error'})

        if cad is not None:
            fig_error_box.add_hline(y=cad, line_dash='dash', line_color='red', name='CAD')
            fig_error_box.add_hline(y=-cad, line_dash='dash', line_color='red', name='CAD')

        if showfig:
            fig_error_box.show()

        if write_html:
            path = os.path.join(dir_out_box_error, f'murphy_box_{measure}_error{outlier_str}.html')
            fig_error_box.write_html(path)

        if write_svg:
            path = os.path.join(dir_out_box_error, f'murphy_box_{measure}_error{outlier_str}.svg')
            fig_error_box.write_image(path, scale=5)

        if write_png:
            path = os.path.join(dir_out_box_error, f'murphy_box_{measure}_error{outlier_str}.png')
            fig_error_box.write_image(path, scale=5)




        progbar.update(1)


def plot_timeseries_blandaltman_scale_location(root_val, kinematic, idx_p=None, idx_t=None, dynamic='dynamic', write_html=True, write_svg=True, show_plots=True):
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
                fig_bland.add_hline(y=0, line_dash='dash', line_color='grey', name='Zero')

                # add limits of agreement
                std_diff = np.std(error)
                sd = 1.96
                upper_limit = + sd * std_diff
                lower_limit = - sd * std_diff

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


def plot_timeseries_boxplot_error_rmse(root_val, showfig=False, write_html=False, write_svg=True, verbose=1):
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

                unit = get_unit(metric)

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

                if write_svg:
                    path = os.path.join(dir_dst_error_box, f'{id_p}_{metric}_{dynamic}_{normal}_error_box.svg')
                    fig_err.write_image(path, scale=5)

                    path = os.path.join(dir_dst_rmse_box, f'{id_p}_{metric}_{dynamic}_{normal}_rmse_box.svg')
                    fig_rmse.write_image(path, scale=5)

                    path = os.path.join(dir_dst_rmse_box_log, f'{id_p}_{metric}_{dynamic}_{normal}_log_rmse_box.svg')
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

def plot_timeseries_averaged(root_val, id_s, id_p, dynamic=False, fig_show=False, write_html=False, write_svg=True,
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

            fig.update_layout(title=f'Averaged Timeseries for {kinematic} of {id_s}_{id_p} - {side} - {affected}',
                              xaxis_title='Normalized Time',
                              yaxis_title=f'{kinematic} [{unit}]', )

            if fig_show:
                fig.show()

            if write_html:
                path = os.path.join(dir_out, f'{id_s}_{id_p}_{kinematic}_{side}_{affected}_averaged.html')
                fig.write_html(path)

            if write_svg:
                path = os.path.join(dir_out, f'{id_s}_{id_p}_{kinematic}_{side}_{affected}_averaged.svg')
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

            fig.update_layout(title=f'Averaged Timeseries for {kinematic} of {id_s}_{id_p}',)

            if fig_show:
                fig.show()

            if write_html:
                path = os.path.join(dir_out, f'{id_s}_{id_p}_{kinematic}_averaged.html')
                fig.write_html(path)

            if write_svg:
                path = os.path.join(dir_out, f'{id_s}_{id_p}_{kinematic}_averaged.svg')
                fig.write_image(path, scale=5)

        progbar.update(1)
    progbar.close()





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
    path = os.path.join(dir_dst, f'CalibrationErrors.svg')
    fig.write_image(path, scale=5)


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


if __name__ == "__main__":
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

    verbose = 1

    csv_calib_errors = os.path.join(root_logs, 'calib_errors.csv')
    caloib_plots_dst = os.path.join(root_stat, '05_calibration')
    calibration_boxplot(csv_calib_errors, caloib_plots_dst, verbose=1, show_fig=False)

    #plot_murphy_blandaltman(root_stat, write_html=True, write_svg=True, show_plots=False, verbose=1)

    for corr in [False]:
        plot_murphy_error_rmse_box_bar_plot(root_val, write_html=True, outlier_corrected=corr)

    #plot_timeseries_averaged(root_val, 'S001', 'P07', dynamic=dynamic)

    #plot_timeseries_boxplot_error_rmse(root_val, showfig=False, write_html=False, write_svg=True, verbose=1)

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

    kinematics =['hand_vel', 'elbow_vel', 'trunk_disp', 'trunk_ang', 'elbow_flex_pos', 'shoulder_flex_pos',
                 'shoulder_abduction_pos']

    # TODO: Make sure, MMC and OMC have always the same colour in Plots MMC - Blue, OMC - Orange check on averaged timeseries

    kinematic = kinematics[4]
    # iterate over all plottable trials and create plots

    for i in range(len(df_plottable)):
        #id_s = df_plottable['id_s'][i]
        id_p = df_plottable['id_p'][i]
        id_t = df_plottable['id_t'][i]


        for kinematic in kinematics:
            plot_timeseries_blandaltman_scale_location(root_val, kinematic=kinematic, idx_p=id_p, idx_t=id_t, dynamic=dynamic_str, write_html=False, write_svg=True, show_plots=False)
            pass

        generate_plots_for_timeseries(root_val, id_p_in = id_p, id_t_in = id_t, dynamic=dynamic,
                                      showfig = False, write_html=False, write_svg=True)



    #plot_timeseries_RMSE(id_s, dir_dst, dir_data, joint_data=True, id_p=None,  verbose=1)
    #plot_measured_vs_errors(data1, data2, id_s='S000', measured_value='Test', path=path, show_plots=True)
