import numpy as np
import pandas
import pandas as pd

import plotly as py
from PIL.ImageOps import scale


def plot_blandaltman(df_murphy, measured_value, id_s, id_p=None, plot_to_val=False, filename=None, filetype='.png', use_smoother=True,
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
    import plotly.express as px
    import statsmodels.api as sm
    import plotly.graph_objects as go
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


def plot_blandaltman_old(df_murphy, measured_value, id_s, id_p=None, path=None, use_smoother=True, show_id_t=False, verbose=1, show_plots=True):
    """
    create bland altman plot.

    OMC Setting is Reference for Plots.

    if filename is given, the plot is saved to the given filename.

    If id_s is given, use only the data of the given id_s.

    If id_p is given, use only the data of the given id_p.
    The individual patients/participants are colour-coded in the plot

    if path is given, the plot is saved to the given path.

    :param dat_ref:
    :param dat_measured:
    :param show_plot:
    :return:
    """
    import plotly.graph_objects as go
    import statsmodels.api as sm


    if id_p is None:
        idx_p = df_murphy[df_murphy['id_s'] == id_s]['id_p'].unique()
    else:
        idx_p = [id_p]

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

    idx_p = dat_ref_all['id_p'].unique()
    #create List with al id_t (index is the same as in the list of values)
    idx_t_all = dat_measured_all['id_t'].to_list()

    mean_all = np.mean([dat_ref_all[measured_value].values, dat_measured_all[measured_value].values], axis=0)
    diff_all = dat_measured_all[measured_value].values - dat_ref_all[measured_value].values

    std_diff = np.std(diff_all)
    sd = 1.96
    upper_limit = mean_all + sd * std_diff
    lower_limit = mean_all - sd * std_diff

    n_colours = len(idx_p)
    colours = py.colors.qualitative.Plotly[:n_colours]

    # creating plot
    fig = go.Figure()

    for id_p, colour in zip(idx_p, colours):
        # calculate mean, difference, mean of differences, standard deviation of differences, upper and lower limits, smoother
        dat_ref = df_murphy[(df_murphy['id_s'] == 'S15133') & (df_murphy['id_p'] == id_p)].sort_values(by='id_t')
        dat_measured = df_murphy[(df_murphy['id_s'] == id_s) & (df_murphy['id_p'] == id_p)].sort_values(by='id_t')

        # Delete all Trials that are not in both datasets
        dat_ref = dat_ref[dat_ref['id_t'].isin(dat_measured['id_t'])]
        dat_measured = dat_measured[dat_measured['id_t'].isin(dat_ref['id_t'])]
        idx_t = dat_measured['id_t'].to_list()
        diff = dat_measured[measured_value].values - dat_ref[measured_value].values
        mean_diff = np.mean(diff)
        mean = np.mean([dat_ref[measured_value].values, dat_measured[measured_value].values], axis=0)

        # Scatter -Plot of means against differences
        fig.add_trace(go.Scatter(x=mean, y=diff, mode='markers', name=f'Differences: {id_p}', marker=dict(color=colour, symbol='circle')))

        if use_smoother:
            lowess = sm.nonparametric.lowess(diff, mean, frac=0.6)
            # smoother
            fig.add_trace(
                go.Scatter(x=lowess[:, 0], y=lowess[:, 1], mode='lines', name=f'Smoother (LOWESS): {id_p}',
                           line=dict(color='red')))
        else:
            # mean of differences
            fig.add_trace(go.Scatter(x=mean, y=[mean_diff]*len(mean), mode='lines', name=f'Mean of Differences: {id_p}', marker=dict(color=colour)))


    # limits of agreement
    fig.add_trace(go.Scatter(x=mean_all, y=[upper_limit]*len(mean_all), mode='lines', name=f'Upper Limit ({sd} SD)', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=mean_all, y=[lower_limit]*len(mean_all), mode='lines', name=f'Lower Limit ({sd} SD)', line=dict(dash='dash')))

    lowess = sm.nonparametric.lowess(diff_all, mean_all, frac=0.6)
    # smoother
    fig.add_trace(
        go.Scatter(x=lowess[:, 0], y=lowess[:, 1], mode='lines', name='Smoother (LOWESS): all', line=dict(color='red')))

    # Add horicontal line at 0
    fig.add_trace(go.Scatter(x=[min(mean_all), max(mean_all)], y=[0, 0], mode='lines', name='Zero Difference', line=dict(color='grey', dash='dash')))

    # update the layout
    if id_p is not None:
        title = f'Bland-Altman Plot for {measured_value} of {id_p}'
    else:
        title = f'Bland-Altman Plot for {measured_value} of {id_s}'


    range_lim = max(abs(min(diff)), abs(max(diff))) * 1.5
    y_range = [-range_lim, range_lim]

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

    if path is not None:
        if path.endswith('.html'):
            py.offline.plot(fig, filename=path)
        elif path.endswith('.png') or path.endswith('.jpg') or path.endswith('.jpeg'):
            fig.write_image(path)


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
    import plotly.graph_objects as go
    import statsmodels.api as sm

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
    fig.add_trace(go.Scatter(x=[min(dat_measured), max(dat_measured)], y=[0, 0], mode='lines', name='Zero Difference', line=dict(color='grey', dash='dash')))

    # Scatter-Plot of dat_measured against differences
    fig.add_trace(go.Scatter(x=dat_measured, y=diff, mode='markers', name='Differences'))

    # mean of differences
    fig.add_trace(go.Scatter(x=dat_measured, y=[mean_diff]*len(dat_measured), mode='lines', name='Mean of Differences'))

    # limits of agreement
    fig.add_trace(go.Scatter(x=dat_measured, y=[upper_limit]*len(dat_measured), mode='lines', name=f'Upper Limit ({sd} SD)', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=dat_measured, y=[lower_limit]*len(dat_measured), mode='lines', name=f'Lower Limit ({sd} SD)', line=dict(dash='dash')))

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


if __name__ == "__main__":
    # Beispiel-Daten (ersetze dies mit deinen eigenen Datensätzen)
    data1 = np.array([1, 2, 3, 4, 5])
    data2 = np.array([1.15554, 2.5, 2.4, 4.8, 8])
    path = r"I:\iDrink\test.png"
    plot_measured_vs_errors(data1, data2, id_s='S000', measured_value='Test', path=path, show_plots=True)