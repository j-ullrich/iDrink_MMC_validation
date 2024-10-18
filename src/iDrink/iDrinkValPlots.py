import numpy as np
import pandas as pd

import plotly as py


def plot_blandaltman(dat_ref, dat_measured, measured_value, id_s, id_p=None, path=None, verbose=1, show_plots=True):
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
    mean = np.mean([dat_ref, dat_measured], axis=0)
    diff = dat_ref - dat_measured
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    sd = 1.96
    upper_limit = mean_diff + sd * std_diff
    lower_limit = mean_diff - sd * std_diff
    lowess = sm.nonparametric.lowess(diff, mean, frac=0.6)

    # creating plot
    fig = go.Figure()

    # Add horicontal line at 0
    fig.add_trace(go.Scatter(x=[min(mean), max(mean)], y=[0, 0], mode='lines', name='Zero Difference', line=dict(color='grey', dash='dash')))

    # Scatter -Plot of means against differences
    fig.add_trace(go.Scatter(x=mean, y=diff, mode='markers', name='Differences'))

    # mean of differences
    fig.add_trace(go.Scatter(x=mean, y=[mean_diff]*len(mean), mode='lines', name='Mean of Differences'))

    # limits of agreement
    fig.add_trace(go.Scatter(x=mean, y=[upper_limit]*len(mean), mode='lines', name=f'Upper Limit ({sd} SD)', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=mean, y=[lower_limit]*len(mean), mode='lines', name=f'Lower Limit ({sd} SD)', line=dict(dash='dash')))

    # smoother
    fig.add_trace(
        go.Scatter(x=lowess[:, 0], y=lowess[:, 1], mode='lines', name='Smoother (LOWESS)', line=dict(color='red')))

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