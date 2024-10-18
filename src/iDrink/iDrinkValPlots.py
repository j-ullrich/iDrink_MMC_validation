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

    # calculate mean, difference, mean of differences, standard deviation of differences, upper and lower limits
    mean = np.mean([dat_ref, dat_measured], axis=0)
    diff = dat_ref - dat_measured
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    # creating plot
    fig = go.Figure()

    # Scatter -Plot of means against differences
    fig.add_trace(go.Scatter(x=mean, y=diff, mode='markers', name='Differenzen'))

    # mean of differences
    fig.add_trace(go.Scatter(x=mean, y=[mean_diff]*len(mean), mode='lines', name='Mittelwert der Differenzen'))

    # limits of agreement
    fig.add_trace(go.Scatter(x=mean, y=[upper_limit]*len(mean), mode='lines', name='Oberes Limit (1.96 SD)', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=mean, y=[lower_limit]*len(mean), mode='lines', name='Unteres Limit (1.96 SD)', line=dict(dash='dash')))

    # update the layout
    if id_p is not None:
        title = f'Bland-Altman Plot for {measured_value} of {id_p}'
    else:
        title = f'Bland-Altman Plot for {measured_value} of {id_s}'


    fig.update_layout(title=title,
                      xaxis_title=f'{measured_value} of OMC',
                      yaxis_title=f'Difference of {id_s} from OMC')
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
    plot_blandaltman(data1, data2, path=path, show_plot=True)