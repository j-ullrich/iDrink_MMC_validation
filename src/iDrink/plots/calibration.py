"""calibration part of iDrinkValPlots.py (moved without changes, see iDrinkValPlots.py for the public names)."""
import pandas as pd
import os
import plotly.express as px


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

    fig.update_layout(title = dict(text='<b>Calibration Errors for camera setups<b>', font=dict(size=20)),
                      xaxis_title = dict(text='Camera Setup', font=dict(size=18)),
                      yaxis_title = dict(text='Reprojection Error', font=dict(size=18))
                      )
    fig.update(layout_showlegend=False)

    if show_fig:
        fig.show()
    # save plot

    os.makedirs(dir_dst, exist_ok=True)
    path = os.path.join(dir_dst, f'CalibrationErrors.svg')
    fig.write_image(path, scale=5)
