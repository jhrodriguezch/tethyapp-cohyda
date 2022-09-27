from tethys_gizmos.gizmo_options import PlotlyView

import numpy as np
import pandas as pd
from plotly import graph_objs as go
from datetime import datetime, timedelta

from .model import last_curvagasto, get_Q2H_H2Q

"""
############################################################
                      helpers.py
############################################################
__author__  : jrc
__version__ : Beta 0.1
__obj__     : Additional functions for controller.py in tethys app
__date__    : 01 - sep - 2022
"""

# Draw functions
def create_time_serie(station_id,
                      obs,
                      sen,
                      nombre_variable,
                      station_data,
                      main_values = [float('nan')],
                      name_main_values = [''],
                      color=['aliceblue'],
                      ensamble_main = None,
                      ensamble_full = None,
                      ensamble_main_color=None,
                      last_sim = None,
                      ):
    """
    Create time series images.
    Input:
        # Station details:
        station_id          : str                -> Code of station.
        station_data        : pandas.DataFrame() -> General information of the station.\n
        # Variable details
        nombre_variable     : str                -> Name and units of variable.\n
        # Time series data:
        obs                 : pandas.DataFrame() -> Time series observed.
        sen                 : pandas.DataFrame() -> Time series sensor.
        last_sim            : pandas.DataFrame() -> Time series historic (short) simulated.
        ensamble_full       : pandas.DataFrame() -> Forecast time series.
        ensamble_main       : pandas.DataFrame() -> Main values for forecast time series.
        ensamble_main_color : list               -> Color fot main values for forecast time series.\n
        # Horizontal lines details:
        main_values         : list               -> Values for horizontal lines.
        name_main_values    : list               -> Names for identification the horizontal lines.
        color               : list               -> Hexadecimal colors for identification the horizontal lines.\n
    Return :
        PlotyView
    """

    # Coupling data in one dataframe
    obs = obs.astype('float').copy()
    sen = sen.astype('float').copy()
    last_sim = last_sim.astype('float').copy()
    rv = pd.concat([obs.add_suffix('_obs'), sen.add_suffix('_sen'), last_sim.add_suffix('_sim')])
    rv = rv.loc[rv.index.dropna()].copy()

    # Config horizontal lines to draw
    name_main_values = [x for _, x in sorted(zip(main_values, name_main_values), reverse=True)]
    color = [x for _, x in sorted(zip(main_values, color), reverse=True)]
    main_values = sorted(main_values, reverse=True)

    # List of images to plot
    data = []

    # ////////// PLOT time //////////

    # Plot full ensemble
    # #### Remove for quick draw
    # if not ensamble_full is None:
    #     for ens_col in ensamble_full.columns:
    #         time_serie_ens = go.Scatter(
    #             x = ensamble_full.index,
    #             y = ensamble_full[ens_col],
    #             mode = 'lines',
    #             line=dict(color='lightgray'),
    #             opacity=0.6,
    #             name=ens_col,
    #             showlegend=False,
    #             hoverinfo='skip'
    #         )
    #         data.append(time_serie_ens)


    # Plot main ensamble
    if not ensamble_main is None:
        for ens_col_ii, ens_color_ii in list(zip(ensamble_main.columns, ensamble_main_color)):
            time_series_ens_main = go.Scatter(
                x = ensamble_main.index,
                y = ensamble_main[ens_col_ii],
                mode='lines',
                name=ens_col_ii,
                line=dict(color= ens_color_ii, dash='dash'),
                legendgroup = '0',
                legendgrouptitle_text = 'Pronóstico',
            )
            data.append(time_series_ens_main)


    # Plot horizontal lines
    if len(main_values) > 0 and len(rv.index) > 0:
        for values_ii, names_ii, color_ii in list(zip(main_values, name_main_values, color)):
            time_series_lines = go.Scatter(
                x = [rv.index[0], datetime.now() + timedelta(days=16)],
                y = [values_ii, values_ii],
                mode = 'lines',
                name = names_ii,
                line=dict(color=color_ii),
                legendgroup='1',
                legendgrouptitle_text = 'Alertas',
            )
            data.append(time_series_lines)


    # Plot Observed time series
    time_series_obs = go.Scatter(
        x    = rv.index,
        y    = rv.data_obs,
        mode = 'lines+markers',
        name = 'Observado',
        line = dict(color='#00b27c'),
    )

    # Plot Sensor time series
    time_series_sen = go.Scatter(
        x   = rv.index,
        y   = rv.data_sen,
        mode='markers',
        name='Sensor',
        line = dict(color='#000000'),
    )

    # Plot historical (short) simulation
    time_series_lastSim = go.Scatter(
        x   = rv.index,
        y   = rv.data_sim,
        mode='markers',
        name='Simulacion',
        line = dict(color='#1368b2'),
    )

    # Build layout
    layout = {
        'title':str(station_data['nombre'].values[0]),
        'xaxis':{'title' : 'Fecha',
                 'range' : [datetime.now() - timedelta(days=5), datetime.now() + timedelta(days=16)]},
        'yaxis':{'title' : nombre_variable},
        'hovermode':'x',
    }

    # Add images to plot
    data.append(time_series_obs)
    data.append(time_series_sen)
    data.append(time_series_lastSim)

    # Print figure for the server
    figure = {'data': data, 'layout':layout}
    time_series_plot = PlotlyView(figure, height='30%', width='100%')
    return time_series_plot


def create_curva_gasto(full_curva_gasto = None,
                       fun_curva_gasto = None):
    """
    Build rate curve image
    Input:
        full_curva_gasto : pandas.DataFrame -> Observation points of the rate curve.
        fun_curva_gasto  : pandas.DataFrame -> Function of the rate curve to plot.
    Return:
        PlotlyView
    """


    # List of images to plot
    data = []

    if len(full_curva_gasto) > 1:
        # Build water level data from rate curve function
        q_vals = list(np.arange(full_curva_gasto['CAUDAL'].min(),
                                full_curva_gasto['CAUDAL'].max(),
                                (full_curva_gasto['CAUDAL'].max() - full_curva_gasto['CAUDAL'].min())/10))

        h_vals, tipo, parametros = get_Q2H_H2Q(original_ts=pd.DataFrame({'data':q_vals}),
                                               change_fun=fun_curva_gasto,
                                               typechange='Q2H')

        # Build data function description
        if tipo == 'lineal':
            func_desc = 'y={0:.2f}*x+{1:.2f}'.format(parametros[0], parametros[1])
        if tipo =='quadratic':
            func_desc = 'y={0:.2f}*x^2+{1:.2f}*x+{2:.2f}'.format(parametros[0], parametros[1], parametros[2])
        if tipo == 'potential':
            func_desc = 'y={0:.2f}*(x+{1:.2f})^{2:.2f}'.format(parametros[0], parametros[1], parametros[2])

        # Plot function rate curve
        fun_res_cgasto = go.Scatter(
            x=q_vals,
            y=h_vals.data,
            mode='lines',
            name='Aproximado : {0}'.format(func_desc),
            line=dict(color='blue'),
        )
        data.append(fun_res_cgasto)

        # Plot scatter points for observation periode
        for num, id_curvagasto in enumerate(full_curva_gasto['NO.'].unique()):

            ini_date = np.datetime_as_string(full_curva_gasto.loc[full_curva_gasto['NO.'] == id_curvagasto, 'F. INICIAL'].values[0], unit='D')
            end_date = np.datetime_as_string(full_curva_gasto.loc[full_curva_gasto['NO.'] == id_curvagasto, 'F. FINAL'].values[0], unit='D')

            points_cgasto = go.Scatter(
                x=full_curva_gasto.loc[full_curva_gasto['NO.'] == id_curvagasto, 'CAUDAL'],
                y=full_curva_gasto.loc[full_curva_gasto['NO.'] == id_curvagasto, 'NIVEL'],
                mode='markers',
                name='{0} a {1}'.format(ini_date, end_date),
                line=dict(color='#000000'),
                opacity= (num + 1) / (len(full_curva_gasto['NO.'].unique()) + 1.0),
                legendgroup='0',
                legendgrouptitle_text='Observados',
            )

            data.append(points_cgasto)
    else:
        data.append(__empy_image__())


    layout = {
        'title': 'Curva de gasto.',
        'xaxis': {'title': 'Caudal [m3/s]',
                  'showspikes':True},
        'yaxis': {'title': 'Nivel [m]',
                  'showspikes':True},
    }

    # Build figure for the server
    figure = {'data': data, 'layout': layout}
    curva_gasto_plot = PlotlyView(figure, height='30%', width='100%')

    return curva_gasto_plot


def __empy_image__():
    """
    Build an empty data frame
    """
    return go.Scatter(
                x=[],
                y=[],
            )