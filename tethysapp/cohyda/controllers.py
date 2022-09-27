from django.shortcuts import render, reverse

from tethys_sdk.gizmos import *
from tethys_sdk.permissions import login_required

import os
import time
import warnings

from .model import Read_DataBase, historical_timeserie, get_Q2H_H2Q, data2dfserie
from .helpers import *

"""
############################################################
                      controller.py
############################################################
__author__  : jrc
__version__ : Beta 0.1
__obj__     : Controller python file for tethys app
__date__    : 01 - sep - 2022
"""

############ Secundary functions #########################
def station_features_fun(station, dict_station):
    """
    From point corrds and properties, build a feature for geojson.
    input:
        station : pandas.DataFrame -> columns with lng coord and
                lat coord (longitude and latitude)
        dict_station : dict -> Properties for the point
    Return:
        dict
    """
    return {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [station['lng'], station['lat']]
        },
        'properties': dict_station,
    }


def station_layer_fun(features, color_hex, rsize=4,
                      description='Estado normal'):
    """
    Build a MvLayer form data features and point radio size, color in
    hexadecimal color and the description name.
    Input:
        features    : dict -> features for the geojson.
        color_hex   : str  -> color in hexadecimal.
        r_size      : int  -> radio size.
        description : str  -> Description of the legend.
    Return:
        MVLayer with the points.

    """
    # Built collection
    station_collection = {
        'type':'FeatureCollection',
        'crs' : {
            'type': 'name',
            'properties':{
                'name' : 'EPSG:4326'
            }
        },
        'features' : features
    }
    # Built station map style
    style = {'ol.style.Style': {
        'image': {'ol.style.Circle': {
            'radius': rsize,
            'fill': {'ol.style.Fill': {
                'color': color_hex #'#1368b2'
            }},
            'stroke': {'ol.style.Stroke': {
                'color': '#ffffff',
                'width': 1
            }}
        }}
    }}
    # Built station layer
    return  MVLayer(
        source            = 'GeoJSON',
        options           = station_collection,
        legend_title      = description,
        layer_options     = {'style': style},
        feature_selection = True
    )


# Global variables : Colors
cnegro      = '#000000'
crojo       = '#FF0000'
cnaranja    = '#FF9900'
camarillo   = '#FFFD00'
cmorado     = '#D000FF'

cazulIdeam  = '#1368b2'
cverdeIdeam = '#00b27c'


# Read dir to database and actualize the database
foo = Read_DataBase(path_dir=os.path.dirname(__file__))
foo()


############ Main controllers #########################
@login_required()
def home(request):
    """
    Controller for the app home page.
    """

    global foo

    # Refrech database
    foo()

    # ////////////////// Build catchment layer //////////////////
    # TODO: Add catchements

    # ////////////////// Build reach layer //////////////////
    reach_map = foo.get_reachgeojson()
    reach_layer = MVLayer(
        source='GeoJSON',
        options=reach_map,
        legend_title='reach map',
        # layer_options={'style': style},
        feature_selection=False
    )

    # ////////////////// Build Station layer //////////////////
    # Call station map
    station_location = foo.get_station_location()

    # Built station map features
    features           = []
    features_nmax      = []
    features_nrojo     = []
    features_nnaranja  = []
    features_namarillo = []
    features_nmin      = []

    if 'lng' and 'lat' in station_location.columns:
        for enu, station in station_location.iterrows():
            dict_station = {}

            for ind in station.index:
                if not 'nan' in str(station[ind]):
                    dict_station.update({str(ind):station[ind]})

            if station['id_alert'] == 6:
                features_nmax.append(station_features_fun(station, dict_station))
            elif station['id_alert'] == 5:
                features_nrojo.append(station_features_fun(station, dict_station))
            elif station['id_alert'] == 4:
                features_nnaranja.append(station_features_fun(station, dict_station))
            elif station['id_alert'] == 3:
                features_namarillo.append(station_features_fun(station, dict_station))
            elif station['id_alert'] == 2:
                features_nmin.append(station_features_fun(station, dict_station))
            else:
                features.append(station_features_fun(station, dict_station))
    else:
        print('Warning:')
        print('ReporteTablaEstaciones.csv no tiene las columnas lat y lng.')

    # Built layer for all warning types
    station_layer_nmax      = station_layer_fun(features_nmax,      cnegro,    rsize=6.5, description='Nivel max. [{0}]'.format(len(features_nmax)))
    station_layer_nrojo     = station_layer_fun(features_nrojo,     crojo,     rsize=6.0, description='Alerta roja [{0}]'.format(len(features_nrojo)))
    station_layer_nnaranja  = station_layer_fun(features_nnaranja,  cnaranja,  rsize=5.5, description='Alerta naranja [{0}]'.format(len(features_nnaranja)))
    station_layer_namarillo = station_layer_fun(features_namarillo, camarillo, rsize=5.0, description='Alerta amarilla [{0}]'.format(len(features_namarillo)))
    station_layer_nmin      = station_layer_fun(features_nmin,      cmorado,   rsize=4.5, description='Nivel minimo [{0}]'.format(len(features_nmin)))
    station_layer           = station_layer_fun(features,           cazulIdeam)

    # ////////////////// General maps controller //////////////////
    view_options = MVView(
		projection = 'EPSG:4326',
		center     = [-71, 4],
		zoom       = 5.0,
	)

    # ////////////////// Build map in home url //////////////////
    main_map = MapView(
        height  = '100%',
        width   = '100%',
        layers  = [station_layer_nmin, station_layer_nmax, station_layer_nrojo,
                   station_layer_nnaranja, station_layer_namarillo, station_layer,
                   reach_layer],
        basemap = [{'CartoDB': {'style': 'dark'}}],
        view    = view_options,
        legend  = True,
    )

    context = {
        'main_map': main_map
    }

    return render(request, 'cohyda/home.html', context)


@login_required()
def station_details(request, station_code):
    """
    Controllers for the station details page.
    """
    print('\n... New Station ...')
    print(station_code)

    start_time = time.time()

    # //////////// Call station data ////////////
    station_loc = foo.get_station_data()
    station_loc = station_loc.loc[station_loc['id'].astype('str') == station_code].reset_index(drop=True)

    print('')
    print(station_loc.T)
    print('')

    # //////////// Call rate curve ////////////
    c_gasto_db = foo.get_curvagasto(station_code)
    c_gasto_fun = foo.get_curvagasto_fun(c_gasto_db)


    # //////////// Call time series ////////////
    # Database load Stream Flow - Observation - from Fews
    qObs_db, qSen_db, _    = foo.get_fewsdata(station_code, 'Q')
    # Database load WaterLevel - Observation - from Fews
    hObs_db, hSen_db, p_db = foo.get_fewsdata(station_code, 'H')


    # Database load Stream Flow - Historic Observation - from hydroshare
    qHist_db = foo.getHistoricalData(station_code, 'Q')
    # Database load WaterLevel - Historic Observation - from hydroshare
    hHist_db = foo.getHistoricalData(station_code, 'H')

    # //////////// Define time series to work ////////////
    qHist = historical_timeserie(original=[qHist_db, qObs_db],
                                 alternative=[hHist_db, hObs_db],
                                 change_fun=c_gasto_fun,
                                 var_type='Q')

    hHist = historical_timeserie(original=[hHist_db, hObs_db],
                                 alternative=[qHist_db, qObs_db],
                                 change_fun=c_gasto_fun,
                                 var_type='H')

    # Database load Stream Flow - Historic Simulation - from ECMWF StreamFlow model
    hist_sim = foo.get_historical_simulation(station_loc,
                                             obsData=qHist)

    # Database load Stream Flow - Historic Simulation (short) - from ECMWF StreamFlow model
    qlast_sim = foo.get_forecast_records(station_loc,
                                         obsData=qHist,
                                         simData=hist_sim)

    print('')
    print('')
    print('Carga base de datos')
    print('--- demora {0} seg ---'.format(time.time() - start_time))
    start_time = time.time()

    # Database load Stream Flow - Forecast - from ECMWF StreamFlow model
    qforecMain, qforecFull = foo.get_forecastdata(station_loc,
                                                  hist_obs=qHist,
                                                  hist_sim=hist_sim)

    print('')
    print('')
    print('--- bias correction --- ')
    print('--- demora {0} seg ---'.format(time.time() - start_time))

    start_time = time.time()

    # ////////////
    # Transform stream flow time series to water level time series with rate curve
    # ////////////

    ###################################################
    ################## JRC ############################
    ###################################################
    # hforecFull, hforecMain = foo.extract_waterlevel_forecast(original_ts = [qforecFull, qforecMain],
    #                                                          hist_sim = hist_sim,
    #                                                          hist_obs = hHist,
    #                                                          change_fun = c_gasto_fun)

    # hlastsim = foo.extract_waterlevel_lastsim(original_ts = qlast_sim,
    #                                           hist_sim = hist_sim,
    #                                           hist_obs = hHist,
    #                                           change_fun = c_gasto_fun)
    ###################################################
    ################## JRC ############################
    ###################################################

    # TODO: Add possibility for calc from bias correction
    hforecMain, _, _ = get_Q2H_H2Q(original_ts=qforecMain,
                                   change_fun=c_gasto_fun,
                                   typechange='Q2H')


    # TODO: Add possibility for calc from bias correction
    hforecFull, _, _ = get_Q2H_H2Q(original_ts=qforecFull,
                                   change_fun=c_gasto_fun,
                                   typechange='Q2H')

    # ////////////
    # Transform water level warnings to stream flows warnings with curve rate
    # ////////////
    # TODO: aff to foo class
    qmaxhist  , _ , _ = get_Q2H_H2Q(original_ts = data2dfserie(station_loc['umaxhis'].values[0]),
                                    change_fun  = c_gasto_fun, typechange='H2Q')
    qroja     , _ , _ = get_Q2H_H2Q(original_ts = data2dfserie(station_loc['uroja'].values[0]),
                                    change_fun  = c_gasto_fun, typechange='H2Q')
    qnaranja  , _ , _ = get_Q2H_H2Q(original_ts = data2dfserie(station_loc['unaranja'].values[0]),
                                    change_fun  = c_gasto_fun, typechange='H2Q')
    qamarilla , _ , _ = get_Q2H_H2Q(original_ts = data2dfserie(station_loc['uamarilla'].values[0]),
                                    change_fun  = c_gasto_fun, typechange='H2Q')
    qbajos    , _ , _ = get_Q2H_H2Q(original_ts = data2dfserie(station_loc['ubajos'].values[0]),
                                    change_fun  = c_gasto_fun, typechange='H2Q')

    # ////////////
    # Transform stream flows forecast simulation (short) to
    # water level simulation with rate curve.
    # ////////////
    # TODO: Add possibility for calc from bias correction
    hlastsim  , _ , _ = get_Q2H_H2Q(original_ts = qlast_sim,
                                    change_fun  = c_gasto_fun,
                                    typechange='Q2H')

    print('')
    print('')
    print('---  Fix data --- ')
    print('--- demora {0} seg ---'.format(time.time() - start_time))

    start_time = time.time()

    # //////////// Build images ////////////
    # Built figure streamflow
    streamflow_serie_plot = create_time_serie(station_code,
                                         obs=qObs_db,
                                         sen=qSen_db,
                                         ensamble_main = qforecMain,
                                         ensamble_full = qforecFull,
                                         ensamble_main_color=[crojo,
                                                              crojo,
                                                              cazulIdeam,
                                                              cazulIdeam,
                                                              cverdeIdeam],
                                         station_data=station_loc,

                                         main_values=[qmaxhist.data.values[0],
                                                     qroja.data.values[0],
                                                     qnaranja.data.values[0],
                                                     qamarilla.data.values[0],
                                                     qbajos.data.values[0],
                                                      ],
                                         name_main_values=['Altos', 'Roja', 'Naranja', 'Amarilla', 'Bajos'],
                                         color=[cnegro, crojo, cnaranja, camarillo, cmorado],
                                         last_sim = qlast_sim,
                                         nombre_variable='Caudal [m3/s]')

    # Built figure water level
    waterlevel_serie_plot = create_time_serie(station_code,
                                        obs=hObs_db,
                                        sen=hSen_db,
                                        ensamble_main=hforecMain,
                                        ensamble_full=hforecFull,
                                        ensamble_main_color=[crojo,
                                                             crojo,
                                                             cazulIdeam,
                                                             cazulIdeam,
                                                             cverdeIdeam],
                                        station_data=station_loc,

                                        main_values=[station_loc['umaxhis'].values[0],
                                                     station_loc['uroja'].values[0],
                                                     station_loc['unaranja'].values[0],
                                                     station_loc['uamarilla'].values[0],
                                                     station_loc['ubajos'].values[0]],
                                        name_main_values=['Altos', 'Roja', 'Naranja', 'Amarilla', 'Bajos'],
                                        color = [cnegro, crojo, cnaranja, camarillo, cmorado],
                                        last_sim = hlastsim,
                                        nombre_variable='Nivel [m]')


    # Build figure curva Gasto
    curva_gasto_plot = create_curva_gasto(full_curva_gasto = c_gasto_db,
                                          fun_curva_gasto  = c_gasto_fun)

    print('')
    print('')
    print('--- draw --- ')
    print('--- demora {0} seg ---'.format(time.time() - start_time))

    context = {'streamflow_serie_plot': streamflow_serie_plot,
               'waterlevel_serie_plot': waterlevel_serie_plot,
               'curva_gasto_plot': curva_gasto_plot}

    return render(request, 'cohyda/station_details.html', context)


@login_required()
def search_data(request):
    """
    Controllers for the search data page
    """

    context = {}
    return render(request, 'cohyda/search_data.html', context)