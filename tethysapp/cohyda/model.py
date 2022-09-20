import os
import io
import geojson
import urllib

import numpy as np
import pandas as pd
import geoglows as ggs
from datetime import datetime

from .auxFun import *


class Read_DataBase:
    def __init__(self, path_dir):
        """
        Load all directories for load data
        Input:
            path_dir : str -> General directory
        """
        # Directory with database
        self.path_dir = path_dir

        # Url for request the data
        self.url_station_loc   = 'http://fews.ideam.gov.co/colombia/data/ReporteTablaEstaciones.csv'
        self.fews_database     = 'http://fews.ideam.gov.co/colombia/'
        self.hidroshare_url_D  = 'https://www.hydroshare.org/resource/d222676fbd984a81911761ca1ba936bf/data/contents/Discharge_Data/'
        self.hydroshare_url_WL = 'https://www.hydroshare.org/resource/b0b5aaabd29c48bd8c5261c128f4762a/data/contents/'

        # Static directory for load the data
        self.dir_station_comid_rel = os.path.join(path_dir, 'public', 'csvs','comid_from_catchment_V1.csv' )
        self.dir_cgasto_list = os.path.join(path_dir, 'public', 'csvs', 'cgasto')
        self.dir_reach_geojson = os.path.join(path_dir, 'public', 'geojson', 'colombiaGeoglowsCatchments.geojson')

        # Review existence url, directorys and files
        # TODO: Revision code of links existence path and url status less than 400


    def __call__(self):
        """
        Load general data (all Colombia data)
        """
        # ---------------- Read: Stations ----------------
        # Read data
        url = self.url_station_loc
        stat_loc = urllib.request.urlopen(url).read()
        stat_loc = pd.read_csv(io.StringIO(stat_loc.decode('latin-1')))

        # Fix data frame
        # Add id_alert identification
        stat_loc['id_alert'] = 1
        # Obtain df for alert type
        # id_alert = 6
        valids_umax      = stat_loc['ultimonivelsen'] > stat_loc['umaxhis']
        # id_alert = 5
        valids_uroja     = (stat_loc['ultimonivelsen'] < stat_loc['umaxhis']) & (stat_loc['ultimonivelsen'] > stat_loc['uroja'])
        # id_alert = 4
        valids_uamarilla = (stat_loc['ultimonivelsen'] < stat_loc['uroja']) & (stat_loc['ultimonivelsen'] > stat_loc['unaranja'])
        # id_alert = 3
        valids_unaranja  = (stat_loc['ultimonivelsen'] < stat_loc['unaranja']) & (stat_loc['ultimonivelsen'] > stat_loc['uamarilla'])
        # id_alert = 2
        valids_umin      = stat_loc['ultimonivelsen'] < stat_loc['ubajos']

        stat_loc.loc[valids_umax, 'id_alert']      = 6
        stat_loc.loc[valids_uroja, 'id_alert']     = 5
        stat_loc.loc[valids_uamarilla, 'id_alert'] = 4
        stat_loc.loc[valids_unaranja, 'id_alert']  = 3
        stat_loc.loc[valids_umin, 'id_alert']      = 2

        # ---------------- Read : Comid from stations ----------------
        # Read data
        df_comid_station = pd.read_csv(self.dir_station_comid_rel, index_col=0)
        # Fix data frame
        df_comid_station['id'] = df_comid_station['id'].astype(int)

        # ---------------- Merge data from stations ----------------
        # var_station = ['id', 'COMID', 'corriente', 'subzona', 'zona', 'cenpoblado', 'municipio', 'depart']
        var_station = ['id', 'COMID','altitud','DEPARTAMEN',
                       'MUNICIPIO','AREA_OPERA','AREA_HIDRO','ZONA_HIDRO',
                       'CORRIENTE','SUBZONA_HI']
        stat_loc = stat_loc.merge(df_comid_station[var_station],
                                                    how='left',
                                                    left_on='id',
                                                    right_on='id')

        # ---------------- Save as object features ----------------
        self.station_loc_df = stat_loc[['id', 'nombre', 'lat', 'lng', 'id_alert']].copy()
        self.stationloc = stat_loc.copy()

        # ---------------- End message ----------------
        print('Última recarga de la base da datos {0}'.format(datetime.now()))


    # Get methods - All Colombia
    def get_reachgeojson(self):
        """
        Get geojson with the reach
        """
        with open(self.dir_reach_geojson) as f:
            rv = geojson.load(f)
        return rv


    def get_station_data(self):
        """
        Get all station data
        """
        return self.stationloc


    def get_station_location(self):
        """
        Get essential station data
        """
        return self.station_loc_df


    # Get methods for reach or station
    def get_forecastdata(self, station_data, hist_obs, hist_sim):
        """
        Obtain forecast data.
        Input:
            station_data : pandas.DataFrame -> Station identification data.
            hist_obs     : pandas.DataFrame -> Time series observated data.
            hist_sim     : pandas.DataFrame -> Time series simulated data.\n
        Return:
            ensemble_main_forecast : pandas.DataFrame -> Main results of the forecast.
            ensemble_forecast      : pandas.DataFrame -> All results of the forecast.\n
        """
        station_id = station_data['id']
        try:
            reach_id = int(station_data['COMID'])
        except:
            print('Estación {0} no tiene COMID.'.format(station_id))
            reach_id = None

        return self.__forecastDataFixed__(hist_obs= hist_obs,
                                          reach_id = reach_id,
                                          hist_sim= hist_sim)


    def get_fewsdata(self, stationID, typeData):
        """
        Get data from the FEWS database.
        Input:
            stationID : str -> Station identification code
            typeData  : str -> Type to data for load. Q for stream flow or H for water level.
        Return:
            data    : pandas.DataFrame -> Observed time series
            dataSen : pandas.DataFrame -> Sensor time series
            dataP   : pandas.DataFrame -> Precipitation time series
        """
        return self.__FewsData__(stationID, typeData)


    def getHistoricalData(self, stationID, type_data):
        """
        Get data from the HYDROSHARE database.
        Input:
            stationID : str -> Station identification code
            typeData  : str -> Type to data for load. Q for stream flow or H for water level.
        Return:
            rv        : pandas.DataFrame -> Time series data
        """

        if 'Q' == type_data:

            fix_units = 1
            url_download = self.hidroshare_url_D + stationID + '.csv'

            if 400 <= requests.get(url_download).status_code:
                print('No hay datos historicos de caudal para la estacion : {0}'.format(stationID))
                return pd.DataFrame(data={'data': [float('nan')]}, index=[np.datetime64('NaT')])

        elif 'H' == type_data:

            fix_units = 1/100.
            url_download = self.hydroshare_url_WL + stationID + '.csv'

            if 400 <= requests.get(url_download).status_code:
                print('No hay datos de nivel historicos para la estacion : {0}'.format(stationID))
                return pd.DataFrame(data={'data': [float('nan')]}, index=[np.datetime64('NaT')])

        else:

            return pd.DataFrame(data={'data':[float('nan')]}, index=[np.datetime64('NaT')])

        # Load hydroshare database
        rv = hydroShare2df(urlDir=url_download)

        # Fix units
        rv.data = rv.data * fix_units

        return rv


    def get_curvagasto(self, stationID):
        """
        Get rate curve data from the public folder.
        Input:
            stationID : str -> Station identification code
        Return:
            rv : pandas.DataFrame -> Rate curve dataframe
        """

        # Read available rate curve
        list_files = [[ii.split('_')[0], os.path.join(self.dir_cgasto_list, ii)] for ii in os.listdir(self.dir_cgasto_list)]
        list_files = pd.DataFrame(list_files)
        list_files.columns = ['id', 'path']
        list_files['id'] = list_files['id'].astype(str)

        # Read curve data
        if stationID in list_files['id'].tolist():
            dir_path_cgasto = list_files.loc[list_files['id'] == stationID, 'path'].values[0]
            return self.__extractCurvasGasto__(dirFile=dir_path_cgasto)
        else:
            # TODO: add option to requests rate curve from dhime or other database
            print('Rate curve does not exists.')
            return pd.DataFrame(data={'id':[None],
                                      'NOMBRE':['NA'],
                                      'NO.':['NA'],
                                      'F. INICIAL':[np.datetime64('NaT')],
                                      'F. FINAL':[np.datetime64('NaT')],
                                      'NIVEL APRO':['NA'],
                                      'NIVEL':[float('nan')],
                                      'CAUDAL':[float('nan')]},
                                )


    def get_curvagasto_fun(self, cgasto_data):
        """
        Get best function for the rate curve data
        Input:
            cgasto_data : pandas.DataFrame -> Data of the curve rate
        Return:
            dict_rv     : dict             -> Best interpolation function
        """

        dict_rv = {'id':cgasto_data['id'].unique()[0]}
        try:
            for no_unique in cgasto_data['NO.'].unique():

                # Read data
                q = cgasto_data.loc[cgasto_data['NO.'] == no_unique, 'CAUDAL'].to_numpy()
                n = cgasto_data.loc[cgasto_data['NO.'] == no_unique, 'NIVEL'].to_numpy()

                # Get interpolation for Q vs N
                rv_lineal    = getInterpolation(x=q, y=n, typeInterpol='lineal')    # n = a * q + b
                rv_quadratic = getInterpolation(x=q, y=n, typeInterpol='quadratic') # n = a * q ** 2 + b * q + c
                rv_potential = getInterpolation(x=q, y=n, typeInterpol='potential') # n = a * (q + b)**c

                # Get interpolation for N vs Q
                rvi_lineal    = getInterpolation(x=n, y=q, typeInterpol='lineal')    # n = a * q +b
                rvi_quadratic = getInterpolation(x=n, y=q, typeInterpol='quadratic') # n = a * q ** 2 + b * q + c
                rvi_potential = getInterpolation(x=n, y=q, typeInterpol='potential') # n = a * (q + b)**c

                # Build dictionary with best interpolation
                tipo_var, para_var, metr_var = self.__bestInterpolation__(int1=rv_lineal,
                                                                          int2=rv_quadratic,
                                                                          int3=rv_potential)

                tipo_var_i, para_var_i, metr_var_i = self.__bestInterpolation__(int1=rvi_lineal,
                                                                          int2=rvi_quadratic,
                                                                          int3=rvi_potential)

                dict_rv.update({'NO.' + no_unique :{'id' : float(no_unique),
                                       'cant data' : len(q),
                                       'Q_H' : {
                                           'tipo' : tipo_var,
                                           'parametros' : para_var,
                                           'metricas': metr_var,
                                       },
                                       'H_Q' : {
                                           'tipo': tipo_var_i,
                                           'parametros': para_var_i,
                                            'metricas': metr_var_i,
                                       },
                                       }})

            return dict_rv
        except Exception as e:
            print('No se encontraron los parametros para la funcion de la curva de gasto.')
            print(e)
            print('')

            dict_rv.update({'NO.':{'id' : float('nan'),
                                   'cant data' :        float('nan'),
                                   'H_Q' : {
                                           'tipo' : '',
                                           'parametros' : [],
                                           'metricas': [],
                                       },
                                       'Q_H' : {
                                           'tipo': '',
                                           'parametros': [],
                                           'metricas': [],
                                       },
                                   }})
            return dict_rv


    @staticmethod
    def get_historical_simulation(station_data):
        """
        Get historical simulation for ECMWF Streamflow
        Input:
            station_data : pandas.DataFrame -> Station identification data.
        Return:
            rv           : pandas.DataFrame -> Time series
        """
        try:
            reach_id = int(station_data['COMID'])
            return changeGGLOWSColNames(ggs.streamflow.historic_simulation(reach_id))
        except Exception as e:
            print('Estación {0} no tiene COMID asociado.'.format(station_id))
            print(e)
            print('')

            return pd.DataFrame(data={'data': None},
                                index=np.datetime64('NaT'), )


    @staticmethod
    def get_forecast_records(station_data, simData, obsData):
        """
        Get last simulated time series
        Input:
            station_data : pandas.DataFrame -> Station identification data.
            simData      : pandas.DataFrame -> Time series simulation data.
            obsData      : pandas.DataFrame -> Time series observed data.
        Return:
            rv           : pandas.DataFrame -> Time series historic simulated data.
        """
        try:
            reach_id = station_data['COMID'].astype(int).values[0]
            print('Reach COMID : {0}'.format(reach_id))
        except Exception as e:
            print('Estación {0} no tiene COMID.'.format(station_data['id'].values[0]))
            print(e)
            print('')

            return pd.DataFrame(data={'data': None},
                                index=[np.datetime64('NaT')])

        rv = changeGGLOWSColNames(ggs.streamflow.forecast_records(reach_id))

        rv = getBiasCorrection(input=rv,
                               simData=simData,
                               obsData=obsData)
        return rv




    # Hiden methods
    @staticmethod
    def __bestInterpolation__(int1, int2, int3):
        """
        Select best interpolation from r2 parameter.
            int1 -> Lineal interpolation
            int2 -> Quadratic interpolation
            int3 -> Potential interpolation
        """

        best_r2_interpol = max(int1[1][1], int2[1][1], int3[1][1])

        if best_r2_interpol == int1[1][1]:
            return 'lineal', int1[0], int1[1]
        elif best_r2_interpol == int2[1][1]:
            return 'quadratic', int2[0], int2[1]
        else:
            return 'potential', int3[0], int3[1]


    @staticmethod
    def __extractCurvasGasto__(dirFile,
                               strToSplitDf="CÓDIGO",
                               headerLenght=10,
                               dateColumns=['F. INICIAL', 'F. FINAL'],
                               formatDate="%d/%m/%Y"):
        """
        Extract rate curve from file.
            dirFile     -> Directory file
            strToSplitDf -> Word for split the rate curve into observations
            headerLenght -> Header length
            dateColumns  -> Date columns
            formatDate   -> Format date for dateColumns
        """

        # read dataframe
        df = pd.read_excel(dirFile,
                           skiprows=range(headerLenght),
                           header=None)

        # Fix dataframe
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # row where the data will be splited
        splitRows = list(df[df[1] == strToSplitDf].index)

        # build result dataframe
        df_res = pd.DataFrame()
        for row, ii in enumerate(splitRows[:-1]):

            # extract principal identification parameters
            names_gnrl = list(df.loc[ii].dropna())
            data_gnrl = list(df.loc[ii + 1].dropna())
            names_esp = list(df.loc[ii + 2].dropna())

            # extract data of the dataframe
            data_col = df.loc[ii + 3: splitRows[row + 1] - 1].copy()
            data_col.reset_index(inplace=True, drop=True)
            data_col.dropna(axis=1, how='all', inplace=True)

            # merge in the temporal dataframe
            df_tmp = pd.DataFrame()
            for namesCol, dataCol in list(zip(names_gnrl, data_gnrl)):
                df_tmp[namesCol] = [dataCol] * len(data_col)

            for num, nameCol in enumerate(names_esp):
                df_tmp[nameCol] = data_col.iloc[:, num]

            # add to result dataframe
            df_res = pd.concat([df_res, df_tmp], ignore_index=True)

            del df_tmp

        # build date columns
        for dateCol in dateColumns:
            df_res[dateCol] = pd.to_datetime(df_res[dateCol], format=formatDate)

        df_res.rename(columns={strToSplitDf: 'id'}, inplace=True)

        # Fix type and units
        df_res['id'] = df_res['id'].astype(str)
        df_res['NIVEL'] = df_res['NIVEL'] * (1/100.)

        return df_res


    @staticmethod
    def __forecastDataFixed__(reach_id, hist_obs, hist_sim):
        """
        Fix forecast time series
            reach_id -> COMID of reach
            hist_obs -> Historic time series observed
            hist_sim -> Historic time series simulated
        """
        # -- Get database
        if None != reach_id:
            # Load forecast ensemble
            ensemble_forecast = changeGGLOWSColNames(ggs.streamflow.forecast_ensembles(reach_id))

            if hist_obs.shape[0] == 1:
                print('No fue posible ejecutarse el método de bias correction.')
            else:
                # Fix with bias correction
                for ens_col in ensemble_forecast.columns:
                    ensemble_forecast[ens_col] = getBiasCorrection(ensemble_forecast[ens_col],
                                                                   hist_sim,
                                                                   hist_obs)

            # Built dataframe with main values of forecast
            ens_forecast_fix = pd.DataFrame()
            ens_forecast_fix['Maximo pronostico'] = ensemble_forecast.max(axis=1)
            ens_forecast_fix['Minimo pronostico'] = ensemble_forecast.min(axis=1)
            ens_forecast_fix['P25 pronostico'] = ensemble_forecast.quantile(0.25, axis=1)
            ens_forecast_fix['P75 pronostico'] = ensemble_forecast.quantile(0.75, axis=1)
            ens_forecast_fix['Promedio pronostico'] = ensemble_forecast.quantile(0.5, axis=1)

            return ens_forecast_fix, ensemble_forecast

        else:
            print('reach_id : {0} no encontrado.'.format(reach_id))

            ens_forecast_fix  = pd.DataFrame(data={'Maximo pronostico': [float('nan')],
                                                   'Minimo pronostico': [float('nan')],
                                                   'P25 pronostico': [float('nan')],
                                                   'P75 pronostico': [float('nan')]},
                                             index=[np.datetime64('NaT')])

            ensemble_forecast = pd.DataFrame(data={'ens': [float('nan')]},
                                             index=[np.datetime64('NaT')])

            return ens_forecast_fix, ensemble_forecast


    def __FewsData__(self, stationID: str, typeData: str):
        """
        Obtain data for fews web server
        Input:
            - stationID : str -> station code.
            - typeData  : str -> Type of data to extract. Shoud be H or Q; H for water depth
                                 and Q for charge.
        return:
            data    : DataFrame -> Data observed. obs.data in json
            dataSen : DataFrame -> Data sensor. obs.sen in json
            dataP   : DataFrame -> Data precipitation. obs.prec in json

        Test results:
            - 24/08/2022
            - Time reading data + dataframe construct:
                //////////////////////////////////////////////////////////////////////////
                Test 1
                Test for function : getFewsData
                Elapsed time in 50 run test = 0.03583229064941406 seconds
                //////////////////////////////////////////////////////////////////////////
            - Time reading data + dataframe construct + dataFrame fix:
                //////////////////////////////////////////////////////////////////////////
                Test 1
                Test for function : getFewsData
                Elapsed time in 50 run test = 0.04199528694152832 seconds
                //////////////////////////////////////////////////////////////////////////
            26 AGO
            //////////////////////////////////////////////////////////////////////////
            Test 1
            Test for function : getFewsData
            Elapsed time in 50 run test = 0.040893268585205075 seconds
            //////////////////////////////////////////////////////////////////////////
        """
        assert type(stationID) == str, 'StationID should be a string format (str).'
        assert type(typeData) == str, 'typeData should be a string format (str).'
        assert typeData.upper() in ["H", "Q"], 'typeData only should be H or Q, not {}'.format(typeData)

        # Build url
        # url_dir = self.staticDataBase['FEWS']['URL'] + '/json' + typeData + '/'
        url_dir = self.fews_database + '/json' + typeData + '/'
        stationFile = '00' + stationID + typeData + 'obs.json'

        # TODO: Add conditional id station id does not exist
        return jfews2df(url_dir + stationFile)


# Main functions
def historical_timeserie(original, alternative, change_fun, var_type):
    """
    Select a time series from all time series possible
    Input:
        original    : list             -> [hydroshare df, fews df]
        alternative : list             -> [hydroshare df alternative, fews df alternative]
        change_fun  : pandas.DataFrame -> Rate curve associate
        var_type    : str              -> Type of data for identify the change.
    """
    # TODO: Change "Guard Clauses" in the if for best code understanding.
    # Identify last curva de gasto
    change = last_curvagasto(change_fun)

    if len(original[0].dropna(axis=0)) > len(original[1].dropna(axis=0)):
        print('Datos historicos de hydroshare.')
        return original[0].dropna(axis=0)
    else:
        if len(original[1].dropna(axis=0)) > 0:
            print('Datos historicos de FEWS.')
            return original[1].dropna(axis=0)
        else:
            if not np.isnan(change['id']):
                print('Curva de gasto cargada.')
                if len(alternative[0].dropna(axis=0)) > 0:
                    print('Datos historicos calculador por alternativos hydroshare.')
                    if var_type == 'Q':
                        return get_Q2H_H2Q(alternative[0].dropna(axis=0),
                                           change_fun=change_fun,
                                           typechange='H_Q')
                    else:
                        return get_Q2H_H2Q(original_ts=alternative[0].dropna(axis=0),
                                           change_fun=change_fun,
                                           typechange='Q_H')
                else:
                    if len(alternative[1].dropna(axis=0)) > 0:
                        print('Datos historicos calculador por alternativos FEWS.')
                        if var_type == 'Q':
                            return get_Q2H_H2Q(original_ts=alternative[1].dropna(axis=0),
                                               change_fun=change_fun,
                                               typechange='H_Q')
                        else:
                            return get_Q2H_H2Q(original_ts=alternative[1].dropna(axis=0),
                                               change_fun=change_fun,
                                               typechange='Q_H')
                    else:
                        # TODO: add static lecture of data or something
                        print('vamo a acer argo')
                        return 0
            else:
                print('Curva de gasto no cargada.')
                print('Datos historicos de hydroshare.')
                return original[0]


def last_curvagasto(cgasto):
    """
    Extract last rate curve observed
    Input:
        cgasto : pandas.DataFrame -> Rate curve database.
    """
    last_cgasto = [ii for ii in cgasto.keys() if 'NO' in ii][-1]
    return cgasto[last_cgasto]


def get_Q2H_H2Q(original_ts, change_fun, typechange):
    """
    Transform time series.
    Q2H form stream flow to water level
    H2Q from water level to stream flow
    Input:
        original_ts : pandas.DataFrame -> Time series.
        change_fun  : dict             -> Function of rate curve.
        typechange  : str              -> Type of change (Q2H or H2Q)
    Return:
        rv         : pandas.DataFrame -> Time series result
        type       : str              -> Type of interpolation used
        parameters : list             -> Parameters of the interpolation
    """
    tmp_ts = original_ts.copy()
    last_fun_cgasto = last_curvagasto(change_fun)

    # Transformation type function
    if typechange == 'Q2H':
        fun_trans_gasto = last_fun_cgasto['Q_H']
    else: # typechange == 'H2Q'
        fun_trans_gasto = last_fun_cgasto['H_Q']


    for col_name in tmp_ts.columns:
        # Transformation
        if fun_trans_gasto['tipo'] == 'lineal':
            tmp_ts[col_name] = fun_trans_gasto['parametros'][0] * tmp_ts[col_name] \
                               + fun_trans_gasto['parametros'][1]
        elif fun_trans_gasto['tipo'] == 'quadratic':
            tmp_ts[col_name] = fun_trans_gasto['parametros'][0] * tmp_ts[col_name] ** 2 \
                               + fun_trans_gasto['parametros'][1] * tmp_ts[col_name] \
                               + fun_trans_gasto['parametros'][2]
        elif fun_trans_gasto['tipo'] == 'potential':
            tmp_ts[col_name] = fun_trans_gasto['parametros'][0] \
                               * ( tmp_ts[col_name] + fun_trans_gasto['parametros'][1] ) \
                               ** fun_trans_gasto['parametros'][2]
        else:
            print('No interpolation')
            tmp_ts[col_name] = [float('nan')] * len(tmp_ts)


    return tmp_ts, fun_trans_gasto['tipo'], fun_trans_gasto['parametros']


def data2dfserie(data):
    """
    Build time series dataframe format from data
    Input:
        data : int/float -> Data
    Return:
        rv   : pandas.DataFrame -> Time series
    """
    return pd.DataFrame(data={'data': [data]},
                        index=[0])