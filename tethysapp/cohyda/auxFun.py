import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from scipy.optimize import curve_fit

import sys

import io
import json
import requests
import geoglows as ggs
from datetime import datetime
from urllib.request import urlopen

"""
############################################################
                      auxFun.py
############################################################
__author__  : jrc
__version__ : Beta 0.1
__obj__     : Additional functions for model.py in tethys app (mathematical analysis).
__date__    : 01 - sep - 2022

__functions__ : jfews2df
                changeGGLOWSColNames
                hydroShare2df
                getBiasCorrection
                getInterpolation
                
"""

def jfews2df(url_dir):
    """
    Load data from FEWS
    """
    # Load json
    response = urlopen(url_dir)
    response = json.loads(response.read())

    # Change json to dataframe data observed
    data    = pd.DataFrame(response['obs']['data'],
                            columns=['date', 'data'])
    data.index = pd.to_datetime(data['date'], format='%Y/%m/%d %H:%M')
    data.drop(['date'], axis=1, inplace=True)

    # Change json to dataframe data sensor
    dataSen = pd.DataFrame(response['sen']['data'],
                            columns=['date', 'data'])
    dataSen.index = pd.to_datetime(dataSen['date'], format='%Y/%m/%d %H:%M')
    dataSen.drop(['date'], axis=1, inplace=True)

    # Change json to dataframe precipitation observed
    dataP    = pd.DataFrame(response['prec']['data'],
                            columns=['date', 'data'])
    dataP.index = pd.to_datetime(dataP['date'], format='%Y/%m/%d %H:%M')
    dataP.drop(['date'], axis=1, inplace=True)

    # Transform to daily datatime
    try:
        data = data.groupby([data.index.date]).mean()
        data.index = pd.to_datetime(data.index, format='%Y/%m/%d')
    except Exception as e:
        print(e)
        print('data observer from fews is empty.')
        print('')

    try:
        dataSen = dataSen.groupby([dataSen.index.date]).mean()
        dataSen.index = pd.to_datetime(dataSen.index, format='%Y/%m/%d')
    except Exception as e:
        print(e)
        print('data sensor from fews is empty.')
        print('')

    try:
        dataP = dataP.groupby([dataP.index.date]).sum()
        dataP.index = pd.to_datetime(dataP.index, format='%Y/%m/%d')
    except Exception as e:
        print(e)
        print('data precipitation from fews is empty.')
        print('')

    return data, dataSen, dataP


def changeGGLOWSColNames(df):
    """
    Change geoglows columns names
    Input:
        df : pandas.DataFrame -> Time series
    Return:
        df : pandas.DataFrame -> Time series rename
    """
    # Rename data axis
    df.rename_axis('date', axis=0, inplace=True)
    df.index = df.index.tz_localize(None)

    if len(df.columns) == 1:
        # Rename column names
        df.rename(columns={df.columns[0]: 'data'}, inplace=True)

    return df


def hydroShare2df(urlDir, dateColName='Datetime', formatDate='%Y-%m-%d'):
    """
    Load hydroshare data
    Input:
        urlDir      : str -> URL
        dateColName : str -> Column name of the date time column
        formatDate  : str -> Format of the datetime for the datetime column
    Return:
         rv : pandas.DataFrame -> Time series
    """
    # Read data as dataframe
    response = requests.get(urlDir, verify=True).content
    rv = pd.read_csv(io.StringIO(response.decode('utf-8')),
                     parse_dates=[dateColName],
                     date_parser=lambda x: datetime.strptime(x, formatDate),
                     index_col=0)\
                    .rename_axis('date')

    # Rename column
    rv.rename(columns={rv.columns[0]: 'data'}, inplace=True)
    return rv


def getBiasCorrection(input, simData, obsData):
    """
    Get bias correction
    Input:
        input   : pandas.DataFrame -> Time series
        simData : pandas.DataFrame -> Time series
        obsData : pandas.DataFrame -> Time series
    Return:
        rv : pandas.DataFrame      -> Time series
    """

    if type(input) != type(pd.DataFrame()):
        # warnings.warn('Pandas library does not define a
        # pandas. Serie as dtype Series.object in the 1.4.3 version.')
        input = input.to_frame()



    try:
        rv = ggs.bias.correct_forecast(forecast_data  = input,
                                       simulated_data = simData,
                                       observed_data  = obsData)
        rv = changeGGLOWSColNames(rv)
    except Exception as e:
        print(e)
        print('No se aplicó la corrección por sesgo - Pronostico.')
        print('')

        rv = input

    return rv


def getInterpolation(x: np.array, y: np.array, typeInterpol):
    """
    Get interpolation
    Input:
        x            : numpy.array -> Independent variable
        y            : numpy.array -> Dependent variable
        typeInterpol : str         -> Type of interpolation to do
    Return:
        para   : list -> Parameter of the interpolation
        metric : list -> Metric results [root-mean-square error, r2 pearson correlation]

    """

    # Remove nan in both data arrays x and y
    tmp_df = pd.DataFrame()
    tmp_df['x'] = x
    tmp_df['y'] = y
    tmp_df.dropna(axis=0, inplace=True)
    x = tmp_df['x'].to_numpy()
    y = tmp_df['y'].to_numpy()

    # Try Lineal interpolation
    if typeInterpol == 'lineal':
        # Interpolation
        try:
            MatPara, _ = curve_fit(lambda x_i, a, b: a * x_i + b , x, y)
        except Exception as e:
            print('')
            print('Error in lineal interpolation')
            print(e)
            print('')
            MatPara = [0, 0]

        y_sim = MatPara[0] * x + MatPara[1]

        # Metrics
        mse = metrics.mean_squared_error(y, y_sim)
        rmse = np.sqrt(mse)  # or mse**(0.5)
        r2 = metrics.r2_score(y, y_sim)

        return MatPara, [rmse, r2]

    # Try quadratic interpolation
    elif typeInterpol == 'quadratic':
        # Interpolacion
        try:
            MatPara, _ = curve_fit(lambda x_i, a, b, c: a * x_i ** 2 + b * x_i + c, x, y)
        except Exception as e:
            print('')
            print('Error in quadratic interpolation')
            print(e)
            print('')
            MatPara = [0, 0, 0]

        y_sim = MatPara[0] * x ** 2 + MatPara[1] * x + MatPara[2]

        # Metrics
        mse = metrics.mean_squared_error(y, y_sim)
        rmse = np.sqrt(mse)  # or mse**(0.5)
        r2 = metrics.r2_score(y, y_sim)

        return MatPara, [rmse, r2]

    # Try potential interpolation
    elif typeInterpol == 'potential':
        # Interpolacion
        try:
            MatPara, _ = curve_fit(lambda x_i, a, b, c: a * (x_i + b) ** c , x, y)
        except Exception as e:
            print('')
            print('Error in potential interpolation')
            print(e)
            print('')

            MatPara = [0, 0, 0]

        y_sim = MatPara[0] * (x + MatPara[1]) ** MatPara[2]

        # Metrics
        mse = metrics.mean_squared_error(y, y_sim)
        rmse = np.sqrt(mse)# or mse**(0.5)
        r2 = metrics.r2_score(y, y_sim)

        return MatPara, [rmse, r2]

    else:
        # Add new interpolation methods
        return [], []


def change_UTC_colombia(input_ts,
                        format_dt = None,
                        dt_colname = None,
                        ):
    """
    __date__ : 20 - sep
    Obj: Change the UTC for Colombian data series
    Input:
        Input_ts   : pandas.DataFrame      = Time series original
        dt_colname : str                   = Name of the column with date time
        format_dt  : str (datetime format) = Format of the column with date time
    Return:
        df         : pandas.DataFrame      = Time series fixed
    """
    # Copy datadframe
    df = input_ts.copy()

    if dt_colname == None:
        # Asserts

        # Main
        df.index = df.index - pd.Timedelta(hours=5)
    else:

        # Asserts
        if not dt_colname in input_ts.columns:
            print('Does not UTC changed.')
            return df

        # Main
        df[dt_colname] = df[dt_colname] - pd.Timedelta(hours=5)

    # Return data
    return df


def getBiasCorrectionHistorical(simData, obsData):
    """
    Get bias correction from historical data
    Input:
        simData : pandas.DataFrame -> Time series
        obsData : pandas.DataFrame -> Time series
    Return:
        rv : pandas.DataFrame      -> Time series
    """

    try:
        print(simData)
        print(obsData)
        rv = ggs.bias.correct_historical(simulated_data = simData,
                                         observed_data = obsData)
        rv = changeGGLOWSColNames(rv)
    except Exception as e:
        print(e)
        print('No se aplicó la corrección por sesgo. - Historica')
        print('')

        rv = simData

    return rv