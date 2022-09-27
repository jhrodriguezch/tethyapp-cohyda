from tethys_sdk.base import TethysAppBase, url_map_maker
from tethys_sdk.app_settings import CustomSetting

"""
############################################################
                         app.py
############################################################
__author__  : jrc
__version__ : Beta 0.1
__obj__     : Application pyrhon file for tethys data
__date__    : 01 - sep - 2022
"""

class Cohyda(TethysAppBase):
    """
    Tethys app class for Colombian Hydraulic Data Analysis.
    """

    name = 'Colombian Hydraulic Data Analysis'
    index = 'cohyda:home'
    icon = 'cohyda/images/icon.gif'
    package = 'cohyda'
    root_url = 'cohyda'
    color = '#1368b2'
    description = 'Análisis de variables hidraúlicas para Colombia'
    tags = '"Hydraulic","Data Analysis"'
    enable_feedback = False
    feedback_emails = []


    def url_maps(self):
        """
        Relate url and controller in the URL three
        """
        UrlMap = url_map_maker(self.root_url)

        url_maps = (
            UrlMap(
                name       = 'home',
                url        = 'cohyda',
                controller = 'cohyda.controllers.home'
            ),
            UrlMap(
                name       = 'station_details',
                url        = 'cohyda/station_details/{station_code}',
                controller = 'cohyda.controllers.station_details'
            ),
            UrlMap(
                name       = 'search_data',
                url        = 'cohyda/search_data',
                controller = 'cohyda.controllers.search_data'
            ),
        )
        return url_maps


    def custom_settings(self):
        """
        Load custom data for the web application
        """
        custom_settings = (
            CustomSetting(
                name='Url_FEWS',
                type=CustomSetting.TYPE_STRING,
                description='URL de FEWS',
                required=True,
            ),
            CustomSetting(
                name='URL_HydroShare_Discharge',
                type=CustomSetting.TYPE_STRING,
                description='URL con datos de caudal historico',
                required=True,
            ),
            CustomSetting(
                name='URL_HydroShare_WaterLevel',
                type=CustomSetting.TYPE_STRING,
                description='Url con datos de nivel historico',
                required=True,
            ),
        )
        return custom_settings


