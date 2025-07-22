"""
Clase: EstadisticasBasicasDatos

Objetivo: Clase enfocada a la codificacion de la data para el EDA basico

1. Creacion de clase pmarin 15-07-2024
"""
import pandas as pd
import io
import logging
from typing import Dict, List, Optional, Union, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EstadisticasBasicasDatos:
    def __init__(self, df):
        self.df = df

    def obtener_info_basico(self):
        """
        Obtiene información básica del DataFrame

        Returns:
            dict: Diccionario con información básica

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener la información
        """
        try:
            memoria_mb = self.df.memory_usage(deep=True).sum() / 1024 ** 2

            return {
                'filas': self.df.shape[0],
                'columnas': self.df.shape[1],
                'memoria_mb': round(memoria_mb, 2)
            }

        except MemoryError:
            logger.error("Error de memoria al calcular el uso de memoria del DataFrame")
            return {
                'filas': 0,
                'columnas': 0,
                'memoria_mb': 0
            }
        except Exception as e:
            logger.error(f"Error al obtener información básica: {str(e)}")
            return {
                'filas':0,
                'columnas': 0,
                'memoria_mb': 0
            }

    def obtener_primeras_filas(self, n: int = 5):
        """
        Obtiene las primeras n filas del DataFrame

        Args:
            n (int): Número de filas a obtener

        Returns:
            pd.DataFrame: Primeras n filas

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener las filas
        """
        try:
            if n <= 0:
                logger.error("El número de filas debe ser positivo")

            if n > len(self.df):
                logger.warning(f"Se solicitaron {n} filas pero el DataFrame solo tiene {len(self.df)} filas")
                n = len(self.df)

            return self.df.head(n)

        except Exception as e:
            logger.error(f"Error al obtener las primeras {n} filas: {str(e)}")
            return None

    def obtener_ultimas_filas(self, n: int = 5):
        """
        Obtiene las últimas n filas del DataFrame

        Args:
            n (int): Número de filas a obtener

        Returns:
            pd.DataFrame: Últimas n filas

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener las filas
        """
        try:
            if n <= 0:
                logger.error("El número de filas debe ser positivo")

            if n > len(self.df):
                logger.warning(f"Se solicitaron {n} filas pero el DataFrame solo tiene {len(self.df)} filas")
                n = len(self.df)

            return self.df.tail(n)

        except Exception as e:
            logger.error(f"Error al obtener las últimas {n} filas: {str(e)}")
            return None

    def obtener_tipo_datos(self):
        """
        Obtiene información detallada sobre tipos de datos

        Returns:
            dict: Diccionario con información de tipos de datos

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener la información
        """
        try:
            # Capturar información detallada
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            info_str = buffer.getvalue()

            # Resumen de tipos de datos
            tipos_resumen = self.df.dtypes.value_counts().to_dict()

            return {
                'info_detallada': info_str,
                'tipos_resumen': tipos_resumen
            }

        except Exception as e:
            logger.error(f"Error al obtener información de tipos de datos: {str(e)}")
            return {
                'info_detallada': "NA",
                'tipos_resumen': "NA"
            }

    def obtener_info_columnas(self):
        """
        Obtiene información detallada de las columnas

        Returns:
            pd.DataFrame: DataFrame con información de columnas

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener la información
        """
        try:
            columnas = self.df.columns.tolist()

            info_columnas_data = []

            for col in columnas:
                try:
                    valores_nulos = self.df[col].isnull().sum()
                    valores_unicos = self.df[col].nunique()
                    porcentaje_nulos = round((valores_nulos / len(self.df)) * 100, 2)

                    info_columnas_data.append({
                        'Columna': col,
                        'Tipo': str(self.df[col].dtype),
                        'Valores_nulos': valores_nulos,
                        'Valores_únicos': valores_unicos,
                        'Porcentaje_nulos': porcentaje_nulos
                    })

                except Exception as e:
                    logger.warning(f"Error al procesar columna '{col}': {str(e)}")
                    info_columnas_data.append({
                        'Columna': col,
                        'Tipo': 'Error',
                        'Valores_nulos': 'N/A',
                        'Valores_únicos': 'N/A',
                        'Porcentaje_nulos': 'N/A'
                    })

            return pd.DataFrame(info_columnas_data)

        except Exception as e:
            logger.error(f"Error al obtener información de columnas: {str(e)}")


    def obtener_nombres_columnas(self):
        """
        Obtiene la lista de nombres de columnas

        Returns:
            list: Lista con nombres de columnas

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener los nombres
        """
        try:
            return self.df.columns.tolist()

        except Exception as e:
            logger.error(f"Error al obtener nombres de columnas: {str(e)}")


    def obtener_resumen_estadistico(self):
        """
        Obtiene estadísticas resumidas del DataFrame

        Returns:
            pd.DataFrame: Estadísticas descriptivas o None si hay error

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener las estadísticas
        """
        try:
            return self.df.describe(include='all')

        except Exception as e:
            logger.error(f"Error al obtener estadísticas descriptivas: {str(e)}")


    def obetener_resumen_valores_faltantes(self):
        """
        Obtiene resumen de valores faltantes

        Returns:
            pd.DataFrame: Resumen de valores faltantes

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener el resumen
        """
        try:
            missing_data_list = []

            for col in self.df.columns:
                try:
                    valores_faltantes = self.df[col].isnull().sum()
                    porcentaje = round((valores_faltantes / len(self.df)) * 100, 2)

                    missing_data_list.append({
                        'Columna': col,
                        'Valores_faltantes': valores_faltantes,
                        'Porcentaje': porcentaje
                    })

                except Exception as e:
                    logger.warning(f"Error al procesar valores faltantes en columna '{col}': {str(e)}")
                    missing_data_list.append({
                        'Columna': col,
                        'Valores_faltantes': 'Error',
                        'Porcentaje': 'Error'
                    })

            missing_df = pd.DataFrame(missing_data_list)
            return missing_df.sort_values('Valores_faltantes', ascending=False)

        except Exception as e:
            logger.error(f"Error al obtener resumen de valores faltantes: {str(e)}")


    def obtener_analisis_completo(self):
        """
        Obtiene análisis completo del DataFrame

        Returns:
            dict: Diccionario con todo el análisis

        Raises:
            DataFrameAnalyzerError: Si ocurre un error crítico
        """
        analysis = {}
        errors = []

        # Ejecutar cada análisis y capturar errores
        analysis_methods = [
            ('info_basica', self.obtener_info_basico()),
            ('primeras_filas', lambda: self.obtener_primeras_filas()),
            ('ultimas_filas', lambda: self.obtener_ultimas_filas()),
            ('tipos_datos', self.obtener_tipo_datos()),
            ('info_columnas', self.obtener_info_columnas()),
            ('nombres_columnas', self.obtener_nombres_columnas()),
            ('estadisticas', self.obtener_resumen_estadistico()),
            ('valores_faltantes', self.obetener_resumen_valores_faltantes())
        ]

        for key, method in analysis_methods:
            try:
                analysis[key] = method()
            except Exception as e:
                logger.error(f"Error en análisis '{key}': {str(e)}")
                errors.append(f"{key}: {str(e)}")
                analysis[key] = None

        # Agregar información de errores si los hay
        if errors:
            analysis['errores'] = errors
            logger.warning(f"Análisis completado con {len(errors)} errores")

        return analysis


