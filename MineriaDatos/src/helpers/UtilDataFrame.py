"""
Clase: UtilDataframe

Objetivo: Utilidades para manipulación de DataFrames

Cambios:
    1. Creacion de la clase
"""
import numpy as np
import pandas as pd
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UtilDataFrame:

    @staticmethod
    def convertir_tipos_json_serializable(df: pd.DataFrame) -> pd.DataFrame:
        """Convierte tipos problemáticos a tipos JSON-serializables"""
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("El parámetro debe ser un DataFrame de pandas")

            if df.empty:
                logger.warning("El DataFrame está vacío, se retorna sin modificaciones")
                return df.copy()

            df_copy = df.copy()

            conversiones = {
                'Float64': 'float64',
                'Int64': 'int64',
                'string': 'object',
                'boolean': 'bool'
            }

            for col in df_copy.columns:
                try:
                    dtype_str = str(df_copy[col].dtype)

                    for tipo_problematico, tipo_destino in conversiones.items():
                        if tipo_problematico in dtype_str:
                            try:
                                df_copy[col] = df_copy[col].astype(tipo_destino)
                                logger.info(f"Columna '{col}' convertida de {tipo_problematico} a {tipo_destino}")
                                break
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"No se pudo convertir la columna '{col}' de {tipo_problematico} a {tipo_destino}: {str(e)}")
                                continue

                except Exception as e:
                    logger.error(f"Error al procesar la columna '{col}': {str(e)}")
                    continue

            return df_copy

        except TypeError as e:
            logger.error(f"Error de tipo en convertir_tipos_json_serializable: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error inesperado en convertir_tipos_json_serializable: {str(e)}")
            return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    @staticmethod
    def corregir_dataframe_para_streamlit(df: pd.DataFrame) -> pd.DataFrame:
        """Prepara DataFrame para Streamlit sin errores de Arrow"""
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("El parámetro debe ser un DataFrame de pandas")

            if df.empty:
                logger.warning("El DataFrame está vacío, se retorna sin modificaciones")
                return df.copy()

            df_copy = df.copy()

            for col in df_copy.columns:
                try:
                    dtype = df_copy[col].dtype

                    if dtype == 'object':
                        try:
                            df_copy[col] = df_copy[col].astype(str)
                        except Exception as e:
                            logger.warning(f"Error al convertir columna '{col}' a string: {str(e)}")
                            # Intentar convertir elemento por elemento
                            df_copy[col] = df_copy[col].apply(lambda x: str(x) if x is not None else 'None')

                    elif dtype.name.startswith('datetime64'):
                        try:
                            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Error al convertir columna '{col}' a datetime: {str(e)}")
                            continue

                    elif dtype.name.startswith('int') and df_copy[col].isnull().any():
                        try:
                            df_copy[col] = df_copy[col].astype('float64')
                        except Exception as e:
                            logger.warning(f"Error al convertir columna '{col}' de int a float64: {str(e)}")
                            continue

                    elif dtype.name == 'category':
                        try:
                            df_copy[col] = df_copy[col].astype(str)
                        except Exception as e:
                            logger.warning(f"Error al convertir columna '{col}' de category a string: {str(e)}")
                            continue

                except Exception as e:
                    logger.error(f"Error al procesar tipo de dato de la columna '{col}': {str(e)}")
                    continue

            # Limpiar valores problemáticos
            try:
                df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                logger.warning(f"Error al reemplazar valores infinitos: {str(e)}")

            # Convertir nombres de columnas a string
            try:
                df_copy.columns = df_copy.columns.astype(str)
            except Exception as e:
                logger.warning(f"Error al convertir nombres de columnas a string: {str(e)}")
                # Intentar renombrar columnas problemáticas
                try:
                    new_columns = []
                    for i, col in enumerate(df_copy.columns):
                        try:
                            new_columns.append(str(col))
                        except:
                            new_columns.append(f"col_{i}")
                    df_copy.columns = new_columns
                except Exception as e2:
                    logger.error(f"Error crítico al renombrar columnas: {str(e2)}")

            return df_copy

        except TypeError as e:
            logger.error(f"Error de tipo en corregir_dataframe_para_streamlit: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error inesperado en corregir_dataframe_para_streamlit: {str(e)}")
            return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    # Método que identifica y separa las columnas según su tipo de dato
    @staticmethod
    def obtener_tipos(df):
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("El parámetro debe ser un DataFrame de pandas")

            if df.empty:
                logger.warning("El DataFrame está vacío")
                return [], []

            # Se seleccionan las columnas que contienen datos numéricos
            try:
                numericas = df.select_dtypes(include='number').columns.tolist()
            except Exception as e:
                logger.error(f"Error al seleccionar columnas numéricas: {str(e)}")
                numericas = []

            # Se seleccionan las columnas categóricas (texto, categorías o booleanos)
            try:
                categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            except Exception as e:
                logger.error(f"Error al seleccionar columnas categóricas: {str(e)}")
                categoricas = []

            # Validación adicional: verificar que las columnas realmente existan
            try:
                columnas_existentes = df.columns.tolist()
                numericas = [col for col in numericas if col in columnas_existentes]
                categoricas = [col for col in categoricas if col in columnas_existentes]
            except Exception as e:
                logger.warning(f"Error al validar existencia de columnas: {str(e)}")

            logger.info(f"Encontradas {len(numericas)} columnas numéricas y {len(categoricas)} columnas categóricas")

            # Se devuelven ambas listas: una con columnas numéricas y otra con categóricas
            return numericas, categoricas

        except TypeError as e:
            logger.error(f"Error de tipo en obtener_tipos: {str(e)}")
            return [], []
        except Exception as e:
            logger.error(f"Error inesperado en obtener_tipos: {str(e)}")
            return [], []