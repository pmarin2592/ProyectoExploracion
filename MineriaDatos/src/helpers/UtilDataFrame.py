"""
Clase: UtilDataframe

Objetivo: Utilidades para manipulación de DataFrames

Cambios:
    1. Creacion de la clase
"""
import numpy as np
import pandas as pd


class UtilDataFrame:


    @staticmethod
    def convertir_tipos_json_serializable(df: pd.DataFrame) -> pd.DataFrame:
        """Convierte tipos problemáticos a tipos JSON-serializables"""
        df_copy = df.copy()

        conversiones = {
            'Float64': 'float64',
            'Int64': 'int64',
            'string': 'object',
            'boolean': 'bool'
        }

        for col in df_copy.columns:
            dtype_str = str(df_copy[col].dtype)

            for tipo_problematico, tipo_destino in conversiones.items():
                if tipo_problematico in dtype_str:
                    df_copy[col] = df_copy[col].astype(tipo_destino)
                    break

        return df_copy

    @staticmethod
    def corregir_dataframe_para_streamlit(df: pd.DataFrame) -> pd.DataFrame:
        """Prepara DataFrame para Streamlit sin errores de Arrow"""
        df_copy = df.copy()

        for col in df_copy.columns:
            dtype = df_copy[col].dtype

            if dtype == 'object':
                df_copy[col] = df_copy[col].astype(str)
            elif dtype.name.startswith('datetime64'):
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            elif dtype.name.startswith('int') and df_copy[col].isnull().any():
                df_copy[col] = df_copy[col].astype('float64')
            elif dtype.name == 'category':
                df_copy[col] = df_copy[col].astype(str)

        # Limpiar valores problemáticos
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
        df_copy.columns = df_copy.columns.astype(str)

        return df_copy

    # Método que identifica y separa las columnas según su tipo de dato
    @staticmethod
    def obtener_tipos(df):
        # Se seleccionan las columnas que contienen datos numéricos
        numericas = df.select_dtypes(include='number').columns.tolist()

        # Se seleccionan las columnas categóricas (texto, categorías o booleanos)
        categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Se devuelven ambas listas: una con columnas numéricas y otra con categóricas
        return numericas, categoricas


