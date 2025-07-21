"""
Clase: EstadisticasBasicasEda

Objetivo: Clase enfocada a la muestra de datos para EDA

1. Creacion de clase pmarin 15-07-2024
"""

from src.datos.EstadisticasBasicasDatos import EstadisticasBasicasDatos

class EstadisticasBasicasEda:
    def __init__(self, df):
        self.df = df
        self.eda = EstadisticasBasicasDatos(self.df)

    def obtener_analisis_completo(self):
        return self.eda.obtener_analisis_completo()

    def obtener_resumen_estadisticas(self):
        return self.obtener_resumen_estadisticas()

    def obtener_valores_faltantes(self):
        return self.obtener_valores_faltantes()

    def obtener_tipo_datos(self):
        return self.obtener_tipo_datos()

    def obtener_info_columnas(self):
        return self.obtener_info_columnas()
    def obtener_info_basica(self):
        return self.obtener_info_basica()

    def obtener_primeras_filas(self):
        return self.obtener_primeras_filas()

    def obtener_ultimas_filas(self):
        return self.obtener_ultimas_filas()

    def obtener_nombres_columnas(self):
        return self.obtener_nombres_columnas()


