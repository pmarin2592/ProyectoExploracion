"""
Clase: EstadisticasBasicasEda

Objetivo: Clase enfocada a la muestra de datos para EDA

1. Creacion de clase pmarin 15-07-2024
"""

from src.datos.EstadisticasBasicasDatos import EstadisticasBasicasDatos

class EstadisticasBasicasEda:
    def __init__(self, df,numericas, categoricas):
        self.df = df
        self.eda = EstadisticasBasicasDatos(self.df,numericas, categoricas)

    def obtener_analisis_completo(self):
        return self.eda.obtener_analisis_completo()

    def obtener_resumen_estadisticas(self):
        return self.eda.obtener_resumen_estadistico()

    def obtener_valores_faltantes(self):
        return self.eda.obetener_resumen_valores_faltantes()

    def obtener_tipo_datos(self):
        return self.eda.obtener_tipo_datos()

    def obtener_info_columnas(self):
        return self.eda.obtener_info_columnas()
    def obtener_info_basica(self):
        return self.eda.obtener_info_basico()

    def obtener_primeras_filas(self):
        return self.eda.obtener_primeras_filas()

    def obtener_ultimas_filas(self):
        return self.eda.obtener_ultimas_filas()

    def obtener_nombres_columnas(self):
        return self.eda.obtener_nombres_columnas()

    def  obtener_analisis_distribucion(self,col):
        return self.eda.obtener_analisis_distribucion(col)

    def obtener_analisis_boxplot(self,col):
        return self.eda.obtener_analisis_boxplot(col)

    def obtener_analisis_correlaccion(self):
        return self.eda.obtener_analisis_correlaccion()

    def obtener_analisis_univariados(self, col):
        return self.eda.obtener_analisis_univariados(col)

