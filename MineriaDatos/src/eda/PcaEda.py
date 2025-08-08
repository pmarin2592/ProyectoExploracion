from src.datos.PcaDatos import PcaDatos


class PcaEda:
    def __init__(self, eda):
        self._eda = eda
        self._pca_datos= PcaDatos(self._eda)
        self._pca_datos.limpiar_escalar_datos()

    def graficar_varianza(self):
        return self._pca_datos.graficar_varianza()

    def biplot(self):
        return self._pca_datos.biplot()

    def graficar_3d(self):
        return self._pca_datos.graficar_3d()

    def graficar_3d_con_planos(self):
        return self._pca_datos.graficar_3d_con_planos()

    def graficar_proyeccion_pc1_pc2(self):
        return self._pca_datos.graficar_proyeccion_pc1_pc2()

    def graficar_proyeccion_pc1_pc3(self):
        return self._pca_datos.graficar_proyeccion_pc1_pc3()

    def graficar_heatmap_loadings(self):
        return self._pca_datos.graficar_heatmap_loadings()

    def graficar_circulo_correlacion(self):
        return self._pca_datos.graficar_circulo_correlacion()

    def graficar_contribuciones_variables(self):
        return self._pca_datos.graficar_contribuciones_variables()

    def graficar_analisis_cuadrantes(self):
        return self._pca_datos.graficar_analisis_cuadrantes()

