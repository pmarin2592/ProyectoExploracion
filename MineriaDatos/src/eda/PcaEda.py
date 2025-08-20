import inspect

from src.datos.PcaDatos import PcaDatos


class PcaEda:
    def __init__(self, eda):
        self.eda = eda
        self.pca_datos= PcaDatos(self.eda)
        self.pca_datos.limpiar_escalar_datos()

    def graficar_varianza(self):
        return self.pca_datos.graficar_varianza(), inspect.getsource(self.pca_datos.graficar_varianza)

    def biplot(self):
        return self.pca_datos.biplot(), inspect.getsource(self.pca_datos.biplot)

    def graficar_3d(self):
        return self.pca_datos.graficar_3d(), inspect.getsource(self.pca_datos.graficar_3d)

    def graficar_3d_con_planos(self):
        return self.pca_datos.graficar_3d_con_planos(), inspect.getsource(self.pca_datos.graficar_3d_con_planos)

    def graficar_proyeccion_pc1_pc2(self):
        return self.pca_datos.graficar_proyeccion_pc1_pc2() , inspect.getsource(self.pca_datos.graficar_proyeccion_pc1_pc2)

    def graficar_proyeccion_pc1_pc3(self):
        return self.pca_datos.graficar_proyeccion_pc1_pc3(), inspect.getsource(self.pca_datos.graficar_proyeccion_pc1_pc3)

    def graficar_heatmap_loadings(self):
        return self.pca_datos.graficar_heatmap_loadings(), inspect.getsource(self.pca_datos.graficar_heatmap_loadings)

    def graficar_circulo_correlacion(self):
        return self.pca_datos.graficar_circulo_correlacion(), inspect.getsource(self.pca_datos.graficar_circulo_correlacion)

    def graficar_contribuciones_variables(self):
        return self.pca_datos.graficar_contribuciones_variables(), inspect.getsource(self.pca_datos.graficar_contribuciones_variables)

    def graficar_analisis_cuadrantes(self):
        return self.pca_datos.graficar_analisis_cuadrantes(), inspect.getsource(self.pca_datos.graficar_analisis_cuadrantes)

