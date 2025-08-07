from src.datos.PcaDatos import PcaDatos


class PcaEda:
    def __init__(self, eda):
        self.eda = eda
        self.pca_datos= PcaDatos(self.eda)
        self.pca_datos.limpiar_escalar_datos()

    def graficar_varianza(self):
        return self.pca_datos.graficar_varianza()

    def biplot(self):
        return self.pca_datos.biplot()

    def graficar_3d(self):
        return self.pca_datos.graficar_3d()

