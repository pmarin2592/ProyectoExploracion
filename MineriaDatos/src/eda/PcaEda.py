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

