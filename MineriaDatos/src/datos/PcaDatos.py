import logging

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.eda.EstadisticasBasicasEda import EstadisticasBasicasEda

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class PcaDatos:


    def __init__(self,eda: EstadisticasBasicasEda):
        self._eda = eda
        self.pca = None
        self.datos_escalados = None
        self.resultado_pca = None
        self.varianza_explicada = None

    def limpiar_escalar_datos(self):
        # Limpiar datos, eliminar filas con nulos en columnas numéricas
        datos = self._eda.eda.df[self._eda.eda.numericas].dropna()
        if datos.empty or len(self._eda.eda.numericas) < 2:
            logger.error(f"Se requieren al menos 2 variables numéricas con datos válidos para PCA")
            raise 'Se requieren al menos 2 variables numéricas con datos válidos para PCA.'
        # Escalar los datos
        escalador = StandardScaler()
        self.datos_escalados = escalador.fit_transform(datos)
        # Aplicar PCA
        self.pca = PCA()
        self.resultado_pca = self.pca.fit_transform(self.datos_escalados)
        self.varianza_explicada = self.pca.explained_variance_ratio_

    def graficar_varianza(self):
        varianza_acumulada = np.cumsum(self.varianza_explicada)

        fig = plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.varianza_explicada) + 1), self.varianza_explicada)
        plt.title('Varianza explicada')
        plt.xlabel('Componentes')
        plt.ylabel('Varianza')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o')
        plt.axhline(0.9, color='orange', linestyle='--', label='90% varianza')
        plt.axhline(0.95, color='red', linestyle='--', label='95% varianza')
        plt.title('Varianza acumulada')
        plt.xlabel('Número de componentes')
        plt.ylabel('Varianza acumulada')
        plt.legend()
        plt.tight_layout()


        return fig


    def biplot(self):
        fig= plt.figure(figsize=(8, 6))
        plt.scatter(self.resultado_pca[:, 0], self.resultado_pca[:, 1], alpha=0.6)
        for i, variable in enumerate(self._eda.eda.numericas):
            plt.arrow(0, 0, self.pca.components_[0, i] * 3, self.pca.components_[1, i] * 3,
                      color='r', head_width=0.1)
            plt.text(self.pca.components_[0, i] * 3.2, self.pca.components_[1, i] * 3.2, variable)
        plt.xlabel(f'PC1 ({self.varianza_explicada[0] * 100:.1f}%)')
        plt.ylabel(f'PC2 ({self.varianza_explicada[1] * 100:.1f}%)')
        plt.title('Biplot PCA')
        plt.grid()
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(0, color='gray', lw=0.5)


        return fig

    def graficar_3d(self):
        if len(self._eda.eda.numericas) < 3:
            logger.error(f"Se requieren al menos 3 variables numéricas con datos válidos para PCA")
            raise 'Se requieren al menos 3 variables numéricas con datos válidos para PCA.'


        figura = plt.figure(figsize=(8, 6))
        eje = figura.add_subplot(111, projection='3d')
        eje.scatter(self.resultado_pca[:, 0], self.resultado_pca[:, 1], self.resultado_pca[:, 2], alpha=0.6)
        eje.set_xlabel(f'PC1 ({self.varianza_explicada[0] * 100:.1f}%)')
        eje.set_ylabel(f'PC2 ({self.varianza_explicada[1] * 100:.1f}%)')
        eje.set_zlabel(f'PC3 ({self.varianza_explicada[2] * 100:.1f}%)')
        eje.set_title('PCA 3D')


        return figura
