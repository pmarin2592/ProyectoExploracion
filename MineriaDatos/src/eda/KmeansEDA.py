"""
Clase: KmeansEDA

Objetivo: Conector para análisis de clustering (K-means)
"""

from src.modelos.Kmeans import Kmeans
from src.eda.EstadisticasBasicasEda import EstadisticasBasicasEda
from src.helpers.ComponentesUI import ComponentesUI

class KmeansEDA:
    def __init__(self, eda: EstadisticasBasicasEda):
        """
        Inicializa el conector KmeansEDA con un objeto de EDA.
        """
        self._eda = eda
        try:
            self._kmeans_modelo = Kmeans(self._eda)
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error al inicializar K-means: {str(e)}")
            self._kmeans_modelo = None

    def limpiar_y_escalar_datos(self):
        """Prepara los datos para clustering"""
        try:
            self._kmeans_modelo.limpiar_escalar_datos()
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error al preparar datos para K-means: {str(e)}")

    def entrenar_kmeans(self, n_clusters: int = 3, random_state: int = 42):
        """Entrena el modelo K-means"""
        try:
            self._kmeans_modelo.entrenar_kmeans(n_clusters=n_clusters, random_state=random_state)
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error al entrenar K-means: {str(e)}")

    def graficar_elbow(self):
        """Gráfico del codo para determinar el número óptimo de clusters"""
        try:
            return self._kmeans_modelo.graficar_elbow()
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error al graficar Elbow: {str(e)}")
            return None

    def graficar_clusters_pca(self):
        """Gráfico de clusters proyectados sobre los componentes principales"""
        try:
            return self._kmeans_modelo.graficar_clusters_pca()
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error al graficar clusters PCA: {str(e)}")
            return None

    def obtener_centroides(self):
        """Devuelve los centroides de los clusters"""
        try:
            return self._kmeans_modelo.centroides
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error al obtener centroides: {str(e)}")
            return None

    def asignar_clusters_a_datos(self):
        """Agrega columna con el cluster asignado al DataFrame"""
        try:
            return self._kmeans_modelo.asignar_clusters()
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error al asignar clusters: {str(e)}")
            return None

    def graficar_radar_clusters(self):
        """Gráfico radar de los clusters"""
        try:
            return self._kmeans_modelo.graficar_radar()
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error al graficar radar clusters: {str(e)}")
            return None
