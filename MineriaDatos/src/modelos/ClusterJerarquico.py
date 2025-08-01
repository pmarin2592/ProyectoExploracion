""""
Clase: ClusteriJerarquico

Objetivo:Generar el cluster jerarquico

Cambios: 1. Creacion -fabarca 30-7-2025
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from src.eda.EstadisticasBasicasEda import EstadisticasBasicasEda


class ClusterJerarquico:
    def __init__(self, eda: EstadisticasBasicasEda):
            self._eda = eda
            self.cluster_data = None
            self.cluster_data_scaled = None
            self.numeric_cols = None
            self.scaler = StandardScaler()
            self.hierarchical_results = {}
            self.optimal_k = None
            self.final_labels = None
            self.metrics_history = {}


    def escalar_datos(self) -> Dict:
        """
        Escala las columnas numéricas del DataFrame original.

        Returns:
            Dict con información sobre la preparación de datos
        """
        try:
            # Seleccionar solo columnas numéricas
            self.numeric_cols = self._eda.select_dtypes(include=['number']).columns.tolist()
            self.cluster_data = self._eda[self.numeric_cols].dropna()

            # Escalado
            self.cluster_data_scaled = self.scaler.fit_transform(self.cluster_data)

            info = {
                'n_samples': self.cluster_data_scaled.shape[0],
                'n_features': self.cluster_data_scaled.shape[1],
                'numeric_columns': self.numeric_cols,
                'data_shape': self.cluster_data_scaled.shape
            }
            return info
        except Exception as e:
            raise ValueError(f"Error al escalar los datos: {str(e)}")

    def plot_dendrograms(self, methods: List[str] = None, figsize: Tuple = (20, 15)) -> plt.Figure:
        """
        Crea dendrogramas para los métodos seleccionados.

        Args:
            methods: Lista de métodos de linkage a usar
            figsize: Tamaño de la figura

        Returns:
            Figura de matplotlib
        """
        if self.cluster_data_scaled is None:
            self.prepare_data()

        if methods is None:
            methods = ['ward', 'complete', 'average', 'single']

        # Calcular matrices de linkage para los métodos seleccionados
        for method in methods:
            if method == 'ward':
                linkage_matrix = linkage(self.cluster_data_scaled, method='ward')
            else:
                distances = pdist(self.cluster_data_scaled, metric='euclidean')
                linkage_matrix = linkage(distances, method=method)

            self.hierarchical_results[method] = linkage_matrix

        # Crear subplots dinámicamente según número de métodos
        # Crear subplots dinámicamente según número de métodos
        n_methods = len(methods)
        cols = 2
        rows = (n_methods + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Asegurar que `axes` sea iterable
        if isinstance(axes, np.ndarray):
            axes = axes.ravel()
        else:
            axes = [axes]

        for idx, method in enumerate(methods):
            linkage_matrix = self.hierarchical_results[method]

            dendrogram(linkage_matrix,
                       ax=axes[idx],
                       truncate_mode='lastp',
                       p=30,
                       leaf_rotation=90,
                       leaf_font_size=8)

            axes[idx].set_title(f'Dendrograma - Método: {method.capitalize()}')
            axes[idx].set_xlabel('Índice de Muestra o (Tamaño del Cluster)')
            axes[idx].set_ylabel('Distancia')

        # Ocultar axes vacíos si hay menos que el total de subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig
