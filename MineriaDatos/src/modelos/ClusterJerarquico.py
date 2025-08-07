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
from src.eda.PcaEda import PcaEda


class ClusterJerarquico:
    def __init__(self, pca_datos: PcaEda):
        """
        Inicializa el cluster jerárquico con los datos escalados desde PCA.
        """
        if pca_datos.pca_datos.datos_escalados is None:
            print("Los datos de PCA no han sido escalados. Ejecute limpiar_escalar_datos() primero.")
            raise ValueError("Los datos de PCA no han sido escalados. Ejecute limpiar_escalar_datos() primero.")
        self.cluster_data_scaled = pca_datos.pca_datos.datos_escalados
        self.hierarchical_results = {}
        self.optimal_k = None
        self.final_labels = None
        self.metrics_history = {}

    def plot_dendrograms(self, methods: List[str] = None, figsize: Tuple = (20, 15),
                         show_cuts: bool = True) -> go.Figure:
        """
        Crea dendrogramas interactivos para los métodos seleccionados usando Plotly.

        Args:
            methods: Lista de métodos de linkage a usar
            figsize: Tamaño de la figura (ancho, alto) - se convierte a píxeles
            show_cuts: Si mostrar las líneas de corte para 2 y 3 clusters

        Returns:
            Figura de Plotly interactiva
        """

        if methods is None:
            methods = ['ward', 'complete', 'average', 'single']

        # Convertir figsize a píxeles (aprox 100 píxeles por pulgada)
        width = int(figsize[0] * 100)
        height = int(figsize[1] * 100)

        # Calcular matrices de linkage para los métodos seleccionados
        for method in methods:
            if method == 'ward':
                linkage_matrix = linkage(self.cluster_data_scaled, method='ward')
            else:
                distances = pdist(self.cluster_data_scaled, metric='euclidean')
                linkage_matrix = linkage(distances, method=method)

            self.hierarchical_results[method] = linkage_matrix

        # Crear subplots dinámicamente según número de métodos
        n_methods = len(methods)
        cols = 2
        rows = (n_methods + 1) // 2

        # Crear títulos para cada subplot
        subplot_titles = [f'Dendrograma - Método: {method.capitalize()}' for method in methods]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.08
        )

        # Colores para las líneas de corte
        cut_colors = ['red', 'orange']
        cut_labels = ['Corte para 2 clusters', 'Corte para 3 clusters']

        for idx, method in enumerate(methods):
            row = (idx // cols) + 1
            col = (idx % cols) + 1

            linkage_matrix = self.hierarchical_results[method]

            # Crear dendrograma usando scipy para obtener coordenadas
            dendro_data = dendrogram(linkage_matrix,
                                     truncate_mode='lastp',
                                     p=30,
                                     no_plot=True)

            # Extraer coordenadas
            x_coords = dendro_data['icoord']
            y_coords = dendro_data['dcoord']

            # Agregar líneas del dendrograma
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                fig.add_trace(
                    go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        line=dict(color='blue', width=1.5),
                        showlegend=False,
                        hovertemplate='Distancia: %{y:.3f}<br>Posición: %{x}<extra></extra>'
                    ),
                    row=row, col=col
                )

            # Agregar líneas de corte si se solicita
            if show_cuts:
                # Calcular las distancias de corte
                n_samples = linkage_matrix.shape[0] + 1

                # Para 2 clusters: corte en la segunda fusión más grande
                cut_2_clusters = linkage_matrix[n_samples - 3, 2]

                # Para 3 clusters: corte en la tercera fusión más grande
                cut_3_clusters = linkage_matrix[n_samples - 4, 2]

                cuts = [cut_2_clusters, cut_3_clusters]

                # Obtener rango x para las líneas horizontales
                x_min = min([min(x) for x in x_coords])
                x_max = max([max(x) for x in x_coords])
                x_range = [x_min, x_max]

                # Dibujar las líneas de corte
                for cut_idx, (cut_distance, color, label) in enumerate(zip(cuts, cut_colors, cut_labels)):
                    show_legend = (idx == 0)  # Solo mostrar leyenda en el primer subplot

                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=[cut_distance, cut_distance],
                            mode='lines',
                            line=dict(color=color, width=2, dash='dash'),
                            name=label,
                            showlegend=show_legend,
                            hovertemplate=f'{label}: %{{y:.3f}}<extra></extra>'
                        ),
                        row=row, col=col
                    )

            # Configurar ejes para cada subplot
            fig.update_xaxes(
                title_text='Índice de Muestra o (Tamaño del Cluster)',
                row=row, col=col
            )
            fig.update_yaxes(
                title_text='Distancia',
                row=row, col=col
            )

        # Configuración general de la figura
        fig.update_layout(
            height=height,
            width=width,
            title_text="Dendrogramas - Clustering Jerárquico Interactivo",
            title_x=0.5,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            hovermode='closest'
        )

        return fig

