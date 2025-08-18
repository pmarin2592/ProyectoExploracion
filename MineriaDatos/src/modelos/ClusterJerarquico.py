""""
Clase: ClusteriJerarquico

Objetivo:Generar el cluster jerarquico

Cambios: 1. Creacion -fabarca 30-7-2025
2. Graficos interacivos -Fabarca 10/8/2025
"""
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple
import inspect

class ClusterJerarquico:
    def __init__(self, pca_datos):
        if pca_datos.pca_datos.datos_escalados is None:
            raise ValueError("Los datos de PCA no han sido escalados. Ejecute limpiar_escalar_datos() primero.")

        self.datos_escalados_cluster = pca_datos.pca_datos.datos_escalados
        self.resultados_jerarquicos = {}
        self.k_optimo = None
        self.etiquetas_finales = None
        self.historial_metricas = {}

    def dendrogramas_interactivos(
        self,
        metodos: List[str] = None,
        tamaño_figura: Tuple[int, int] = (1200, 800),
        mostrar_cortes: bool = True
    ):
        """
        Crea dendrogramas interactivos para los métodos seleccionados usando Plotly.
        """
        try:
            if metodos is None:
                metodos = ['ward', 'complete', 'average', 'single']

            fig = make_subplots(
                rows=(len(metodos) + 1) // 2,
                cols=2,
                subplot_titles=[f"Método: {m.capitalize()}" for m in metodos]
            )

            for idx, metodo in enumerate(metodos):
                try:
                    # Calcular linkage
                    if metodo == 'ward':
                        linkage_matrix = linkage(self.datos_escalados_cluster, method='ward')
                    else:
                        distances = pdist(self.datos_escalados_cluster, metric='euclidean')
                        linkage_matrix = linkage(distances, method=metodo)

                    self.resultados_jerarquicos[metodo] = linkage_matrix

                    # Crear datos del dendrograma
                    dendro = dendrogram(linkage_matrix, no_plot=True, truncate_mode='lastp', p=30)

                    # Dibujar las líneas del dendrograma
                    for xs, ys in zip(dendro['icoord'], dendro['dcoord']):
                        fig.add_trace(
                            go.Scatter(
                                x=xs,
                                y=ys,
                                mode='lines',
                                line=dict(color='blue', width=1),
                                hoverinfo='none',
                                showlegend=False
                            ),
                            row=(idx // 2) + 1,
                            col=(idx % 2) + 1
                        )

                    # Líneas de corte (2 y 3 clusters)
                    if mostrar_cortes:
                        n_samples = linkage_matrix.shape[0] + 1
                        cut_2_clusters = linkage_matrix[n_samples - 3, 2]
                        cut_3_clusters = linkage_matrix[n_samples - 4, 2]

                        fig.add_hline(
                            y=cut_2_clusters,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Corte para 2 clusters",
                            row=(idx // 2) + 1,
                            col=(idx % 2) + 1
                        )

                        fig.add_hline(
                            y=cut_3_clusters,
                            line_dash="dash",
                            line_color="orange",
                            annotation_text="Corte para 3 clusters",
                            row=(idx // 2) + 1,
                            col=(idx % 2) + 1
                        )
                except Exception as e_metodo:
                    print(f"⚠️ Error al procesar el método {metodo}: {e_metodo}")

            fig.update_layout(
                width=tamaño_figura[0],
                height=tamaño_figura[1],
                showlegend=False,
                title_text="Dendrogramas Interactivos",
                hovermode="closest"
            )

            return fig

        except Exception as e:
            print(f"❌ Error al generar dendrogramas interactivos: {e}")
            return None

    def generar_codigo_cj(self, metodos):
        try:
            fig = self.dendrogramas_interactivos(metodos)
            codigo_fuente = inspect.getsource(self.dendrogramas_interactivos)
            return fig, codigo_fuente
        except Exception as e:
            print(f"❌ Error al generar código del dendrograma: {e}")
            return None, ""
