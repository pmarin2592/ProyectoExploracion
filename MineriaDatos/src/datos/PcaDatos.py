import logging

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go
from src.eda.EstadisticasBasicasEda import EstadisticasBasicasEda

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PcaDatos:

    def __init__(self, eda: EstadisticasBasicasEda):
        try:
            if eda is None:
                raise ValueError("El objeto EstadisticasBasicasEda no puede ser None")
            self._eda = eda
            self.pca = None
            self.datos_escalados = None
            self.resultado_pca = None
            self.varianza_explicada = None
            logger.info("PcaDatos inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar PcaDatos: {e}")
            raise

    def limpiar_escalar_datos(self):
        try:
            # Limpiar datos, eliminar filas con nulos en columnas numéricas
            if not hasattr(self._eda, 'eda') or not hasattr(self._eda.eda, 'df'):
                raise AttributeError("El objeto EDA no tiene la estructura esperada")

            if not hasattr(self._eda.eda, 'numericas') or not self._eda.eda.numericas:
                raise ValueError("No se encontraron columnas numéricas en el dataset")

            datos = self._eda.eda.df[self._eda.eda.numericas].dropna()

            if datos.empty:
                raise ValueError("No hay datos válidos después de eliminar valores nulos")

            if len(self._eda.eda.numericas) < 2:
                logger.error(f"Se requieren al menos 2 variables numéricas con datos válidos para PCA")
                raise ValueError('Se requieren al menos 2 variables numéricas con datos válidos para PCA.')

            # Escalar los datos
            escalador = StandardScaler()
            self.datos_escalados = escalador.fit_transform(datos)

            # Aplicar PCA
            self.pca = PCA()
            self.resultado_pca = self.pca.fit_transform(self.datos_escalados)
            self.varianza_explicada = self.pca.explained_variance_ratio_

            logger.info(f"PCA aplicado correctamente con {len(self._eda.eda.numericas)} variables")

        except ValueError as ve:
            logger.error(f"Error de validación en limpiar_escalar_datos: {ve}")
            raise
        except AttributeError as ae:
            logger.error(f"Error de atributo en limpiar_escalar_datos: {ae}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado en limpiar_escalar_datos: {e}")
            raise

    def graficar_varianza(self):
        """Gráfico interactivo de varianza explicada"""
        try:
            if self.varianza_explicada is None:
                raise ValueError("Debe ejecutar limpiar_escalar_datos() primero")

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Varianza Explicada', 'Varianza Acumulada')
            )

            # Varianza explicada
            fig.add_trace(
                go.Bar(
                    x=list(range(1, len(self.varianza_explicada) + 1)),
                    y=self.varianza_explicada,
                    name='Varianza Explicada',
                    marker_color='lightblue',
                    hovertemplate='PC%{x}: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )

            # Varianza acumulada
            varianza_acumulada = np.cumsum(self.varianza_explicada)
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(varianza_acumulada) + 1)),
                    y=varianza_acumulada,
                    mode='lines+markers',
                    name='Varianza Acumulada',
                    line=dict(color='orange', width=3),
                    marker=dict(size=8),
                    hovertemplate='PC%{x}: %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )

            # Líneas de referencia
            fig.add_hline(y=0.9, line_dash="dash", line_color="red",
                          annotation_text="90%", row=1, col=2)
            fig.add_hline(y=0.95, line_dash="dash", line_color="green",
                          annotation_text="95%", row=1, col=2)

            fig.update_layout(
                title='Análisis de Varianza Explicada - Interactivo',
                height=400,
                showlegend=False
            )

            return fig

        except ValueError as ve:
            logger.error(f"Error de validación en graficar_varianza: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error al crear gráfico de varianza: {e}")
            raise

    def biplot(self):
        """Biplot interactivo con vectores de variables"""
        try:
            if self.resultado_pca is None or self.pca is None:
                raise ValueError("Debe ejecutar limpiar_escalar_datos() primero")

            if self.resultado_pca.shape[1] < 2:
                raise ValueError("Se necesitan al menos 2 componentes principales para el biplot")

            fig = go.Figure()

            # Puntos de datos
            fig.add_trace(go.Scatter(
                x=self.resultado_pca[:, 0],
                y=self.resultado_pca[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color='lightblue',
                    opacity=0.7,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name='Observaciones',
                hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
            ))

            # Vectores de variables
            for i, variable in enumerate(self._eda.eda.numericas):
                try:
                    x_end = self.pca.components_[0, i] * 3
                    y_end = self.pca.components_[1, i] * 3

                    # Vector
                    fig.add_trace(go.Scatter(
                        x=[0, x_end],
                        y=[0, y_end],
                        mode='lines+markers',
                        line=dict(color='red', width=2),
                        marker=dict(size=[0, 8], color='red'),
                        name=variable,
                        showlegend=True,
                        hovertemplate=f'{variable}<br>PC1: {self.pca.components_[0, i]:.3f}<br>PC2: '
                                      f'{self.pca.components_[1, i]:.3f}<extra></extra>'
                    ))

                    # Etiqueta
                    fig.add_annotation(
                        x=x_end * 1.1,
                        y=y_end * 1.1,
                        text=variable,
                        showarrow=False,
                        font=dict(size=12, color='red')
                    )
                except IndexError as ie:
                    logger.warning(f"Error al procesar variable {variable}: {ie}")
                    continue

            fig.update_layout(
                title=f'Biplot Interactivo - PC1: {self.varianza_explicada[0] * 100:.1f}% | PC2:'
                      f' {self.varianza_explicada[1] * 100:.1f}%',
                xaxis_title=f'PC1 ({self.varianza_explicada[0] * 100:.1f}%)',
                yaxis_title=f'PC2 ({self.varianza_explicada[1] * 100:.1f}%)',
                width=800,
                height=600,
                hovermode='closest'
            )

            return fig

        except ValueError as ve:
            logger.error(f"Error de validación en biplot: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error al crear biplot: {e}")
            raise

    def graficar_3d(self):
        """Gráfico 3D interactivo con Plotly"""
        try:
            if self.resultado_pca is None or self.varianza_explicada is None:
                raise ValueError("Debe ejecutar limpiar_escalar_datos() primero")

            if len(self._eda.eda.numericas) < 3:
                logger.error(f"Se requieren al menos 3 variables numéricas con datos válidos para PCA")
                raise ValueError('Se requieren al menos 3 variables numéricas con datos válidos para PCA.')

            if self.resultado_pca.shape[1] < 3:
                raise ValueError("Se necesitan al menos 3 componentes principales para gráfico 3D")

            hover_text = [
                (f'Punto {i}<br>PC1: {self.resultado_pca[i, 0]:.3f}<br>PC2: {self.resultado_pca[i, 1]:.3f}'
                 f'<br>PC3: {self.resultado_pca[i, 2]:.3f}')
                for i in range(len(self.resultado_pca))]

            fig = go.Figure(data=[go.Scatter3d(
                x=self.resultado_pca[:, 0],
                y=self.resultado_pca[:, 1],
                z=self.resultado_pca[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.resultado_pca[:, 0],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="PC1 Values")
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Datos PCA'
            )])

            fig.update_layout(
                title=f'PCA 3D Interactivo<br>PC1: {self.varianza_explicada[0] * 100:.1f}% | PC2:'
                      f' {self.varianza_explicada[1] * 100:.1f}% | PC3: {self.varianza_explicada[2] * 100:.1f}%',
                scene=dict(
                    xaxis_title=f'PC1 ({self.varianza_explicada[0] * 100:.1f}%)',
                    yaxis_title=f'PC2 ({self.varianza_explicada[1] * 100:.1f}%)',
                    zaxis_title=f'PC3 ({self.varianza_explicada[2] * 100:.1f}%)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=800,
                height=600
            )

            return fig

        except ValueError as ve:
            logger.error(f"Error de validación en graficar_3d: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error al crear gráfico 3D: {e}")
            raise

    def graficar_3d_con_planos(self):
        """Gráfico 3D con planos de proyección usando Plotly"""
        try:
            if self.resultado_pca is None or self.varianza_explicada is None:
                raise ValueError("Debe ejecutar limpiar_escalar_datos() primero")

            if len(self._eda.eda.numericas) < 3:
                logger.error(f"Se requieren al menos 3 variables numéricas con datos válidos para PCA")
                raise ValueError('Se requieren al menos 3 variables numéricas con datos válidos para PCA.')

            if self.resultado_pca.shape[1] < 3:
                raise ValueError("Se necesitan al menos 3 componentes principales para gráfico 3D con planos")

            fig = go.Figure()

            # Puntos de datos 3D
            hover_text = [
                (f'Punto {i}<br>PC1: {self.resultado_pca[i, 0]:.3f}<br>PC2: {self.resultado_pca[i, 1]:.3f}<br>PC3: '
                 f'{self.resultado_pca[i, 2]:.3f}')
                for i in range(len(self.resultado_pca))]

            fig.add_trace(go.Scatter3d(
                x=self.resultado_pca[:, 0],
                y=self.resultado_pca[:, 1],
                z=self.resultado_pca[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color='red',
                    opacity=0.8
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Datos PCA'
            ))

            # Rangos para los planos
            x_range = [self.resultado_pca[:, 0].min(), self.resultado_pca[:, 0].max()]
            y_range = [self.resultado_pca[:, 1].min(), self.resultado_pca[:, 1].max()]
            z_range = [self.resultado_pca[:, 2].min(), self.resultado_pca[:, 2].max()]

            # Crear meshgrid para los planos
            x_plane = np.linspace(x_range[0], x_range[1], 10)
            y_plane = np.linspace(y_range[0], y_range[1], 10)
            z_plane = np.linspace(z_range[0], z_range[1], 10)

            # Plano XY (Z=0)
            X_mesh, Y_mesh = np.meshgrid(x_plane, y_plane)
            Z_mesh = np.zeros_like(X_mesh)

            fig.add_trace(go.Surface(
                x=X_mesh,
                y=Y_mesh,
                z=Z_mesh,
                opacity=0.3,
                colorscale=[[0, 'blue'], [1, 'blue']],
                showscale=False,
                name='Plano XY (Z=0)',
                hovertemplate='Plano XY<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: 0<extra></extra>'
            ))

            # Plano XZ (Y=0)
            X_mesh, Z_mesh = np.meshgrid(x_plane, z_plane)
            Y_mesh = np.zeros_like(X_mesh)

            fig.add_trace(go.Surface(
                x=X_mesh,
                y=Y_mesh,
                z=Z_mesh,
                opacity=0.3,
                colorscale=[[0, 'green'], [1, 'green']],
                showscale=False,
                name='Plano XZ (Y=0)',
                hovertemplate='Plano XZ<br>X: %{x:.3f}<br>Y: 0<br>Z: %{z:.3f}<extra></extra>'
            ))

            # Plano YZ (X=0)
            Y_mesh, Z_mesh = np.meshgrid(y_plane, z_plane)
            X_mesh = np.zeros_like(Y_mesh)

            fig.add_trace(go.Surface(
                x=X_mesh,
                y=Y_mesh,
                z=Z_mesh,
                opacity=0.3,
                colorscale=[[0, 'orange'], [1, 'orange']],
                showscale=False,
                name='Plano YZ (X=0)',
                hovertemplate='Plano YZ<br>X: 0<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title=f'PCA 3D con Planos de Proyección<br>PC1: {self.varianza_explicada[0] * 100:.1f}% | '
                      f'PC2: {self.varianza_explicada[1] * 100:.1f}% | PC3: {self.varianza_explicada[2] * 100:.1f}%',
                scene=dict(
                    xaxis_title=f'PC1 ({self.varianza_explicada[0] * 100:.1f}%)',
                    yaxis_title=f'PC2 ({self.varianza_explicada[1] * 100:.1f}%)',
                    zaxis_title=f'PC3 ({self.varianza_explicada[2] * 100:.1f}%)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    aspectmode='cube'
                ),
                width=900,
                height=700,
                showlegend=True
            )

            return fig

        except ValueError as ve:
            logger.error(f"Error de validación en graficar_3d_con_planos: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error al crear gráfico 3D con planos: {e}")
            raise

    def graficar_proyeccion_pc1_pc2(self):
        """Proyección 2D interactiva del plano PC1-PC2"""
        try:
            if self.resultado_pca is None or self.varianza_explicada is None:
                raise ValueError("Debe ejecutar limpiar_escalar_datos() primero")

            if self.resultado_pca.shape[1] < 2:
                raise ValueError("Se necesitan al menos 2 componentes principales")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.resultado_pca[:, 0],
                y=self.resultado_pca[:, 1],
                mode='markers',
                marker=dict(color='red', size=8, opacity=0.7),
                hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
                name='Datos PCA'
            ))

            fig.update_layout(
                title='Proyección Plano PC1-PC2 - Interactivo',
                xaxis_title=f'PC1 ({self.varianza_explicada[0] * 100:.1f}%)',
                yaxis_title=f'PC2 ({self.varianza_explicada[1] * 100:.1f}%)',
                width=600,
                height=500
            )
            return fig

        except ValueError as ve:
            logger.error(f"Error de validación en graficar_proyeccion_pc1_pc2: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error al crear proyección PC1-PC2: {e}")
            raise

    def graficar_proyeccion_pc1_pc3(self):
        """Proyección 2D interactiva del plano PC1-PC3"""
        try:
            if self.resultado_pca is None or self.varianza_explicada is None:
                raise ValueError("Debe ejecutar limpiar_escalar_datos() primero")

            if len(self._eda.eda.numericas) < 3:
                logger.error(f"Se requieren al menos 3 variables numéricas con datos válidos para PCA")
                raise ValueError('Se requieren al menos 3 variables numéricas con datos válidos para PCA.')

            if self.resultado_pca.shape[1] < 3:
                raise ValueError("Se necesitan al menos 3 componentes principales")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.resultado_pca[:, 0],
                y=self.resultado_pca[:, 2],
                mode='markers',
                marker=dict(color='green', size=8, opacity=0.7),
                hovertemplate='PC1: %{x:.3f}<br>PC3: %{y:.3f}<extra></extra>',
                name='Datos PCA'
            ))

            fig.update_layout(
                title='Proyección Plano PC1-PC3 - Interactivo',
                xaxis_title=f'PC1 ({self.varianza_explicada[0] * 100:.1f}%)',
                yaxis_title=f'PC3 ({self.varianza_explicada[2] * 100:.1f}%)',
                width=600,
                height=500
            )
            return fig

        except ValueError as ve:
            logger.error(f"Error de validación en graficar_proyeccion_pc1_pc3: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error al crear proyección PC1-PC3: {e}")
            raise

    def graficar_heatmap_loadings(self):
        """Heatmap interactivo de contribuciones (loadings)"""
        try:
            if self.pca is None:
                raise ValueError("Debe ejecutar limpiar_escalar_datos() primero")

            if not hasattr(self.pca, 'components_'):
                raise AttributeError("El objeto PCA no tiene componentes calculados")

            loadings_matrix = self.pca.components_[:min(5, len(self.pca.components_))].T

            fig = go.Figure(data=go.Heatmap(
                z=loadings_matrix,
                x=[f'PC{i + 1}' for i in range(loadings_matrix.shape[1])],
                y=self._eda.eda.numericas,
                colorscale='RdBu',
                zmid=0,
                hovertemplate='Variable: %{y}<br>Componente: %{x}<br>Loading: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Loading Value")
            ))

            fig.update_layout(
                title='Matriz de Loadings Interactiva',
                xaxis_title='Componentes Principales',
                yaxis_title='Variables',
                width=800,
                height=600
            )

            return fig

        except ValueError as ve:
            logger.error(f"Error de validación en graficar_heatmap_loadings: {ve}")
            raise
        except AttributeError as ae:
            logger.error(f"Error de atributo en graficar_heatmap_loadings: {ae}")
            raise
        except Exception as e:
            logger.error(f"Error al crear heatmap de loadings: {e}")
            raise

    def graficar_circulo_correlacion(self):
        """Círculo de correlación interactivo"""
        try:
            if self.pca is None or self.varianza_explicada is None:
                raise ValueError("Debe ejecutar limpiar_escalar_datos() primero")

            if not hasattr(self.pca, 'components_'):
                raise AttributeError("El objeto PCA no tiene componentes calculados")

            if self.pca.components_.shape[0] < 2:
                raise ValueError("Se necesitan al menos 2 componentes principales")

            fig = go.Figure()

            # Círculo unitario
            theta = np.linspace(0, 2 * np.pi, 100)
            fig.add_trace(go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='Círculo unitario',
                hoverinfo='skip'
            ))

            # Vectores de variables
            for i, feature in enumerate(self._eda.eda.numericas):
                try:
                    pc1_load = self.pca.components_[0, i]
                    pc2_load = self.pca.components_[1, i]

                    # Vector
                    fig.add_trace(go.Scatter(
                        x=[0, pc1_load],
                        y=[0, pc2_load],
                        mode='lines+markers',
                        line=dict(color='red', width=3),
                        marker=dict(size=[0, 10], color='red'),
                        name=feature,
                        hovertemplate=f'{feature}<br>PC1: {pc1_load:.3f}<br>PC2: {pc2_load:.3f}<extra></extra>'
                    ))

                    # Etiqueta
                    fig.add_annotation(
                        x=pc1_load * 1.1,
                        y=pc2_load * 1.1,
                        text=feature,
                        showarrow=False,
                        font=dict(size=12, color='red')
                    )
                except IndexError as ie:
                    logger.warning(f"Error al procesar variable {feature}: {ie}")
                    continue

            # Líneas de referencia
            fig.add_hline(y=0, line_color="gray", line_width=1, opacity=0.5)
            fig.add_vline(x=0, line_color="gray", line_width=1, opacity=0.5)

            # Líneas diagonales
            x_diag = np.linspace(-1.2, 1.2, 100)
            fig.add_trace(go.Scatter(
                x=x_diag, y=x_diag,
                mode='lines',
                line=dict(color='blue', dash='dot'),
                name='y=x',
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=x_diag, y=-x_diag,
                mode='lines',
                line=dict(color='green', dash='dot'),
                name='y=-x',
                hoverinfo='skip'
            ))

            fig.update_layout(
                title='Círculo de Correlación Interactivo',
                xaxis_title=f'PC1 ({self.varianza_explicada[0] * 100:.1f}%)',
                yaxis_title=f'PC2 ({self.varianza_explicada[1] * 100:.1f}%)',
                xaxis=dict(range=[-1.3, 1.3]),
                yaxis=dict(range=[-1.3, 1.3]),
                width=700,
                height=700,
                showlegend=True
            )

            return fig

        except ValueError as ve:
            logger.error(f"Error de validación en graficar_circulo_correlacion: {ve}")
            raise
        except AttributeError as ae:
            logger.error(f"Error de atributo en graficar_circulo_correlacion: {ae}")
            raise
        except Exception as e:
            logger.error(f"Error al crear círculo de correlación: {e}")
            raise

    def graficar_contribuciones_variables(self):
        try:
            if self.pca is None:
                raise ValueError("Debe ejecutar limpiar_escalar_datos() primero")

            if not hasattr(self.pca, 'components_'):
                raise AttributeError("El objeto PCA no tiene componentes calculados")

            if self.pca.components_.shape[0] < 3:
                raise ValueError("Se necesitan al menos 3 componentes principales")

            contributions = pd.DataFrame(
                self.pca.components_[:3].T,
                columns=[f'PC{i + 1}' for i in range(3)],
                index=self._eda.eda.numericas
            )

            fig = go.Figure()

            # Crear barras para cada componente
            colors = ['lightblue', 'orange', 'lightgreen']
            for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
                fig.add_trace(go.Bar(
                    name=pc,
                    x=self._eda.eda.numericas,
                    y=contributions[pc],
                    marker_color=colors[i],
                    hovertemplate=f'{pc}<br>Variable: %{{x}}<br>Contribución: %{{y:.3f}}<extra></extra>'
                ))

            fig.update_layout(
                title='Contribución de Variables a los Primeros 3 Componentes',
                xaxis_title='Variables',
                yaxis_title='Contribución',
                barmode='group',
                width=900,
                height=500,
                xaxis={'categoryorder': 'total descending'}
            )

            return fig

        except ValueError as ve:
            logger.error(f"Error de validación en graficar_contribuciones_variables: {ve}")
            raise
        except AttributeError as ae:
            logger.error(f"Error de atributo en graficar_contribuciones_variables: {ae}")
            raise
        except Exception as e:
            logger.error(f"Error al crear gráfico de contribuciones: {e}")
            raise

    def graficar_analisis_cuadrantes(self):
        """Análisis interactivo de cuadrantes en el plano PC1-PC2"""
        try:
            if self.resultado_pca is None or self.varianza_explicada is None:
                raise ValueError("Debe ejecutar limpiar_escalar_datos() primero")

            if self.resultado_pca.shape[1] < 2:
                raise ValueError("Se necesitan al menos 2 componentes principales")

            # Colorear puntos según cuadrante
            cuadrantes_data = {'Q I': [], 'Q II': [], 'Q III': [], 'Q IV': []}
            colors = {'Q I': 'red', 'Q II': 'blue', 'Q III': 'green', 'Q IV': 'orange'}

            for i in range(len(self.resultado_pca)):
                try:
                    pc1, pc2 = self.resultado_pca[i, 0], self.resultado_pca[i, 1]
                    if pc1 >= 0 and pc2 >= 0:
                        cuadrantes_data['Q III'].append((pc1, pc2, i))
                    else:
                        cuadrantes_data['Q IV'].append((pc1, pc2, i))
                except IndexError as ie:
                    logger.warning(f"Error al procesar punto {i}: {ie}")
                    continue

            fig = go.Figure()

            # Crear scatter por cada cuadrante
            for q, color in colors.items():
                if cuadrantes_data[q]:
                    try:
                        x_vals = [point[0] for point in cuadrantes_data[q]]
                        y_vals = [point[1] for point in cuadrantes_data[q]]
                        indices = [point[2] for point in cuadrantes_data[q]]

                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='markers',
                            marker=dict(color=color, size=8, opacity=0.7),
                            name=q,
                            hovertemplate=f'{q} - Punto %{{customdata}}<br>PC1: %{{x:.3f}}<br>PC2: %'
                                          f'{{y:.3f}}<extra></extra>',
                            customdata=indices
                        ))
                    except Exception as e:
                        logger.warning(f"Error al procesar cuadrante {q}: {e}")
                        continue

            # Líneas de división
            fig.add_hline(y=0, line_color="black", line_width=2, opacity=0.7)
            fig.add_vline(x=0, line_color="black", line_width=2, opacity=0.7)

            fig.update_layout(
                title='Análisis por Cuadrantes - Interactivo',
                xaxis_title=f'PC1 ({self.varianza_explicada[0] * 100:.1f}%)',
                yaxis_title=f'PC2 ({self.varianza_explicada[1] * 100:.1f}%)',
                width=700,
                height=600,
                showlegend=True
            )
            return fig

        except ValueError as ve:
            logger.error(f"Error de validación en graficar_analisis_cuadrantes: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error al crear análisis de cuadrantes: {e}")
