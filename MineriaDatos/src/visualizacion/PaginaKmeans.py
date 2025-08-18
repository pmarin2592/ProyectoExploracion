import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modelos.Kmeans import Kmeans, Clustering
from kneed import KneeLocator
import inspect
import logging

logger = logging.getLogger(__name__)

class PaginaKmeans:
    def __init__(self):
        self.df = None
        self.cluster_manual = None
        self.kmeans_obj = None

        # Verificar si el dataset está cargado
        if hasattr(st.session_state, 'df_cargado') and st.session_state.df_cargado is not None:
            self.df = st.session_state.df_cargado.copy()

            # 🔹 Eliminar todas las filas que tengan al menos un nulo
            self.df = self.df.dropna(how='any')

            # Eliminar columnas duplicadas
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]

            self.tiene_datos = True
        else:
            self.tiene_datos = False

    def render(self):
        st.markdown("""
            <h1 style='
                text-align: center;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 3rem;
                margin-bottom: 2rem;
                font-weight: bold;
            '>K-Means</h1>
        """, unsafe_allow_html=True)

        if not self.tiene_datos:
            st.warning("❌ No hay dataset cargado. Por favor, carga un dataset primero.")
            return

        # Columnas numéricas
        columnas_numericas = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(columnas_numericas) < 2:
            st.warning("El dataset debe tener al menos 2 columnas numéricas para clustering.")
            return

        # Instanciar Clustering y escalar
        self.cluster_manual = Clustering(self.df)
        datos_escalados, columnas_escaladas = self.cluster_manual.limpiar_y_escalar(columnas_numericas)
        if datos_escalados is None:
            st.warning("Error al escalar las variables numéricas.")
            return

        self.kmeans_obj = Kmeans(datos_escalados)

        # Crear pestañas
        tab1, tab2, tab3, tab4 = st.tabs([
            "K-means Automático",
            "Clustering Manual",
            "Selecciona Cluster",
            "Código Fuente"
        ])

        # ---------- TAB 1: Codo Jambu ----------
        with tab1:
            st.subheader("🔹 Codo Jambu")
            inercias = self.kmeans_obj.evaluar_varios_k()
            k_range = list(range(2, len(inercias) + 2))

            kneedle = KneeLocator(k_range, inercias, curve='convex', direction='decreasing')
            k_opt = kneedle.knee
            inercia_opt = inercias[k_opt - 2] if k_opt else inercias[0]

            fig_codo = go.Figure()
            fig_codo.add_trace(go.Scatter(x=k_range, y=inercias, mode='lines+markers',
                                          line=dict(color='royalblue'), marker=dict(size=8), name='Inercia'))
            if k_opt:
                fig_codo.add_trace(go.Scatter(x=[k_opt], y=[inercia_opt], mode='markers+text',
                                              marker=dict(color='red', size=12), text=["Codo"],
                                              textposition="top center",
                                              name='K óptimo'))
            fig_codo.update_layout(title='📉 Codo de Jambú - Número óptimo de clusters',
                                   xaxis_title='Número de clusters (k)',
                                   yaxis_title='Inercia',
                                   template='plotly_white', height=500)
            st.plotly_chart(fig_codo, use_container_width=True)

            if k_opt:
                st.info(f"🔴 Se ha detectado automáticamente {k_opt} clusters como óptimos para aplicar K-means.")

                modelo_final, etiquetas_kmeans = self.kmeans_obj.entrenar(k_opt)
                etiquetas_completas_kmeans = pd.Series(index=self.df.index, dtype="float64")
                etiquetas_completas_kmeans[self.cluster_manual.df_numerico.index] = etiquetas_kmeans
                self.df['Cluster_KMeans'] = etiquetas_completas_kmeans
                st.session_state['df_clusterizado'] = self.df.copy()

        # ---------- TAB 2: Clusters ----------
        with tab2:
            st.subheader("🔹 Clusters")
            k_cluster = st.slider("Elige número de clusters para el análisis manual", 2, 10, 3, key="slider_clustering")

            # Limpiar filas con nulos
            self.cluster_manual.df_numerico = self.cluster_manual.df_numerico.dropna(how='any')

            resultados = self.cluster_manual.clustering_kmeans(k=k_cluster)

            etiquetas_completas_manual = pd.Series(data=resultados['etiquetas'],
                                                   index=self.cluster_manual.df_numerico.index)
            self.df = self.df.drop(columns=['Cluster_Manual'], errors='ignore')
            self.df.loc[etiquetas_completas_manual.index, 'Cluster_Manual'] = etiquetas_completas_manual

            # DataFrame para graficar
            df_plot = self.cluster_manual.df_numerico.copy()
            df_plot['Cluster'] = resultados['etiquetas']

            columnas_graf = df_plot.columns[:2]
            palette = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
            colores = {cluster: palette[i % len(palette)] for i, cluster in enumerate(sorted(df_plot['Cluster'].unique()))}

            # Scatter plot
            fig_clusters = go.Figure()
            for cluster in sorted(df_plot['Cluster'].unique()):
                df_cluster = df_plot[df_plot['Cluster'] == cluster]
                fig_clusters.add_trace(go.Scatter(
                    x=df_cluster[columnas_graf[0]],
                    y=df_cluster[columnas_graf[1]],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(size=8, color=colores[cluster])
                ))

            fig_clusters.update_layout(
                title="Clustering Manual",
                xaxis_title=columnas_graf[0],
                yaxis_title=columnas_graf[1],
                template='plotly_white',
                height=500,
            )
            st.plotly_chart(fig_clusters, use_container_width=True)

            # Gráfico de barras con conteos
            conteos = df_plot['Cluster'].value_counts().sort_index()
            fig_barras = go.Figure()
            for cluster, count in conteos.items():
                fig_barras.add_trace(go.Bar(
                    x=[f"Cluster {cluster}"],
                    y=[count],
                    name=f'Cluster {cluster}',
                    marker_color=colores[cluster]
                ))
            fig_barras.update_layout(
                title="Número de puntos por Cluster",
                xaxis_title="Cluster",
                yaxis_title="Cantidad de puntos",
                template='plotly_white',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_barras, use_container_width=True)

        # ---------- TAB 3: Selecciona Cluster ----------
        with tab3:
            st.subheader("🔹 Selecciona Cluster")
            if 'Cluster_Manual' in self.df.columns:
                clusters_validos = self.df['Cluster_Manual'].dropna().unique()
                clusters_validos.sort()
                cluster_seleccionado = st.selectbox("Selecciona un cluster", clusters_validos)
                st.dataframe(self.df[self.df['Cluster_Manual'] == cluster_seleccionado])
            else:
                st.info("❌ No hay clusters manuales asignados. Ve al TAB 'Clusters' para generar los clusters primero.")

        # ---------- TAB 4: Código Fuente ----------
        with tab4:
            st.subheader("📄 Código Generado")

            try:
                # Obtener todo el código de la clase PaginaKmeans
                codigo_completo = inspect.getsource(PaginaKmeans)
                st.code(codigo_completo, language="python")
            except Exception as e:
                st.warning(f"Error mostrando código: {str(e)}")

