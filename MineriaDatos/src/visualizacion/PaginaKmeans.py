import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modelos.Kmeans import Kmeans, Clustering
from kneed import KneeLocator  # <-- importamos kneed

class PaginaKmeans:
    def __init__(self):
        self.df = None
        self.cluster_manual = None
        self.kmeans_obj = None

        # Verificar si el dataset está cargado
        if hasattr(st.session_state, 'df_cargado') and st.session_state.df_cargado is not None:
            self.df = st.session_state.df_cargado.copy()
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
            '>
                K-Means
            </h1>
        """, unsafe_allow_html=True)

        if not self.tiene_datos:
            st.warning("❌ No hay dataset cargado. Por favor, carga un dataset primero.")
            return

        # Limpiar dataset: eliminar filas y columnas duplicadas
        self.df = self.df.dropna(how='all')
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]

        # Columnas numéricas
        columnas_numericas = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(columnas_numericas) < 2:
            st.warning("El dataset debe tener al menos 2 columnas numéricas para clustering.")
            return

        # Instanciar Clustering manual
        self.cluster_manual = Clustering(self.df)
        datos_escalados, columnas_escaladas = self.cluster_manual.limpiar_y_escalar(columnas_numericas)
        if datos_escalados is None:
            st.warning("Error al escalar las variables numéricas.")
            return

        self.kmeans_obj = Kmeans(datos_escalados)

        # ===== Crear pestañas para cada sección =====
        tab1, tab2, tab3, tab4 = st.tabs(["K-means Automático", "Clustering Manual", "Datos", "Selecciona Cluster"])

        # ---------- TAB 1: Codo Jambu ----------
        with tab1:
            st.subheader("🔹 Codo Jambu")
            inercias = self.kmeans_obj.evaluar_varios_k()
            k_range = list(range(2, len(inercias)+2))

            # Detectar el codo usando kneed
            kneedle = KneeLocator(k_range, inercias, curve='convex', direction='decreasing')
            k_opt = kneedle.knee
            inercia_opt = inercias[k_opt - 2] if k_opt else inercias[0]

            # Gráfico interactivo con punto rojo
            fig_codo = go.Figure()
            fig_codo.add_trace(go.Scatter(
                x=k_range,
                y=inercias,
                mode='lines+markers',
                line=dict(color='royalblue'),
                marker=dict(size=8),
                name='Inercia'
            ))
            if k_opt:
                fig_codo.add_trace(go.Scatter(
                    x=[k_opt],
                    y=[inercia_opt],
                    mode='markers+text',
                    marker=dict(color='red', size=12),
                    text=["Codo"],
                    textposition="top center",
                    name='K óptimo'
                ))
            fig_codo.update_layout(
                title='📉 Codo de Jambú - Número óptimo de clusters',
                xaxis_title='Número de clusters (k)',
                yaxis_title='Inercia',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig_codo, use_container_width=True)

            if k_opt:
                st.info(f"🔴 Se ha detectado automáticamente {k_opt} clusters como óptimos para aplicar K-means.")

            # Entrenar K-means con k óptimo
            if k_opt:
                modelo_final, etiquetas_kmeans = self.kmeans_obj.entrenar(k_opt)
                etiquetas_completas_kmeans = pd.Series(index=self.df.index, dtype="float64")
                etiquetas_completas_kmeans[self.cluster_manual.df_numerico.index] = etiquetas_kmeans

                if 'Cluster_KMeans' in self.df.columns:
                    self.df.drop(columns=['Cluster_KMeans'], inplace=True)
                self.df['Cluster_KMeans'] = etiquetas_completas_kmeans

        # ---------- TAB 2: Clusters ----------
        with tab2:
            st.subheader("🔹 Clusters ")

            k_cluster = st.slider("Elige número de clusters para el análisis manual", 2, 10, 3, key="slider_clustering")
            resultados = self.cluster_manual.clustering_kmeans(k=k_cluster)

            etiquetas_completas_manual = pd.Series(index=self.df.index, dtype="float64")
            etiquetas_completas_manual[self.cluster_manual.df_numerico.index] = resultados['etiquetas']

            if 'Cluster_Manual' in self.df.columns:
                self.df.drop(columns=['Cluster_Manual'], inplace=True)
            self.df['Cluster_Manual'] = etiquetas_completas_manual

            # ===== Graficar con Plotly y mostrar conteo por cluster =====
            df_plot = self.cluster_manual.df_numerico.copy()
            df_plot['Cluster'] = resultados['etiquetas']
            columnas_graf = df_plot.columns[:2]  # usar primeras 2 columnas numéricas para el scatter

            fig_clusters = go.Figure()
            for cluster in sorted(df_plot['Cluster'].unique()):
                df_cluster = df_plot[df_plot['Cluster'] == cluster]
                fig_clusters.add_trace(go.Scatter(
                    x=df_cluster[columnas_graf[0]],
                    y=df_cluster[columnas_graf[1]],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(size=8)
                ))

            # Anotaciones tipo "legend" en esquina superior izquierda
            conteos = df_plot['Cluster'].value_counts().to_dict()
            anotaciones = []
            offset = 0
            for cluster, count in sorted(conteos.items()):
                anotaciones.append(dict(
                    x=-0.02,  # fuera del área de trazado, esquina izquierda
                    y=1 - offset,  # descendiendo por cluster
                    xref='paper',
                    yref='paper',
                    text=f"Cluster {cluster}: {count} pts",
                    showarrow=False,
                    font=dict(size=12, color='black'),
                    align='left'
                ))
                offset += 0.08  # espacio entre filas

            fig_clusters.update_layout(
                title="Clustering Manual",
                xaxis_title=columnas_graf[0],
                yaxis_title=columnas_graf[1],
                annotations=anotaciones,
                template='plotly_white',
                height=500,
                margin=dict(l=120)  # espacio a la izquierda para las anotaciones
            )

            st.plotly_chart(fig_clusters, use_container_width=True)

        # ---------- TAB 3: Datos ----------
        with tab3:
            st.subheader("🔹 Datos ")
            st.dataframe(self.df[['Cluster_Manual'] + columnas_numericas])

        # ---------- TAB 4: Selecciona Cluster ----------
        with tab4:
            st.subheader("🔹 Selecciona Cluster")
            cluster_seleccionado = st.selectbox("Selecciona un cluster", self.df['Cluster_Manual'].unique())
            st.dataframe(self.df[self.df['Cluster_Manual'] == cluster_seleccionado])
