'''
Clase: PaginaKmeans

Objetivo: Clase que mantiene la página para visualización del análisis de clustering K-means
'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modelos.Kmeans import Kmeans, Clustering
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import plotly.express as px

class PaginaKmeans:
    def __init__(self):
        self.df = None
        self.cluster_manual = None
        self.kmeans_obj = None

        # Verificar si el dataset está cargado localmente
        if hasattr(st.session_state, 'df_cargado') and st.session_state.df_cargado is not None:
            self.df = st.session_state.df_cargado
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

        # Columnas numéricas para clustering
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

        # ---------- K-means automático ----------
        st.subheader("Análisis K-means Automático")

        inercias = self.kmeans_obj.evaluar_varios_k()
        fig_codo = self.kmeans_obj.graficar_metricas(inercias)

        # 🔄 Mostrar gráfico de codo interactivo
        st.plotly_chart(fig_codo, use_container_width=True)

        # Estimar K óptimo automáticamente con heurística simple (codo)
        deltas = [inercias[i] - inercias[i + 1] for i in range(len(inercias) - 1)]
        if deltas:
            k_opt = deltas.index(max(deltas)) + 2  # +2 porque el K inicial suele ser 2
        else:
            k_opt = 3  # fallback por si no se puede calcular

        st.info(f"Se ha detectado automáticamente {k_opt} clusters como óptimos para aplicar K-means.")

        modelo_final, etiquetas_kmeans = self.kmeans_obj.entrenar(k_opt)

        # Crear Serie con NaN para mantener tamaño original
        etiquetas_completas_kmeans = pd.Series(index=self.df.index, dtype="float64")
        etiquetas_completas_kmeans[self.cluster_manual.df_numerico.index] = etiquetas_kmeans
        self.df['Cluster_KMeans'] = etiquetas_completas_kmeans

        # ---------- Clustering manual ----------
        st.subheader("Clustering Manual")

        k_cluster = st.slider("Elige número de clusters para el análisis manual", 2, 10, 3, key="slider_clustering")
        resultados = self.cluster_manual.clustering_kmeans(k=k_cluster)

        etiquetas_completas_manual = pd.Series(index=self.df.index, dtype="float64")
        etiquetas_completas_manual[self.cluster_manual.df_numerico.index] = resultados['etiquetas']
        self.df['Cluster_Manual'] = etiquetas_completas_manual

        fig_clusters = self.cluster_manual.graficar_clusters_kmeans(resultados['etiquetas'])

        # 🔄 Reemplazo de matplotlib por Plotly
        st.plotly_chart(fig_clusters, use_container_width=True)

        st.dataframe(self.df[['Cluster_Manual'] + columnas_numericas].head())



