"""
Clase: PaginaKmeans

Objetivo: Clase que mantiene la página para visualización del análisis de clustering K-means
"""

import streamlit as st
import pandas as pd
from modelos.Kmeans import Kmeans, Clustering


class PaginaKmeans:
    def __init__(self):
        self.df = None
        self.cluster_manual = None
        self.kmeans_obj = None

        # Verificar si el dataset está cargado globalmente
        if hasattr(st.session_state, 'df_cargado') and st.session_state.df_cargado is not None:
            self.df = st.session_state.df_cargado
            self.tiene_datos = True
        else:
            self.tiene_datos = False

    def render(self):
        st.title("🎯 K-Means")

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

        self.kmeans_obj = Kmeans(datos_escalados)  # Corrección de mayúscula en Kmeans

        # ---------- K-means automático ----------
        st.subheader("Análisis K-means Automático")

        inercias = self.kmeans_obj.evaluar_varios_k()
        fig_codo = self.kmeans_obj.graficar_metricas(inercias)
        st.pyplot(fig_codo)

        k_opt = st.slider("Seleccione el número de clusters (K)", 2, 10, 3)
        st.info(f"Has seleccionado {k_opt} clusters para aplicar K-means.")

        modelo_final, etiquetas_kmeans = self.kmeans_obj.entrenar(k_opt)
        self.df['Cluster_KMeans'] = etiquetas_kmeans

        st.dataframe(self.df[['Cluster_KMeans'] + columnas_numericas].head())

        # ---------- Clustering manual (Jerárquico) ----------
        st.subheader("Clustering Manual (Jerárquico)")

        k_cluster = st.slider("Elige número de clusters para el análisis manual", 2, 10, 3, key="slider_clustering")
        resultados = self.cluster_manual.clustering_kmeans(k=k_cluster)

        self.df['Cluster_Manual'] = resultados['etiquetas']

        fig_clusters = self.cluster_manual.graficar_clusters_kmeans(resultados['etiquetas'])
        st.pyplot(fig_clusters)

        st.dataframe(self.df[['Cluster_Manual'] + columnas_numericas].head())

        # Gráfico radar para clusters manuales
        st.subheader("Perfil tipo radar por Cluster (Manual)")
        fig_radar = self.cluster_manual.graficar_radar_por_cluster(resultados['etiquetas'], titulo="Radar - Clustering Manual")
        st.pyplot(fig_radar)
