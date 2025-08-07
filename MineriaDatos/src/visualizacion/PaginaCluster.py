"""
Clase: PaginaCluster

Objetivo: Clase que mantiene la pagina para visualizacion del Cluster Jerarquico

Cambios:

    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
"""
import streamlit as st
from src.modelos.ClusterJerarquico import ClusterJerarquico

class PaginaCluster:
    def __init__(self):
        self.df = None
        if 'pca' in st.session_state and st.session_state['pca'] is not None:
            try:
                self.pca = getattr(st.session_state, 'pca', None)
                self.df = self.pca.eda.df
            except Exception as e:
                st.error(f"Error al cargar datos: {e}")

    def render(self):
        st.set_page_config(page_title="Clustering Jerárquico", layout="wide")
        st.title("🔗 Análisis de Clustering Jerárquico")
        st.markdown("---")

        if self.df is None:
            st.warning("No hay datos para analizar.")
            return


        try:
            analyzer = ClusterJerarquico(self.pca)
            n_samples, n_features = analyzer.cluster_data_scaled.shape
            st.write(f"📊 Datos escalados: {n_samples} muestras, {n_features} características")
            st.markdown("---")

            st.subheader("🌳 Dendrogramas")
            selected_methods = st.multiselect(
                "Seleccionar métodos de linkage:",
                ['ward', 'complete', 'average', 'single'],
                default=['ward', 'complete']
            )

            if selected_methods and st.button("Generar Dendrogramas"):
                with st.spinner("Generando dendrogramas..."):
                    fig = analyzer.plot_dendrograms(methods=selected_methods)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error en el análisis de clustering: {e}")
