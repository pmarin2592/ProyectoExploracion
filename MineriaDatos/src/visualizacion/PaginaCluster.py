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
        if 'eda' in st.session_state and st.session_state['eda'] is not None:
            try:
                self.df = st.session_state['eda'].df  # <- accedemos al DataFrame real
            except Exception as e:
                st.error(f"Error al cargar datos del EDA: {e}")

    def render(self):
        st.set_page_config(page_title="Análisis de Clustering Jerárquico", layout="wide")
        # Título principal de la página
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
                           Análisis de Clustering Jerárquico
                       </h1>
                       """, unsafe_allow_html=True)
        st.markdown("---")

        if self.df is None:
            st.warning("No se encontraron datos para el análisis.")
            return

        try:
            analyzer = ClusterJerarquico(self.df)
            info = analyzer.escalar_datos()

            st.write(f"📊 Datos escalados: {info['n_samples']} muestras, {info['n_features']} características")
            st.markdown("---")

            st.subheader("🌳 Dendrogramas")

            selected_methods = st.multiselect(
                "Seleccionar métodos de linkage:",
                ['ward', 'complete', 'average', 'single'],
                default=['ward', 'complete']
            )

            if selected_methods and st.button("Generar Dendrogramas"):
                with st.spinner("Generando dendrogramas..."):
                    fig_dendro = analyzer.plot_dendrograms(methods=selected_methods)
                    st.pyplot(fig_dendro)

        except Exception as e:
            st.error(f"Error en el análisis de clustering: {e}")