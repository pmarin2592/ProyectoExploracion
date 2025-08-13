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
        self.pca = None

        if 'pca' in st.session_state and st.session_state['pca'] is not None:
            try:
                self.pca = getattr(st.session_state, 'pca', None)
                self.df = self.pca.eda.df
            except Exception as e:
                st.error(f"Error al cargar datos: {e}")

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
            st.warning("⚠️ No hay datos para analizar.")
            return

        try:
            analizador = ClusterJerarquico(self.pca)
            n_muestras, n_caracteristicas = analizador.datos_escalados_cluster.shape
            st.write(f"📊 Datos escalados: **{n_muestras} muestras**, **{n_caracteristicas} características**")
            st.markdown("---")

            st.subheader("🌳 Dendrogramas Interactivos")
            metodos_seleccionados = st.multiselect(
                "Seleccionar métodos de linkage:",
                ['ward', 'complete', 'average', 'single'],
                default=['ward', 'complete']
            )

            # Por defecto siempre mostrar líneas de corte, sin checkbox para ocultar
            mostrar_cortes = True

            if metodos_seleccionados and st.button("Generar Dendrogramas"):
                with st.spinner("Generando dendrogramas interactivos..."):
                    fig = analizador.dendrogramas_interactivos(
                        metodos=metodos_seleccionados,
                        mostrar_cortes=mostrar_cortes
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error en el análisis de clustering: {e}")