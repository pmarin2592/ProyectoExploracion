"""
Clase: PaginaAC

Objetivo: Clase que mantiene la pagina para visualizacion del ACP

Cambios:

    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
"""
import streamlit as st
from matplotlib import pyplot as plt

from src.eda.PcaEda import PcaEda


class PaginaACP:
    def __init__(self):
        if getattr(st.session_state, 'eda', None) is not None:
            try:
                self.pca = PcaEda(getattr(st.session_state, 'eda', None))
                st.session_state.pca = self.pca
            except Exception as  e:
                var = None
    def render(self):
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
                    ACP (Componentes Principales)
                </h1>
                """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
            ["Varianza", "Biplot", "Gráficos 3d","Gráficos 3d con planos","Proyección 2 componentes","Proyección 3 componentes",
             "Mapa de Calor", "Circulo de correlación", "Contribucciones", "Cuadrantes"]
        )

        with tab1:

            st.plotly_chart(self.pca.graficar_varianza())

        with tab2:
            st.plotly_chart(self.pca.biplot())

        with tab3:
            st.plotly_chart(self.pca.graficar_3d())

        with tab4:
            st.plotly_chart(self.pca.graficar_3d_con_planos())

        with tab5:
            st.plotly_chart(self.pca.graficar_proyeccion_pc1_pc2())

        with tab6:
            st.plotly_chart(self.pca.graficar_proyeccion_pc1_pc3())

        with tab7:
            st.plotly_chart(self.pca.graficar_heatmap_loadings())

        with tab8:
            st.plotly_chart(self.pca.graficar_circulo_correlacion())

        with tab9:
            st.plotly_chart(self.pca.graficar_contribuciones_variables())

        with tab10:
            st.plotly_chart(self.pca.graficar_analisis_cuadrantes())

