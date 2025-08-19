"""
Clase: PaginaACP

Objetivo: Clase que mantiene la pagina para visualizacion del ACP

Cambios:
    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
"""
import logging
import streamlit as st
from matplotlib import pyplot as plt

from src.eda.PcaEda import PcaEda

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaginaACP:
    def __init__(self):
        try:
            if getattr(st.session_state, 'eda', None) is not None:
                try:
                    self.pca = PcaEda(getattr(st.session_state, 'eda', None))
                    st.session_state.pca = self.pca
                    logger.info("PCA inicializado correctamente")
                except ValueError as ve:
                    logger.error(f"Error de validación al crear PCA: {ve}")
                    self.pca = None
                except AttributeError as ae:
                    logger.error(f"Error de atributo al crear PCA: {ae}")
                    self.pca = None
                except Exception as e:
                    logger.error(f"Error inesperado al crear PCA: {e}")
                    self.pca = None
            else:
                logger.warning("No hay objeto EDA disponible en session_state")
                self.pca = None
        except Exception as e:
            logger.error(f"Error crítico en inicialización de PaginaACP: {e}")
            self.pca = None

    def render(self):
        try:
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
                try:
                    if self.pca is not None:
                        fig, codigo = self.pca.graficar_varianza()
                        st.plotly_chart(fig)
                        try:
                            with st.expander("📋 **Ver Código Gráfico**"):
                                st.subheader("📄 Código generado:")
                                st.code(codigo, language="python")
                        except Exception as e:
                            logger.warning(f"Error en expander de código: {str(e)}")
                    else:
                        st.error("Error: No hay datos PCA disponibles")
                except ValueError as ve:
                    logger.error(f"Error de validación en graficar_varianza: {ve}")
                    st.error(f"Error de validación: {ve}")
                except Exception as e:
                    logger.error(f"Error en graficar_varianza: {e}")
                    st.error(f"Error al generar gráfico de varianza: {e}")

            with tab2:
                try:
                    if self.pca is not None:
                        fig, codigo = self.pca.biplot()
                        st.plotly_chart(fig)
                        try:
                            with st.expander("📋 **Ver Código Gráfico**"):
                                st.subheader("📄 Código generado:")
                                st.code(codigo, language="python")
                        except Exception as e:
                            logger.warning(f"Error en expander de código: {str(e)}")
                    else:
                        st.error("Error: No hay datos PCA disponibles")
                except ValueError as ve:
                    logger.error(f"Error de validación en biplot: {ve}")
                    st.error(f"Error de validación: {ve}")
                except Exception as e:
                    logger.error(f"Error en biplot: {e}")
                    st.error(f"Error al generar biplot: {e}")

            with tab3:
                try:
                    if self.pca is not None:
                        fig, codigo =self.pca.graficar_3d()
                        st.plotly_chart(fig)
                        try:
                            with st.expander("📋 **Ver Código Gráfico**"):
                                st.subheader("📄 Código generado:")
                                st.code(codigo, language="python")
                        except Exception as e:
                            logger.warning(f"Error en expander de código: {str(e)}")
                    else:
                        st.error("Error: No hay datos PCA disponibles")
                except ValueError as ve:
                    logger.error(f"Error de validación en graficar_3d: {ve}")
                    st.error(f"Error de validación: {ve}")
                except Exception as e:
                    logger.error(f"Error en graficar_3d: {e}")
                    st.error(f"Error al generar gráfico 3D: {e}")

            with tab4:
                try:
                    if self.pca is not None:
                        fig, codigo = self.pca.graficar_3d_con_planos()
                        st.plotly_chart(fig)
                        try:
                            with st.expander("📋 **Ver Código Gráfico**"):
                                st.subheader("📄 Código generado:")
                                st.code(codigo, language="python")
                        except Exception as e:
                            logger.warning(f"Error en expander de código: {str(e)}")
                    else:
                        st.error("Error: No hay datos PCA disponibles")
                except ValueError as ve:
                    logger.error(f"Error de validación en graficar_3d_con_planos: {ve}")
                    st.error(f"Error de validación: {ve}")
                except Exception as e:
                    logger.error(f"Error en graficar_3d_con_planos: {e}")
                    st.error(f"Error al generar gráfico 3D con planos: {e}")

            with tab5:
                try:
                    if self.pca is not None:
                        fig, codigo = self.pca.graficar_proyeccion_pc1_pc2()
                        st.plotly_chart(fig)
                        try:
                            with st.expander("📋 **Ver Código Gráfico**"):
                                st.subheader("📄 Código generado:")
                                st.code(codigo, language="python")
                        except Exception as e:
                            logger.warning(f"Error en expander de código: {str(e)}")
                    else:
                        st.error("Error: No hay datos PCA disponibles")
                except ValueError as ve:
                    logger.error(f"Error de validación en graficar_proyeccion_pc1_pc2: {ve}")
                    st.error(f"Error de validación: {ve}")
                except Exception as e:
                    logger.error(f"Error en graficar_proyeccion_pc1_pc2: {e}")
                    st.error(f"Error al generar proyección PC1-PC2: {e}")

            with tab6:
                try:
                    if self.pca is not None:
                        fig, codigo = self.pca.graficar_proyeccion_pc1_pc3()
                        st.plotly_chart(fig)
                        try:
                            with st.expander("📋 **Ver Código Gráfico**"):
                                st.subheader("📄 Código generado:")
                                st.code(codigo, language="python")
                        except Exception as e:
                            logger.warning(f"Error en expander de código: {str(e)}")
                    else:
                        st.error("Error: No hay datos PCA disponibles")
                except ValueError as ve:
                    logger.error(f"Error de validación en graficar_proyeccion_pc1_pc3: {ve}")
                    st.error(f"Error de validación: {ve}")
                except Exception as e:
                    logger.error(f"Error en graficar_proyeccion_pc1_pc3: {e}")
                    st.error(f"Error al generar proyección PC1-PC3: {e}")

            with tab7:
                try:
                    if self.pca is not None:
                        fig, codigo = self.pca.graficar_heatmap_loadings()
                        st.plotly_chart(fig)
                        try:
                            with st.expander("📋 **Ver Código Gráfico**"):
                                st.subheader("📄 Código generado:")
                                st.code(codigo, language="python")
                        except Exception as e:
                            logger.warning(f"Error en expander de código: {str(e)}")
                    else:
                        st.error("Error: No hay datos PCA disponibles")
                except ValueError as ve:
                    logger.error(f"Error de validación en graficar_heatmap_loadings: {ve}")
                    st.error(f"Error de validación: {ve}")
                except Exception as e:
                    logger.error(f"Error en graficar_heatmap_loadings: {e}")
                    st.error(f"Error al generar heatmap de loadings: {e}")

            with tab8:
                try:
                    if self.pca is not None:
                        fig, codigo = self.pca.graficar_circulo_correlacion()
                        st.plotly_chart(fig)
                        try:
                            with st.expander("📋 **Ver Código Gráfico**"):
                                st.subheader("📄 Código generado:")
                                st.code(codigo, language="python")
                        except Exception as e:
                            logger.warning(f"Error en expander de código: {str(e)}")
                    else:
                        st.error("Error: No hay datos PCA disponibles")
                except ValueError as ve:
                    logger.error(f"Error de validación en graficar_circulo_correlacion: {ve}")
                    st.error(f"Error de validación: {ve}")
                except Exception as e:
                    logger.error(f"Error en graficar_circulo_correlacion: {e}")
                    st.error(f"Error al generar círculo de correlación: {e}")

            with tab9:
                try:
                    if self.pca is not None:
                        fig, codigo = self.pca.graficar_contribuciones_variables()
                        st.plotly_chart(fig)
                        try:
                            with st.expander("📋 **Ver Cádigo Gráfico**"):
                                st.subheader("📄 Código generado:")
                                st.code(codigo, language="python")
                        except Exception as e:
                            logger.warning(f"Error en expander de código: {str(e)}")
                    else:
                        st.error("Error: No hay datos PCA disponibles")
                except ValueError as ve:
                    logger.error(f"Error de validación en graficar_contribuciones_variables: {ve}")
                    st.error(f"Error de validación: {ve}")
                except Exception as e:
                    logger.error(f"Error en graficar_contribuciones_variables: {e}")
                    st.error(f"Error al generar gráfico de contribuciones: {e}")

            with tab10:
                try:
                    if self.pca is not None:
                        fig, codigo = self.pca.graficar_analisis_cuadrantes()
                        st.plotly_chart(fig)
                        try:
                            with st.expander("📋 **Ver Código Gráfico**"):
                                st.subheader("📄 Código generado:")
                                st.code(codigo, language="python")
                        except Exception as e:
                            logger.warning(f"Error en expander de código: {str(e)}")
                    else:
                        st.error("Error: No hay datos PCA disponibles")
                except ValueError as ve:
                    logger.error(f"Error de validación en graficar_analisis_cuadrantes: {ve}")
                    st.error(f"Error de validación: {ve}")
                except Exception as e:
                    logger.error(f"Error en graficar_analisis_cuadrantes: {e}")
                    st.error(f"Error al generar análisis de cuadrantes: {e}")

        except Exception as e:
            logger.error(f"Error crítico al renderizar página ACP: {e}")
            st.error(f"Error crítico al cargar la página: {e}")