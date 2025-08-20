"""
Clase: PaginaArbolDecision

Objetivo: Página para visualización y generación de código del Árbol de Decisión
"""

import streamlit as st
import pandas as pd
import logging
from src.eda.ArbolDecisionEda import ArbolDecisionEda
from src.visualizacion.PaginaDatos import PaginaDatos
from kneed import KneeLocator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaginaArbolDecision(PaginaDatos):
    def __init__(self):
        super().__init__()
        self.tiene_datos = (hasattr(st.session_state, 'df_cargado') and
                           st.session_state.df_cargado is not None)

    def render(self):
        st.title("🌳 Clasificación con Árbol de Decisión")

        if not self.tiene_datos:
            st.error("❌ No hay dataset cargado. Por favor, carga un dataset primero.")
            return

        df = st.session_state.df_cargado  # df de PaginaDatos

        st.write("Vista previa del dataset cargado:")
        self._crear_tabla_con_tooltips(df.head())

        col1, col2 = st.columns(2)

        with col1:
            target_col = st.selectbox("Variable objetivo:", df.columns)

        with col2:
            aplicar_binning = st.checkbox("Aplicar binning a variables continuas", value=True)
            if aplicar_binning:
                n_bins = st.slider("Número de bins:", 3, 10, 5)
            else:
                n_bins = 5

        cols_a_excluir = st.multiselect(
            "Columnas a excluir:",
            options=[col for col in df.columns if col != target_col]
        )

        columnas_predictoras = [
            col for col in df.columns if col not in cols_a_excluir + [target_col]
        ]
        st.info(f"Usando {len(columnas_predictoras)} variables predictoras")

        if len(columnas_predictoras) == 0:
            st.error("❌ Se excluyeron todas las variables. Debes elegir al menos una columna para predecir.")
            return

        try:
            df_modelo = df[columnas_predictoras + [target_col]].copy()
        except KeyError as e:
            st.error(f"❌ Error en selección de columnas: {e}")
            return

        if st.button("Entrenar árbol de decisión"):
            try:
                # === Entrenamiento ===
                arbol = ArbolDecisionEda(
                    df_modelo,
                    target_col,
                    aplicar_binning,
                    n_bins
                )

                st.success("Modelo entrenado exitosamente! ✅")


                # === Pestañas con resultados ===
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Matriz Confusión",
                    "🌳 Árbol",
                    "📈 Importancia",
                    "📋 Evaluación"
                ])

                with tab1:
                    fig, codigo_funcion = arbol.graficar_matriz_confusion()
                    st.pyplot(fig)

                    try:
                        with st.expander("📋 **Ver Código Gráfico**"):
                            st.subheader("📄 Código generado:")
                            st.code(codigo_funcion, language="python")
                    except Exception as e:
                        logger.warning(f"Error en expander de código: {str(e)}")

                with tab2:
                    fig, codigo_funcion = arbol.graficar_arbol()
                    st.pyplot(fig)

                    try:
                        with st.expander("📋 **Ver Código Gráfico**"):
                            st.subheader("📄 Código generado:")
                            st.code(codigo_funcion, language="python")
                    except Exception as e:
                        logger.warning(f"Error en expander de código: {str(e)}")

                with tab3:
                    fig, codigo_funcion = arbol.graficar_importancia_variables()
                    st.pyplot(fig)

                    try:
                        with st.expander("📋 **Ver Código Gráfico**"):
                            st.subheader("📄 Código generado:")
                            st.code(codigo_funcion, language="python")
                    except Exception as e:
                        logger.warning(f"Error en expander de código: {str(e)}")

                with tab4:
                    resultados, codigo_funcion = arbol.evaluar()
                    acc, rep, conf = resultados

                    st.metric("Precisión", f"{acc:.3f}")
                    st.subheader("Reporte de Clasificación")
                    rep_df = pd.DataFrame(rep).transpose()
                    st.dataframe(rep_df)

                    try:
                        with st.expander("📋 **Ver Código Evaluación**"):
                            st.subheader("📄 Código generado:")
                            st.code(codigo_funcion, language="python")
                    except Exception as e:
                        logger.warning(f"Error en expander de código: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error: {e}")
