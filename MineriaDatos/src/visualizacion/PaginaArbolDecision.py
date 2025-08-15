"""
Clase: PaginaArbolDecision

Objetivo: Página para visualización del Árbol de Decisión
    1. Se usa df cargado en PaginaDatos
    2. Se hace el historial de código generado desde GeneradorCodigoAbrol
"""

import streamlit as st
import pandas as pd
from src.eda.ArbolDecisionEda import ArbolDecisionEda
from src.visualizacion.PaginaDatos import PaginaDatos
from src.helpers.GeneradorCodigoArbol import GeneradorCodigoArbol

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
            n_bins = st.slider("Número de bins:", 3, 10, 5) if aplicar_binning else 5

        cols_a_excluir = st.multiselect(
            "Columnas a excluir:",
            options=[col for col in df.columns if col != target_col]
        )

        columnas_predictoras = [col for col in df.columns if col not in cols_a_excluir + [target_col]]
        st.info(f"Usando {len(columnas_predictoras)} variables predictoras")

        df_modelo = df[columnas_predictoras + [target_col]].copy()

        if st.button("Entrenar árbol de decisión"):
            try:
                arbol = ArbolDecisionEda(df_modelo, target_col, aplicar_binning, n_bins)
                st.success("Modelo entrenado exitosamente!")

                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Matriz Confusión",
                    "🌳 Árbol",
                    "📈 Importancia",
                    "📋 Evaluación",
                    "💻 Código Generado"
                ])

                with tab1:
                    st.pyplot(arbol.graficar_matriz_confusion())

                with tab2:
                    st.pyplot(arbol.graficar_arbol())

                with tab3:
                    st.pyplot(arbol.graficar_importancia_variables())

                with tab4:
                    acc, rep, conf = arbol.evaluar()
                    st.metric("Precisión", f"{acc:.3f}")
                    st.subheader("Reporte de Clasificación")
                    rep_df = pd.DataFrame(rep).transpose()
                    st.dataframe(rep_df)

                with tab5:
                    if "historial_codigo_arbol" not in st.session_state:
                        st.session_state.historial_codigo_arbol = []

                    generador = GeneradorCodigoArbol(
                        target_col=target_col,
                        columnas_predictoras=columnas_predictoras,
                        aplicar_binning=aplicar_binning,
                        n_bins=n_bins
                    )
                    codigo_actual = generador.generar_codigo()

                    # Agregar al historial
                    st.session_state.historial_codigo_arbol.append(codigo_actual)

                    # Mostrar solo el último bloque generado
                    st.subheader("📄 Código generado (última ejecución):")
                    st.code(codigo_actual, language="python")

                    # Unir todo el historial para descarga
                    codigo_completo = "\n\n".join(st.session_state.historial_codigo_arbol)
                    st.download_button(
                        label="Descargar todo el código .py",
                        data=codigo_completo.encode("utf-8"),
                        file_name="codigo_Arbol_Decision.py",
                        mime="text/x-python"
                    )

            except Exception as e:
                st.error(f"❌ Error: {e}")
