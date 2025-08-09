"""
Clase: PaginaArbolDecision

Objetivo: Clase que mantiene la página para visualización del Árbol de Decisión

Cambios:
    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
    2. INtegración Final Funcional del Arbol de Decisión ggranados 31-07-2025
"""
import streamlit as st
import pandas as pd
from src.eda.ArbolDecisionEda import ArbolDecisionEda

class PaginaArbolDecision:
    def __init__(self):
        if getattr(st.session_state, 'eda', None) is not None:
            try:
                # Verificar que hay un dataset cargado
                if hasattr(st.session_state, 'df_cargado') and st.session_state.df_cargado is not None:
                    self.tiene_datos = True
                else:
                    self.tiene_datos = False
            except Exception as e:
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
                                          Clasificación con Árbol de Decisión
                                      </h1>
                                      """, unsafe_allow_html=True)

        if not self.tiene_datos:
            st.error("❌ No hay dataset cargado. Por favor, carga un dataset primero.")
            return

        df = st.session_state.df_cargado

        # Vista previa del dataset
        st.write("Vista previa del dataset cargado:")
        st.dataframe(df.head(), use_container_width=True)

        # Configuración básica
        col1, col2 = st.columns(2)

        with col1:
            target_col = st.selectbox("Variable objetivo:", df.columns)

        with col2:
            aplicar_binning = st.checkbox("Aplicar binning a variables continuas", value=True)
            if aplicar_binning:
                n_bins = st.slider("Número de bins:", 3, 10, 5)
            else:
                n_bins = 5

        # Columnas a excluir
        cols_a_excluir = st.multiselect(
            "Columnas a excluir:",
            options=[col for col in df.columns if col != target_col]
        )

        # Variables predictoras finales
        columnas_predictoras = [col for col in df.columns if col not in cols_a_excluir + [target_col]]
        st.info(f"Usando {len(columnas_predictoras)} variables predictoras")

        # Subset del DataFrame
        df_modelo = df[columnas_predictoras + [target_col]].copy()

        # Entrenamiento

        if st.button("Entrenar árbol de decisión"):
            try:
                # Crear el clasificador usando el patrón similar a PCA
                arbol = ArbolDecisionEda(
                    getattr(st.session_state, 'eda', None),
                    df_modelo,
                    target_col,
                    aplicar_binning,
                    n_bins
                )

                st.success("Modelo entrenado exitosamente!")

                # Crear tabs como en el patrón ACP
                tab1, tab2, tab3, tab4 = st.tabs([
                    " Matriz Confusión",
                    " Árbol",
                    " Importancia",
                    " Evaluación"
                ])

                with tab1:
                    st.pyplot(arbol.graficar_matriz_confusion())

                with tab2:
                    st.pyplot(arbol.graficar_arbol())

                with tab3:
                    st.pyplot(arbol.graficar_importancia_variables())

                with tab4:
                    # Mostrar métricas de evaluación
                    acc, rep, conf = arbol.evaluar()
                    st.metric("Precisión", f"{acc:.3f}")

                    st.subheader("Reporte de Clasificación")
                    rep_df = pd.DataFrame(rep).transpose()
                    st.dataframe(rep_df)

            except Exception as e:
                st.error(f"❌ Error: {e}")