import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import inspect
import logging

from src.eda.RegresionEDA import RegresionEda
from src.helpers.UtilDataFrame import UtilDataFrame
from src.helpers.ComponentesUI import ComponentesUI, ConfigMetrica

logger = logging.getLogger(__name__)


class PaginaRegresion:
    def __init__(self):
        pass

    def _obtener_df(self):
        """Obtiene el DataFrame desde la sesión con control de errores"""
        try:
            return getattr(st.session_state, "df_cargado", None)
        except Exception as e:
            logger.error(f"Error obteniendo DataFrame: {e}")
            return None

    def mostrar_eda_basico(self, df, x_vars, y_var, tipo):
        st.subheader(f"Exploración de datos (EDA) - {tipo}")

        try:
            with st.expander("Resumen estadístico de las variables seleccionadas"):
                cols = list(filter(lambda c: c in df.columns, x_vars + ([y_var] if y_var else [])))
                if not cols:
                    st.write("No hay columnas válidas seleccionadas.")
                    return
                st.write(df[cols].describe(include="all"))
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error mostrando resumen estadístico: {e}")

        try:
            with st.expander("Distribuciones"):
                for idx, col in enumerate(x_vars + ([y_var] if y_var else [])):
                    if col not in df.columns:
                        continue
                    st.markdown(f"**{col}**")
                    key_base = f"eda_{tipo}_{col}_{idx}"
                    try:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fig = px.histogram(df, x=col, marginal="box", nbins=30, title=f"Histograma de {col}")
                            st.plotly_chart(fig, use_container_width=True, key=f"{key_base}_hist")
                        else:
                            vc = df[col].value_counts().reset_index()
                            vc.columns = [col, "conteo"]
                            fig = px.bar(vc, x=col, y="conteo", title=f"Distribución de {col}")
                            st.plotly_chart(fig, use_container_width=True, key=f"{key_base}_bar")
                    except Exception as e:
                        ComponentesUI.mostrar_error(f"Error generando gráfico para {col}: {e}")
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error mostrando distribuciones: {e}")

    def mostrar_codigo_funcion(self, codigo: str):
        """Muestra el código fuente recibido dentro de un expander con control de errores"""
        try:
            with st.expander("📋 **Ver Código Gráfico**"):
                st.subheader("📄 Código generado:")
                st.code(codigo, language="python")
        except Exception as e:
            logger.warning(f"Error mostrando código: {str(e)}")

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
                Modelos de Regresión
            </h1>
        """, unsafe_allow_html=True)

        df = self._obtener_df()
        if df is None:
            st.error("No hay dataset cargado. Ve a 'Datos' y carga uno primero.")
            return

        try:
            df = UtilDataFrame.corregir_dataframe_para_streamlit(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error procesando el DataFrame: {e}")
            return

        if len(numeric_cols) < 2:
            st.warning("El dataset necesita al menos 2 variables numéricas para aplicar regresión.")
            return

        # ========= 🔑 Reinicialización si cambia el dataset =========
        dataset_signature = (tuple(df.columns), df.shape)
        if "dataset_signature" not in st.session_state or st.session_state.dataset_signature != dataset_signature:
            st.session_state.dataset_signature = dataset_signature

            # Reiniciar variables de regresión
            st.session_state.rls_x = numeric_cols[0]
            st.session_state.rls_y = numeric_cols[1]
            st.session_state.rlm_y = numeric_cols[0]
            st.session_state.rlm_xvars = []

        # Inicializar el objeto de EDA con el DataFrame
        reg_eda = RegresionEda(df)

        tab1, tab2 = st.tabs(["Regresión lineal simple", "Regresión lineal múltiple"])

        # ==================== REGRESIÓN LINEAL SIMPLE ====================
        with tab1:
            try:
                y = st.selectbox("Variable objetivo (Y)", numeric_cols,
                                 index=numeric_cols.index(st.session_state.rls_y),
                                 key="rls_y_select")

                # Excluir la variable Y de las opciones de X
                x_options = [c for c in numeric_cols if c != y]
                x = st.selectbox("Variable predictora (X)", x_options,
                                 index=x_options.index(st.session_state.rls_x) if st.session_state.rls_x in x_options else 0,
                                 key="rls_x_select")

                st.session_state.rls_x = x
                st.session_state.rls_y = y

                if x and y and x != y:
                    self.mostrar_eda_basico(df, [x], y, tipo="Lineal_simple")

                if st.button("Ejecutar regresión lineal simple", key="btn_rls"):
                    try:
                        res, codigo = reg_eda.regresion_lineal_simple(x, y)

                        metricas = [
                            ("", "#45B7D1", f"{res['resultados']['coeficiente']:.3f}", "Coeficiente"),
                            ("", "#96CEB4", f"{res['resultados']['intercepto']:.3f}", "Intercepto"),
                            ("", "#4ECDC4", f"{res['resultados']['R2_entrenamiento']:.3f}", "R² Entrenamiento"),
                            ("", "#FF6B6B", f"{res['resultados']['R2_prueba']:.3f}", "R² Prueba"),
                            ("", "#FFEAA7", f"{res['resultados']['MSE_entrenamiento']:.3f}", "MSE Entrenamiento"),
                            ("", "#FFB347", f"{res['resultados']['MSE_prueba']:.3f}", "MSE Prueba"),
                            ("", "#C8A2C8", f"{res['resultados']['RMSE_entrenamiento']:.3f}", "RMSE Entrenamiento"),
                            ("", "#FF7F50", f"{res['resultados']['RMSE_prueba']:.3f}", "RMSE Prueba"),
                        ]
                        cols = st.columns(len(metricas))
                        for col, (emoji, color, valor, label) in zip(cols, metricas):
                            config = ConfigMetrica(emoji, color)
                            col.markdown(ComponentesUI.crear_metrica_visual(config, valor, label),
                                         unsafe_allow_html=True)

                        fig = px.scatter(x=res["X_test"], y=res["Y_test"], labels={"x": x, "y": y},
                                         title=f"{y} vs {x} (conjunto de prueba)")
                        line_df = pd.DataFrame({x: res["X_test"], y: res["y_pred_test"]})
                        fig.add_traces(px.line(line_df, x=x, y=y).data)
                        st.plotly_chart(fig, use_container_width=True, key=f"rls_plot_{x}_{y}")

                        # Mostrar código fuente
                        self.mostrar_codigo_funcion(codigo)

                    except Exception as e:
                        ComponentesUI.mostrar_error(f"Error en regresión lineal simple: {e}")
            except Exception as e:
                ComponentesUI.mostrar_error(f"Error configurando regresión simple: {e}")

        # ==================== REGRESIÓN LINEAL MÚLTIPLE ====================
        with tab2:
            try:
                y = st.selectbox("Variable objetivo (Y)", numeric_cols,
                                 index=numeric_cols.index(st.session_state.rlm_y), key="rlm_y_select")
                x_vars = st.multiselect(
                    "Variables predictoras (X)",
                    [c for c in numeric_cols if c != y],
                    key="rlm_xvars_select"
                )
                st.session_state.rlm_y = y
                st.session_state.rlm_xvars = x_vars

                if x_vars:
                    self.mostrar_eda_basico(df, x_vars, y, tipo="Lineal_multiple")

                if st.button("Ejecutar regresión lineal múltiple", key="btn_rlm"):
                    try:
                        res, codigo = reg_eda.regresion_lineal_multiple(x_vars, y)

                        metricas = [
                            ("", "#45B7D1", str(len(res['resultados']['coeficientes'])), "N° Variables"),
                            ("", "#96CEB4", f"{res['resultados']['intercepto']:.3f}", "Intercepto"),
                            ("", "#4ECDC4", f"{res['resultados']['R2_entrenamiento']:.3f}", "R² Entrenamiento"),
                            ("", "#FF6B6B", f"{res['resultados']['R2_prueba']:.3f}", "R² Prueba"),
                            ("", "#FFEAA7", f"{res['resultados']['MSE_entrenamiento']:.3f}", "MSE Entrenamiento"),
                            ("", "#FFB347", f"{res['resultados']['MSE_prueba']:.3f}", "MSE Prueba"),
                            ("", "#C8A2C8", f"{res['resultados']['RMSE_entrenamiento']:.3f}", "RMSE Entrenamiento"),
                            ("", "#FF7F50", f"{res['resultados']['RMSE_prueba']:.3f}", "RMSE Prueba"),
                        ]
                        cols = st.columns(len(metricas))
                        for col, (emoji, color, valor, label) in zip(cols, metricas):
                            config = ConfigMetrica(emoji, color)
                            col.markdown(ComponentesUI.crear_metrica_visual(config, valor, label),
                                         unsafe_allow_html=True)

                        # Gráficos individuales
                        for idx, xi in enumerate(x_vars):
                            try:
                                df_pred = df[x_vars].copy().fillna(df[x_vars].mean())
                                for var in x_vars:
                                    if var != xi:
                                        df_pred[var] = df_pred[var].mean()
                                y_pred_line = res["modelo"].predict(df_pred)

                                fig = px.scatter(x=df[xi], y=df[y], labels={"x": xi, "y": y}, title=f"{y} vs {xi}")
                                line_df = pd.DataFrame({xi: df[xi], y: y_pred_line}).sort_values(by=xi)
                                fig.add_traces(px.line(line_df, x=xi, y=y).data)
                                st.plotly_chart(fig, use_container_width=True, key=f"rlm_plot_{xi}_{idx}")
                            except Exception as e:
                                ComponentesUI.mostrar_error(f"Error generando gráfico para {xi}: {e}")

                        # Mostrar código fuente
                        self.mostrar_codigo_funcion(codigo)

                    except Exception as e:
                        ComponentesUI.mostrar_error(f"Error en regresión lineal múltiple: {e}")
            except Exception as e:
                ComponentesUI.mostrar_error(f"Error configurando regresión múltiple: {e}")
