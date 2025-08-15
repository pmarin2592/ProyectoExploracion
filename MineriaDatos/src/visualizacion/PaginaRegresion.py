import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Importamos módulos personalizados
from modelos.Regresion import Regresion
from src.helpers.UtilDataFrame import UtilDataFrame
from src.helpers.ComponentesUI import ComponentesUI, ConfigMetrica


class PaginaRegresion:
    def __init__(self):
        pass

    def _obtener_df(self):
        return getattr(st.session_state, "df_cargado", None)

    def mostrar_eda_basico(self, df, x_vars, y_var, tipo):
        st.subheader("Exploración de datos (EDA) rápida")

        with st.expander("Resumen estadístico de las variables seleccionadas"):
            cols = list(filter(lambda c: c in df.columns, x_vars + ([y_var] if y_var else [])))
            if not cols:
                st.write("No hay columnas válidas seleccionadas.")
                return
            st.write(df[cols].describe(include="all"))

        with st.expander("Distribuciones"):
            for col in x_vars + ([y_var] if y_var else []):
                if col not in df.columns:
                    continue
                st.markdown(f"**{col}**")
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.histogram(df, x=col, marginal="box", nbins=30, title=f"Histograma de {col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    vc = df[col].value_counts().reset_index()
                    vc.columns = [col, "conteo"]
                    fig = px.bar(vc, x=col, y="conteo", title=f"Distribución de {col}")
                    st.plotly_chart(fig, use_container_width=True)

        if tipo == "Logística" and y_var:
            st.subheader(f"Distribución de la variable objetivo ({y_var})")
            if y_var in df.columns:
                counts = df[y_var].value_counts().reset_index()
                counts.columns = ["clase", "conteo"]
                fig = px.bar(counts, x="clase", y="conteo", title="Clases en variable objetivo")
                st.plotly_chart(fig, use_container_width=True)

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
            st.error(" No hay dataset cargado. Ve a 'Datos' y carga uno primero.")
            return

        df = UtilDataFrame.corregir_dataframe_para_streamlit(df)

        # Estado inicial
        if "regresion_tipo" not in st.session_state:
            st.session_state.regresion_tipo = "Regresión lineal simple"
        if "regresion_x" not in st.session_state:
            st.session_state.regresion_x = None
        if "regresion_y" not in st.session_state:
            st.session_state.regresion_y = None
        if "regresion_x_vars" not in st.session_state:
            st.session_state.regresion_x_vars = []
        if "regresion_logistica_y" not in st.session_state:
            st.session_state.regresion_logistica_y = None
        if "regresion_logistica_x_vars" not in st.session_state:
            st.session_state.regresion_logistica_x_vars = []

        modelo_tipo = st.selectbox(
            "Selecciona tipo de modelo",
            ["Regresión lineal simple", "Regresión lineal múltiple", "Regresión logística"],
            index=["Regresión lineal simple", "Regresión lineal múltiple", "Regresión logística"].index(
                st.session_state.regresion_tipo
            ),
        )
        st.session_state.regresion_tipo = modelo_tipo
        st.markdown("---")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # ----------------- REGRESIÓN LINEAL SIMPLE -----------------
        if modelo_tipo == "Regresión lineal simple":
            x = st.selectbox(
                "Variable predictora (X)",
                numeric_cols,
                index=numeric_cols.index(st.session_state.regresion_x)
                if st.session_state.regresion_x in numeric_cols
                else 0,
            )
            y = st.selectbox(
                "Variable objetivo (Y)",
                numeric_cols,
                index=numeric_cols.index(st.session_state.regresion_y)
                if st.session_state.regresion_y in numeric_cols
                else 0,
            )
            st.session_state.regresion_x = x
            st.session_state.regresion_y = y

            if x and y:
                if x == y:
                    st.warning("La variable X y Y no deben ser iguales.")
                else:
                    self.mostrar_eda_basico(df, [x], y, tipo="Lineal simple")

                if st.button("Ejecutar regresión lineal simple"):
                    try:
                        res = Regresion.regresion_lineal_simple(df, x, y)

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
                            col.markdown(ComponentesUI.crear_metrica_visual(config, valor, label), unsafe_allow_html=True)

                        # Gráfico
                        fig = px.scatter(
                            x=res["X_test"],
                            y=res["Y_test"],
                            labels={"x": x, "y": y},
                            title=f"{y} vs {x} (conjunto de prueba)",
                        )
                        line_df = pd.DataFrame({x: res["X_test"], y: res["y_pred_test"]})
                        fig.add_traces(px.line(line_df, x=x, y=y).data)
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        ComponentesUI.mostrar_error(f"{e}")

        # ----------------- REGRESIÓN LINEAL MÚLTIPLE -----------------
        elif modelo_tipo == "Regresión lineal múltiple":
            y = st.selectbox(
                "Variable objetivo (Y)",
                numeric_cols,
                index=numeric_cols.index(st.session_state.regresion_y)
                if st.session_state.regresion_y in numeric_cols
                else 0,
            )
            x_vars = st.multiselect(
                "Variables predictoras (X)",
                [c for c in numeric_cols if c != y],
                default=st.session_state.regresion_x_vars,
            )
            st.session_state.regresion_y = y
            st.session_state.regresion_x_vars = x_vars

            if y and x_vars:
                self.mostrar_eda_basico(df, x_vars, y, tipo="Lineal múltiple")
                if st.button("Ejecutar regresión lineal múltiple"):
                    try:
                        res = Regresion.regresion_lineal_multiple(df, x_vars, y)

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
                            col.markdown(ComponentesUI.crear_metrica_visual(config, valor, label), unsafe_allow_html=True)

                        # Visualizaciones
                        for xi in x_vars:
                            df_pred = df[x_vars].copy()
                            df_pred = df_pred.fillna(df_pred.mean())
                            for var in x_vars:
                                if var != xi:
                                    df_pred[var] = df_pred[var].mean()

                            y_pred_line = res["modelo"].predict(df_pred)
                            fig = px.scatter(
                                x=df[xi],
                                y=df[y],
                                labels={"x": xi, "y": y},
                                title=f"{y} vs {xi}",
                            )
                            line_df = pd.DataFrame({xi: df[xi], y: y_pred_line})
                            line_df = line_df.sort_values(by=xi)
                            fig.add_traces(px.line(line_df, x=xi, y=y).data)
                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        ComponentesUI.mostrar_error(f"{e}")

        # ----------------- REGRESIÓN LOGÍSTICA -----------------
        elif modelo_tipo == "Regresión logística":
            y = st.selectbox(
                "Variable objetivo (binaria)",
                numeric_cols,
                index=numeric_cols.index(st.session_state.regresion_logistica_y)
                if st.session_state.regresion_logistica_y in numeric_cols
                else 0,
            )
            valid_x_options = [c for c in numeric_cols if c != y]
            x_vars = st.multiselect(
                "Variables predictoras (X)",
                valid_x_options,
                default=st.session_state.regresion_logistica_x_vars,
            )
            st.session_state.regresion_logistica_y = y
            st.session_state.regresion_logistica_x_vars = x_vars

            if y and x_vars:
                if y in df.columns:
                    unique_vals = df[y].dropna().unique()
                    if len(unique_vals) != 2:
                        st.warning(f"La variable objetivo tiene {len(unique_vals)} clases; se requiere exactamente 2.")

                self.mostrar_eda_basico(df, x_vars, y, tipo="Logística")

                if y in df.columns:
                    counts = df[y].value_counts(normalize=True)
                    if len(counts) == 2:
                        min_pct = counts.min()
                        if min_pct < 0.2:
                            st.warning(f"Posible desbalanceo: la clase minoritaria tiene {min_pct*100:.1f}%.")

                if st.button("Ejecutar regresión logística"):
                    try:
                        res = Regresion.regresion_logistica(df, x_vars, y)

                        metricas = [
                            ("", "#4ECDC4", f"{res['resultados']['accuracy']:.3f}", "Accuracy"),
                            ("", "#FF6B6B", f"{res['resultados']['roc_auc']:.3f}", "ROC AUC"),
                            ("", "#45B7D1", str(len(res['resultados']['feature_names'])), "N° Features")
                        ]
                        cols = st.columns(len(metricas))
                        for col, (emoji, color, valor, label) in zip(cols, metricas):
                            config = ConfigMetrica(emoji, color)
                            col.markdown(ComponentesUI.crear_metrica_visual(config, valor, label), unsafe_allow_html=True)

                        st.subheader("Matriz de confusión")
                        st.write(np.array(res["resultados"]["confusion_matrix"]))

                        st.subheader("Reporte de clasificación")
                        st.json(res["resultados"]["classification_report"])

                        # Curva ROC
                        fpr = res["fpr"]
                        tpr = res["tpr"]
                        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
                        fig = px.area(
                            roc_df,
                            x="fpr",
                            y="tpr",
                            title="Curva ROC",
                            labels={"fpr": "False Positive Rate", "tpr": "True Positive Rate"},
                        )
                        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        ComponentesUI.mostrar_error(f"{e}")
