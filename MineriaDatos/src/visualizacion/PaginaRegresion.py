"""
Clase: PaginaKmeans

Objetivo: Clase que mantiene la pagina para visualizacion del K medias

Cambios:

    1. Creacion de la clase y cascarazon visual, aquesada 21-07-2025
"""
import streamlit as st  # Framework para crear aplicaciones web interactivas
import pandas as pd     # Manejo de dataframes
import plotly.express as px  # Visualizaciones interactivas
import numpy as np      # Operaciones numéricas

# Importamos módulos personalizados del proyecto
from modelos.Regresion import Regresion  # Contiene métodos para regresión lineal y logística
from src.helpers.UtilDataFrame import UtilDataFrame  # Funciones de limpieza/corrección del dataframe
from src.helpers.ComponentesUI import ComponentesUI  # Componentes visuales y manejo de errores en la UI


class PaginaRegresion:
    def __init__(self):
        pass  # Constructor vacío

    def _obtener_df(self):
        # Obtiene el DataFrame cargado desde la sesión de Streamlit
        return getattr(st.session_state, "df_cargado", None)

    def mostrar_eda_basico(self, df, x_vars, y_var, tipo):
        # Sección para mostrar análisis exploratorio básico (EDA)
        st.subheader("Exploración de datos (EDA) rápida")

        # Muestra resumen estadístico de las variables seleccionadas
        with st.expander("Resumen estadístico de las variables seleccionadas"):
            cols = list(filter(lambda c: c in df.columns, x_vars + ([y_var] if y_var else [])))
            if not cols:
                st.write("No hay columnas válidas seleccionadas.")
                return
            st.write(df[cols].describe(include="all"))  # Resumen estadístico de pandas

        # Gráficos de distribución para variables numéricas y categóricas
        with st.expander("Distribuciones"):
            for col in x_vars + ([y_var] if y_var else []):
                if col not in df.columns:
                    continue
                st.markdown(f"**{col}**")
                # Si es numérica, histograma con boxplot
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.histogram(df, x=col, marginal="box", nbins=30, title=f"Histograma de {col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Si es categórica, gráfico de barras con frecuencias
                    vc = df[col].value_counts().reset_index()
                    vc.columns = [col, "conteo"]
                    fig = px.bar(vc, x=col, y="conteo", title=f"Distribución de {col}")
                    st.plotly_chart(fig, use_container_width=True)

        # En caso de regresión logística, mostramos distribución de clases objetivo
        if tipo == "Logística" and y_var:
            st.subheader(f"Distribución de la variable objetivo ({y_var})")
            if y_var in df.columns:
                counts = df[y_var].value_counts().reset_index()
                counts.columns = ["clase", "conteo"]
                fig = px.bar(counts, x="clase", y="conteo", title="Clases en variable objetivo")
                st.plotly_chart(fig, use_container_width=True)

    def render(self):
        # Título principal de la página
        st.title("📈 Modelos de Regresión")

        # Intentamos obtener el dataset desde la sesión
        df = self._obtener_df()
        if df is None:
            st.error(" No hay dataset cargado. Ve a 'Datos' y carga uno primero.")
            return

        # Limpiamos el DataFrame para evitar errores en Streamlit
        df = UtilDataFrame.corregir_dataframe_para_streamlit(df)

        # Inicializamos variables de sesión si no existen
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

        # Selector para tipo de modelo
        modelo_tipo = st.selectbox(
            "Selecciona tipo de modelo",
            ["Regresión lineal simple", "Regresión lineal múltiple", "Regresión logística"],
            index=["Regresión lineal simple", "Regresión lineal múltiple", "Regresión logística"].index(
                st.session_state.regresion_tipo
            ),
        )
        st.session_state.regresion_tipo = modelo_tipo
        st.markdown("---")

        #  REGRESIÓN LINEAL SIMPLE

        if modelo_tipo == "Regresión lineal simple":
            # Solo variables numéricas son válidas para X e Y
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Selector de variable X
            x = st.selectbox(
                "Variable predictora (X)",
                numeric_cols,
                index=numeric_cols.index(st.session_state.regresion_x)
                if st.session_state.regresion_x in numeric_cols
                else 0,
            )
            # Selector de variable Y
            y = st.selectbox(
                "Variable objetivo (Y)",
                numeric_cols,
                index=numeric_cols.index(st.session_state.regresion_y)
                if st.session_state.regresion_y in numeric_cols
                else 0,
            )
            # Guardamos selección en la sesión
            st.session_state.regresion_x = x
            st.session_state.regresion_y = y

            # Validaciones y ejecución del modelo
            # Validaciones y ejecución del modelo
            if x and y:
                if x == y:
                    st.warning("La variable X y Y no deben ser iguales. Selecciona columnas diferentes.")
                else:
                    self.mostrar_eda_basico(df, [x], y, tipo="Lineal simple")

                if st.button("Ejecutar regresión lineal simple"):
                    try:
                        res = Regresion.regresion_lineal_simple(df, x, y)
                        st.subheader("Resultados (Entrenamiento vs Prueba)")
                        st.json(res["resultados"])

                        # Gráfico de dispersión + línea de regresión
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

        #  REGRESIÓN LINEAL MÚLTIPLE

        elif modelo_tipo == "Regresión lineal múltiple":
            all_cols = df.columns.tolist()

            # Selector de variable Y
            y = st.selectbox(
                "Variable objetivo (Y)",
                all_cols,
                index=all_cols.index(st.session_state.regresion_y) if st.session_state.regresion_y in all_cols else 0,
            )

            # Selector de variables X
            x_vars = st.multiselect(
                "Variables predictoras (X)",
                [c for c in all_cols if c != y],
                default=st.session_state.regresion_x_vars,
            )

            # Guardamos en sesión
            st.session_state.regresion_y = y
            st.session_state.regresion_x_vars = x_vars

            # Ejecutar modelo si se seleccionaron variables
            if y and x_vars:
                self.mostrar_eda_basico(df, x_vars, y, tipo="Lineal múltiple")
                if st.button("Ejecutar regresión lineal múltiple"):
                    try:
                        res = Regresion.regresion_lineal_multiple(df, x_vars, y)
                        st.subheader("Resultados")
                        st.json(res["resultados"])
                    except Exception as e:
                        ComponentesUI.mostrar_error(f"{e}")

        #  REGRESIÓN LOGÍSTICA

        elif modelo_tipo == "Regresión logística":
            all_cols = df.columns.tolist()

            # Selector de variable Y (binaria)
            y = st.selectbox(
                "Variable objetivo (binaria)",
                all_cols,
                index=all_cols.index(st.session_state.regresion_logistica_y)
                if st.session_state.regresion_logistica_y in all_cols
                else 0,
            )

            # Variables X posibles (excluyendo la Y)
            valid_x_options = [c for c in all_cols if c != y]

            # Validar valores por defecto en sesión
            default_x_vars = [
                var for var in st.session_state.regresion_logistica_x_vars if var in valid_x_options
            ]

            # Multiselect para variables X
            x_vars = st.multiselect(
                "Variables predictoras (X)",
                valid_x_options,
                default=st.session_state.regresion_logistica_x_vars,
            )

            # Guardamos en sesión
            st.session_state.regresion_logistica_y = y
            st.session_state.regresion_logistica_x_vars = x_vars

            # Validación de clases y advertencias
            if y and x_vars:
                if y in df.columns:
                    unique_vals = df[y].dropna().unique()
                    if len(unique_vals) != 2:
                        st.warning(
                            f"La variable objetivo tiene {len(unique_vals)} clases ({unique_vals.tolist()}); se requiere exactamente 2 para regresión logística."
                        )

                # Mostramos EDA
                self.mostrar_eda_basico(df, x_vars, y, tipo="Logística")

                # Advertencia por desbalanceo
                if y in df.columns:
                    counts = df[y].value_counts(normalize=True)
                    if len(counts) == 2:
                        min_pct = counts.min()
                        if min_pct < 0.2:
                            st.warning(
                                f"Posible desbalanceo en la clase objetivo: la clase minoritaria tiene {min_pct*100:.1f}% de los ejemplos."
                            )

                # Ejecutamos el modelo
                if st.button("Ejecutar regresión logística"):
                    try:
                        res = Regresion.regresion_logistica(df, x_vars, y)
                        st.subheader("Resultados")
                        st.metric("Accuracy", res["resultados"]["accuracy"])
                        st.metric("ROC AUC", res["resultados"]["roc_auc"])

                        # Matriz de confusión
                        st.subheader("Matriz de confusión")
                        cm = np.array(res["resultados"]["confusion_matrix"])
                        st.write(cm)

                        # Reporte de clasificación
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
