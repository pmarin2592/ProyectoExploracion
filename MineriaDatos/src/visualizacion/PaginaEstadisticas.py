"""
Clase: PaginaEstadisticas

Objetivo: Clase que mantiene la pagina de las estadisticas basicas

Cambios:

    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
    2. Implementación completa de estadísticas básicas usando session_state aquesada 19-07-2025
"""
import pandas as pd
import streamlit as st
<<<<<<< Updated upstream
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


class PaginaEstadisticas:
    def __init__(self):
        pass

    def _obtener_estadisticas_numericas(self, df):
        """Genera estadísticas descriptivas para columnas numéricas"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None

        stats_df = df[numeric_cols].describe().round(2)
        return stats_df

    def _obtener_estadisticas_categoricas(self, df):
        """Genera estadísticas para columnas categóricas"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) == 0:
            return None

        cat_stats = []
        for col in categorical_cols:
            unique_count = df[col].nunique()
            most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
            frequency = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0

            cat_stats.append({
                'Columna': col,
                'Valores únicos': unique_count,
                'Valor más frecuente': most_frequent,
                'Frecuencia máxima': frequency,
                'Valores nulos': df[col].isnull().sum()
            })

        return pd.DataFrame(cat_stats)

    def _crear_grafico_distribucion(self, df, columna):
        """Crea gráfico de distribución para una columna numérica"""
        fig = px.histogram(
            df,
            x=columna,
            title=f'Distribución de {columna}',
            nbins=30,
            marginal="box"
        )
        fig.update_layout(
            showlegend=False,
            height=400,
            title_x=0.5
        )
        return fig

    def _crear_grafico_barras_categorica(self, df, columna, top_n=10):
        """Crea gráfico de barras para una columna categórica"""
        value_counts = df[columna].value_counts().head(top_n)

        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Top {min(top_n, len(value_counts))} valores más frecuentes en {columna}',
            labels={'x': columna, 'y': 'Frecuencia'}
        )
        fig.update_layout(
            height=400,
            title_x=0.5,
            xaxis_tickangle=-45
        )
        return fig

    def _crear_matriz_correlacion(self, df):
        """Crea matriz de correlación para variables numéricas"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None

        corr_matrix = numeric_df.corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlación",
            color_continuous_scale="RdBu"
        )
        fig.update_layout(
            height=500,
            title_x=0.5
        )
        return fig

    def _crear_boxplots_numericas(self, df):
        """Crea boxplots para variables numéricas"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None

        # Limitar a las primeras 6 columnas para no sobrecargar
        cols_to_plot = numeric_cols[:6]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=cols_to_plot
        )

        for i, col in enumerate(cols_to_plot):
            row = (i // 3) + 1
            col_pos = (i % 3) + 1

            fig.add_trace(
                go.Box(y=df[col], name=col),
                row=row, col=col_pos
            )

        fig.update_layout(
            height=500,
            title_text="Boxplots de Variables Numéricas",
            title_x=0.5,
            showlegend=False
        )
        return fig

    def render(self):
        st.title("📊 Estadísticas Básicas")

        # Verificar si hay datos cargados
        if 'df_cargado' not in st.session_state:
            st.warning("⚠️ No hay datos cargados. Por favor, ve a la página 'Datos' y carga un archivo.")
            return

        df = st.session_state.df_cargado

        # Información general del dataset
        st.subheader("📋 Resumen General del Dataset")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total de Filas", f"{len(df):,}")
        with col2:
            st.metric("📋 Total de Columnas", len(df.columns))
        with col3:
            st.metric("🔢 Columnas Numéricas", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("📝 Columnas Categóricas", len(df.select_dtypes(include=['object']).columns))

        # Información de valores nulos
        st.subheader("❌ Análisis de Valores Nulos")

        null_info = df.isnull().sum()
        null_percent = (null_info / len(df)) * 100

        null_df = pd.DataFrame({
            'Columna': null_info.index,
            'Valores Nulos': null_info.values,
            'Porcentaje': null_percent.values.round(2)
        })
        null_df = null_df[null_df['Valores Nulos'] > 0].sort_values('Valores Nulos', ascending=False)

        if len(null_df) > 0:
            st.dataframe(null_df, hide_index=True, use_container_width=True)

            # Gráfico de valores nulos
            if len(null_df) > 0:
                fig_nulls = px.bar(
                    null_df,
                    x='Columna',
                    y='Valores Nulos',
                    title='Distribución de Valores Nulos por Columna'
                )
                fig_nulls.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_nulls, use_container_width=True)
        else:
            st.success("✅ No hay valores nulos en el dataset")

        # Estadísticas descriptivas para variables numéricas
        st.subheader("🔢 Estadísticas Descriptivas - Variables Numéricas")
        stats_num = self._obtener_estadisticas_numericas(df)

        if stats_num is not None:
            st.dataframe(stats_num, use_container_width=True)

            # Selector para visualizar distribuciones
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            col_selected = st.selectbox(
                "Selecciona una columna numérica para ver su distribución:",
                numeric_cols
            )

            if col_selected:
                fig_dist = self._crear_grafico_distribucion(df, col_selected)
                st.plotly_chart(fig_dist, use_container_width=True)

                # Estadísticas adicionales de la columna seleccionada
                col_data = df[col_selected].dropna()
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📊 Media", f"{col_data.mean():.2f}")
                with col2:
                    st.metric("📊 Mediana", f"{col_data.median():.2f}")
                with col3:
                    st.metric("📊 Desv. Estándar", f"{col_data.std():.2f}")
                with col4:
                    st.metric("📊 Coef. Variación", f"{(col_data.std()/col_data.mean()*100):.2f}%")
        else:
            st.info("No hay columnas numéricas en el dataset")

        # Estadísticas para variables categóricas
        st.subheader("📝 Estadísticas Descriptivas - Variables Categóricas")
        stats_cat = self._obtener_estadisticas_categoricas(df)

        if stats_cat is not None:
            st.dataframe(stats_cat, hide_index=True, use_container_width=True)

            # Selector para visualizar distribuciones categóricas
            categorical_cols = df.select_dtypes(include=['object']).columns
            cat_selected = st.selectbox(
                "Selecciona una columna categórica para ver su distribución:",
                categorical_cols
            )

            if cat_selected:
                fig_cat = self._crear_grafico_barras_categorica(df, cat_selected)
                st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No hay columnas categóricas en el dataset")

        # Matriz de correlación
        st.subheader("🔗 Matriz de Correlación")
        fig_corr = self._crear_matriz_correlacion(df)

        if fig_corr is not None:
            st.plotly_chart(fig_corr, use_container_width=True)

            # Top correlaciones
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] >= 2:
                corr_matrix = numeric_df.corr()

                # Obtener correlaciones más altas (excluyendo diagonal)
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        correlations.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlación': corr_matrix.iloc[i, j]
                        })

                corr_df = pd.DataFrame(correlations)
                corr_df['Correlación Abs'] = abs(corr_df['Correlación'])
                top_corr = corr_df.nlargest(5, 'Correlación Abs')[['Variable 1', 'Variable 2', 'Correlación']]

                st.write("**🎯 Top 5 Correlaciones más Fuertes:**")
                st.dataframe(top_corr, hide_index=True, use_container_width=True)
        else:
            st.info("Se necesitan al menos 2 variables numéricas para calcular correlaciones")

        # Boxplots
        st.subheader("📦 Análisis de Outliers - Boxplots")
        fig_box = self._crear_boxplots_numericas(df)

        if fig_box is not None:
            st.plotly_chart(fig_box, use_container_width=True)
            st.caption("Los puntos fuera de las cajas representan posibles valores atípicos (outliers)")
        else:
            st.info("No hay variables numéricas para analizar outliers")

        # Información adicional
        st.subheader("ℹ️ Información Adicional")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**🏷️ Tipos de Datos:**")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"• {dtype}: {count} columnas")

        with col2:
            st.write("**📏 Uso de Memoria:**")
            memory_usage = df.memory_usage(deep=True)
            total_memory = memory_usage.sum() / 1024**2
            st.write(f"• Total: {total_memory:.2f} MB")
            st.write(f"• Promedio por columna: {total_memory/len(df.columns):.2f} MB")
=======
import logging
from src.eda.EstadisticasBasicasEda import EstadisticasBasicasEda
# Configurar logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class PaginaEstadisticas:
    def __init__(self):
        self.eda = None

    def mostrar_error(mensaje: str, tipo_error: str = "error"):
        """
        Muestra un mensaje de error en Streamlit

        Args:
            mensaje (str): Mensaje de error
            tipo_error (str): Tipo de error (error, warning, info)
        """
        if tipo_error == "error":
            st.error(f"❌ {mensaje}")
        elif tipo_error == "warning":
            st.warning(f"⚠️ {mensaje}")
        elif tipo_error == "info":
            st.info(f"ℹ️ {mensaje}")

        logger.error(f"Error en Streamlit: {mensaje}")

    def _mostrar_info_dataframe_streamlit(self):
        """
        Muestra información básica del DataFrame en Streamlit usando EstadisticasBasicasEda
        con control de excepciones robusto

        Args:
            df (pd.DataFrame): El DataFrame a analizar
        """
        try:

            # Crear instancia del analizador
            eda = self.eda

            # Título principal
            st.header("📊 Información Básica del DataFrame")

            # Obtener información básica con manejo de errores
            try:
                info_basica = eda.obtener_info_basica()

                # Métricas principales en columnas
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Filas", f"{info_basica['filas']:,}")

                with col2:
                    st.metric("Columnas", info_basica['columnas'])

                with col3:
                    st.metric("Tamaño en memoria", f"{info_basica['memoria_mb']:.2f} MB")

            except Exception as e:
                self.mostrar_error(f"Error inesperado al obtener información básica: {str(e)}")
                logger.exception(f"Error inesperado al obtener información básica: {str(e)}")
                return

            # Separador
            st.divider()

            # Primeras y últimas filas
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔝 Primeras 5 filas")
                try:
                    primeras_filas = eda.obtener_primeras_filas()
                    st.dataframe(primeras_filas, use_container_width=True)
                except Exception as e:
                    self.mostrar_error(f"Error inesperado al obtener primeras filas: {str(e)}", "warning")
                    logger.exception(f"Error inesperado al obtener primeras filas: {str(e)}")

            with col2:
                st.subheader("🔚 Últimas 5 filas")
                try:
                    ultimas_filas = eda.obtener_ultimas_filas()
                    st.dataframe(ultimas_filas, use_container_width=True)
                except Exception as e:
                    self.mostrar_error(f"Error inesperado al obtener últimas filas: {str(e)}", "warning")
                    logger.exception(f"Error inesperado al obtener últimas filas: {str(e)}")

            # Información de tipos de datos
            st.subheader("📋 Información de tipos de datos")

            try:
                tipos_info = eda.obtener_tipo_datos()

                # Mostrar en un expander para ahorrar espacio
                with st.expander("Ver información detallada de tipos de datos"):
                    st.text(tipos_info['info_detallada'])

                # Resumen de tipos de datos en formato más visual
                st.write("**Resumen de tipos de datos:**")
                for dtype, count in tipos_info['tipos_resumen'].items():
                    st.write(f"- {dtype}: {count} columnas")


            except Exception as e:
                self.mostrar_error(f"Error inesperado al obtener información de tipos: {str(e)}", "warning")
                logger.exception(f"Error inesperado al obtener información de tipos: {str(e)}")

            # Nombres de columnas con información adicional
            st.subheader("📝 Información de columnas")

            try:
                info_columnas = eda.obtener_info_columnas()
                st.dataframe(info_columnas, use_container_width=True)

                # Mostrar lista de columnas en formato compacto
                with st.expander("Ver lista de nombres de columnas"):
                    nombres_columnas = eda.obtener_nombres_columnas()
                    st.write(nombres_columnas)

            except Exception as e:
                self.mostrar_error(f"Error inesperado al obtener información de columnas: {str(e)}", "warning")
                logger.exception(f"Error inesperado al obtener información de columnas: {str(e)}")

        except Exception as e:
            self.mostrar_error(f"Error inesperado: {str(e)}")
            logger.exception("Error inesperado en mostrar_info_dataframe_streamlit")

    def _mostrar_analisis_completo(self):
        """
        Muestra análisis completo del DataFrame con control de excepciones

        Args:
            df (pd.DataFrame): El DataFrame a analizar
        """
        try:


            eda = self.eda

            # Crear tabs para organizar mejor la información
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📊 Información Básica", "📈 Estadísticas", "❌ Valores Faltantes", "🔍 Análisis Detallado"])

            with tab1:
                self._mostrar_info_dataframe_streamlit()

            with tab2:
                st.subheader("📈 Estadísticas Descriptivas")
                try:
                    estadisticas = eda.obtener_resumen_estadisticas()
                    if estadisticas is not None:
                        st.dataframe(estadisticas, use_container_width=True)
                    else:
                        st.warning("No se pudieron generar estadísticas descriptivas")
                except Exception as e:
                    self.mostrar_error(f"Error inesperado al obtener estadísticas: {str(e)}", "warning")

            with tab3:
                st.subheader("❌ Valores Faltantes")
                try:
                    valores_faltantes = eda.obtener_valores_faltantes()

                    if valores_faltantes is not None:
                        # Verificar si hay valores faltantes válidos
                        valores_numericos = valores_faltantes[valores_faltantes['Valores_faltantes'] != 'Error']

                        if len(valores_numericos) > 0 and valores_numericos['Valores_faltantes'].sum() == 0:
                            st.success("¡No hay valores faltantes en el DataFrame!")
                        else:
                            st.dataframe(valores_faltantes, use_container_width=True)

                            # Visualizar valores faltantes solo si hay datos válidos
                            try:
                                if len(valores_numericos) > 0:
                                    chart_data = valores_numericos[valores_numericos['Porcentaje'] != 'Error']
                                    if len(chart_data) > 0:
                                        st.bar_chart(chart_data.set_index('Columna')['Porcentaje'])
                            except Exception as e:
                                st.warning("No se pudo generar el gráfico de valores faltantes")
                    else:
                        st.warning("No se pudo obtener información de valores faltantes")

                except Exception as e:
                    self.mostrar_error(f"Error inesperado al obtener valores faltantes: {str(e)}", "warning")

            with tab4:
                st.subheader("🔍 Análisis Detallado")

                # Información por columna
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Distribución de tipos de datos:**")
                    try:
                        tipos_info = eda.obtener_tipo_datos()
                        chart_data = pd.DataFrame(
                            list(tipos_info['tipos_resumen'].items()),
                            columns=['Tipo', 'Cantidad']
                        )
                        st.bar_chart(chart_data.set_index('Tipo'))
                    except Exception as e:
                        st.warning("No se pudo generar el gráfico de tipos de datos")

                with col2:
                    st.write("**Top 10 columnas con más valores únicos:**")
                    try:
                        info_columnas = eda.obtener_info_columnas()
                        # Filtrar solo las columnas con valores numéricos válidos
                        columnas_validas = info_columnas[info_columnas['Valores_únicos'] != 'N/A']
                        if len(columnas_validas) > 0:
                            top_unique = columnas_validas.nlargest(10, 'Valores_únicos')[['Columna', 'Valores_únicos']]
                            st.dataframe(top_unique, use_container_width=True)
                        else:
                            st.warning("No se encontraron columnas con valores únicos válidos")
                    except Exception as e:
                        st.warning("No se pudo generar el top de columnas con valores únicos")

        except Exception as e:
            self.mostrar_error(f"Error inesperado en análisis completo: {str(e)}")
            logger.exception("Error inesperado en mostrar_analisis_completo")


    def render(self):
        st.title("📊 Estadisticas Basicas")
        if  self.eda is None:
            self.eda =  st.session_state.eda
            self._mostrar_info_dataframe_streamlit()
>>>>>>> Stashed changes
