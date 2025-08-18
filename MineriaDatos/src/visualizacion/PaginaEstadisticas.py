"""
Clase: PaginaEstadisticas

Objetivo: Clase que mantiene la pagina de las estadisticas basicas

Cambios:
    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
    2. Mejoras visuales y diseño moderno - Versión mejorada
    3. Optimización completa del código - Versión optimizada
    4. Cambios en UI aquesada 02-08-25
    5. Eliminación del selector de vista y título principal "Análisis Completo"
    6. Se agrega codigo de graficos pmarin 15-08-2025
"""
import inspect

import pandas as pd
import streamlit as st
import logging
import plotly.express as px
from typing import Dict, Optional
from src.eda.EstadisticasBasicasEda import EstadisticasBasicasEda
from src.eda.ManejadorDatos import ManejadorDatos
from src.helpers.ComponentesUI import ComponentesUI, ConfigMetrica, ConfiguracionTipos, TipoVista
from src.helpers.UtilDataFrame import UtilDataFrame

# Configurar logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class PaginaEstadisticas:
    """Clase optimizada para mostrar estadísticas básicas"""

    def __init__(self):
        try:
            self.eda: Optional[EstadisticasBasicasEda] = None
            self.manejador_datos: Optional[ManejadorDatos] = None
        except Exception as e:
            logger.error(f"Error en la inicialización de PaginaEstadisticas: {str(e)}")
            raise

    def _inicializar_dependencias(self) -> bool:
        """Inicializa dependencias necesarias"""
        try:
            if self.eda is None:
                try:
                    self.eda = getattr(st.session_state, 'eda', None)
                except AttributeError as e:
                    logger.error(f"Error al acceder a st.session_state: {str(e)}")
                    ComponentesUI.mostrar_error("Error al acceder al estado de la sesión")
                    return False

            if self.eda is None:
                ComponentesUI.mostrar_error("No hay datos cargados")
                return False

            if self.manejador_datos is None:
                try:
                    self.manejador_datos = ManejadorDatos(self.eda)
                except Exception as e:
                    logger.error(f"Error al inicializar ManejadorDatos: {str(e)}")
                    ComponentesUI.mostrar_error("Error al inicializar el manejador de datos")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error inesperado en _inicializar_dependencias: {str(e)}")
            ComponentesUI.mostrar_error("Error inesperado al inicializar dependencias")
            return False

    def _mostrar_metricas_principales(self) -> None:
        """Muestra las métricas principales del DataFrame"""
        try:
            if not self.manejador_datos:
                logger.error("ManejadorDatos no está inicializado")
                return

            info_basica = self.manejador_datos.obtener_info_basica_segura()
            if not info_basica:
                logger.warning("No se pudo obtener información básica")
                return

            try:
                metrics_config = [
                    ConfigMetrica('📏', '#FF6B6B'),
                    ConfigMetrica('📋', '#4ECDC4'),
                    ConfigMetrica('💾', '#45B7D1')
                ]

                valores = [
                    f"{info_basica.get('filas', 0):,}",
                    f"{info_basica.get('columnas', 0)}",
                    f"{info_basica.get('memoria_mb', 0):.2f} MB"
                ]

                labels = ['Filas', 'Columnas', 'Memoria']

                cols = st.columns(3)
                for i, (config, valor, label) in enumerate(zip(metrics_config, valores, labels)):
                    try:
                        with cols[i]:
                            st.markdown(
                                ComponentesUI.crear_metrica_visual(config, valor, label),
                                unsafe_allow_html=True
                            )
                    except Exception as e:
                        logger.warning(f"Error al mostrar métrica {label}: {str(e)}")
                        with cols[i]:
                            st.metric(label, valor)

            except Exception as e:
                logger.error(f"Error al crear métricas visuales: {str(e)}")
                # Fallback con métricas simples
                try:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Filas", f"{info_basica.get('filas', 0):,}")
                    with col2:
                        st.metric("Columnas", info_basica.get('columnas', 0))
                    with col3:
                        st.metric("Memoria", f"{info_basica.get('memoria_mb', 0):.2f} MB")
                except Exception as e2:
                    logger.error(f"Error en fallback de métricas: {str(e2)}")

        except Exception as e:
            logger.error(f"Error inesperado en _mostrar_metricas_principales: {str(e)}")

    def _crear_grafico_tipos_datos(self, tipos_resumen: Dict[str, int], total_columnas: int) -> None:
        """Crea gráfico de distribución de tipos de datos"""
        try:
            if not tipos_resumen or total_columnas == 0:
                st.warning("No hay datos suficientes para crear el gráfico")
                return

            try:
                chart_data = pd.DataFrame([
                    {
                        'Tipo': dtype,
                        'Cantidad': count,
                        'Porcentaje': (count / total_columnas) * 100 if total_columnas > 0 else 0
                    }
                    for dtype, count in tipos_resumen.items()
                ])

                if chart_data.empty:
                    st.warning("No hay datos para mostrar en el gráfico")
                    return

                fig = px.bar(
                    chart_data,
                    x='Cantidad',
                    y='Tipo',
                    orientation='h',
                    color='Tipo',
                    text='Cantidad',
                    title='Distribución de Tipos de Datos',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )

                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    title_font_size=16,
                    height=300
                )

                fig.update_traces(
                    texttemplate='%{text}',
                    textposition='outside',
                    marker=dict(line=dict(width=1, color='white'))
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                logger.error(f"Error al crear gráfico de tipos de datos: {str(e)}")
                # Fallback con gráfico simple
                st.bar_chart(pd.Series(tipos_resumen))

        except Exception as e:
            logger.error(f"Error inesperado en _crear_grafico_tipos_datos: {str(e)}")

    def _mostrar_vista_previa_datos(self) -> None:
        """Muestra vista previa de primeras y últimas filas"""
        try:
            st.markdown("### 🔍 **Vista Previa de los Datos**")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🔝 **Primeras 5 filas**")
                try:
                    if not self.eda:
                        st.error("EDA no está inicializado")
                        return

                    primeras_filas = self.eda.obtener_primeras_filas()
                    if primeras_filas is not None and not primeras_filas.empty:
                        st.dataframe(primeras_filas, use_container_width=True, height=200)
                    else:
                        st.warning("No se pudieron obtener las primeras filas")
                except Exception as e:
                    logger.error(f"Error al obtener primeras filas: {str(e)}")
                    ComponentesUI.mostrar_error(f"Error al obtener primeras filas: {str(e)}", "warning")

            with col2:
                st.markdown("#### 🔚 **Últimas 5 filas**")
                try:
                    if not self.eda:
                        st.error("EDA no está inicializado")
                        return

                    ultimas_filas = self.eda.obtener_ultimas_filas()
                    if ultimas_filas is not None and not ultimas_filas.empty:
                        st.dataframe(ultimas_filas, use_container_width=True, height=200)
                    else:
                        st.warning("No se pudieron obtener las últimas filas")
                except Exception as e:
                    logger.error(f"Error al obtener últimas filas: {str(e)}")
                    ComponentesUI.mostrar_error(f"Error al obtener últimas filas: {str(e)}", "warning")

        except Exception as e:
            logger.error(f"Error inesperado en _mostrar_vista_previa_datos: {str(e)}")

    def _mostrar_info_columnas_detallada(self) -> None:
        """Muestra información detallada de columnas"""
        try:
            st.markdown("### 📝 **Información Detallada de Columnas**")

            try:
                if not self.eda:
                    st.error("EDA no está inicializado")
                    return

                info_columnas = self.eda.obtener_info_columnas()

                if info_columnas is None or info_columnas.empty:
                    st.warning("No se pudo obtener información de columnas")
                    return

                try:
                    info_columnas_procesado = UtilDataFrame.convertir_tipos_json_serializable(info_columnas)
                except Exception as e:
                    logger.warning(f"Error al convertir tipos JSON: {str(e)}")
                    info_columnas_procesado = info_columnas

                try:
                    st.dataframe(
                        info_columnas_procesado,
                        use_container_width=True,
                        height=300,
                        column_config={
                            "Columna": st.column_config.TextColumn("📝 Columna", width="medium"),
                            "Tipo": st.column_config.TextColumn("🏷️ Tipo", width="small"),
                            "Valores_únicos": st.column_config.NumberColumn("🔢 Únicos", format="%d"),
                            "Valores_nulos": st.column_config.NumberColumn("❌ Nulos", format="%d"),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error al configurar columnas del dataframe: {str(e)}")
                    st.dataframe(info_columnas_procesado, use_container_width=True, height=300)

                try:
                    with st.expander("📋 **Ver Lista Completa de Columnas**"):
                        nombres_columnas = self.eda.obtener_nombres_columnas()
                        if nombres_columnas:
                            print(nombres_columnas)
                            try:
                                chips_html = ComponentesUI.crear_chips_columnas(nombres_columnas)
                                st.html(chips_html)
                            except Exception as e:
                                logger.warning(f"Error al crear chips HTML: {str(e)}")
                                # Fallback: mostrar como lista simple
                                st.write(", ".join(nombres_columnas))
                        else:
                            st.warning("No se pudieron obtener los nombres de las columnas")
                except Exception as e:
                    logger.warning(f"Error en expander de columnas: {str(e)}")

            except Exception as e:
                logger.error(f"Error al obtener información de columnas: {str(e)}")
                ComponentesUI.mostrar_error(f"Error al obtener información de columnas: {str(e)}", "warning")

        except Exception as e:
            logger.error(f"Error inesperado en _mostrar_info_columnas_detallada: {str(e)}")

    def _mostrar_vista_mejorada(self) -> None:
        """Muestra la vista mejorada con diseño moderno"""
        try:
            # Título con gradiente
            try:
                ComponentesUI.crear_titulo_principal(
                    "📊 Información Básica del DataFrame",
                    "Análisis completo de la estructura de tus datos"
                )
            except Exception as e:
                logger.warning(f"Error al crear título principal: {str(e)}")
                st.header("📊 Información Básica del DataFrame")
                st.subheader("Análisis completo de la estructura de tus datos")

            # Métricas principales
            self._mostrar_metricas_principales()
            st.markdown("<br>", unsafe_allow_html=True)

            # Información de tipos de datos
            try:
                st.markdown("### 🏷️ **Información de Tipos de Datos**")

                if not self.manejador_datos:
                    st.error("Manejador de datos no está inicializado")
                    return

                tipos_resumen, total_columnas = self.manejador_datos.obtener_tipos_datos_procesados()

                if tipos_resumen and total_columnas > 0:
                    # Mostrar métricas por tipo en columnas
                    try:
                        num_tipos = len(tipos_resumen)
                        cols = st.columns(min(4, num_tipos))

                        for i, (dtype, count) in enumerate(tipos_resumen.items()):
                            try:
                                config = ConfiguracionTipos.obtener_config_tipo(dtype)

                                with cols[i % len(cols)]:
                                    st.markdown(f"""
                                    <div style="
                                        background: linear-gradient(135deg, {config.color}22, {config.color}11);
                                        padding: 1rem;
                                        border-radius: 10px;
                                        border-left: 4px solid {config.color};
                                        margin-bottom: 0.5rem;
                                    ">
                                        <div style="text-align: center;">
                                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{config.emoji}</div>
                                            <div style="font-size: 1.5rem; font-weight: bold; color: {config.color};">{count}</div>
                                            <div style="font-size: 0.9rem; color: #666; font-weight: 500;">{dtype}</div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            except Exception as e:
                                logger.warning(f"Error al mostrar métrica de tipo {dtype}: {str(e)}")
                                with cols[i % len(cols)]:
                                    st.metric(dtype, count)

                    except Exception as e:
                        logger.error(f"Error al crear métricas de tipos: {str(e)}")

                    st.markdown("---")

                    # Gráfico de distribución
                    try:
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown("#### 📊 **Distribución de Tipos de Datos**")
                            self._crear_grafico_tipos_datos(tipos_resumen, total_columnas)

                        with col2:
                            st.markdown("#### 🎯 **Resumen Rápido**")

                            try:
                                tipo_principal = max(tipos_resumen.items(), key=lambda x: x[1]) if tipos_resumen else (
                                    'N/A', 0)

                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                            padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
                                    <div style="text-align: center;">
                                        <div style="font-size: 1.8rem; font-weight: bold;">{total_columnas}</div>
                                        <div style="font-size: 0.9rem;">Total Columnas</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                            padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
                                    <div style="text-align: center;">
                                        <div style="font-size: 1.2rem; font-weight: bold;">{tipo_principal[0]}</div>
                                        <div style="font-size: 0.9rem;">Tipo Predominante</div>
                                        <div style="font-size: 1.5rem; font-weight: bold;">{tipo_principal[1]}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Barras de progreso para porcentajes
                                st.markdown("**📈 Porcentajes:**")
                                for dtype, count in tipos_resumen.items():
                                    try:
                                        porcentaje = (count / total_columnas) * 100 if total_columnas > 0 else 0
                                        st.progress(porcentaje / 100, text=f"{dtype}: {porcentaje:.1f}%")
                                    except Exception as e:
                                        logger.warning(f"Error al crear barra de progreso para {dtype}: {str(e)}")

                            except Exception as e:
                                logger.error(f"Error al crear resumen rápido: {str(e)}")
                                st.metric("Total Columnas", total_columnas)

                    except Exception as e:
                        logger.error(f"Error al crear sección de distribución: {str(e)}")
                else:
                    st.warning("No hay información de tipos de datos disponible")

            except Exception as e:
                logger.error(f"Error en sección de tipos de datos: {str(e)}")

            # Vista previa de datos
            try:
                st.markdown("---")
                self._mostrar_vista_previa_datos()
            except Exception as e:
                logger.error(f"Error en vista previa de datos: {str(e)}")

            # Información detallada de columnas
            try:
                st.markdown("---")
                self._mostrar_info_columnas_detallada()
            except Exception as e:
                logger.error(f"Error en información detallada de columnas: {str(e)}")

        except Exception as e:
            logger.error(f"Error inesperado en _mostrar_vista_mejorada: {str(e)}")

    def _mostrar_analisis_completo(self) -> None:
        """Muestra análisis completo con tabs organizadas"""
        try:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📊 Información Básica", "📈 Estadísticas", "❌ Valores Faltantes", "🔍 Análisis Detallado", "📈 Gráficos"]
            )

            with tab1:
                try:
                    self._mostrar_vista_mejorada()
                except Exception as e:
                    logger.error(f"Error en tab Información Básica: {str(e)}")
                    st.error("Error al cargar información básica")

            with tab2:
                try:
                    st.subheader("📈 Estadísticas Descriptivas")

                    if not self.eda:
                        st.error("EDA no está inicializado")
                        return

                    estadisticas = self.eda.obtener_resumen_estadisticas()
                    if estadisticas is not None and not estadisticas.empty:
                        st.dataframe(estadisticas, use_container_width=True)
                    else:
                        st.warning("No se pudieron generar estadísticas descriptivas")
                except Exception as e:
                    logger.error(f"Error al obtener estadísticas: {str(e)}")
                    ComponentesUI.mostrar_error(f"Error al obtener estadísticas: {str(e)}", "warning")

            with tab3:
                try:
                    st.subheader("❌ Valores Faltantes")

                    if not self.eda:
                        st.error("EDA no está inicializado")
                        return

                    valores_faltantes = self.eda.obtener_valores_faltantes()

                    if valores_faltantes is not None and not valores_faltantes.empty:
                        try:
                            valores_numericos = valores_faltantes[valores_faltantes['Valores_faltantes'] != 'Error']

                            if len(valores_numericos) > 0 and valores_numericos['Valores_faltantes'].sum() == 0:
                                st.success("¡No hay valores faltantes en el DataFrame!")
                            else:
                                st.dataframe(valores_faltantes, use_container_width=True)

                                # Gráfico de valores faltantes
                                try:
                                    if len(valores_numericos) > 0:
                                        chart_data = valores_numericos[valores_numericos['Porcentaje'] != 'Error']
                                        if len(chart_data) > 0:
                                            chart_data_clean = chart_data.set_index('Columna')['Porcentaje']
                                            if not chart_data_clean.empty:
                                                st.bar_chart(chart_data_clean)
                                except Exception as e:
                                    logger.warning(f"Error al generar gráfico de valores faltantes: {str(e)}")
                                    st.warning("No se pudo generar el gráfico de valores faltantes")
                        except Exception as e:
                            logger.error(f"Error al procesar valores faltantes: {str(e)}")
                            st.dataframe(valores_faltantes, use_container_width=True)
                    else:
                        st.warning("No se pudo obtener información de valores faltantes")
                except Exception as e:
                    logger.error(f"Error al obtener valores faltantes: {str(e)}")
                    ComponentesUI.mostrar_error(f"Error al obtener valores faltantes: {str(e)}", "warning")

            with tab4:
                try:
                    st.subheader("🔍 Análisis Detallado")

                    col1, col2 = st.columns(2)

                    with col1:
                        try:
                            st.write("**Distribución de tipos de datos:**")
                            if self.manejador_datos:
                                tipos_resumen, _ = self.manejador_datos.obtener_tipos_datos_procesados()
                                if tipos_resumen:
                                    chart_data = pd.DataFrame(
                                        list(tipos_resumen.items()),
                                        columns=['Tipo', 'Cantidad']
                                    )
                                    if not chart_data.empty:
                                        st.bar_chart(chart_data.set_index('Tipo'))
                                    else:
                                        st.warning("No hay datos para mostrar")
                                else:
                                    st.warning("No se pudieron obtener tipos de datos")
                            else:
                                st.error("Manejador de datos no está inicializado")
                        except Exception as e:
                            logger.error(f"Error en distribución de tipos: {str(e)}")
                            st.warning("Error al mostrar distribución de tipos de datos")

                    with col2:
                        try:
                            st.write("**Top 10 columnas con más valores únicos:**")

                            if not self.eda:
                                st.error("EDA no está inicializado")
                                return

                            info_columnas = self.eda.obtener_info_columnas()
                            if info_columnas is not None and not info_columnas.empty:
                                columnas_validas = info_columnas[info_columnas['Valores_únicos'] != 'N/A']
                                if len(columnas_validas) > 0:
                                    top_unique = columnas_validas.nlargest(10, 'Valores_únicos')[
                                        ['Columna', 'Valores_únicos']]
                                    st.dataframe(top_unique, use_container_width=True)
                                else:
                                    st.warning("No se encontraron columnas con valores únicos válidos")
                            else:
                                st.warning("No se pudo obtener información de columnas")
                        except Exception as e:
                            logger.error(f"Error en top columnas únicas: {str(e)}")
                            st.warning("No se pudo generar el top de columnas con valores únicos")
                except Exception as e:
                    logger.error(f"Error en tab Análisis Detallado: {str(e)}")

            with tab5:
                try:
                    st.subheader("📈 Gráficos")

                    if not self.eda:
                        st.error("EDA no está inicializado")
                        return

                    try:
                        tipo_grafico = st.selectbox("Seleccione el tipo de gráfico",
                                                    ["Selecciona", "Distribucciones", "Boxplot",
                                                     "Matriz de Correlación", "Univariado"])

                        if hasattr(self.eda, 'df') and self.eda.df is not None:
                            columna = st.selectbox("Seleccione la variable para el eje X", self.eda.df.columns)
                        else:
                            st.error("DataFrame no está disponible")
                            return

                        if tipo_grafico == "Distribucciones":
                            try:
                                fig, codigo = self.eda.obtener_analisis_distribucion(columna)
                                if fig is not None:
                                    st.pyplot(fig)
                                    try:
                                        with st.expander("📋 **Ver Código Gráfico**"):
                                            st.subheader("📄 Código generado:")
                                            st.code(codigo, language="python")
                                    except Exception as e:
                                        logger.warning(f"Error en expander de código: {str(e)}")
                                else:
                                    st.warning("No se pudo generar el gráfico de distribución")
                            except Exception as e:
                                logger.error(f"Error en gráfico de distribución: {str(e)}")
                                st.error("Error al generar gráfico de distribución")

                        elif tipo_grafico == "Boxplot":
                            try:
                                fig, codigo = self.eda.obtener_analisis_boxplot(columna)
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                    try:
                                        with st.expander("📋 **Ver Código Gráfico**"):
                                            st.subheader("📄 Código generado:")
                                            st.code(codigo, language="python")
                                    except Exception as e:
                                        logger.warning(f"Error en expander de código: {str(e)}")
                                else:
                                    st.warning("No se pudo generar el boxplot")
                            except Exception as e:
                                logger.error(f"Error en boxplot: {str(e)}")
                                st.error("Error al generar boxplot")

                        elif tipo_grafico == "Matriz de Correlación":
                            try:
                                fig, codigo = self.eda.obtener_analisis_correlaccion()
                                if fig is not None:
                                    st.pyplot(fig, use_container_width=True)
                                    try:
                                        with st.expander("📋 **Ver Código Gráfico**"):
                                            st.subheader("📄 Código generado:")
                                            st.code(codigo, language="python")
                                    except Exception as e:
                                        logger.warning(f"Error en expander de código: {str(e)}")
                                else:
                                    st.warning("No se pudo generar la matriz de correlación")
                            except Exception as e:
                                logger.error(f"Error en matriz de correlación: {str(e)}")
                                st.error("Error al generar matriz de correlación")

                        elif tipo_grafico == "Univariado":
                            try:
                                fig, codigo = self.eda.obtener_analisis_univariados(columna)
                                if fig is not None:
                                    st.pyplot(fig, use_container_width=True)
                                    try:
                                        with st.expander("📋 **Ver Código Gráfico**"):
                                            st.subheader("📄 Código generado:")
                                            st.code(codigo, language="python")
                                    except Exception as e:
                                        logger.warning(f"Error en expander de código: {str(e)}")
                                else:
                                    st.warning("No se pudo generar el análisis univariado")
                            except Exception as e:
                                logger.error(f"Error en análisis univariado: {str(e)}")
                                st.error("Error al generar análisis univariado")

                    except Exception as e:
                        logger.error(f"Error en controles de gráficos: {str(e)}")
                        st.error("Error al configurar controles de gráficos")

                except Exception as e:
                    logger.error(f"Error en tab Gráficos: {str(e)}")

        except Exception as e:
            logger.error(f"Error inesperado en _mostrar_analisis_completo: {str(e)}")

    def render(self) -> None:
        """Método principal para renderizar la página de estadísticas"""
        try:
            if not self._inicializar_dependencias():
                return

            # Título principal de la página
            try:
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
                    Análisis Completo
                </h1>
                """, unsafe_allow_html=True)
            except Exception as e:
                logger.warning(f"Error al crear título principal: {str(e)}")
                st.title("Análisis Completo")

            # Mostrar directamente el análisis completo
            try:
                self._mostrar_analisis_completo()
            except Exception as e:
                logger.error(f"Error inesperado en mostrar análisis completo: {str(e)}")
                ComponentesUI.mostrar_error(f"Error inesperado: {str(e)}")

        except Exception as e:
            logger.error(f"Error inesperado en render: {str(e)}")
            try:
                ComponentesUI.mostrar_error(f"Error inesperado en la página: {str(e)}")
            except:
                st.error(f"Error crítico en la página: {str(e)}")
            logger.exception("Error inesperado en render")