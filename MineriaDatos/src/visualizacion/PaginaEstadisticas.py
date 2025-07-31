"""
Clase: PaginaEstadisticas

Objetivo: Clase que mantiene la pagina de las estadisticas basicas

Cambios:
    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
    2. Mejoras visuales y diseño moderno - Versión mejorada
    3. Optimización completa del código - Versión optimizada
"""
import pandas as pd
import streamlit as st
import logging
import plotly.express as px
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
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
        self.eda: Optional[EstadisticasBasicasEda] = None
        self.manejador_datos: Optional[ManejadorDatos] = None

    def _inicializar_dependencias(self) -> bool:
        """Inicializa dependencias necesarias"""
        if self.eda is None:
            self.eda = getattr(st.session_state, 'eda', None)

        if self.eda is None:
            ComponentesUI.mostrar_error("No hay datos cargados")
            return False

        if self.manejador_datos is None:
            self.manejador_datos = ManejadorDatos(self.eda)

        return True

    def _mostrar_metricas_principales(self) -> None:
        """Muestra las métricas principales del DataFrame"""
        info_basica = self.manejador_datos.obtener_info_basica_segura()
        if not info_basica:
            return

        metrics_config = [
            ConfigMetrica('📏', '#FF6B6B'),
            ConfigMetrica('📋', '#4ECDC4'),
            ConfigMetrica('💾', '#45B7D1')
        ]

        valores = [
            f"{info_basica['filas']:,}",
            f"{info_basica['columnas']}",
            f"{info_basica['memoria_mb']:.2f} MB"
        ]

        labels = ['Filas', 'Columnas', 'Memoria']

        cols = st.columns(3)
        for i, (config, valor, label) in enumerate(zip(metrics_config, valores, labels)):
            with cols[i]:
                st.markdown(
                    ComponentesUI.crear_metrica_visual(config, valor, label),
                    unsafe_allow_html=True
                )

    def _crear_grafico_tipos_datos(self, tipos_resumen: Dict[str, int], total_columnas: int) -> None:
        """Crea gráfico de distribución de tipos de datos"""
        if not tipos_resumen or total_columnas == 0:
            st.warning("No hay datos suficientes para crear el gráfico")
            return

        chart_data = pd.DataFrame([
            {
                'Tipo': dtype,
                'Cantidad': count,
                'Porcentaje': (count / total_columnas) * 100
            }
            for dtype, count in tipos_resumen.items()
        ])

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

    def _mostrar_vista_previa_datos(self) -> None:
        """Muestra vista previa de primeras y últimas filas"""
        st.markdown("### 🔍 **Vista Previa de los Datos**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔝 **Primeras 5 filas**")
            try:
                primeras_filas = self.eda.obtener_primeras_filas()
                st.dataframe(primeras_filas, use_container_width=True, height=200)
            except Exception as e:
                ComponentesUI.mostrar_error(f"Error al obtener primeras filas: {str(e)}", "warning")

        with col2:
            st.markdown("#### 🔚 **Últimas 5 filas**")
            try:
                ultimas_filas = self.eda.obtener_ultimas_filas()
                st.dataframe(ultimas_filas, use_container_width=True, height=200)
            except Exception as e:
                ComponentesUI.mostrar_error(f"Error al obtener últimas filas: {str(e)}", "warning")

    def _mostrar_info_columnas_detallada(self) -> None:
        """Muestra información detallada de columnas"""
        st.markdown("### 📝 **Información Detallada de Columnas**")

        try:

            info_columnas = self.eda.obtener_info_columnas()
            info_columnas_procesado = UtilDataFrame.convertir_tipos_json_serializable(info_columnas)

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

            with st.expander("📋 **Ver Lista Completa de Columnas**"):
                nombres_columnas = self.eda.obtener_nombres_columnas()
                print(nombres_columnas)
                chips_html = ComponentesUI.crear_chips_columnas(nombres_columnas)
                st.html(chips_html)
                #st.markdown(chips_html, unsafe_allow_html=True)

        except Exception as e:
            ComponentesUI.mostrar_error(f"Error al obtener información de columnas: {str(e)}", "warning")

    def _mostrar_vista_mejorada(self) -> None:
        """Muestra la vista mejorada con diseño moderno"""
        # Título con gradiente
        ComponentesUI.crear_titulo_principal(
            "📊 Información Básica del DataFrame",
            "Análisis completo de la estructura de tus datos"
        )

        # Métricas principales
        self._mostrar_metricas_principales()
        st.markdown("<br>", unsafe_allow_html=True)

        # Información de tipos de datos
        st.markdown("### 🏷️ **Información de Tipos de Datos**")
        tipos_resumen, total_columnas = self.manejador_datos.obtener_tipos_datos_procesados()

        if tipos_resumen and total_columnas > 0:
            # Mostrar métricas por tipo en columnas
            num_tipos = len(tipos_resumen)
            cols = st.columns(min(4, num_tipos))

            for i, (dtype, count) in enumerate(tipos_resumen.items()):
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

            st.markdown("---")

            # Gráfico de distribución
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("#### 📊 **Distribución de Tipos de Datos**")
                self._crear_grafico_tipos_datos(tipos_resumen, total_columnas)

            with col2:
                st.markdown("#### 🎯 **Resumen Rápido**")

                tipo_principal = max(tipos_resumen.items(), key=lambda x: x[1]) if tipos_resumen else ('N/A', 0)

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
                    porcentaje = (count / total_columnas) * 100 if total_columnas > 0 else 0
                    st.progress(porcentaje / 100, text=f"{dtype}: {porcentaje:.1f}%")

        # Vista previa de datos
        st.markdown("---")
        self._mostrar_vista_previa_datos()

        # Información detallada de columnas
        st.markdown("---")
        self._mostrar_info_columnas_detallada()

    def _mostrar_vista_clasica(self) -> None:
        """Muestra la vista clásica simplificada"""
        st.header("📊 Información Básica del DataFrame")

        # Información básica
        info_basica = self.manejador_datos.obtener_info_basica_segura()
        if info_basica:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Filas", f"{info_basica['filas']:,}")
            with col2:
                st.metric("Columnas", info_basica['columnas'])
            with col3:
                st.metric("Tamaño en memoria", f"{info_basica['memoria_mb']:.2f} MB")

        st.divider()
        self._mostrar_vista_previa_datos()

        # Tipos de datos simplificado
        st.subheader("📋 Información de tipos de datos")
        tipos_resumen, _ = self.manejador_datos.obtener_tipos_datos_procesados()

        if tipos_resumen:
            for dtype, count in tipos_resumen.items():
                st.write(f"- {dtype}: {count} columnas")

        # Información de columnas
        st.subheader("📝 Información de columnas")
        try:
            info_columnas = self.eda.obtener_info_columnas()
            st.dataframe(info_columnas, use_container_width=True)
        except Exception as e:
            ComponentesUI.mostrar_error(f"Error al obtener información de columnas: {str(e)}", "warning")

    def _mostrar_analisis_completo(self) -> None:
        """Muestra análisis completo con tabs organizadas"""
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Información Básica", "📈 Estadísticas", "❌ Valores Faltantes", "🔍 Análisis Detallado", "📈 Gráficos"]
        )

        with tab1:
            self._mostrar_vista_mejorada()

        with tab2:
            st.subheader("📈 Estadísticas Descriptivas")
            try:
                estadisticas = self.eda.obtener_resumen_estadisticas()
                if estadisticas is not None:
                    st.dataframe(estadisticas, use_container_width=True)
                else:
                    st.warning("No se pudieron generar estadísticas descriptivas")
            except Exception as e:
                ComponentesUI.mostrar_error(f"Error al obtener estadísticas: {str(e)}", "warning")

        with tab3:
            st.subheader("❌ Valores Faltantes")
            try:
                valores_faltantes = self.eda.obtener_valores_faltantes()

                if valores_faltantes is not None:
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
                                    st.bar_chart(chart_data.set_index('Columna')['Porcentaje'])
                        except Exception:
                            st.warning("No se pudo generar el gráfico de valores faltantes")
                else:
                    st.warning("No se pudo obtener información de valores faltantes")
            except Exception as e:
                ComponentesUI.mostrar_error(f"Error al obtener valores faltantes: {str(e)}", "warning")

        with tab4:
            st.subheader("🔍 Análisis Detallado")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Distribución de tipos de datos:**")
                tipos_resumen, _ = self.manejador_datos.obtener_tipos_datos_procesados()
                if tipos_resumen:
                    chart_data = pd.DataFrame(
                        list(tipos_resumen.items()),
                        columns=['Tipo', 'Cantidad']
                    )
                    st.bar_chart(chart_data.set_index('Tipo'))

            with col2:
                st.write("**Top 10 columnas con más valores únicos:**")
                try:
                    info_columnas = self.eda.obtener_info_columnas()
                    columnas_validas = info_columnas[info_columnas['Valores_únicos'] != 'N/A']
                    if len(columnas_validas) > 0:
                        top_unique = columnas_validas.nlargest(10, 'Valores_únicos')[['Columna', 'Valores_únicos']]
                        st.dataframe(top_unique, use_container_width=True)
                    else:
                        st.warning("No se encontraron columnas con valores únicos válidos")
                except Exception:
                    st.warning("No se pudo generar el top de columnas con valores únicos")
        with tab5:
            st.subheader("📈 Gráficos")

            tipo_grafico = st.selectbox("Seleccione el tipo de gráfico",
                                        ["Selecciona","Distribucciones", "Boxplot", "Matriz de Correlación", "Univariado"])

            columna = st.selectbox("Seleccione la variable para el eje X", self.eda.df.columns)

            if tipo_grafico == "Distribucciones":
                st.pyplot(self.eda.obtener_analisis_distribucion(columna))
            elif tipo_grafico == "Boxplot":
                st.plotly_chart(self.eda.obtener_analisis_boxplot(columna), use_container_width=True)
            elif tipo_grafico == "Matriz de Correlación":
                st.pyplot(self.eda.obtener_analisis_correlaccion(), use_container_width=True)
            elif tipo_grafico == "Univariado":
                st.pyplot(self.eda.obtener_analisis_univariados(columna), use_container_width=True)

    def render(self) -> None:
        """Método principal para renderizar la página de estadísticas"""
        ComponentesUI.crear_titulo_principal(
            "📊 Estadísticas Básicas",
            "Análisis completo y visual de tus datos"
        )

        if not self._inicializar_dependencias():
            return

        # Selector de vista
        vista_option = st.radio(
            "Selecciona el tipo de análisis:",
            [vista.value for vista in TipoVista],
            horizontal=True
        )

        # Mapeo de vistas a métodos
        vista_methods = {
            TipoVista.MEJORADA.value: self._mostrar_vista_mejorada,
            TipoVista.CLASICA.value: self._mostrar_vista_clasica,
            TipoVista.COMPLETA.value: self._mostrar_analisis_completo
        }

        # Ejecutar vista seleccionada
        method = vista_methods.get(vista_option)
        if method:
            try:
                method()
            except Exception as e:
                ComponentesUI.mostrar_error(f"Error inesperado: {str(e)}")
                logger.exception("Error inesperado en render")
        else:
            ComponentesUI.mostrar_error("Vista no reconocida", "warning")