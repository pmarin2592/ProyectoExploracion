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

# Configurar logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class TipoVista(Enum):
    """Enum para los tipos de vista disponibles"""
    MEJORADA = "🚀 Vista Mejorada"
    CLASICA = "📋 Vista Clásica"
    COMPLETA = "🔍 Análisis Completo"


@dataclass
class ConfigMetrica:
    """Configuración para métricas visuales"""
    emoji: str
    color: str
    label: str = ""


class ConfiguracionTipos:
    """Configuración centralizada para tipos de datos y colores"""

    TIPO_CONFIG = {
        'object': ConfigMetrica('📝', '#FF6B6B'),
        'float64': ConfigMetrica('🔢', '#4ECDC4'),
        'int64': ConfigMetrica('🔢', '#45B7D1'),
        'datetime64': ConfigMetrica('📅', '#96CEB4'),
        'bool': ConfigMetrica('✅', '#FFEAA7'),
        'category': ConfigMetrica('📊', '#DDA0DD')
    }

    COLORES_METRICAS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    @classmethod
    def obtener_config_tipo(cls, dtype: str) -> ConfigMetrica:
        """Obtiene configuración para un tipo de dato"""
        return cls.TIPO_CONFIG.get(dtype, ConfigMetrica('❓', '#95A5A6'))


class UtilDataFrame:
    """Utilidades para manipulación de DataFrames"""

    @staticmethod
    def convertir_tipos_json_serializable(df: pd.DataFrame) -> pd.DataFrame:
        """Convierte tipos problemáticos a tipos JSON-serializables"""
        df_copy = df.copy()

        conversiones = {
            'Float64': 'float64',
            'Int64': 'int64',
            'string': 'object',
            'boolean': 'bool'
        }

        for col in df_copy.columns:
            dtype_str = str(df_copy[col].dtype)

            for tipo_problematico, tipo_destino in conversiones.items():
                if tipo_problematico in dtype_str:
                    df_copy[col] = df_copy[col].astype(tipo_destino)
                    break

        return df_copy

    @staticmethod
    def corregir_dataframe_para_streamlit(df: pd.DataFrame) -> pd.DataFrame:
        """Prepara DataFrame para Streamlit sin errores de Arrow"""
        df_copy = df.copy()

        for col in df_copy.columns:
            dtype = df_copy[col].dtype

            if dtype == 'object':
                df_copy[col] = df_copy[col].astype(str)
            elif dtype.name.startswith('datetime64'):
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            elif dtype.name.startswith('int') and df_copy[col].isnull().any():
                df_copy[col] = df_copy[col].astype('float64')
            elif dtype.name == 'category':
                df_copy[col] = df_copy[col].astype(str)

        # Limpiar valores problemáticos
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
        df_copy.columns = df_copy.columns.astype(str)

        return df_copy


class ComponentesUI:
    """Componentes reutilizables de interfaz"""

    @staticmethod
    def mostrar_error(mensaje: str, tipo_error: str = "error") -> None:
        """Muestra mensaje de error en Streamlit"""
        iconos = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}

        getattr(st, tipo_error)(f"{iconos.get(tipo_error, '⚠️')} {mensaje}")
        logger.error(f"Error en Streamlit: {mensaje}")

    @staticmethod
    def crear_titulo_principal(titulo: str, subtitulo: str) -> None:
        """Crea título principal con gradiente"""
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        ">
            <h1 style="color: white; margin: 0; font-size: 3rem;">{titulo}</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 1rem 0 0 0; font-size: 1.2rem;">
                {subtitulo}
            </p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def crear_metrica_visual(config: ConfigMetrica, valor: str, label: str) -> str:
        """Crea HTML para métrica visual"""
        return f"""
        <div style="
            background: linear-gradient(135deg, {config.color}22, {config.color}11);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            border: 2px solid {config.color}33;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{config.emoji}</div>
            <div style="font-size: 2rem; font-weight: bold; color: {config.color}; margin-bottom: 0.3rem;">
                {valor}
            </div>
            <div style="font-size: 1.1rem; color: #666; font-weight: 500;">
                {label}
            </div>
        </div>
        """

    @staticmethod
    def crear_chips_columnas(nombres_columnas: List[str]) -> str:
        """Crea chips visuales para nombres de columnas"""
        colores = ConfiguracionTipos.COLORES_METRICAS
        cols_html = ""

        for i, col in enumerate(nombres_columnas):
            color = colores[i % len(colores)]
            cols_html += f"""
            <span style="
                background: {color}33;
                color: {color};
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                margin: 0.2rem;
                display: inline-block;
                font-weight: 500;
                border: 1px solid {color}66;
            ">{col}</span>
            """
        return f"<div style='line-height: 2.5;'>{cols_html}</div>"


class ManejadorDatos:
    """Maneja la obtención y procesamiento de datos"""

    def __init__(self, eda: EstadisticasBasicasEda):
        self.eda = eda

    def obtener_info_basica_segura(self) -> Optional[Dict[str, Any]]:
        """Obtiene información básica con manejo de errores"""
        try:
            return self.eda.obtener_info_basica()
        except Exception as e:
            logger.exception("Error al obtener información básica")
            ComponentesUI.mostrar_error(f"Error al obtener información básica: {str(e)}")
            return None

    def obtener_tipos_datos_procesados(self) -> Tuple[Dict[str, int], int]:
        """Obtiene y procesa información de tipos de datos"""
        try:
            tipos_info = self.eda.obtener_tipo_datos()
            tipos_resumen = tipos_info['tipos_resumen']

            # Limpiar tipos problemáticos
            tipos_resumen_limpio = {}
            total_columnas = 0

            for dtype, count in tipos_resumen.items():
                dtype_str = str(dtype)
                count_int = int(count) if pd.notna(count) else 0
                tipos_resumen_limpio[dtype_str] = count_int
                total_columnas += count_int

            return tipos_resumen_limpio, total_columnas
        except Exception as e:
            logger.exception("Error al procesar tipos de datos")
            return {}, 0


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
                chips_html = ComponentesUI.crear_chips_columnas(nombres_columnas)
                st.markdown(chips_html, unsafe_allow_html=True)

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
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Información Básica", "📈 Estadísticas", "❌ Valores Faltantes", "🔍 Análisis Detallado"]
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