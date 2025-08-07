"""
Clase: UtilDataframe

Objetivo: Utilidades para manipulación de DataFrames

Cambios:
    1. Componentes reutilizables de interfaz
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

import streamlit as st

# Configurar logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class TipoVista(Enum):
    """Enum para los tipos de vista disponibles"""
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

