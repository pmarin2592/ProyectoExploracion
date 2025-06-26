"""
Clase: MenuPrincipal

Objetivo: Clase que mantiene el menu principal del proyecto

Cambios:

    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
"""
import os
import streamlit as st
from PIL import Image

from src.visualizacion.PaginaACP import PaginaACP
from src.visualizacion.PaginaAFC import PaginaAFC
from src.visualizacion.PaginaAcerca import PaginaAcerca
from src.visualizacion.PaginaCluster import PaginaCluster
from src.visualizacion.PaginaDatos import PaginaDatos
from src.visualizacion.PaginaEstadisticas import PaginaEstadisticas
from src.visualizacion.PaginaKmeans import PaginaKmeans


class MenuPrincipal:
    def __init__(self, base_dir):
        self.__menu ={
                    "🗂️ Datos":PaginaDatos(),                               # Representa archivos/datasets
                    "📈 Estadísticas Básicas": PaginaEstadisticas(),        # Para resúmenes y gráficos básicos
                    "🧮 ACP (Componentes Principales)": PaginaACP(),        # Matemática / reducción dimensional
                    "🔠 AFC (Correspondencias)": PaginaAFC(),               # Relacionado con datos categóricos
                    "🧬 Clúster Jerárquico": PaginaCluster(),               # Jerarquía y análisis agrupado
                    "🔢 K-Means": PaginaKmeans(),                           # Algoritmo de clustering
                    "💡 Acerca de": PaginaAcerca()                          # Información general / autores / contexto
                }
        self.__base_dir = base_dir
        # Inicializar la página seleccionada si no existe
        if 'pagina_seleccionada' not in st.session_state:
            st.session_state.pagina_seleccionada = "🗂️ Datos"

    def menu_principal(self):
        # Abre la imagen desde la misma carpeta que main.py
        logo_path = os.path.join(self.__base_dir, "logo-cuc.png")
        logo = Image.open(logo_path)

        st.set_page_config(page_title="Aplicativo de Mineria", layout="wide", page_icon="📊")

        # Header del sidebar con logo y título
        st.sidebar.image(logo, width=120)
        st.sidebar.markdown("""
            <div style='text-align: center; margin-bottom: 30px;'>
                <h2 style='color: #1f77b4; margin-bottom: 5px;'>Menú Principal</h2>
                <hr style='margin: 10px 0; border: 1px solid #dee2e6;'>
            </div>
        """, unsafe_allow_html=True)

        # CSS para ocultar botones por defecto y estilizar
        st.sidebar.markdown("""
            <style>
            .stButton button {
                background-color: #f8f9fa;
                color: #333333;
                border: 2px solid #dee2e6;
                border-radius: 10px;
                padding: 12px 16px;
                width: 100%;
                text-align: left;
                font-size: 14px;
                font-weight: 400;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
                margin: 5px 0;
            }
            .stButton button:hover {
                background-color: #e9ecef;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
            }
            .stButton button:focus {
                background-color: #1f77b4 !important;
                color: #ffffff !important;
                border-color: #1f77b4 !important;
                box-shadow: 0 4px 8px rgba(31, 119, 180, 0.3) !important;
            }
            .menu-button-active .stButton button {
                background-color: #1f77b4 !important;
                color: #ffffff !important;
                border-color: #1f77b4 !important;
                box-shadow: 0 4px 8px rgba(31, 119, 180, 0.3) !important;
                font-weight: 600 !important;
            }
            .menu-description {
                font-size: 11px;
                color: #6c757d;
                margin-top: -10px;
                margin-bottom: 10px;
                margin-left: 5px;
                font-style: italic;
            }
            </style>
        """, unsafe_allow_html=True)

        # Diccionario con descripciones para cada opción
        descripciones = {
            "🗂️ Datos": "Gestión de datasets",
            "📈 Estadísticas Básicas": "Análisis exploratorio",
            "🧮 ACP (Componentes Principales)": "Reducción dimensional",
            "🔠 AFC (Correspondencias)": "Análisis categórico",
            "🧬 Clúster Jerárquico": "Agrupamiento jerárquico",
            "🔢 K-Means": "Clustering K-Means",
            "💡 Acerca de": "Información del proyecto"
        }

        # Crear botones para cada opción del menú
        for nombre_opcion in self.__menu.keys():
            es_activo = st.session_state.pagina_seleccionada == nombre_opcion
            descripcion = descripciones.get(nombre_opcion, "")

            # Contenedor con clase condicional para botón activo
            if es_activo:
                st.sidebar.markdown('<div class="menu-button-active">', unsafe_allow_html=True)

            # Botón principal
            if st.sidebar.button(nombre_opcion, key=f"btn_{nombre_opcion}",
                               use_container_width=True):
                st.session_state.pagina_seleccionada = nombre_opcion
                st.rerun()

            # Descripción debajo del botón
            st.sidebar.markdown(f'<div class="menu-description">{descripcion}</div>',
                              unsafe_allow_html=True)

            if es_activo:
                st.sidebar.markdown('</div>', unsafe_allow_html=True)

        # Espaciador
        st.sidebar.markdown("<br>", unsafe_allow_html=True)

        # Renderizar la página seleccionada
        pagina = self.__menu[st.session_state.pagina_seleccionada]
        pagina.render()

        # Footer mejorado
        st.sidebar.markdown(
            """
            <hr style='margin-top: 50px; border: 1px solid #dee2e6;'>
            <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; 
                        text-align: center; margin-top: 10px;'>
                <div style='font-size: 0.9em; color: #1f77b4; font-weight: 600;'>
                    © 2025 | Minería De Datos I
                </div>
                <div style='font-size: 0.8em; color: #6c757d; margin-top: 5px;'>
                    Big Data CUC
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )