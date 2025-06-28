"""
Clase: PaginaDatos

Objetivo: Clase que mantiene la pagina de la carga de datos

Cambios:

    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
"""

import streamlit as st
import pandas as pd
import time

class PaginaDatos:
    def __init__(self):
        self.archivo_cargado = None
        self.analisis_realizado = False

    def render(self):
        try:
            st.title("🗃️ Carga de Datos")

            # Selección de delimitador y carga de archivo
            col1, col2 = st.columns([1, 2])
            with col1:
                delimitador = st.selectbox(
                    "Selecciona el delimitador de tu archivo CSV:",
                    options=[",", ";", "\t"],
                    format_func=lambda x: f"{x}"
                )
            with col2:
                archivo = st.file_uploader(
                    "Sube tu archivo (.csv o .xlsx)",
                    type=["csv", "xlsx"],
                    key="uploader"
                )

            # Si se subió un archivo
            if archivo is not None:
                nombre = archivo.name
                prev_name = st.session_state.get('file_name')
                # Si es un archivo nuevo o cambia:
                if prev_name != nombre:
                    try:
                        # Leer según extensión
                        if nombre.lower().endswith('.csv'):
                            df = pd.read_csv(archivo, delimiter=delimitador)
                        else:
                            df = pd.read_excel(archivo, engine='openpyxl')

                        # Guardar en session_state
                        st.session_state.df_cargado = df
                        st.session_state.file_name = nombre
                        st.session_state.analisis_generado = False

                        st.success(f"✅ Archivo '{nombre}' cargado exitosamente.")
                    except Exception as e:
                        st.error(f"Error al cargar el archivo: {e}")

                # Siempre asignar el df a self para mostrar
                self.archivo_cargado = st.session_state.get('df_cargado')

            # Mostrar la tabla si existe df cargado
            if 'df_cargado' in st.session_state:
                st.subheader("Vista previa del dataset")
                st.dataframe(st.session_state.df_cargado)

                # Botón para análisis
                col3, col4, col5 = st.columns([3, 2, 3])
                with col4:
                    if st.button("📊 Generar análisis"):
                        progreso = st.progress(0, text="Iniciando análisis...")
                        for i in range(1, 101):
                            time.sleep(0.02)
                            progreso.progress(i, text=f"Analizando datos... {i}%")

                        st.success("✅ Análisis completado exitosamente.")
                        self.analisis_realizado = True
                        st.session_state.analisis_generado = True
                        st.rerun()

            else:
                st.info("Por favor sube un archivo CSV o Excel (.xlsx).")

        except Exception as e:
            st.error(f"Ocurrió un error en la pantalla de datos: {e}")