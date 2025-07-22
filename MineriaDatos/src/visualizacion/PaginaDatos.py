"""
Clase: PaginaDatos

Objetivo: Clase que mantiene la pagina de la carga de datos

Cambios:

    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
    2. Validar que haya datos seleccionados para habilitar opciones y
    mantener por medio de variable de entorno st.session_state.analisis_generado
    la tabla que muestra la informacion, aun navegando por las diversas pantallas
    aquesada 28-06-2025
    3. Cambios de mejoras sugeridas, cambio en String, tooltip con info de la columna aquesada 07-07-2025
    4. Solucionado problema de actualización cuando cambia el delimitador - 07-07-2025

"""

import streamlit as st
import pandas as pd
import time

from src.eda.EstadisticasBasicasEda import EstadisticasBasicasEda


class PaginaDatos:
    def __init__(self):
        self.archivo_cargado = None
        self.analisis_realizado = False

    def _obtener_tipo_dato_legible(self, tipo_pandas):
        """Convierte el tipo de dato de pandas a una descripción más legible"""
        tipo_str = str(tipo_pandas)

        if 'int' in tipo_str:
            return "🔢 Entero"
        elif 'float' in tipo_str:
            return "🔢 Decimal"
        elif 'bool' in tipo_str:
            return "☑️ Booleano"
        elif 'datetime' in tipo_str:
            return "📅 Fecha/Hora"
        elif 'object' in tipo_str:
            return "📝 Texto"
        elif 'category' in tipo_str:
            return "🏷️ Categoría"
        else:
            return f"❓ {tipo_str}"

    def _crear_tabla_con_tooltips(self, df):
        """Crea la tabla con tooltips en las columnas"""
        # Crear información de tooltips
        tooltips = {}
        for col in df.columns:
            tipo_dato = self._obtener_tipo_dato_legible(df[col].dtype)
            valores_nulos = df[col].isnull().sum()
            total_valores = len(df)
            porcentaje_nulos = (valores_nulos / total_valores) * 100

            tooltip_text = f"{tipo_dato}\n📊 {total_valores - valores_nulos} valores válidos\n❌ {valores_nulos} valores nulos ({porcentaje_nulos:.1f}%)"
            tooltips[col] = tooltip_text

        # Mostrar la tabla principal
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                col: st.column_config.Column(
                    col,
                    help=tooltips[col],
                    width="medium"
                ) for col in df.columns
            }
        )

    def render(self):
        try:
            st.title("🗃️ Carga de Datos")

            # Selección de delimitador y carga de archivo
            col1, col2 = st.columns([1, 2])
            with col1:
                delimitador = st.selectbox(
                    "Selecciona el delimitador del archivo:",
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
                prev_delimitador = st.session_state.get('delimitador')

                # Si es un archivo nuevo o cambia el delimitador:
                if prev_name != nombre or prev_delimitador != delimitador:
                    try:
                        # Leer según extensión
                        if nombre.lower().endswith('.csv'):
                            df = pd.read_csv(archivo, delimiter=delimitador)
                        else:
                            df = pd.read_excel(archivo, engine='openpyxl')

                        # Guardar en session_state
                        st.session_state.df_cargado = df
                        st.session_state.file_name = nombre
                        st.session_state.delimitador = delimitador
                        st.session_state.analisis_generado = False

                        st.success(f"✅ Archivo '{nombre}' cargado exitosamente.")
                    except Exception as e:
                        st.error(f"Error al cargar el archivo: {e}")

                # Siempre asignar el df a self para mostrar
                self.archivo_cargado = st.session_state.get('df_cargado')

            # Mostrar la tabla si existe df cargado
            if 'df_cargado' in st.session_state:
                st.subheader("Vista previa del dataset")

                # Mostrar estadísticas básicas
                df = st.session_state.df_cargado
                col_stats1, col_stats2, col_stats3 = st.columns(3)

                with col_stats1:
                    st.metric("📊 Filas", f"{len(df):,}")
                with col_stats2:
                    st.metric("📋 Columnas", len(df.columns))
                with col_stats3:
                    st.metric("📏 Tamaño", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

                # Crear tabla con tooltips
                self._crear_tabla_con_tooltips(df)

                # Botón para análisis
                col3, col4, col5 = st.columns([3, 2, 3])
                with col4:
                    if st.button("📊 Generar análisis"):
                        progreso = st.progress(0, text="Iniciando análisis...")
                        for i in range(1, 101):
                            time.sleep(0.02)
                            progreso.progress(i, text=f"Analizando datos... {i}%")

                        st.session_state.eda = EstadisticasBasicasEda(df)
                        st.success("✅ Análisis completado exitosamente.")
                        self.analisis_realizado = True
                        st.session_state.analisis_generado = True
                        st.rerun()

            else:
                st.info("Por favor sube un archivo CSV o Excel (.xlsx).")

        except Exception as e:
            st.error(f"Ocurrió un error en la pantalla de datos: {e}")