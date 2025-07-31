import pandas as pd

from src.eda.EstadisticasBasicasEda import EstadisticasBasicasEda
from typing import Dict, Any, Optional, Tuple, List
import logging

from src.helpers.ComponentesUI import ComponentesUI

# Configurar logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
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
