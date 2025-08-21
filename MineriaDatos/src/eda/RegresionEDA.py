import inspect
import pandas as pd
from typing import Dict, Any, List

from src.modelos.Regresion import Regresion


class RegresionEda:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def regresion_lineal_simple(self, x: str, y: str) -> tuple[Dict[str, Any], str]:
        """
        Ejecuta regresión lineal simple y devuelve resultados + código de la función usada.
        """
        resultados = Regresion.regresion_lineal_simple(self.df, x, y)
        codigo = inspect.getsource(Regresion.regresion_lineal_simple)
        return resultados, codigo

    def regresion_lineal_multiple(self, x_vars: List[str], y: str) -> tuple[Dict[str, Any], str]:
        """
        Ejecuta regresión lineal múltiple y devuelve resultados + código de la función usada.
        """
        resultados = Regresion.regresion_lineal_multiple(self.df, x_vars, y)
        codigo = inspect.getsource(Regresion.regresion_lineal_multiple)
        return resultados, codigo
