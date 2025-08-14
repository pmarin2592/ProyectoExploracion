"""
Clase: GeneradorCodigoArbol

Objetivo: Generar el código del árbol
    1. Se genera el codigo basandose con lo que se habia mostrado del st_code pero aplicado al a´rbol de decision
"""

class GeneradorCodigoArbol:
    def __init__(self, target_col, columnas_predictoras, aplicar_binning, n_bins):
        self.target_col = target_col
        self.columnas_predictoras = columnas_predictoras
        self.aplicar_binning = aplicar_binning
        self.n_bins = n_bins

    def generar_codigo(self):
        binning_code = ""
        if self.aplicar_binning:
            binning_code = f"""
# Binning de variables numéricas
from sklearn.preprocessing import KBinsDiscretizer
num_cols = df_modelo.select_dtypes(include=['int64', 'float64']).columns.drop(target_col)
if len(num_cols) > 0:
    kb = KBinsDiscretizer(n_bins={self.n_bins}, encode='ordinal', strategy='uniform')
    df_modelo[num_cols] = kb.fit_transform(df_modelo[num_cols])
"""

        codigo = f"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv("tu_dataset.csv")

# Variables
target_col = "{self.target_col}"
columnas_predictoras = {self.columnas_predictoras}

# Preparar datos
df_modelo = df[columnas_predictoras + [target_col]].dropna()
{binning_code if self.aplicar_binning else ""}
# One-hot encoding
df_modelo = pd.get_dummies(df_modelo, drop_first=False)

# División train-test
X = df_modelo.drop(columns=[target_col])
y = df_modelo[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar modelo
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Evaluar modelo
acc = modelo.score(X_test, y_test)
print("Precisión:", acc)

# Graficar árbol
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(modelo, feature_names=X.columns, class_names=modelo.classes_, filled=True)
plt.show()
"""
        return codigo
