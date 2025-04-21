import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV, realiza un análisis básico y retorna un DataFrame.
    """

    # Cargar el dataset
    df = pd.read_csv(file_path)
    return df


# if __name__ == "__main__":
#     # Especificar la ruta del archivo CSV
#     file_path = 'telco_churn.csv'
#
#     # Llamar la función para cargar el dataset
#     dataset = load_dataset(file_path)
