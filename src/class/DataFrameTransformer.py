import pandas as pd

class DataFrameTransformer:
    def __init__(self, df):
        """
        Initialise la classe avec un DataFrame.
        
        Args:
        df (pd.DataFrame): Le DataFrame à transformer.
        """
        self.df = df

    def average_speed(self, distance_col, time_col, dest_col):
        """
        Calcule la vitesse moyenne et l'ajoute en tant que nouvelle colonne au DataFrame.
        
        Args:
        distance_col (str): Le nom de la colonne de distance.
        time_col (str): Le nom de la colonne de temps en secondes.
        dest_col (str): Le nom de la colonne de destination pour la vitesse moyenne.
        
        Returns:
        pd.DataFrame: Le DataFrame modifié avec une nouvelle colonne pour la vitesse moyenne.
        """
        self.df[dest_col] = self.df[distance_col] / (self.df[time_col] / 3600)
        return self.df

# Exemple d'utilisation
# df_merged = pd.DataFrame({'distance': [10, 20], 'AttendanceTimeSeconds': [3600, 7200]})
# transformer = DataFrameTransformer(df_merged)
# df_merged = transformer.average_speed('distance', 'AttendanceTimeSeconds', 'VitesseMoy')
# print(df_merged)