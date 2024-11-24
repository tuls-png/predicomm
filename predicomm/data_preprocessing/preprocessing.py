import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Preprocessing:

    def treatNA(df):
        df = df.ffill()
        df = df.bfill()
        return df
    
    def is_numeric(value):
        try:
            float(value)
            return True
        except ValueError:
            return False    

    
    def preprocessing(raw_data):
        df = raw_data.copy()
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
        date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max())
        df = df.set_index('Date').reindex(date_range)
        df = df.reset_index()
        df.rename(columns={'index': 'Date'}, inplace=True)
        df = df.set_index('Date')

        df.to_csv(r'C:\Users\tulik\Desktop\IGDTUW\ML\ML Lab\predicomm\predicomm\dataset\commodity_futures_alldates.csv', index=False)
        print('wooh')
        df = df[df.index.year >= 2018]

        df = Preprocessing.treatNA(df)

        df = df[df.applymap(Preprocessing.is_numeric).all(axis=1)]

        return df