import pandas as pd
from data_preprocessing.preprocessing import Preprocessing
from eda.eda import EDA
from feature_selection.feature_selection import Feature_Selection
from modelling.modelling import Modelling

class Tulika_Predicomm:
    def predicomm(raw_data_path, commodity, category, category_commodity):
        raw_data = pd.read_csv(f'{raw_data_path}')
        commodity = commodity
        category = category

        df = Preprocessing.preprocessing(raw_data=raw_data)
        EDA.overall_eda(df=df,category_commodity=category_commodity)
        print(df.columns)
        EDA.pca_clustering(df, category_commodity)
        df_summary = EDA.target_eda(df, commodity)

        _, selected_features = Feature_Selection.feature_analysis(commodity, df)
        print(' ')
        print('Features selected after every method : ')
        for i, j in _.items():
            print(i)
            print(j)
        print(selected_features)
        print(' ')


        best_model = Modelling.modelling(df, selected_features, commodity)
        Modelling.future_forecast(df, commodity, best_model)
        return df
    

category_commodity = {
    "ENERGY": [
        "NATURAL GAS",
        "LOW SULPHUR GAS OIL",
        "WTI CRUDE",
        "BRENT CRUDE",
        "ULS DIESEL",
        "GASOLINE",
    ], 
    "INDUSTRIAL METALS": [
        "COPPER",
        "ALUMINIUM",
        "ZINC",
        "NICKEL",
    ],
    "PRECIOUS METALS": [
        "GOLD",
        "SILVER",
    ],
    "GRAINS": [
        "CORN",
        "SOYBEANS",
        "WHEAT",
        "SOYBEAN OIL",
        "SOYBEAN MEAL",
        "HRW WHEAT",
    ],
    "LIVESTOCK": [
        "LIVE CATTLE",
        "LEAN HOGS",
    ],
    "SOFTS": [
        "SUGAR",
        "COFFEE",
        "COTTON",
    ],
}


file_path = r'C:\Users\tulik\Desktop\IGDTUW\ML\ML Lab\predicomm\predicomm\dataset\commodity_futures.csv'
commodity = 'SILVER'
category = 'PRECIOUS METALS'
print(Tulika_Predicomm.predicomm(raw_data_path=file_path,
                           commodity=commodity,
                           category=category,
                           category_commodity=category_commodity))


