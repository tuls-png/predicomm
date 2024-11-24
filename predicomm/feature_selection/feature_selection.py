import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Feature_Selection:
    def feature_analysis(target_column, data):
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Standardize the features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # 1. Correlation
        correlations = X.corrwith(y).abs()
        top_corr_features = correlations[correlations > 0.6].index.tolist()

        # 2. Multicollinearity (VIF)
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
        top_vif_features = vif_data[vif_data['VIF'] < 5]['Feature'].tolist()

        # 3. Mutual Information Gain
        mi_scores = mutual_info_regression(X_scaled, y)
        mi_features = X.columns[mi_scores > np.mean(mi_scores)].tolist()

        # 4. K Best Features
        selector = SelectKBest(score_func=f_regression, k=5)
        selector.fit(X_scaled, y)
        k_best_features = X.columns[selector.get_support()].tolist()

        feature_counts = pd.Series(
            top_corr_features + top_vif_features + mi_features + k_best_features
        ).value_counts()
        selected_features = feature_counts[feature_counts >= 3].index.tolist()

        return {
            "Correlation": top_corr_features,
            "High Multicollinearity Features": top_vif_features,
            "Mutual Information": mi_features,
            "K Best Features": k_best_features,
            "Selected Features (>= 3 methods)": selected_features
        }, selected_features

