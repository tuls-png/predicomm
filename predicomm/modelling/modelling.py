import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import timedelta
from sklearn.metrics import r2_score, mean_squared_error


class Modelling:    
    def modelling(df, selected_features, target_column):
        df["Year"] = df.index.year  

        train = df[df["Year"] <= 2021]
        validate = df[(df["Year"] > 2021) & (df["Year"] <= 2022)]
        test = df[df["Year"] > 2022]

        print("Train set size:", train.shape)
        print("Validation set size:", validate.shape)
        print("Test set size:", test.shape)

        
        X_train, y_train = train[selected_features], train[target_column]
        X_validate, y_validate = validate[selected_features], validate[target_column]
        X_test, y_test = test[selected_features], test[target_column]

        
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42)
        }

        
        results = {}
        for model_name, model in models.items():            
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_validate)
            y_test_pred = model.predict(X_test)

            rmse_val = np.sqrt(mean_squared_error(y_validate, y_val_pred))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

            accuracy_val = 1 - (np.sum(np.abs(y_validate - y_val_pred)) / np.sum(np.abs(y_validate)))
            accuracy_test = 1 - (np.sum(np.abs(y_test - y_test_pred)) / np.sum(np.abs(y_test)))
            
            results[model_name] = {
                "Validation RMSE": rmse_val,
                "Test RMSE": rmse_test,
                "Validation Accuracy": accuracy_val,
                "Test Accuracy": accuracy_test,
            }

        
        best_model_name = max(results, key=lambda x: results[x]["Test Accuracy"])
        best_model = models[best_model_name]

        
        print("Model Performance Metrics:")
        for model_name, metrics in results.items():
            print(f"{model_name}:")
            print(f"  Validation RMSE: {metrics['Validation RMSE']:.4f}")
            print(f"  Test RMSE: {metrics['Test RMSE']:.4f}")
            print(f"  Validation Accuracy: {metrics['Validation Accuracy']:.4f}")
            print(f"  Test Accuracy: {metrics['Test Accuracy']:.4f}\n")

        print(f"Best Model: {best_model_name}")

        
        best_model.fit(X_train, y_train)  
        y_test_pred = best_model.predict(X_test)

        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label="Actual Values", color="blue")
        plt.plot(y_test_pred, label=f"Predicted Values ({best_model_name})", color="orange")
        plt.title(f"Actual vs Predicted Values ({best_model_name})")
        plt.xlabel("Sample Index")
        plt.ylabel("Target Value")
        plt.legend()
        plt.show()
        return best_model
        
    def future_forecast(df, target_column, best_model):

        forecast_period = 30
        df['Shifted_Target'] = df[target_column].shift(-forecast_period)

        train_data = df.dropna(subset=['Shifted_Target'])

        # Test data starts where the shifted target becomes NaN
        test_data = df[df['Shifted_Target'].isna()]

        # Prepare training and test data
        X_train = train_data.drop(columns=[target_column, 'Shifted_Target'])
        y_train = train_data['Shifted_Target']

        X_test = test_data.drop(columns=[target_column, 'Shifted_Target']).iloc[:forecast_period]
        last_date = df.index[-1]

        # Train a model
        model = best_model
        model.fit(X_train, y_train)

        # Make predictions for test data
        y_pred = model.predict(X_test)

        # Generate future dates starting from the last known date
        future_dates = [last_date + timedelta(days=i + 1) for i in range(len(y_pred))]

        # Combine results into a DataFrame
        results = pd.DataFrame({
            'Date': future_dates,
            f'Predicted {target_column}': y_pred
        })

        # Display results
        print("Future Predictions:")
        print(results)

        # Filter data for 2023
        df_2023 = df[df.index.year == 2023]

        plt.figure(figsize=(12, 6))

        # Plot historical values for 2023
        plt.plot(df_2023.index, df_2023[target_column], label='Historical Values (2023)', color='blue')

        # Plot future predictions for 2023
        plt.plot(results['Date'], results[f'Predicted {target_column}'], label='Future Predictions', color='orange', linestyle='--')

        plt.title(f"Historical and Future Predictions for {target_column} in 2023")
        plt.xlabel("Date")
        plt.ylabel(target_column)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
