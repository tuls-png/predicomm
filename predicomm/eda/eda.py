import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

class EDA:

    def correlation_heatmap(df):
        correlation_matrix = df.corr()
        plt.figure(figsize=(14, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Commodity Prices')
        plt.show()

    def histogram(df, category_commodity):
        histogram_colors = ["skyblue", "lightcoral", "mediumseagreen", "gold", "plum", "darkorange"]
        count = 0
        for category, columns in category_commodity.items():
            available_columns = [col for col in columns if col in df.columns]
            # Create a histogram for each category
            plt.figure(figsize=(15, 10))            
            for i, column in enumerate(available_columns, 1):
                plt.subplot(2, 3, i)  
                df[column].dropna().hist(bins=30, alpha=0.75, color=histogram_colors[count])
                plt.title(column)
                plt.xlabel("Values")
                plt.ylabel("Frequency")
            count+=1

            plt.suptitle(f"Histograms for {category}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()  

    def pca_clustering(df, category_commodity):
        commodities = [commodity for category in category_commodity.values() for commodity in category]

        print(commodities)

        pivoted_df = df[commodities]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pivoted_df)

        # Perform K-Means Clustering
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        pivoted_df['Cluster'] = kmeans.fit_predict(scaled_data)

        # Apply PCA to reduce dimensionality to 2D
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(scaled_data)        

        
        plt.figure(figsize=(10, 8))
        palette = sns.color_palette('pastel', n_clusters)

        for cluster in range(n_clusters):
            cluster_points = pca_components[pivoted_df['Cluster'] == cluster]
            
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=100, label=f"Cluster {cluster}",
                        alpha=0.7, edgecolor='k')
            if len(cluster_points) > 2:  
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                plt.fill(hull_points[:, 0], hull_points[:, 1], color=palette[cluster], alpha=0.2)

        plt.title("K-Means Clustering of Commodities with Cluster Boundaries (2D PCA)", fontsize=16)
        plt.xlabel("PCA Component 1", fontsize=14)
        plt.ylabel("PCA Component 2", fontsize=14)
        plt.legend(title="Cluster")
        plt.show()

        # Analyzing the contributions of each feature to each PCA component
        explained_variance = pca.explained_variance_ratio_

        print(f"PCA Component 1 explains {explained_variance[0]:.2%} of the variance.")
        print(f"PCA Component 2 explains {explained_variance[1]:.2%} of the variance.\n")

      
        feature_contributions = pd.DataFrame(
            pca.components_.T,  
            index=commodities, 
            columns=["PCA Component 1", "PCA Component 2"] 
        )

        print("Feature Contributions to Each Principal Component:")
        print(feature_contributions)

        # Commodities in each cluster and their time frame
        commodities_repeated = np.tile(commodities, len(pivoted_df) // len(commodities))
        commodities_repeated = np.concatenate([commodities_repeated, commodities[:len(pivoted_df) % len(commodities)]])

        pivoted_df['Commodity'] = commodities_repeated  

        pivoted_df = pivoted_df.reset_index()
        cluster_time_frame = pivoted_df.groupby(['Cluster', 'Date'])['Commodity'].apply(list)
        cluster_date_range = pivoted_df.groupby('Cluster')['Date'].agg(['min', 'max'])
        print("\nTime Frame for Each Cluster:")
        print(cluster_date_range)
        print("\nCommodities and Time Frame for Each Cluster:")
        for cluster in range(n_clusters):
            print(f"\nCluster {cluster}:")
            time_range = cluster_date_range.loc[cluster]
            print(f"Time frame: {time_range['min']} to {time_range['max']}")
            commodities_in_cluster = cluster_time_frame[cluster_time_frame.index.get_level_values('Cluster') == cluster]
            unique_commodities = set([item for sublist in commodities_in_cluster for item in sublist])              
            print(f"Commodities: {sorted(unique_commodities)}")  



    def overall_eda(df, category_commodity):
        
        # Correlation heatmap for all commodities
        EDA.correlation_heatmap(df)

        # Histogram for each category
        EDA.histogram(df, category_commodity)       

        




    def target_eda(df, commodity):
        print(f"\nSummary statistics of target {commodity}:")
        df_summary = df[commodity].describe()

        rolling_mean = df[commodity].rolling(window=30).mean()

        # Plot the actual values and rolling mean
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[commodity], label=f"{commodity} Actual", alpha=0.7)
        plt.plot(df.index, rolling_mean, label=f"{commodity} 30-day Rolling Mean", color='red', linestyle='--', linewidth=2)
        plt.title(f"Trend Analysis of {commodity} Over Time")
        plt.xlabel("Date")
        plt.ylabel(commodity)
        plt.legend()
        plt.grid()
        plt.show()
        
        # Aggregate data yearly
        df['Year'] = df.index.year  
        yearly_data = df.groupby('Year')[commodity].mean()  # Replace commodity with your column of interest

        # Plot yearly averages
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_data.index, yearly_data, marker='o', linestyle='-', color='b', label='Yearly Average')
        plt.title(f"Yearly Average of {commodity} Prices")
        plt.xlabel("Year")
        plt.ylabel(f"Average {commodity} Price")
        plt.grid()
        plt.legend()
        plt.show()

        # Calculate a rolling mean with a yearly window (365 days)
        rolling_year = df[commodity].rolling(window=365).mean()

        # Plot actual vs. yearly rolling mean
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[commodity], label=f'Actual {commodity} Prices', alpha=0.5)
        plt.plot(df.index, rolling_year, label='Yearly Rolling Mean (365 days)', color='red', linestyle='--', linewidth=2)
        plt.title(f"Yearly Rolling Mean of {commodity} Prices")
        plt.xlabel("Date")
        plt.ylabel(f"{commodity} Prices")
        plt.legend()
        plt.grid()
        plt.show()

 
        df['Month'] = df.index.month
        monthly_data = df.groupby(['Year', 'Month'])[commodity].mean().unstack(level=0)

        # Heatmap to visualize seasonality
        plt.figure(figsize=(10, 8))
        sns.heatmap(monthly_data, cmap='coolwarm', annot=False, cbar=True)
        plt.title(f"Monthly Seasonality of {commodity} Prices (Year-on-Year)")
        plt.xlabel("Year")
        plt.ylabel("Month")
        plt.show()

        # Calculate YoY percentage change
        yoy_change = df[commodity].resample('Y').mean().pct_change() * 100
        plt.figure(figsize=(10, 6))
        plt.bar(yoy_change.index.year, yoy_change, color='skyblue', alpha=0.7)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.title(f"Year-on-Year Percentage Change in {commodity} Prices")
        plt.xlabel("Year")
        plt.ylabel("Percentage Change (%)")
        plt.grid()
        plt.show()

        return df_summary
