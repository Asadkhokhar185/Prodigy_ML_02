import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Loading the dataset
data = pd.read_csv('C:/Users/PMLS/Downloads/archive/Mall_Customers.csv')

# features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Normalizing the data
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Using the Elbow Method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(features_normalized)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Choosing the optimal number of clusters
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(features_normalized)

# Add the cluster labels to the original data
data['Cluster'] = clusters

# Display the first few rows with the cluster labels
print(data.head())

# Optional: Visualize the clusters
plt.scatter(features_normalized[:, 0], features_normalized[:, 1], c=clusters, cmap='rainbow')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$) [Normalized]')
plt.ylabel('Spending Score (1-100) [Normalized]')
plt.show()