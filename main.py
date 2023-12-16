# Import necessary libraries import numpy as np
import pandas as pd
from sklearn.cluster import KMeans import matplotlib.pyplot as plt from sklearn import datasets
from sklearn.preprocessing import StandardScaler from sklearn.decomposition import PCA

# Load the Iris dataset iris = datasets.load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

# Standardize the features scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.iloc[:, :-1])

# Reduce the dimensions using PCA (optional) pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Determine the optimal number of clusters using the Elbow Method wcss = []
for i in range(1, 11):

kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(data_scaled) wcss.append(kmeans.inertia_)

# Plot the Elbow Method plt.plot(range(1, 11), wcss) plt.title('Elbow Method') plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares plt.show()

# Choose the optimal number of clusters (e.g., 3 in this case) and fit the K- means model
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(data_scaled)

# Add the cluster labels to the original dataset data['cluster'] = kmeans.labels_

# Visualize the clusters in 2D (PCA-reduced) space
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data['cluster'], cmap='viridis') plt.title('K-means Clustering of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2') plt.show()
