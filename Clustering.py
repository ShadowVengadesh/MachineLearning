from sklearn.datasets import make_blobs 

from sklearn.cluster import KMeans, AgglomerativeClustering 

import matplotlib.pyplot as plt 

# Generate a random dataset with 100 samples and 4 clusters 

X, y = make_blobs(n_samples=100, centers=4, random_state=42) 

# Create a K-Means clustering object with 4 clusters 

kmeans = KMeans(n_clusters=4, random_state=42) 

# Fit the K-Means model to the dataset 

kmeans.fit(X) 

# Create a scatter plot of the data colored by K-Means cluster assignment 

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_) 

plt.title("K-Means Clustering") 

plt.show() 

# Create a Hierarchical clustering object with 4 clusters 

hierarchical = AgglomerativeClustering(n_clusters=4) 

# Fit the Hierarchical model to the dataset hierarchical.fit(X) 

# Create a scatter plot of the data colored by Hierarchical cluster assignment 

plt.scatter(X[:, 0], X[:, 1], c=hierarchical.labels_) 

plt.title("Hierarchical Clustering")
plt.show()
