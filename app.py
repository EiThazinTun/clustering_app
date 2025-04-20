import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Page configuration
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

# Title
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Sidebar slider for selecting k
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 3)

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Transform cluster centers for plotting
centers_2d = pca.transform(kmeans.cluster_centers_)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='tab10', s=50)
ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', s=200, marker='X', label='Centroids')
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

# Add custom legend
for i in range(k):
    ax.scatter([], [], color=scatter.cmap(i / k), label=f'Cluster {i}')
ax.legend()

# Display in Streamlit
st.pyplot(fig)
