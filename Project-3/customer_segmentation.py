import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("First 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove CustomerID column
if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

# Convert Gender into numerical values
df = pd.get_dummies(df, drop_first=True)

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

print("\nPCA Variance Ratio:")
print(pca.explained_variance_ratio_)

# Elbow Method
wcss = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Silhouette Scores
print("\nSilhouette Scores:")

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_data)

    score = silhouette_score(pca_data, labels)

    print(f"K = {i}: {score:.3f}")

# Final Model
optimal_k = 5

kmeans = KMeans(n_clusters=optimal_k,
                random_state=42,
                n_init=10)

clusters = kmeans.fit_predict(pca_data)

df['Cluster'] = clusters

# Visualization
plt.figure(figsize=(10, 7))

plt.scatter(pca_data[:, 0],
            pca_data[:, 1],
            c=clusters,
            cmap='viridis')

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            c='red',
            marker='X')

plt.title("Customer Segmentation")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.show()

# Cluster Summary
print("\nCluster Summary:")
print(df.groupby('Cluster').mean())

# Save Output
df.to_csv("clustered_customers.csv",
          index=False)

print("\nOutput saved as clustered_customers.csv")