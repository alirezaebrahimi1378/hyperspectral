import os
import numpy as np
import rasterio as rio
import constants as C
from spectral.algorithms import rx
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv


def UINT8(Data):
    shape = Data.shape
    for i in range(shape[2]):
        data = Data[:, :, i]
        data = data / data.max()
        data = 255 * data
        Data[:, :, i] = data.astype(np.uint8)
    return Data


def normalize(data):
    min_val = data.min()
    max_val = data.max()
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def plot_results(rx_global, rx_local, combined):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(rx_global, cmap='hot', interpolation='nearest')
    plt.title('Global RX Anomaly Detection')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(rx_local, cmap='hot', interpolation='nearest')
    plt.title('Local RX Anomaly Detection')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(combined, cmap='hot', interpolation='nearest')
    plt.title('Combined KMeans & Mahalanobis')
    plt.colorbar()

    plt.show()


def plot_hist(rx_global, rx_local, combined):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.hist(rx_global.ravel(), bins=100, color='blue', log=True)
    plt.title('Histogram of Global RX Anomaly Detection')
    plt.xlabel('RX value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(rx_local.ravel(), bins=100, color='green', log=True)
    plt.title('Histogram of Local RX Anomaly Detection')
    plt.xlabel('RX value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(combined.ravel(), bins=100, color='red', log=True)
    plt.title('Histogram of Combined KMeans & Mahalanobis')
    plt.xlabel('Anomaly score')
    plt.ylabel('Frequency')

    plt.show()


class processor:

    def __init__(self):
        self.root = C.root_path  # _____ root path of project
        self.file_path = C.image_path  # _____ path to image
        self.image_file = rio.open(self.file_path)
        data = self.image_file.read()
        # _____ Rearrange to (rows, cols, bands)
        data = np.moveaxis(data, 0, -1)
        self.data = data.astype(np.float32)
        self.window_size = C.window_size

    def RX_global(self):
        rx_global = rx(self.data)
        rx_global_normalized = normalize(rx_global)
        return rx_global_normalized

    def RX_local(self):
        window_size = self.window_size
        rows, cols, bands = self.data.shape
        rx_local_result = np.zeros((rows, cols), dtype=np.float32)

        pad_size = window_size // 2
        padded_data = np.pad(
            self.data, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

        for i in range(rows):
            for j in range(cols):
                window = padded_data[i:i + window_size, j:j + window_size, :]
                rx_local_result[i, j] = rx(window)[pad_size, pad_size]
        rx_local_normalized = normalize(rx_local_result)
        return rx_local_normalized

    def kmeans_mahalanobis(self):
        rows, cols, bands = self.data.shape
        data_reshaped = self.data.reshape(-1, bands)

        # Apply KMeans clustering
        n_clusters = 5  # Number of clusters, adjust as needed
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=42).fit(data_reshaped)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Compute Mahalanobis distance for each pixel from its cluster centroid
        mahalanobis_distances = np.zeros(
            data_reshaped.shape[0], dtype=np.float32)

        for cluster in range(n_clusters):
            cluster_points = data_reshaped[labels == cluster]
            if cluster_points.shape[0] == 0:
                continue  # Skip empty clusters

            cluster_mean = centroids[cluster]
            cov_matrix = np.cov(cluster_points, rowvar=False)

            # Regularize covariance matrix if it's singular
            try:
                inv_cov_matrix = inv(cov_matrix)
            except LinAlgError:
                inv_cov_matrix = inv(
                    cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-10)

            cluster_indices = np.where(labels == cluster)[0]
            for i, index in enumerate(cluster_indices):
                distance = mahalanobis(
                    cluster_points[i], cluster_mean, inv_cov_matrix)
                mahalanobis_distances[index] = distance

        # Reshape back to the original image dimensions
        mahalanobis_image = mahalanobis_distances.reshape(rows, cols)

        # Handle any potential NaN values
        mahalanobis_image = np.nan_to_num(
            mahalanobis_image, nan=0.0, posinf=1.0, neginf=0.0)

        mahalanobis_normalized = normalize(mahalanobis_image)
        return mahalanobis_normalized

    def main(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            rx_global = self.RX_global()
            rx_local = self.RX_local()
            combined = self.kmeans_mahalanobis()
            plot_results(rx_global=rx_global, rx_local=rx_local,
                         combined=combined)
            plot_hist(rx_global=rx_global, rx_local=rx_local,
                      combined=combined)


if __name__ == '__main__':
    anomaly_detector = processor()
    anomaly_detector.main()
