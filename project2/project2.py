import os
import rasterio as rio
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import skdim
import warnings
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random
import constants as C


# _____ gets a multi-band image and returns it as uint8 image
def UINT8(Data):
    shape = Data.shape
    for i in range(shape[2]):
        data = Data[:, :, i]
        data = data / data.max()
        data = 255 * data
        Data[:, :, i] = data.astype(np.uint8)
    return Data


# _____ used to plot reduced images after applying feature extraction
def plot_reduced_img(reduced_data, method, shape):
    fig = plt.figure(figsize=(14, 10))
    for i in range(reduced_data.shape[1]):
        if method == "lda":
            plt.subplot(1, 3, i + 1)
        else:
            plt.subplot(5, 5, i + 1)
        img_band = reduced_data[:, i]
        img_band = img_band.reshape(shape[1], shape[2])
        plt.imshow(img_band, cmap="gray")
        plt.title(f"{method}_{i + 1}")
        plt.axis("off")  # Turn off the axis
    plt.show()


# _____ used for ploting 3D scatter plot from first three bands after feature extraction
def scatter_plot(reduced_image, label, method):
    roi_pixels = reduced_image[label > 0]
    roi_labels = label[label > 0]
    band1 = roi_pixels[:, 0]
    band2 = roi_pixels[:, 1]
    band3 = roi_pixels[:, 2]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"scatter plot for method {method}")
    colors = {1: "g", 2: "r", 3: "b", 4: "y"}
    point_colors = [colors[label] for label in roi_labels]
    scatter = ax.scatter(band1, band2, band3, c=point_colors)

    ax.set_xlabel("Band 1")
    ax.set_ylabel("Band 2")
    ax.set_zlabel("Band 3")
    classes = ["tree", "urban", "water body", "grass"]
    for class_value in np.unique(roi_labels):

        ax.scatter(
            [], [], [
            ], c=colors[class_value], label=f"{classes[int(class_value)-1]}"
        )
    ax.legend()
    plt.show()

# _____ ploting accuracy for different methods of feature extraction at once


def plot_accuracies(accuracies):
    methods = list(accuracies.keys())
    values = list(accuracies.values())

    plt.figure(figsize=(10, 6))
    plt.bar(methods, values, color='skyblue')
    plt.xlabel('Feature Extraction Method')
    plt.ylabel('Accuracy')
    plt.title('KNN Classification Accuracy for Different Feature Extraction Methods')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.show()


class Processor:
    def __init__(self):
        self.root = os.getcwd()  # _____ root path of project
        self.file_path = os.path.join(
            self.root, "data/data.tif")  # _____ path to image
        self.label_path = os.path.join(
            self.root, "data/label.tif"
        )  # _____ path to label
        self.image_file = rio.open(self.file_path)
        self.label_file = rio.open(self.label_path)
        self.label = self.label_file.read()
        self.image = self.image_file.read()
        self.shape = self.image.shape

    # _____ used to calculate intrinsic dimension and make data square from 3d data
    def ins_dim(self, calc_dim=C.calc_dim):
        image = self.image
        data = image.transpose(1, 2, 0)
        data_uint8 = UINT8(data)
        image = np.zeros((data.shape[0] * data.shape[1], 1))

        for i in range(data.shape[2]):
            band = data_uint8[:, :, i]
            band = band.flatten()
            band = band.reshape((-1, 1))
            image = np.append(image, band, axis=1)

        self.flattened_img = image[:, 1:].astype(np.uint8)
        self.flattened_lbl = self.label.flatten()

        if calc_dim:
            lpca = skdim.id.lPCA().fit_pw(self.flattened_img, n_neighbors=100, n_jobs=1)
            print(np.mean(lpca.dimension_pw_))

    # _____ applying PCA method for feature extraction
    def PCA(self):
        pca = PCA(n_components=C.n_components)
        pca_img = pca.fit_transform(self.flattened_img)
        shape = self.shape
        plot_reduced_img(reduced_data=pca_img, method="pca", shape=self.shape)
        scatter_plot(reduced_image=pca_img,
                     label=self.flattened_lbl, method="PCA")
        return pca_img

    # _____ applying incremental PCA for feature extraction
    def IPCA(self):
        n_components = C.n_components
        ipca = IncrementalPCA(n_components=n_components, batch_size=1000)
        for batch in np.array_split(self.flattened_img, 80):
            ipca.partial_fit(batch)
        ipca_img = ipca.transform(self.flattened_img)
        plot_reduced_img(reduced_data=ipca_img,
                         method="ipca", shape=self.shape)
        scatter_plot(
            reduced_image=ipca_img, label=self.flattened_lbl, method="Incremental PCA"
        )
        return ipca_img

    # _____ applying kernel PCA for feature extraction
    def KPCA(self):
        n_components = C.n_components
        kpca = KernelPCA(kernel=C.kernel_type, n_components=n_components)
        subset_size = 5000
        # _____ because of processing limits , parameters are calculated for a sample of data
        subset = shuffle(self.flattened_img, random_state=100)[:subset_size]
        kpca_subset = kpca.fit(subset)
        # _____ parameters from sample are applyed to whole image
        kpca_img = kpca_subset.transform(self.flattened_img)
        plot_reduced_img(reduced_data=kpca_img,
                         method="kpca", shape=self.shape)
        scatter_plot(
            reduced_image=kpca_img, label=self.flattened_lbl, method="Kernel PCA"
        )
        return kpca_img

    # _____ applying method of isomap for dimension reduction
    def ISOMAP(self):
        n_components = 25
        isomap = Isomap(n_components=n_components, n_neighbors=5)
        subset_size = 5000
        # _____ again because of processing limits need to calculate parameters for a sebset of data
        subset = shuffle(self.flattened_img, random_state=42)[:subset_size]
        isomap_subset = isomap.fit(subset)
        # _____ applying subset parammeters for entire image
        isomap_img = isomap_subset.transform(self.flattened_img)
        plot_reduced_img(reduced_data=isomap_img,
                         method="isomap", shape=self.shape)
        scatter_plot(
            reduced_image=isomap_img, label=self.flattened_lbl, method="ISOMAP"
        )
        return isomap_img

    # _____ applying LDA method for feature extraction
    def LDA(self):
        # _____ keeping pixels that have value in ground truth
        roi_pixels = self.flattened_img[self.flattened_lbl > 0]
        roi_labels = self.flattened_lbl[self.flattened_lbl > 0]

        lda = LinearDiscriminantAnalysis(
            n_components=min(C.n_components, C.n_lda_classes - 1))

        # _____ in lda method number of components cant be more than number of classes - 1
        lda.fit(roi_pixels, roi_labels)
        lda_img = lda.transform(self.flattened_img)
        plot_reduced_img(reduced_data=lda_img, method="lda", shape=self.shape)

        scatter_plot(reduced_image=lda_img,
                     label=self.flattened_lbl, method="LDA")
        return lda_img

    # _____ using K nearest neighbour to classify pixels and calculate accuracy for reduced image
    def KNN(self, reduced_image):
        roi_pixels = reduced_image[self.flattened_lbl > 0]
        roi_labels = self.flattened_lbl[self.flattened_lbl > 0]
        train, test, train_gt, test_gt = train_test_split(
            roi_pixels, roi_labels, test_size=0.1, random_state=42
        )
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train, train_gt)
        test_pred = knn.predict(test)
        accuracy = accuracy_score(test_gt, test_pred)
        return accuracy

    def main(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            accuracy = {}
            self.ins_dim(calc_dim=False)
            reduced_img_pca = self.PCA()
            reduced_img_ipca = self.IPCA()
            reduced_img_kpca = self.KPCA()
            reduced_img_iso = self.ISOMAP()
            reduced_img_lda = self.LDA()

            accuracy['PCA'] = self.KNN(reduced_image=reduced_img_pca)
            accuracy['iPCA'] = self.KNN(reduced_image=reduced_img_ipca)
            accuracy['kPCA'] = self.KNN(reduced_image=reduced_img_kpca)
            accuracy['isomap'] = self.KNN(reduced_image=reduced_img_iso)
            accuracy['LDA'] = self.KNN(reduced_image=reduced_img_lda)

            plot_accuracies(accuracies=accuracy)


if __name__ == '__main__':
    change_process = Processor()
    change_process.main()
