from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def fit_pca(dim, data):
    pca = PCA(n_components=dim, whiten=True, svd_solver='full')
    return pca.fit(data)


def compute_pca_features(visual_features, pca_model):
    print('Computing PCA!')

    # Standardizing the features
    # x = StandardScaler().fit_transform(visual_features)

    principal_components = pca_model.transform(visual_features)
    principal_components = normalize(principal_components)

    print('input shape: {} -----> {}'.format(visual_features.shape, principal_components.shape))

    return principal_components


def fit_kmeans(num_clusters, features):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=None)
    print('kmeans fitted!')
    return kmeans.fit(features)
