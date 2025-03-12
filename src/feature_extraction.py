import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FeatureExtraction:
    def __init__(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data
        self.default_freq_bands = [
            (0.5, 4),
            (4, 8),
            (8, 12),
            (12, 30),
            (30, 45),
        ]  # default bands (delta, theta, alpha, beta, gamma)

    def extract_psd_feature(self, window_size=1, freq_bands=None):
        """Extract Power Spectral Density (PSD) features for specific frequency bands."""
        freq_bands = self.default_freq_bands if not freq_bands else freq_bands
        psd_features = []
        for start, stop in freq_bands:
            # Compute the Power Spectral Density (PSD) using Welch's method
            f, Pxx = welch(
                self.preprocessed_data,
                fs=self.preprocessed_data.info["sfreq"],
                nperseg=window_size,
            )
            # Extract PSD values for the given frequency band
            band_psd = np.mean(Pxx[(f >= start) & (f <= stop)])
            psd_features.append(band_psd)
        return np.array(psd_features)

    def extract_time_domain_features(self):
        """
        Extract simple time-domain features like mean, variance, skewness, and kurtosis.
        """
        data = self.preprocessed_data.get_data()
        features = []
        for ch_data in data:
            # Compute mean, variance, skewness, and kurtosis for each channel
            mean_val = np.mean(ch_data)
            var_val = np.var(ch_data)
            skewness_val = skew(ch_data)
            kurtosis_val = kurtosis(ch_data)
            features.extend([mean_val, var_val, skewness_val, kurtosis_val])
        return np.array(features)

    def extract_bandpower_features(self, freq_bands=None):
        """
        Extract bandpower features for different frequency bands (alpha, beta, etc.).
        """
        freq_bands = self.default_freq_bands if not freq_bands else freq_bands
        bandpower_features = []
        for start, stop in freq_bands:
            # Compute power in each frequency band using Welch's method
            f, Pxx = welch(
                self.preprocessed_data, fs=self.preprocessed_data.info["sfreq"]
            )
            band_power = np.sum(Pxx[(f >= start) & (f <= stop)])
            bandpower_features.append(band_power)
        return np.array(bandpower_features)

    def extract_all_features(self, window_size=1, freq_bands=None):
        psd_features = self.extract_psd_feature(window_size, freq_bands)
        time_features = self.extract_time_domain_features()
        freq_features = self.extract_bandpower_features(freq_bands)
        return np.concatenate([psd_features, time_features, freq_features])

    @staticmethod
    def normalize_features(features):
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    @staticmethod
    def apply_pca(features, n_components=0.95):
        """
        n_components can be number of dimensions (e.g., 10) or percentage of variance (e.g., 0.95).
        """
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features)
        print(
            f"Original features: {features.shape[1]}, Reduced to: {reduced_features.shape[1]}"
        )
        return reduced_features

    @staticmethod
    def select_best_features(features, labels, k=10):
        selector = SelectKBest(score_func=f_classif, k=k)
        best_features = selector.fit_transform(features, labels)
        print(f"Selected top {k} features from {features.shape[1]}")
        return best_features
