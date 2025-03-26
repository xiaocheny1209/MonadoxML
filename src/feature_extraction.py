import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal.windows import hann


class FeatureExtraction:
    def __init__(self, preprocessed_data, freq_bands):
        self.preprocessed_data = preprocessed_data
        if freq_bands:
            self.default_freq_bands = freq_bands
        else:
            self.default_freq_bands = [
                (0.5, 4),
                (4, 8),
                (8, 12),
                (12, 30),
                (30, 45),
            ]  # default bands (delta, theta, alpha, beta, gamma)

    def _get_relative_psd(self, relative_energy_graph, sample_freq, stft_n=256):
        start_index = int(np.floor(self.default_freq_bands[0] / sample_freq * stft_n))
        end_index = int(np.floor(self.default_freq_bands[1] / sample_freq * stft_n))
        # print(start_index, end_index)
        psd = np.mean(
            relative_energy_graph[:, start_index - 1 : end_index] ** 2, axis=1
        )
        # print('psd:', psd.shape)
        return psd

    def extract_psd_feature(self, window_size, freq, stft_n=256):
        sample_freq = freq
        # Ptr operation
        if len(self.preprocessed_data.shape) > 2:
            self.preprocessed_data = np.squeeze(self.preprocessed_data)
        n_channels, n_samples = self.preprocessed_data.shape
        point_per_window = int(sample_freq * window_size)
        window_num = int(n_samples // point_per_window)
        psd_feature = np.zeros((window_num, len(self.default_freq_bands), n_channels))
        # print('psd feature shape:', psd_feature.shape)
        for window_index in range(window_num):
            start_index, end_index = (
                point_per_window * window_index,
                point_per_window * (window_index + 1),
            )
            window_data = self.preprocessed_data[:, start_index:end_index]
            hdata = window_data * hann(point_per_window)
            fft_data = np.fft.fft(hdata, n=stft_n)
            # print('fft_data shape:',fft_data.shape)
            energy_graph = np.abs(fft_data[:, 0 : int(stft_n / 2)])
            # print('energy_graph.shape:', energy_graph.shape)
            relative_energy_graph = energy_graph / np.sum(energy_graph)
            for band_index, band in enumerate(self.default_freq_bands):
                band_relative_psd = self._get_relative_psd(
                    relative_energy_graph, sample_freq, stft_n
                )
                psd_feature[window_index, band_index, :] = band_relative_psd
        return psd_feature

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
        # reshaped_features = np.array(features).reshape(1, -1) # reshape if the feature is 1D
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
