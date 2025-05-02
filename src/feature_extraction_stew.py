import numpy as np
from scipy.signal.windows import hann

class FeatureExtractionSTEW:
    def __init__(self, preprocessed_data, freq_bands):
        self.preprocessed_data = preprocessed_data  # shape: (channels, timepoints)
        self.freq_bands = freq_bands

    def _get_relative_psd(self, energy_graph, sample_freq, band, stft_n=256):
        start_freq, end_freq = band
        start_index = int(np.floor(start_freq / sample_freq * stft_n))
        end_index = int(np.floor(end_freq / sample_freq * stft_n))
        psd = np.mean(energy_graph[:, start_index:end_index] ** 2, axis=1)
        return psd

    def extract_psd_feature(self, window_size, sample_freq, stft_n=256):
        if len(self.preprocessed_data.shape) > 2:
            self.preprocessed_data = np.squeeze(self.preprocessed_data)

        n_channels, n_samples = self.preprocessed_data.shape
        point_per_window = int(sample_freq * window_size)
        window_num = int(n_samples // point_per_window)
        psd_feature = np.zeros((window_num, len(self.freq_bands), n_channels))

        for window_index in range(window_num):
            start_index = point_per_window * window_index
            end_index = point_per_window * (window_index + 1)
            window_data = self.preprocessed_data[:, start_index:end_index]
            hdata = window_data * hann(point_per_window)
            fft_data = np.fft.fft(hdata, n=stft_n)
            energy_graph = np.abs(fft_data[:, 0 : int(stft_n / 2)])
            relative_energy_graph = energy_graph / np.sum(energy_graph)

            for band_index, band in enumerate(self.freq_bands):
                band_relative_psd = self._get_relative_psd(
                    relative_energy_graph, sample_freq, band, stft_n
                )
                psd_feature[window_index, band_index, :] = band_relative_psd

        return psd_feature
