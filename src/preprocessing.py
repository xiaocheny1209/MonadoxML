import mne
import matplotlib.pyplot as plt
import numpy as np


class Preprocessing:
    def __init__(self, data_path):
        self.data = mne.io.read_raw_bdf(data_path, preload=True)
        print(self.data.info)

    def filter_data(self, l_freq, h_freq):
        """Apply a bandpass filter to remove unwanted frequencies (e.g., noise, artifacts)."""
        filtered_data = self.data.copy().filter(l_freq=l_freq, h_freq=h_freq)
        return filtered_data

    def remove_bad_channels(self, bad_channels=None):
        """Remove bad channels (e.g., noisy or broken channels) from raw EEG data."""
        if bad_channels:
            self.data.info["bads"] = bad_channels
            self.data = self.data.drop_channels(bad_channels)
        return self.data

    def downsample_data(self, new_sfreq):
        """Downsample the EEG data to a lower sampling frequency."""
        self.data = self.data.resample(new_sfreq)
        return self.data

    def apply_ica(self, n_components=20):
        """Apply ICA to remove artifacts (e.g., eye blinks)."""
        ica = mne.preprocessing.ICA(
            n_components=n_components, random_state=97, max_iter=800
        )
        ica.fit(self.data)
        # Automatically find the components to exclude (e.g., eye movement)
        ica.exclude = [0, 1]  # example component exclusion
        self.data = ica.apply(self.data)
        return self.data

    # below functions are not yet used in the pipeline
    def describe(self):
        # inspect data
        # data, times = self.data[:, :]  # EEG signal values and the corresponding time vector
        # Plot the raw data
        self.data.plot()

        # Plot the PSD
        # plt.plot(freqs, psd.mean(axis=0))
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Power Spectral Density')
        # plt.show()

    def epoch(self):
        # epoching data
        # Define events based on triggers (e.g., stimuli in the data)
        events, _ = mne.find_events(self.data, stim_channel="STI 014")
        # Create epochs based on the events
        epochs = mne.Epochs(
            self.data,
            events,
            event_id=1,
            tmin=-0.2,
            tmax=0.5,
            baseline=(None, 0),
            detrend=1,
        )

    def save(self, path):  # "data/FACED/processed_data.fif"
        self.data.save(path, overwrite=True)
