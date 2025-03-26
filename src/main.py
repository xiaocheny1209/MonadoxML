import os
import argparse
import copy
from re import I, sub
import pandas as pd
import numpy as np
import mne
import pickle as pkl
import hdf5storage
from sklearn.model_selection import KFold
from tqdm import tqdm
import matpltlib.pyplot as plt
from glob import glob
import scipy.io as sio
import time
import joblib
from preprocessing.FACED import (
    FACEDPreprocessing,
    read_data,
    data_concat,
    eeg_save,
    channel_modify,
    load_srt_de,
    load_srt_pretrainFeat,
)
from feature_extraction import FeatureExtraction
from models.svm import train_svm, evaluate_svm


parser = argparse.ArgumentParser(description="Data preprocessing")
parser.add_argument(
    "--clisa-or-not",
    default="no",
    type=str,
    help="implement the clisa preprocessing step, yes or no",
)
args = parser.parse_args()
clisa_or_not = args.clisa_or_not


if __name__ == "__main__":
    ## ============ Data Preprocessing ===============
    foldPaths = "data/FACED/Recording_info.csv"
    data_dir = "data/FACED"
    save_dir = "data/FACED/Processed_data"
    if clisa_or_not == "yes":
        clisa_save_dir = (
            "data/FACED/Validation/Classification_validation/Clisa_analysis/Clisa_data"
        )
        print("Also do the preprocess for CLISA.")

    pd_data = pd.read_csv(foldPaths, low_memory=False)
    sub_info = pd_data["sub"]
    sub_batch = pd_data["Cohort"]
    print(sub_info)

    # Read the data
    for idx, sub in tqdm(enumerate(sub_info)):
        sub_path = os.path.join(data_dir, sub)
        print("Current processing subject:", sub)
        trigger, onset, duration, rawdata, [unit, impedance, experiments] = read_data(
            sub_path
        )
        # read_the_remark_data
        remark_data = hdf5storage.loadmat(os.path.join(sub_path, "After_remarks.mat"))[
            "After_remark"
        ]
        vids = np.squeeze(remark_data["vid"])
        frequency = rawdata.info["sfreq"]
        events = np.transpose(
            np.vstack((np.vstack((onset, duration)), trigger))
        ).astype(int)
        # The first batch and the second batch have different unit (uV and V)
        original_raw = rawdata.copy()

        # Epochs cutting
        cut_seconds = -30
        event_id = 102
        epochs = mne.Epochs(
            original_raw,
            events,
            event_id=event_id,
            tmin=cut_seconds,
            tmax=0,
            preload=True,
        )

        # Trigger segmentation
        video_trigger_index = np.where((trigger != 0) & (trigger < 29))[0]

        eeg_Data_saved = None
        if clisa_or_not == "yes":
            eeg_clisa = None

        # print(len(video_trigger_index))
        for index, pos in enumerate(video_trigger_index):
            print("Processing video trigger", index)
            video = trigger[pos]

            # The final 30s trial
            processed_epoch_ = FACEDPreprocessing(epochs[index])
            processed_epoch_.down_sample(250)
            # processed_epoch_.band_pass_filter(0.5, 47)
            processed_epoch_.band_pass_filter(0.05, 47)
            processed_epoch_.bad_channels_interpolate(thresh1=3, proportion=0.3)
            processed_epoch_.eeg_ica()

            if clisa_or_not == "yes":
                processed_epoch_clisa = copy.deepcopy(processed_epoch_)
                processed_epoch_clisa.band_pass_filter(4, 47)
                processed_epoch_clisa.average_ref()
                eeg_clisa = data_concat(
                    eeg_clisa, processed_epoch_clisa.raw.get_data(), video
                )

            processed_epoch_.average_ref()

            # Save the data
            eeg_Data_saved = data_concat(
                eeg_Data_saved, processed_epoch_.raw.get_data(), video
            )

        # Modify the channels
        if int(sub_batch[idx] == 1):
            batch = 1
            eeg_Data_saved = channel_modify(eeg_Data_saved, batch)
        elif int(sub_batch[idx] == 2):
            batch = 2
            eeg_Data_saved = channel_modify(eeg_Data_saved, batch)
        if clisa_or_not == "yes":
            eeg_clisa = channel_modify(eeg_clisa, batch)

        # Saved as pkl
        eeg_save(sub, eeg_Data_saved, save_dir)
        if clisa_or_not == "yes":
            eeg_save(sub, eeg_clisa, clisa_save_dir)

    ## ============ Feature Extraction ===============
    n_vids = 28
    freq = 250
    nsec = 30
    nchn = 32
    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]

    datapath = "data/FACED/Processed_data"
    PSD_savepath = "data/FACED/EEG_Features/PSD"
    if not os.path.exists(PSD_savepath):
        os.makedirs(PSD_savepath)
    DE_savepath = "data/FACED/EEG_Features/DE"
    if not os.path.exists(DE_savepath):
        os.makedirs(DE_savepath)

    subs_psd = np.zeros((n_vids, nchn, nsec, len(freq_bands)))
    subs_de = np.zeros((n_vids, nchn, nsec, len(freq_bands)))  # shape (28,32,30,5)

    for idx, sub in tqdm(enumerate(sub_info)):
        f = open(os.path.join(datapath, sub), "rb")
        data_sub = pkl.load(f)
        # print(data_sub.shape) # (28, 32, 7500)

        # PSD features
        for video in range(0, n_vids):
            # (30, 5, 32)
            feature_extractor = FeatureExtraction(data_sub[video, :, :], freq_bands)
            psd_data = feature_extractor.extract_psd_feature(1, freq)
            # (32, 30, 5)
            psd_data = np.transpose(psd_data, (2, 0, 1))
            subs_psd[video, :, :, :] = psd_data

        # save the PSD and DE features
        f = open(PSD_savepath + "/" + str(sub) + ".pkl", "wb")
        pkl.dump(subs_psd, f)
        f.close()

        # DE features
        for i in range(len(freq_bands)):
            for video in range(0, n_vids):
                data_video = data_sub[video, :, :]
                low_freq = freq_bands[i][0]
                high_freq = freq_bands[i][1]
                # band pass filter for the specific data band
                # shape (32, 7500)
                data_video_filt = mne.filter.filter_data(
                    data_video, freq, l_freq=low_freq, h_freq=high_freq
                )
                # shape (32, 30, 250)
                data_video_filt = data_video_filt.reshape(nchn, -1, freq)
                print("data filtered :", data_video_filt.shape)
                de_one = 0.5 * np.log(
                    2 * np.pi * np.exp(1) * (np.var(data_video_filt, 2))
                )
                print(de_one.shape)
                # n_subs, video, channel,  second, frequency
                subs_de[video, :, :, i] = de_one

        g = open(DE_savepath + "/" + str(sub) + ".pkl", "wb")
        pkl.dump(subs_de, g)
        g.close()

    ## ============ Model Training ===============
    feat_dir = "data/FACED/EEG_Features/DE"
    all_features = []
    all_labels = []

    for sub in sub_info[:1]:
        with open(f"{feat_dir}/{sub}.pkl.pkl", "rb") as f:
            features = pkl.load(f)  # Shape (28, 30, 32, 5)
            labels = np.array(
                [0] * 15 + [1] * 15
            )  # Example labels for 30 time steps per subject
            # data, labels, n_samples = load_srt_pretrainFeat(
            #     features, channel_norm, timeLen, timeStep
            # )
            features_reshaped = features.reshape(
                -1, 32, 5
            )  # Shape becomes (28*30, 32, 5)
            labels_reshaped = np.tile(labels, 28)  # Fatten the labels

            # print(f"Reshaped features shape: {features_reshaped.shape}")
            # print(f"Reshaped labels shape: {labels_reshaped.shape}")

            all_features.append(features_reshaped)
            all_labels.append(labels_reshaped)

    # Convert all_features and all_labels to numpy arrays
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Perform 10-fold cross-validation on the reshaped data
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    best_accuracy = 0  # Variable to track the best validation accuracy
    best_fold = -1  # Variable to track the best fold number

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(all_labels)))):
        print(f"\n  Fold {fold + 1}")

        # Get the training and validation data for this fold
        X_train = all_features[train_idx]
        X_val = all_features[val_idx]
        y_train = all_labels[train_idx]
        y_val = all_labels[val_idx]

        # print(f"    Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        # print(
        #     f"    Train labels: {np.bincount(y_train)}, Val labels: {np.bincount(y_val)}"
        # )

        # Flatten the training data to fit the model
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        # Train the model
        svm = train_svm(X_train_flat, y_train)

        # Make predictions on the validation set
        accuracy = evaluate_svm(svm, X_val_flat, y_val)
        print(f"    Validation Accuracy: {accuracy:.4f}")

        # Check if the current fold has the best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_fold = fold + 1  # Store the best fold (1-indexed)

    # After cross-validation, print the best fold and its accuracy
    print(f"\nBest fold: {best_fold} with Validation Accuracy: {best_accuracy:.4f}")
