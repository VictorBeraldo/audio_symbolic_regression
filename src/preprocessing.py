import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC, AmplitudeToDB
import os
import pickle
from tqdm import tqdm
import librosa
import numpy as np
import matplotlib.pyplot as plt

def load_audio(filepath, target_sample_rate=16000):
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".flac" or ext == ".wav":
        waveform, sample_rate = torchaudio.load(filepath)
    else:
        raise ValueError(f"Unsupported audio format: {ext}")
    
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform

def extract_melspectrogram(waveform, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
    mel_spec_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
        norm="slaney"
    )
    mel_spectrogram = mel_spec_transform(waveform)
    mel_spectrogram_db = AmplitudeToDB()(mel_spectrogram)
    return mel_spectrogram_db

def save_melspectrograms_from_folder(folder_path, save_path='../data/processed/melspectrograms.pkl', N=-1):
    """
    Extrai e salva os Mel Spectrograms em dB para todos os áudios da pasta.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav') or f.endswith('.flac')]
    if N > 0:
        files = files[:N]

    spectrograms_list = []

    for file in tqdm(files, desc="Extracting Mel Spectrograms"):
        filepath = os.path.join(folder_path, file)
        waveform = load_audio(filepath)
        spectrogram = extract_melspectrogram(waveform)
        
        spectrograms_list.append({
            'file': file,
            'spectrogram': spectrogram.numpy()
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(spectrograms_list, f)

    print(f"Saved {len(spectrograms_list)} Mel Spectrograms to {save_path}")

def extract_filterbank_features(waveform, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
    """
    Extrai atributos a partir dos filterbanks (coeficientes do Mel Spectrogram antes da conversão para dB).
    Retorna um vetor de atributos com n_mels*5 estatísticas.
    """
    mel_spec_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
        norm="slaney"
    )
    
    mel_spectrogram = mel_spec_transform(waveform).squeeze().numpy()  # Remove batch e converte para numpy
    
    # Estatísticas globais sobre os filterbanks
    mean_features = np.mean(mel_spectrogram, axis=1)  # Média por filtro ao longo do tempo
    std_features = np.std(mel_spectrogram, axis=1)    # Desvio padrão por filtro
    min_features = np.min(mel_spectrogram, axis=1)    # Mínimo por filtro
    max_features = np.max(mel_spectrogram, axis=1)    # Máximo por filtro
    median_features = np.median(mel_spectrogram, axis=1)  # Mediana por filtro
    
    # Concatenar todas as features em um único vetor
    filterbank_features = np.concatenate([mean_features, std_features, min_features, max_features, median_features])

    return filterbank_features


def save_filterbank_features_from_folder(folder_path, save_path='../data/processed/filterbank_features.pkl', N=-1):
    """
    Extrai e salva features de filterbanks para todos os áudios da pasta.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav') or f.endswith('.flac')]
    if N > 0:
        files = files[:N]

    features_list = []

    for file in tqdm(files, desc="Extracting Filterbank Features"):
        filepath = os.path.join(folder_path, file)
        waveform = load_audio(filepath)
        features = extract_filterbank_features(waveform)
        
        features_list.append({
            'file': file,
            'features': features
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(features_list, f)

    print(f"Saved {len(features_list)} filterbank features to {save_path}")


def extract_mfcc(waveform, sample_rate=16000, n_mfcc=13, n_fft=1024, hop_length=512, n_mels=40):
    """
    Extrai coeficientes MFCC a partir de um waveform, com parâmetros explícitos.
    """
    melkwargs = {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_mels': n_mels,
        'power': 2.0,
        'norm': 'slaney'
    }

    mfcc_transform = MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs=melkwargs
    )
    mfcc = mfcc_transform(waveform)
    return mfcc

def save_mfccs_from_folder(folder_path, save_path='../data/processed/mfccs.pkl', N=-1):
    """
    Extrai e salva MFCCs para todos os áudios da pasta.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav') or f.endswith('.flac')]
    if N > 0:
        files = files[:N]

    mfccs_list = []

    for file in tqdm(files, desc="Extracting MFCCs"):
        filepath = os.path.join(folder_path, file)
        waveform = load_audio(filepath)
        mfcc = extract_mfcc(waveform)
        
        mfccs_list.append({
            'file': file,
            'mfcc': mfcc.squeeze().numpy()  # Remove batch dimension
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(mfccs_list, f)

    print(f"Saved {len(mfccs_list)} MFCCs to {save_path}")