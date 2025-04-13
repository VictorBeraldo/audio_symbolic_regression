import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC, AmplitudeToDB
import os
import pickle
from tqdm import tqdm
import librosa
import numpy as np
import matplotlib.pyplot as plt
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def process_file(file, folder_path):
    """
    Carrega o áudio, extrai o espectrograma e retorna um dicionário
    com o nome do arquivo e o espectrograma (como numpy array).
    """
    filepath = os.path.join(folder_path, file)
    waveform = load_audio(filepath)
    spectrogram = extract_melspectrogram(waveform)
    return {'file': file, 'spectrogram': spectrogram.numpy()}

def process_files_chunk(file_chunk, folder_path):
    """
    Processa uma lista (chunk) de arquivos de forma sequencial.
    Retorna uma lista com os resultados.
    """
    results = []
    for file in file_chunk:
        results.append(process_file(file, folder_path))
    return results

def process_files_in_parallel(file_list, folder_path, desc, chunk_size=100, max_workers=None):
    """
    Processa a lista de arquivos em paralelo, agrupando-os em chunks para reduzir o overhead.
    
    Parâmetros:
      - file_list: lista de nomes de arquivos.
      - folder_path: pasta onde os arquivos estão localizados.
      - desc: descrição para o tqdm.
      - chunk_size: quantidade de arquivos por tarefa (default = 100).
      - max_workers: número máximo de threads a serem usados; se None, utiliza os núcleos disponíveis.
    
    Retorna uma lista com os resultados de todos os arquivos.
    """
    # Divide a file_list em chunks
    chunks = [file_list[i:i+chunk_size] for i in range(0, len(file_list), chunk_size)]
    total_files = len(file_list)
    results = []
    
    if max_workers is None:
        max_workers = os.cpu_count()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_files_chunk, chunk, folder_path): len(chunk) for chunk in chunks}
        pbar = tqdm(total=total_files, desc=desc, unit='file', ncols=100)
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                print(f"Erro no processamento de um chunk: {e}")
            pbar.update(futures[future])
        pbar.close()
    
    return results

def save_melspectrograms_in_parts_parallel(files, folder_path, base_save_path, num_parts=4, chunk_size=100, max_workers=None):
    """
    Divide a lista de arquivos (já preparada) em 'num_parts' partes e, para cada parte,
    reinicializa o pool de threads (para evitar acúmulo de recursos/travamentos) e salva os resultados.
    
    Parâmetros:
      - files: lista de nomes de arquivos a processar.
      - folder_path: pasta contendo os arquivos de áudio.
      - base_save_path: caminho base para salvar os arquivos PKL (por exemplo,
        '../data/processed/melspectrograms_ASVspoof2021_DF_eval_part00.pkl').
      - num_parts: número de partes em que os dados serão divididos (default = 4).
      - chunk_size: quantidade de arquivos processados por tarefa paralela (default = 100).
      - max_workers: número máximo de threads a usar; se None, utiliza os núcleos disponíveis.
    """
    # Cria o diretório de saída com base no caminho fornecido
    os.makedirs(os.path.dirname(base_save_path), exist_ok=True)
    parts = np.array_split(files, num_parts)
    
    for part_idx, file_array in enumerate(parts, start=1):
        file_list = list(file_array)
        print(f"Processando parte {part_idx} com {len(file_list)} arquivos...")
        
        # Cria um novo pool de threads para esta parte
        results = process_files_in_parallel(file_list, folder_path, desc=f"Parte {part_idx}",
                                              chunk_size=chunk_size, max_workers=max_workers)
        
        # Define o caminho de saída para a parte atual usando o base_save_path.
        # Exemplo: se base_save_path for '.../melspectrograms_ASVspoof2021_DF_eval_part00.pkl'
        # o arquivo de parte 1 será '.../melspectrograms_ASVspoof2021_DF_eval_part00_part1.pkl'
        base, ext = os.path.splitext(base_save_path)
        save_path_part = f"{base}_part{part_idx}{ext}"
        
        with open(save_path_part, 'wb') as f:
            pickle.dump(results, f)
        print(f"Salvos {len(results)} espectrogramas em {save_path_part}")
        
        # Libera memória
        del results
        gc.collect()

def save_melspectrograms_from_folder_parallel(folder_path, save_path, N=-1, num_parts=4, chunk_size=400, max_workers=None):
    """
    Função wrapper que lista os arquivos na pasta (filtrando por extensão),
    aplica o limite se N > 0 e chama a função que processa os dados em partes,
    reinicializando os pools de threads.
    
    Parâmetros:
      - folder_path: pasta contendo os arquivos de áudio (.wav, .flac, etc.)
      - save_path: caminho base para salvar os arquivos PKL. As partes serão salvas
                   com sufixos, por exemplo, '_part1', '_part2', etc.
      - N: número máximo de arquivos a processar; se N <= 0, usa todos.
      - num_parts: número de partes em que os dados serão divididos (default = 4).
      - chunk_size: quantidade de arquivos processados por tarefa paralela (default = 400).
      - max_workers: número máximo de threads a usar; se None, utiliza os núcleos disponíveis.
    """
    # Lista os arquivos na pasta filtrando por extensão
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.flac'))]
    if N > 0:
        files = files[:N]
    
    save_melspectrograms_in_parts_parallel(files, folder_path, save_path,
                                             num_parts=num_parts, chunk_size=chunk_size,
                                             max_workers=max_workers)



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

def extract_mfcc_features(waveform, sample_rate=16000, n_mfcc=13, n_fft=1024, hop_length=512, n_mels=40):
    """
    Extrai os coeficientes MFCC a partir de um waveform e calcula
    estatísticas globais (média, desvio padrão, mínimo, máximo e mediana) para cada coeficiente.
    Retorna um vetor de features com tamanho n_mfcc*5.
    
    Parâmetros:
      - waveform: tensor (ou similar) com o áudio carregado.
      - sample_rate: taxa de amostragem (default 16000)
      - n_mfcc: número de coeficientes MFCC (default 13)
      - n_fft: tamanho da FFT (default 1024)
      - hop_length: tamanho do hop (default 512)
      - n_mels: número de bandas mel para o cálculo dos MFCC (default 40)
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
    
    # Extrai os MFCCs; assume-se que o resultado possui shape (n_mfcc, tempo)
    mfcc = mfcc_transform(waveform).squeeze().numpy()
    
    # Estatísticas globais para cada coeficiente, calculadas ao longo do tempo (eixo=1)
    mean_features   = np.mean(mfcc, axis=1)
    std_features    = np.std(mfcc, axis=1)
    min_features    = np.min(mfcc, axis=1)
    max_features    = np.max(mfcc, axis=1)
    median_features = np.median(mfcc, axis=1)
    
    # Concatena todas as estatísticas em um único vetor
    mfcc_features = np.concatenate([mean_features, std_features, min_features, max_features, median_features])
    
    return mfcc_features

def save_mfcc_features_from_folder(folder_path, save_path='../data/processed/mfcc_features.pkl', N=-1):
    """
    Extrai e salva as features de MFCC para todos os áudios da pasta.
    
    Para cada áudio, extrai os MFCCs e calcula estatísticas globais (média, desvio, 
    mínimo, máximo e mediana) para cada coeficiente, gerando um vetor de tamanho n_mfcc*5,
    de forma similar à função de filterbanks.
    
    Parâmetros:
      - folder_path: pasta contendo os arquivos de áudio (.wav ou .flac)
      - save_path: caminho onde o arquivo pickle será salvo
      - N: número máximo de arquivos a processar; se N <= 0, processa todos.
    """
    # Lista arquivos com extensões desejadas
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.flac'))]
    if N > 0:
        files = files[:N]
    
    mfcc_features_list = []
    
    for file in tqdm(files, desc="Extracting MFCC Features"):
        filepath = os.path.join(folder_path, file)
        waveform = load_audio(filepath)
        # Extrai as features de MFCC (vetor de tamanho n_mfcc*5)
        features = extract_mfcc_features(waveform)
        
        mfcc_features_list.append({
            'file': file,
            'features': features
        })
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(mfcc_features_list, f)
    
    print(f"Saved {len(mfcc_features_list)} MFCC features to {save_path}")