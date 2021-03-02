"""Transforms on the short time fourier transforms of wav samples."""

import random

import numpy as np
import librosa
import torch
import torchaudio
from torch.utils.data import Dataset

from .transforms_wav import should_apply_transform

class ToSTFT(object):
    """Applies on an audio the short time fourier transform."""

    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        data['n_fft'] = self.n_fft
        data['hop_length'] = self.hop_length
        data['stft'] = librosa.stft(samples, n_fft=self.n_fft, hop_length=self.hop_length)
        data['stft_shape'] = data['stft'].shape
        return data

class StretchAudioOnSTFT(object):
    """Stretches an audio on the frequency domain."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        stft = data['stft']
        sample_rate = data['sample_rate']
        hop_length = data['hop_length']
        scale = random.uniform(-self.max_scale, self.max_scale)
        stft_stretch = librosa.core.phase_vocoder(stft, 1+scale, hop_length=hop_length)
        data['stft'] = stft_stretch
        return data

class TimeshiftAudioOnSTFT(object):
    """A simple timeshift on the frequency domain without multiplying with exp."""

    def __init__(self, max_shift=8):
        self.max_shift = max_shift

    def __call__(self, data):
        if not should_apply_transform():
            return data

        stft = data['stft']
        shift = random.randint(-self.max_shift, self.max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        stft = np.pad(stft, ((0, 0), (a, b)), "constant")
        if a == 0:
            stft = stft[:,b:]
        else:
            stft = stft[:,0:-a]
        data['stft'] = stft
        return data

class AddBackgroundNoiseOnSTFT(Dataset):
    """Adds a random background noise on the frequency domain."""

    def __init__(self, bg_dataset, max_percentage=0.45):
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage

    def __call__(self, data):
        if not should_apply_transform():
            return data

        noise = random.choice(self.bg_dataset)['stft']
        percentage = random.uniform(0, self.max_percentage)
        data['stft'] = data['stft'] * (1 - percentage) + noise * percentage
        return data

class FixSTFTDimension(object):
    """Either pads or truncates in the time axis on the frequency domain, applied after stretching, time shifting etc."""

    def __call__(self, data):
        stft = data['stft']
        t_len = stft.shape[1]
        orig_t_len = data['stft_shape'][1]
        if t_len > orig_t_len:
            stft = stft[:,0:orig_t_len]
        elif t_len < orig_t_len:
            stft = np.pad(stft, ((0, 0), (0, orig_t_len-t_len)), "constant")

        data['stft'] = stft
        return data

class ToMelSpectrogramFromSTFT(object):
    """Creates the mel spectrogram from the short time fourier transform of a file. The result is a 32x32 matrix."""

    def __init__(self, n_mels=32):
        self.n_mels = n_mels

    def __call__(self, data):
        stft = data['stft']
        sample_rate = data['sample_rate']
        n_fft = data['n_fft']
        mel_basis = librosa.filters.mel(sample_rate, n_fft, self.n_mels)
        s = np.dot(mel_basis, np.abs(stft)**2.0)
        data['mel_spectrogram'] = librosa.power_to_db(s, ref=np.max)
        return data

class DeleteSTFT(object):
    """Pytorch doesn't like complex numbers, use this transform to remove STFT after computing the mel spectrogram."""

    def __call__(self, data):
        del data['stft']
        return data

class AudioFromSTFT(object):
    """Inverse short time fourier transform."""

    def __call__(self, data):
        stft = data['stft']
        data['istft_samples'] = librosa.core.istft(stft, dtype=data['samples'].dtype)
        return data



class RandomPitchShift(object):
    def __init__(self, sample_rate=22050, pitch_shift=(-1.0, 1.0)):
        if isinstance(pitch_shift, (tuple, list)):
            self.min_pitch_shift = pitch_shift[0]
            self.max_pitch_shift = pitch_shift[1]
        else:
            self.min_pitch_shift = -pitch_shift
            self.max_pitch_shift = pitch_shift
        self.sample_rate=sample_rate

    def __call__(self, waveform):
        waveform = waveform.numpy()
        pitch_shift = random.uniform(self.min_pitch_shift, self.max_pitch_shift)
        waveform = librosa.effects.pitch_shift(waveform, sr=self.sample_rate,
                                               n_steps=pitch_shift)
        return torch.from_numpy(waveform)


class RandomVolume(object):
    def __init__(self, gain_db=(-50.0, 50.0)):
        self.gain = gain_db

    def __call__(self, waveform):
        rand_gain = random.uniform(self.gain[0], self.gain[1])
        return torch.clamp(torchaudio.functional.gain(waveform, rand_gain), -1.0, 1.0)


class AudioNoise(object):
    def __init__(self, scale=0.25, sample_rate=22050, examples=None):
        self.scale = scale
        self.sample_rate = sample_rate
        if examples is None:
            examples = ['brahms', 'choice', 'fishin', 'nutcracker', 'trumpet', 'vibeace']
            self.examples = []

            for example in examples:
                waveform, sample_rate = librosa.load(librosa.example(example))
                if sample_rate != self.sample_rate:
                    waveform = librosa.core.resample(waveform, sample_rate, self.sample_rate)
                self.examples.append(torch.from_numpy(waveform))
        else:
            self.examples = examples

    def __call__(self, waveform):
        noise = random.choice(self.examples)
        if noise.shape[0] < waveform.shape[0]:
            noise = noise.repeat(waveform.shape[0] // noise.shape[0] + 1)

        rand_pos = random.randrange(noise.shape[0] - waveform.shape[0] + 1)
        noise = noise[rand_pos:rand_pos + waveform.shape[0]]
        return waveform + self.scale * noise


class GaussianNoise(object):
    def __init__(self, scale=0.01):
        self.scale = scale

    def __call__(self, data):
        return data + self.scale * torch.randn(data.shape)


class SpectogramNormalize(object):
    def __init__(self, mean=-7.0, std=6.0, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = 1e-8

    def __call__(self, spec):
        spec = torch.log(spec + self.eps)
        spec = (spec - self.mean) / self.std
        return spec