import numpy as np
import torch
import torchaudio.compliance.kaldi


class ToMono:
    def __call__(self, x):
        if x.ndim == 1:
            return x[None]
        elif x.ndim == 2:
            return x.mean(0, keepdims=True)
        else:
            raise ValueError('Audio tensor should have at most 2 dimensions (c,t)')


class CAV_Wav2fbank:
    def __init__(self, sampling_rate: int, num_mel_bins: int, target_length: int, freqm: int, timem: int):
        self.sr = sampling_rate
        self.melbins = num_mel_bins
        self.target_length = target_length
        if freqm != 0:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm)
        else:
            self.freqm = None
        if timem != 0:
            self.timem = torchaudio.transforms.TimeMasking(timem)
        else:
            self.timem = None

    def __call__(self, x):
        fbank = torchaudio.compliance.kaldi.fbank(x, htk_compat=True, sample_frequency=self.sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                  frame_shift=10)
        n_frames = fbank.shape[0]

        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]

        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm is not None:
            fbank = self.freqm(fbank)
        if self.timem is not None:
            fbank = self.timem(fbank)
        fbank = torch.transpose(fbank, 1, 2)

        return fbank


class CAV_Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = (x.squeeze() - self.mean) / self.std
        return x.unsqueeze(0)


class Noise:
    def __init__(self, target_length):
        self.target_length = target_length
    def __call__(self, x):
        x = x.squeeze(0)
        device = x.device
        noise = torch.rand(x.shape[0], x.shape[1]) * np.random.rand() / 10
        noise = noise.to(device)
        x = x + noise
        x = torch.roll(x, np.random.randint(-self.target_length, self.target_length), 0)

        return x.unsqueeze(0)


class ToTensor:

    def __call__(self, array):
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
        elif isinstance(array, torch.Tensor):
            tensor = array
        else:
            raise ValueError("Input array is neither numpy array nor torch tensor")
        return tensor