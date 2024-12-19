import random
import numpy as np
from scipy.signal import resample
from scipy import signal


class BasicAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if np.random.random() < self.p:
            return self.augment(x)
        return x

    def augment(self, x):
        raise NotImplementedError


class RandomGaussianNoise(BasicAugmentation):
    def __init__(self, p=0.5, mean=0, std=0.1):
        super().__init__(p)
        self.mean = mean
        self.std = std

    def augment(self, x):
        return x + np.random.normal(self.mean, self.std, x.shape)


class RandomLaplaceNoise(BasicAugmentation):
    def __init__(self, p=0.5, loc=0.0, scale=0.01):
        super().__init__(p)
        self.loc = loc
        self.scale = scale

    def augment(self, x):
        noise = np.random.laplace(self.loc, self.scale, x.shape)
        return x + noise


class RandomTimeRoll(BasicAugmentation):
    def __init__(self, p=0.5, max_shift_prop=0.2):
        super().__init__(p)
        self.max_shift_prop = max_shift_prop

    def augment(self, x):
        shift = np.random.randint(-int(self.max_shift_prop * x.shape[-1]), 
                                int(self.max_shift_prop * x.shape[-1]))
        return np.roll(x, shift, axis=-1)


class RandomTimeWarp(BasicAugmentation):
    def __init__(self, p=0.5, warp_mean=0.8, warp_std=1.2):
        super().__init__(p)
        self.warp_mean = warp_mean
        self.warp_std = warp_std

    def augment(self, x):
        time_indices = np.arange(x.shape[-1])
        warp_factor = np.random.uniform(self.warp_mean, self.warp_std, size=x.shape[-1])
        new_indices = np.clip(np.round(time_indices * warp_factor).astype(int), 
                            0, x.shape[-1]-1)
        return x[..., new_indices]


class RandomFrequencyNoise(BasicAugmentation):
    def __init__(self, p=0.5, mean=0, std=0.1):
        super().__init__(p)
        self.mean = mean
        self.std = std

    def augment(self, x):
        x_fft = np.fft.fft(x)
        noise = np.random.normal(self.mean, self.std, x_fft.shape)
        return np.fft.ifft(x_fft + noise).real


class RandomFrequencyDropout(BasicAugmentation):
    def __init__(self, p=0.5, drop_prob=0.1):
        super().__init__(p)
        self.drop_prob = drop_prob

    def augment(self, x):
        x_fft = np.fft.fft(x)
        mask = np.random.choice([0, 1], size=x_fft.shape, p=[self.drop_prob, 1-self.drop_prob])
        return np.fft.ifft(x_fft * mask).real


class RandomRegionShuffle(BasicAugmentation):
    def __init__(self, p=0.5):
        super().__init__(p)

    def augment(self, x):
        indices = np.random.permutation(x.shape[0])
        return x[indices]


class RandomAmplitudeScaling(BasicAugmentation):
    def __init__(self, p=0.5, min_scale=0.8, max_scale=1.2):
        super().__init__(p)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def augment(self, x):
        scale_factor = np.random.uniform(self.min_scale, self.max_scale)
        return x * scale_factor


class RandomRegionDropout(BasicAugmentation):
    def __init__(self, p=0.5, drop_prob=0.1):
        super().__init__(p)
        self.drop_prob = drop_prob

    def augment(self, x):
        mask = np.random.choice([0, 1], size=x.shape[0], p=[self.drop_prob, 1-self.drop_prob])
        return x * mask[:, np.newaxis]


class RandomTimeMasking(BasicAugmentation):
    def __init__(self, p=0.5, max_mask_prop=0.2):
        super().__init__(p)
        self.max_mask_prop = max_mask_prop

    def augment(self, x):
        mask_size = int(x.shape[-1] * self.max_mask_prop)
        if mask_size > 0:
            start = np.random.randint(0, x.shape[-1] - mask_size)
            x = x.copy()
            x[..., start:start+mask_size] = 0
        return x


class RandomFrequencyShift(BasicAugmentation):
    def __init__(self, p=0.5, max_shift=5):
        super().__init__(p)
        self.max_shift = max_shift

    def augment(self, x):
        x_fft = np.fft.fft(x)
        shift = np.random.randint(-self.max_shift, self.max_shift)
        return np.fft.ifft(np.roll(x_fft, shift)).real


class RandomFlip(BasicAugmentation):
    def __init__(self, p=0.5):
        super().__init__(p)

    def augment(self, x):
        return np.flip(x, axis=-1)


class TimeFrequencyMasking(BasicAugmentation):
    def __init__(self, p=0.5, max_time_mask=0.2, max_freq_mask=0.2):
        super().__init__(p)
        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask

    def augment(self, x):
        x = x.copy()
        # Time masking
        time_mask_size = int(x.shape[-1] * self.max_time_mask)
        if time_mask_size > 0:
            time_start = np.random.randint(0, x.shape[-1] - time_mask_size)
            x[..., time_start:time_start+time_mask_size] = 0

        # Frequency masking
        freq_mask_size = int(x.shape[0] * self.max_freq_mask)
        if freq_mask_size > 0:
            freq_start = np.random.randint(0, x.shape[0] - freq_mask_size)
            x[freq_start:freq_start+freq_mask_size, ...] = 0

        return x


class RandomPhaseShuffling(BasicAugmentation):
    def __init__(self, p=0.5):
        super().__init__(p)

    def augment(self, x):
        x_fft = np.fft.fft(x)
        random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, x_fft.shape))
        return np.fft.ifft(x_fft * random_phase).real


class DynamicRangeCompression(BasicAugmentation):
    def __init__(self, p=0.5, compression_factor=0.5):
        super().__init__(p)
        self.compression_factor = compression_factor

    def augment(self, x):
        return np.sign(x) * (np.abs(x) ** self.compression_factor)


class RandomResampling(BasicAugmentation):
    def __init__(self, p=0.5, min_rate=0.8, max_rate=1.2):
        super().__init__(p)
        self.min_rate = min_rate
        self.max_rate = max_rate

    def augment(self, x):
        rate = np.random.uniform(self.min_rate, self.max_rate)
        num_samples = int(x.shape[-1] * rate)
        if num_samples < 1:
            return x
        return resample(x, num_samples, axis=-1)


class RandomCropOrPad(BasicAugmentation):
    def __init__(self, p=1, max_pad=120):
        super().__init__(p)
        self.max_pad = max_pad

    def augment(self, x):
        if x.shape[-1] > self.max_pad:
            start = np.random.randint(0, x.shape[-1] - self.max_pad)
            return x[..., start:start+self.max_pad]
        else:
            pad_width = ((0, 0), (0, self.max_pad - x.shape[-1]))
            return np.pad(x, pad_width, 'constant', constant_values=0)


class StaticCropOrPad:
    def __init__(self, max_pad=10):
        self.max_pad = max_pad

    def __call__(self, x):
        if x.shape[-1] > self.max_pad:
            return x[..., :self.max_pad]
        else:
            pad_width = ((0, 0), (0, self.max_pad - x.shape[-1]))
            return np.pad(x, pad_width, 'constant', constant_values=0)


class RandomBaselineShift(BasicAugmentation):
    def __init__(self, p=0.5, max_shift=0.1):
        super().__init__(p)
        self.max_shift = max_shift

    def augment(self, x):
        drift = np.linspace(0, np.random.uniform(-self.max_shift, self.max_shift), x.shape[-1])
        return x + drift


class RandomLowFrequencyTrend(BasicAugmentation):
    def __init__(self, p=0.5, max_amplitude=0.1, max_freq=0.1):
        super().__init__(p)
        self.max_amplitude = max_amplitude
        self.max_freq = max_freq

    def augment(self, x):
        t = np.arange(x.shape[-1])
        freq = np.random.uniform(0, self.max_freq)
        amplitude = np.random.uniform(0, self.max_amplitude)
        trend = amplitude * np.sin(2 * np.pi * freq * t)
        return x + trend


class RandomSpikes(BasicAugmentation):
    def __init__(self, p=0.5, num_spikes=3, max_amplitude=2.0):
        super().__init__(p)
        self.num_spikes = num_spikes
        self.max_amplitude = max_amplitude

    def augment(self, x):
        x_aug = x.copy()
        positions = np.random.choice(x.shape[-1], self.num_spikes)
        amplitudes = np.random.uniform(-self.max_amplitude, self.max_amplitude, self.num_spikes)
        x_aug[..., positions] += amplitudes
        return x_aug


class RandomSmoothing(BasicAugmentation):
    def __init__(self, p=0.5, min_sigma=0.5, max_sigma=2.0):
        super().__init__(p)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def augment(self, x):
        from scipy.ndimage import gaussian_filter1d
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)
        return gaussian_filter1d(x, sigma, axis=-1)


class RandomHemodynamicResponse(BasicAugmentation):
    def __init__(self, p=0.5, response_variation=0.2):
        super().__init__(p)
        self.response_variation = response_variation

    def augment(self, x):
        x_fft = np.fft.fft(x)
        freq_response = 1 + np.random.uniform(-self.response_variation, 
                                            self.response_variation, 
                                            x_fft.shape)
        return np.fft.ifft(x_fft * freq_response).real


class RandomBandpassFilter(BasicAugmentation):
    def __init__(self, p=0.5, low_freq=0.01, high_freq=0.1):
        super().__init__(p)
        self.low_freq = low_freq
        self.high_freq = high_freq

    def augment(self, x):
        nyquist = 0.5
        low = np.random.uniform(0, self.low_freq)
        high = np.random.uniform(self.high_freq, nyquist)
        b, a = signal.butter(3, [low, high], btype='band')
        return signal.filtfilt(b, a, x, axis=-1)


class RandomDetrend(BasicAugmentation):
    def __init__(self, p=0.5, order=3, detrend_type='linear'):
        super().__init__(p)
        self.order = order
        self.detrend_type = detrend_type

    def augment(self, x):
        return signal.detrend(x, axis=-1, type=self.detrend_type, bp=self.order)


class RandomROICorrelation(BasicAugmentation):
    def __init__(self, p=0.5, correlation_strength=0.3):
        super().__init__(p)
        self.correlation_strength = correlation_strength

    def augment(self, x):
        if x.shape[0] < 2:  # Need at least 2 ROIs
            return x
        noise = np.random.normal(0, 1, x.shape)
        shared_noise = np.random.normal(0, 1, (1, x.shape[-1]))
        mixed_noise = (1 - self.correlation_strength) * noise + \
                     self.correlation_strength * shared_noise
        return x + 0.1 * mixed_noise


class RandomMotionArtifact(BasicAugmentation):
    def __init__(self, p=0.5, max_displacement=0.2):
        super().__init__(p)
        self.max_displacement = max_displacement

    def augment(self, x):
        displacement = np.random.uniform(-self.max_displacement, 
                                       self.max_displacement, 
                                       x.shape[-1])
        return x + displacement * np.random.randn(*x.shape)


class RandomPhysiologicalNoise(BasicAugmentation):
    """Adds physiological noise patterns"""
    def __init__(self, p=0.5, cardiac_freq=1.2, respiratory_freq=0.3):
        super().__init__(p)
        self.cardiac_freq = cardiac_freq
        self.respiratory_freq = respiratory_freq

    def augment(self, x):
        t = np.arange(x.shape[-1])
        cardiac = 0.1 * np.sin(2 * np.pi * self.cardiac_freq * t)
        respiratory = 0.15 * np.sin(2 * np.pi * self.respiratory_freq * t)
        return x + cardiac + respiratory


class Compose:
    def __init__(self, transforms: list, p=1):
        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            for t in self.transforms:
                x = t(x)
        return x


class OneOf:
    def __init__(self, transforms: list, p: float = 0.5):
        self.transforms = transforms
        self.p = p
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, x):
        if random.random() < self.p:
            transform = np.random.choice(self.transforms, p=self.transforms_ps)
            return transform(x)
        return x


class Transforms:
    def __init__(self):
        self.train_transforms = Compose([
            # Signal Processing Group
            OneOf([
                RandomDetrend(p=0.7, order=3),
                RandomBandpassFilter(p=0.7, low_freq=0.01, high_freq=0.1),
            ], p=0.8),
            
            # Noise Simulation Group
            OneOf([
                RandomGaussianNoise(p=0.5, mean=0, std=0.02),
                RandomLaplaceNoise(p=0.5, loc=0, scale=0.01),
                RandomFrequencyNoise(p=0.5, mean=0, std=0.05),
            ], p=0.7),
            
            # Temporal Manipulation Group
            OneOf([
                RandomTimeRoll(p=0.5, max_shift_prop=0.05),
                RandomTimeWarp(p=0.5, warp_mean=0.95, warp_std=1.05),
                RandomTimeMasking(p=0.5, max_mask_prop=0.05),
                RandomFlip(p=0.3),
            ], p=0.7),
            
            # Frequency Domain Group
            OneOf([
                RandomFrequencyDropout(p=0.5, drop_prob=0.05),
                RandomFrequencyShift(p=0.5, max_shift=3),
                RandomPhaseShuffling(p=0.5),
            ], p=0.6),
            
            # ROI Manipulation Group
            OneOf([
                RandomRegionShuffle(p=0.4),
                RandomRegionDropout(p=0.4, drop_prob=0.05),
                RandomROICorrelation(p=0.5, correlation_strength=0.2),
            ], p=0.6),
            
            # Amplitude and Scale Group
            OneOf([
                RandomAmplitudeScaling(p=0.5, min_scale=0.9, max_scale=1.1),
                DynamicRangeCompression(p=0.5, compression_factor=0.9),
                RandomResampling(p=0.5, min_rate=0.9, max_rate=1.1),
            ], p=0.7),
            
            # Artifact Simulation Group
            OneOf([
                RandomSpikes(p=0.4, num_spikes=2, max_amplitude=1.5),
                RandomBaselineShift(p=0.4, max_shift=0.05),
                RandomLowFrequencyTrend(p=0.4, max_amplitude=0.05, max_freq=0.05),
            ], p=0.5),
            
            # Physiological Noise Group
            OneOf([
                RandomPhysiologicalNoise(p=0.5, cardiac_freq=1.2, respiratory_freq=0.3),
                RandomMotionArtifact(p=0.5, max_displacement=0.1),
                RandomHemodynamicResponse(p=0.5, response_variation=0.1),
            ], p=0.6),
            
            # Signal Processing Group
            OneOf([
                RandomSmoothing(p=0.5, min_sigma=0.5, max_sigma=1.5),
                TimeFrequencyMasking(p=0.5, max_time_mask=0.1, max_freq_mask=0.1),
            ], p=0.5),
        ])
        
        # Set test transforms to None
        self.test_transforms = None