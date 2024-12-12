import random
import numpy as np
from scipy.signal import resample


class BasicAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def augment(self, x):
        raise NotImplementedError
    
    def __call__(self, x):
        if random.random() < self.p:
            return self.augment(x)
        return x


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
        # Generate Laplace noise
        noise = np.random.laplace(self.loc, self.scale, x.shape)
        return x + noise


class RandomTimeRoll(BasicAugmentation):
    def __init__(self, p=0.5, max_shift_prop=0.2):
        super().__init__(p)
        self.max_shift_prop = max_shift_prop

    def augment(self, x):
        shift = np.random.randint(-self.max_shift_prop * x.shape[-1], self.max_shift_prop * x.shape[-1])
        return np.roll(x, shift, axis=-1) # x: [num_rois, time_steps]


class RandomTimeWarp(BasicAugmentation):
    def __init__(self, p=0.5, warp_mean=0.8, warp_std=1.2):
        super().__init__(p)
        self.warp_mean = warp_mean
        self.warp_std = warp_std

    def augment(self, x):
        time_indices = np.arange(x.shape[-1])
        warp_factor = np.random.uniform(0.8, 1.2, size=x.shape[-1])
        new_indices = np.clip(np.round(time_indices * warp_factor).astype(int), 0, x.shape[1]-1)
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
        np.random.shuffle(x)
        return x


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
        start = np.random.randint(0, x.shape[-1] - mask_size)
        x[..., start:start+mask_size] = 0
        return x


class RandomFrequencyShift(BasicAugmentation):
    def __init__(self, p=0.5, max_shift=5):
        super().__init__(p)
        self.max_shift = max_shift

    def augment(self, x):
        x_fft = np.fft.fft(x)
        shift = np.random.randint(-self.max_shift, self.max_shift)
        x_fft = np.roll(x_fft, shift)
        return np.fft.ifft(x_fft).real


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
        # Time masking
        time_mask_size = int(x.shape[-1] * self.max_time_mask)
        time_start = np.random.randint(0, x.shape[-1] - time_mask_size)
        x[..., time_start:time_start+time_mask_size] = 0

        # Frequency masking
        freq_mask_size = int(x.shape[0] * self.max_freq_mask)
        freq_start = np.random.randint(0, x.shape[0] - freq_mask_size)
        x[freq_start:freq_start+freq_mask_size, ...] = 0

        return x


class RandomPhaseShuffling(BasicAugmentation):
    def __init__(self, p=0.5):
        super().__init__(p)

    def augment(self, x):
        x_fft = np.fft.fft(x)
        random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, x_fft.shape))
        x_fft *= random_phase
        return np.fft.ifft(x_fft).real


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
            return np.pad(x, ((0, 0), (0, self.max_pad - x.shape[-1])), 'constant', constant_values=0)


class StaticCropOrPad:
    def __init__(self, max_pad=10):
        self.max_pad = max_pad

    def __call__(self, x):
        if x.shape[-1] > self.max_pad:
            return x[..., :self.max_pad]
        else:
            return np.pad(x, ((0, 0), (0, self.max_pad - x.shape[-1])), 'constant', constant_values=0)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x):
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
            OneOf([
                RandomGaussianNoise(),
                RandomLaplaceNoise(),
            ], p=0.5),
            
            OneOf([
                RandomAmplitudeScaling(),
                DynamicRangeCompression(),
            ], p=0.5),
            
            OneOf([
                RandomTimeRoll(),
                RandomTimeWarp(),
                RandomTimeMasking(),
            ], p=0.5),

            OneOf([
                RandomFrequencyNoise(),
                RandomFrequencyDropout(),
                RandomFrequencyShift(),
                RandomPhaseShuffling(),
            ], p=0.5),
            
            OneOf([
                TimeFrequencyMasking(),
                RandomRegionShuffle(),
                RandomRegionDropout(),
            ], p=0.5),

            OneOf([
                RandomCropOrPad(),
                RandomResampling(),
            ], p=0.5),
        ])
        self.test_transforms = None
