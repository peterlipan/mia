import random
import numpy as np


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


class RandomRegionDropout(BasicAugmentation):
    def __init__(self, p=0.5, drop_prob=0.1):
        super().__init__(p)
        self.drop_prob = drop_prob

    def augment(self, x):
        mask = np.random.choice([0, 1], size=x.shape[0], p=[self.drop_prob, 1-self.drop_prob])
        return x * mask[:, np.newaxis]


class RandomCropOrPad(BasicAugmentation):
    def __init__(self, p=1, max_pad=10):
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
            RandomRegionShuffle(),
            RandomRegionDropout(),
            RandomGaussianNoise(),
            OneOf([
                RandomFrequencyNoise(),
                RandomFrequencyDropout(),
            ], p=0.5),
            OneOf([
                RandomTimeRoll(max_shift_prop=0.5),
                RandomTimeWarp(),
            ], p=0.5),
        ])
        self.test_transforms = None
