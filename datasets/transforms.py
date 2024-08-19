import torch
import numpy as np
import torchio as tio


class FrameTransform:
    def __init__(self, img_size: int, training: bool = True):
        if training:
            self.transform = tio.Compose([
                tio.ToCanonical(),
                tio.Resample(3),
                tio.CropOrPad(img_size),
                # artifact
                tio.OneOf({
                    tio.RandomGhosting(): 0.25,
                    tio.RandomMotion(): 0.25,
                    tio.RandomSpike(): 0.25,
                    tio.RandomBiasField(): 0.25,
                }),
                tio.RandomBlur(p=0.3),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                tio.RandomNoise(p=0.5),
                tio.RandomSwap(p=0.3),
                tio.RandomFlip(),
                tio.OneOf({
                    tio.RandomAffine(): 0.8,
                    tio.RandomElasticDeformation(): 0.2,
                }),
            ])

        else:
            self.transform = tio.Compose([
                tio.ToCanonical(),
                tio.Resample(3),
                tio.CropOrPad(img_size),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            ])
    
    def __call__(self, sample):
        return self.transform(sample).data.float()


class FmriTransform:
    def __init__(self, img_size: int, training: bool = True):

        self.training = training

        self.transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(3),
            tio.CropOrPad(img_size),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        ])

    @staticmethod
    def random_sampling(sample):
        # pseudo bags
        n = sample.shape[0]
        p = np.random.uniform(0.3, 1)
        k = int(n * p)
        start_idx = np.random.randint(0, n - k + 1)
        return sample[start_idx:start_idx + k]
    
    def __call__(self, sample):
        sample = self.transform(sample).data.float()
        if self.training:
            sample = self.random_sampling(sample)
        return sample
