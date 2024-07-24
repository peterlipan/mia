import torch
import torchio as tio


class Transform:
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
