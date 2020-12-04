from art.defences.preprocessor.gaussian_augmentation import GaussianAugmentation
from art.defences.preprocessor import Preprocessor
from art.config import ART_NUMPY_DTYPE
import numpy as np
from eval2.defenses.filter import ASNRWiener, SSFPreprocessor
import logging
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)

class SpeechNoiseAugmentation(GaussianAugmentation):
    def __init__(self,*args,high_freq=False, filter=None,filter_kwargs={},enhancer=None,enhancer_kwargs={},**kwargs):
        super(SpeechNoiseAugmentation,self).__init__(*args,**kwargs)
        self.filter=None
        self.enhancer=None
        self.high_freq=high_freq
        if filter is not None:
            if filter=="asnr_wiener":
                self.filter=ASNRWiener(gaussian_sigma=self.sigma,high_freq=high_freq, **filter_kwargs)
            elif filter=="ssf_enhancer":
                self.filter=SSFPreprocessor(**filter_kwargs)
            else:
                raise ValueError("Unrecognized filter %s"%filter)
        
        if enhancer is not None:
            from eval2.defenses.enhancer import SEGANEnhancer
            if enhancer=="segan":
                self.enhancer=SEGANEnhancer(**enhancer_kwargs)
            else:
                raise ValueError("Unrecognized enhancer %s"%enhancer)
    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Augment the sample `(x, y)` with Gaussian noise. The result is either an extended dataset containing the
        original sample, as well as the newly created noisy samples (augmentation=True) or just the noisy counterparts
        to the original samples.

        :param x: Sample to augment with shape `(batch_size, width, height, depth)`.
        :param y: Labels for the sample. If this argument is provided, it will be augmented with the corresponded
                  original labels of each sample point.
        :return: The augmented dataset and (if provided) corresponding labels.
        """
        # Select indices to augment
        if self.augmentation:
            logger.info("Original dataset size: %d", x.shape[0])
            size = int(x.shape[0] * self.ratio)
            indices = np.random.randint(0, x.shape[0], size=size)

            # Generate noisy samples
            x_aug = self.augment(x[indices])
            x_aug = np.vstack((x, x_aug))
            if y is not None:
                y_aug = np.concatenate((y, y[indices]))
            else:
                y_aug = y
            logger.info("Augmented dataset size: %d", x_aug.shape[0])
        else:
            x_aug = self.augment(x)
            y_aug = y

        if self.clip_values is not None:
            x_aug = np.clip(x_aug, self.clip_values[0], self.clip_values[1])
        #print("SNR :",[self.snr_db(x[i],x_aug[i]) for i in range(len(x))])
        if self.filter:
            x_filt, y_filt=self.filter(x_aug, y_aug)
        else:
            x_filt, y_filt=x_aug, y_aug
        if self.enhancer:
            x_enh, y_enh=self.enhancer(x_filt, y_filt)
        else:
            x_enh, y_enh=x_filt, y_filt
        
        return x_enh, y_enh

    def snr_db(self,x,x_aug):
        signal_power = (x ** 2).mean()
        noise_power = ((x - x_aug) ** 2).mean()
        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr)
        return snr_db

    def augment(self,x: np.ndarray) -> np.ndarray:
        #x_aug = np.random.normal(x, scale=self.sigma, size=x.shape).astype(ART_NUMPY_DTYPE)
        x_aug = np.copy(x)
        for i in range(x.shape[0]):
            assert len(x[i].shape)==1
            if self.high_freq:
                noise = np.random.normal(0, scale=self.sigma, size=(x[i].shape[0]+1,))
                noise = 0.5 * (noise[1:]-noise[:-1])
            else:
                noise = np.random.normal(0, scale=self.sigma, size=x[i].shape)
            x_aug[i] = (x[i]+noise).astype(ART_NUMPY_DTYPE)
        return x_aug

