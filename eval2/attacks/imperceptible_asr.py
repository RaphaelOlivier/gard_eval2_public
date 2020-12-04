from __future__ import absolute_import, division, print_function, unicode_literals

# version of the attack that runs with smoothed model (and its preprocessing defenses)

import logging
from typing import Tuple, Optional, Union, TYPE_CHECKING

import numpy as np
import scipy
from art.config import ART_NUMPY_DTYPE

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
from art.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
logger = logging.getLogger(__name__)

class PreprocessingInPytorch(Function):
    @staticmethod
    def forward(ctx,x,npmask, preprocessing_defences):
        x_np = x.detach().cpu().numpy()
        
        x_np_var_len = np.array([x_np[i,:int(npmask[i].sum())] for i in range(len(npmask))])
        for defense in preprocessing_defences:
            x_np_var_len,_ = defense(x_np_var_len)
        x_np_preprocessed = np.zeros(shape=x_np.shape)
        for i in range(len(npmask)):
            x_np_preprocessed[i,:len(x_np_var_len[i])]=x_np_var_len[i]

        x_preprocessed = torch.tensor(x_np).to(x.device)
        x_preprocessed.requires_grad = x.requires_grad
        return x_preprocessed
    
    @staticmethod
    def backward(ctx,grad_output):
        grad_input=grad_output.clone()
        return grad_input, None, None


class PreprocessorWrapper(nn.Module):
    def __init__(self,preprocessing_defences):
        super(PreprocessorWrapper,self).__init__()
        self.npmodule=preprocessing_defences
    def forward(self,x, npmask):
        if self.npmodule is None or self.npmodule == []:
            return x
        x_preprocessed=PreprocessingInPytorch.apply(x,npmask, self.npmodule)
        return x_preprocessed

class ImperceptibleASRWithPreprocessing(ImperceptibleASRPytorch):
    def __init__(
        self,
        estimator: PyTorchDeepSpeech,
        initial_eps: float = 0.001,
        max_iter_1st_stage: int = 1000,
        max_iter_2nd_stage: int = 4000,
        learning_rate_1st_stage: float = 0.1,
        learning_rate_2nd_stage: float = 0.001,
        optimizer_1st_stage: Optional["torch.optim.Optimizer"] = None,
        optimizer_2nd_stage: Optional["torch.optim.Optimizer"] = None,
        global_max_length: int = 10000,
        initial_rescale: float = 1.0,
        rescale_factor: float = 0.8,
        num_iter_adjust_rescale: int = 10,
        initial_alpha: float = 0.05,
        increase_factor_alpha: float = 1.2,
        num_iter_increase_alpha: int = 20,
        decrease_factor_alpha: float = 0.8,
        num_iter_decrease_alpha: int = 50,
        batch_size: int = 1,
        use_amp: bool = False,
        opt_level: str = "O1",
        loss_scale: Optional[Union[float, str]] = 1.0,
        niters_gradients:int = 1,
        batch_backward:bool = False
    ):
        
        self.wrapper_preprocessing_defences=None
        self.niters_gradients = niters_gradients
        if hasattr(estimator, "preprocessing_defences") and (estimator.preprocessing_defences is not None and estimator.preprocessing_defences != []):
            self.wrapper_preprocessing_defences=PreprocessorWrapper(estimator.preprocessing_defences)

        if hasattr(estimator, "preprocessing") and (estimator.preprocessing is not None and estimator.preprocessing != (0, 1)):
            raise NotImplementedError(
                "The framework-specific implementation currently does not apply preprocessing."
            )
        EvasionAttack.__init__(self,estimator=estimator)

        # Set attack attributes
        self.initial_eps = initial_eps
        self.max_iter_1st_stage = max_iter_1st_stage
        self.max_iter_2nd_stage = max_iter_2nd_stage
        self.learning_rate_1st_stage = learning_rate_1st_stage
        self.learning_rate_2nd_stage = learning_rate_2nd_stage
        self.global_max_length = global_max_length
        self.initial_rescale = initial_rescale
        self.rescale_factor = rescale_factor
        self.num_iter_adjust_rescale = num_iter_adjust_rescale
        self.initial_alpha = initial_alpha
        self.increase_factor_alpha = increase_factor_alpha
        self.num_iter_increase_alpha = num_iter_increase_alpha
        self.decrease_factor_alpha = decrease_factor_alpha
        self.num_iter_decrease_alpha = num_iter_decrease_alpha
        self.batch_size = batch_size
        self._use_amp = use_amp
        self.batch_backward=batch_backward

        # Create the main variable to optimize
        self.global_optimal_delta = Variable(
            torch.zeros(self.batch_size, self.global_max_length).type(torch.FloatTensor), requires_grad=True
        )
        self.global_optimal_delta.to(self.estimator.device)

        # Create the optimizers
        if optimizer_1st_stage is None:
            self.optimizer_1st_stage = torch.optim.SGD(
                params=[self.global_optimal_delta], lr=self.learning_rate_1st_stage
            )
        else:
            self.optimizer_1st_stage = optimizer_1st_stage(
                params=[self.global_optimal_delta], lr=self.learning_rate_1st_stage
            )
        if optimizer_2nd_stage is None:
            self.optimizer_2nd_stage = torch.optim.SGD(
                params=[self.global_optimal_delta], lr=self.learning_rate_1st_stage
            )
        else:
            self.optimizer_2nd_stage = optimizer_2nd_stage(
                params=[self.global_optimal_delta], lr=self.learning_rate_1st_stage
            )

        # Setup for AMP use
        if self._use_amp:
            from apex import amp

            if self.estimator.device.type == "cpu":
                enabled = False
            else:
                enabled = True

            self.estimator._model, [self.optimizer_1st_stage, self.optimizer_2nd_stage] = amp.initialize(
                models=self.estimator._model,
                optimizers=[self.optimizer_1st_stage, self.optimizer_2nd_stage],
                enabled=enabled,
                opt_level=opt_level,
                loss_scale=loss_scale,
            )

        # Check validity of attack attributes
        self._check_params()

    def _forward_1st_stage(
        self,
        original_input: np.ndarray,
        original_output: np.ndarray,
        local_batch_size: int,
        local_max_length: int,
        rescale: np.ndarray,
        input_mask: np.ndarray,
        real_lengths: np.ndarray,
    ) -> Tuple["torch.Tensor", "torch.Tensor", np.ndarray, "torch.Tensor", "torch.Tensor"]:
        """
        The forward pass of the first stage of the attack.

        :param original_input: Samples of shape (nb_samples, seq_length). Note that, sequences in the batch must have
                               equal lengths. A possible example of `original_input` could be:
                               `original_input = np.array([np.array([0.1, 0.2, 0.1]), np.array([0.3, 0.1, 0.0])])`.
        :param original_output: Target values of shape (nb_samples). Each sample in `original_output` is a string and
                                it may possess different lengths. A possible example of `original_output` could be:
                                `original_output = np.array(['SIXTY ONE', 'HELLO'])`.
        :param local_batch_size: Current batch size.
        :param local_max_length: Max length of the current batch.
        :param rescale: Current rescale coefficients.
        :param input_mask: Masks of true inputs.
        :param real_lengths: Real lengths of original sequences.
        :return: A tuple of (loss, local_delta, decoded_output, masked_adv_input)
                    - loss: The loss tensor of the first stage of the attack.
                    - local_delta: The delta of the current batch.
                    - decoded_output: Transcription output.
                    - masked_adv_input: Perturbed inputs.
        """
        import torch  # lgtm [py/repeated-import]
        from warpctc_pytorch import CTCLoss

        # Compute perturbed inputs
        local_delta = self.global_optimal_delta[:local_batch_size, :local_max_length]
        local_delta_rescale = torch.clamp(local_delta, -self.initial_eps, self.initial_eps).to(self.estimator.device)
        local_delta_rescale *= torch.tensor(rescale).to(self.estimator.device)
        adv_input = local_delta_rescale + torch.tensor(original_input).to(self.estimator.device)
        masked_adv_input = adv_input * torch.tensor(input_mask).to(self.estimator.device)

        # Transform data into the model input space
        masked_adv_input_batch = masked_adv_input.expand(self.niters_gradients,masked_adv_input.size(1))
        input_mask=np.repeat(input_mask,self.niters_gradients,0)
        real_lengths=np.repeat(real_lengths,self.niters_gradients,0)
        original_output = np.repeat(original_output,self.niters_gradients,0)
        masked_adv_input_batch=self.wrapper_preprocessing_defences(masked_adv_input_batch,input_mask)
        inputs, targets, input_rates, target_sizes, batch_idx = self.estimator.transform_model_input(
            x=masked_adv_input_batch.to(self.estimator.device),
            y=original_output,
            compute_gradient=False,
            tensor_input=True,
            real_lengths=real_lengths,
        )
        # Compute real input sizes
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()

        # Call to DeepSpeech model for prediction
        outputs, output_sizes = self.estimator.model(
            inputs.to(self.estimator.device), input_sizes.to(self.estimator.device)
        )
        outputs_ = outputs.transpose(0, 1)
        float_outputs = outputs_.float()

        # Loss function
        criterion = CTCLoss()
        loss = criterion(float_outputs, targets, output_sizes, target_sizes).to(self.estimator.device)
        loss = loss / inputs.size(0)

        # Compute transcription
        decoded_output, _, _ = self.estimator.decoder.decode(outputs, output_sizes)
        decoded_output = [do[0] for do in decoded_output]
        decoded_output = np.array(decoded_output)

        # Rearrange to the original order
        decoded_output_ = decoded_output.copy()
        decoded_output[batch_idx] = decoded_output_

        return loss, local_delta, decoded_output, masked_adv_input, local_delta_rescale

    def _attack_1st_stage(self, x: np.ndarray, y: np.ndarray) -> Tuple["torch.Tensor", np.ndarray]:
        """
        The first stage of the attack.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`. Note that, this
                  class only supports targeted attack.
        :return: A tuple of two tensors:
                    - A tensor holding the candidate adversarial examples.
                    - An array holding the original inputs.
        """
        import torch  # lgtm [py/repeated-import]

        # Compute local shape
        local_batch_size = len(x)
        real_lengths = np.array([x_.shape[0] for x_ in x])
        local_max_length = np.max(real_lengths)

        # Initialize rescale
        rescale = np.ones([local_batch_size, local_max_length], dtype=np.float32) * self.initial_rescale

        # Reformat input
        input_mask = np.zeros([local_batch_size, local_max_length], dtype=np.float32)
        original_input = np.zeros([local_batch_size, local_max_length], dtype=np.float32)

        for local_batch_size_idx in range(local_batch_size):
            input_mask[local_batch_size_idx, : len(x[local_batch_size_idx])] = 1
            original_input[local_batch_size_idx, : len(x[local_batch_size_idx])] = x[local_batch_size_idx]

        # Optimization loop
        successful_adv_input = [None] * local_batch_size
        trans = [None] * local_batch_size

        for iter_1st_stage_idx in range(self.max_iter_1st_stage):
            # Zero the parameter gradients
            self.optimizer_1st_stage.zero_grad()

            # Call to forward pass
            loss, local_delta, decoded_output, masked_adv_input, _ = self._forward_1st_stage(
                original_input=original_input,
                original_output=y,
                local_batch_size=local_batch_size,
                local_max_length=local_max_length,
                rescale=rescale,
                input_mask=input_mask,
                real_lengths=real_lengths,
            )

            # Actual training
            if self._use_amp:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer_1st_stage) as scaled_loss:
                    scaled_loss.backward()

            else:
                loss.backward()

            # Get sign of the gradients
            self.global_optimal_delta.grad = torch.sign(self.global_optimal_delta.grad)

            # Do optimization
            self.optimizer_1st_stage.step()

            # Save the best adversarial example and adjust the rescale coefficient if successful
            if iter_1st_stage_idx % self.num_iter_adjust_rescale == 0:
                for local_batch_size_idx in range(local_batch_size):
                    if decoded_output[local_batch_size_idx] == y[local_batch_size_idx]:
                        # Adjust the rescale coefficient
                        max_local_delta = np.max(np.abs(local_delta[local_batch_size_idx].detach().numpy()))

                        if rescale[local_batch_size_idx][0] * self.initial_eps > max_local_delta:
                            rescale[local_batch_size_idx] = max_local_delta / self.initial_eps
                        rescale[local_batch_size_idx] *= self.rescale_factor

                        # Save the best adversarial example
                        successful_adv_input[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                        trans[local_batch_size_idx] = decoded_output[local_batch_size_idx]

            # If attack is unsuccessful
            if iter_1st_stage_idx == self.max_iter_1st_stage - 1:
                for local_batch_size_idx in range(local_batch_size):
                    if successful_adv_input[local_batch_size_idx] is None:
                        successful_adv_input[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                        trans[local_batch_size_idx] = decoded_output[local_batch_size_idx]
            self.estimator.model.train()

        result = torch.stack(successful_adv_input)

        return result, original_input

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`. Note that, this
                  class only supports targeted attack.
        :return: An array holding the adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        if y is None:
            raise ValueError("`ImperceptibleASRPytorch` is a targeted attack and requires the definition of target"
                             "labels `y`. Currently `y` is set to `None`.")

        # Start to compute adversarial examples
        adv_x = x.copy()

        # Put the estimator in the training mode
        self.estimator.model.train()

        # Compute perturbation with batching
        num_batch = int(np.ceil(len(x) / float(self.batch_size)))

        for m in range(num_batch):
            # Batch indexes
            batch_index_1, batch_index_2 = (
                m * self.batch_size,
                min((m + 1) * self.batch_size, len(x)),
            )

            # First reset delta
            self.global_optimal_delta.data = torch.zeros(self.batch_size, self.global_max_length).type(torch.float32)

            # Then compute the batch
            adv_x_batch = self._generate_batch(adv_x[batch_index_1:batch_index_2], y[batch_index_1:batch_index_2])

            for i in range(len(adv_x_batch)):
                adv_x[batch_index_1 + i] = adv_x_batch[i, : len(adv_x[batch_index_1 + i])]
        return adv_x