"""
Automatic speech recognition model
"""
import os
import numpy as np
import torch
from armory import paths
from art.estimators.speech_recognition import PyTorchDeepSpeech
from art.utils import get_file
from deepspeech_pytorch.utils import load_model
from art.config import ART_DATA_PATH
from eval2.models.vote import MajorityVote, Rover, MultipleProbsVote, VoteEnsemble, ROVER_MAX_HYPS
from eval2.models.decoder import load_decoder_with_scores
import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)

class SmoothedDeepSpeech(PyTorchDeepSpeech):
    def __init__(self,*args,voting_kwargs, niters_forward=1, niters_backward=1,batch_backward=0,batch_forward=0,load_weights_file=None, use_half=False, random_init=False, **kwargs):
        filename = load_weights_file if load_weights_file else "librispeech_pretrained_v2.pth"
        saved_model_dir = paths.runtime_paths().saved_model_dir
        model_path = os.path.join(saved_model_dir, filename)
        model=load_model(device="cpu", model_path=model_path, use_half=use_half)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4, weight_decay=1e-5, amsgrad=False)
        super(SmoothedDeepSpeech,self).__init__(model,*args,optimizer=optimizer,**kwargs)
        self.model_path=model_path
        self.use_half=use_half
        self.niters_forward=niters_forward
        self.niters_backward= niters_backward
        if random_init:
            for p in self._model.parameters():
                if p.dim()>1:
                    torch.nn.init.xavier_uniform(p)
                else:
                    torch.nn.init.zeros_(p)
        self.decoder = load_decoder_with_scores(self.decoder)
        self.set_voting_module(**voting_kwargs, **kwargs)
        self.batch_backward=batch_backward
        self.batch_forward=batch_forward

    def set_voting_module(self,voting="majority",rover_bin_path=None, vote_on_nbest=False, decoder_type="greedy", use_alignments=False,use_confidence=False, **kwargs):
        assert (not vote_on_nbest) or (decoder_type=="beam"), "option vote_on_nbest is incompatible with greedy decoding"
        assert (not use_confidence) or decoder_type=="beam" or voting=="majority", "use_confidence is currently not compatible with greedy search. You can use beam search with width 1"
        if voting in ["rover","rover_freq"]: # Rover by Word frequency
            if self.niters_forward>ROVER_MAX_HYPS:
                self.voting_module = VoteEnsemble(Rover(scheme='freq', exec_path=rover_bin_path, return_all=True),Rover(scheme='freq', exec_path=rover_bin_path))
            else:
                self.voting_module=Rover(scheme='freq', exec_path=rover_bin_path)
        elif voting=="rover_conf": # Rover by Average Confidence Scores 
            self.voting_module=Rover(scheme='conf', exec_path=rover_bin_path)
        elif voting=="rover_max": # Rover by Word Maximum Confidence Scores 
            self.voting_module=Rover(scheme='max', exec_path=rover_bin_path)
        elif voting=="probs_sum":
            self.voting_module=MultipleProbsVote(scheme="sum", decoder=self.decoder, device=self._device)
        elif voting=="probs_max":
            self.voting_module=MultipleProbsVote(scheme="max", decoder=self.decoder, device=self._device)
        elif voting=="probs_rover":
            self.voting_module = VoteEnsemble(MultipleProbsVote(scheme="sum", decoder=self.decoder, device=self._device, return_all = True), 
            Rover(scheme='freq', exec_path=rover_bin_path), agg_by=12)
        else:
            assert voting=="majority"
            self.voting_module=MajorityVote()

        self.vote_on_nbest=vote_on_nbest
        self.transcription_output = not voting.startswith("probs")
        self.use_alignments=use_alignments
        self.use_confidence=use_confidence
    def fit(self,*args,save_weights_file=None, **kwargs):
        super(SmoothedDeepSpeech,self).fit(*args,**kwargs)
        if save_weights_file:
            saved_model_dir = paths.runtime_paths().saved_model_dir
            save_weights_path=os.path.join(saved_model_dir,save_weights_file)
            dic=self._model.state_dict()
            torch.save(dic,save_weights_path)
    
    def reload_model(self):
        model=load_model(device="cpu", model_path=self.model_path, use_half=self.use_half)
        self._model=model.to(self._device)
    def predict(self,x: np.ndarray,reload_model=False,**kwargs):
        if reload_model:
            self.reload_model()
        if self.niters_forward<=1:
            decoded_output = self.predict_once(x, **kwargs)
        else:
            random_outputs=[]
            alignments=[]
            confidence=[]
            kwargs["transcription_output"]=self.transcription_output
            kwargs["return_alignments"]=True
            kwargs["return_confidence"]=True
            if self.vote_on_nbest:
                for i in range(self.niters_forward):
                    out=self.predict_once(x, return_nbest=True, **kwargs)
                    out, algns, scores = out
                    for k in range(len(out[0])):
                        decoded_output = [do[k] for do in out]
                        decoded_output = np.array(decoded_output)
                        random_outputs.append(decoded_output)
                        decoded_alignments = [al[k].cpu().numpy() for al in algns]
                        decoded_alignments = np.array(decoded_alignments)
                        alignments.append(decoded_alignments)
                        decoded_scores = [sc[k].cpu().numpy() for sc in scores]
                        decoded_scores = np.array(decoded_scores)
                        confidence.append(decoded_scores)
                        
            else:
                if self.batch_forward>0 and x.shape[0]==1:
                    for i in range(0,self.niters_forward,self.batch_forward):
                        batch_size = min(self.batch_forward,self.niters_forward-i)
                        x_batch = np.repeat(x,batch_size,axis=0)
                        out=self.predict_once(x_batch, **kwargs)
                        out, algns ,scores = out
                        for j in range(batch_size):
                            alignments.append(algns[j].reshape(1,-1))
                            confidence.append(scores[j].reshape(1,-1))
                            random_outputs.append(out[j].reshape(1))
                else:
                    for i in range(self.niters_forward):
                        out=self.predict_once(x, **kwargs)
                        out, algns ,scores = out
                        alignments.append(algns)
                        confidence.append(scores)
                        random_outputs.append(out)
            decoded_output=self.voting_module.run(asr_outputs=random_outputs, alignments=alignments, confidence=confidence)
        return decoded_output


    def predict_once(
        self, x: np.ndarray, batch_size: int = 128, return_nbest=False,return_alignments=False, return_confidence=False, **kwargs
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:

        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param batch_size: Batch size.
        :param transcription_output: Indicate whether the function will produce probability or transcription as
                                     prediction output. If transcription_output is not available, then probability
                                     output is returned.
        :type transcription_output: `bool`
        :return: Probability (if transcription_output is None or False) or transcription (if transcription_output is
                 True) predictions:
                 - Probability return is a tuple of (probs, sizes), where `probs` is the probability of characters of
                 shape (nb_samples, seq_length, nb_classes) and `sizes` is the real sequence length of shape
                 (nb_samples,).
                 - Transcription return is a numpy array of characters. A possible example of a transcription return
                 is `np.array(['SIXTY ONE', 'HELLO'])`.
        """
        import torch  # lgtm [py/repeated-import]
        
        x_ = np.array([x_i for x_i in x] + [np.array([0.1]), np.array([0.1, 0.2])])[:-2]

        # Put the model in the eval mode
        self._model.eval()
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x_, y=None, fit=False)

        # Transform x into the model input space
        inputs, targets, input_rates, target_sizes, batch_idx = self.transform_model_input(x=x_preprocessed)

        # Compute real input sizes
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()

        # Run prediction with batch processing
        results = []
        result_output_sizes = np.zeros(x_preprocessed.shape[0], dtype=np.int)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))

        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            # Call to DeepSpeech model for prediction
            with torch.no_grad():
                outputs, output_sizes = self._model(
                    inputs[begin:end].to(self._device), input_sizes[begin:end].to(self._device)
                )

            results.append(outputs)
            result_output_sizes[begin:end] = output_sizes.detach().cpu().numpy()

        # Aggregate results
        result_outputs = np.zeros(
            (x_preprocessed.shape[0], result_output_sizes.max(), results[0].shape[-1]), dtype=np.float32
        )

        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            # Overwrite results
            result_outputs[begin:end, : results[m].shape[1], : results[m].shape[-1]] = results[m].cpu().numpy()

        # Rearrange to the original order
        result_output_sizes_ = result_output_sizes.copy()
        result_outputs_ = result_outputs.copy()
        result_output_sizes[batch_idx] = result_output_sizes_
        result_outputs[batch_idx] = result_outputs_
        if np.isnan(result_outputs).any():
            logger.warning("NaN output encountered ; reloading model weights.")
            self.reload_model()
        # Check if users want transcription outputs
        transcription_output = kwargs.get("transcription_output")
        if transcription_output is None or transcription_output is False:
            return (result_outputs, result_output_sizes), None, None 
        
        # Now users want transcription outputs
        # Compute transcription

        decoded_output, offsets, scores = self.decoder.decode(
            torch.tensor(result_outputs, device=self._device), torch.tensor(result_output_sizes, device=self._device)
        )
        if not return_nbest:
            decoded_output = [do[0] for do in decoded_output]
            decoded_output = np.array(decoded_output)
            offsets = [ofs[0].cpu().numpy() for ofs in offsets]
            offsets = np.array(offsets)
            scores = [sc[0].cpu().numpy() for sc in scores]
            scores = np.array(scores)
        
        if return_alignments or return_confidence:
            results=[decoded_output] + ([offsets] if return_alignments else []) + ([scores] if return_confidence else [])
            return tuple(results)
        return decoded_output
    
    
    def loss_gradient(self,x: np.ndarray, y: np.ndarray,**kwargs):
        if self.niters_backward<=1:
            return super(SmoothedDeepSpeech,self).loss_gradient(x,y,**kwargs)
        else:
            random_gradients=[]
            if self.batch_backward>0 and x.shape[0]==1:
                for i in range(0,self.niters_backward,self.batch_backward):
                    batch_size = min(self.batch_forward,self.niters_forward-i)
                    x_batch = np.repeat(x,batch_size,axis=0)
                    grad=super(SmoothedDeepSpeech,self).loss_gradient(x_batch,y_batch,**kwargs)
                    random_gradients=random_gradients + [np.expand_dims(g,0) for g in grad]
            else:
                for i in range(self.niters_backward):
                    grad=super(SmoothedDeepSpeech,self).loss_gradient(x,y,**kwargs)
                    random_gradients.append(grad)
            result_gradient=self.estimate_input_gradient(random_gradients)
            return result_gradient

    def estimate_input_gradient(self,gradients_list):
        nsamples=len(gradients_list)
        gradient=1./nsamples*sum(gradients_list)
        return gradient


def get_art_model(model_kwargs, wrapper_kwargs, weights_path=None):
    return SmoothedDeepSpeech(**wrapper_kwargs)
