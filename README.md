# GARD Evaluation phase 2 ASR scenario.
Submission by Perspecta Labs/Carnegie Mellon University

## Description and code organization
The provided defense is a variation of randomized smoothing (https://arxiv.org/pdf/1902.02918.pdf) suited for Speech Recognition. Gaussian noise added to the inputs, and implemented in the `SpeechNoiseAugmentation` Preprocessor (controlled in the defense configuration) . Estimation from multiple predictions is controlled directly in the model configuration (see Postprocessing). On top of that, this repository's original contributions are as follows :
### Preprocessing
ASR models are highly susceptible to gaussian noise, so a preprocessing/speech enhancement step to "clean" inputs improves results significantly. We implemented several enhancement methods, such as : 
* Wiener filtering (https://ieeexplore.ieee.org/document/1163086)
* **A priori SNR (ASNR) filtering** (https://www.researchgate.net/publication/3644389_Speech_enhancement_based_on_a_priori_signal_to_noise_estimation) (by far the most effective)
* Speech Enhancement GAN (https://arxiv.org/abs/1703.09452)
These methods require the pyaudlib and segan external dependencies. The local implementation for filtering and gaussian noise can be found in the 'eval2/defenses' folder. The `filter`, `filter_kwargs`, `enhancer`, `enhancer_kwargs` attributes in the defense configuration control it. We recommend keeping the parameters in the provided configuration (asnr_wiener filter and no enhancer) as they yield the best results.

### Postprocessing
Estimating an output from multiple sentence predictions is not as trivial as it is for classification. We implemented several methods, controlled with the `voting_kwargs` argument of the `SmoothedDeepSpeech` class. The number of votes can be controlled with the `niters_forward` parameter. These predictions are by default ran sequentially. The user can choose to batch these forward passes if the dataset batch size is 1, by setting the "batch_forward" parameter to a value greater than 0. Voting strategies are :
* Usual majority vote between sentences, i.e. treating predictions as class labels. This method is limited by the high sparsity of noisy speech transcriptions and the high computation time required to output hundreds or thousands of predictions to mitigate that sparsity. Use `voting="majority"`
* Taking the average or the frame-wise max of the output probabilites prior to transcription decoding. Averaging does improve results compared to majority vote. Use `voting="probs_sum"` or `"probs_max"` in that case
* Using the ROVER post-processing method (http://my.fit.edu/~vkepuska/ece5527/sctk-2.3-rc1/doc/rover/rover.htm) from the NIST scoriing toolkit. This advanced voting method was popular in previous ASR work to ensemble weak models. We propose a novel use of that method for smoothing noisy inputs passing throught the same models. The method computes an optimal mix of all output sentences using criteria such as word frequency (`voting="rover_freq"`) or word confidence (`"rover_conf"` and `"rover_max"`). It is difficult to obtain word scores from deepspeech, so we used extensively the "rover_freq" paradigm and recommend evaluators do the same.

#### Implementation details
Voting is implemented in the `eval2.models.vote` file.
* *Rover implementation* : ROVER is not implemented in python. To use it, we first write output sentences in files, then call a bash command. This pipeline rely on the SCTK dependency being installed, for which using one of our docker images is required.
* *Maximum outputs* ROVER is powerful but unstable. Its computation time increases exponentially with the number of sentences, and it starts running into segmentation faults when using many sentences. We handle that last case by falling back on majority voting when it happens. However, the current implementation does not allow more than 50 predictions with one rover call. If the user wants to vote on more outputs with rover, the behavior will be a "divide and conquer" strategy, where outputs are split in 50 batches, votes are run for each batch, then run again on the batch outputs. We however recommend to stick to 50 predictions, as the WER improvement when using more is marginal.
* Rover uses alignments and sentence confidence to improve its results. Fortunately the `CTCBeamDecoder` class can return these values. To do so we wrote custom decoder wrappers in `eval2.models.decoder`, and control what the decoder outputs with the `use_alignments` and `use_confidence` parameters, which we recomment keeping to True in any case. Because the output format is different, ART attacks such as `imperceptible_asr` will run into errors. We wrote a custom version of that attack handling that case (see below)
* Since using `CTCBeamDecoder`, we can wonder if using a beam size greater than 1, and voting on all beams rather than just the best one, may improve results in any way. Our experiments show that it doesn't, and we therefore use a beam size of 1 (equivalent to greedy search). However, the user can change that with the `beam_width` and `vote_on_nbest` parameters

### Attacks
* The preprocessing step is non-differentiable. So far we have approximated it witht hee identity function for gradient purposes.
* To apply EoT on the postprocessing step, we rewrote the `loss_gradient` method to let the user average gradients on multiple steps, (`niters_backward` and `batch_backward` parameters). This works fine with the PGD attack, for instance.
* The `imperceptible_asr` attack was more tricky, as it does not use `loss_gradient`, decodes internally and does not handle numpy preprocessing steps. We rewrote this attack for our classifier in `eval2.attacks.imperceptible_asr` to handle those issues. The algorithm is otherwise identical.

### Pretrained models.
The above pipeline can be used on the default pretrained weights. However its results improve significantly when using models fine-tuned on noisy and filtered inputs. We trained such models using a forked Deepspeech repository to which we added gaussian noise and ASNR filtering support (https://github.com/RaphaelOlivier/deepspeech.pytorch). Unfortunately this repository uses Pytorch 1.6 and the weights are incompatible with Pytorch 1.4. Besides, when running many attacks on a speech input in pytorch 1.6 we observe NaN issues, like other teams. We sadly do not have a fix for that problem so far. The `smoothing_imperceptible.json` config seems in practice to encounter a NaN issue for ~1/4 inputs, the `smoothing_pgd.json` config for ~1/20. These inputs end up artificially degrading the adversarial WER that Armory reports.

Therefore we propose two docker images
* One uses Pytorch 1.4 and can only be used with the default model weights. The `smoothing_pgd_pt14.json` file uses it.
* One uses Pytorch 1.6 and can use any weights, but will encounter the aforementioned problem. To estimate accurately the performance of these models, we believe it necessary to manually remove all NaN-issuing inputs from metric computation.
