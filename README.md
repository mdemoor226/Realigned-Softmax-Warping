# Realigned-Softmax-Warping-For-Deep-Metric-Learning
## Image Retrieval
You will need to download the datasets (CUB, Cars, SOP) and place them into their proper datasets directory.

CUB:
https://www.vision.caltech.edu/datasets/cub_200_2011/

SOP:
https://cvgl.stanford.edu/projects/lifted_struct/

The Cars dataset link homepage has been removed. If you wish to download it you will need to find an alternative source.

To Run:
<ol>
  <li> Download the datasets</li>
  <li> Enter ./run to run the code (comment out any dataset you do not want to train on).</li>
</ol>

The config is already loaded with parameters. Feel free to make any changes. <br>
Cars and CUB can be trained using a simple GPU (e.g. 1080/2080 ti). However, for SOP we recommend something bigger (e.g. a V100) due to the increased memory
constraints. 

## Face Recognition
The Face Recognition code is built off of the ArcFace (InsightFace) repository (https://github.com/deepinsight/insightface). You will need to download the necessary datasets.
The insightface repository contains links to download some common ones: <br> 
https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_

We made use of MS1M-ArcFace.

To Run:
<ol>
  <li> Download the datasets and place in the data directory.</li>
  <li> Secure access to a cluster node (or two) of GPUs (we used A100s).</li>
  <li> Enter ./run to run the code on a single node of 4 GPUs (this script can be easily edited to fit your environment).</li>
  <li> Alternatively, the scripts folder contains scripts for running on 8 GPUs across 2 nodes.</li>
</ol>

To evaluate some trained weights simply run the respective python script (Eval.py). <br>
We contained a log file of the training run used for the results in the paper. <br>
Note: PartialFC is untested for this repository and may not work properly.

## Appendices
Code for Appendix G will be released in the future.

## Miscellaneous Details:

<ul>
  <li> Installing the dependencies for image retrieval should be fairly straightforward (PyTorch, TorchVision, Numpy etc.). For face recognition follow the Insightface repository. We included the environment.yml files for both of them in their respective subdirectories. </li>
  <li> An extensive random/grid search was used to obtain the Image retrieval hyperparameters. A grid search was used for face recognition. </li>
  <li> We recommend tuning the non-warp parameters (e.g. learning rates, temperature) with warping turned off. Then turn warping on when ready to find the warp parameters. </li>
  <li> One trick we used was to split the hyperparameter space between the regular and followup phases (each contains its set). After finding a suitable set.
for the regular training phase we would optimize the followup set using the checkpoint obtained from the regular training phase. </li>
  <li> The bounds in the followup phase are tighter than the first phase with alpha constrained to be lower. k1, k2, and alpha can be obtained from a grid search off the checkpoint mentioned above. Increaded Delta, if used, was not changed at all during all the training. </li> 
  <li> For face recognition, the peak performance on the standard datasets was reported in the paper (this was done for each loss). </li>
</ul>
