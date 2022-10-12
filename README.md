# ORBITRON Team
# A Subspace Approach for the ORBIT Challenge

This code repo is provided as a part of the [ORBIT few-shot recognition challenge](https://eval.ai/web/challenges/challenge-page/1438/leaderboard/3580).
The method that we use here is inspired by the [subspace approach](https://github.com/chrysts/dsn_fewshot) with some modification.

In this work, we use PyTorch 1.8.1+ and Python 3.7.

# Installation

1. Clone or download this repository
2. Install dependencies
   ```
   cd ORBIT-Dataset

   # if using Anaconda
   conda env create -f environment.yml
   conda activate orbit-dataset

   # if using pip
   # pip install -r requirements.txt
   ```


# Download ORBIT Benchmark Dataset


The following script downloads the benchmark dataset into a folder called `orbit_benchmark_<FRAME_SIZE>` at the path `folder/to/save/dataset`. Use `FRAME_SIZE=224` to download the dataset already re-sized to 224x224 frames. For other values of `FRAME_SIZE`, the script will dynamically re-size the frames accordingly:
```
bash scripts/download_benchmark_dataset.sh folder/to/save/dataset FRAME_SIZE
```

Alternatively, the 224x224 train/validation/test ZIPs can be manually downloaded [here](https://city.figshare.com/articles/dataset/_/14294597). Each should be unzipped as a separate train/validation/test folder into `folder/to/save/dataset/orbit_benchmark_224`. The full-size (1080x1080) ZIPs can also be manually downloaded and `scripts/resize_videos.py` can be used to re-size the frames if needed.
   
The following script summarizes the dataset statistics:
```
python3 scripts/summarize_dataset.py --data_path path/to/save/dataset/orbit_benchmark_<FRAME_SIZE> --with_modes 
# to aggregate stats across train, validation, and test collectors, add --combine_modes
```
The Jupyter notebook `scripts/plot_dataset.ipynb` can be used to plot bar charts summarizing the dataset (uses Plotly).



## Subspace Classifier (CORE ALGORITHM)
Implementation of the model-based few-shot learner using the subspace method, inspired by [DSN](https://openaccess.thecvf.com/content_CVPR_2020/html/Simon_Adaptive_Subspaces_for_Few-Shot_Learning_CVPR_2020_paper.html) (Simon et al., _CVPR 2020_). 

Given video frames for each class as the context (in the support set), we create a subspace for each class as a classifier for target video frames. The rationale of this approach is that the object of the user's interest is located in a few frames and these frames can be represented by a few principal components. In addition, the subspace method produces a higher-order representation for classification meaning that the representation is more expressive compared to the prototypical solution.

In our setup, we replace the classifiers using subspaces while the CNN backbone remains unmodified ([Code Ref](https://github.com/chrysts/ORBITRON_Team_ORBIT_Challenge/blob/main/models/classifiers.py#L249)). 
The feature extractor initially worked only for a GPU, we extend it to use DataParallel to distribute memory usage among GPUs ([Code Ref](https://github.com/chrysts/ORBITRON_Team_ORBIT_Challenge/blob/main/models/few_shot_recognisers.py#L74)).
During training we only update the CNN backbone and there is no heavy data augmentation involved.  
Our method has a hyper-parameter which is the subspace dimension. In our setup, the subspace dimension is the number of images per class (K) - 1.

**Our setup** is run with 224x224 frames and an EfficientNet-B0 feature extractor. It is trained on 4 x 16GB GPUs (Tesla P100).

To highlight the benefit of using our approach:
- It is not required to update the model parameters in testing, thus the inference time is quick.
- The computational time of our approach might take a slightly longer time compared to the prototypical solution, but this weakness can be neglected with the trade-off of high performance gain using our method. 
- Even though there are some bad performances (< 0.5) for some users (see the table at the bottom), but the overall performance is remarkable for all users.

In our experiments, our method can achieve ~9% improvement over the best baseline performance. 

The empirical result of our method is shown below:

```
Frame Accuracy (avg): 0.740
Video Accuracy (avg): 0.760
Frames to Recognition (avg): 0.176
```

For a complete description to the method that we use, please see our presentation:

The video of our approach can be viewed at [https://www.youtube.com/watch?v=yfc7qI83eFY](https://www.youtube.com/watch?v=yfc7qI83eFY)


Run our method using 4 GPUs (TRAINING):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 single-step-learner.py --data_path folder/to/save/dataset/orbit_benchmark_224 --frame_size 224 \
                         --feature_extractor efficientnetb0 --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                         --classifier subspace --adapt_features  --learn_extractor \
                         --context_video_type clean --target_video_type clutter \
                         --with_lite --num_lite_samples 16 --batch_size 60 --epochs 35 --learning_rate 0.0002 
```

To summarize and plot the unfiltered dataset, use `scripts/summarize_dataset.py` and `scripts/plot_dataset.ipynb` similar to above.

Grab the best performing model, then generate a JSON File for EVALUATION:
```
CUDA_VISIBLE_DEVICES=0 python3 test_to_json.py --data_path folder/to/save/dataset/orbit_benchmark_224 --checkpoint_path checkpoint/best.pt 
```


More results for each test user:

```
         	Frame Accuracy	Video Accuracy	Frames to Recognition
user P177 (1/17)	0.9804 (0.0356)	1.0000 (0.0000)	0.0103 (0.0186)
user P198 (2/17)	0.7064 (0.2332)	0.7000 (0.2840)	0.1292 (0.1836)
user P204 (3/17)	0.6823 (0.0889)	0.7079 (0.0945)	0.2586 (0.0887)
user P233 (4/17)	0.3212 (0.1466)	0.4167 (0.2789)	0.1783 (0.1575)
user P271 (5/17)	0.8108 (0.1778)	0.8462 (0.1961)	0.0620 (0.0906)
user P421 (6/17)	1.0000 (0.0000)	1.0000 (0.0000)	0.0000 (0.0000)
user P452 (7/17)	0.7260 (0.2393)	0.7500 (0.2450)	0.1769 (0.2087)
user P455 (8/17)	0.5062 (0.2738)	0.5455 (0.2943)	0.4653 (0.2860)
user P485 (9/17)	0.9902 (0.0187)	1.0000 (0.0000)	0.0000 (0.0000)
user P554 (10/17)	0.6743 (0.2487)	0.7000 (0.2840)	0.1823 (0.2033)
user P609 (11/17)	0.8051 (0.2417)	0.8000 (0.2479)	0.1301 (0.1881)
user P642 (12/17)	0.7516 (0.2262)	0.7692 (0.2290)	0.2367 (0.2275)
user P753 (13/17)	0.6903 (0.1648)	0.7200 (0.1760)	0.2447 (0.1666)
user P900 (14/17)	0.5506 (0.2895)	0.5000 (0.3465)	0.1658 (0.2057)
user P901 (15/17)	0.8298 (0.1195)	0.8276 (0.1375)	0.1175 (0.1058)
user P953 (16/17)	0.9707 (0.0452)	1.0000 (0.0000)	0.0059 (0.0110)
user P999 (17/17)	0.8169 (0.1548)	0.8182 (0.2279)	0.1005 (0.0986)
User avg	0.7537 (0.0846)	0.7707 (0.0806)	0.1449 (0.0550)
Video avg	0.7402 (0.0444)	0.7600 (0.0483)	0.1760 (0.0405)

```

Please kindly cite our work if you use this code implementation and compare the performance with ours.

```
@inproceedings{simon2020dsn,
        author       = {C. Simon}, {P. Koniusz}, {R. Nock}, and {M. Harandi}
        title        = {Adaptive Subspaces for Few-Shot Learning},
        booktitle    = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
        year         = 2020
        }
```        
