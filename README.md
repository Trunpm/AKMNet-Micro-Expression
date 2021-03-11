# AKMNet for Micro-Expression
Recognizing Micro-Expression in Video Clip  with Adaptive Key-Frame Mining

![alt text](docs/akmnetoverview.png 'overview of the network')

# Purpose
The existing representation based on various deep learning techniques learned from a full video clip is usually redundant. In addition, methods utilizing the single apex frame of each video clip require expert annotations and sacrifice the temporal dynamics. In our paper, we propose a novel end-to-end deep learning architecture, referred to as adaptive key-frame mining network (AKMNet). Operating on the video clip of micro-expression, AKMNet is able to learn discriminative spatio-temporal representation by combining spatial features of self-learned local key frames and their global-temporal dynamics.

# Citation
Peng, Min, Chongyang Wang, Yuan Gao, Tao Bi, Tong Chen, Yu Shi, and Xiang-Dong Zhou. "[Recognizing Micro-expression in Video Clip with Adaptive Key-frame Mining](https://arxiv.org/abs/2009.09179)", arXiv preprint arXiv:2009.09179 (2020).

# Platform and dependencies
Ubuntu 14.04  Python 3.7  CUDA8.0 CuDNN6.0+  
pytorch==1.0.0  

# Data Preparation (option)
* Download the dataset  
  CASMEI: http://fu.psych.ac.cn/CASME/casme-en.php   
  CASMEII: http://fu.psych.ac.cn/CASME/casme2-en.php  
  SAMM: http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php  
  SMIC: https://www.oulu.fi/cmvs/node/41319  
* preprocessing  
  1.you can also use the data in *cropped* fold to conduct the experiment. For SAMM dataset, the face detect and align method same as the paper *CASME II: An Improved Spontaneous Micro-Expression Database and the Baseline Evaluation*  
  2. for phase-based video magnification method, please ref to http://people.csail.mit.edu/nwadhwa/phase-video/  
  In our method, we do not need the frame normalization, the design of all modules in the AKMNet is independent on the length of input video clip  

# Method
![image](https://github.com/Trunpm/AKMNet-Micro-Expression/blob/main/docs/module.jpg)  

# Experiment
* Comparison Experiment  

|    *Methods*         |*CASMEI*|*CASMEII*|*SMIC*|*SAMM*|
|:-----------------:|:--------:|:----------:|:----------:|:----------:|
| `LBP-TOP`                   |   0.6618  |   0.3843   |   0.3598  |  0.3899  |
| `LBP-SIP`                    |   0.6026  |   0.4784   |   0.3842  |  0.5220  |
| `STCLQP`                     |   0.6349  |   0.5922   |   0.5366  |  0.5283  |
| `HIGO`                       |   0.5781  |   0.5137   |   0.3720  |  0.4465  |
| `FHOFO `                     |   0.6720  |   0.6471   |   0.5366  |  0.6038  |
| `MDMO `                      |   0.6825  |   0.6314   |   0.5793  |  0.6164  |
| `Macro2Micro`               |   0.6772  |   0.6078   |     -     |  0.6436  |
| `MicroAttention`             |   0.6825  |   0.6431   |     -     |  0.6489  |
| `ATNet   `                   |   0.6720  |   0.6039   |     -     |  0.6543  |
| `STSTNet `                   |   0.6349  |   0.5529   |   0.5488  |  0.6289  |
| `STRCN-G`                    |   0.7090  |   0.6039   |   0.6280  |  0.6478  |
| # **AKMNet**                 |**0.7566** |**0.6706**  |**0.7256** |**0.7170**|

* Justification of the Adaptive Key-Frame Mining Module  

|    *Methods*         |*CASMEI*|*CASMEII*|*SMIC*|*SAMM*|
|:-----------------:|:--------:|:----------:|:----------:|:----------:|
| `AKMNetva-all`     |   0.6618  |   0.3843   |   0.3598  |  0.3899  |
| `AKMNetva-random` |0.6138 |0.6118 |0.5427 |0.6289  |
| `AKMNetva-norm16` |0.6667 |0.6314 |0.5976 |0.6604  |
| `AKMNetva-norm32` |0.6825 |0.6392 |0.6434 |0.6478  |
|  `AKMNetva-norm64` |0.7090 |0.6392 |0.6463 |0.6164  |
|  `AKMNetva-norm128` |0.6984 |0.6431 |0.6646 |0.6792  |
| `**AKMNet**`      |**0.7566** |**0.6706**  |**0.7256** |**0.7170**|

* Ablation Experiment  

|    *Methods*         |*CASMEI*|*CASMEII*|*SMIC*|*SAMM*|
|:-----------------:|:--------:|:----------:|:----------:|:----------:|
| `AKMNet-s12` |0.6984 |0.6392 |0.6463 |0.6667  |
| `AKMNet-s13` |0.7354 |0.6431 |0.6463 |0.6604  |
| `AKMNet-s23` |0.7249 |0.6549 |0.6707 |0.6918 |
| `AKMNet-s123` |0.7566 |0.6706 |0.7256| 0.7170  |

* Annotated Apex Frame VS ‘Most Informative’ Frame  

|    *Methods*         |*CASMEI*|*CASMEII*|*SMIC*|*SAMM*|
|:-----------------:|:--------:|:----------:|:----------:|:----------:|
| `Resnet18` |Apex frame |0.6772 |0.6078 |0.6436  |
| `Resnet18` |Max-key frame |0.6825| 0.6392| 0.6486  |
| `VGG-11` |Apex frame |0.6667 |0.6235| 0.6277 |
| `VGG-11` |May-key frame |0.6931 |0.6353 |0.6649  |

* how to use:
  for each LOSO exp:    
  first： set *list_file_train* and *list_file_test* in `main.py` properly, each of them is a list file, contents in file like this:  
  */home/XXX/fold/sub01/EP01_12__alpha15 19 3*  
  *...*  
  where */home/XXX/fold/sub01/EP01_12__alpha15* is a fold which contain a image sequence of a micro-expression, 19 is the len of the clips, 3 is the label  
  second: set *premodel* in `main.py` if you have the pretrained model  
  third: run `python main.py` in your terminal.  
  
 * 
  If you have questions, post them in GitHub issues.
