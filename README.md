# AKMNet-Micro-Expression
Recognizing Micro-Expression in Video Clip  with Adaptive Key-Frame Mining
# Purpose
  The existing representation based on various deep learning techniques learned from a full video clip is usually redundant. In addition, methods utilizing the single apex frame of each video clip require expert annotations and sacrifice the temporal dynamics. In our paper, we propose a novel end-to-end deep learning architecture, referred to as adaptive key-frame mining network (AKMNet). Operating on the video clip of micro-expression, AKMNet is able to learn discriminative spatio-temporal representation by combining spatial features of self-learned local key frames and their global-temporal dynamics  



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
|    *Name*         |*# Params*|*Top-1 Acc.*|
|:-----------------:|:--------:|:----------:|
| `efficientnet-b0` |   5.3M   |    76.3    |
| `efficientnet-b1` |   7.8M   |    78.8    |
| `efficientnet-b2` |   9.2M   |    79.8    |
| `efficientnet-b3` |    12M   |    81.1    |
| `efficientnet-b4` |    19M   |    82.6    |
| `efficientnet-b5` |    30M   |    83.3    |
| `efficientnet-b6` |    43M   |    84.0    |
| `efficientnet-b7` |    66M   |    84.4    |

|    *Methods*         |*CASMEI*|*CASMEII*|*SMIC*|*SAMM*|
|:-----------------:|:--------:|:----------:|:----------:|:----------:|
| # LBP-TOP                     |   0.6618  |   0.3843   |   0.3598  |  0.3899  |
| # LBP-SIP                    |   0.6026  |   0.4784   |   0.3842  |  0.5220  |
| # STCLQP                     |   0.6349  |   0.5922   |   0.5366  |  0.5283  |
| # HIGO                       |   0.5781  |   0.5137   |   0.3720  |  0.4465  |
| # FHOFO                      |   0.6720  |   0.6471   |   0.5366  |  0.6038  |
| # MDMO                       |   0.6825  |   0.6314   |   0.5793  |  0.6164  |
|------------------------------|-----------|------------|-----------|----------|
| # Macro2Micro                |   0.6772  |   0.6078   |     -     |  0.6436  |
| # MicroAttention             |   0.6825  |   0.6431   |     -     |  0.6489  |
| # ATNet                      |   0.6720  |   0.6039   |     -     |  0.6543  |
| # STSTNet                    |   0.6349  |   0.5529   |   0.5488  |  0.6289  |
| # STRCN-G                    |   0.7090  |   0.6039   |   0.6280  |  0.6478  |
|------------------------------|-----------|------------|-----------|----------|
| # **AKMNet**                 |**0.7566** |**0.6706**  |**0.7256** |**0.7170**|

* Justification of the Adaptive Key-Frame Mining Module
  ![image](https://github.com/Trunpm/AKMNet-Micro-Expression/blob/main/docs/Table_IV.jpg) 
  RECOGNITION RESULTS (%) ON CASME I/II, SMIC, AND SAMM DATASETS IN THE COMPARISON EXPERIMENT

* Ablation Experiment
  ![image](https://github.com/Trunpm/AKMNet-Micro-Expression/blob/main/docs/Table_V.jpg) 
  RECOGNITION RESULTS (%) ON CASME I/II, SMIC, AND SAMM DATASETS IN THE ABLATION EXPERIMENT

* Annotated Apex Frame VS ‘Most Informative’ Frame
  ![image](https://github.com/Trunpm/AKMNet-Micro-Expression/blob/main/docs/Table_VII.jpg) 
  RECOGNITION RESULTS (%) OF MODELS WITH INPUT OF APEX  FRAMES AND MAX-KEY FRAMES, RESPECTIVELY

