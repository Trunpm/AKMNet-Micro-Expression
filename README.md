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
  ![image](https://github.com/Trunpm/AKMNet-Micro-Expression/blob/main/docs/Table_III.jpg)  
  RECOGNITION RESULTS (%) ON CASME I/II, SMIC, AND SAMM DATASETS IN THE COMPARISON EXPERIMENT

* Justification of the Adaptive Key-Frame Mining Module
  ![image](https://github.com/Trunpm/AKMNet-Micro-Expression/blob/main/docs/Table_IV.jpg) 
  RECOGNITION RESULTS (%) ON CASME I/II, SMIC, AND SAMM DATASETS IN THE COMPARISON EXPERIMENT

* Ablation Experiment
  ![image](https://github.com/Trunpm/AKMNet-Micro-Expression/blob/main/docs/Table_V.jpg) 
  RECOGNITION RESULTS (%) ON CASME I/II, SMIC, AND SAMM DATASETS IN THE ABLATION EXPERIMENT

* Annotated Apex Frame VS ‘Most Informative’ Frame
  ![image](https://github.com/Trunpm/AKMNet-Micro-Expression/blob/main/docs/Table_VII.jpg) 
  RECOGNITION RESULTS (%) OF MODELS WITH INPUT OF APEX  FRAMES AND MAX-KEY FRAMES, RESPECTIVELY

