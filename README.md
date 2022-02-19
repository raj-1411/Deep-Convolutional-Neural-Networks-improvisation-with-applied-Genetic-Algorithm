# Deep-Convolutional-Neural-Networks-improvisation-with-applied-Genetic-Algorithm
## Project Summary
Performed feature extraction on `pre-trained` and `fine-tuned` Convolution Neural Networks for `enhanced` accuracy on dataset of BreakHis through application of `novel Genetic Algorithm`

## Project Description
This is a python-based project with the motivation of `enhanced` accuracy on predictions made by highly complex and `state-of-the-art` CNNs with the use of `Genetic Algorithm`. For this project popular `BreakHis` dataset was employed to verify feature extraction, classifying between benign and malignant types of tumors detected under histopathological images of affected tissue. For feature extraction, three Convolution Neural Networks are used with their `pre-final` layer modified for the need of the objective. As the models get trained and evaluated on different datasets, the features with the best accuracy on validation set is retained for further operation. The final features undergo different levels in Genetic Algorithm where the `non-redundant` and efficient features are selected for prediction. These filtered features procure significant increment in accuracy on the validation dataset.

## Dataset description
The Breast Cancer Histopathological Image Classification (BreakHis) is  composed of `9,109` microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X).  To date, it contains `2,480`  benign and `5,429` malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format).  
The dataset is available at:    
https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

## Classes of Division
In this project, the histopathological image samples of human breast tissue has been classified into two categories, namely:  
- `Benign`  
- `Malignant`  

## Convolution Neural Network models used
Three CNN models may be applied one at a time on the dataset for feature extraction, namely:  
-	`Visual Geometry Group (VGG-19)`  
-	`ResNet-18`  
-	`GoogLeNet` 

## Gentic Algorithm Visualization:
- To be added

## Accuracy Plots Over Generations
Different extractors paired with MLP classifer for GA gives three plots of accuracy vs generations:
Epoch-`10`
Generations-`10`
-     GoogLeNet with MLP
     ![image](https://user-images.githubusercontent.com/89198752/154793527-b9dc5c33-5c7b-494e-bf51-31b6909852a6.png)
-     VGG-19 with MLP
     ![image](https://user-images.githubusercontent.com/89198752/154793609-fe21f00b-5b80-42dc-a2e4-6fedbdb05c09.png)
-     ResNet-18 with MLP
     ![image](https://user-images.githubusercontent.com/89198752/154793637-f36ce72c-6483-4755-9d04-622327210d48.png)

## Dependencies
Since the entire project is based on `Python` programming language, it is necessary to have Python installed in the system. It is recommended to use Python with version `>=3.9`.
The Python packages which are in use in this project are  `matplotlib`, `numpy`, `pandas`,`scikit-learn`, `torch` and `torchvision`. All these dependencies can be installed just by the following command line argument
- pip install `requirements.txt`

## Code implementation
 ### Data paths :
      Current directory ---->   data
                                  |
                                  |
                                  |               
                                  ------------------>  train
                                  |                      |
                                  |             -------------------------
                                  |             |        |              |
                                  |             V        V              V
                                  |          class 1   class 2 ..... class n
                                  |
                                  |
                                  |              
                                  ------------------>   val
                                                         |
                                                -------------------------
                                                |        |              |
                                                V        V              V
                                             class 1   class 2 ..... class n
                                              
                               
- Where the folders `train` and `val` contain the folders for different classes of histopathological images of respective type of breast tissue tumor in `.jpg`/`.png` format.

 ### Training and Evaluating different CNN models :
      usage: main.py [-h] [-data DATA_PATH] [-classes NUM_CLASSES] [-ext EXT_TYPE] [-classif CLASSIF_TYPE]

      Application of Genetic Algorithm

      optional arguments:
        -h, --help            show this help message and exit
        -data DATA_PATH, --data_path DATA_PATH
                              Path to data
        -classes NUM_CLASSES, --num_classes NUM_CLASSES
                              Number of data classes
        -ext EXT_TYPE, --ext_type EXT_TYPE
                              Choice of extractor
        -classif CLASSIF_TYPE, --classif_type CLASSIF_TYPE
                              Choice of classifier for GA
        
  ### Run the following for training and validation :
  
      `python main.py -data data -classes 2 -ext resnet -classif MLP`
      
  ### Specific tokens :

          GoogLeNet: 'googlenet'
          VGG-19: 'vgg'
          ResNet-18: 'resnet'
          SVM: 'SVM'
          MLP: 'MLP'
          KNN: 'KNN'          
