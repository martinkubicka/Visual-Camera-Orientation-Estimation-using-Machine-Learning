# Visual Camera Orientation Estimation using Machine Learning

## General information
This repository contains implementations for a bachelor's thesis: Visual Camera Orientation Estimation using Machine Learning created by Martin Kubička.

## Requirements
Scripts were implemented tested with ```Python 3.9``` and in case of training ```S2CNNModel``` ```Python 3.6``` must be used.

Required libraries can be found in ```requirements.txt```.

## Tree of contained files

```
.
├── README.md                         - Documentation and manual
├── requirements.txt                  - Required python dependencies
├── install                           - Script for installation
├── run                               - Script for executing programs
└── src                               - Scripts and programs
    ├── COESCNN                       - Models
    │   ├── S2CNNModel                - S2CNN model
    │   ├── SphereNetModel            - SphereNet model
    │   ├── SphereNetSegModel         - SphereNet segmentation model
    ├── datasetGenerator              - SPPAI dataset generator
    ├── preprocessing                 - Preprocessing
    │   ├── PAC                       - PAC dataset generator
    │   ├── Segmentation              - Segmentation datasets generator
    │   ├── equir2stereo              - Script for transforming equirectangular
    │   │                               to stereographic projection
    │   ├── geoPose                   - Equirectangular and stereographic
    │   │                               GeoPose dataset generator
    │   ├── lib                       - Library for creating cutouts
    │   └── utils                     - Utilities for dataset generators
    └── testing                       - Scripts for models testing and loading
```

## Setup

```
1. Install Python 3.9 or 3.6 in case you want to train S2CNN model (must be in PATH)
```

```
2. Install pip3 (example for ubuntu: sudo apt install python3-pip) (must be in PATH)
```

```
3. pip3 install virtualenv
```

```
4. chmod +x install
```
```
5. ./install
```

6. For NumPy and SciPy for GPU please install cupy:
(YZ represent version of your CUDA - for example cupy-cuda12x)
```
virtualenv venv
source venv/bin/activate
pip3 install cupy-cuda-YZx
```

7. For converting equirectangular projection to stereographic please have installed ```g++ compiler```.

8. If you want to generate SPPAI dataset setup .env file in src/datasetGenerator/
Example:
```
API_GOOGLE_KEY=YOUR_API_GOOGLE_KEY
API_MAPILLARY_CLIENT_TOKEN=YOUR_MAPILLARY_CLIENT_TOKEN
```
Note: For Google key you must have activated StreetView API on your account.

## Run
```
chmod +x run
```

Note: Preprocessing programs always include examples in the input folder set by default. However, when training models, it is necessary to upload a dataset or change the path to the dataset. For model testing (Model evaluation), you need to upload the trained model and the dataset on which it will be tested or change paths to trained model and dataset.
```
./run
```

```
Choose program
```
