# multiearth-challenge
The Multimodal Learning for Earth and Environment Workshop (MultiEarth 2023) is the second annual CVPR workshop aimed at leveraging the significant amount of remote sensing data that is continuously being collected to aid in the monitoring and analysis of the health of Earth ecosystems. Detailed information, including public challenge data download links, can be found on the workshop’s webpage at https://sites.google.com/view/rainforest-challenge. The goal of the workshop is to bring together the Earth and environmental science communities as well as the multimodal representation learning communities to examine new ways to leverage technological advances in support of environmental monitoring. In addition, through a series of public challenges, the MultiEarth Workshop hopes to provide a common benchmark for remote sensing multimodal information processing. There are two main categories of public challenges as part of the MultiEarth Workshop

### 1. Rapid Detection of Environmental Change
There are three sub-challenges associated with this effort.  
**1A. Detection of deforestation**  
**1B. Detection of fire / burned regions**  
**1C. SAR to visible image translation; enables easily interpretable all-weather monitoring**  

### 2. Long-term Prediction of Environmental Trends
There are two sub-challenges associated with this effort.  
**1A. Prediction of deforestation**  
**1B. Prediction of collected imagery. An image generation task given historical data.**  

## Challenge Data
Multi-modal data collected from Landsat-5, Landsat-8, Sentinel-1, and Sentinel-2 along with deforestation and fire datasets are provided for use as part of these challenges at https://sites.google.com/view/rainforest-challenge. The data is provided in the form of NetCDF files as well as zipped tiff images. A small sample of the data formats is provided in this repository at ?????.

This repository holds tools for working with the large quantity of remote sensing data provided for these challenges.

## Installation
Clone the repository and install with pip by running

```shell
$ git clone https://github.com/MIT-AI-Accelerator/multiearth-challenge && cd multiearth-challenge
$ pip install .
```

This will automatically install the jupyter, matplotlib, netcdf4, numba, numpy, and xarray dependencies along with the package.

## Provided Dataset Classes
There are several dataset classes that are provided for loading MultiEarth data held in NetCDF formatted files and iterating through paired samples that are applicable to different types of challenge tasks. These paired samples will include imagery along with associated metadata such as collection date and sensor band.

### ImageSegmentationDataset
[ImageSegmentationDataset](https://github.com/MIT-AI-Accelerator/multiearth-challenge/blob/main/src/multiearth_challenge/datasets/segmentation_dataset.py) is a dataset designed for an image segmentation task, such as might be used in the detection of deforestation or burned region sub-challenges. It stores MultiEarth data and can be used to retrieve an image and its corresponding segmented image. Example usage and additional details can be found in the example_segmentation.ipynb notebook.

### SARToVisibleDataset
[SARToVisibleDataset](https://github.com/MIT-AI-Accelerator/multiearth-challenge/blob/main/src/multiearth_challenge/datasets/translation_dataset.py) is a dataset designed for sub-challenge involving the generation of an EO image from a SAR image. It stores MultiEarth data and can be used to retrieve a SAR image and its corresponding EO image. Example usage and additional details can be found in the example_translation.ipynb notebook.

### Prediction Dataset
[ImagePredictionDataset](https://github.com/MIT-AI-Accelerator/multiearth-challenge/blob/main/src/multiearth_challenge/datasets/prediction_dataset.py) is a dataset designed for the "Long-term Prediction of Environmental Trends" category of challenges. It stores MultiEarth data and can be used to retrieve a set of past images and a target image from a future date, all for the same location in the Amazon. Example usage and additional details can be found in the example_prediction.ipynb notebook.

## Zip File Utilities
In addition to providing the data in the form of NetCDF files, zip files in the same format as the data provided as part of the MultiEarth 2022 challenge are also being made available. This is the same data as that in the NetCDF files, but provided in an alternate format. Simple functions for parsing these files are provided in https://github.com/MIT-AI-Accelerator/multiearth-challenge/blob/main/src/multiearth_challenge/zip_file_tools.py. Example usage and additional details can be found in the example_zip_utils.ipynb notebook.


## Acknowledgement
Research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.


## Copyright
MIT License

Copyright (c) 2023 USAF - MIT Artificial Intelligence Accelerator

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
