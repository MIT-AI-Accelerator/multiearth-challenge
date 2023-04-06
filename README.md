# multiearth-challenge
This repository holds a few simple tools for working with the imagery supplied by the [MultiEarth 2022 public challenge](https://sites.google.com/view/rainforest-challenge/home).

## Installation
Clone this repository and install:

``` shell
$ git clone https://github.com/MIT-AI-Accelerator/multiearth-challenge && cd multiearth-challenge
$ pip install .
```

#### Verifying Installation
To verify installation, ensure that [pytest](https://docs.pytest.org/en/latest/) is installed, then run `pytest` from the `multiearth-challenge` base directory.

## MultiEarth Utilities
### Filename parsing
The MultiEarth 2022 image files have encoded into their filename the collecting sensor's name, the sensor band, the latitude and longitude position of the image, and the collection date.  The function [parse_filename_parts](https://github.com/MIT-AI-Accelerator/multiearth-challenge/blob/4c8e358837d4492f562b63f92f98a3bafd8ff554/src/multiearth_challenge/data_tools.py#L9) will extract this metadata from the filename.

```
import multiearth_challenge.data_tools as dt

file_info = dt.parse_filename_parts("/path/to/data/Sentinel1_VH_-54.48_-3.33_2016_10_09.jpg")
print(file_info)
```
```
{'sensor': 'Sentinel1', 'band': 'VH', 'lat': -3.33, 'lon': -54.48, 'date': datetime.date(2016, 10, 9)}
```

### Aligned Imagery
One of the MultiEarth sub-challenges involves adapting SAR imagery to have the characteristics of a corresponding electro-optical (EO) image.  Many image translation algorithms make use of paired imagery. The function [get_sar_to_eo_aligned_images](https://github.com/MIT-AI-Accelerator/multiearth-challenge/blob/4c8e358837d4492f562b63f92f98a3bafd8ff554/src/multiearth_challenge/data_tools.py#L129) will identify Sentinel-2 EO satellite images that are geographically and temporally aligned with Sentinel-1 SAR satellite images.

```
import multiearth_challenge.data_tools as dt

# get_sar_to_eo_aligned_images requires the directory that holds the Sentinel-1 (sent1) and Sentinel-2 (sent2) image sub-directories.  
# This would be the directory that either of the MultiEarth Challenge supplied train or test zip files were unpacked in.
aligned_image_paths = dt.get_sar_to_eo_aligned_images("/path/to/data/")
```

## Managing Datasets
### Prediction Dataset
[ImagePredictionDataset](https://github.com/MIT-AI-Accelerator/multiearth-challenge/blob/4c8e358837d4492f562b63f92f98a3bafd8ff554/src/multiearth_challenge/datasets/prediction_dataset.py#L12) is a dataset designed for a predictive generative model. It stores MultiEarth data and can be used to retrieve a set of past images and a target image from a future date, all in the same location.
```
import multiearth_challenge.data_tools as dt
import multiearth_challenge.prediction_dataset as pd

# Use the first 10 images from the "SR_B4" band on Landsat-8 to form the source data, the target data will be all other images from those
# locations within 15 days
band = "SR_B4"
landsat8_images = dt.get_image_paths("/path/to/landsat8/data/", band)
src_images = landsat8_images[:10]
target_images = dt.get_aligned_images(src_images, landsat8_images[10:])

relevant_bands = {"Landsat-8": band}
dataset = pd.ImagePredictionDataset(src_images, target_images, relevant_bands, relevant_bands)
```
### ImageSegmentationDataset
[ImageSegmentationDataset](https://github.com/MIT-AI-Accelerator/multiearth-challenge/blob/4c8e358837d4492f562b63f92f98a3bafd8ff554/src/multiearth_challenge/datasets/segmentation_dataset.py#L12) is a dataset designed for an image segmentation model. It stores MultiEarth data and can be used to retrieve an image and its corresponding segmented image.
```
import multiearth_challenge.data_tools as dt
import multiearth_challenge.segmentation_dataset as sd

# Use the first 10 images from the "SR_B4" band on Landsat-8 to form the source data, the target data will be the corresponding burn data for those images
band = "SR_B4"
landsat8_images = dt.get_image_paths("/path/to/landsat8/data/", band)
burn_data = dt.get_image_paths("/path/to/burn/data/", "ConfidenceLevel")

src_images = landsat8_images[:10]
target_images = dt.get_aligned_images(src_images, burn_data, 0)

relevant_bands = {"Landsat-8": band}
dataset = sd.ImagePredictionDataset(src_images, target_images, relevant_bands)
```

### SARToVisibleDataset
[SARToVisibleDataset](https://github.com/MIT-AI-Accelerator/multiearth-challenge/blob/4c8e358837d4492f562b63f92f98a3bafd8ff554/src/multiearth_challenge/datasets/translation_dataset.py#L11) is a dataset designed for a model that generates EO images from SAR images. It stores MultiEarth data and can be used to retrieve a SAR image and its corresponding EO image.
```
import multiearth_challenge.data_tools as dt
import multiearth_challenge.translation_dataset as td

# Use the first 10 images from the "SR_B4" band on Landsat-8 to form the source data, the target data will be the corresponding SAR images from Sentinel-1
band = "SR_B4"
landsat8_images = dt.get_image_paths("/path/to/landsat8/data/", band)
sent1_images = dt.get_image_paths("/path/to/sentinel1/data/", "VV")

src_images = landsat8_images[:10]
target_images = dt.get_aligned_images(src_images, sent1_images, 0)

relevant_bands = {"Landsat-8": band}
dataset = td.ImagePredictionDataset(src_images, target_images)
```

## Copyright
MIT License

Copyright (c) 2022 USAF - MIT Artificial Intelligence Accelerator

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
