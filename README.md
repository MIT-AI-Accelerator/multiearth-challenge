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

## Aligned Imagery
One of the MultiEarth sub-challenges involves adapting SAR imagery to have the characteristics of a corresponding electro-optical (EO) image.  Many image translation algorithms make use of paired imagery. The function [get_sar_to_eo_aligned_images](https://github.com/MIT-AI-Accelerator/multiearth-challenge/blob/4c8e358837d4492f562b63f92f98a3bafd8ff554/src/multiearth_challenge/data_tools.py#L129) will identify Sentinel-2 EO satellite images that are geographically and temporally aligned with Sentinel-1 SAR satellite images.

```
import multiearth_challenge.data_tools as dt

# get_sar_to_eo_aligned_images requires the directory that holds the Sentinel-1 (sent1) and Sentinel-2 (sent2) image sub-directories.  
# This would be the directory that either of the MultiEarth Challenge supplied train or test zip files were unpacked in.
aligned_image_paths = dt.get_sar_to_eo_aligned_images("/path/to/data/")
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
