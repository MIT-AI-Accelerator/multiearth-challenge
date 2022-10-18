import pkg_resources

import numpy as np
import numpy.testing as npt
import pytest
import torch
import torchvision as tv
import xarray as xr

import multiearth_challenge.datasets.data_filters as df
import multiearth_challenge.datasets.segmentation_dataset as sd


@pytest.fixture
def data_file():
    return pkg_resources.resource_filename("multiearth_challenge", "data/sample_dataset/landsat5.nc")


def test_netcdf_dataset_init(data_file):
    data_filters = []
    data_transforms = []
    dataset = sd._NetCDFDataset(data_file, data_filters, data_transforms)
    assert len(dataset) == 3
    exp_locations = [
        [-4.11, -55.14],
        [-4.01, -54.80],
        [-3.35, -54.84],
    ]
    npt.assert_almost_equal(dataset.locations, exp_locations)

    filter = df.LocationFilter(min_lat=-3.34)
    dataset = sd._NetCDFDataset(data_file, [filter], data_transforms)
    assert len(dataset) == 0
    npt.assert_array_equal(dataset.locations, [])

    filter = df.LocationFilter(min_lat=-4.10)
    dataset = sd._NetCDFDataset(data_file, [filter], data_transforms)
    assert len(dataset) == 2
    exp_locations = [
        [-4.01, -54.80],
        [-3.35, -54.84],
    ]
    npt.assert_almost_equal(dataset.locations, exp_locations)
    

def test_netcdf_dataset_call(data_file):
    data_filters = []
    data_transforms = []
    dataset = sd._NetCDFDataset(data_file, data_filters, data_transforms)
    image_count = 0
    for images in dataset:
        image_count += len(images)
        assert images.ndim == 4
        assert images.shape[1] == 1
        assert images.shape[2] == 85
        assert images.shape[3] == 85
        assert images.dtype == np.uint16
    assert image_count == 80

    filter = df.LocationFilter(min_lat=-4.10)
    data_transforms = [lambda xx : xx.astype(float), lambda xx : torch.from_numpy(xx), tv.transforms.Resize((128, 128))]
    dataset = sd._NetCDFDataset(data_file, [filter], data_transforms)
    image_count = 0
    for images in dataset:
        image_count += len(images)
        assert images.ndim == 4
        assert images.shape[1] == 1
        assert images.shape[2] == 128
        assert images.shape[3] == 128
        assert images.dtype == torch.float64
    assert image_count == 56
    
