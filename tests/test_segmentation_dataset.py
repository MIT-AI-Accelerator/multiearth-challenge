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
def data_files():
    return {
        "image_files": (
            pkg_resources.resource_filename("multiearth_challenge", "data/sample_dataset/landsat5.nc"),
            pkg_resources.resource_filename("multiearth_challenge", "data/sample_dataset/sentinel1.nc"),
        ),
        "deforestation_files": (
            pkg_resources.resource_filename("multiearth_challenge", "data/sample_dataset/deforestation.nc"),
        ),
    }

def test_segmentation_dataset_init(data_files):
    image_files = data_files["image_files"]
    deforestation_files = data_files["deforestation_files"]
    dataset = sd.SegmentationDataset(image_files, deforestation_files)
    assert len(dataset) == 3
    exp_locations = [
        [-4.11, -55.14],
        [-4.01, -54.80],
        [-3.35, -54.84],
    ]
    npt.assert_almost_equal(dataset.locations, exp_locations)

    filter = df.LocationFilter(min_lat=-3.34)
    with pytest.raises(ValueError):
        # Not data
        dataset = sd.SegmentationDataset(
            image_files,
            deforestation_files,
            image_data_filters=[filter],
            deforestation_data_filters=[filter]
        )
    
    dataset = sd.SegmentationDataset(
        image_files,
        deforestation_files,
        image_data_filters=[filter],
        deforestation_data_filters=[filter],
        error_on_empty=False,
    )
    assert len(dataset) == 0
    npt.assert_array_equal(dataset.locations, [])

    filter = df.LocationFilter(min_lat=-4.10)
    with pytest.raises(ValueError):
        # Mismatch in image and deforestation image locations
        dataset = sd.SegmentationDataset(
            image_files,
            deforestation_files,
            image_data_filters=[filter]
        )

    dataset = sd.SegmentationDataset(
        image_files,
        deforestation_files,
        image_data_filters=[filter],
        deforestation_data_filters=[filter],
    )        
    assert len(dataset) == 2
    exp_locations = [
        [-4.01, -54.80],
        [-3.35, -54.84],
    ]
    npt.assert_almost_equal(dataset.locations, exp_locations)


def test_segmentation_dataset_call(data_files):
    image_files = data_files["image_files"]
    deforestation_files = data_files["deforestation_files"]
    dataset = sd.SegmentationDataset(image_files, deforestation_files)
    with pytest.raises(ValueError):       
        # Shape mismatch between landsat5 and sent1 images
        res = dataset[0]

    data_transforms = [
        lambda xx : xx.astype(float),
        lambda xx : torch.from_numpy(xx),
        tv.transforms.Resize((128, 128))
    ]
    dataset = sd.SegmentationDataset(
        image_files,
        deforestation_files,
        image_data_transforms=data_transforms
    )
    assert len(dataset) == 3
    exp_image_samples = [30, 30, 40]
    exp_deforest_samples = [11, 11, 11]
    for sample, exp_image_smp, exp_deforest_smp in zip(dataset, exp_image_samples, exp_deforest_samples):
        assert len(sample) == 2
        images = sample[0]
        deforest = sample[1]
        assert images.ndim == 4
        assert images.shape[0] == exp_image_smp
        assert images.shape[1] == 1
        assert images.shape[2] == 128
        assert images.shape[3] == 128
        assert images.dtype == torch.float64

        assert deforest.ndim == 4
        assert deforest.shape[0] == exp_deforest_smp
        assert deforest.shape[1] == 1
        assert deforest.shape[2] == 256
        assert deforest.shape[3] == 256
        assert deforest.dtype == np.uint8

    dataset = sd.SegmentationDataset(
        image_files,
        deforestation_files,
        image_data_transforms=data_transforms,
        deforestation_data_transforms=data_transforms,
    )
    assert len(dataset) == 3
    exp_image_samples = [30, 30, 40]
    exp_deforest_samples = [11, 11, 11]
    for sample, exp_image_smp, exp_deforest_smp in zip(dataset, exp_image_samples, exp_deforest_samples):
        assert len(sample) == 2
        images = sample[0]
        deforest = sample[1]
        assert images.ndim == 4
        assert images.shape[0] == exp_image_smp
        assert images.shape[1] == 1
        assert images.shape[2] == 128
        assert images.shape[3] == 128
        assert images.dtype == torch.float64

        assert deforest.ndim == 4
        assert deforest.shape[0] == exp_deforest_smp
        assert deforest.shape[1] == 1
        assert deforest.shape[2] == 128
        assert deforest.shape[3] == 128
        assert deforest.dtype == torch.float64

    filters = [df.LocationFilter(min_lat=-4.10)]
    dataset = sd.SegmentationDataset(
        image_files,
        deforestation_files,
        image_data_transforms=data_transforms,
        deforestation_data_transforms=data_transforms,
        image_data_filters=filters,
        deforestation_data_filters=filters,
    )
    assert len(dataset) == 2
    exp_image_samples = [30, 40]
    exp_deforest_samples = [11, 11]
    for sample, exp_image_smp, exp_deforest_smp in zip(dataset, exp_image_samples, exp_deforest_samples):
        assert len(sample) == 2
        images = sample[0]
        deforest = sample[1]
        assert images.ndim == 4
        assert images.shape[0] == exp_image_smp
        assert images.shape[1] == 1
        assert images.shape[2] == 128
        assert images.shape[3] == 128
        assert images.dtype == torch.float64

        assert deforest.ndim == 4
        assert deforest.shape[0] == exp_deforest_smp
        assert deforest.shape[1] == 1
        assert deforest.shape[2] == 128
        assert deforest.shape[3] == 128
        assert deforest.dtype == torch.float64
        
