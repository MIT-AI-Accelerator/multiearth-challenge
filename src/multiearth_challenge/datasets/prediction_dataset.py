from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np
import torch

import multiearth_challenge.base_datasets as bd
import multiearth_challenge.data_filters as df
import multiearth_challenge.relative_data_filters as rdf


class ImagePredictionDataset(MultiEarthDatasetBase):
    """A dataset class intended to be used for supervised learning of a
    predictive generative model. It loads MultiEarth data and returns
    samples consisting of a multiple historical source images paired
    with a target image with a future collection date at the same
    location..

    """

    def __init__(
        self,
        source_files: Iterable[Path],
        target_files: Iterable[Path],
        source_bands: Dict[str, Optional[Iterable[str]]],
        target_bands: Dict[str, Optional[Iterable[str]]],
        source_date_window: Tuple[float, float],
        max_source_cloud_coverage: float = 0.0,
        max_target_cloud_coverage: float = 0.0,
        source_image_transforms: Iterable[Callable[[Any], Any]] = [],
        target_image_transforms: Iterable[Callable[[Any], Any]] = [],
        merge_rgb: bool = False,
        merge_vv_vh: bool = False,
        error_on_empty: bool = True,
    ):
        """Note, this initialization will open file handles to a NetCDF
        file. These file handles are released by calling the close
        member function of this class.

        Parameters
        ----------
        source_files: Iterable[Path]
            MultiEarth image NetCDF files whose data should be
            considered for this dataset.

        target_files: Iterable[Path]
            MultiEarth NetCDF files whose images are truth targets for
            this dataset.

        source_bands: Dict[str, Optional[Iterable[str]]]
            A dictionary with keys specifying the sensor name and
            values specifying the bands of interest for source
            imagery. If the value is None, no filtering is done for
            that sensor's band and all associated bands are
            included. Acceptable sensor names and associated bands are
            as follows:

            "Landsat-5": ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'ST_B6', 'SR_B7', 'QA_PIXEL'],
            "Landsat-8": ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL'],
            "Sentinel-1": ['VV', 'VH'],
            "Sentinel-2": ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'QA60'],

            Additional documentation on the details of each data band
            can be found in the NetCDF data files or on the sensors'
            datasheets.

        target_bands: Dict[str, Optional[Iterable[str]]]
            A dictionary with keys specifying the sensor name and
            values specifying the bands of interest for target
            imagery. If the value is None, no filtering is done for
            that sensor's band and all associated bands are
            included. Acceptable sensor names and associated bands are
            as follows:

            "Landsat-5": ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'ST_B6', 'SR_B7', 'QA_PIXEL'],
            "Landsat-8": ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL'],
            "Sentinel-1": ['VV', 'VH'],
            "Sentinel-2": ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'QA60'],

            Additional documentation on the details of each data band
            can be found in the NetCDF data files or on the sensors'
            datasheets.

        source_date_window: Tuple[Optional[float], Optional[float]], default=(0, 30)
            The minimum and non-imclusive maximum relative time window
            in days around the segmentation image from which source
            imagery is pulled.  For example, with the default of (0,
            30) only source imagery within the interval of 0 days
            before and 29 days after a segmentation image date will be
            returned as source imagery. If the minimum is None, there
            is no filter on the minimum relative date. Similarly, no
            maximum can be specified with a value of None.

        max_source_cloud_coverage: float default=0.0
            The maximum allowable cloud coverage allowed in the source
            imagery as a fraction [0, 1]. Setting the maximum above 0
            may be useful to incorporate additional samples even if
            the source image is slightly obscured.

        max_target_cloud_coverage: float default=0.0
            The maximum allowable cloud coverage allowed in the target
            imagery as a fraction [0, 1]. Setting the maximum above 0
            may be useful to incorporate additional samples even if
            the target image is slightly obscured.

        source_image_transforms: Iterable[Callable[[Any], Any]], default=[],
            A series of callables which are each passed an source
            image and return a transformed image. These may be
            standard Pytorch Torchvision transforms or any function
            with a compatible signature.

        target_image_transforms: Iterable[Callable[[Any], Any]], default=[],
            A series of callables which are each passed a target image
            and return a transformed image. These may be standard
            Pytorch Torchvision transforms or any function with a
            compatible signature.

        merge_rgb: bool, default=False
            If True, when retrieving images from sensors that collect
            visible imagery, merges separate red, green, and blue
            channels into a single image.

            Note, this will only stack the 3 visible bands in red,
            green, blue order and will not perform any color balancing
            across the channels.

            If true and all 3 visible bands are not available due to a
            data filter, this parameter raises a ValueError.

            This parameter is ignored for sensors that do not collect
            visible imagery.

        merge_vv_vh: bool, default=False
            If True, when retrieving SAR images, merges separate VV
            and VH polarization channels into a single two channel
            image.

            Note, this will only stack the 2 bands in VV, VH order and
            will not perform any normalization across the channels.

            If true and both polarization bands are not available due
            to a data filter, this parameter raises a ValueError.

            This parameter is ignored for sensors that do not collect
            SAR imagery.

        error_on_empty: bool, default=True
            If True, if no source or target image remain after
            data filtering, raise a ValueError, otherwise this dataset
            will have length 0.

        """

        if len(source_date_window) != 2:
            raise ValueError(
                f"Parameter source_date_window should be a tuple of length 2, not length ({len(source_date_window)})"
            )
        relative_date_filter = rdf.RelativeDateFilter(
            relative_min_date=source_date_window[0],
            relative_max_date=source_date_window[1],
            include=True,
        )

        source_data_filters = [
            # Cloud filtering must come before the QA bands may be
            # filtered out
            df.CloudCoverFilter(
                min_cover_fraction=0.0,
                max_cover_fraction=max_source_cloud_coverage,
                filter_shadow_landsat=False,
                include=True,
            ),
            df.DataBandFilter(source_bands, include=True),
        ]

        target_data_filters = [
            # Cloud filtering must come before the QA bands may be
            # filtered out
            df.CloudCoverFilter(
                min_cover_fraction=0.0,
                max_cover_fraction=max_target_cloud_coverage,
                filter_shadow_landsat=False,
                include=True,
            ),
            df.DataBandFilter(target_bands, include=True),
        ]

        self.super().__init__(
            source_files,
            target_files,
            source_data_filters,
            target_data_filters,
            source_image_transforms,
            target_image_transforms,
            relative_date_filter,
            merge_rgb,
            merge_vv_vh,
            error_on_empty,
            single_source_image=False,
        )
