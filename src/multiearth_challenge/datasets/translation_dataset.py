from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import torch

import multiearth_challenge.base_datasets as bd
import multiearth_challenge.relative_data_filters as rdf


class SARToVisibleDataset(MultiEarthDatasetBase):
    """A dataset class intended to be used for supervised learning of the
    translation of a SAR image to a visible image. It loads MultiEarth
    data and returns samples consisting of a target image along with
    associated imagery at the same location.

    """

    def __init__(
        self,
        sar_files: Iterable[Path],
        visible_files: Iterable[Path],
        sar_bands: Iterable[str] = ["VV"],
        sar_date_window: Tuple[float, float] = [-10, 10],
        sar_image_transforms: Iterable[Callable[[Any], Any]] = [],
        visible_bands: Dict[str, Iterable[str]] = bd.RGB_BANDS,
        visible_image_transforms: Iterable[Callable[[Any], Any]] = [],
        max_visible_cloud_coverage: float = 0.0,
        merge_rgb: bool = False,
        merge_vv_vh: bool = False,
        error_on_empty: bool = True,
    ):
        """Note, this initialization will open file handles to a NetCDF
        file. These file handles are released by calling the close
        member function of this class.

        Parameters
        ----------
        sar_files: Iterable[Path]
            MultiEarth image NetCDF files containing Sentinel-1 SAR
            images whose data should be considered for this dataset.

        visible_files: Iterable[Path]
            MultiEarth NetCDF files containing visible images which
            are truth targets for this dataset.

        sar_bands: Iterable[str], default=["VV"]
            A series specifying whether to include VV polarized
            imagery, VH polarized imagery, or both in this dataset.

        sar_date_window: Tuple[Optional[float], Optional[float]], default=(-10, 11)
            The minimum and non-imclusive maximum relative time window
            in days around the visible image from which SAR imagery is
            pulled.  For example, with the default of (10, 11) only
            SAR imagery within the interval of 10 days before and 10
            days after a visible image date will be returned. If the
            minimum is None, there is no filter on the minimum
            relative date. Similarly, no maximum can be specified with
            a value of None.

        sar_image_transforms: Iterable[Callable[[Any], Any], default=[]
            A series of callables which are each passed a SAR image
            and return a transformed image. These may be standard
            Pytorch Torchvision transforms or any function with a
            compatible signature.

        visible_bands: dict[str, Iterable[str]], default=bd.RGB_BANDS
            A dictionary with keys specifying the sensor name and
            values specifying the bands of interest. If the value is
            None, no filtering is done for that sensor's band and all
            associated bands are included. Acceptable sensor names and
            associated bands are defined in the default value which is
            dict(
                "Landsat-5": ['SR_B1', 'SR_B2', 'SR_B3'],
                "Landsat-8": ['SR_B2', 'SR_B3', 'SR_B4'],
                "Sentinel-2": ['B2', 'B3', 'B4'],
            )

            Additional documentation on the details of each data band
            can be found in the NetCDF data files or on the sensors'
            datasheets.

        visible_image_transforms: Iterable[Callable[[Any], Any]], default=[]
            A sereies of callables which are each passed a visible
            image and return a transformed image. These may be
            standard Pytorch Torchvision transforms or any function
            with a compatible signature.

        max_visible_cloud_coverage: float default=0.0
            The maximum allowable cloud coverage allowed in visible
            imagery as a fraction [0, 1]. Setting the maximum above 0
            may be useful to incorporate additional samples even if
            the truth visible image is slightly obscured.

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
        if len(sar_date_window) != 2:
            raise ValueError(
                f"Parameter sar_date_window should be a tuple of length 2, not length ({len(sar_date_window)})"
            )
        sar_data_filters = [
            df.DataBandFilter({"Sentinel-1": sar_bands}, include=True),
        ]
        relative_date_filter = rdf.RelativeDateFilter(
            relative_min_date=sar_date_window[0],
            relative_max_date=sar_date_window[1],
            include=True,
        )

        visible_data_filters = [
            # Cloud filtering must come before the QA bands may be
            # filtered out
            df.CloudCoverFilter(
                min_cover_fraction=0.0,
                max_cover_fraction=max_visible_cloud_coverage,
                filter_shadow_landsat=False,
                include=True,
            ),
            df.DataBandFilter(visible_bands, include=True),
        ]
        self.super().__init__(
            sar_files,
            visible_files,
            sar_data_filters,
            visible_data_filters,
            sar_image_transforms,
            visible_image_transforms,
            relative_date_filter,
            merge_rgb,
            merge_vv_vh,
            error_on_empty,
            single_source_image=True,
        )

    def __getitem__(self, index: int) -> Tuple[bd.DatasetData, bd.DatasetData]:
        """Returns paired SAR and visible data held by this dataset.

        This function extends the inherited __getitem__ function.

        Parameters
        ----------
        index: int
            The location index to retrieve imagery for.

        Returns
        -------
        Tuple[DatasetData, DatasetData]
        A tuple where the first element is a SAR image meant to serve
        as an input to a image translation modela along with
        associated metadata.

        The second element holds a visible image at the same location
        along with associated metadata. Each sample returned will be a
        unique pairing of data, but a SAR image may be paired with
        more than one target and vice-versa. The data will be filtered
        by any data filters and modified by any transforms that have
        been set.

        The DatasetData type is a dictionary that holds sample imagery
        as well as the image's collection date and the latitude /
        longitude of the center of the image. The dictionary has keys:

            "image": Any - The image of interest.

            "lat_lon": Tuple[float, float] - A tuple holding the
            latitude and longitude in decimal degrees for the center
            of the image.

           "date": np.datetime64 - The collection date for the image.

        The return type of each image will nominally be a numpy array
        with shape (C0, H0, W0) where C0 is the image channels, H0 the
        image height, and W0 the image width. This shape and return
        type may be modified by this dataset's data transforms. The
        bit depth of the imagery varies depending on the collecting
        sensor and applied data transforms. The paired target image
        will also nominally be a numpy array with dimensions (C1, H1,
        W1) that may be arbitrarily modified by specified data
        transforms.

        """
        source_data, target_data = super().__getitem__(index)
        assert len(source_data) == 1
        return (source_data[0], target_data)
