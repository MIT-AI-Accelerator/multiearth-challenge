from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import multiearth_challenge.base_datasets as bd
import multiearth_challenge.data_filters as df
import multiearth_challenge.relative_data_filters as rdf


class ImageSegmentationDataset(bd.MultiEarthDatasetBase):
    """A dataset class intended to be used for supervised learning of a
    image segmentation task. It loads MultiEarth data and returns
    samples consisting of a source image paired with a target
    segmentation image at the same location.

    """

    def __init__(
        self,
        source_files: Iterable[Path],
        segmentation_files: Iterable[Path],
        source_bands: Dict[str, Optional[Iterable[str]]],
        source_cloud_coverage: float = (0.0, 0.0),
        source_date_window: Tuple[Optional[float], Optional[float]] = (-10, 10),
        source_image_transforms: Iterable[Callable[[Any], Any]] = [],
        segmentation_image_transforms: Iterable[Callable[[Any], Any]] = [],
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

        segmentation_files: Iterable[Path]
            MultiEarth NetCDF files containing deforestation or fire
            segmentation images which are truth targets for this
            dataset.

        source_bands: Dict[str, Optional[Iterable[str]]]
            A dictionary with keys specifying the sensor name and
            values specifying the bands of interest. If the value is
            None, no filtering is done for that sensor's band and all
            associated bands are included. Acceptable sensor names and
            associated bands are as follows:

            "Landsat-5": ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'ST_B6', 'SR_B7', 'QA_PIXEL'],
            "Landsat-8": ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL'],
            "Sentinel-1": ['VV', 'VH'],
            "Sentinel-2": ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'QA60'],

            Additional documentation on the details of each data band
            can be found in the NetCDF data files or on the sensors'
            datasheets.

        source_cloud_coverage: Tuple[float, float], default=(0.0, 0.0)
            The minimum and maximum allowable cloud coverage allowed
            in visible and IR imagery as a fraction [0, 1]. Setting
            the maximum above 0 may be useful when incorporating SAR
            imagery into a multimodal model where a large fraction of
            cloud coverage may be acceptable. Similarly, evaluation in
            challenging cases with a minimum on the amount of cloud
            coverage may be desired.

        source_date_window: Tuple[Optional[float], Optional[float]], default=(0, 30)
            The minimum and non-imclusive maximum relative time window
            in days around the segmentation image from which source
            imagery is pulled.  For example, with the default of (0,
            30) only source imagery within the interval of 0 days
            before and 29 days after a segmentation image date will be
            returned as source imagery. If the minimum is None, there
            is no filter on the minimum relative date. Similarly, no
            maximum can be specified with a value of None.

        source_image_transforms: Iterable[Callable[[Any], Any]], default=[],
            A series of callables which are each passed an source
            image and return a transformed image. These may be
            standard Pytorch Torchvision transforms or any function
            with a compatible signature.

        segmentation_image_transforms: Iterable[Callable[[Any], Any]], default=[],
            A series of callables which are each passed a
            segmentation image and return a transformed image. These
            may be standard Pytorch Torchvision transforms or any
            function with a compatible signature.

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
        if len(source_cloud_coverage) != 2:
            raise ValueError(
                f"Parameter source_cloud_coverage should be a tuple of length 2, not length ({len(source_cloud_coverage)})"
            )
        source_data_filters = [
            # Cloud filtering must come before the QA bands may be
            # filtered out
            df.CloudCoverFilter(
                min_cover_fraction=source_cloud_coverage[0],
                max_cover_fraction=source_cloud_coverage[1],
                filter_shadow_landsat=False,
                include=True,
            ),
            df.DataBandFilter(source_bands, include=True),
        ]
        relative_date_filter = rdf.RelativeDateFilter(
            relative_min_date=source_date_window[0],
            relative_max_date=source_date_window[1],
            include=True,
        )

        # Data bands for deforestation and fire segmentation images
        segmentation_bands = {"Deforestation": None, "Fire": ["ConfidenceLevel"]}
        segmentation_data_filters = [
            df.DataBandFilter(segmentation_bands, include=True)
        ]
        self.super().__init__(
            source_files,
            segmentation_files,
            source_data_filters,
            segmentation_data_filters,
            source_image_transforms,
            segmentation_image_transforms,
            relative_date_filter,
            merge_rgb,
            merge_vv_vh,
            error_on_empty,
            single_source_image=True,
        )

    def __getitem__(self, index: int) -> Tuple[bd.DatasetData, bd.DatasetData]:
        """Returns paired source and segmentation data held by this dataset.

        This function extends the inherited __getitem__ function.

        Parameters
        ----------
        index: int
            The location index to retrieve imagery for.

        Returns
        -------
        Tuple[DatasetData, DatasetData]
        A tuple where the first element is an image meant to serve as
        an input to a segmentation modela along with associated
        metadata.

        The second element holds a segmentation image at the same
        location along with associated metadata. Each sample returned
        will be a unique pairing of data, but source imagery may be
        paired with more than one target and vice-versa. The data will
        be filtered by any data filters and modified by any transforms
        that have been set.

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
