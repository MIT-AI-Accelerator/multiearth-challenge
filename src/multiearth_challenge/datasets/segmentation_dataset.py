from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import multiearth_challenge.datasets.base_datasets as bd
import multiearth_challenge.datasets.data_filters as df
import multiearth_challenge.datasets.relative_data_filters as rdf


class ImageSegmentationDataset(bd.MultiEarthDatasetBase):
    """A dataset class intended to be used for supervised learning of a
    image segmentation task. It loads MultiEarth data and returns
    samples consisting of a target segmentation image paired with
    source imagery at the same location along with associated
    metadata. The segmentation image may be None if no imagery is
    contained in the NetCDF file. This will be the case for data
    supplied as test targets for the MultiEarth challenge where only
    the target metadata is supplied.

    The segmentation images are either Deforestation boolean masks or
    fire confidence levels from the FireCCI51 dataset.

    """

    def __init__(
        self,
        source_files: Sequence[Path],
        segmentation_files: Sequence[Path],
        source_bands: Dict[str, Optional[Iterable[str]]],
        merge_source_bands: bool = False,
        source_cloud_coverage: Tuple[float, float] = (0.0, 0.0),
        source_date_window: Tuple[Optional[int], Optional[int]] = (-7, 7),
        single_source_image: bool = True,
        error_on_empty: bool = True,
    ):
        """Note, this initialization will open file handles to a NetCDF
        file. These file handles are released by calling the close
        member function of this class.

        Parameters
        ----------
        source_files: Sequence[Path]
            MultiEarth image NetCDF files whose data should be
            considered for this dataset.

        segmentation_files: Sequence[Path]
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

        merge_source_bands: bool, default=False
            If True, returned source images will have multiple
            channels in increasing order of frequency (e.g., red,
            green, blue for visible), co-pol before cross-pol, and
            with bands not originating from collected imagery coming
            last and in alphabetical order. The metadata returned with
            the imagery will also specify the channel order. If False,
            each band is treated as a separate sample.

        source_cloud_coverage: Tuple[float, float], default=(0.0, 0.0)
            The minimum and maximum allowable cloud coverage allowed
            in visible and IR imagery as a fraction [0, 1]. Setting
            the maximum above 0 may be useful when incorporating SAR
            imagery into a multimodal model where a large fraction of
            cloud coverage may be acceptable. Similarly, evaluation in
            challenging cases with a minimum on the amount of cloud
            coverage may be desired.

            Note, there may be some innacuracies in the identified
            cloud coverage provided by the sensor's QA bands. This is
            especially true for Sentinel-2 data.

        source_date_window: Tuple[Optional[int], Optional[int]], default=(-7, 7)
            The minimum and maximum inclusive relative time window in
            days around the segmentation image from which source
            imagery is pulled. For example, with the default of (-7,
            7) only source imagery within the interval of -7 days
            before and 7 days after a segmentation image date will be
            returned as source imagery. If the minimum is None, there
            is no filter on the minimum relative date. Similarly, no
            maximum can be specified with a value of None.

        single_source_image: bool, default=True
            If True, for each target image only a single source image
            is returned in a unique pair. A single source image may be
            paired with multiple target images and vice-versa
            depending on data filters applied. If False, each target
            image is returned with all source images at the same
            location that satisfy applied data filters.

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
        segmentation_bands = {"Deforestation": None, "FireCCI51": ["ConfidenceLevel"]}
        segmentation_data_filters = [
            df.DataBandFilter(segmentation_bands, include=True)
        ]
        merge_target_bands = False
        super().__init__(
            source_files,
            segmentation_files,
            source_data_filters,
            segmentation_data_filters,
            merge_source_bands,
            merge_target_bands,
            relative_date_filter,
            single_source_image,
            error_on_empty,
        )
