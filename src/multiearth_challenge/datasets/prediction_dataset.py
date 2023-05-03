from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import multiearth_challenge.datasets.base_datasets as bd
import multiearth_challenge.datasets.data_filters as df
import multiearth_challenge.datasets.relative_data_filters as rdf


class ImagePredictionDataset(bd.MultiEarthDatasetBase):
    """A dataset class intended to be used for supervised learning of a
    predictive generative model. It loads MultiEarth data and returns
    samples consisting of multiple historical source images paired
    with a future target image at the same location along with
    associated metadata. The target image may be None if no imagery is
    contained in the NetCDF file. This will be the case for data
    supplied as test targets for the MultiEarth challenge where only
    the target metadata is supplied.

    """

    def __init__(
        self,
        source_files: Sequence[Path],
        target_files: Sequence[Path],
        source_bands: Dict[str, Optional[Iterable[str]]],
        target_bands: Dict[str, Optional[Iterable[str]]],
        source_date_window: Tuple[Optional[int], Optional[int]],
        max_source_cloud_coverage: float = 0.0,
        max_target_cloud_coverage: float = 0.0,
        merge_source_bands: bool = False,
        merge_target_bands: bool = False,
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

        target_files: Sequence[Path]
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

        source_date_window: Tuple[Optional[int], Optional[int]]
            The minimum and maximum inclusive relative time window in
            days around the target image from which source imagery is
            pulled.  If the minimum is None, there is no filter on the
            minimum relative date. Similarly, no maximum can be
            specified with a value of None. For example, a value of
            (None, -365) would result in only source imagery collected
            at least 365 days before the target image date being
            returned.

        max_source_cloud_coverage: float default=0.0
            The maximum allowable cloud coverage allowed in the source
            imagery as a fraction [0, 1]. Setting the maximum above 0
            may be useful to incorporate additional samples even if
            the source image is slightly obscured.

            Note, there may be some innacuracies in the identified
            cloud coverage provided by the sensor's QA bands. This is
            especially true for Sentinel-2 data.

        max_target_cloud_coverage: float default=0.0
            The maximum allowable cloud coverage allowed in the target
            imagery as a fraction [0, 1]. Setting the maximum above 0
            may be useful to incorporate additional samples even if
            the target image is slightly obscured.

            Note, there may be some innacuracies in the identified
            cloud coverage provided by the sensor's QA bands. This is
            especially true for Sentinel-2 data.

        merge_source_bands: bool, default=False
            If True, returned source images will have multiple
            channels in increasing order of frequency (e.g., red,
            green, blue for visible), co-pol before cross-pol, and
            with bands not originating from collected imagery coming
            last and in alphabetical order. The metadata returned with
            the imagery will also specify the channel order. If False,
            each band is treated as a separate sample.

        merge_target_bands: bool, default=False
            If True, returned source images will have multiple
            channels in increasing order of frequency (e.g., red,
            green, blue for visible), co-pol before cross-pol, and
            with bands not originating from collected imagery coming
            last and in alphabetical order. The metadata returned with
            the imagery will also specify the channel order. If False,
            each band is treated as a separate sample.

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

        super().__init__(
            source_files,
            target_files,
            source_data_filters,
            target_data_filters,
            merge_source_bands,
            merge_target_bands,
            relative_date_filter,
            single_source_image=False, # Always want all relevant source imagery paired with target
            error_on_empty=error_on_empty,
        )
