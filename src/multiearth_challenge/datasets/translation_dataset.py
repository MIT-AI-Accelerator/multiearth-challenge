from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import multiearth_challenge.datasets.base_datasets as bd
import multiearth_challenge.datasets.data_filters as df
import multiearth_challenge.datasets.relative_data_filters as rdf


class SARToVisibleDataset(bd.MultiEarthDatasetBase):
    """A dataset class intended to be used for supervised learning of the
    translation of a SAR image to a visible image. It loads MultiEarth
    data and returns samples consisting of a target image paired with
    visible imagery at the same location along with associated
    metadata. The target image may be None if no imagery is contained
    in the NetCDF file. This will be the case for data supplied as
    test targets for the MultiEarth challenge where only the target
    metadata is supplied.

    """

    def __init__(
        self,
        sar_files: Sequence[Path],
        visible_files: Sequence[Path],
        sar_bands: Iterable[str] = ["VV"],
        merge_sar_bands: bool = False,
        sar_date_window: Tuple[Optional[int], Optional[int]] = (-7, 7),
        visible_bands: Dict[str, Iterable[str]] = df.RGB_BANDS,
        merge_visible_bands: bool = False,
        max_visible_cloud_coverage: float = 0.0,
        single_source_image: bool = True,
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

        merge_sar_bands: bool, default=False
            If True, and both polarizations are listed in sar_bands,
            returned SAR images will be two channel with the first
            being 'VV' and the second being 'VH'. The metadata
            returned with the imagery will also specify the channel
            order. If False, each band is treated as a separate
            sample.

        sar_date_window: Tuple[Optional[int], Optional[int]], default=(-7, 7)
            The minimum and maximum inclusive relative time window in
            days around the segmentation image from which source
            imagery is pulled. For example, with the default of (-7,
            7) only SAR imagery within the interval of -7 days
            before and 7 days after a visible image date will be
            returned as source imagery. If the minimum is None, there
            is no filter on the minimum relative date. Similarly, no
            maximum can be specified with a value of None.

        visible_bands: dict[str, Iterable[str]], default=df.RGB_BANDS
            A dictionary with keys specifying the sensor name and
            values specifying the bands of interest. If the value is
            None, no filtering is done for that sensor's band and all
            associated bands are included. Acceptable sensor names and
            associated bands are defined in the default value which is
            dict(
                "Landsat-5": ['SR_B3', 'SR_B2', 'SR_B1'],
                "Landsat-8": ['SR_B4', 'SR_B3', 'SR_B2'],
                "Sentinel-2": ['B4', 'B3', 'B2'],
            )

            Additional documentation on the details of each data band
            can be found in the NetCDF data files or on the sensors'
            datasheets.

        merge_visible_bands: bool, default=False
            If True, and multiple visible bands are listed in
            visible_bands, returned visible images will be
            multi-channel in ascending frequency (red, green,
            blue). The metadata returned with the imagery will also
            specify the channel order. If False, each band is treated
            as a separate sample.

        max_visible_cloud_coverage: float default=0.0
            The maximum allowable cloud coverage allowed in visible
            imagery as a fraction [0, 1]. Setting the maximum above 0
            may be useful to incorporate additional samples even if
            the truth visible image is slightly obscured.

            Note, there may be some innacuracies in the identified
            cloud coverage provided by the sensor's QA bands. This is
            especially true for Sentinel-2 data.

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
        super().__init__(
            source_files=sar_files,
            target_files=visible_files,
            source_data_filters=sar_data_filters,
            target_data_filters=visible_data_filters,
            merge_source_bands=merge_sar_bands,
            merge_target_bands=merge_visible_bands,
            relative_date_filter=relative_date_filter,
            single_source_image=single_source_image,
            error_on_empty=error_on_empty,
        )
