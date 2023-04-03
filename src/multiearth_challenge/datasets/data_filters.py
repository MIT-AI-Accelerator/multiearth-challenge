from typing import Dict, Iterable, Optional

import numpy as np
import xarray as xr

"""This module holds functions for filtering MultiEarth imagery based on various collection conditions."""


class DateFilter:
    """Filters imagery based on collection date."""

    def __init__(
        self,
        *,
        min_date: Optional[np.datetime64] = None,
        max_date: Optional[np.datetime64] = None,
        include: bool = True,
    ):
        """
        Parameters
        ----------
        min_date: Optional[np.datetime64], default=None
            Specifies the minimum of the date interval of
            interest. If None, there is no minimum date.

        max_date: Optional[np.datetime64], default=None
            Specifies the maximum of the date interval of
            interest. If None, there is no maximum date.

        include: bool, default=True
            If True, will include all imagery within the inclusive
            interval specified by the min_date and
            max_date. Otherwise, will exclude imagery within the date
            interval.

        """
        self.min_date = min_date
        self.max_date = max_date
        if (
            self.min_date is not None
            and self.max_date is not None
            and self.min_date >= self.max_date
        ):
            raise ValueError(
                f"Passed min_date ({self.min_date}) must be greater than max_date ({self.max_date})."
            )
        self.include = include

    def __call__(self, dataset: xr.Dataset) -> Dict[str, np.array]:
        """Calculates a boolean mask based on the filter parameters.

        Parameters
        ----------
        dataset: xr.Dataset
            The MultiEarth xarray dataset to filter. This can be
            loaded from a MultiEarth NetCDF file.

        Returns
        -------
        Dict[str, np.array]
            A dictionary with key 'index' (corresponding to the
            relevant Xarray Dataset coordinate) and a value of a
            boolean mask specifying which images satisfy the filter
            parameters used to initialize this DateFilter. This
            boolean mask will have length equal to the length of the
            dataset's index coordinate.

        """
        mask = np.array([True] * len(dataset.index))
        if self.min_date is not None:
            mask &= dataset.image_dates.data >= self.min_date
        if self.max_date is not None:
            mask &= dataset.image_dates.data <= self.max_date
        if not self.include:
            mask = np.logical_not(mask)
        return {"index": mask}


class LocationFilter:
    """Filters imagery based on their center latitude and longitude."""

    def __init__(
        self,
        *,
        min_lat: Optional[float] = None,
        min_lon: Optional[float] = None,
        max_lat: Optional[float] = None,
        max_lon: Optional[float] = None,
        include: bool = True,
    ):
        """
        Parameters
        ----------
        min_lat: Optional[float], default=None
             Specifies the minimum latitude of the region of interest.
             If None, there is no minimum latitude.

        min_lon: Optiona[float], default=None
             Specifies the minimum longitude of the region of
             interest.  If None, there is no minimum latitude.

        max_lat: Optional[float], default=None
             Specifies the maximum latitude of the region of interest.
             If None, there is no minimum latitude.

        max_lon: Optional[float], default=None
             Specifies the maximum longitude of the region of
             interest.  If None, there is no minimum latitude.

        include: bool=True
            If True, will include all imagery within the inclusive
            region specified by the min_lat, min_lon, max_lat, and
            max_lon.  Otherwise, will exclude imagery within the
            region of interest.

        """
        if min_lon is not None and max_lon is not None and max_lon < min_lon:
            raise ValueError(
                f"Passed maximum longitude ({max_lon}) must be greater than the passed minimum longitude ({min_lon})."
            )
        if min_lat is not None and max_lat is not None and max_lat < min_lat:
            raise ValueError(
                f"Passed maximum longitude ({max_lat}) must be greater than the passed minimum longitude ({min_lat})."
            )

        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon

        self.include = include

    def __call__(self, dataset: xr.Dataset) -> Dict[str, np.array]:
        """Calculates a boolean mask based on the filter parameters.

        Parameters
        ----------
        dataset: xr.Dataset
            The MultiEarth xarray dataset to filter. This can be
            loaded from a MultiEarth NetCDF file.

        Returns
        -------
        Dict[str, np.array]
            A dictionary with key 'index' (corresponding to the
            relevant Xarray Dataset coordinate) and a value of a
            boolean mask specifying which images satisfy the filter
            parameters used to initialize this LocationFilter. It will
            have length equal to the length of the dataset's index
            coordinate.

        """
        mask = np.array([True] * len(dataset.index))
        if self.min_lat is not None:
            mask &= dataset.center_lat_lons[:, 0].data >= self.min_lat
        if self.max_lat is not None:
            mask &= dataset.center_lat_lons[:, 0].data <= self.max_lat
        if self.min_lon is not None:
            mask &= dataset.center_lat_lons[:, 1].data >= self.min_lon
        if self.max_lon is not None:
            mask &= dataset.center_lat_lons[:, 1].data <= self.max_lon
        if not self.include:
            mask = np.logical_not(mask)
        return {"index": mask}


class DataBandFilter:
    """Filters imagery based on collecting sensor/type of data and band."""

    def __init__(
        self, data_band_names: dict[str, Optional[Iterable[str]]], include: bool = True
    ):
        """
        Parameters
        ----------
        data_band_names: dict[str, Optional[Iterable[str]]]
            A dictionary with keys specifying the data name and values
            specifying the bands of interest. If the value associated
            with a data key is None, all associated bands are
            included. Acceptable data source names and associated
            bands are as follows:

            "Landsat-5": ['SR_B4', 'SR_B5', 'SR_B7', 'SR_B2', 'ST_B6', 'SR_B3', 'QA_PIXEL', 'SR_B1'],
            "Landsat-8": ['SR_B4', 'SR_B5', 'SR_B7', 'SR_B2', 'SR_B6', 'ST_B10', 'SR_B3', 'QA_PIXEL', 'SR_B1'],
            "Sentinel-1": ['VV', 'VH'],
            "Sentinel-2": ['B3', 'B12', 'B11', 'B1', 'B6', 'B8A', 'B2', 'QA60', 'B4', 'B8', 'B5', 'B9', 'B7'],
            "Fire": ['BurnDate', 'ConfidenceLevel', 'LandCover', 'ObservedFlag'],
            "Deforestation": None, There is only a single unnamed band associated with the deforestation data

            Additional documentation on the details of each data band
            can be found in the NetCDF data files or on the sensors'
            datasheets.

        include: bool=True
            If True, will include all imagery from the specified data
            sources and bands. Otherwise, this imagery will be
            excluded.

        """
        self.data_band_names = data_band_names

        acceptable_bands = {
            "Landsat-5": [
                "SR_B4",
                "SR_B5",
                "SR_B7",
                "SR_B2",
                "ST_B6",
                "SR_B3",
                "QA_PIXEL",
                "SR_B1",
            ],
            "Landsat-8": [
                "SR_B4",
                "SR_B5",
                "SR_B7",
                "SR_B2",
                "SR_B6",
                "ST_B10",
                "SR_B3",
                "QA_PIXEL",
                "SR_B1",
            ],
            "Sentinel-1": ["VV", "VH"],
            "Sentinel-2": [
                "B3",
                "B12",
                "B11",
                "B1",
                "B6",
                "B8A",
                "B2",
                "QA60",
                "B4",
                "B8",
                "B5",
                "B9",
                "B7",
            ],
            "FireCCI51": ["BurnDate", "ConfidenceLevel", "LandCover", "ObservedFlag"],
            "Deforestation": None,
        }
        self.deforestation_flag = False
        for data_source, bands in self.data_band_names.items():
            if data_source not in acceptable_bands.keys():
                raise ValueError(
                    f"Passed data source name of ({data_source}) not recognized.\n"
                    f"Expected data source names: {list(acceptable_bands.keys())}."
                )
            if data_source == "Deforestation":
                self.deforestation_flag = True
                if bands is not None:
                    raise ValueError(
                        f"Deforestation data has a single unnamed data band. Cannot specify filtering on bands: {bands}"
                    )

            else:
                for band in bands:
                    if band not in acceptable_bands[data_source]:
                        raise ValueError(
                            f"Passed band name of ({band}) for data_source ({data_source}).\n"
                            f"Expected band names: {acceptable_bands[data_source]}"
                        )

        self.include = include

    def __call__(self, dataset: xr.Dataset) -> Dict[str, np.array]:
        """Calculates a boolean mask based on the filter parameters.

        Parameters
        ----------
        dataset: xr.Dataset
            The MultiEarth xarray dataset to filter. This can be
            loaded from a MultiEarth NetCDF file.

        Returns
        -------
        Dict[str, np.array]
            A dictionary with key 'data_band' (corresponding to the
            relevant Xarray Dataset coordinate) and a value of a
            boolean mask specifying which images satisfy the filter
            parameters used to initialize this DataBandFilter. It will
            have length equal to the length of the dataset's data_band
            coordinate.

        """
        if dataset.attrs["data_source"] not in self.data_band_names.keys():
            # Remove all data
            if self.deforestation_flag:
                return {"index": np.array([not self.include] * len(dataset.index))}
            else:
                return {
                    "data_band": np.array([not self.include] * len(dataset.data_band))
                }
        bands = self.data_band_names[dataset.attrs["data_source"]]
        if bands is None:
            # If bands is None, include all bands for the sensor
            if self.deforestation_flag:
                return {"index": np.array([self.include] * len(dataset.index))}
            else:
                return {"data_band": np.array([self.include] * len(dataset.data_band))}
        mask = np.array([False] * len(dataset.data_band))
        for band in bands:
            mask |= dataset.data_band.data == band
        if not self.include:
            mask = np.logical_not(mask)
        return {"data_band": mask}


class CloudCoverFilter:
    """Filters imagery based on the cloud coverage.
    Cloud coverage is determined by the information contained in the
    QA channels for the respective sensors.  Additional details can be
    found in the links below.
    Landsat-5 - https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1336_Landsat4-5-TM-C2-L2-DFCB-v4.pdf section 3.2
    Landsat-8 - https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1328_Landsat8-9-OLI-TIRS-C2-L2-DFCB-v6.pdf section 3.2
    Sentinel-2 - https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2 bit

    """

    def __init__(
        self,
        *,
        min_cover_fraction: float,
        max_cover_fraction: float,
        filter_shadow_landsat: bool = True,
        include: bool = True,
    ):
        """
        Parameters
        ----------
        min_cover_fraction: float
            A value in [0, 1] specifying a minimum fraction that the
            image is covered by clouds. This value must be <=
            max_cover_fraction.

        max_cover_fraction: float
            A value in [0, 1] specifying a maximum fraction that the
            image is covered by clouds. This value must be >=
            min_cover_fraction.

        filter_shadow_landsat: bool, default=True
            Landsat imagery contains information about cloud
            shadows. If True, will include cloud shadows when
            calculating cloud coverage. Otherwise cloud shadows will
            be ignored.

        include: bool, default=True
            If True, will include all imagery with cloud coverage
            fraction across an image with less than or equal to the
            specified max_cover_fraction. Otherwise, will exclude
            imagery equal to or below this threshold.

        """
        if min_cover_fraction < 0.0 or min_cover_fraction > 1.0:
            raise ValueError(
                "min_cover_Fraction must be in the [0, 1] interval. "
                f"Was passed {min_cover_fraction}."
            )
        if max_cover_fraction < 0.0 or max_cover_fraction > 1.0:
            raise ValueError(
                "max_cover_Fraction must be in the [0, 1] interval. "
                f"Was passed {max_cover_fraction}."
            )
        if min_cover_fraction > max_cover_fraction:
            raise ValueError(
                f"min_cover_fraction must be <= max_cover_fraction. Was passed\n"
                f"min_cover_fraction: ({min_cover_fraction})\n"
                f"max_cover_fraction: ({max_cover_fraction})"
            )

        self.max_cover_fraction = max_cover_fraction
        self.coverage_band = {
            "Landsat-5": {
                "qa_channel": "QA_PIXEL",
                "cloud_bits": [1, 3, 4]
                if filter_shadow_landsat
                else [
                    1,
                    3,
                ],  # bit 1 is for dilated cloud, 3 for cloud, 4 for cloud shadow
            },
            "Landsat-8": {
                "qa_channel": "QA_PIXEL",
                "cloud_bits": [1, 2, 3, 4]
                if filter_shadow_landsat
                else [
                    1,
                    2,
                    3,
                ],  # bit 1 is for dilated cloud, 2 for cirrus, 3 for cloud, 4 for cloud shadow
            },
            "Sentinel-2": {
                "qa_channel": "QA60",
                "cloud_bits": [10, 11],  # bit 10 is for clouds, 11 for cirrus
            },
        }
        self.unoccluded_data_sources = ["Sentinel-1", "Deforestation"]
        self.include = include

    def __call__(self, dataset: xr.Dataset) -> Dict[str, np.array]:
        """Calculates a boolean mask based on the filter parameters.

        Parameters
        ----------
        dataset: xr.Dataset
            The MultiEarth xarray dataset to filter. This can be
            loaded from a MultiEarth NetCDF file.

        Returns
        -------
        Dict[str, np.array]
            A dictionary with keys 'data_band' and 'index'
            (corresponding to the relevant Xarray Dataset
            coordinates). Each key has a value of a boolean mask
            specifying which images satisfy the filter parameters used
            to initialize this CloudCoverFilter. Each mask will have
            length equal to the length of the dataset's corresponding
            coordinate.

        """
        data_source = dataset.attrs["data_source"]
        if data_source == self.sent1_name:
            # Sentinal-1 SAR imagery, and deforestation masks not occluded by clouds
            return np.array([self.include] * len(dataset.index))
        if data_source not in self.coverage_band:
            data_source_names = list(self.coverage_band.keys()) + [self.sent1_name]
            raise ValueError(
                f"Unrecognized data source ({data_source}). Expected one of {data_source_names}."
            )
        coverage_band = self.coverage_band[data_source]
        coverage_band_name = coverage_band["qa_channel"]
        bit_mask = 0
        for cloud_bit in coverage_band["cloud_bits"]:
            bit_mask |= 1 << (cloud_bit)
        cov_dataset = dataset.isel(index=dataset.data_bands.data == coverage_band_name)
        mask = cov_dataset.images.data & bit_mask
        final_mask = np.array([True] * len(dataset.index))
        for final_mask_val, cloud_mask in zip(final_mask, mask):
            frac = np.count_nonzero(cloud_mask) / cloud_mask.size
            if frac > self.max_cover_fraction:
                final_mask_val = False

        if not self.include:
            final_mask = np.logical_not(final_mask)

        return final_mask


class MatchedDateLocFilter:
    """Filters imagery based on whether each sample matches within a
    specified time window and the same location of any sample from a
    set of datasets. This filter is used as part of the initialization
    of the MultiEarthDatasetBase class to filter out any data that is
    guaranteed to not be used since it won't ever be paired with other
    data as part of a dataset sample.

    """

    def __init___(
        self,
        comp_datasets: xr.Dataset,
        *,
        relative_min_date: Optional[np.timedelta64] = None,
        relative_max_date: Optional[np.timedelta64] = None,
        date_include: bool = True,
    ):
        """Parameters
        ----------
        comp_datasets: xr.Datasets
            The set of datasets to compare against to see if any of
            their samples matches the location and relative time
            window for the data to be filtered.

        relative_min_date: Optional[np.timedelta64], default=None
            Specifies the minimum of the relative date interval of
            interest for the comp_datasets data. Negative values
            indicate a time before the date of the data to be filtered
            and positive after. If None, there is no minimum date.

        relative_max_date: Optional[np.timedelta64], default=None
            Specifies the maximum of the relative date interval of
            interest for the comp_datasets data. Negative values
            indicate a time before the date of the data to be filtered
            and positive after. If None, there is no maximum date.

        date_include: bool, default=True
            If True, will include all data that has at least one
            position and relative time match with samples from
            comp_datasets. Otherwise, will include all data that has
            at least one position match and is outside the relative
            time window.

        """
        self.comp_datasets = comp_datasets
        if not len(self.comp_datasets.index):
            raise ValueError(f"Expected at least 1 dataset to compare against")
        self.min_date = relative_min_date
        self.max_date = relative_max_date
        if (
            self.min_date is not None
            and self.max_date is not None
            and self.min_date >= self.max_date
        ):
            raise ValueError(
                f"Passed parameter relative_min_date ({self.min_date}) must be greater than relative_max_date ({self.max_date})."
            )
        self.date_include = date_include

    def __call__(self, dataset: xr.Dataset) -> Dict[str, np.array]:
        """Calculates a boolean mask based on the filter parameters.

        Parameters
        ----------
        dataset: xr.Dataset
            The MultiEarth xarray dataset to filter. This can be
            loaded from a MultiEarth NetCDF file.

        Returns
        -------
        Dict[str, np.array]
            A dictionary with key 'index' (corresponding to the
            relevant Xarray Dataset coordinate) and a value of a
            boolean mask specifying which images satisfy the filter
            parameters used to initialize this LocationFilter. It will
            have length equal to the length of the dataset's index
            coordinate.

        """
        dist_thresh = 1e-5  # degrees, center lat/lons are spaced by 0.02 degrees
        mask = [False] * len(self.comp_datasets.index)
        for ii, (target_date, target_loc) in enumerate(
            zip(dataset.collection_dates.data, dataset.center_lat_lons.data)
        ):
            for comp_dataset in self.comp_datasets:
                date_mask = np.any(self.relative_date_filter(comp_dataset, target_date))
                if not self.date_include:
                    date_mask = np.logical_not(date_mask)
                loc_mask = np.any(
                    np.all(
                        np.abs(comp_dataset.center_lat_lons.data - target_loc)
                        < dist_thresh,
                        axis=1,
                    )
                )
                mask[ii] |= np.logical_and(date_mask, loc_mask)
            return {"index": mask_val}
