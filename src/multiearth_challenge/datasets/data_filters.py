import numpy as np
import xarray as xr


class DateFilter:
    def __init__(self, *, min_date=None, max_date=None, include=True):
        self.min_date = min_date
        self.max_date = max_date
        self.include = include

    def __call__(self, dataset):
        mask = np.array([True] * len(dataset.index))
        if self.min_date is not None:
            mask &= dataset.image_dates >= self.min_date
        if self.max_date is not None:
            mask &= dataset.image_dates <= self.max_date
        if not self.include:
            mask = np.logical_not(mask)
        return mask
    

class LocationFilter:
    def __init__(self, *, min_lat=None, min_lon=None, max_lat=None, max_lon=None, include=True):
        if min_lon is not None and max_lon is not None and max_lon < min_lon:
            raise ValueError(f"Passed maximum longitude ({max_lon}) is less than the passed minimum longitude ({min_lon})")
        if min_lat is not None and max_lat is not None and max_lat < min_lat:
            raise ValueError(f"Passed maximum longitude ({max_lat}) is less than the passed minimum longitude ({min_lat})")

        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon

        self.include = include

    def __call__(self, dataset):
        mask = np.array([True] * len(dataset.index))
        if self.min_lat is not None:
            mask &= dataset.center_lat_lons[:, 0] >= self.min_lat
        if self.max_lat is not None:
            mask &= dataset.center_lat_lons[:, 0] <= self.max_lat
        if self.min_lon is not None:
            mask &= dataset.center_lat_lons[:, 1] >= self.min_lon
        if self.max_lon is not None:
            mask &= dataset.center_lat_lons[:, 1] <= self.max_lon
        if not self.include:
            mask = np.logical_not(mask)
        return mask
        

class SensorBandFilter:
    def __init__(self, sensor_band_names, include=True):
        self.sensor_band_names = sensor_band_names
        acceptable_keys = [
            "Landsat-5",
            "Landsat-8",
            "Sentinel-1",
            "Sentinel-2",
        ]
        self.sensors = sensor_band_names.keys()
        for sensor in self.sensors:
            if sensor not in acceptable_keys:
                raise ValueError(f"Passed sensor name of {sensor} not recognized.\n"
                                 f"Expected sensor names: {acceptable_keys}")
        # TODO Add error checking for band names
        self.include = include

    def __call__(self, dataset):
        if dataset.attrs["sensor"] not in self.sensors:
            return np.array([not self.include] * len(dataset.index))
        bands = self.sensor_band_names[dataset.attrs["sensor"]]
        if bands is None:
            # If bands is None, include all bands for the sensor
            return np.array([self.include] * len(dataset.index))
        mask = np.array([False] * len(dataset.index))
        for band in bands:
            mask |= dataset.sensor_bands == band
        if not self.include:
            mask = np.logical_not(mask)
        return mask


class CloudCoverFilter:
    """
    https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
    Landsat-5 = https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1336_Landsat4-5-TM-C2-L2-DFCB-v4.pdf section 3.2
    Landsat-8 - https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands section 3.2
    Sentinel-2 https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2 bit 10 is for clouds, 11 cirrus. This stackexchange comment indicates the other bits aren't set or aren't relevant (https://gis.stackexchange.com/questions/307974/what-are-the-other-bits-in-sentinels-qa60-band)
    """
    def __init__(self, max_cover_fraction, filter_shadow_landsat=True, include=True):
        if max_cover_fraction < 0.0 or max_cover_fraction > 1.0:
            raise ValueError("max_cover_Fraction must be in [0, 1] interval. "
                             f"Was passed {max_cover_fraction}")
        self.max_cover_fraction = max_cover_fraction
        self.coverage_band = {
            "Landsat-5": {
                "qa_channel": "QA_PIXEL",
                "cloud_bits": [4, 6] if filter_shadow_landsat else [6],
                "cloud_free": False,
            },
            "Landsat-8": {
                "qa_channel": "QA_PIXEL",
                "cloud_bits": [4, 6] if filter_shadow_landsat else [6],
                "cloud_free": True,
            },
            "Sentinel-2": {
                "qa_channel": "QA60",
                "cloud_bits": [10, 11],
                "cloud_free": False,
            },
        }
        self.sent1_name = "Sentinel-1"
        self.include = include

    def __call__(self, dataset):
        sensor = dataset.attrs["sensor"]
        if sensor == self.sent1_name:
            # Sentinal-1 SAR imagery not occluded by clouds
            return np.array([self.include] * len(dataset.index))
        if sensor not in self.coverage_band:
            sensor_names = list(self.coverage_band.keys()) + [self.sent1_name]
            raise ValueError(f"Unrecognized sensor ({sensor}). Expected one of {sensor_names}")
        coverage_band = self.coverage_band[sensor]
        coverage_band_name = coverage_band["qa_channel"]
        bit_mask = 0
        for cloud_bit in coverage_band["cloud_bits"]:
            bit_mask |= 1 << (cloud_bit)
        cov_dataset = dataset.isel(index=dataset.sensor_bands == coverage_band_name)
        mask = cov_dataset.images.data & bit_mask
        final_mask = np.array([True] * len(dataset.index))
        for final_mask_val, cloud_mask in zip(final_mask, mask):
            frac = np.count_nonzero(cloud_mask) / cloud_mask.size
            if frac > self.max_cover_fraction:
                final_mask_val = False

        if not self.include:
            final_mask = np.logical_not(final_mask)

        return final_mask
