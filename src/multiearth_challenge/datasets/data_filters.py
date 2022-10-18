import numpy as np
import xarray as xr


class DateFilter:
    def __init__(self, *, min_date=None, max_date=None):
        self.min_date = min_date
        self.max_date = max_date

    def __call__(self, dataset):
        mask = np.array([True] * len(dataset.index))
        if self.min_date is not None:
            mask &= dataset.image_dates >= self.min_date
        if self.max_date is not None:
            mask &= dataset.image_dates <= self.max_date
        return mask
    

class LocationFilter:
    def __init__(self, *, min_lat=None, min_lon=None, max_lat=None, max_lon=None):
        if min_lon is not None and max_lon is not None and max_lon < min_lon:
            raise ValueError(f"Passed maximum longitude ({max_lon}) is less than the passed minimum longitude ({min_lon})")
        if min_lat is not None and max_lat is not None and max_lat < min_lat:
            raise ValueError(f"Passed maximum longitude ({max_lat}) is less than the passed minimum longitude ({min_lat})")

        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon

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
        return mask
        

class SensorBandFilter:
    def __init__(self, sensor_band_names):
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

    def __call__(self, dataset):
        if dataset.attrs["sensor"] not in self.sensors:
            return np.array([False] * len(dataset.index))
        bands = self.sensor_band_names[dataset.attrs["sensor"]]
        if bands is None:
            # If bands is None, include all bands for the sensor
            return np.array([True] * len(dataset.index))
        mask = np.array([False] * len(dataset.index))
        for band in bands:
            mask |= dataset.sensor_bands == band
        return mask
