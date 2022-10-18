import pkg_resources

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

import multiearth_challenge.datasets.data_filters as df


@pytest.fixture
def data_file():
    return pkg_resources.resource_filename("multiearth_challenge", "data/sample_dataset/landsat5.nc")


def test_date_filter(data_file):
    print(data_file)
    with xr.open_dataset(data_file) as data:
        filter = df.DateFilter()
        res = filter(data)
        exp = np.array([True] * len(data.index))
        npt.assert_array_equal(res, exp)

        dates = data.image_dates.data
        sorted_dates = np.sort(dates)
        
        min_date = sorted_dates[0]
        above_min_date = min_date + np.timedelta64(1, 'us')
        min_indices = np.where(dates == min_date)[0]
        filter = df.DateFilter(min_date=above_min_date)
        res = filter(data)
        exp[min_indices] = False
        npt.assert_array_equal(res, exp)

        max_date = sorted_dates[-1]
        below_max_date = max_date - np.timedelta64(1, 'us')
        max_indices = np.where(dates == max_date)[0]
        filter = df.DateFilter(min_date=above_min_date, max_date=below_max_date)
        res = filter(data)
        exp[max_indices] = False
        npt.assert_array_equal(res, exp)

def test_location_filter(data_file):
    with xr.open_dataset(data_file) as data:
        filter = df.LocationFilter()
        res = filter(data)
        exp = np.array([True] * len(data.index))
        npt.assert_array_equal(res, exp)

        center_lat_lons = data.center_lat_lons.data
        sorted_lats = np.sort(center_lat_lons[:, 0])
        sorted_lons = np.sort(center_lat_lons[:, 1])
        
        min_lat = sorted_lats[0]
        above_min_lat = min_lat + 1e-6
        min_lat_indices = np.where(center_lat_lons[:, 0] == min_lat)[0]
        filter = df.LocationFilter(min_lat=above_min_lat)
        res = filter(data)
        exp[min_lat_indices] = False
        npt.assert_array_equal(res, exp)

        max_lat = sorted_lats[-1]
        below_max_lat = max_lat - 1e-6
        max_lat_indices = np.where(center_lat_lons[:, 0] == max_lat)[0]
        filter = df.LocationFilter(min_lat=above_min_lat, max_lat=below_max_lat)
        res = filter(data)
        exp[max_lat_indices] = False
        npt.assert_array_equal(res, exp)

        min_lon = sorted_lons[40]
        above_min_lon = min_lon + 1e-6
        min_lon_indices = np.where(center_lat_lons[:, 1] <= above_min_lon)[0]
        filter = df.LocationFilter(min_lat=above_min_lat, max_lat=below_max_lat, min_lon=above_min_lon)
        res = filter(data)
        exp[min_lon_indices] = False
        npt.assert_array_equal(res, exp)

        max_lon = sorted_lons[-1]
        below_max_lon = max_lon - 1e-6
        max_lon_indices = np.where(center_lat_lons[:, 1] == max_lon)[0]
        filter = df.LocationFilter(
            min_lat=above_min_lat,
            max_lat=below_max_lat,
            min_lon=above_min_lon,
            max_lon=below_max_lon
        )
        res = filter(data)
        exp[max_lon_indices] = False
        npt.assert_array_equal(res, exp)

def test_sensor_band_filter(data_file):
    with xr.open_dataset(data_file) as data:
        bands = data.sensor_bands.data

        filter = df.SensorBandFilter(sensor_band_names={"Landsat-8": None})
        res = filter(data)
        exp = np.array([False] * len(data.index))
        npt.assert_array_equal(res, exp)

        filter = df.SensorBandFilter(sensor_band_names={"Landsat-5": None})
        res = filter(data)
        exp = np.array([True] * len(data.index))
        npt.assert_array_equal(res, exp)

        requested_bands = ["SR_B1"]
        filter = df.SensorBandFilter(sensor_band_names={"Landsat-5": requested_bands})
        res = filter(data)
        exp = np.array([False] * len(data.index))
        indices = np.where(bands == requested_bands[0])[0]
        exp[indices] = True
        npt.assert_array_equal(res, exp)
        
        requested_bands = ["SR_B1", "ST_B6"]
        filter = df.SensorBandFilter(sensor_band_names={"Landsat-5": requested_bands})
        res = filter(data)
        indices = np.where(bands == requested_bands[1])[0]
        exp[indices] = True
        npt.assert_array_equal(res, exp)

