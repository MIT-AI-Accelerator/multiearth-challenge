import pytest

import multiearth_challenge.data_tools as dt


def test_parse_filename_parts():
    test_filenames = [
        ("Landsat5_QA_PIXEL_-54.48_-4.19_1984_05_20.jpg", "Landsat5", "QA_PIXEL", "-54.48", "-4.19", 1984, 5, 20),
        ("Landsat5_ST_B6_-55.20_-4.39_2011_09_04.jpg", "Landsat5", "ST_B6", "-55.20", "-4.39", 2011, 9, 4),
        ("Landsat5_SR_B7_-55.16_-4.33_2010_02_05.jpg", "Landsat5", "SR_B7", "-55.16", "-4.33", 2010, 2, 5),
        ("Landsat8_QA_PIXEL_-54.64_-4.19_2013_03_27.jpg", "Landsat8", "QA_PIXEL", "-54.64", "-4.19", 2013, 3, 27),
        ("Landsat8_SR_B1_-54.76_-4.21_2017_10_06.jpg", "Landsat8", "SR_B1", "-54.76", "-4.21", 2017, 10, 6),
        ("Landsat8_ST_B10_-54.64_-4.23_2013_03_27.jpg", "Landsat8", "ST_B10", "-54.64", "-4.23", 2013, 3, 27),
        ("Sentinel1_VH_-54.52_-4.33_2021_03_30.jpg", "Sentinel1", "VH", "-54.52", "-4.33", 2021, 3, 30),
        ("Sentinel1_VV_-55.20_-4.39_2021_09_07.jpg", "Sentinel1", "VV", "-55.20", "-4.39", 2021, 9, 7),
        ("Sentinel2_B11_-54.54_-4.19_2021_02_05.jpg", "Sentinel2", "B11", "-54.54", "-4.19", 2021, 2, 5),
        ("Sentinel2_B9_-55.20_-4.39_2021_12_27.jpg", "Sentinel2", "B9", "-55.20", "-4.39", 2021, 12, 27),
        ("Sentinel2_QA60_-54.56_-4.19_2020_05_11.jpg", "Sentinel2", "QA60", "-54.56", "-4.19", 2020, 5, 11),
    ]

    for item in test_filenames:
        res = dt.parse_filename_parts(item[0], pos_float=True)
        assert res["sensor"] == item[1]
        assert res["band"] == item[2]
        assert res["lon"] == pytest.approx(float(item[3]))
        assert res["lat"] == pytest.approx(float(item[4]))
        assert res["date"].year == item[5]
        assert res["date"].month == item[6]
        assert res["date"].day == item[7]

        res = dt.parse_filename_parts(item[0], pos_float=False)
        assert res["lon"] == item[3]
        assert res["lat"] == item[4]
        
