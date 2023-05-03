from collections import OrderedDict
import datetime
import json
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Tuple, Union


"""This module holds simple functions for working with zip files of
individual images, rather than NetCDF files. The image's position,
collection date, and data band are specified in the filename."""

def parse_filename_parts(
        image_path: Union[str, Path],
        pos_float: bool=True,
) -> Dict[str, Union[str, float, datetime.datetime]]:
    """Image paths hold information on the sensor, band, position, and
    date. This function parses and returns these values.

    Parameters
    ----------
    image_path: Union[str, Path]
        The image file of interest.

    pos_float: bool, default=True
        If true, the latitude and longitude is returned as a float.
        Otherwise, it is left as a string which may be more convenient
        for comparison.

    Return
    ------
    Dict[str, Union[str, float, datetime.datetime]]
        A dictionary with keys: 'sensor', 'band', 'lat', 'lon', and
        'date' holding the parsed information.

    """
    name = Path(image_path).stem
    parts = name.split("_")
    sensor = parts[0]
    band = "_".join(parts[1:-5])
    lat = parts[-4]
    lon = parts[-5]
    if pos_float:
        lat = float(lat)
        lon = float(lon)
    date = datetime.date(year=int(parts[-3]), month=int(parts[-2]), day=int(parts[-1].split(".")[0]))
    return {
        "sensor": sensor,
        "band": band,
        "lat": lat,
        "lon": lon,
        "date": date
    }


def get_image_paths(image_dir: Union[str, Path], band: str)->List[Path]:
    """Retrieve paths to images in the passed directory that correspond to
    the specified sensor band.

    Parameters
    ----------
    image_dir: Union[str, Path]
        The directory that holds image files.

    band: str
        The sensor band of interest.

    Return
    ------
    List[Path]
        All image paths within the passed directory that correspond to
        the specified sensor band.
    """
    image_paths = list(Path(image_dir).glob(f"*_{band}_*.tiff"))
    if not len(image_paths):
        raise ValueError(f"Could not find images in directory [{image_dir}] with band [{band}]")
    return image_paths


def get_aligned_images(
        images1: Sequence[Union[str, Path]],
        images2: Sequence[Union[str, Path]],
        date_window: int=15
) -> Dict[Path, List[Path]]:
    """For each image path in images1, retrieve all spatially aligned
    images that were collected within +-date_window days.

    Parameters
    ----------
    images1: Sequence[Union[str, Path]], len=M
        The paths to images to find spatially and temporally aligned images for.

    images2: Sequence[Union[str, Path]], len=N
        The paths images to search to find images aligned with images1.

    min_date: Optional[int]
        Specifies the inclusive minimum of a time window in days
        indicating whether two collects are close enough in time to be
        consdered temporally aligned. If None, there is no minimum
        date.

    max_date: Optional[np.datetime64], default=None
        Specifies the inclusive maximum of a time window in days
        indicating whether two collects are close enough in time to be
        consdered temporally aligned. If None, there is no maximum
        date.

    date_window: Tuple[Optional[float], Optional[float]]
        The minimum and non-inclusive maximum relative time window in
        days specifying whether two collects are close enough in time
        to be considered temporally aligned. If the minimum is None,
        there is no filter on the minimum relative date. Similarly, no
        maximum can be specified with a value of None.

    Return
    ------
    Dict[Path, List[Path]], len=M
        For each image path in images1 (dictionary keys), holds paths
        for all spatially and temporally aligned images from images2
        (dictionary values).

    """
    if len(date_window) != 2:
        raise ValueError(f"Expected a two element minimum and maximum relative date, not ({date_window}).")
    if (
        date_window[0] is not None
        and date_window[1] is not None
        and date_window[0] > date_window[1]
    ):
        raise ValueError(
            f"The minimum date ({date_window[0]}) must not be greater than maximum date ({date_window[1]})."
        )


    image2_data = {}
    for ii, image2 in enumerate(images2):
        image2 = Path(image2)
        parts = parse_filename_parts(image2, pos_float=False)
        key = parts["lat"] + parts["lon"]
        if key not in image2_data:
            image2_data[key] = []
        image2_data[key].append((image2, parts))

    aligned_images = OrderedDict()
    for ii, image1 in enumerate(images1):
        image1 = Path(image1)
        aligned_images[image1] = []
        image1_parts = parse_filename_parts(image1, pos_float=False)
        key = image1_parts["lat"] + image1_parts["lon"]
        if key in image2_data:
            for image2_path, image2_parts in image2_data[key]:

                aligned = True
                if date_window[0] is not None and (image2_parts["date"] - image1_parts["date"]).days < date_window[0]:
                    aligned = False
                if date_window[1] is not None and (image2_parts["date"] - image1_parts["date"]).days > date_window[1]:
                    aligned = False
                if aligned:
                    aligned_images[image1].append(image2_path)

    return aligned_images
