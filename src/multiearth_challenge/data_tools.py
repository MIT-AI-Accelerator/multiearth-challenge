from collections import OrderedDict
import datetime
import json
from pathlib import Path
from typing import Dict, List, Sequence, Union, Any, Optional


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

    date_window: int, default=15
        A window in +- days specifying whether two collects are close
        enough in time to be considered aligned (inclusive).

    Return
    ------
    Dict[Path, List[Path]], len=M
        For each image path in images1 (dictionary keys), holds paths
        for all spatially and temporally aligned images from images2
        (dictionary values).

    """
    if date_window <= 0:
        raise ValueError(f"Passed parameter date_window should be a positive, not [{date_window}]")

    image2_data = {}
    for ii, image2 in enumerate(images2):
        parts = parse_filename_parts(image2, pos_float=False)
        key = parts["lat"] + parts["lon"]
        if key not in image2_data:
            image2_data[key] = []
        image2_data[key].append((Path(image2), parts))

    aligned_images = OrderedDict()
    for ii, image1 in enumerate(images1):
        aligned_images[Path(image1)] = []
        image1_parts = parse_filename_parts(image1, pos_float=False)
        key = image1_parts["lat"] + image1_parts["lon"]
        if key in image2_data:
            for image2_path, image2_parts in image2_data[key]:
                if abs((image1_parts["date"] - image2_parts["date"]).days) <= date_window:
                    aligned_images[image1].append(image2_path)

    return aligned_images


def get_sar_to_eo_aligned_images(
        base_dir: Union[str, Path],
        date_window: int=2,
        output_file: Optional[Path]=None
)-> Dict[Path, List[Path]]:
    """Given a directory that holds sub-directories 'sent1', and 'sent2',
    for each VV or VH SAR image in 'sent1', will find and return all
    B4, B3, or B2 band (RGB) images in 'sent2' that are spatially and
    temporally aligned.

    Parameters
    ----------
    base_dir: Union[str, Path]
        A parent directory to sub-directories 'sent1' and 'sent2' that
        hold VV / VH SAR and B4 / B3 / B2 band satellite images respectively.

    date_window: int, default=15
        A window in +- days specifying whether two collects are close
        enough in time to be considered aligned (inclusive).

    output_file: Optional[Path], default=None
        Optional path to an output JSON file to write the aligned
        images filenames (not the full paths).  The JSON file will
        have format

        {
          "ex_sar_file1.tiff": ["ex_eo_file1.tiff", "ex_eo_file2.tiff"],
          ...
        }

    Return
    ------
    Dict[Path, List[Path]]
        The path for each VV or VH SAR image path found in
        sub-directory 'sent1' serves as dictionary keys. The
        corresponding dictionary value is a list of all spatially and
        temporally aligned B4, B2, or B2 band (RGB) images from
        sub-directory 'sent2'.

    """
    sent1_dir = Path(base_dir, "sent1")
    if not sent1_dir.is_dir():
        raise ValueError(f"Passed directory [{base_dir}] does not contain sub-directory 'sent1'")
    sent2_dir = Path(base_dir, "sent2")
    if not sent2_dir.is_dir():
        raise ValueError(f"Passed directory [{base_dir}] does not contain sub-directory 'sent2'")

    sent1_vv_images = get_image_paths(sent1_dir, "VV")
    sent1_vh_images = get_image_paths(sent1_dir, "VH")
    sent1_images = sorted(sent1_vv_images + sent1_vh_images)

    sent2_b4_images = get_image_paths(sent2_dir, "B4")
    sent2_b3_images = get_image_paths(sent2_dir, "B3")
    sent2_b2_images = get_image_paths(sent2_dir, "B2")
    sent2_images = sorted(sent2_b4_images + sent2_b3_images + sent2_b2_images)

    aligned_imgs = get_aligned_images(sent1_images, sent2_images, date_window)

    if output_file is not None:
        aligned_imgs_filenames = OrderedDict()
        for key, val in aligned_imgs.items():
            aligned_imgs_filenames[key.name] = [ii.name for ii in val]
        with open(output_file, 'w') as fout:
            json.dump(aligned_imgs_filenames, fout, indent=4, sort_keys=True)

    return aligned_imgs
