from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
import xarray as xr

import multiearth_challenge.dataset.relative_data_filters as rdf


"""This module holds base classes for datasets used to access
MultiEarth imagery.

"""


"""Dataset __getitem__ member functions will return a DatasetData type
which is a dictionary that holds an image of interest as well as the
image's collection date and the latitude / longitude of the center of
the image. The dictionary has keys:
    "image": Any - The image of interest.
    "lat_lon": Tuple[float, float] - A tuple holding the latitude and
                                     longitude in decimal degrees for
                                     the center of the image.
    "date": np.datetime64 - The collection date for the image.
"""
DatasetData = Dict[str, Union[Tuple[float, float], np.datetime64, Any]]

"""A list of sensors and their corresponding Red, Green, Blue (RGB)
visible bands in RGB order."""
RGB_BANDS = {
    "Landsat-5": ["SR_B3", "SR_B2", "SR_B1"],
    "Landsat-8": ["SR_B4", "SR_B3", "SR_B2"],
    "Sentinel-2": ["B4", "B3", "B2"],
}


class NetCDFDataset(torch.Dataset):
    """A helper class for MultiEarth dataset classes. This class handles
    loading, filtering, and parsing a single MultiEarth NetCDF data file."""

    def __init__(
        self,
        netcdf_file: Union[str, Path],
        data_filters: Iterable[Callable[[xr.Dataset], Sequence[bool]]],
        data_transforms: Iterable[Callable[[Any], Any]],
        merge_rgb: bool,
        merge_vv_vh: bool,
        target_dataset=None,
    ):
        """Note, this initialization will open a file handle to a NetCDF
        file. This file handle is released by calling the close member
        function of this class.

        Parameters
        ----------
        netcdf_file: Union[str, Path]
            The path to the MultiEarth NetCDF data file to load.

        data_filters: Iterable[Callable[[xr.Dataset], Sequence[bool]]
            A series of callables which specify what data held
            within a MultiEarth NetCDF file should be included in this
            dataset. Each callable should return a boolean mask for
            each sample held in the NetCDF file to indicate whether it
            should be included or discarded. Standard filters for
            date, position, and sensor band are supplied in
            data_filters.py

        data_transforms: Iterable[Callable[[Any], Any]]
            A series of callables which are each passed an image
            and return a transformed image.

        merge_rgb: bool
            If True, when retrieving images will merge separate red,
            green, and blue visible imagery channels into a single
            image.

            Note, this will only stack the 3 visible bands in red,
            green, blue order and will not perform any color balancing
            across the channels.

            If the passed NetCDF file is for a sensor that does not
            collect RGB imagery, this parameter is ignored. Otherwise,
            if all 3 visible bands are not available, this parameter
            raises a ValueError.

        merge_vv_vh: bool, default=False
            If True, when retrieving SAR images, merges separate VV
            and VH polarization channels into a single image.

            Note, this will only stack the 2 bands in VV, VH order and
            will not perform any normalization across the channels.

            If true and both polarization bands are not available due
            to a data filter, this parameter raises a ValueError.

            This parameter is ignored for sensors that do not collect
            SAR imagery.

        """
        self.data_transforms = data_transforms
        self.data = xr.open_dataset(netcdf_file, cache=False)
        self.data_source = self.data.attrs["data_source"]
        self.merge_rgb = merge_rgb
        for filter in data_filters:
            selected_data = filter(self.data)
            self.data = self.data.isel(selected_data)
            if not np.all(self.data.images.shape):
                # Filtered all data
                self.data = None
                break
        self.locations = [] if self.data is None else self._get_center_lat_lons()
        self.rgb_band_names = None
        if self.merge_rgb:
            if self.data_source in rgb_bands:
                self.rgb_band_names = rgb_bands[self.data_source]
                for band_name in self.rgb_band_names:
                    if band_name not in self.data.data_band:
                        raise ValueError(
                            f"Cannot merge RGB bands since band ({band_name}) for sensor ({self.data_source}) is not part of the dataset."
                        )

    def close(self) -> None:
        """Closes the NetCDF file handle owned by this class."""
        self.data.close()

    def _get_center_lat_lons(self) -> List[Tuple[float, float]]:
        """Retrieves the unique set of center latitude/longitude pairs for all
        images in this dataset.

        Note: This function leverages the fact that MultiEarth image
        chips have center latitudes and longitudes that fall on
        exactly hundredths of a degree.

        Returns
        -------
        List[Tuple[float, float]]
            The unique set of center latitudes and longitudes for the held data.

        """
        locations = set()
        for pos in self.data.center_lat_lons.data:
            locations.add(tuple(pos))
        return list(locations)

    def get_num_locations(self) -> int:
        """Get the number of unique center latitude/longitude pairs for all
        images in this dataset.

        Returns
        -------
        int
            The number of unique positions in this dataset.

        """
        return len(self.locations)

    def __len__(self) -> int:
        """Get the number of images in this dataset.

        Returns
        -------
        int
            The number of images in this dataset.
        """
        return len(self.data.index)

    def get_data_by_loc(
        self,
        loc: Sequence[float],
        relative_date_filter: mcr.RelativeDateFilter = None,
        target_date: np.datetime64 = None,
    ) -> Sequence[Any]:
        """Returns all imagery associated with the passed location. The
        imagery will be modified by any dataset transforms that have
        been set.

        Note: The passed location is assumed to be one of the held
        locations and therefore exactly equal.

        Parameters
        ----------
        loc: Sequence[float], len=2
            The latitude / longitude position of interest.

        relative_date_filter: mcr.RelativeDateFilter, default=None
            For a given target date, filters the imagery
            returned based on its relative date. For example,
            you can specify that only imagery collected within a week
            after the date associated with a fire segmentation mask be
            returned. If this is not None, the target_date must be
            set as well.

        target_date: np.datetime64
            If relative_date_filter is specified, the target date to
            calculate a relative minimum and maximum date around.

        Returns
        -------
        List[Any]
            Returns a list of images with center latitude and
            longitude that match the passed location. The return type
            of the image will nominally be a list of numpy arrays with
            shape (C, H, W) where C the image channels, H the image
            height, and W the image width. This shape and return type
            may be modified by this dataset's data transforms. The bit
            depth of the imagery varies depending on the collecting
            sensor and applied data transforms.

        """
        date_filtered_data = self.data
        if relative_date_filter is not None:
            if target_date is None:
                raise ValueError(
                    f"Parameter relative_date_filter was passed so a target_date must be set as well."
                )
            date_mask = relative_date_filter(self.data, target_date)
            date_filtered_data = self.data.isel(date_mask)
        mask = np.all(date_filtered_data.center_lat_lons.data == loc, axis=1)
        images = [image for image in self.data.images.data[mask]]
        for transform in self.data_transforms:
            for image in images:
                image = transform(image)
        return images

    def __getitem__(self, index: int) -> DatasetData:
        """Returns the image associated with the passed index along with
        position and date information. The image will be modified by
        any dataset transforms that have been set.

        Parameters
        ----------
        index: int
            The image index of interest.

        Returns
        -------
        DatasetData
            A dictionary that holds the image of interest as well as
            the image's collection date and the latitude / longitude
            of the center of the image. The dictionary has keys:
               "image": Any - The image of interest.
               "lat_lon": Tuple[float, float] - A tuple holding the
                                                latitude and longitude
                                                in decimal degrees for
                                                the center of the
                                                image.
               "date": np.datetime64 - The collection date for the image.

            The image return type will nominally be a numpy array with
            shape (N, C, H, W) where N is the number of images, C the
            image channels, H the image height, and W the image
            width. This shape and return type may be modified by this
            dataset's data transforms. The bit depth of the imagery
            varies depending on the collecting sensor and applied data
            transforms.

        """
        image = self.data.images.data[index]
        lat_lon = self.data.center_lat_lons.data[index]
        date = self.data.collection_dates.data[index]
        for transform in self.data_transforms:
            image = transform(image)
        return {"lat_lon": lat_lon, "date": date, "image": image}


class MultiEarthDatasetBase(torch.Dataset):
    """A class for holding common logic used across MultiEarth dataset
    classes.

    """

    def __init__(
        self,
        source_files: Iterable[Path],
        target_files: Iterable[Path],
        source_data_filters: Iterable[Callable[[xr.Dataset], Sequence[bool]]] = [],
        target_data_filters: Iterable[Callable[[xr.Dataset], Sequence[bool]]] = [],
        source_data_transforms: Iterable[Callable[[Any], Any]] = [],
        target_data_transforms: Iterable[Callable[[Any], Any]] = [],
        relative_date_filter: rdf.RelativeDateFilter = None,
        merge_rgb: bool = False,
        merge_vv_vh: bool = False,
        error_on_empty: bool = True,
        single_source_image: bool = True,
    ):
        """Note, this initialization will open file handles to a NetCDF
        file. These file handles are released by calling the close
        member function of this class.

        Parameters
        ----------
        source_files: Iterable[Path]
            MultiEarth image NetCDF files whose data should be
            considered for this dataset.

        target_files: Iterable[Path]
            MultiEarth NetCDF files whose images are considered truth
            targets for this dataset. For example, deforestation masks
            in an image segmentation task.

        source_data_filters: Iterable[Callable[[xr.Dataset], Sequence[bool]]], default=[]
            A series of callables which specify what data held
            within the passed MultiEarth NetCDF image files should be
            included in this dataset. Each callable should return a
            boolean mask for each sample held in a single NetCDF file
            to indicate whether it should be included or
            discarded. Standard filters date, position, and sensor
            band are supplied in data_filters.py.

        target_data_filters: Iterable[Callable[[xr.Dataset], Sequence[bool]]], default=[]
            A series of callables which specify what data held
            within the passed MultiEarth NetCDF target files should be
            included in this dataset.  Each callable should return a
            boolean mask for each sample held in a single NetCDF file
            to indicate whether it should be included or
            discarded. Standard filters for date, position, and sensor
            band are supplied in data_filters.py.

        source_data_transforms: Iterable[Callable[[Any], Any]], default=[]
            A series of callables which are each passed an image
            and return a transformed image.

        target_data_transforms: Iterable[Callable[[Any], Any]], default=[]
            A series of callables which are each passed an target
            image and return a transformed target image.

        relative_date_filter: rdf.RelativeDateFilter, default=None
            For a given target image, filters the imagery returned
            with it based on its relative date. For example, for a
            fire segmentation task you might specify that only
            imagery collected within a week after the date associated
            with the burn date be returned.

        merge_rgb: bool, default=False
            If True, when retrieving images from sensors that collect
            visible imagery, merges separate red, green, and blue
            channels into a single image.

            Note, this will only stack the 3 visible bands in red,
            green, blue order and will not perform any color balancing
            across the channels.

            If true and all 3 visible bands are not available due to a
            data filter, this parameter raises a ValueError.

            This parameter is ignored for sensors that do not collect
            visible imagery.

        merge_vv_vh: bool, default=False
            If True, when retrieving SAR images, merges separate VV
            and VH polarization channels into a single image.

            Note, this will only stack the 2 bands in VV, VH order and
            will not perform any normalization across the channels.

            If true and both polarization bands are not available due
            to a data filter, this parameter raises a ValueError.

            This parameter is ignored for sensors that do not collect
            SAR imagery.

        error_on_empty: bool, default=True
            If True, if no source or target image remain after
            data filtering, raise a ValueError, otherwise this dataset
            will have length 0.

        single_source_image: bool, default=True
            If True, for each target image only a single source image
            is returned in a unique pair. A single source image may be
            paired with multiple target images and vice-versa
            depending on data filters applied. If False, each target
            image is returned with all source images at the same
            location that satisfy applied data filters.

        """
        if not len(source_files):
            raise ValueError(f"No source image files passed.")
        if not len(target_files):
            raise ValueError(f"No target image files passed.")

        self.source_data = [
            NetCDFDataset(ii, source_data_filters, source_data_transforms, merge_rgb)
            for ii in source_files
        ]

        self.relative_date_filter = relative_date_filter
        if self.relative_date_filter is not None:
            # Create a data filter to remove all source images that do not
            # have a source image that satisfies the source data filters.
            target_datasets = [xr.open_dataset(filename) for filename in target_files]
            target_date_loc_match = df.MatchedDateLocFilter(
                target_datasets,
                relative_min_date=relative_date_filter.min_date,
                relative_max_date=relative_date_filter.max_date,
                date_include=relative_date_filter.include,
            )
            # Ensure the source_date_loc_match filter is processed
            # first since it was generated with all target data before
            # filtering.
            source_data_filters.insert(0, target_date_loc_match)
            for dataset in target_datasets:
                dataset.close()

            # Create a data filter to remove all target images that do not
            # have a source image that satisfies the source data filters.
            source_datasets = [xr.open_dataset(filename) for filename in source_files]
            source_date_loc_match = df.MatchedDateLocFilter(
                source_datasets,
                relative_min_date=relative_date_filter.min_date,
                relative_max_date=relative_date_filter.max_date,
                date_include=relative_date_filter.include,
            )
            # Ensure the target_date_loc_match filter is processed
            # first since it was generated with all source data before
            # filtering.
            target_data_filters.insert(0, source_date_loc_match)
            for dataset in source_datasets:
                dataset.close()

        self.target_data = [
            NetCDFDataset(
                ii, target_data_filters, target_data_transforms, merge_rgb=False
            )
            for ii in target_files
        ]
        image_locations = set()
        for data in self.source_data:
            image_locations.update(data.locations)

        target_locations = set()
        for data in self.target_data:
            target_locations.update(data.locations)
        if error_on_empty and not len(image_locations):
            raise ValueError(f"After filtering, no image data remains")

        if error_on_empty and not len(target_locations):
            raise ValueError(f"After filtering, no target data remains")

        if image_locations != target_locations:
            image_pos = image_locations - target_locations
            target_pos = target_locations - image_locations
            raise ValueError(
                f"Mismatch between filtered image positions and filtered target masks.\n"
                f"Positions contained only in image data: {image_pos}\n"
                f"Positions contained only in target data: {target_pos}"
            )
        self.locations = list(image_locations)

        # Save mapping from absolute sample index to source and target
        # indices to extract sample data from appropriate dataset locations
        self.indices = []
        for targ_ds_idx, targ_dataset in enumerate(target_data):
            for targ_img_idx, (targ_data) in enumerate(targ_dataset):
                sample_data = []
                for source_ds_idx, source_dataset in enumerate(self.source_data):
                    source_imgs = source_dataset.get_data_by_loc(
                        targ_data["lat_lon"],
                        self.relative_date_filter,
                        targ_date["date"],
                    )
                    if not len(source_imgs):
                        raise ValueError(
                            f"Failed to find source images for target image ({targ_img_idx}) in target dataset ({targ_ds_idx})"
                        )
                    for source_img_idx, source_img in enumerate(source_imgs):
                        if source_img_idx == 0 or self.single_source_image:
                            self.indices.append(
                                (
                                    [],
                                    {
                                        "target_ds_idx": targ_ds_idx,
                                        "target_img_idx": targ_img_idx,
                                    },
                                )
                            )
                        self.indices[-1][0].append(
                            {
                                "source_ds_idx": source_ds_idx,
                                "source_img_idx": source_img_idx,
                            }
                        )

    def close(self) -> None:
        """Closes the NetCDF file handles owned by this class."""
        for data in self.source_data:
            data.close()
        for data in self.target_data:
            data.close()

    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[List[DatasetData], DatasetData]:
        """Returns paired source and target data held by this dataset.

        Parameters
        ----------
        index: int
            The location index to retrieve imagery for.

        Returns
        -------
        Tuple[List[DatasetData], DatasetData]
        A tuple where the first element is list of source data meant
        to serve as an input to a model. 

        The second element holds target data at the same
        location. Each sample returned will be a unique pairing of
        data, but source data may be paired with more than one target
        and target data and vice-versa. The data will be filtered by
        any data filters and modified by any transforms that have been
        set.

        The DatasetData type is a dictionary that holds sample imagery
        as well as the image's collection date and the latitude /
        longitude of the center of the image. The dictionary has keys:

            "image": Any - The image of interest.

            "lat_lon": Tuple[float, float] - A tuple holding the
            latitude and longitude in decimal degrees for the center
            of the image.

           "date": np.datetime64 - The collection date for the image.

        The return type of each image will nominally be a numpy array
        with shape (C0, H0, W0) where C0 is the image channels, H0 the
        image height, and W0 the image width. This shape and return
        type may be modified by this dataset's data transforms. The
        bit depth of the imagery varies depending on the collecting
        sensor and applied data transforms. The paired target image
        will also nominally be a numpy array with dimensions (C1, H1,
        W1) that may be arbitrarily modified by specified data
        transforms.

        """
        source_idxs, target_idx = self.indices[index]
        source_data = [
            self.source_data[idxs["source_ds_idx"]][idxs["source_img_idx"]]
            for idxs in source_idxs
        ]
        target_data = self.target_data[idxs["target_ds_idx"]][idxs["target_img_idx"]]
        return (source_data, target_data)
