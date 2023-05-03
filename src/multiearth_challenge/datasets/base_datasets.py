from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

import multiearth_challenge.datasets.data_filters as df
import multiearth_challenge.datasets.relative_data_filters as rdf


"""This module holds base classes for datasets used to access
MultiEarth imagery.

"""


"""Dataset __getitem__ member functions will return a DatasetData type
which is a dictionary that holds an image of interest as well as the
image's source, bands, collection date, and the latitude / longitude of
the center of the image. The dictionary has keys:
    "image": np.ndarray - The image of interest.
    "data_source": str - The sensor or dataset source of the imagery.
    "bands": List[str] - The data bands that comprise the image channels.
                         If the data source is the Deforestation dataset
                         which does not have data bands, this
                         will be None.
    "lat_lon": Tuple[float, float] - A tuple holding the latitude and
                                     longitude in decimal degrees for
                                     the center of the image.
    "date": np.datetime64 - The collection date for the image.
"""
DatasetData = Dict[str, Union[np.ndarray, str, List[str], Tuple[float, float], np.datetime64, None]]


class NetCDFDataset:
    """A helper class for MultiEarth dataset classes. This class handles
    loading, filtering, and parsing a single MultiEarth NetCDF data file."""

    def __init__(
        self,
        netcdf_file: Union[str, Path],
        data_filters: Iterable[Callable[[xr.Dataset], Dict[str, np.ndarray]]],
        merge_bands: bool=False,
    ):
        """Note, this initialization will open a file handle to a NetCDF
        file. This file handle is released by calling the close member
        function of this class.

        Note, some functionality depends on the knowledge that
        MultiEarth data has a 0.02 degree latitude / longitude
        spacing.

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

        merge_bands: bool, default=False
            If True, returned images will have multiple channels in
            increasing order of frequency (e.g., red, green, blue for
            visible), co-pol before cross-pol, and with bands not
            originating from collected imagery coming last and in
            alphabetical order. The metadata returned with the imagery
            will also specify the channel order. If False, each band
            is treated as a separate sample.

        """
        self.merge_bands = merge_bands
        self.data = xr.open_dataset(netcdf_file, cache=False)

        # Modify the index coordinate to simply be a index position
        self.data.assign_coords(index = np.array(range(len(self.data.index))))
        self.data_source = self.data.attrs["data_source"]
        for data_filter in data_filters:
            selected_data = data_filter(self.data)
            self.data = self.data.isel(selected_data)
            if not len(self.data.index) or ("data_band" in self.data.coords and not len(self.data.data_band)):
                # Filtered all data
                self.data = None
                break

        # Index lat/lon positions in data for fast lookup.
        if self.data is not None:
            self.center_lat_lons = self.data.center_lat_lons.data
            self.collection_dates = self.data.collection_dates.data
            self.data_bands = None if "data_band" not in self.data.coords else self.data.data_band.data
            self.locations = np.unique(self.center_lat_lons, axis=0)
            lat_lons = np.rint(self.center_lat_lons*100).astype(int)
            lat_lons = list(map(tuple, lat_lons))
            self.pos_indices: Dict[Tuple[int, int], List[int]] = {}
            for ii, pos in enumerate(lat_lons):
                if pos not in self.pos_indices:
                    self.pos_indices[pos] = []
                self.pos_indices[pos].append(ii)

            self.use_separate_bands = False
            self.ordered_band_indices = None
            if self.merge_bands:
                self.ordered_band_indices = []
                for band in df.DATA_BANDS[self.data_source]:
                    if band in self.data.data_band:
                        self.ordered_band_indices.append(np.where(self.data.data_band == band)[0][0])
            elif "data_band" in self.data.coords:
                self.use_separate_bands = True                
        else:
            self.center_lat_lons = np.array([])
            self.collection_dates = np.array([])
            self.locations = np.array([])
            self.data_bands = np.array([])
            self.pos_indices = {}
            self.use_separate_bands = False
            self.ordered_band_indices = None

    def close(self) -> None:
        """Closes the NetCDF file handle owned by this class."""
        self.data.close()

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
        if self.data is None:
            return 0
        if self.use_separate_bands:
            return len(self.data.index) * len(self.data.data_band)
        return len(self.data.index)

    def get_indices_by_loc(
        self,
        loc: Sequence[float],
        relative_date_filter: Optional[rdf.RelativeDateFilter] = None,
        target_date: Optional[np.datetime64] = None,
    ) -> List[int]:
        
        """Returns the indices for imagery associated with the passed
        location.

        Parameters
        ----------
        loc: Sequence[float], len=2
            The latitude / longitude position of interest.

        relative_date_filter: rdf.RelativeDateFilter, default=None
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
        List[int]
            Returns a list of indices for images with center latitude
            and longitude that match the passed location.

        """
        pos_key = tuple(np.rint(np.array(loc) * 100).astype(int))
        if pos_key not in self.pos_indices:
            return [] # No matching images with that position
        loc_indices = self.pos_indices[pos_key]
        loc_indices = np.array(loc_indices)
        loc_filtered_dates = self.collection_dates[loc_indices]
        if relative_date_filter is not None:
            if target_date is None:
                raise ValueError(
                    f"Parameter relative_date_filter was passed so a target_date must be set as well."
                )
            date_mask = relative_date_filter.calc_relative_date_mask(loc_filtered_dates, target_date)
        else:
            date_mask = np.ones((len(loc_indices),), dtype=bool)
        idxs = list(loc_indices[date_mask])
        if self.use_separate_bands:
            # Include indices for data bands separately
            num_idxs = len(idxs)
            for ii in range(num_idxs):
                for jj in range(1, len(self.data.data_band)):
                    idxs.append(idxs[ii] + jj * len(self.data.index))
        return idxs

    def get_metadata(self, index: int) -> Tuple[np.ndarray, np.datetime64]:
        """Returns the position and date information associated with
        the passed index.

        Parameters
        ----------
        index: int
            The dataset index of interest.

        Returns
        -------
        Tuple[np.ndarray, np.datetime64]
            A tuple whose first element is a two element numpy array
            with the latitude / longitude of the center of the
            image. The second element holds the collection date of the
            image.

        """
        if self.use_separate_bands:
            img_index = index % len(self.data.index)
            band_index = index // len(self.data.index)
            lat_lon = self.center_lat_lons[img_index]
            date = self.collection_dates[img_index]
        else:
            lat_lon = self.center_lat_lons[index]
            date = self.collection_dates[index]
        return (lat_lon, date)

    def __getitem__(self, index: int) -> DatasetData:
        """Returns the image associated with the passed index along with
        position and date information. The image may be None if no
        imagery is contained in the NetCDF file. This will be the case
        for data supplied as test targets for the MultiEarth challenge
        where only the target metadata is supplied.

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

               "image": Optional[np.ndarray] - The image of
               interest. This will be None if no imagery is in the
               NetCDF file which will be the case for MultiEarth
               challenge test targets.

               "data_source": str - The sensor or dataset source of
                                    the imagery.
               "bands": List[str] - The data bands that comprise the
                                    image channels. If the data source
                                    is the Deforestation dataset which
                                    does not list multiple data bands,
                                    this will be None.
               "lat_lon": Tuple[float, float] - A tuple holding the
                                                latitude and longitude
                                                in decimal degrees for
                                                the center of the
                                                image.
               "date": np.datetime64 - The collection date for the image.

            The image is a numpy array with shape (C, H, W) where C is
            the image channels, H the image height, and W the image
            width. The bit depth of the imagery varies depending on
            the collecting sensor.

        """
        # Images will not exist if the dataset holds information for
        # MultieEarth challenge test targets
        images_exist = "images" in self.data.keys()
        if not images_exist:
            image = None
        if self.use_separate_bands:
            img_index = index % len(self.data.index)
            band_index = index // len(self.data.index)
            if images_exist:
                image = self.data.images[img_index, band_index].data
            lat_lon = self.center_lat_lons[img_index]
            date = self.collection_dates[img_index]
            bands = [str(self.data.data_band[band_index].data)]
        elif "data_band" not in self.data.coords:
            if images_exist:
                image = self.data.images[index].data
            lat_lon = self.center_lat_lons[index]
            date = self.collection_dates[index]
            bands = None
        else:
            if images_exist:
                # Remove redundant channel dimension
                image = self.data.images[index, self.ordered_band_indices, 0].data
            lat_lon = self.center_lat_lons[index]
            date = self.collection_dates[index]
            bands = [str(ii) for ii in self.data_bands[self.ordered_band_indices]]
        return {"image": image, "data_source": self.data_source, "bands": bands, "lat_lon": lat_lon, "date": date}


class MultiEarthDatasetBase:
    """A class for holding common logic used across MultiEarth dataset
    classes.

    """

    def __init__(
        self,
        source_files: Sequence[Path],
        target_files: Sequence[Path],
        source_data_filters: Iterable[Callable[[xr.Dataset], Dict[str, np.ndarray]]] = [],
        target_data_filters: Iterable[Callable[[xr.Dataset], Dict[str, np.ndarray]]] = [],
        merge_source_bands: bool = False,
        merge_target_bands: bool = False,
        relative_date_filter: Optional[rdf.RelativeDateFilter] = None,
        single_source_image: bool = True,
        error_on_empty: bool = True,
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

        source_data_filters: Iterable[Callable[[xr.Dataset], Dict[str, np.ndarray]]], default=[]
            A series of callables which specify what data held
            within the passed MultiEarth NetCDF image files should be
            included in this dataset. Each callable should return a
            boolean mask for each sample held in a single NetCDF file
            to indicate whether it should be included or
            discarded. Standard filters date, position, and sensor
            band are supplied in data_filters.py.

        target_data_filters: Iterable[Callable[[xr.Dataset], Dict[str, np.ndarray]]], default=[]
            A series of callables which specify what data held
            within the passed MultiEarth NetCDF target files should be
            included in this dataset.  Each callable should return a
            boolean mask for each sample held in a single NetCDF file
            to indicate whether it should be included or
            discarded. Standard filters for date, position, and sensor
            band are supplied in data_filters.py.

        merge_source_bands: bool, default=False
            If True, returned source images will have multiple
            channels in increasing order of frequency (e.g., red,
            green, blue for visible), co-pol before cross-pol, and
            with bands not originating from collected imagery coming
            last and in alphabetical order. The metadata returned with
            the imagery will also specify the channel order. If False,
            each band is treated as a separate sample.

        merge_target_bands: bool, default=False
            If True, returned target images will have multiple
            channels in increasing order of frequency (e.g., red,
            green, blue for visible), co-pol before cross-pol, and
            with bands not originating from collected imagery coming
            last and in alphabetical order. The metadata returned with
            the imagery will also specify the channel order. If False,
            each band is treated as a separate sample.

        relative_date_filter: rdf.RelativeDateFilter, default=None
            For a given target image, filters the imagery returned
            with it based on its relative date. For example, for a
            fire segmentation task you might specify that only
            imagery collected within a week after the date associated
            with the burn date be returned.

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
        if not len(source_files):
            raise ValueError(f"No source image files passed.")
        if not len(target_files):
            raise ValueError(f"No target image files passed.")
        self.single_source_image = single_source_image
        self.source_data = [
            NetCDFDataset(ii, source_data_filters, merge_source_bands)
            for ii in source_files
        ]

        self.relative_date_filter = relative_date_filter
        self.target_data = [
            NetCDFDataset(ii, target_data_filters, merge_target_bands)
            for ii in target_files
        ]

        # Save mapping from absolute sample index to source and target
        # indices to extract sample data from appropriate dataset
        # locations.
        # self.indices will hold a list of tuples where the first
        # element is a list of source dataset / image indices and the
        # second element is the target dataset / image index.
        self.indices = []
        for targ_ds_idx, targ_dataset in enumerate(self.target_data):
            for targ_idx in range(len(targ_dataset)):
                self.indices.append(
                    (
                        [],
                        {
                            "target_ds_idx": targ_ds_idx,
                            "target_img_idx": targ_idx,
                        },
                    )
                )
                found_match = False
                for source_ds_idx, source_dataset in enumerate(self.source_data):
                    target_pos, target_date = targ_dataset.get_metadata(targ_idx)
                    source_idxs = source_dataset.get_indices_by_loc(
                        target_pos,
                        self.relative_date_filter,
                        target_date,
                    )
                    if len(source_idxs):
                        found_match = True
                    for source_idx in source_idxs:
                        if self.single_source_image and len(self.indices[-1][0]):
                            self.indices.append(
                                (
                                    [],
                                    {
                                        "target_ds_idx": targ_ds_idx,
                                        "target_img_idx": targ_idx,
                                    },
                                )
                            )
                        self.indices[-1][0].append(
                            {
                                "source_ds_idx": source_ds_idx,
                                "source_img_idx": source_idx,
                            }
                        )
                        new_target = False
                if not found_match:
                    # No source images satisfy the location and date
                    # requirements, remove the target
                    self.indices.pop()

        if error_on_empty and not len(self.indices):
            raise ValueError(f"After filtering, no common source and target data remains")

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
        any data filters that have been set.

        The DatasetData type is a dictionary that holds sample imagery
        as well as the image's collection date and the latitude /
        longitude of the center of the image. The dictionary has keys:

            "image": np.ndarray - The image of interest.

            "data_source": str - The sensor or dataset source of
            the imagery.

            "bands": List[str] - The data bands that comprise the
            image channels. If the data source is the Deforestation
            dataset which does not list multiple data bands, this
            will be None.

            "lat_lon": Tuple[float, float] - A tuple holding the
            latitude and longitude in decimal degrees for the center
            of the image.

            "date": np.datetime64 - The collection date for the image.

        The returned image is a numpy array with shape
        (C0, H0, W0) where C0 is the image channels, H0 the image
        height, and W0 the image width. The bit depth of the imagery
        varies depending on the collecting sensor. The paired target
        image is also a numpy array with dimensions C1, H1, W1.

        """
        source_idxs, target_idx = self.indices[index]
        source_data = [
            self.source_data[idxs["source_ds_idx"]][idxs["source_img_idx"]]
            for idxs in source_idxs
        ]
        target_data = self.target_data[target_idx["target_ds_idx"]][target_idx["target_img_idx"]]
        return (source_data, target_data)
