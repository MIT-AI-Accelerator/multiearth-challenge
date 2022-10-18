from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
import xarray as xr
  

class _NetCDFDataset():
    """A helper class for MultiEarth dataset classes.  This class handles
    loading, filtering, and parsing a single MultiEarth NetCDF data file."""

    def __init__(self, netcdf_file, data_filters, data_transforms):
        """Standard init.

        Parameters
        ----------
        netcdf_file: Union[str, Path]
            The path to the MultiEarth NetCDF data file to load.

        data_filters: Sequence[Callable[[xr.Dataset], Sequence[bool]]
            A sequence of callables which specify what data held
            within a MultiEarth NetCDF file should be included in this
            dataset.  Each callable should return a boolean mask for
            each sample held in the NetCDF file to indicate whether it
            should be included or discarded. Standard filters for
            date, position, and sensor band are supplied in
            data_filters.py

        data_transforms: Sequence[Callable]
            A sequence of callables which are passed a set of images
            and return a sequence of transformed images.

        """
        self.data_transforms = data_transforms
        with xr.open_dataset(netcdf_file, cache=False) as in_data:
            mask = np.array([True] * len(in_data.index))
            for filter in data_filters:
                mask &= np.asarray(filter(in_data))
            if np.any(mask):
                self.data = in_data.isel(index=mask)
                self.locations = self._get_center_lat_lons()
            else:
                self.data = None
                self.locations = []

    def _get_center_lat_lons(self):
        """Retrieves the unique set of center latitudes and longitudes for all
        image chips held by the MultiEarth NetCDF data file parsed by
        this NetCDFDataset.

        Note: This function leverages the fact that MultiEarth image
        chips have center latitudes and longitudes that fall on
        exactly hundredths of a degree.

        Returns
        -------
        np.array
            The unique set of center latitudes and longitudes for the held data.
        """
        locations = set()
        for pos in self.data.center_lat_lons.data:
            locations.add(tuple(pos))
        return list(locations)

    def __len__(self):
        """Gets the number of samples in this dataset.

        Returns
        -------
        int
            The number of samples in this dataset.
        """
        return len(self.locations)

    def _get_data_by_loc(self, loc):
        """Returns all imagery associated with the passed location.  The
        imagery will be modified by any dataset transforms that have
        been set.
        

        Note: The passed location is assumed to be one of the held
        locations and therefore exactly equal.
        
        Parameters
        ----------
        loc: Sequence[float], len=2
            The latitude / longitude position of interest.

        Returns
        -------
        Sequence[Any]
            Returns images with center latitude and longitude that
            match the passed location.  The return type will nominally
            be a numpy array with shape (N, C, H, W) where N is the
            number of images, C the image channels, H the image
            height, and W the image width.  This shape and return type
            may be modified by this dataset's data transforms.  The bit
            depth of the imagery varies depending on the collecting
            sensor and applied data transforms.

        """
        mask = self.data.center_lat_lons.data == loc
        # Two columns of mask are identical
        mask = mask[:, 0]
        images = self.data.images.data[mask]
        for transform in self.data_transforms:
            images = transform(images)
        return images

    def __getitem__(self, index):
        """Returns all imagery associated with the latitude / longitude
        location corresponding to the passed index. The imagery will
        be modified by any dataset transforms that have been set.
        
        Parameters
        ----------
        index: int
            The location index to retrieve imagery for.

        Returns
        -------
        Sequence[Any]
            Returns images with center latitude and longitude that
            match the passed location.  The return type will nominally
            be a numpy array with shape (N, C, H, W) where N is the
            number of images, C the image channels, H the image
            height, and W the image width.  This shape and return type
            may be modified by this dataset's data transforms.  The bit
            depth of the imagery varies depending on the collecting
            sensor and applied data transforms.

        """
        return self._get_data_by_loc(self.locations[index])
    
class SegmentationDataset():
    """A dataset class that loads MultiEarth data and returns image chips
    at a paired with deforestation segmentation masks at the same locations.
    """

    def __init__(
            self,
            image_files,
            deforestation_files,
            image_data_filters=[],
            deforestation_data_filters=[],
            image_data_transforms=[],
            deforestation_data_transforms=[],
            error_on_empty=True,
    ):
        """Standard init.
    
        Parameters
        ----------
        image_files: Sequence[Path]
            MultiEarth NetCDF image files whose data should be
            considered for this dataset.

        deforestation_files: Sequence[Path]
            MultiEarth NetCDF deforestation segmentation files whose
            data should be considered for this dataset

        image_data_filters: Sequence[Callable[[xr.Dataset], Sequence[bool]], default=[]
            A sequence of callables which specify what data held
            within the passed MultiEarth NetCDF image files should be
            included in this dataset.  Each callable should return a
            boolean mask for each sample held in a single NetCDF file
            to indicate whether it should be included or
            discarded. Standard filters date, position, and sensor
            band are supplied in data_filters.py.

        deforestation_data_filters: Sequence[Callable[[xr.Dataset], Sequence[bool]], default=[]
            A sequence of callables which specify what data held
            within the passed MultiEarth NetCDF deforestation
            segmentation files should be included in this dataset.
            Each callable should return a boolean mask for each sample
            held in a single NetCDF file to indicate whether it should
            be included or discarded. Standard filters for date,
            position, and sensor band are supplied in data_filters.py.
            
        image_data_transforms: Sequence[Callable]
            A sequence of callables which are passed a set of images
            and return a sequence of transformed images.

        deforestation_data_transforms: Sequence[Callable]
            A sequence of callables which are passed a set of
            deforestation segmentation masks and return a sequence of
            transformed masks.
        
        error_on_empty: bool, default=True
            If True, if no images or deforestation segmentation masks
            remain after data filtering, raise a ValueError, otherwise
            this dataset will have length 0.

        """
        if not len(image_files):
            raise ValueError(f"No image files passed.")
        if not len(deforestation_files):
            raise ValueError(f"No deforestation files passed.")

        self.image_data_transforms = image_data_transforms
        self.deforestation_data_transforms = deforestation_data_transforms
        self.image_data = [_NetCDFDataset(ii, image_data_filters, self.image_data_transforms) for ii in image_files]
        self.deforestation_data = [_NetCDFDataset(ii, deforestation_data_filters, self.deforestation_data_transforms) for ii in deforestation_files]

        image_locations = set()
        for data in self.image_data:
            image_locations.update(data.locations)
        
        deforestation_locations = set()
        for data in self.deforestation_data:
            deforestation_locations.update(data.locations)
        if error_on_empty and not len(image_locations):
            raise ValueError(f"After filtering, no image data remains") 

        if error_on_empty and not len(deforestation_locations):
            raise ValueError(f"After filtering, no deforestation data remains") 
        
        if image_locations != deforestation_locations:
            image_pos = image_locations - deforestation_locations
            deforestation_pos = deforestation_locations - image_locations
            raise ValueError(f"Mismatch between filtered image positions and filtered deforestation masks.\n"
                             f"Positions contained only in image data: {image_pos}\n"
                             f"Positions contained only in deforestation data: {deforestation_pos}")
        self.locations = list(image_locations)
        
    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        loc = self.locations[index]
        image_list = []
        for data in self.image_data:
            image_list.append(data._get_data_by_loc(loc))
        if not len(image_list):
            raise RuntimeError(f"Did not find iamge data for location {loc}")
        if isinstance(image_list[0], np.ndarray):
            images = np.stack(image_list)
        elif torch.is_tensor(image_list[0]):
            images = torch.cat(image_list)
        else:
            raise TypeError(f"Unrecognized image type of {type(image_list[0])}")
            
        mask_list = []
        for data in self.deforestation_data:
            mask_list.append(data._get_data_by_loc(loc))
        if not len(mask_list):
            raise RuntimeError(f"Did not find deforestation data for location {loc}")
        if isinstance(mask_list[0], np.ndarray):
            masks = np.concatenate(mask_list, axis=0)
        elif torch.is_tensor(mask_list[0]):
            masks = torch.cat(mask_list)
        else:
            raise TypeError(f"Unrecognized deforestation mask type of {type(mask_list[0])}")
            
        return (images, masks)
