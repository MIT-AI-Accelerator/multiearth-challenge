import datetime
from typing import Dict, Optional, Union

import numpy as np
import numpy.ma as ma
import xarray as xr


"""This module holds functions for filtering MultiEarth source imagery
based on various conditions relative to a target image's
metadata. When passed as a parameter to a MultiEarth dataset this
enables filtering source data that is paired with each target based on
the sample's target data characteristics.

"""


class RelativeDateFilter:
    """Filters source imagery based on it's relative collection date
    compared to a target image's collection date.  For example, if you
    want to train on imagery earlier than one week before the target
    image was collected, you can specify a relative_min_date of None
    and a relative_max_date of np.timedelta64(-7, 'D').

    """

    def __init__(
        self,
        *,
        relative_min_date: Optional[Union[int, np.timedelta64]] = None,
        relative_max_date: Optional[Union[int, np.timedelta64]] = None,
        include: bool = True,
    ):
        """Parameters
        ----------
        relative_min_date: Optional[np.timedelta64], default=None
            Specifies the inclusive minimum of the relative date
            interval of interest. Negative values indicate a time
            before the target date and positive after. If an integer
            is passed, the relative_min_date is in units of days. If
            None, there is no minimum date.

        relative_max_date: Optional[np.timedelta64], default=None
            Specifies the inclusive maximum of the relative date
            interval of interest. Negative values indicate a time
            before the target date and positive after. If an integer
            is passed, the relative_min_date is in units of days. If
            None, there is no maximum date.

        include: bool, default=True
            If True, will include all imagery within the inclusive
            interval specified by the relative_min_date and
            relative_max_date. Otherwise, will exclude imagery within
            the date interval.

        """
        self.min_date = relative_min_date
        self.max_date = relative_max_date
        if self.min_date is not None and not isinstance(self.min_date, np.timedelta64):
            self.min_date = np.timedelta64(self.min_date, 'D')
        if self.max_date is not None and not isinstance(self.max_date, np.timedelta64):
            self.max_date = np.timedelta64(self.max_date, 'D')

        if (
            self.min_date is not None
            and self.max_date is not None
            and self.min_date > self.max_date
        ):
            raise ValueError(
                f"Passed parameter relative_min_date ({self.min_date}) must not be greater than relative_max_date ({self.max_date})."
            )
        self.include = include

    def calc_relative_date_mask(self, collection_dates: np.ndarray, target_date: np.datetime64) -> np.ndarray:
        """Calculates a boolean mask based on the filter parameters.

        Parameters
        ----------
        collection_dates: np.array
            An array of numpy datetime64 dates to filter.

        target_date: np.datetime64
            The target date to calculate a relative minimum and maximum date around.

        Returns
        -------
        np.ndarray
            A numpy boolean mask specifying which images satisfy the
            filter parameters used to initialize this
            RelativeDateFilter. It will have length equal to the
            length of the passed collection_dates array.

        """
        if isinstance(target_date, datetime.date):
            target_date = np.datetime64(target_date)
        if self.min_date is not None and self.max_date is not None:
            mask = ma.masked_inside(
                collection_dates,
                target_date + self.min_date,
                target_date + self.max_date,
            ).mask
        elif self.min_date is not None:
            mask = collection_dates >= (target_date + self.min_date)
        elif self.max_date is not None:
            mask = collection_dates <= (target_date + self.max_date)
        else:
            # There is no minimum or maximum date
            mask = np.ones(collection_dates.shape, dtype=bool)
        if not self.include:
            mask = np.logical_not(mask)
        return mask
        
    def __call__(self, dataset: xr.Dataset, target_date: np.datetime64) -> Dict[str, np.ndarray]:
        """Calculates a boolean mask based on the filter parameters.

        Parameters
        ----------
        dataset: xr.Dataset
            The MultiEarth xarray dataset to filter. This can be
            loaded from a MultiEarth NetCDF file.

        target_date: np.datetime64
            The target date to calculate a relative minimum and maximum date around.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary with key 'index' (corresponding to the
            relevant Xarray Dataset coordinate) and a value of a
            boolean mask specifying which images satisfy the filter
            parameters used to initialize this RelativeDateFilter. It
            will have length equal to the length of the dataset's
            index coordinate.

        """
        mask = self.calc_relative_date_mask(dataset.collection_dates.data, target_date)
        return {"index": mask}
