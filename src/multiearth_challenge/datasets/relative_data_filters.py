from typing import Optional

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
        relative_min_date: Optional[np.timedelta64] = None,
        relative_max_date: Optional[np.timedelta64] = None,
        include: bool = True,
    ):
        """
        Parameters
        ----------
        relative_min_date: Optional[np.timedelta64], default=None
            Specifies the minimum of the relative date interval of
            interest. Negative values indicate a time before the
            target date and positive after. If None, there is no
            minimum date.

        relative_max_date: Optional[np.timedelta64], default=None
            Specifies the non-inclusive maximum of the relative date
            interval of interest. Negative values indicate a time
            before the target date and positive after. If None, there
            is no maximum date.

        include: bool, default=True
            If True, will include all imagery within the inclusive
            interval specified by the relative_min_date and
            relative_max_date. Otherwise, will exclude imagery within the date
            interval.

        """
        self.min_date = relative_min_date
        self.max_date = relative_max_date
        if (
            self.min_date is not None
            and self.max_date is not None
            and self.min_date >= self.max_date
        ):
            raise ValueError(
                f"Passed parameter relative_min_date ({self.min_date}) must be greater than relative_max_date ({self.max_date})."
            )
        self.include = include

    def __call__(self, dataset: xr.Dataset, target_date: np.datetime64) -> np.array:
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
        Dict[str, np.array]
            A dictionary with key 'index' (corresponding to the
            relevant Xarray Dataset coordinate) and a value of a
            boolean mask specifying which images satisfy the filter
            parameters used to initialize this RelativeDateFilter. It
            will have length equal to the length of the dataset's
            index coordinate.

        """
        if self.min_date is not None and self.max_date is not None:
            mask = ma.masked_inside(
                dataset.collection_dates.data,
                target_date + self.min_date,
                target_date + self.max_date,
            )
        elif self.min_date is not None:
            mask = dataset.collection_dates.data >= target_date + self.min_date
        elif self.max_date is not None:
            mask = dataset.collection_dates.data < target_date + self.max_date

        if not self.include:
            mask = np.logical_not(mask)
        return {"index": mask}
