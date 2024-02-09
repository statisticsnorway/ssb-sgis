import numpy as np


def get_gradient(raster, degrees: bool = False, copy: bool = False):
    """Get the slope of an elevation raster.

    Calculates the absolute slope between the grid cells
    based on the image resolution.

    For multiband images, the calculation is done for each band.

    Args:
        degrees: If False (default), the returned values will be in ratios,
            where a value of 1 means 1 meter up per 1 meter forward. If True,
            the values will be in degrees from 0 to 90.
        copy: Whether to copy or overwrite the original Raster.
            Defaults to False to save memory.

    Returns:
        The class instance with new array values, or a copy if copy is True.

    Examples
    --------
    Making an array where the gradient to the center is always 10.

    >>> import sgis as sg
    >>> import numpy as np
    >>> arr = np.array(
    ...         [
    ...             [100, 100, 100, 100, 100],
    ...             [100, 110, 110, 110, 100],
    ...             [100, 110, 120, 110, 100],
    ...             [100, 110, 110, 110, 100],
    ...             [100, 100, 100, 100, 100],
    ...         ]
    ...     )

    Now let's create a Raster from this array with a resolution of 10.

    >>> r = sg.Raster.from_array(arr, crs=None, bounds=(0, 0, 50, 50), res=10)

    The gradient will be 1 (1 meter up for every meter forward).
    The calculation is by default done in place to save memory.

    >>> r.gradient()
    >>> r.array
    array([[0., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1.],
        [1., 1., 0., 1., 1.],
        [1., 1., 1., 1., 1.],
        [0., 1., 1., 1., 0.]])
    """
    if len(raster.array.shape) == 2:
        array = _slope_2d(raster.array, raster.res, degrees=degrees)
    else:
        out_array = []
        for array in raster.array:
            results = _slope_2d(array, raster.res, degrees=degrees)
            out_array.append(results)
        array = np.array(out_array)

    return raster._return_self_or_copy(array, copy)


def _slope_2d(array, res, degrees) -> np.ndarray:
    gradient_x, gradient_y = np.gradient(array, res, res)

    gradient = abs(gradient_x) + abs(gradient_y)

    if not degrees:
        return gradient

    radians = np.arctan(gradient)
    degrees = np.degrees(radians)

    assert np.max(degrees) <= 90

    return degrees
