"""Module containing core functionality of ``astrowidgets``."""

# STDLIB
import functools
import os
import tempfile
import warnings

# THIRD-PARTY
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyWarning

# Jupyter widgets
import ipywidgets as ipyw

# Firefly
import firefly_client

__all__ = ['FireflyWidget']

# Allowed locations for cursor display
ALLOWED_CURSOR_LOCATIONS = ['top', 'bottom', None]

# List of marker names that are for internal use only
RESERVED_MARKER_SET_NAMES = ['all']


class FireflyWidget:
    """
    Image widget for Jupyter notebook using Firefly viewer.

    .. todo:: Any property passed to constructor has to be valid keyword.

    Parameters
    ----------
    logger : obj or ``None``
        Ginga logger. For example::

            from ginga.misc.log import get_logger
            logger = get_logger('my_viewer', log_stderr=False,
                                log_file='ginga.log', level=40)

    image_width, image_height : int
        Dimension of Jupyter notebook's image widget.

    use_opencv : bool
        Let Ginga use ``opencv`` to speed up image transformation;
        e.g., rotation and mosaic. If this is enabled and you
        do not have ``opencv``, you will get a warning.

    pixel_coords_offset : int, optional
        An offset, typically either 0 or 1, to add/subtract to all
        pixel values when going to/from the displayed image.
        *In almost all situations the default value, ``0``, is the
        correct value to use.*

    """

    def __init__(self, start_browser_tab=False, *args, **kwargs):
        
        html_file = kwargs.get('html_file',
                               os.environ.get('FIREFLY_HTML', 'slate.html'))
        
        self._viewer = firefly_client.FireflyClient.make_lab_client(
                start_tab=True, html_file=html_file)

        if start_browser_tab:
            self._viewer.launch_browser()


    def load_fits(self, fitsorfn, numhdu=None, memmap=None):
        """
        Load a FITS file into the viewer.

        Parameters
        ----------
        fitsorfn : str or HDU
            Either a file name or an HDU (*not* an HDUList).
            If file name is given, WCS in primary header is automatically
            inherited. If a single HDU is given, WCS must be in the HDU
            header.

        numhdu : int or ``None``
            Extension number of the desired HDU.
            If ``None``, it is determined automatically.

        memmap : bool or ``None``
            Memory mapping.
            If ``None``, it is determined automatically.

        """
        if isinstance(fitsorfn, str):
            if numhdu == 0 or numhdu is None:
                fname = fitsorfn
 
            else:
                hdulist = fits.open(fitsorfn)
                hdu = hdulist[numhdu]
                fname = self._write_temp_fits(hdu)

        elif isinstance(fitsorfn, (fits.ImageHDU, fits.CompImageHDU,
                                   fits.PrimaryHDU)):
            fname = self._write_temp_fits(fitsorfn)
        
        f = self._viewer.upload_file(fname)
        self._viewer.show_fits(f)

    @property
    def zoom_level(self):
        """
        Zoom level:

        * 1 means real-pixel-size.
        * 2 means zoomed in by a factor of 2.
        * 0.5 means zoomed out by a factor of 2.

        """
        return self._viewer.get_scale()

    @zoom_level.setter
    def zoom_level(self, val):
        if val == 'fit':
            self._viewer.zoom_fit()
        else:
            self._viewer.scale_to(val, val)

    def zoom(self, val):
        """
        Zoom in or out by the given factor.

        Parameters
        ----------
        val : int
            The zoom level to zoom the image.
            See `zoom_level`.

        """
        self.zoom_level = self.zoom_level * val


    def _write_temp_fits(hdu):
        """
        Write a FITS HDU to a temporary location.

        Parameters
        ----------
        hdu : astropy.io.fits.HDU
            The header-data unit to write

        Returns
        -------
        fname : str
            The filename of the image on disk
        """

        with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as fd:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                hdu.writeto(fd, overwrite=True)
        return(fd)


def _offset_is_pixel_or_sky(x):
    if isinstance(x, u.Quantity):
        if x.unit in (u.dimensionless_unscaled, u.pix):
            coord = 'data'
            val = x.value
        else:
            coord = 'wcs'
            val = x.to_value(u.deg)
    else:
        coord = 'data'
        val = x

    return val, coord
