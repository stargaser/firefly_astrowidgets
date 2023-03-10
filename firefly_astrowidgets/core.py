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
from astropy.nddata import NDData, CCDData, fits_ccddata_writer
from astropy.table import Table, vstack
import astropy.units as u
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

        self._zlevel = 1.0
        self._pixel_offset = 1
        self._stretch_options = ['linear', 'log', 'loglog', 'equal', 'squared',
                                 'sqrt', 'asinh', 'powerlaw_gamma']
        self._current_stretch = 'linear'
        self._autocut_methods = ['minmax', 'zscale']
        self._stype = self._cut_levels = 'zscale'

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
                temp = False

            else:
                hdulist = fits.open(fitsorfn)
                hdu = hdulist[numhdu]
                fname = self._write_temp_fits(hdu)
                temp = True

        elif isinstance(fitsorfn, (fits.ImageHDU, fits.CompImageHDU,
                                   fits.PrimaryHDU)):
            fname = self._write_temp_fits(fitsorfn)
            temp = True

        f = self._viewer.upload_file(fname)
        self._viewer.show_fits(f, plot_id='main')
        if temp:
            os.remove(fname)

    def load_nddata(self, nddata):
        """
        Load an ``NDData`` object into the viewer.

        .. todo:: Add flag/masking support, etc.

        Parameters
        ----------
        nddata : `~astropy.nddata.NDData`
            ``NDData`` with image data and WCS.

        """
        if nddata.unit is None:
            unit = 'DN'
        else:
            unit = nddata.unit
        ccd = CCDData(nddata, unit=unit)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as fd:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                fits_ccddata_writer(ccd, fd.name)
        f = self._viewer.upload_file(fd.name)
        self._viewer.show_fits(f, plot_id='main')
        os.remove(fd.name)

    def load_array(self, arr):
        """
        Load a 2D array into the viewer.

        .. note:: Use :meth:`load_nddata` for WCS support.

        Parameters
        ----------
        arr : array-like
            2D array.

        """
        nd = NDData(arr, unit='DN')
        self.load_nddata(nd)

    @property
    def zoom_level(self):
        """
        Zoom level:

        * 1 means real-pixel-size.
        * 2 means zoomed in by a factor of 2.
        * 0.5 means zoomed out by a factor of 2.

        """
        return self._zlevel

    @zoom_level.setter
    def zoom_level(self, val):
        self._zlevel = val
        self._viewer.set_zoom(plot_id='main', factor=val)

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

    def center_on(self, point):
        """
        Centers the view on a particular point.

        Parameters
        ----------
        point : tuple or `~astropy.coordinates.SkyCoord`
            If tuple of ``(X, Y)`` is given, it is assumed
            to be in data coordinates.
        """
        if isinstance(point, SkyCoord):
            self._viewer.set_pan(plot_id='main',
                                 x=point.ra.deg, y=point.dec.deg,
                                 coord='J2000')
        else:
            self._viewer.set_pan(plot_id='main',
                                 x=point[0] - self._pixel_offset + 1,
                                 y=point[1] - self._pixel_offset + 1,
                                 coord='image')

    @property
    def stretch_options(self):
        """
        List all available options for image stretching.
        """
        return self._stretch_options

    @property
    def stretch(self):
        """
        The image stretching algorithm in use.
        """
        return self._current_stretch

    @stretch.setter
    def stretch(self, val):
        valid_vals = self.stretch_options
        if val not in valid_vals:
            raise ValueError('Value must be one of: {}'.format(valid_vals))

        if self._stype in self.autocut_options:
            self._viewer.set_stretch(plot_id='main', stype=self._stype,
                                     algorithm=val)
        else:
            self._viewer.set_stretch(plot_id='main', stype=self._stype,
                                     algorithm=val,
                                     lower_value=self.cuts[0],
                                     upper_value=self.cuts[1])
        self._current_stretch = val

    @property
    def autocut_options(self):
        """
        List all available options for image auto-cut.
        """
        return self._autocut_methods

    @property
    def cuts(self):
        """
        Current image cut levels.
        To set new cut levels, either provide a tuple of
        ``(low, high)`` values or one of the options from
        `autocut_options`.
        """
        return self._cut_levels

    @cuts.setter
    def cuts(self, val):
        if isinstance(val, str):  # Autocut
            valid_vals = self.autocut_options
            if val not in valid_vals:
                raise ValueError('Value must be one of: {}'.format(valid_vals))
            self._viewer.set_stretch(plot_id='main', stype=val,
                                     algorithm=self._current_stretch)
            self._stype = val
            self._cut_levels = val
        else:  # (low, high)
            if len(val) > 2:
                raise ValueError('Value must have length 2.')
            self._viewer.set_stretch(plot_id='main', stype='absolute',
                                     lower_value=val[0], upper_value=val[1])
            self._cut_levels = val
            self._stype = 'absolute'

    def _write_temp_fits(self, hdu):
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
                hdu.writeto(fd.name, overwrite=True)
        return(fd.name)


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
