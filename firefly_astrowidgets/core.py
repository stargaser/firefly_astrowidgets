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

    def __init__(self, use_extension=True, launch_browser=False, *args, **kwargs):

        html_file = kwargs.get('html_file',
                               os.environ.get('FIREFLY_HTML', 'slate.html'))

        if use_extension:
            self._viewer = firefly_client.FireflyClient.make_lab_client(
                start_tab=True, html_file=html_file)
        else:
            self._viewer = firefly_client.FireflyClient.make_client(
                    html_file=html_file, launch_browser=launch_browser)
            if launch_browser is False:
                self._viewer.display_url()

        self.plot_id = 'main-display'
        self._zlevel = 1.0
        self._pixel_offset = 1
        self._stretch_options = ['linear', 'log', 'loglog', 'equal', 'squared',
                                 'sqrt', 'asinh', 'powerlaw_gamma']
        self._current_stretch = 'linear'
        self._autocut_methods = ['minmax', 'zscale']
        self._stype = self._cut_levels = 'zscale'
        # Maintain marker tags as a set because we do not want
        # duplicate names.
        self._marktags = set()
        # Let's have a default name for the tag too:
        self._default_mark_tag_name = 'default-marker-name'
        # available color maps
        self._cmaps = dict(gray=0, reversegray=1, colorcube=2, spectrum=3,
                           falsecolor=4, reversefalsecolor=5,
                           compressedfalsecolor=6, difference=7, ds9_a=8,
                           ds9_b=9, ds9_bb=10, ds9_he=11, ds9_i8=12,
                           ds9_aips=13, ds9_sls=14, ds9_hsv=15, heat=16,
                           cool=17, rainbow=18, standard=19,
                           staircase=20, color=21)

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
        self._viewer.show_fits(f, plot_id=self.plot_id)
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
        self._viewer.show_fits(f, plot_id=self.plot_id)
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
        self._viewer.set_zoom(plot_id=self.plot_id, factor=val)

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
            self._viewer.set_pan(plot_id=self.plot_id,
                                 x=point.ra.deg, y=point.dec.deg,
                                 coord='J2000')
        else:
            self._viewer.set_pan(plot_id=self.plot_id,
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
            self._viewer.set_stretch(plot_id=self.plot_id, stype=self._stype,
                                     algorithm=val)
        else:
            self._viewer.set_stretch(plot_id=self.plot_id, stype=self._stype,
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
            self._viewer.set_stretch(plot_id=self.plot_id, stype=val,
                                     algorithm=self._current_stretch)
            self._stype = val
            self._cut_levels = val
        else:  # (low, high)
            if len(val) > 2:
                raise ValueError('Value must have length 2.')
            self._viewer.set_stretch(plot_id=self.plot_id, stype='absolute',
                                     lower_value=val[0], upper_value=val[1])
            self._cut_levels = val
            self._stype = 'absolute'

    @property
    def colormap_options(self):
        """List of colormap names."""
        return list(self._cmaps.keys())

    def set_colormap(self, cmap):
        """
        Set colormap to the given colormap name.

        Parameters
        ----------
        cmap : str
            Colormap name. Possible values can be obtained from
            :meth:`colormap_options`.

        """
        cbar_id = self._cmaps[cmap]
        self._viewer.dispatch('ImagePlotCntlr.ColorChange',
                              payload=dict(plotId=self.plot_id,
                                           cbarId=cbar_id))

    def add_markers(self, table, x_colname='x', y_colname='y',
                    skycoord_colname='coord', use_skycoord=False,
                    marker_name=None):
        """
        Creates markers in the image at given points.

        .. todo::

            Later enhancements to include more columns
            to control size/style/color of marks,

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table containing marker locations.

        x_colname, y_colname : str
            Column names for X and Y.
            Coordinates can be 0- or 1-indexed, as
            given by ``self.pixel_offset``.

        skycoord_colname : str
            Column name with ``SkyCoord`` objects.

        use_skycoord : bool
            If `True`, use ``skycoord_colname`` to mark.
            Otherwise, use ``x_colname`` and ``y_colname``.

        marker_name : str, optional
            Name to assign the markers in the table. Providing a name
            allows markers to be removed by name at a later time.
        """
        if marker_name is None:
            marker_name = self._default_mark_tag_name
        if use_skycoord is False:
            print("Using pixel coordinates is not supported")
            return
        else: # Using sky coordinates
            upload_table = table.copy()
            if 'ra' not in upload_table.colnames:
                upload_table['ra'] = upload_table[skycoord_colname].ra.deg
            if 'dec' not in upload_table.colnames:
                upload_table['dec'] = upload_table[skycoord_colname].dec.deg
            with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as fd:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', AstropyWarning)
                    upload_table.write(fd.name, format="fits", overwrite=True)
            tval = self._viewer.upload_file(fd.name)
            self._viewer.show_table(tval, tbl_id=marker_name,
                                    title=marker_name)
            os.remove(fd.name)
        self._marktags.add(marker_name)

    def remove_markers(self, marker_name=None):
        """
        Remove some but not all of the markers by name used when
        adding the markers

        Parameters
        ----------

        marker_name : str, optional
            Name used when the markers were added.
        """
        if marker_name is None:
            marker_name = self._default_mark_tag_name

        if marker_name not in self._marktags:
            # This shouldn't have happened, raise an error
            raise ValueError('Marker name {} not found in current markers.'
                             ' Markers currently in use are '
                             '{}'.format(marker_name,
                                         sorted(self._marktags)))

        self._viewer.dispatch('table.remove',
                              payload=dict(tbl_id=marker_name))
        self._marktags.remove(marker_name)

    def reset_markers(self):
        """
        Delete all markers.
        """

        # Grab the entire list of marker names before iterating
        # otherwise what we are iterating over changes.
        for marker_name in list(self._marktags):
            self.remove_markers(marker_name)

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
