API Reference
=============

.. autosummary::
   :toctree: api

   astroimred.imops
   astroimred.imops.ccdutils
   astroimred.imops.imstat
   astroimred.imops.mathutils
   astroimred.imops.pixels
   astroimred.reduction
   astroimred.reduction.combutil
   astroimred.reduction.preproc
   astroimred.reduction.imutil
   astroimred.mgmt
   astroimred.mgmt.airmass
   astroimred.mgmt.headers
   astroimred.mgmt.io
   astroimred.mgmt.logging
   astroimred.mgmt.misc
   astroimred.mgmt.paths
   astroimred.mgmt.summary
   astroimred.mgmt.wcstools

Root-level convenience imports such as ``air.load_ccd`` and compatibility module
aliases such as ``astroimred.io`` remain available, but the canonical module
files live under ``astroimred.mgmt`` and ``astroimred.imops``.

Optional visualization modules
------------------------------

The visualization helpers require ``astroimred[full]``.

.. autosummary::
   :toctree: api

   astroimred.imops.viz
