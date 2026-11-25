
EconSieve - Transposed-Ensemble Kalman Filter (TEnKF) and Nonlinear Path-Adjusting Smoother (NPAS)
================================================================================================================


Apart from the smoother (`npas`) and TEnKF, I stole quite some of the code from these two projects:

* https://github.com/rlabbe/filterpy
* https://github.com/pykalman/pykalman


They deserve most of the merits. I just made everything look way more complicated. Sometimes ``filterpy`` was more efficient, sometimes ``pykalman``. Unfortunately the ``pykalman`` project is orphaned. I tweaked something here and there:

* treating numerical errors in the UKF covariance matrix by looking for the nearest positive semi-definite matrix
* eliminating identical sigma points (yields speedup assuming that evaluation of each point is costly)
* extracting functions from classes and compile them using the @njit flag (speedup)
* major cleanup

NPAS is built from scratch. I barely did any testing as a standalone filter and just always used it in combination with the 'pydsge', where it works very well.

Some very rudimentary documentation `can be found here <https://econsieve.readthedocs.io/en/latest/readme.html>`_.

TEnKF precision
---------------

``TEnKF(..., fallback_decimal_digits=200)`` normally uses Float64. When the
scaled forecast factor is severely ill-conditioned or a correction nearly
cancels a member's forecast state, the filter recomputes that observation's
update and requested log-likelihood with the given number of significant decimal
digits. It uses the original forecast ensemble and the same measurement draws,
then rounds the updated ensemble and score to Float64. These checks are
heuristics. Precision already lost in the supplied forecasts cannot be recovered.

With ``fallback_decimal_digits=None``, the same checks run and an observation
that requires a retry raises ``ForecastPrecisionRequired``. Each call to
``batch_filter`` resets four counters on the filter: ``precision_requests``,
``high_precision_updates`` (attempts), ``high_precision_completed``, and
``high_precision_seconds`` (elapsed time spent in attempts, including failures).
They also describe the evaluated prefix when a call raises.

Developing with the local grgrlib fork
--------------------------------------

Put the grgrlib fork in a sibling directory named ``grgrlib``. From the
EconSieve checkout, run:

.. code-block:: bash

   uv sync --locked
   uv run --locked pytest -q

This installs both checkouts in editable mode in EconSieve's ``.venv``.
Edits to either package are picked up the next time Python starts.

Installation with ``pip``
-------------------------------------------------------

Be sure that you are on Python 3.x. Then it's as simple as:

.. code-block:: bash

   pip install econsieve


Installation of bleeding-edge version using ``git``
---------------------------------------------------

First install ``git``. Linux users just use their respective repos. 

Windows users probably use anaconda and can do

.. code-block:: bash

   conda install -c anaconda git

in the conda shell `as they kindly tell us here <https://anaconda.org/anaconda/git>`_. Otherwise you can probably `get it here <https://git-scm.com/download/win>`_.

Then you can simply do

.. code-block:: bash

   pip install git+https://github.com/gboehl/econsieve

If you run it and it complains about missing packages, please let me know so that I can update the `setup.py`!

Alternatively you can clone the repository and then from within the cloned folder run (Windows user from the Anaconda Prompt):

.. code-block:: bash

   pip install .


Updating
--------

The package is updated very frequently (find the history of latest commits `here <https://github.com/gboehl/econsieve/commits/master>`_). I hence recommend pulling and reinstalling whenever something is not working right. Run:

.. code-block:: bash

   pip install --upgrade econsieve
   
Citation
--------

**pydsge** is developed by Gregor Boehl to simulate, filter, and estimate DSGE models with the zero lower bound on nominal interest rates in various applications (see [gregorboehl.com](https://gregorboehl.com) for research papers using the package). Please cite it with:

.. code-block:: latex

    @TechReport{boehl2022meth,
      author={Boehl, Gregor and Strobel, Felix},
      title={{Estimation of DSGE Models with the Effective Lower Bound}},
      year=2022,
      type     = {CRC 224 Discussion Papers},
      institution={University of Bonn and University of Mannheim, Germany}
    }

We appreciate citations for **pydsge** because it helps us to find out how people have
been using the package and it motivates further work.
