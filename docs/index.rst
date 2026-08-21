JAX BSSN
========

.. container:: hero

   JAX BSSN is a research-oriented implementation of the Cartesian
   Baumgarte--Shapiro--Shibata--Nakamura evolution system in JAX. It keeps the
   evolved fields, finite-difference operators, boundary treatment, algebraic
   projections, and fixed mesh refinement transfers visible as direct array
   operations.

   .. container:: hero-actions

      :doc:`Run a physical demo <demos>`
      :doc:`Understand one RK4 step <evolution>`

.. container:: hero-callout

   **Focused for numerical-relativity researchers**

   The code favors explicit equations and immutable JAX state over a large
   framework layer. This makes it practical to inspect numerical ownership,
   prototype changes, and verify that refinement and boundary operations stay
   at their intended locations in the timestep.

Current scope
-------------

The installed package provides the vacuum Cartesian BSSN equations, fourth-
order finite differences, Kreiss--Oliger dissipation, periodic, Super-Gaussian,
and Sommerfeld boundary treatments, synchronous openPMD diagnostics, and one
stage-synchronous 2:1 refinement patch. The supported physical examples are
exactly gauge wave, linear wave, and single puncture.

Quick navigation
----------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Install JAX BSSN
      :link: installation
      :link-type: doc

      Set up the solver, test dependencies, and documentation toolchain.

   .. grid-item-card:: Follow the architecture
      :link: architecture
      :link-type: doc

      See package ownership and the complete execution flow.

   .. grid-item-card:: Inspect the BSSN system
      :link: bssn
      :link-type: doc

      Review every evolved variable and equation owner.

   .. grid-item-card:: Trace an RK4 step
      :link: evolution
      :link-type: doc

      Locate derivatives, dissipation, boundaries, and projections.

   .. grid-item-card:: Understand FMR
      :link: fmr
      :link-type: doc

      Examine patch geometry, guard filling, and synchronized stages.

   .. grid-item-card:: Run the demos
      :link: demos
      :link-type: doc

      Launch the three supported physical examples explicitly.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   architecture
   bssn
   evolution
   fmr
   diagnostics
   demos
   development
   contributing

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
