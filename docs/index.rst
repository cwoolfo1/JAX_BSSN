JAX BSSN
========

.. container:: hero

   JAX BSSN is a autodifferentiable and jit-compiled implementation of the Cartesian
   Baumgarte--Shapiro--Shibata--Nakamura evolution system in JAX.

   .. container:: hero-actions

      :doc:`Run a physical demo <demos>`
      :doc:`Understand one RK4 step <evolution>`

.. container:: hero-callout

   .. rubric:: Current scope

   The installed package provides the vacuum Cartesian BSSN equations, fourth-
   order finite differences, Kreiss--Oliger dissipation, periodic and Sommerfeld
   boundary treatments, spherical and axisymmetric Cartoon reconstruction, synchronous openPMD
   diagnostics, and a stage-synchronous nested 2:1 refinement hierarchy. The supported
   physical examples are gauge wave, linear wave, Cartesian single puncture,
   spherical Cartoon puncture, and axisymmetric Cartoon puncture.

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

   .. grid-item-card:: Use Cartoon symmetry
      :link: cartoon
      :link-type: doc

      Follow spherical and axisymmetric storage, reconstruction, and RK4 staging.

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
   cartoon
   fmr
   diagnostics
   demos

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
