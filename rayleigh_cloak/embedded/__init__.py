"""Fully embedded circular cloak for bulk P/S waves (Step 1: continuous ideal cloak).

Modules: ``config`` (YAML schema), ``mesh`` (tagged gmsh mesh), ``problem``
(jax-fem weak form), ``linsolve`` (PETSc-free direct solve), ``sampling``
(point evaluation + metrics), ``run`` (cases A-D), ``plots``.
"""

import jax

jax.config.update("jax_enable_x64", True)
