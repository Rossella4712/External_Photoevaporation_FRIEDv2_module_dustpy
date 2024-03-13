Module to take into account the mass loss due to external photoevaporation in DustPy (Stammler & Birnstiel, 2022). The mass loss rate is computed interpolating the FRIEDv2 grid of mass loss rate (Haworth, 2023), and the numerical approach follows Sellek et al.(2020).
This module is based on the implementation by M. Gárate (...), but includes FRIEDv2 instead of FRIED.

The FRIEDv2 tables included here are a subset of the entire table. To explore different options for the dust component, upload the new tables in the kword "fried_filename" in the line: 
setup_externalPhotoevaporation_FRIED(sim, fried_filename = "./friedgrid.dat", fuv)

See the run_test.py code for an example.

If you use this module, please cite Gárate et al. 2023
