#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cProfile
from pathlib import Path
import numpy as np
import gstools as gs
from gstools.field.generator import RandMeth
from gstools.field.summator import summate, summate_incompr, summate_fourier
from gstools.krige.krigesum import calc_field_krige_and_variance, calc_field_krige
from gstools.variogram.estimator import (
    structured,
    ma_structured,
    unstructured,
    directional,
)
import gstools_core as gsc


if __name__ == '__main__':
    path = Path('input')

    ##########################################################################

    x = np.loadtxt(path / 'field_x.txt')
    y = np.loadtxt(path / 'field_y.txt')
    z = np.loadtxt(path / 'field_z.txt')
    cov_samples_2d = np.loadtxt(path / 'field_cov_samples_2d.txt')
    cov_samples_3d = np.loadtxt(path / 'field_cov_samples_3d.txt')
    z_1 = np.loadtxt(path / 'field_z_1.txt')
    z_2 = np.loadtxt(path / 'field_z_2.txt')
    spectrum_factor_2d = np.loadtxt(path / 'field_fourier_factor_2d.txt')
    modes_2d = np.loadtxt(path / 'field_fourier_modes_2d.txt')
    z_1_fourier = np.loadtxt(path / 'field_fourier_z_1.txt')
    z_2_fourier = np.loadtxt(path / 'field_fourier_z_2.txt')
    pos_2d = np.array((x, y))
    pos_3d = np.array((x, y, z))

    print(f'SUMMATE 2D')
    print(f'\tCYTHON IMPLEMENTATION')
    cProfile.run(
        'summate(cov_samples_2d, z_1, z_2, pos_2d)'
    )

    print(f'\tRUST IMPLEMENTATION')
    cProfile.run(
        'gsc.summate(cov_samples_2d, z_1, z_2, pos_2d)'
    )

    print(f'SUMMATE 3D')
    print(f'\tCYTHON IMPLEMENTATION')
    cProfile.run(
        'summate(cov_samples_3d, z_1, z_2, pos_3d)'
    )

    print(f'\tRUST IMPLEMENTATION')
    cProfile.run(
        'gsc.summate(cov_samples_3d, z_1, z_2, pos_3d)'
    )

    print(f'SUMMATE INCOMPR 2D')
    print(f'\tCYTHON IMPLEMENTATION')
    cProfile.run(
        'summate_incompr(cov_samples_2d, z_1, z_2, pos_2d)'
    )

    print(f'\tRUST IMPLEMENTATION')
    cProfile.run(
        'gsc.summate_incompr(cov_samples_2d, z_1, z_2, pos_2d)'
    )

    print(f'SUMMATE INCOMPR 3D')
    print(f'\tCYTHON IMPLEMENTATION')
    cProfile.run(
        'summate_incompr(cov_samples_3d, z_1, z_2, pos_3d)'
    )

    print(f'\tRUST IMPLEMENTATION')
    cProfile.run(
        'gsc.summate_incompr(cov_samples_3d, z_1, z_2, pos_3d)'
    )

    print(f'SUMMATE FOURIER')
    print(f'\tCYTHON IMPLEMENTATION')
    cProfile.run(
        'summate_fourier(spectrum_factor_2d, modes_2d, z_1_fourier, z_2_fourier, pos_2d)'
    )

    print(f'\tRUST IMPLEMENTATION')
    cProfile.run(
        'gsc.summate_fourier(spectrum_factor_2d, modes_2d, z_1_fourier, z_2_fourier, pos_2d)'
    )

    ##########################################################################

    krige_mat = np.loadtxt(path / 'krige_krige_mat.txt')
    k_vec = np.loadtxt(path / 'krige_k_vec.txt')
    krige_cond = np.loadtxt(path / 'krige_krige_cond.txt')

    print(f'KRIGE_AND_VAR')
    print(f'\tCYTHON IMPLEMENTATION')
    cProfile.run(
        'calc_field_krige_and_variance(krige_mat, k_vec, krige_cond)'
    )

    print(f'\tRUST IMPLEMENTATION')
    cProfile.run(
        'gsc.calc_field_krige_and_variance(krige_mat, k_vec, krige_cond)'
    )
    
    print(f'KRIGE')
    print(f'\tCYTHON IMPLEMENTATION')
    cProfile.run(
        'calc_field_krige(krige_mat, k_vec, krige_cond)'
    )

    print(f'\tRUST IMPLEMENTATION')
    cProfile.run(
        'gsc.calc_field_krige(krige_mat, k_vec, krige_cond)'
    )
