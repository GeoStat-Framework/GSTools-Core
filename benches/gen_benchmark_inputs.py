#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from gstools import Gaussian
from gstools.field.generator import RandMeth, IncomprRandMeth


def gen_field_summate(path, seed):
    pos_no = 1000
    mode_no = 100
    x = np.linspace(0.0, 10.0, pos_no)
    y = np.linspace(-5.0, 5.0, pos_no)
    z = np.linspace(-6.0, 8.0, pos_no)

    model = Gaussian(dim=3, var=1.0, len_scale=1.0)

    rm = RandMeth(model, mode_no, seed)
    np.savetxt(path / 'field_bench_x.txt', x)
    np.savetxt(path / 'field_bench_y.txt', y)
    np.savetxt(path / 'field_bench_z.txt', z)
    np.savetxt(path / 'field_bench_cov_samples.txt', rm._cov_sample)
    np.savetxt(path / 'field_bench_z_1.txt', rm._z_1)
    np.savetxt(path / 'field_bench_z_2.txt', rm._z_2)

    pos_no = 100000
    mode_no = 1000
    x = np.linspace(0.0, 10.0, pos_no)
    y = np.linspace(-5.0, 5.0, pos_no)
    z = np.linspace(-6.0, 8.0, pos_no)

    model = Gaussian(dim=3, var=1.0, len_scale=1.0)

    rm = RandMeth(model, mode_no, seed)
    np.savetxt(path / 'field_vs_x.txt', x)
    np.savetxt(path / 'field_vs_y.txt', y)
    np.savetxt(path / 'field_vs_z.txt', z)
    np.savetxt(path / 'field_vs_cov_samples.txt', rm._cov_sample)
    np.savetxt(path / 'field_vs_z_1.txt', rm._z_1)
    np.savetxt(path / 'field_vs_z_2.txt', rm._z_2)


if __name__ == '__main__':
    path = Path('input')
    seed = 19031977

    gen_field_summate(path, seed)
