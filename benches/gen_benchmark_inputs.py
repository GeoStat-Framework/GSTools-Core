#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import gstools as gs
from gstools.field.generator import RandMeth, IncomprRandMeth, Fourier


def gen_field_summate(path, seed):
    pos_no = 10_000
    mode_no = 1_000
    x = np.linspace(0.0, 10.0, pos_no)
    y = np.linspace(-5.0, 5.0, pos_no)
    z = np.linspace(-6.0, 8.0, pos_no)

    model_2d = gs.Gaussian(dim=2, var=1.0, len_scale=1.0)
    model_3d = gs.Gaussian(dim=3, var=1.0, len_scale=1.0)

    rm_2d = RandMeth(model_2d, mode_no=mode_no, seed=seed)
    rm_3d = RandMeth(model_3d, mode_no=mode_no, seed=seed)
    f_2d = Fourier(model_2d, period=[10.0, 10.0] , seed=seed)
    np.savetxt(path / 'field_x.txt', x)
    np.savetxt(path / 'field_y.txt', y)
    np.savetxt(path / 'field_z.txt', z)
    np.savetxt(path / 'field_cov_samples_2d.txt', rm_2d._cov_sample)
    np.savetxt(path / 'field_cov_samples_3d.txt', rm_3d._cov_sample)
    np.savetxt(path / 'field_z_1.txt', rm_2d._z_1)
    np.savetxt(path / 'field_z_2.txt', rm_2d._z_2)
    np.savetxt(path / 'field_fourier_z_1.txt', f_2d._z_1)
    np.savetxt(path / 'field_fourier_z_2.txt', f_2d._z_2)
    np.savetxt(path / 'field_fourier_factor_2d.txt', f_2d._spectrum_factor)
    np.savetxt(path / 'field_fourier_modes_2d.txt', f_2d._modes)

def gen_krige(path, seed):
    def prepare_data(pos_no, cond_no):
        rng = np.random.default_rng(seed)
        pos = np.array((
            rng.uniform(0., 10., pos_no),
            rng.uniform(-5., 5., pos_no),
            rng.uniform(-10., 0., pos_no),
        ))
        cond_pos = (
            rng.uniform(0., 10., cond_no),
            rng.uniform(-5., 5., cond_no),
            rng.uniform(-10., 0., cond_no),
        )
        model = gs.Gaussian(dim=3, var=2., len_scale=2., anis=[0.9, 0.8], angles=[2., 1., 0.5])
        srf = gs.SRF(model, seed=seed)
        cond_val = srf(cond_pos)

        simple = gs.krige.Simple(
            model, cond_pos, cond_val
        )
        krige_mat = simple._krige_mat
        k_vec = simple._get_krige_vecs(simple.pre_pos(pos, 'unstructured')[0])
        krige_cond = simple._krige_cond

        return krige_mat, k_vec, krige_cond

    krige_mat, k_vec, krige_cond = prepare_data(pos_no=10_000, cond_no=500)

    np.savetxt(path / 'krige_krige_mat.txt', krige_mat)
    np.savetxt(path / 'krige_k_vec.txt', k_vec)
    np.savetxt(path / 'krige_krige_cond.txt', krige_cond)


if __name__ == '__main__':
    path = Path('input')
    seed = 19031977

    gen_field_summate(path, seed)
    gen_krige(path, seed)
