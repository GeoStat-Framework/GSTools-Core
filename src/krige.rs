use ndarray::{Array1, ArrayView1, ArrayView2};

pub fn calculator_field_krige_and_variance(
    krig_mat: ArrayView2<'_, f64>,
    krig_vecs: ArrayView2<'_, f64>,
    cond: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    let mat_i = krig_mat.shape()[0];
    let res_i = krig_vecs.shape()[1];

    let mut field = Array1::<f64>::zeros(res_i);
    let mut error = Array1::<f64>::zeros(res_i);

    //TODO make parallel
    (0..res_i).into_iter().for_each(|k| {
        (0..mat_i).into_iter().for_each(|i| {
            let mut krig_fac = 0.0;
            (0..mat_i).into_iter().for_each(|j| {
                krig_fac += krig_mat[[i, j]] * krig_vecs[[j, k]];
            });
            error[k] += krig_vecs[[i, k]] * krig_fac;
            field[k] += cond[i] * krig_fac;
        });
    });

    (field, error)
}

pub fn calculator_field_krige(
    krig_mat: ArrayView2<'_, f64>,
    krig_vecs: ArrayView2<'_, f64>,
    cond: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mat_i = krig_mat.shape()[0];
    let res_i = krig_vecs.shape()[1];

    let mut field = Array1::<f64>::zeros(res_i);

    //TODO make parallel
    (0..res_i).into_iter().for_each(|k| {
        (0..mat_i).into_iter().for_each(|i| {
            let mut krig_fac = 0.0;
            (0..mat_i).into_iter().for_each(|j| {
                krig_fac += krig_mat[[i, j]] * krig_vecs[[j, k]];
            });
            field[k] += cond[i] * krig_fac;
        });
    });

    field
}
