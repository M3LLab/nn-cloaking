import jax.numpy as jnp

def C_eff_tensor(x1, lam, mu):
    F = F_tensor(x1)

    F21 = F[1, 0]
    F22 = F[1, 1]

    C = jnp.array([
        [(lam + 2*mu)/F22,
         lam,
         0.0,
         (F21/F22)*(lam + 2*mu)],

        [lam,
         (F21**2 * mu + F22**2 * (lam + 2*mu)) / F22,
         (F21/F22)*mu,
         F21*(lam + mu)],

        [0.0,
         (F21/F22)*mu,
         mu/F22,
         mu],

        [(F21/F22)*(lam + 2*mu),
         F21*(lam + mu),
         mu,
         (F21**2 * (lam + 2*mu) + F22**2 * mu) / F22]
    ])

    return C