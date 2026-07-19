import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import (
    BSSNParameters,
    BSSNVariables,
    compute_momentum_constraint,
    evolve_traceless_extrinsic_curvature,
)
from JAX_BSSN.derivatives import diff1_field
from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.tensor_algebra import (
    christoffel_symbols_second_kind,
    invert_3x3_metric,
    trace_tensor,
    traceless_part,
)


class TestBSSNEquationRegressions(unittest.TestCase):
    def setUp(self):
        self.n = 24
        x = jnp.linspace(-jnp.pi, jnp.pi, self.n, endpoint=False)
        self.dx = float(x[1] - x[0])
        self.X, self.Y, self.Z = jnp.meshgrid(x, x, x, indexing="ij")
        self.shape = (self.n, self.n, self.n)

    def nontrivial_vars(self):
        q = 0.03 * jnp.sin(self.X + self.Y) + 0.02 * jnp.cos(self.Y - self.Z)

        gamma = jnp.zeros((3, 3) + self.shape)
        gamma = gamma.at[0, 0].set(jnp.exp(2.0 * q))
        gamma = gamma.at[1, 1].set(jnp.exp(-q))
        gamma = gamma.at[2, 2].set(jnp.exp(-q))

        inv_gamma = invert_3x3_metric(gamma)

        A_seed = jnp.zeros((3, 3) + self.shape)
        A_seed = A_seed.at[0, 0].set(0.02 * jnp.sin(self.X))
        A_seed = A_seed.at[1, 1].set(0.015 * jnp.cos(self.Y))
        A_seed = A_seed.at[2, 2].set(-0.01 * jnp.sin(self.Z))
        A_seed = A_seed.at[0, 1].set(0.012 * jnp.sin(self.X - self.Y))
        A_seed = A_seed.at[1, 0].set(0.012 * jnp.sin(self.X - self.Y))
        A_seed = A_seed.at[0, 2].set(0.009 * jnp.cos(self.X + self.Z))
        A_seed = A_seed.at[2, 0].set(0.009 * jnp.cos(self.X + self.Z))
        A_ij = traceless_part(A_seed, gamma, inv_gamma)

        W = 1.0 + 0.04 * jnp.sin(self.X - self.Z)
        K = 0.03 * jnp.cos(self.X + self.Y + self.Z)

        metric_derivs = jnp.stack(
            [diff1_field(gamma, d + 2, self.dx) for d in range(3)],
            axis=0,
        )
        christoffel = christoffel_symbols_second_kind(inv_gamma, metric_derivs)
        conformal_connection = jnp.einsum(
            "mn...,imn...->i...", inv_gamma, christoffel
        )

        return BSSNVariables(
            conformal_metric=gamma,
            conformal_factor=W,
            traceless_K=A_ij,
            trace_K=K,
            conformal_connection=conformal_connection,
            lapse=jnp.ones(self.shape),
            shift=jnp.zeros((3,) + self.shape),
        )

    def test_momentum_constraint_includes_contracted_connection_term(self):
        vars = self.nontrivial_vars()
        gamma = vars.conformal_metric
        inv_gamma = invert_3x3_metric(gamma)
        A_ij = vars.traceless_K
        W = vars.conformal_factor
        K = vars.trace_K

        params = BSSNParameters(dx=self.dx, dt=0.01, nu=0.0)
        momentum = compute_momentum_constraint(vars, params)

        dA_ij_dk = jnp.stack(
            [diff1_field(A_ij, d + 2, self.dx) for d in range(3)],
            axis=0,
        )
        dgamma_ij_dk = jnp.stack(
            [diff1_field(gamma, d + 2, self.dx) for d in range(3)],
            axis=0,
        )
        dWdi = jnp.stack([diff1_field(W, d, self.dx) for d in range(3)], axis=0)
        dKdi = jnp.stack([diff1_field(K, d, self.dx) for d in range(3)], axis=0)

        A_ij_raised = jnp.einsum(
            "ik...,jl...,kl...->ij...", inv_gamma, inv_gamma, A_ij
        )
        A_i_up_j = jnp.einsum("jk...,ik...->ij...", inv_gamma, A_ij)

        expected = jnp.einsum("jl...,lij...->i...", inv_gamma, dA_ij_dk)
        expected += -0.5 * jnp.einsum(
            "jk...,ijk...->i...", A_ij_raised, dgamma_ij_dk
        )
        expected += -jnp.einsum(
            "j...,ij...->i...", vars.conformal_connection, A_ij
        )
        expected += -3.0 * jnp.einsum("ij...,j...->i...", A_i_up_j, dWdi) / W
        expected += -(2.0 / 3.0) * dKdi

        np.testing.assert_allclose(momentum, expected, atol=3.0e-6)

    def test_kappa_term_changes_traceless_extrinsic_curvature_rhs(self):
        vars = self.nontrivial_vars()
        params0 = BSSNParameters(dx=self.dx, dt=0.01, nu=0.0, kappa=0.0)
        params1 = params0._replace(kappa=2.5)

        rhs0 = evolve_traceless_extrinsic_curvature(vars, params0)
        rhs1 = evolve_traceless_extrinsic_curvature(vars, params1)

        inv_gamma = invert_3x3_metric(vars.conformal_metric)
        metric_derivs = jnp.stack(
            [
                diff1_field(vars.conformal_metric, d + 2, self.dx)
                for d in range(3)
            ],
            axis=0,
        )
        christoffel = christoffel_symbols_second_kind(inv_gamma, metric_derivs)
        M_i = compute_momentum_constraint(vars, params1)

        dMidj = jnp.zeros((3,) + M_i.shape)
        for i in range(3):
            for j in range(3):
                dMidj = dMidj.at[i, j].set(diff1_field(M_i[i], j, self.dx))

        DjMi = dMidj - jnp.einsum("kij...,k...->ij...", christoffel, M_i)
        DiMj = jnp.swapaxes(DjMi, 0, 1)
        expected_difference = 0.5 * params1.kappa * vars.lapse * (DjMi + DiMj)

        np.testing.assert_allclose(rhs1 - rhs0, expected_difference, atol=4.0e-7)

    def test_rk4_projects_pure_trace_A_before_first_rhs(self):
        gamma = jnp.eye(3)[:, :, None, None, None] * jnp.ones(
            (3, 3) + self.shape
        )
        pure_trace_A = 0.02 * jnp.sin(self.X)
        A_ij = gamma * pure_trace_A

        vars = BSSNVariables(
            conformal_metric=gamma,
            conformal_factor=jnp.ones(self.shape),
            traceless_K=A_ij,
            trace_K=jnp.zeros(self.shape),
            conformal_connection=jnp.zeros((3,) + self.shape),
            lapse=jnp.ones(self.shape),
            shift=jnp.zeros((3,) + self.shape),
        )
        params = BSSNParameters(
            dx=self.dx, dt=0.01, nu=0.0, kappa=0.0, g=0.0, eta=0.0
        )

        evolved = rk4_step(vars, params)

        np.testing.assert_allclose(evolved.conformal_metric, gamma, atol=1.0e-12)
        trace_A = trace_tensor(
            evolved.traceless_K, invert_3x3_metric(evolved.conformal_metric)
        )
        np.testing.assert_allclose(trace_A, jnp.zeros_like(trace_A), atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
