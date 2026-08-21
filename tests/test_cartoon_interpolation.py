import jax
import jax.numpy as jnp
import numpy as np

from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic


jax.config.update("jax_enable_x64", True)


def test_nonperiodic_lagrange6_reproduces_degree_five_polynomials():
    source_x = jnp.arange(12, dtype=jnp.float64)
    targets = jnp.asarray([0.1, 1.75, 5.4, 10.8, 11.2])

    for degree in range(6):
        interpolated = lagrange6_nonperiodic(source_x**degree, targets)
        np.testing.assert_allclose(
            interpolated,
            targets**degree,
            rtol=2.0e-12,
            atol=2.0e-10,
        )


def test_nonperiodic_lagrange6_preserves_leading_and_target_axes():
    source_x = jnp.arange(12, dtype=jnp.float64)
    targets = jnp.asarray([[0.25, 4.5], [9.75, 11.1]])
    profiles = jnp.stack([source_x**5, 3.0 * source_x**5])

    interpolated = lagrange6_nonperiodic(profiles, targets)

    assert interpolated.shape == (2, 2, 2)
    np.testing.assert_allclose(
        interpolated[0], targets**5, rtol=2.0e-12, atol=2.0e-10
    )
    np.testing.assert_allclose(
        interpolated[1], 3.0 * targets**5, rtol=2.0e-12, atol=5.0e-10
    )
