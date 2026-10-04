"""Uniform Yee coordinates, analytic stencils, and cylindrical initial data."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolution.coordinates import axis_coordinates, grid_coordinates, cylindrical_volume_weights
from JAX_BSSN.evolution.spatial_derivatives import diff1_physical, diff2_physical, diff1_upwind_physical
from JAX_BSSN.evolution.derivatives import diff1_field, diff2_field
from JAX_BSSN.EM.first_order.equations import (
    _forward_difference, _backward_difference, curl_E_to_densitized_B,
    curl_H_to_densitized_D, densitized_displacement_divergence, densitized_magnetic_divergence,
)
from JAX_BSSN.EM.first_order.staggering import DISPLACEMENT_FIELD_LOCATIONS, MAGNETIC_FIELD_LOCATIONS
from JAX_BSSN.EM.second_order.derivatives import physical_first_derivative, physical_second_derivative
from tests.EM.demo_helpers import load_em_demo_module
from tests.EM.em_helpers import flat_bssn_variables
jax.config.update('jax_enable_x64', True)


def params(n=24):
    h=4.0/n
    return BSSNParameters(dx=h, dt=.1*h, x_min=-3.5*h, y_min=-4*h,
        z_min=-(2*n-1)*h/2, xr_bc=1, zl_bc=1, zr_bc=1)


@pytest.mark.parametrize("formulation", ["first_order", "second_order"])
def test_uniform_hamiltonian_reports_failed_cg(formulation):
    metric = load_em_demo_module(formulation, "initial_metric")
    with pytest.raises(RuntimeError, match="Hamiltonian CG did not converge"):
        metric.solve_electromagnetic_conformal_factor(
            jnp.ones((12, 24)) * .01, 4 / 12, max_cg_iterations=1,
        )


@pytest.mark.parametrize('order',[2,4])
def test_all_physical_derivatives_converge(order):
    errors=[]
    first=diff1_physical if order==4 else physical_first_derivative
    second=diff2_physical if order==4 else physical_second_derivative
    for n in (24,48,96):
        p=params(n); X,Y,Z=grid_coordinates((n+4,9,2*n),p,jnp.float64)
        f=jnp.exp(-X*X-Z*Z)*jnp.cos(.3+Y)
        exact1=[-2*X*f,-jnp.exp(-X*X-Z*Z)*jnp.sin(.3+Y),-2*Z*f]
        exact2=[(4*X*X-2)*f,-f,(4*Z*Z-2)*f]
        interior=(slice(4,-4),4,slice(4,-4))
        errs=[]
        for d in range(3):
            errs.extend(float(jnp.max(jnp.abs(a-b)[interior])) for a,b in ((first(f,d,p),exact1[d]),(second(f,d,p),exact2[d])))
        mixed=first(first(f,0,p),2,p)
        errs.append(float(jnp.max(jnp.abs(mixed-4*X*Z*f)[interior])))
        if order==4:
            for sign in (-1.,1.):
                errs.append(float(jnp.max(jnp.abs(diff1_upwind_physical(f,sign,2,p)-exact1[2])[interior])))
        errors.append(errs)
    rates=np.log2(np.asarray(errors[-2])/np.asarray(errors[-1]))
    assert np.min(rates)>order-.4,(errors,rates)


def test_yee_axial_derivatives_and_div_curl():
    errors=[]
    for n in (24,48,96):
        p=params(n)
        zc=axis_coordinates(2*n,2,p,jnp.float64,'C')[None,None,:]
        zv=axis_coordinates(2*n,2,p,jnp.float64,'V')[None,None,:]
        errors.append(max(float(jnp.max(jnp.abs(a-b)[...,4:-4])) for a,b in (
            (_backward_difference(jnp.exp(-zc**2),2,p),-2*zv*jnp.exp(-zv**2)),
            (_forward_difference(jnp.exp(-zv**2),2,p),-2*zc*jnp.exp(-zc**2)))))
    assert np.log2(errors[-2]/errors[-1])>1.8,errors
    p=params(24);f=jnp.asarray(np.random.default_rng(1).normal(size=(3,28,9,48)))
    for curl,div in ((curl_E_to_densitized_B,densitized_magnetic_divergence),(curl_H_to_densitized_D,densitized_displacement_divergence)):
        np.testing.assert_allclose(div(curl(f,p),p)[5:-4,2:-2,2:-2],0,atol=5e-12)


@pytest.mark.parametrize("formulation", ["first_order", "second_order"])
def test_newton_sparse_reference_and_convergence(formulation):
    from scipy.sparse import diags, kron
    from scipy.sparse.linalg import spsolve
    metric=load_em_demo_module(formulation,'initial_metric')
    p=params(12);h=p.dx
    nr,nz=11,22
    u=(np.arange(nr)+.5)*h
    weights=u; lo=u-h/2; hi=u+h/2
    Ar=diags((lo[1:],-lo-hi,hi[:-1]),(-1,0,1))
    Az=diags((np.ones(nz-1),-2*np.ones(nz),np.ones(nz-1)),(-1,0,1))
    lap=(kron(Ar,diags(np.ones(nz)))+kron(diags(weights),Az))/h**2
    energy=np.ones((nr,nz))*.01; psi=np.ones((nr,nz))*1.05
    volume=np.broadcast_to(weights[:,None],(nr,nz))
    A=lap-diags((3*np.pi*energy*psi**-4*volume).ravel())
    rng=np.random.default_rng(9);x=rng.normal(size=(nr,nz));y=rng.normal(size=(nr,nz))
    op=lambda q:metric.linearized_electromagnetic_hamiltonian_operator(jnp.asarray(q),jnp.asarray(psi),jnp.asarray(energy),h)
    np.testing.assert_allclose(op(x).ravel(),A@x.ravel(),rtol=2e-13,atol=2e-12)
    np.testing.assert_allclose(np.vdot(x,op(y)),np.vdot(op(x),y),rtol=2e-13)
    assert np.vdot(x,-op(x))>0
    solved,correction,history=metric.solve_electromagnetic_conformal_factor(jnp.ones((12,24))*.01,h,cg_tolerance=1e-12)
    ref=np.zeros(nr*nz)
    for _ in range(8):
        F=lap@ref+np.pi*(volume*.01).ravel()*(1+ref)**-3
        jac=lap-diags(3*np.pi*(volume*.01).ravel()*(1+ref)**-4)
        ref-=spsolve(jac,F)
    np.testing.assert_allclose(correction[:-1,1:-1].ravel(),ref,atol=1e-10)
    assert history[-1]<1e-10
    errors=[]
    for n in (24,48,96):
        p=params(n);r=axis_coordinates(n+4,0,p,jnp.float64)[4:,None];z=axis_coordinates(2*n,2,p,jnp.float64)[None,:]
        u=jnp.exp(-r*r-z*z)
        residual=metric.electromagnetic_hamiltonian_residual(u,jnp.zeros_like(u),p.dx)
        weight=cylindrical_volume_weights((n+4,1,2*n),p,jnp.float64)[4:-1,1:-1]
        exact=(4*(r*r+z*z)-6)*u
        errors.append(float(jnp.max(jnp.abs(residual/weight-exact[:-1,1:-1])[:-4,3:-3])))
    assert np.log2(errors[-2]/errors[-1])>1.8,errors


def test_native_reconstruction_z_dependence_converges():
    from JAX_BSSN.EM.first_order.cartoon.axisymmetry import _compact_maxwell_rhs
    errors=[]
    for n in (24,48,96):
        p=params(n);shape=(n+4,1,2*n);bssn=flat_bssn_variables(shape)
        r=axis_coordinates(n+4,0,p,jnp.float64)[:,None];z=axis_coordinates(2*n,2,p,jnp.float64)[None,:]
        rv=axis_coordinates(n+4,0,p,jnp.float64,'V')[:,None];zv=axis_coordinates(2*n,2,p,jnp.float64,'V')[None,:]
        D=jnp.zeros((3,)+shape).at[1,:,0,:].set(.001*r*jnp.exp(-r*r-z*z))
        _,Bdot=_compact_maxwell_rhs(bssn,D,jnp.zeros_like(D),p)
        expected=[-.002*r*zv*jnp.exp(-r*r-zv*zv),-.002*(1-rv*rv)*jnp.exp(-rv*rv-z*z)]
        errors.append(max(float(jnp.max(jnp.abs(Bdot[d,:,0,:]-e)[4:-4,4:-4])) for d,e in zip((0,2),expected)))
    assert np.log2(errors[-2]/errors[-1])>1.8,errors


def test_uniform_norms_use_physical_volume():
    from JAX_BSSN.bssn.constraints import ConstraintViolations
    from JAX_BSSN.cartoon.axisymmetry import compute_axisymmetric_constraint_norms
    p=params(12);shape=(16,1,24)
    f=jnp.arange(np.prod(shape),dtype=float).reshape(shape)
    v=jnp.stack((f,2*f,3*f));zero=jnp.zeros_like(f)
    violations=ConstraintViolations(f,v,zero,zero,v)
    norms=compute_axisymmetric_constraint_norms(violations,params=p)
    w=cylindrical_volume_weights(shape,p,f.dtype)[4:-4,4:-4]
    expected=jnp.sqrt(jnp.sum(w*f[4:-4,0,4:-4]**2)/jnp.sum(w))
    assert float(norms['hamiltonian_l2'])==pytest.approx(float(expected))
    assert float(norms['momentum_l2'])==pytest.approx(float(expected)*np.sqrt(14/3))


@pytest.mark.parametrize('location', DISPLACEMENT_FIELD_LOCATIONS + MAGNETIC_FIELD_LOCATIONS)
def test_native_coordinates_and_cylindrical_volume_weights(location):
    p = params(12)
    shape = (16, 9, 24)
    for d, coordinate in enumerate(grid_coordinates(shape, p, jnp.float64, location)):
        expected = (p.x_min, p.y_min, p.z_min)[d] + p.dx * (
            np.arange(shape[d]) - (0.5 if location[d] == 'V' else 0.0)
        )
        np.testing.assert_allclose(np.ravel(coordinate), expected, atol=1e-15)
    rho = (np.arange(shape[0]) - 3.5 - (0.5 if location[0] == 'V' else 0.0)) * p.dx
    expected_weight = np.broadcast_to(np.abs(rho[:, None]), (16, 24))
    np.testing.assert_allclose(cylindrical_volume_weights(shape, p, jnp.float64, location), expected_weight)


def test_uniform_wrapper_is_exact_legacy_stencil():
    p = params()
    field = jnp.sin(jnp.arange(28 * 9 * 9).reshape(28, 9, 9) * .1)
    np.testing.assert_array_equal(diff1_physical(field, 0, p), diff1_field(field, 0, p.dx, 0, 1))
    np.testing.assert_array_equal(diff2_physical(field, 0, p), diff2_field(field, 0, p.dx, 0, 1))


@pytest.mark.parametrize('formulation,location', [
    ('first_order', ('C', 'C', 'C')),
    ('first_order', ('V', 'C', 'V')),
    ('second_order', ('C', 'C', 'C')),
])
def test_demo_norms_keep_native_radial_volume_weights(formulation, location):
    demo = load_em_demo_module(formulation, 'run_collapse')
    p = params(12)
    field = jnp.arange(16 * 24, dtype=float).reshape((16, 1, 24))
    kwargs = {'location': location} if formulation == 'first_order' else {}
    l2, linf = demo._axisymmetric_scalar_norms(field, p, **kwargs)
    rho = (np.arange(16) - 3.5 - (0.5 if location[0] == 'V' else 0.0)) * p.dx
    weights = np.broadcast_to(np.abs(rho[:, None]), (16, 24))[4:-4, 4:-4].copy()
    if formulation == "first_order" and location[0] == "V":
        weights[0, :] = p.dx / 8.0
    interior = np.asarray(field)[4:-4, 0, 4:-4]
    assert float(l2) == pytest.approx(np.sqrt(np.sum(weights * interior**2) / np.sum(weights)))
    assert float(linf) == pytest.approx(np.max(np.abs(interior)))


def test_physical_derivatives_import_without_initializing_bssn_first():
    import subprocess
    import sys
    from pathlib import Path
    subprocess.run(
        [sys.executable, "-c",
         "from JAX_BSSN.evolution.spatial_derivatives import diff1_physical; "
         "from JAX_BSSN.evolution.boundaries import sommerfeld"],
        cwd=Path(__file__).resolve().parents[2], check=True, capture_output=True,
    )


@pytest.mark.parametrize('spherical', [False, True])
@pytest.mark.parametrize('wave', [False, True])
@pytest.mark.parametrize('dx', [0.0, -0.1, np.nan, np.inf])
def test_uniform_cartoon_rejects_invalid_spacing(spherical, wave, dx):
    from JAX_BSSN.EM.second_order.variables import EMVariables
    from JAX_BSSN.cartoon.axisymmetry import validate_axisymmetric_grid
    from JAX_BSSN.cartoon.spherical_symmetry import validate_cartoon_grid
    from JAX_BSSN.EM.second_order.cartoon.axisymmetry import validate_axisymmetric_wave_grid
    from JAX_BSSN.EM.second_order.cartoon.spherical_symmetry import validate_cartoon_wave_grid
    p = params(12)._replace(dx=dx, x_min=-3.5*dx, y_min=-4*dx, z_min=-4*dx)
    if spherical:
        p = p._replace(zl_bc=0, zr_bc=0)
    shape = (16, 1, 1 if spherical else 24)
    state = flat_bssn_variables(shape)
    if wave:
        zero = jnp.zeros((3,) + shape)
        state = EMVariables(zero, zero, zero, zero)
        validate = validate_cartoon_wave_grid if spherical else validate_axisymmetric_wave_grid
    else:
        validate = validate_cartoon_grid if spherical else validate_axisymmetric_grid
    with pytest.raises(ValueError, match='finite dx > 0'):
        validate(state, p)
