from bssn import BSSNVariables, BSSNParameters, bssn_evolution_step
from kreiss_oliger import apply_ko_dissipation_bssn
from jax import jit


@jit 
def forward_euler_step(vars: BSSNVariables, params: BSSNParameters,
                        ko_sigma: float) -> BSSNVariables:
    """
    Perform one forward Euler timestep with dissipation.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        ko_sigma: Kreiss-Oliger dissipation coefficient
        
    Returns:
        Updated BSSN variables
    """
    dt = params.dt
    
    # Compute BSSN time derivatives
    new_vars = bssn_evolution_step(vars, params)
    
    # Extract time derivatives
    dt_gamma = (new_vars.conformal_metric - vars.conformal_metric) / dt
    dt_W = (new_vars.conformal_factor - vars.conformal_factor) / dt
    dt_A = (new_vars.traceless_K - vars.traceless_K) / dt
    dt_K = (new_vars.trace_K - vars.trace_K) / dt
    dt_Gamma = (new_vars.conformal_connection - vars.conformal_connection) / dt
    dt_alpha = (new_vars.lapse - vars.lapse) / dt
    dt_beta = (new_vars.shift - vars.shift) / dt
    
    # Add Kreiss-Oliger dissipation
    dissipation = apply_ko_dissipation_bssn(vars, ko_sigma, params.dx)
    
    dt_gamma += dissipation.conformal_metric
    dt_W += dissipation.conformal_factor
    dt_A += dissipation.traceless_K
    dt_K += dissipation.trace_K
    dt_Gamma += dissipation.conformal_connection
    dt_alpha += dissipation.lapse
    dt_beta += dissipation.shift
    
    # Update variables
    new_vars = BSSNVariables(
        conformal_metric=vars.conformal_metric + dt * dt_gamma,
        conformal_factor=vars.conformal_factor + dt * dt_W,
        traceless_K=vars.traceless_K + dt * dt_A,
        trace_K=vars.trace_K + dt * dt_K,
        conformal_connection=vars.conformal_connection + dt * dt_Gamma,
        lapse=vars.lapse + dt * dt_alpha,
        shift=vars.shift + dt * dt_beta
    )
    
    return new_vars

def backward_euler_step(vars: BSSNVariables, params: BSSNParameters,
                         ko_sigma: float) -> BSSNVariables:

    """
    Perform one backward Euler timestep with dissipation.
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        ko_sigma: Kreiss-Oliger dissipation coefficient
    Returns:
        Updated BSSN variables
    """

    # using fixed point iteration to solve implicit equations
    dt = params.dt
    new_vars = vars  # initial guess

    for _ in range(10):  # 5 iterations
        print("  Backward Euler iteration")
        bssn_derivs = bssn_evolution_step(new_vars, params)
        print("  Completed BSSN evolution step in iteration")
        
        # Extract time derivatives
        dt_gamma = (bssn_derivs.conformal_metric - vars.conformal_metric) / dt
        dt_W = (bssn_derivs.conformal_factor - vars.conformal_factor) / dt
        dt_A = (bssn_derivs.traceless_K - vars.traceless_K) / dt
        dt_K = (bssn_derivs.trace_K - vars.trace_K) / dt
        dt_Gamma = (bssn_derivs.conformal_connection - vars.conformal_connection) / dt
        dt_alpha = (bssn_derivs.lapse - vars.lapse) / dt
        dt_beta = (bssn_derivs.shift - vars.shift) / dt
        
        # Add Kreiss-Oliger dissipation
        dissipation = apply_ko_dissipation_bssn(new_vars, ko_sigma, params.dx)
        
        dt_gamma += dissipation.conformal_metric
        dt_W += dissipation.conformal_factor
        dt_A += dissipation.traceless_K
        dt_K += dissipation.trace_K
        dt_Gamma += dissipation.conformal_connection
        dt_alpha += dissipation.lapse
        dt_beta += dissipation.shift
        
        # Update variables
        new_vars = BSSNVariables(
            conformal_metric=vars.conformal_metric + dt * dt_gamma,
            conformal_factor=vars.conformal_factor + dt * dt_W,
            traceless_K=vars.traceless_K + dt * dt_A,
            trace_K=vars.trace_K + dt * dt_K,
            conformal_connection=vars.conformal_connection + dt * dt_Gamma,
            lapse=vars.lapse + dt * dt_alpha,
            shift=vars.shift + dt * dt_beta
        )
    
    return new_vars
