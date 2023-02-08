############################################################################################
# Define methods that need to be dispatched on if new AD engines are added
"""
$(TYPEDEF)
Abstract super type for Tuning structs of differentiable functions.
"""
abstract type AbstractDifferentiableTune end

"""
$(SIGNATURES)
Write config file for AD wrapper.

# Examples
```julia
```

"""
function _config end

"""
    $(FUNCTIONNAME)(objective, θᵤ)
Compute log density of 'objective' at 'θᵤ'.

# Examples
```julia
```

"""
function log_density end

"""
    $(FUNCTIONNAME)(objective, θᵤ)
Compute log density and gradient of 'objective' at 'θᵤ'.

# Examples
```julia
```

"""
function log_density_and_gradient end

"""
    $(FUNCTIONNAME)(objective, θᵤ)
Compute log density, gradient and hessian of 'objective' at 'θᵤ'.

# Examples
```julia
```

"""
function log_density_and_gradient_and_hessian end

############################################################################################
# Include
include("utility.jl")

include("analytic/analytic.jl")
include("automatic/automatic.jl")

include("diffobjective.jl")
include("results.jl")
include("checks.jl")

############################################################################################
# Export
export update!, log_density, log_density_and_gradient, log_density_and_gradient_and_hessian
