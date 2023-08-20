module BaytesDiffForwardDiffExt

############################################################################################
import BaytesDiff: 
    BaytesDiff, 
    AutomaticDifferentiationMethod, 
    AutomaticDiffTune,
    _config,
    _log_density_and_gradient,
    _log_density_and_gradient_and_hessian,
    ADForward

using ModelWrappers, BaytesDiff, DiffResults
using ForwardDiff, DiffResults

using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
    
############################################################################################
#!NOTE: Need to define new struct in actual Code, ext only here to dispatch existing functions
#struct ADForward <: AutomaticDifferentiationMethod end

############################################################################################
"""
$(SIGNATURES)
Initiate DiffResults.MutableDiffResult struct buffer for gradients. Not exported.

# Examples
```julia
```

"""
function _diffresults_gradientbuffer(θᵤ::AbstractVector{T}) where {T<:Real}
    #NOTE: Adjusted from: https://github.com/tpapp/LogDensityProblems.jl/blob/master/src/DiffResults_helpers.jl
    S = T <: Real ? T : Float64
    return DiffResults.MutableDiffResult(zero(S), (similar(θᵤ, S),))
end

"""
$(SIGNATURES)
Initiate DiffResults.MutableDiffResult struct buffer for hessian. Not exported.

# Examples
```julia
```

"""
function _diffresults_hessianbuffer(θᵤ::AbstractVector{T}) where {T<:Real}
    S = T <: Real ? T : Float64
    len = length(θᵤ)
    return DiffResults.MutableDiffResult(zero(S), (similar(θᵤ, S), zeros(S, len, len)))
end

##############################################
function _config(
    differentiation::ADForward, order::DiffOrderZero, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return nothing
end
function _config(
    differentiation::ADForward, order::DiffOrderOne, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return ForwardDiff.GradientConfig(objective, θᵤ)
end
function _config(
    differentiation::ADForward, order::DiffOrderTwo, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    result = _diffresults_hessianbuffer(θᵤ)
    cf = ForwardDiff.HessianConfig(objective, result, θᵤ)
    return cf
end

##############################################
function AutomaticDiffTune(
    objective::Objective,
    backend::Val{:ForwardDiff},
    order::AbstractDiffOrder,
    config::C=_config(ADForward(), order, objective, unconstrain_flatten(objective.model, objective.tagged)),
) where {C}
    return AutomaticDiffTune(ADForward(), order, config)
end

##############################################
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune{<:ADForward}, order::DiffOrderOne, θᵤ::AbstractVector{T}
) where {T<:Real}
    buffer = _diffresults_gradientbuffer(θᵤ)
    result = ForwardDiff.gradient!(buffer, objective, θᵤ, tune.config)
    return DiffResults.value(result), DiffResults.gradient(result)
end
function _log_density_and_gradient_and_hessian(
    objective::Objective, tune::AutomaticDiffTune{ADForward}, order::DiffOrderTwo, θᵤ::AbstractVector{T}
) where {T<:Real}
    buffer = _diffresults_hessianbuffer(θᵤ)
    result = ForwardDiff.hessian!(buffer, objective, θᵤ, tune.config)
    return DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
end

############################################################################################
# Export

end