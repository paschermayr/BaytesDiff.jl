module BaytesDiffReverseDiffExt

############################################################################################
import BaytesDiff: 
    BaytesDiff, 
    AutomaticDifferentiationMethod, 
    AutomaticDiffTune,
    _config,
    checkfinite,
    _log_density_and_gradient,
    _log_density_and_gradient_and_hessian,
    ADReverse,
    ADReverseUntaped

using ModelWrappers, BaytesDiff, DiffResults
#import ModelWrappers: ModelWrappers, _checkfinite
using ReverseDiff, DiffResults

using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
    
############################################################################################
#Helper function for ReverseDiff package
function checkfinite(θ::ReverseDiff.TrackedArray{T}, max_val::R=max_val) where {T,R<:Real}
    @inbounds @simd for iter in eachindex(θ)
        if !checkfinite(θ[iter], max_val=max_val)
            return false
        end
    end
    return true
end

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
    differentiation::ADReverse, order::DiffOrderZero, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return nothing
end
function _config(
    differentiation::ADReverse, order::DiffOrderOne, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return ReverseDiff.compile(ReverseDiff.GradientTape(objective, θᵤ))
end
function _config(
    differentiation::ADReverse, order::DiffOrderTwo, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return ReverseDiff.compile(ReverseDiff.HessianTape(objective, θᵤ))
end

##############################################
function AutomaticDiffTune(
    objective::Objective,
    backend::Val{:ReverseDiff},
    order::AbstractDiffOrder,
    config::C=_config(ADReverse(), order, objective, unconstrain_flatten(objective.model, objective.tagged)),
) where {C}
    return AutomaticDiffTune(ADReverse(), order, config)
end

##############################################
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune{ADReverse}, order::DiffOrderOne, θᵤ::AbstractVector{T}
) where {T<:Real}
    buffer = _diffresults_gradientbuffer(θᵤ)
    result = ReverseDiff.gradient!(buffer, tune.config, θᵤ)
    return DiffResults.value(result), DiffResults.gradient(result)
end
function _log_density_and_gradient_and_hessian(
    objective::Objective, tune::AutomaticDiffTune{ADReverse}, order::DiffOrderTwo, θᵤ::AbstractVector{T}
) where {T<:Real}
    buffer = _diffresults_hessianbuffer(θᵤ)
    result = ReverseDiff.hessian!(buffer, tune.config, θᵤ)
    return DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
end
############################################################################################

##############################################
function _config(
    differentiation::ADReverseUntaped, order::AbstractDiffOrder, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return nothing
end

##############################################
function AutomaticDiffTune(
    objective::Objective,
    backend::Val{:ReverseDiffUntaped},
    order::AbstractDiffOrder,
    config::C=_config(ADReverseUntaped(), order, objective, unconstrain_flatten(objective.model, objective.tagged)),
) where {C}
    return AutomaticDiffTune(ADReverseUntaped(), order, config)
end

##############################################
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune{ADReverseUntaped}, order::DiffOrderOne, θᵤ::AbstractVector{T}
) where {T<:Real}
    buffer = _diffresults_gradientbuffer(θᵤ)
    result = ReverseDiff.gradient!(buffer, objective, θᵤ)
    return DiffResults.value(result), DiffResults.gradient(result)
end
function _log_density_and_gradient_and_hessian(
    objective::Objective, tune::AutomaticDiffTune{ADReverseUntaped}, order::DiffOrderTwo, θᵤ::AbstractVector{T}
) where {T<:Real}
    buffer = _diffresults_hessianbuffer(θᵤ)
    result = ReverseDiff.hessian!(buffer, objective, θᵤ)
    return DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
end

############################################################################################
# Export
export checkfinite

end