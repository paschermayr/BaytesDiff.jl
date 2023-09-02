module BaytesDiffFiniteDifferencesExt

############################################################################################
import BaytesDiff: 
    BaytesDiff, 
    AutomaticDifferentiationMethod, 
    AutomaticDiffTune,
    _config,
    _log_density_and_gradient,
    _log_density_and_gradient_and_hessian,
    ADFiniteDifferences

using ModelWrappers, BaytesDiff
using FiniteDifferences

using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
    
##############################################
function _config(
    differentiation::ADFiniteDifferences, order::DiffOrderZero, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return FiniteDifferences.central_fdm(5, 1)
end
function _config(
    differentiation::ADFiniteDifferences, order::DiffOrderOne, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return FiniteDifferences.central_fdm(5, 1)
end
function _config(
    differentiation::ADFiniteDifferences, order::DiffOrderTwo, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return FiniteDifferences.central_fdm(5, 1)
end

##############################################
function AutomaticDiffTune(
    objective::Objective,
    backend::Val{:FiniteDifferences},
    order::AbstractDiffOrder,
    config::C=_config(ADFiniteDifferences(), order, objective, unconstrain_flatten(objective.model, objective.tagged)),
) where {C}
    return AutomaticDiffTune(ADFiniteDifferences(), order, config)
end

##############################################
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune{<:ADFiniteDifferences}, order::DiffOrderOne, θᵤ::AbstractVector{T}
) where {T<:Real}
    val = objective(θᵤ)
    ∇val = only(FiniteDifferences.grad(tune.config, objective, θᵤ))
    return T(val), ∇val
end
function _log_density_and_gradient_and_hessian(
    objective::Objective, tune::AutomaticDiffTune{ADFiniteDifferences}, order::DiffOrderTwo, θᵤ::AbstractVector{T}
) where {T<:Real}
    return error("Hessian for FiniteDifferences AD framework currently not implemented")
end

############################################################################################
# Export

end