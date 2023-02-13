############################################################################################
"""
$(TYPEDEF)
Stores information for evaluating and taking the gradient of an objective function.

# Fields
$(TYPEDFIELDS)
"""
struct AnalyticalDiffTune{G<:Function, H<:Union{Nothing, Function}} <: AbstractDifferentiableTune
    "Gradient as function of ℓobjective and parameter vector in unconstrained space, gradient(ℓobjective, θᵤ)."
    gradient::G
    "Hessian as function of ℓobjective and parameter vector in unconstrained space, hessian(ℓobjective, θᵤ)."
    hessian::H
end

############################################################################################
function update(tune::AnalyticalDiffTune, objective::Objective)
    return tune
end

############################################################################################
function _log_density(
    objective::Objective, tune::AnalyticalDiffTune, θᵤ::AbstractVector{T}
) where {T<:Real}
    return objective(θᵤ)
end
function _log_density_and_gradient(
    objective::Objective, tune::AnalyticalDiffTune, θᵤ::AbstractVector{T}
) where {T<:Real}
    return objective(θᵤ), tune.gradient(objective, θᵤ)
end
function _log_density_and_gradient_and_hessian(
    objective::Objective, tune::AnalyticalDiffTune, θᵤ::AbstractVector{T}
) where {T<:Real}
    return objective(θᵤ), tune.gradient(objective, θᵤ), tune.hessian(objective, θᵤ)
end

############################################################################################
# Export
export AnalyticalDiffTune
