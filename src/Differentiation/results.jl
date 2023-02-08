############################################################################################
"""
$(TYPEDEF)
Abstract super type for AbstractDifferentiableObjective results.
"""
abstract type ℓObjectiveResult <: BaytesCore.AbstractResult end

"""
$(TYPEDEF)
Stores result for log density and parameter for 'ℓobjective' evaluation at 'parameter'.

# Fields
$(TYPEDFIELDS)
"""
struct ℓDensityResult{T,S} <: ℓObjectiveResult
    "Parameter in unconstrained space."
    θᵤ::T
    "Log density at θᵤ."
    ℓθᵤ::S
    function ℓDensityResult(θᵤ::AbstractVector{S}, ℓθᵤ::S) where {S<:Real}
        return new{typeof(θᵤ),S}(θᵤ, ℓθᵤ)
    end
end
function ℓDensityResult(objective::Objective, θᵤ::AbstractVector{T}) where {T<:Real}
    return ℓDensityResult(θᵤ, T(objective(θᵤ)))
end
function ℓDensityResult(objective::Objective)
    return ℓDensityResult(objective, unconstrain_flatten(objective.model, objective.tagged))
end
function log_density(
    objective::Objective,
    tune::AbstractDifferentiableTune,
    θᵤ::AbstractVector{T}=unconstrain_flatten(objective.model, objective.tagged),
) where {T<:Real}
    ℓθᵤ = objective(θᵤ)
    if isfinite(ℓθᵤ)
        ℓDensityResult(θᵤ, T(ℓθᵤ))
    else
        ℓDensityResult(θᵤ, T(-Inf))
    end
end

############################################################################################
"""
$(TYPEDEF)
Stores result for log density, gradient, and parameter for 'ℓobjective' evaluation at 'parameter'.

# Fields
$(TYPEDFIELDS)
"""
struct ℓGradientResult{T<:AbstractVector,S<:Real,G<:AbstractVector} <: ℓObjectiveResult
    "Parameter in unconstrained space."
    θᵤ::T
    "Log density at θᵤ."
    ℓθᵤ::S
    "Gradient of log density at θᵤ."
    ∇ℓθᵤ::G
    function ℓGradientResult(
        θᵤ::T, ℓθᵤ::S, ∇ℓθᵤ::G
    ) where {T<:AbstractVector,S<:Real,G<:AbstractVector}
        @argcheck length(θᵤ) == length(∇ℓθᵤ)
        return new{T,S,G}(θᵤ, ℓθᵤ, ∇ℓθᵤ)
    end
end
function log_density_and_gradient(
    objective::Objective,
    tune::AbstractDifferentiableTune,
    θᵤ::AbstractVector{T}=unconstrain_flatten(objective.model, objective.tagged),
) where {T<:Real}
    ℓθᵤ, ∇ℓθᵤ = _log_density_and_gradient(objective, tune, θᵤ)
    if isfinite(ℓθᵤ)
        ℓGradientResult(θᵤ, ℓθᵤ, ∇ℓθᵤ)
    else
        len = length(θᵤ)
        ℓGradientResult(θᵤ, oftype(ℓθᵤ, -Inf), zeros(T, len))
    end
end

############################################################################################
"""
$(TYPEDEF)
Stores result for log density, gradient, hessian and parameter for 'ℓobjective' evaluation at 'parameter'.

# Fields
$(TYPEDFIELDS)
"""
struct ℓHessianResult{T<:AbstractVector,S<:Real,G<:AbstractVector,H<:AbstractMatrix} <: ℓObjectiveResult
    "Parameter in unconstrained space."
    θᵤ::T
    "Log density at θᵤ."
    ℓθᵤ::S
    "Gradient of log density at θᵤ."
    ∇ℓθᵤ::G
    "Hessian of log density at θᵤ."
    Δℓθᵤ::H
    function ℓHessianResult(
        θᵤ::T, ℓθᵤ::S, ∇ℓθᵤ::G, Δℓθᵤ::H
    ) where {T<:AbstractVector,S<:Real,G<:AbstractVector,H<:AbstractMatrix}
        @argcheck length(θᵤ) == length(∇ℓθᵤ) == size(Δℓθᵤ, 1)
        return new{T,S,G,H}(θᵤ, ℓθᵤ, ∇ℓθᵤ, Δℓθᵤ)
    end
end
function log_density_and_gradient_and_hessian(
    objective::Objective,
    tune::AbstractDifferentiableTune,
    θᵤ::AbstractVector{T}=unconstrain_flatten(objective.model, objective.tagged),
) where {T<:Real}
    ℓθᵤ, ∇ℓθᵤ, Δℓθᵤ = _log_density_and_gradient_and_hessian(objective, tune, θᵤ)
    if isfinite(ℓθᵤ)
        ℓHessianResult(θᵤ, ℓθᵤ, ∇ℓθᵤ, Δℓθᵤ)
    else
        len = length(θᵤ)
        ℓHessianResult(θᵤ, oftype(ℓθᵤ, -Inf), zeros(T, len), zeros(T, len, len) )
    end
end

############################################################################################
# Export
export
    ℓObjectiveResult,
    ℓDensityResult,
    ℓGradientResult,
    ℓHessianResult,
    log_density,
    log_density_and_gradient,
    log_density_and_gradient_and_hessian
