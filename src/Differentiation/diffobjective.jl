############################################################################################
"""
$(TYPEDEF)

Objective struct with additional information about AD backend and configuration.

# Fields
$(TYPEDFIELDS)
"""
struct DiffObjective{O<:Objective,T<:AbstractDifferentiableTune}
    "Objective as function of a parameter vector in unconstrained space."
    objective::O
    "Automatic Differentiation configurations."
    tune::T
end

############################################################################################
function log_density(
    diff::DiffObjective,
    θᵤ::AbstractVector{T}=unconstrain_flatten(diff.objective.model, diff.objective.tagged),
) where {T<:Real}
    return log_density(diff.objective, diff.tune, θᵤ)
end
function log_density_and_gradient(
    diff::DiffObjective,
    θᵤ::AbstractVector{T}=unconstrain_flatten(diff.objective.model, diff.objective.tagged),
) where {T<:Real}
    return log_density_and_gradient(diff.objective, diff.tune, θᵤ)
end
function log_density_and_gradient_and_hessian(
    diff::DiffObjective,
    θᵤ::AbstractVector{T}=unconstrain_flatten(diff.objective.model, diff.objective.tagged),
) where {T<:Real}
    return log_density_and_gradient_and_hessian(diff.objective, diff.tune, θᵤ)
end

############################################################################################
#export
export DiffObjective,
    log_density,
    log_density_and_gradient,
    log_density_and_gradient_and_hessian
