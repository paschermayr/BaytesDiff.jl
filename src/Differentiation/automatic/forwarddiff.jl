############################################################################################
struct ADForward <: AutomaticDifferentiationMethod end

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
export ADForward
