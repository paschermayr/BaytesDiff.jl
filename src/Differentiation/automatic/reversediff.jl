############################################################################################
struct ADReverse <: AutomaticDifferentiationMethod end
struct ADReverseUntaped <: AutomaticDifferentiationMethod end

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
export ADReverse, ADReverseUntaped
