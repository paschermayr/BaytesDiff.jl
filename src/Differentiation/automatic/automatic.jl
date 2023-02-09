############################################################################################
"""
$(TYPEDEF)
Abstract super type for Supported Automatic Differentiation backends.
"""
abstract type AutomaticDifferentiationMethod end

############################################################################################
struct AutomaticDiffTune{M<:AutomaticDifferentiationMethod,O<:AbstractDiffOrder,C} <: AbstractDifferentiableTune
    "Automatic Differentiation (AD) backend."
    backend::M
    "Determines differentiability of objective function and configuration that is created when initiating a AutomaticDiffTune struct."
    order::O
    "Chunck size configuration for AD backend."
    config::C
    function AutomaticDiffTune(backend::M, order::O, config::C
    ) where {M<:AutomaticDifferentiationMethod,O<:AbstractDiffOrder,C}
        return new{M,O,C}(backend, order, config)
    end
end
function AutomaticDiffTune(objective::Objective, backend::M, order::O = DiffOrderOne()) where {M<:AutomaticDifferentiationMethod, O<:AbstractDiffOrder}
    return AutomaticDiffTune(
        backend, order, _config(backend, order, objective, unconstrain_flatten(objective.model, objective.tagged))
    )
end
function AutomaticDiffTune(objective::Objective, backend::Symbol, order::AbstractDiffOrder = DiffOrderOne())
    #!NOTE: Make Symbol intialization easier
    return AutomaticDiffTune(objective, Val(backend), order)
end

############################################################################################
include("forwarddiff.jl")
include("reversediff.jl")
include("zygote.jl")
include("enzyme.jl")

############################################################################################
function update(
    tune::AutomaticDiffTune,
    objective::Objective)
    return AutomaticDiffTune(objective, tune.backend, tune.order)
end

############################################################################################
function _log_density(
    objective::Objective, tune::AutomaticDiffTune, order::AbstractDiffOrder, θᵤ::AbstractVector{T}
) where {T<:Real}
    return objective(θᵤ)
end

function _log_density(
    objective::Objective, tune::AutomaticDiffTune, θᵤ::AbstractVector{T}
) where {T<:Real}
    return _log_density(objective, tune, tune.order, θᵤ)
end
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune, θᵤ::AbstractVector{T}
) where {T<:Real}
    return _log_density_and_gradient(objective, tune, tune.order, θᵤ)
end
function _log_density_and_gradient_and_hessian(
    objective::Objective, tune::AutomaticDiffTune, θᵤ::AbstractVector{T}
) where {T<:Real}
    return _log_density_and_gradient_and_hessian(objective, tune, tune.order, θᵤ)
end


############################################################################################
# Export
export AutomaticDiffTune
