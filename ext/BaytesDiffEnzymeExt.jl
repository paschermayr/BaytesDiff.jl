module BaytesDiffEnzymeExt

############################################################################################
import BaytesDiff: 
    BaytesDiff, 
    AutomaticDifferentiationMethod, 
    AutomaticDiffTune,
    _config,
    _log_density_and_gradient,
    _log_density_and_gradient_and_hessian,
    ADEnzymeForward,
    ADEnzymeReverse

using ModelWrappers, BaytesDiff
using Enzyme

##############################################
function _config(
    differentiation::ADEnzymeForward, order::AbstractDiffOrder, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return nothing
end
##############################################
function AutomaticDiffTune(
    objective::Objective,
    backend::Val{:EnzymeForward},
    order::AbstractDiffOrder,
    config::C=_config(ADEnzymeForward(), order, objective, unconstrain_flatten(objective.model, objective.tagged)),
) where {C}
    return AutomaticDiffTune(ADEnzymeForward(), order, config)
end

##############################################
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune{ADEnzymeForward}, order::DiffOrderOne, θᵤ::AbstractVector{T}
) where {T<:Real}
#    error("Enzyme AD Forward Mode framework currently not implemented")
#=
    _shadow = zeros(T, ModelWrappers.length(θᵤ))
    val, _ = Enzyme.autodiff(Enzyme.ForwardMode(), objective, Enzyme.Duplicated,
        Enzyme.Duplicated(θᵤ, _shadow),
        Enzyme.Const(objective.model.arg),
        Enzyme.Const(objective.data),
    )
    T(val), _shadow
=#
    _shadow = Enzyme.onehot(θᵤ)
    ℓ, ∂θᵤ = Enzyme.autodiff(Enzyme.Forward, objective, Enzyme.BatchDuplicated,
        Enzyme.BatchDuplicated(θᵤ, _shadow),
        Enzyme.Const(objective.model.arg),
        Enzyme.Const(objective.data),
    )
    #NOTE: need to do T.(collect(∂θᵤ)) for gradient vector as Enzyme seems to fix it to Float64 for some reason
    return T(ℓ), T.(collect(∂θᵤ))
end

function _log_density_and_gradient_and_hessian(
    objective::Objective, tune::AutomaticDiffTune{ADEnzymeForward}, order::DiffOrderTwo, θᵤ::AbstractVector{T}
) where {T<:Real}
    error("Hessian for Enzyme AD framework currently not implemented")
end

############################################################################################

##############################################
function _config(
    differentiation::ADEnzymeReverse, order::AbstractDiffOrder, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return nothing
end

##############################################
function AutomaticDiffTune(
    objective::Objective,
    backend::Val{:EnzymeReverse},
    order::AbstractDiffOrder,
    config::C=_config(ADEnzymeReverse(), order, objective, unconstrain_flatten(objective.model, objective.tagged)),
) where {C}
    return AutomaticDiffTune(ADEnzymeReverse(), order, config)
end

##############################################
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune{ADEnzymeReverse}, order::DiffOrderOne, θᵤ::AbstractVector{T}
) where {T<:Real}
    # _shadow = zeros(T, ModelWrappers.length(θᵤ))
    #=
    #!NOTE: Version that computes both density and gradient that should be available soon
    val, grad = Enzyme.autodiff(Enzyme.ReverseWithPrimal(), objective, Enzyme.Duplicated,
        Enzyme.Duplicated(θᵤ, _shadow),
        Enzyme.Const(objective.model.arg),
        Enzyme.Const(objective.data),
    )
    return val, grad
    
    #!NOTE: Need to explicitly state fields of objective as constant, otherwise mutation occurs for objective.data and objective.model.arg.
    Enzyme.autodiff(Enzyme.ReverseWithPrimal(), objective, Enzyme.Active,
        Enzyme.Duplicated(θᵤ, _shadow),
        Enzyme.Const(objective.model.arg),
        Enzyme.Const(objective.data),
    )
    return T(objective(θᵤ)), _shadow
    =#
    _shadow = zeros(eltype(θᵤ), length(θᵤ))
    _, ℓ = Enzyme.autodiff(ReverseWithPrimal, objective, Enzyme.Active,
        Enzyme.Duplicated(θᵤ, _shadow),
        Enzyme.Const(objective.model.arg),
        Enzyme.Const(objective.data),
    )
    return T(ℓ), _shadow
end

function _log_density_and_gradient_and_hessian(
    objective::Objective, tune::AutomaticDiffTune{ADEnzymeReverse}, order::DiffOrderTwo, θᵤ::AbstractVector{T}
) where {T<:Real}
    error("Hessian for Enzyme AD framework currently not implemented")
end

############################################################################################
# Export
#export 

end