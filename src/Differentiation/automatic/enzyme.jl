############################################################################################
struct ADEnzymeForward <: AutomaticDifferentiationMethod end
struct ADEnzymeReverse <: AutomaticDifferentiationMethod end

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
    error("Enzyme AD Forward Mode framework currently not implemented")
#=
    _shadow = zeros(T, ModelWrappers.length(θᵤ))
    val, _ = Enzyme.autodiff(Enzyme.ForwardMode(), objective, Enzyme.Duplicated,
        Enzyme.Duplicated(θᵤ, _shadow),
        Enzyme.Const(objective.model.arg),
        Enzyme.Const(objective.data),
    )
    T(val), _shadow
=#
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
    _shadow = zeros(T, ModelWrappers.length(θᵤ))
    #=
    #!NOTE: Version that computes both density and gradient that should be available soon
    val, grad = Enzyme.autodiff(Enzyme.ReverseWithPrimal(), objective, Enzyme.Duplicated,
        Enzyme.Duplicated(θᵤ, _shadow),
        Enzyme.Const(objective.model.arg),
        Enzyme.Const(objective.data),
    )
    return val, grad
    =#
    #!NOTE: Need to explicitly state fields of objective as constant, otherwise mutation occurs for objective.data and objective.model.arg.
    Enzyme.autodiff(Enzyme.ReverseWithPrimal(), objective, Enzyme.Active,
        Enzyme.Duplicated(θᵤ, _shadow),
        Enzyme.Const(objective.model.arg),
        Enzyme.Const(objective.data),
    )
    return T(objective(θᵤ)), _shadow
end

function _log_density_and_gradient_and_hessian(
    objective::Objective, tune::AutomaticDiffTune{ADEnzymeReverse}, order::DiffOrderTwo, θᵤ::AbstractVector{T}
) where {T<:Real}
    error("Hessian for Enzyme AD framework currently not implemented")
end

############################################################################################
# Export
export ADEnzymeForward, ADEnzymeReverse
