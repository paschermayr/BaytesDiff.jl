module BaytesDiffZygoteExt

############################################################################################
import BaytesDiff: 
    BaytesDiff, 
    AutomaticDifferentiationMethod, 
    AutomaticDiffTune,
    _config,
    _log_density_and_gradient,
    _log_density_and_gradient_and_hessian,
    ADZygote

using ModelWrappers, BaytesDiff
using Zygote

##############################################
function _config(
    differentiation::ADZygote, order::AbstractDiffOrder, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return nothing
end

##############################################
function AutomaticDiffTune(
    objective::Objective,
    backend::Val{:Zygote},
    order::AbstractDiffOrder,
    config::C=_config(ADZygote(), order, objective, unconstrain_flatten(objective.model, objective.tagged)),
) where {C}
    return AutomaticDiffTune(ADZygote(), order, config)
end

##############################################
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune{ADZygote}, order::DiffOrderOne, θᵤ::AbstractVector{T}
) where {T<:Real}
    _val, back = Zygote.pullback(objective, θᵤ)
    return T(_val), first(back(Zygote.sensitivity(_val)))
end
function _log_density_and_gradient_and_hessian(
    objective::Objective, tune::AutomaticDiffTune{ADZygote}, order::DiffOrderTwo, θᵤ::AbstractVector{T}
) where {T<:Real}
    error("Hessian for Zygote AD framework currently not implemented")
end

############################################################################################
# Export
#export

end