#=
############################################################################################
"""
$(SIGNATURES)
Initiate DiffResults.MutableDiffResult struct buffer for gradients. Not exported.

# Examples
```julia
```

"""
function _diffresults_gradientbuffer(θᵤ::AbstractVector{T}) where {T<:Real}
    #NOTE: Adjusted from: https://github.com/tpapp/LogDensityProblems.jl/blob/master/src/DiffResults_helpers.jl
    S = T <: Real ? T : Float64
    return DiffResults.MutableDiffResult(zero(S), (similar(θᵤ, S),))
end

"""
$(SIGNATURES)
Initiate DiffResults.MutableDiffResult struct buffer for hessian. Not exported.

# Examples
```julia
```

"""
function _diffresults_hessianbuffer(θᵤ::AbstractVector{T}) where {T<:Real}
    S = T <: Real ? T : Float64
    len = length(θᵤ)
    return DiffResults.MutableDiffResult(zero(S), (similar(θᵤ, S), zeros(S, len, len)))
end
=#
############################################################################################
#Export
#export checkfinite
