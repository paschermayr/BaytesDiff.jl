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

############################################################################################
#Helper function for ReverseDiff package
function checkfinite(θ::ReverseDiff.TrackedArray{T}, max_val::R=max_val) where {T,R<:Real}
    @inbounds @simd for iter in eachindex(θ)
        if !checkfinite(θ[iter], max_val=max_val)
            return false
        end
    end
    return true
end

############################################################################################
#Export
export checkfinite
