################################################################################
dat = randn(100)
μ₀ = 1.0
σ₀ = 2.0
################################################################################
# Create custom Model ~ Name it to avoid name collision
_param = (μ=Param(μ₀, Distributions.Normal()), σ=Param(σ₀, Distributions.Exponential()))
struct SossBenchmark <: ModelName end
modelSossBM = ModelWrapper(SossBenchmark(), _param)
obectiveSossBM = Objective(modelSossBM, dat)

function (objective::Objective{<:ModelWrapper{SossBenchmark}})(θ::NamedTuple)
    lp =
        Distributions.logpdf(Distributions.Normal(), θ.μ) +
        Distributions.logpdf(Distributions.Exponential(), θ.σ)
    ll = sum(
        Distributions.logpdf(Distributions.Normal(θ.μ, θ.σ), objective.data[iter]) for
        iter in eachindex(objective.data)
    )
    return lp + ll
end

############################################################################################
# Test Gradients of Model with custom function
modelExample = ModelWrapper(ExampleModel(), _val_examplemodel)
data1 = randn(N)
data2 = rand(Distributions.MvNormal(Diagonal(map(abs2, [1.0, 1.0]))), N)
data3 = rand(Distributions.Categorical(3), N)
_idx = rand(1:2, N)
objectiveExample = Objective(modelExample, (data1, data2, data3, _idx))

function (objective::Objective{<:ModelWrapper{ExampleModel}})(θ::NamedTuple)
    data1 = objective.data[1]
    data2 = objective.data[2]
    data3 = objective.data[3]
    σ2 = Symmetric(Diagonal(θ.σ2) * θ.ρ2 * Diagonal(θ.σ2))
    σ4 = [
        Symmetric(Diagonal(θ.σ4[iter]) * θ.ρ4[iter] * Diagonal(θ.σ4[iter])) for
        iter in eachindex(θ.σ4)
    ]
    _dist1 = Distributions.Normal(θ.μ1, θ.σ1)
    _dist2 = Distributions.MvNormal(θ.μ2, σ2)
    _dist3 = [Distributions.Normal(θ.μ3[iter], θ.σ3[iter]) for iter in eachindex(θ.μ3)]
    _dist4 = [Distributions.MvNormal(θ.μ4[iter], σ4[iter]) for iter in eachindex(θ.μ3)]
    _dist5 = Distributions.Categorical(θ.p)

    ll = sum(Distributions.logpdf(_dist1, data1[iter]) for iter in eachindex(data1))
    ll2 = sum(
        Distributions.logpdf(_dist2, @view(data2[:, iter])) for iter in size(data2, 2)
    )

    ll3 = sum(
        Distributions.logpdf(_dist3[_idx[iter]], data1[iter]) for iter in eachindex(data1)
    )
    ll4 = sum(
        Distributions.logpdf(_dist4[_idx[iter]], @view(data2[:, iter])) for
        iter in size(data2, 2)
    )

    ll5 = sum(Distributions.logpdf(_dist5, data3[iter]) for iter in eachindex(data3))

    return ll + ll2 + ll3 + ll4 + ll5
end
############################################################################################
