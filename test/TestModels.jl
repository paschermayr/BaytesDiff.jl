################################################################################
dat = randn(_RNG, 100)
dat[1:5] = collect(1:5) .+ 0.0
μ₀ = 1.0
σ₀ = 2.0
_args = (a =  1., b = collect(1:100) .+ 0.0)
################################################################################
# Create custom Model ~ Name it to avoid name collision
_param = (μ=Param(Distributions.Normal(), μ₀,), σ=Param(Distributions.Exponential(), σ₀,))
struct Benchmark <: ModelName end
modelBM = ModelWrapper(Benchmark(), _param)
obectiveBM = Objective(modelBM, dat)

function (objective::Objective{<:ModelWrapper{Benchmark}})(θ::NamedTuple)
    lp =
        Distributions.logpdf(Distributions.Normal(), θ.μ) +
        Distributions.logpdf(Distributions.Exponential(), θ.σ)
    ll = sum(
        Distributions.logpdf(Distributions.Normal(θ.μ, θ.σ), objective.data[iter]) for
        iter in eachindex(objective.data)
    )
    return lp + ll
end

################################################################################
# Create custom Model ~ Name it to avoid name collision
_paramHBM = (μ=Param(Distributions.Normal(), μ₀,), σ=Param(Distributions.Exponential(), σ₀,))
struct HessianBM <: ModelName end
modelHBM = ModelWrapper(HessianBM(), _paramHBM)
obectiveHBM = Objective(modelHBM, dat)

function (objective::Objective{<:ModelWrapper{HessianBM}})(θ::NamedTuple)
    return 0.0
end

################################################################################
# Create custom Model that works with Enzyme and checks if constant fields are mutated
struct EnzymeBM <: ModelName end
_paramEnzyme = (μ=Param(Distributions.Normal(), μ₀,), σ=Param(Distributions.Exponential(), σ₀,))
modelEBM = ModelWrapper(EnzymeBM(), _paramEnzyme, _args)
obectiveEBM = Objective(modelEBM, dat)

function (objective::Objective{<:ModelWrapper{EnzymeBM}})(θ::NamedTuple, arg::A = objective.model.arg, data::D = objective.data) where {A, D}
    μ = θ.μ + arg.a + mean(arg.b)
    #=
    #!NOTE: Up until at least Enzyme 10.16, Enzyme mutates model.arg and objective.data even if specified as constant
    lp =
        Distributions.logpdf(Distributions.Normal(), μ) +
        Distributions.logpdf(Distributions.Exponential(), θ.σ)
    ll = sum( logpdf(Normal(μ, θ.σ), dat) for dat in data )
    return ll #+ lp
    =#
    ll = sum( logpdf(Normal(μ, θ.σ), dat) for dat in data )
    return ll
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
