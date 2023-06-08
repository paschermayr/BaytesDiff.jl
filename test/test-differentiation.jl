############################################################################################
# Differentiation - Benchmark Model
modelExample = ModelWrapper(ExampleModel(), _val_examplemodel)
objectiveExample = Objective(modelExample, (data1, data2, data3, _idx))

@testset "AutoDiffContainer - Log Objective Results" begin
    AutomaticDiffTune(objectiveExample, :ForwardDiff)
    tune_fwd = AutomaticDiffTune(objectiveExample, :ForwardDiff, DiffOrderOne())
    fwd = DiffObjective(objectiveExample, tune_fwd)
    theta_unconstrained = randn(_RNG, length(modelExample))
    ## Compute logdensity
    log_density(objectiveExample, tune_fwd, theta_unconstrained)
    theta_unconstrained2 = deepcopy(theta_unconstrained)
    #!NOTE: 10th parameter in likelihood for example, so is not compiled away in Reverse Tape
    theta_unconstrained2[10] = Inf
    ## Check if result is finite
    _ld_fin = log_density(objectiveExample, tune_fwd, copy(theta_unconstrained))
    _ld_inf = log_density(objectiveExample, tune_fwd, copy(theta_unconstrained2))
    _ld_inf_fault = log_density(objectiveExample, tune_fwd, copy(theta_unconstrained))
    _ld_inf_fault.θᵤ[10] = Inf

    @test BaytesDiff.checkfinite(theta_unconstrained)
    @test !BaytesDiff.checkfinite(theta_unconstrained2)

    @test BaytesDiff.checkfinite(_ld_fin)
    @test !BaytesDiff.checkfinite(_ld_inf)
    @test !BaytesDiff.checkfinite(_ld_inf_fault)

    @test BaytesDiff.checkfinite(_ld_inf, _ld_fin) #Infinite to Finite
    @test !BaytesDiff.checkfinite(_ld_fin, _ld_inf) #Finite to Infinite

    @test BaytesDiff.checkfinite(-Inf, 10.0, _ld_fin) #Infinite to Finite
    @test !BaytesDiff.checkfinite(10.0, -Inf, _ld_fin) #Finite to Infinite
    @test !BaytesDiff.checkfinite(-Inf, 10.0, _ld_inf) #Finite to Infinite


    #!ToDo: Check for correct error message in case logdensity cannot be evaluated
    BaytesDiff.checkfinite(objectiveExample, theta_unconstrained)
    BaytesDiff.checkfinite(objectiveExample, _ld_fin)
    BaytesDiff.checkfinite(objectiveExample, _ld_inf, _ld_fin)
    BaytesDiff.checkfinite(objectiveExample, -Inf, 10.0, _ld_fin)

    err = ObjectiveError(objectiveExample, -Inf, theta_unconstrained)
    @test isa(err, ArgCheck.Exception)

end

@testset "AutoDiffContainer - Log Objective AutoDiff compatibility - Vectorized Model" begin
    ## Assign DiffTune
    tune_fwd = AutomaticDiffTune(objectiveExample, :ForwardDiff,)
    tune_rd = AutomaticDiffTune(objectiveExample, :ReverseDiff)
    tune_zyg = AutomaticDiffTune(objectiveExample, :Zygote,)
    fwd = DiffObjective(objectiveExample, tune_fwd)
    rd = DiffObjective(objectiveExample, tune_rd)
    zyg = DiffObjective(objectiveExample, tune_zyg)
    theta_unconstrained = randn(_RNG, length(modelExample))
    ## Compute logdensity
    log_density(objectiveExample, tune_fwd, theta_unconstrained)
    log_density(fwd, theta_unconstrained)
    theta_unconstrained2 = deepcopy(theta_unconstrained)
    #!NOTE: 10th parameter in likelihood for example, so is not compiled away in Reverse Tape
    theta_unconstrained2[10] = Inf
    _ld = log_density(objectiveExample, tune_fwd, theta_unconstrained2)
    @test isinf(_ld.ℓθᵤ)
    _ld = log_density(fwd, theta_unconstrained2)
    @test isinf(_ld.ℓθᵤ)
    _ld = log_density(rd, theta_unconstrained2)
    @test isinf(_ld.ℓθᵤ)
    _ld = log_density(zyg, theta_unconstrained2)
    @test isinf(_ld.ℓθᵤ)

    ld1 = log_density(fwd, theta_unconstrained)
    ld2 = log_density(rd, theta_unconstrained)
    ld3 = log_density(zyg, theta_unconstrained)
    _ld1 = BaytesDiff._log_density(objectiveExample, tune_fwd, theta_unconstrained)
    _ld2 = BaytesDiff._log_density(objectiveExample, tune_rd, theta_unconstrained)
    _ld3 = BaytesDiff._log_density(objectiveExample, tune_zyg, theta_unconstrained)
    ## Compute Diffresult
    log_density_and_gradient(objectiveExample, tune_fwd, theta_unconstrained2)
    _grad = log_density_and_gradient(fwd, theta_unconstrained2)
    @test isinf(_grad.ℓθᵤ)
    log_density_and_gradient(objectiveExample, tune_rd, theta_unconstrained2)
    _grad = log_density_and_gradient(rd, theta_unconstrained2)
    #!TODO: Need an example where tape does not compile parameter in infi
    @test isinf(_grad.ℓθᵤ)
    log_density_and_gradient(objectiveExample, tune_zyg, theta_unconstrained2)
    _grad = log_density_and_gradient(zyg, theta_unconstrained2)
    @test isinf(_grad.ℓθᵤ)

    _grad1 = BaytesDiff._log_density_and_gradient(objectiveExample, tune_fwd, theta_unconstrained)
    _grad2 = BaytesDiff._log_density_and_gradient(objectiveExample, tune_rd, theta_unconstrained)
    _grad3 = BaytesDiff._log_density_and_gradient(objectiveExample, tune_zyg, theta_unconstrained)
    grad1 = log_density_and_gradient(fwd, theta_unconstrained)
    grad2 = log_density_and_gradient(rd, theta_unconstrained)
    grad3 = log_density_and_gradient(zyg, theta_unconstrained)

    ## Compute manual call ~ Already checked for equality
    ld = objectiveExample(theta_unconstrained)
    grad_mod_fd = ForwardDiff.gradient(objectiveExample, theta_unconstrained)
    grad_mod_rd = ReverseDiff.gradient(objectiveExample, theta_unconstrained)
    grad_mod_zy = Zygote.gradient(objectiveExample, theta_unconstrained)[1]
    ## Compare results
    @test ld - ld1.ℓθᵤ ≈ 0 atol = _TOL
    @test ld - ld2.ℓθᵤ ≈ 0 atol = _TOL
    @test ld - ld3.ℓθᵤ ≈ 0 atol = _TOL
    @test sum(abs.(_grad1[2] - grad1.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(_grad2[2] - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(_grad3[2] - grad3.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad1.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_rd - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_zy - grad3.∇ℓθᵤ)) ≈ 0 atol = _TOL
    ## Checks
    _output = check_gradients(_RNG, objectiveExample, [:ForwardDiff, :ReverseDiff, :Zygote]; printoutput = false)
    @test sum(abs.(_output.ℓobjective_gradient_diff)) ≈ 0 atol = _TOL
    ## Update DiffTune
    BaytesDiff.update(tune_fwd, objectiveExample)
    BaytesDiff.update(tune_rd, objectiveExample)
    BaytesDiff.update(tune_zyg, objectiveExample)
    ## Config DiffTune
    theta_unconstrained2 = randn(_RNG, length(objectiveExample))
    BaytesDiff._config(BaytesDiff.ADForward(), DiffOrderZero(), objectiveExample, theta_unconstrained2)
    BaytesDiff._config(BaytesDiff.ADReverse(), DiffOrderZero(), objectiveExample, theta_unconstrained2)
    BaytesDiff._config(BaytesDiff.ADReverseUntaped(), DiffOrderZero(), objectiveExample, theta_unconstrained2)
    BaytesDiff._config(BaytesDiff.ADZygote(), DiffOrderZero(), objectiveExample, theta_unconstrained2)

    BaytesDiff._config(BaytesDiff.ADForward(), DiffOrderOne(), objectiveExample, theta_unconstrained2)
    BaytesDiff._config(BaytesDiff.ADReverse(), DiffOrderOne(), objectiveExample, theta_unconstrained2)
    BaytesDiff._config(BaytesDiff.ADReverseUntaped(), DiffOrderOne(), objectiveExample, theta_unconstrained2)
    BaytesDiff._config(BaytesDiff.ADZygote(), DiffOrderOne(), objectiveExample, theta_unconstrained2)

    BaytesDiff._config(BaytesDiff.ADForward(), DiffOrderTwo(), objectiveExample, theta_unconstrained2)
#    BaytesDiff._config(BaytesDiff.ADReverse(), DiffOrderTwo(), objectiveExample, theta_unconstrained2)
    BaytesDiff._config(BaytesDiff.ADReverseUntaped(), DiffOrderTwo(), objectiveExample, theta_unconstrained2)
    BaytesDiff._config(BaytesDiff.ADZygote(), DiffOrderTwo(), objectiveExample, theta_unconstrained2)
end


objectiveHessian = Objective(modelHBM, data1)
@testset "AutoDiffContainer - Log Objective AutoDiff compatibility - Vectorized Model Hessian" begin
    ## Assign DiffTune
    _objective = objectiveExample #objectiveHessian
    tune_fwd = AutomaticDiffTune(_objective, :ForwardDiff, DiffOrderTwo())
#    tune_rd = AutomaticDiffTune(_objective, :ReverseDiff, DiffOrderTwo())
    tune_rdu = AutomaticDiffTune(_objective, :ReverseDiffUntaped, DiffOrderTwo())
    tune_zyg = AutomaticDiffTune(_objective, :Zygote, DiffOrderTwo())

    fwd = DiffObjective(_objective, tune_fwd)
#    rd = DiffObjective(_objective, tune_rd)
    rdu = DiffObjective(_objective, tune_rdu)
    zyg = DiffObjective(_objective, tune_zyg)
    theta_unconstrained = randn(_RNG, length(modelExample))

    ## Compute logdensity
    _ld = log_density(_objective, tune_fwd, theta_unconstrained)
    ## Compute Diffresult

    _grad1 = BaytesDiff._log_density_and_gradient_and_hessian(_objective, tune_fwd, theta_unconstrained)
#    _grad2 = BaytesDiff._log_density_and_gradient(_objective, tune_rd, theta_unconstrained)
#    _grad22 = BaytesDiff._log_density_and_gradient_and_hessian(_objective, tune_rdu, theta_unconstrained)
#    _grad3 = BaytesDiff._log_density_and_gradient_and_hessian(_objective, tune_zyg, theta_unconstrained)
    grad1 = BaytesDiff.log_density_and_gradient_and_hessian(fwd, theta_unconstrained)
#    grad2 = BaytesDiff.log_density_and_gradient_and_hessian(rd, theta_unconstrained)
#    grad22 = BaytesDiff.log_density_and_gradient_and_hessian(rdu, theta_unconstrained)
#    grad3 = BaytesDiff.log_density_and_gradient_and_hessian(zyg, theta_unconstrained)

    ## Compute manual call ~ Already checked for equality
    grad_mod_fd = ForwardDiff.gradient(_objective, theta_unconstrained)
    hess_mod_fd = ForwardDiff.hessian(_objective, theta_unconstrained)

#    grad_mod_rd = ReverseDiff.gradient(_objective, theta_unconstrained)
#    hess_mod_rd = ReverseDiff.hessian(_objective, theta_unconstrained)

#    grad_mod_zy = Zygote.gradient(_objective, theta_unconstrained)[1]    grad_mod_fd = ForwardDiff.hessian(_objective, theta_unconstrained)
#    hess_mod_zy = Zygote.hessian(_objective, theta_unconstrained)[1]
    ## Compare results
    @test sum(abs.(grad_mod_fd - grad1.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(hess_mod_fd - grad1.Δℓθᵤ)) ≈ 0 atol = _TOL

#    @test sum(abs.(grad_mod_rd - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL
#    @test sum(abs.(grad_mod_rd - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL

#    @test sum(abs.(grad_mod_zy - grad3.∇ℓθᵤ)) ≈ 0 atol = _TOL
#    @test sum(abs.(grad_mod_zy - grad3.∇ℓθᵤ)) ≈ 0 atol = _TOL

end

############################################################################################
# Differentiation - Enzyme
#=
@testset "AutoDiffContainer - Log Objective AutoDiff compatibility - Enzyme" begin
    ## Assign DiffTune
    _objective = obectiveEBM
    tune_fwd0 = AutomaticDiffTune(_objective, :EnzymeForward, DiffOrderZero())
    tune_fwd1 = AutomaticDiffTune(_objective, :EnzymeForward, DiffOrderOne())
    tune_fwd2 = AutomaticDiffTune(_objective, :EnzymeForward, DiffOrderTwo())
    tune_rd0 = AutomaticDiffTune(_objective, :EnzymeReverse, DiffOrderZero())
    tune_rd1 = AutomaticDiffTune(_objective, :EnzymeReverse, DiffOrderOne())
    tune_rd2 = AutomaticDiffTune(_objective, :EnzymeReverse, DiffOrderTwo())

    fwd0 = DiffObjective(_objective, tune_fwd0)
    fwd1 = DiffObjective(_objective, tune_fwd1)
    fwd2 = DiffObjective(_objective, tune_fwd2)
    rd0 = DiffObjective(_objective, tune_rd0)
    rd1 = DiffObjective(_objective, tune_rd1)
    rd2 = DiffObjective(_objective, tune_rd2)

    theta_unconstrained = randn(_RNG, length(modelExample))

    ## Compute logdensity
    _ld = log_density(_objective, tune_fwd0, theta_unconstrained)

    ## Compute Gradient
    # grad1 = BaytesDiff.log_density_and_gradient(fwd1, theta_unconstrained)
    grad2 = BaytesDiff.log_density_and_gradient(rd1, theta_unconstrained)

    ## Compute Hessian
    # hess1 = BaytesDiff.log_density_and_gradient_and_hessian(fwd2, theta_unconstrained)
    # hess2 = BaytesDiff.log_density_and_gradient_and_hessian(rd2, theta_unconstrained)

    ## Compute manual call ~ Already checked for equality
    grad_mod_fd = ForwardDiff.gradient(_objective, theta_unconstrained)
    hess_mod_fd = ForwardDiff.hessian(_objective, theta_unconstrained)

    ## Compare results
#    @test sum(abs.(grad_mod_fd - grad1.∇ℓθᵤ)) ≈ 0 atol = _TOL
#    @test sum(abs.(hess_mod_fd - grad1.Δℓθᵤ)) ≈ 0 atol = _TOL

    @test sum(abs.(grad_mod_fd - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL
#    @test sum(abs.(hess_mod_fd - hess2.Δℓθᵤ)) ≈ 0 atol = _TOL

end
=#
############################################################################################
# Differentiation - Lower dimensions
modelLowerDim = ModelWrapper(LowerDims(), _val_lowerdims)
objectiveLowerDim = Objective(modelLowerDim, nothing)
function (objective::Objective{<:ModelWrapper{LowerDims}})(θ::NamedTuple)
    return 0.0
end

@testset "AutoDiffContainer - Log Objective AutoDiff compatibility - Lower dimensions" begin
    ## Assign DiffTune
    autodiff_fd = AutomaticDiffTune(objectiveLowerDim, :ForwardDiff)
    autodiff_rd = AutomaticDiffTune(objectiveLowerDim, :ReverseDiff)
    autodiff_zyg = AutomaticDiffTune(objectiveLowerDim, :Zygote)
    fwd = DiffObjective(objectiveLowerDim, autodiff_fd)
    rd = DiffObjective(objectiveLowerDim, autodiff_rd)
    zyg = DiffObjective(objectiveLowerDim, autodiff_zyg)
    theta_unconstrained = randn(_RNG, length(objectiveLowerDim))
    ## Compute Diffresult
    ld1 = log_density(fwd, theta_unconstrained)
    ld2 = log_density(rd, theta_unconstrained)
    ld3 = log_density(zyg, theta_unconstrained)
    _ld1 = BaytesDiff._log_density(objectiveLowerDim, autodiff_fd, theta_unconstrained)
    _ld2 = BaytesDiff._log_density(objectiveLowerDim, autodiff_rd, theta_unconstrained)
    _ld3 = BaytesDiff._log_density(objectiveLowerDim, autodiff_zyg, theta_unconstrained)
    grad1 = log_density_and_gradient(fwd, theta_unconstrained)
    grad2 = log_density_and_gradient(rd, theta_unconstrained)
    grad3 = log_density_and_gradient(zyg, theta_unconstrained)
    ## Compute manual call ~ Already checked for equality
    ld = objectiveLowerDim(theta_unconstrained)
    grad_mod_fd = ForwardDiff.gradient(objectiveLowerDim, theta_unconstrained)
    grad_mod_rd = ReverseDiff.gradient(objectiveLowerDim, theta_unconstrained)
    grad_mod_zy = Zygote.gradient(objectiveLowerDim, theta_unconstrained)[1]
    ## Compare results
    @test ld - ld1.ℓθᵤ ≈ 0 atol = _TOL
    @test ld - ld2.ℓθᵤ ≈ 0 atol = _TOL
    @test ld - ld3.ℓθᵤ ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad1.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_rd - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_zy - grad3.∇ℓθᵤ)) ≈ 0 atol = _TOL
    ## Checks
    _output = check_gradients(_RNG, objectiveLowerDim, [:ForwardDiff, :ReverseDiff, :Zygote]; printoutput = false)
    @test sum(abs.(_output.ℓobjective_gradient_diff)) ≈ 0 atol = _TOL
    ## Results
    ℓDensityResult(objectiveLowerDim, theta_unconstrained)
    ℓDensityResult(objectiveLowerDim)
    ℓGradientResult(grad1.θᵤ , grad1.ℓθᵤ , grad1.∇ℓθᵤ)
end



############################################################################################
# Differentiation - Float32
modelExample2 = ModelWrapper(ExampleModel(), _val_examplemodel, (;), FlattenDefault(; output = Float32))
objectiveExample2 = Objective(modelExample2, (data1, data2, data3, _idx))
objectiveExample2(objectiveExample2.model.val)

fwd = DiffObjective(objectiveExample2, AutomaticDiffTune(objectiveExample2, :ForwardDiff, ))
rd = DiffObjective(objectiveExample2, AutomaticDiffTune(objectiveExample2, :ReverseDiff))
rd2 = DiffObjective(objectiveExample2, AutomaticDiffTune(objectiveExample2, :ReverseDiffUntaped))
zyg = DiffObjective(objectiveExample2, AutomaticDiffTune(objectiveExample2, :Zygote))

@testset "AutoDiffContainer - Float32 compatibility" begin
    T = Float32
    theta_unconstrained = randn(T, length(objectiveExample2))
    ## Compute Diffresult
    ld1 = log_density(fwd, theta_unconstrained)
    ld2 = log_density(rd, theta_unconstrained)
    ld22 = log_density(rd2, theta_unconstrained)
    ld3 = log_density(zyg, theta_unconstrained)
    grad1 = log_density_and_gradient(fwd, theta_unconstrained)
    grad2 = log_density_and_gradient(rd, theta_unconstrained)
    grad22 = log_density_and_gradient(rd2, theta_unconstrained)
    grad3 = log_density_and_gradient(zyg, theta_unconstrained)
    ## Compare types
    @test ld1.ℓθᵤ isa T && eltype(ld1.θᵤ) == T
    @test ld2.ℓθᵤ isa T && eltype(ld2.θᵤ) == T
    @test ld22.ℓθᵤ isa T && eltype(ld22.θᵤ) == T
    @test ld3.ℓθᵤ isa T && eltype(ld3.θᵤ) == T

    @test grad1.ℓθᵤ isa T && eltype(grad1.θᵤ) == eltype(grad1.∇ℓθᵤ) == T
    @test grad2.ℓθᵤ isa T && eltype(grad2.θᵤ) == eltype(grad2.∇ℓθᵤ) == T
    @test grad22.ℓθᵤ isa T && eltype(grad22.θᵤ) == eltype(grad22.∇ℓθᵤ) == T
    @test grad3.ℓθᵤ isa T && eltype(grad3.θᵤ) == eltype(grad3.∇ℓθᵤ) == T
end

############################################################################################
#Tune Analytic
function fun1(objective::Objective{<:ModelWrapper{M}}, θᵤ::AbstractVector{T}) where {M<:ExampleModel, T<:Real}
    return zeros(size(θᵤ))
end
function fun2(objective::Objective{<:ModelWrapper{M}}, θᵤ::AbstractVector{T}) where {M<:ExampleModel, T<:Real}
    return zeros(size(θᵤ, 1), size(θᵤ, 1))
end
θᵤ = randn(_RNG, length(objectiveExample))
fun1(objectiveExample, θᵤ)
@testset "AnalyticDiffTune - " begin
    AnalyticalDiffTune(fun1, nothing)
    tune_analytic = AnalyticalDiffTune(fun1, fun2)
    ModelWrappers.update(tune_analytic, objectiveExample)
    _ld = BaytesDiff._log_density(objectiveExample, tune_analytic, θᵤ)
    _ldg =BaytesDiff._log_density_and_gradient(objectiveExample, tune_analytic, θᵤ)
    _ldh =BaytesDiff._log_density_and_gradient_and_hessian(objectiveExample, tune_analytic, θᵤ)

    @test _ld == _ldg[1]
    _ldgresult = log_density_and_gradient(objectiveExample, tune_analytic, θᵤ)
    @test _ld == _ldgresult.ℓθᵤ
    @test all(_ldgresult.θᵤ .== θᵤ)
end
############################################################################################
#Check result type conversion
objectives = [
    Objective(ModelWrapper(obectiveEBM.model.id, _paramEnzyme, _args, FlattenDefault()), dat),
    Objective(ModelWrapper(obectiveEBM.model.id, _paramEnzyme, _args, FlattenDefault(; output = Float32)), dat)
]
backends = [:ForwardDiff, :ReverseDiff, :ReverseDiffUntaped, :Zygote]#, :EnzymeReverse]#, :EnzymeForward, :EnzymeReverse]

@testset "AbstractDifferentiation - correct Type conversion" begin
    ## Gradient backends
    for backend in backends
        for _objective in objectives
            #Check type
            valtype = typeof(_objective.temperature)
            θᵤ = randn(valtype, length(_objective))

            #Create tune
            difftune0 = AutomaticDiffTune(_objective, backend, DiffOrderZero())
            difftune1 = AutomaticDiffTune(_objective, backend, DiffOrderOne())
#            difftune2 = AutomaticDiffTune(_objective, backend, DiffOrderTwo())

            diffobjective0 = DiffObjective(_objective, difftune0)
            diffobjective1 = DiffObjective(_objective, difftune1)
#            diffobjective2 = DiffObjective(_objective, difftune2)

            # create different log results
            result0 = BaytesDiff.log_density(diffobjective0, θᵤ)
            result1 = BaytesDiff.log_density_and_gradient(diffobjective1, θᵤ)
#            result2 = BaytesDiff.log_density_and_gradient_and_hessian(diffobjective2, θᵤ)

            @test typeof(result0.ℓθᵤ) == valtype && eltype(result0.θᵤ) == valtype
            @test typeof(result1.ℓθᵤ) == valtype && eltype(result1.θᵤ) == valtype && eltype(result1.∇ℓθᵤ) == valtype
#            @test typeof(result2.ℓθᵤ) == valtype && eltype(result2.θᵤ) == valtype && eltype(result2.∇ℓθᵤ) == valtype  && eltype(result2.Δℓθᵤ) == valtype
        end
    end
end
