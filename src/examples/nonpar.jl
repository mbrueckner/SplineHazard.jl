using Distributions: rand, quantile, sample, Poisson, InverseGamma, Uniform, Binomial

include("../SplineHazard.jl")

using Main.SplineHazard
using Main.SplineHazard.Spline

function test_data(n=200, beta=0.0, tmax=36)
    z = rand(Binomial(1, 0.5), n) .- 0.5

    ## 25 patients per month
    max_entry = n / 25
    
    entry = rand(Uniform(0, max_entry), n)
    U = -log.(1 .-rand(Uniform(0, 1), n))

    ## 12 months median survival time
    T = U.*exp.(-beta*z)./(log(2)/12)

    ## approx: 10% admin cens, 10% drop-out
    C = min.(rand(Uniform(0, 3*36), n), tmax)
   
    time = min.(T, C)
    status = T .< C
    id = 1:n
    
    (id=id, entry=entry, time=time, status=status, z=z)
end

function nonpar_loglik(s, beta, data)
    h = hazard(data.time[data.status], s)
    ch = cumhaz(data.time, s)
    sum(log.(h) .+ beta.*data.z[data.status]) - sum(ch.*exp.(beta.*data.z ))
end

function nonpar(data, M=10000)
    N_max = 100
    cand_knots = quantile(data.time[data.status], Array(0.0:1/(N_max+1):1.0))[2:(end-1)]
    
    prior = Prior(Poisson(4), InverseGamma(0.01, 0.01), (theta, v) -> sum(theta.^2)/(2*v),
                  4, N_max, cand_knots, (0.0, maximum(data.time[data.status])))
    
    s = create_sampler((sp, beta) -> nonpar_loglik(sp, beta, data), prior)

    knots0 = sort(sample(1:length(cand_knots), 4))
    
    sample!(s, 10000, Param(4, zeros(Float64, 8), knots0, 1.0))

    summary(s, data.time)
end

function nonpar_example()
    data = test_data()
    nonpar(data, 10000)
end
