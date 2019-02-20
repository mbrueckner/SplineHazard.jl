using Distributions
using SplineHazard
using SplineHazard.Spline

function test_data(n=200, tmax=36)
    ## 25 patients per month
    max_entry = n / 25
    
    entry = rand(Uniform(0, max_entry), n)
    
    ## 12 months median survival time
    T = -log.(1 .-rand(Uniform(0, 1), n)) ./ (log(2)/12)

    ## approx: 10% admin cens, 10% drop-out
    C = min.(rand(Uniform(0, 3*36), n), tmax)
   
    time = min.(T, C)
    status = T .< C
    id = 1:n
    
    (id=id, entry=entry, time=time, status=status)
end

nonpar_loglik(s, data) = sum(log.(hazard(data.time[data.status], s))) - sum(cumhaz(data.time, s))

function nonpar(data, M=10000)
    s = setup_sampler(M, data.time[data.status], sp -> nonpar_loglik(sp, data))
    sample!(s, M, Param(4, zeros(Float64, 8), knots0, 1.0))
    summary(s, data.time)
end

function nonpar_example()
    data = test_data()
    nonpar(data, 10000)
end
