using Distributions
using SplineHazard
using SplineHazard.Spline

function ph_data(n=200, beta=0.0, tmax=36)
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

function ph_loglik(s, beta, data)    
    h = hazard(data.time[data.status], s)
    ch = cumhaz(data.time, s)

    sum(log.(h) .+ beta.*data.z[data.status]) - sum(ch.*exp.(beta.*data.z))
end

function sample_beta(beta::Float64, s::Sampler, data, prior::UnivariateDistribution) 
    X = data.z
    time = data.time
    status = data.status

    ch = extract_cumhaz(s, time, s.t)
        
    function log_target(x::Float64)
        logpdf(prior, x) + sum(status.*x.*X - ch.*exp.(x.*X))
    end

    prop = rand(Normal(beta, 1.0), 1)[1]
    r = log_target(prop) - log_target(beta)
    
    if log(rand(1)[1]) < r
        return prop, true
    else
        return beta, false
    end
end


function ph(data, M=10000)
    s = setup_sampler(M, data.time[data.status], (sp, beta) -> ph_loglik(sp, beta, data))
    
    beta = Vector{Float64}(undef, M+1)
    beta[1] = 0.0
    beta_ac = 0

    prior_beta = Normal(0, 10)
    
    for t in 1:M
        beta[t+1], ac = sample_beta(beta[t], s, data, prior_beta)
        beta_ac += ac
        
        update!(s, beta[t+1])
    end

    println("Acceptance rate for beta: $(beta_ac/M)")
    println("Posterior mean: $(mean(beta[(s.warmup+1):end]))")
    println("Posterior variance: $(var(beta[(s.warmup+1):end]))")

    ##summary(s, data.time)
    
    beta[(s.warmup+1):end], beta_ac, s
end

function ph_example()
    data = ph_data(200, log(0.67))
    ph(data, 10000)
end
