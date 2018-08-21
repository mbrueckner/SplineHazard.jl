SplineHazard.jl
===============

*Julia library for Bayesian non- and semi-parametric hazard models using splines*

This is a Julia library implementing the reversible jump MCMC for Bayesian B-spline hazard models
as described in [Bayesian adaptive B-spline estimation in proportional hazards frailty models](https://projecteuclid.org/euclid.ejs/1278439436) by Sharef et al.

It can be used either as standalone for non-parametric estimation of hazard, cumulative hazard and survival functions,
or as part of semi-parametric model, e.g. as baseline hazard in a proportional hazards model.

It uses the [Dierckx](https://github.com/kbarbary/Dierckx.jl) Julia package for B-spline evaluation.

### Features
- Univariate hazard models using 1-dimensional B-splines with number and location of knot positions
determined by the data via reversible jump MCMC
- Designed to be easily integrated into existing MCMC procedures with only a few lines of code without placing
restrictions on the rest of the model (only the likelihood function needs to be supplied by the user)
- Plot mean posterior hazard / cumulative hazard / survival functions with posterior credible intervals
- Estimate coverage probability of posterior credible intervals for given true hazard / cumulative hazard / survival functions

### TODO
- More example models (additive hazard model, ...)
- parametric hazard penalty

Install (Julia 0.7 and later)
-----------------------------

```julia
(v1.0) pkg> add SplineHazard
```
(Type `]` to enter package mode.) Requires [Dierckx](https://github.com/kbarbary/Dierckx.jl), [DataFrames](https://github.com/JuliaData/DataFrames.jl) and [Distributions](https://github.com/JuliaStats/Distributions.jl) Julia packages to be installed.

Example Usage (Semi-parametric proportional hazards model)
----------------------------------------------------------
This example shows how to implement a semi-parametric proportional hazards model with a cubic B-spline baseline hazard
and a single covariate. The full code is in [src/examples/ph.jl](src/examples/ph.jl).

Generate an example survival dataset
```julia
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
    
    DataFrame(id=id, entry=entry, time=time, status=status, z=z)
end

data = ph_data(500, log(0.67)) 
```
Load the package

```julia
using SplineHazard
```

Define the log-likelihood function. First argument is the B-spline (object of type `Dierckx.Spline1D`) defined by the current
number and location of the knots and the spline weights at the current iteration of the MCMC algorithm. Second argument
is the current value of the log-hazard ratio. Third argument is a `DataFrame` or `NamedTuple` with fields `time` (vector of
censored event times), `status` (boolean vector of censoring indicators) and `z` (vector of group indicators).
```julia
function loglik(s::Sampler, beta, data)
    h = hazard(data.time[data.status], s)
    ch = cumhaz(data.time, s)
    sum(log.(h) .+ beta.*data.z[data.status]) - sum(ch.*exp.(beta.*data.z))
end
```

Function to sample from the full conditional distribution of `beta` given all other parameters
```julia
function sample_beta(beta::Float64, s::Sampler, data, prior::UnivariateDistribution) 
    X = data.z
    time = data.time
    status = data.status

    ch = extract_cumhaz(s, time, s.t)
        
    function log_target(x::Float64)
        logpdf(prior, x) + sum(status.*x.*X - ch.*exp.(x.*X))
    end

    prop = rand(Normal(beta, 2.38), 1)[1]
    r = log_target(prop) - log_target(beta)
    
    if log(rand(1)[1]) < r
        return prop, true
    else
        return beta, false
    end
end
```

Place 100 candidate knots at quantiles of observed failure times
```julia
N_max = 100
cand_knots = quantile(data.time[data.status], Vector(0.0:1/(N_max+1):1.0))[2:(end-1)]
```

Use Poisson distribution as prior on number of knots, inverse gamma distribution for penalty parameter, use cubic splines,
and place the two outer knots at 0.0 and the largest observed failure time.
```julia
prior = SplineHazard.Prior(Poisson(4), InverseGamma(0.01, 0.01), (theta, v) -> sum(theta.^2)/(2*v),
                               4, N_max, cand_knots, (0.0, maximum(data.time[data.status])))
```

Create sampler using log-likelihood function and prior object
```julia
s = create_sampler((sp, beta) -> loglik(sp, beta, data), prior)
```

Randomly select 4 candidate knots as initial knots for the spline
```julia
knots0 = sort(sample(1:length(cand_knots), 4))
```

Set initial values for all parameters. All spline weights set to 0.0 (constant hazard function) and the penalty parameter set to 1.0
```julia
M = 10000
set_initial_state!(s, M, Param(4, zeros(Float64, 8), knots0, 1.0))
```

Tune penalty parameter during warmup iterations aiming for 25% acceptance rate for the spline weight parameters `theta`. This uses
the simple linear regression based tuning as proposed by Sharef et al.
```julia
set_tuner!(s, Tuner(Int(M/2))) 
```

Allocate vector to hold posterior samples of log-hazard ratio and set initial value to 0.0
```julia
beta = Vector{Float64}(undef, M+1)
beta[1] = 0.0
beta_ac = 0 ## count acceptances of beta proposals
```
Normal prior for log-hazard ratio
```julia
prior_beta = Normal(0, 10)
```

Gibbs sampler requires only call to `update!` function at each iteration to use the RJ-MCMC for the baseline hazard model
```julia
for t in 1:M
    beta[t+1], ac = sample_beta(beta[t], s, data, prior_beta)
    beta_ac += ac
        
    update!(s, beta[t+1])  ## beta[t+1] gets passed to the log-likelihood function without modification
end
```

Print summary information of posterior samples
```julia
print("Acceptance rate for beta: $(beta_ac/M)")
print("Posterior mean: $(mean(beta[(s.warmup+1):end]))")
print("Posterior variance: $(var(beta[(s.warmup+1):end]))")

df = summary(s)
head(df)
```

References
----------
[Sharef et al.: Bayesian adaptive B-spline estimation in proportional hazards frailty models](https://projecteuclid.org/euclid.ejs/1278439436)