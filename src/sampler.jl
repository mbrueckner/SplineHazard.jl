struct Prior
    N::DiscreteUnivariateDistribution
    v::ContinuousUnivariateDistribution
    theta_penalty::Function
        
    ## known parameters
    Q         ::Int ## order of spline
    N_max     ::Int ## maximum number of knots
    cand_knots::Vector{Float64} ## candidate knot locations
    outer     ::NTuple{2,Float64} ## outer knot locations
end

Prior() = Prior(Poisson(4), InverseGamma(0.01, 0.01), (theta, v) -> sum(theta.^2)/(2*v), 4, 100, Vector(0.1:0.1:0.9), (0.0, 1.0))

mutable struct Sampler
    loglik::Function    
    prior ::Prior
    tuner ::Tuner
    
    ## state variables
    t      ::Int
    warmup ::Int
    attempt::Vector{Int}
    accept ::Vector{Int}
      
    ## posterior samples of model parameters
    theta::Array{Float64,2} ## log control points
    v    ::Vector{Float64} ## variance of prior for theta (smoothness penalty parameter)
    N    ::Vector{Int} ## number of internal knots
    knots::Array{Int,2} ## inner knot locations

    knot_ocp::Vector{Bool}  ## knot occupation indicators (same length as prior.cand_knots)

    ## when number and location of knots are fixed then only the weights change
    ## matrices of the basis spline functions evaluated at each event time and its integrals can pre-computed once
    ## and stored for substantial speed-up
    fixed_knots   ::Bool
    precalc_haz   ::Array{Float64,2}
    precalc_cumhaz::Array{Float64,2}
end

mutable struct Param
    N::Int
    theta::Vector{Float64}
    knots::Vector{Int}
    v::Float64
end

"""
    create_sampler(loglik::Function, prior::Prior=Prior())

Create `Sampler` object from log-likelihood `loglik` and prior object `prior`.
The number of iterations of the MCMC and the initial state must be specified either in a subsequent call to `set_initial_state!` or
in the call to `sample!`.
"""
function create_sampler(loglik::Function, prior::Prior=Prior())
    Sampler(loglik, prior, Tuner(0), 0, 0, Vector{Int}(undef, 0), Vector{Int}(undef,0),
            Array{Float64,2}(undef,0,0), Vector{Float64}(undef,0), Vector{Int}(undef,0), Array{Float64,2}(undef,0,0),
            Vector{Bool}(undef,0), false, Array{Float64,2}(undef,0,0), Array{Float64,2}(undef,0,0))
end

"""
    set_initial_state!(s::Sampler, M::Int, init::Param; warmup=Int(floor(M/2)), fixed_knots::Bool=false,
                                time::Vector{Float64}=Vector{Float64}(undef,0), status::Vector{Bool}=Vector{Bool}(undef,0))

Set the initial state of the Markov chain to `init` and allocate arrays holding the posterior samples of `M` total iterations.
When number and location of the knots are fixed (`fixed_knots == true`) then the vector of event times `time` and the censoring
indicators `status` 
"""
function set_initial_state!(s::Sampler, M::Int, init::Param; warmup=Int(floor(M/2)), fixed_knots::Bool=false,
                            time::Vector{Float64}=Vector{Float64}(undef,0), status::Vector{Bool}=Vector{Bool}(undef,0))
    ## number of inner knots
    s.N = zeros(Int, M+1)
    s.N[1] = length(init.knots)

    ## candidate knots
    ##s.cand_knots = s.prior.cand_knots ##quantile(data.time[data.status], Array(0.0:1/(N_max+1):1.0))[2:(end-1)]
    n_candk = length(s.prior.cand_knots)

    ## inner knots
    s.knots = Array{Int,2}(undef, M+1, s.prior.N_max)
    s.knots[1,1:s.N[1]] = init.knots

    ## knot occupation indicators
    s.knot_ocp = zeros(Int, n_candk)
    ##ck = sort(sample(1:n_candk, N[1])) ##sort(randperm(n_candk)[1:N[1]])
    s.knot_ocp[init.knots] .= true

    Q = s.prior.Q
    
    ## log basis spline weights
    s.theta = Array{Float64,2}(undef, M+1, s.prior.N_max + Q)
    @assert length(init.theta) == s.N[1] + Q
    s.theta[1,1:(s.N[1]+Q)] = init.theta
    
    if fixed_knots        
        ## we need the baseline hazard basis function at all failure times, and their
        ## integrals (cumulative hazard basis functions) at every event time
        s.precalc_haz, s.precalc_cumhaz = Spline.precalc_spline(time[status], time, init.knots, s.prior.outer, Q)
    end
      
    ## prior variance of log basis spline weights
    s.v = Vector{Float64}(undef,M+1)
    s.v[1] = init.v

    s.warmup = warmup
    s.attempt = zeros(Int, 5)
    s.accept = zeros(Int, 5)
    s.t = 1
end

function set_tuner!(s::Sampler, tuner::Tuner)
    s.tuner = tuner
end

"""
    sample!(s::Sampler, M::Int, init::Param; warmup=Int(floor(M/2)), tuner=Tuner(warmup))

Perform a total of `M` iterations of the RJ-MCMC starting from initial state `init`. By default half of the total number of iterations are warmup
iterations where the standard deviation of the proposal distribution for the spline weights is tuned.
"""
function sample!(s::Sampler, M::Int, init::Param; warmup=Int(floor(M/2)), tuner=Tuner(warmup))
    set_initial_state!(s, M, init; warmup=warmup)
    set_tuner!(s, tuner)
    
    for t in 1:M
        update!(s, [0.0])
    end
end

"""
    extract(s::Sampler, t::Int)

Return posterior sample of parameters at iteration `t`.
"""
function extract(s::Sampler, t::Int)
    Param(s.N[t], s.theta[t,1:(s.N[t]+s.prior.Q)], s.knots[t,1:s.N[t]], s.v[t])
end

"""
    extract_spline(s::Sampler, t::Int)

Returns the posterior spline object of type `Dierckx.Spline1D` of iteration `t`.
"""
function extract_spline(s::Sampler, t::Int)
    p = extract(s, t)
    Spline.spline(s.prior.outer, s.prior.cand_knots[p.knots], exp.(p.theta), s.prior.Q-1)
end

"""
    extract_hazard(s::Sampler, time::Vector{Float64})

Evaluate the posterior hazard function sample of iteration `t` at each element of `time`.
"""
function extract_hazard(s::Sampler, time::Vector{Float64}, t::Int)
    p = extract_spline(s, t)
    Spline.hazard(time, p)
end

"""
    extract_hazard(s::Sampler, time::Vector{Float64}; return_warmup=false)

Evaluate all posterior hazard function samples at each element of `time`. Ignores warmup iterations by default.
"""
function extract_hazard(s::Sampler, time::Vector{Float64}; return_warmup=false)
    start = 1 + s.warmup*(1 - return_warmup)
    res = Array{Float64,2}(undef, tmax, length(time))
    for t in start:s.t
        res[t,:] = extract_hazard(s, time, t)
    end
    res
end

"""
    extract_cumhaz(s::Sampler, time::Vector{Float64})

Evaluate the posterior cumulative hazard function sample of iteration `t` at each element of `time`.
"""
function extract_cumhaz(s::Sampler, time::Vector{Float64}, t::Int)
    p = extract_spline(s, t)
    Spline.cumhaz(time, p)
end

"""
    extract_cumhaz(s::Sampler, time::Vector{Float64}; return_warmup=false)

Evaluate all posterior cumulative hazard function samples at each element of `time`. Ignores warmup iterations by default.
"""
function extract_cumhaz(s::Sampler, time::Vector{Float64}; return_warmup=false)
    start = 1 + s.warmup*(1 - return_warmup)
    res = Array{Float64,2}(undef, tmax, length(time))
    for t in start:s.t
        res[t,:] = extract_cumhaz(s, time, t)
    end
    res
end

"""Evaluate posterior survival function sample of iteration `t` at each element of `time`."""
extract_survival(s::Sampler, time::Vector{Float64}, t::Int) = exp.(.-extract_cumhaz(s, time, t))

"""Evaluate all posterior survival function samples at each element of `time`. Ignores warmup iterations by default."""
extract_survival(s::Sampler, time::Vector{Float64}; return_warmup=false) = exp.(.-extract_cumhaz(s, time; return_warmup=return_warmup))

"""
    summary(s::Sampler, time::Vector{Float64})

Create `DataFrame` with mean posterior hazard, cumulative hazard and survival functions evaluated at each element of `time` and the corresponding
(pointwise) 95% credible intervals.
"""
function summary(s::Sampler, time::Vector{Float64})
    haz = extract_hazard(s, time)
    cumhaz = extract_cumhaz(s, time)
    surv = exp.(.-cumhaz)

    function msci(x)
        mean(x, dims=1)[1,:], sqrt.(var(x, dims=1))[1,:], [quantile(x[:,k], 0.025) for k in 1:size(x)[2]], [quantile(x[:,k], 0.975) for k in 1:size(x)[2]]
    end

    msci_haz = msci(haz)
    msci_cumhaz = msci(cumhaz)
    msci_surv = msci(surv)

    DataFrame(time=time,
              hazard=msci_haz[1], hazard_sd=msci_haz[2], hazard_lo=msci_haz[3], hazard_hi=msci_haz[4],
              cumhazard=msci_cumhaz[1], cumhazard_sd=msci_cumhaz[2], cumhazard_lo=msci_cumhaz[3], cumhazard_hi=msci_cumhaz[4],
              survival=msci_surv[1], survival_sd=msci_surv[2], survival_lo=msci_surv[3], survival_hi=msci_surv[4])
    
end

"""
    plot(s::Sampler, time::Vector{Float64}, var=:hazard)

Plot `var` together with (pointwise) 95% posterior credible intervals. `var` can be `:hazard`, `:cumhazard` or `:survival`.
"""

function plot(s::Sampler, time::Vector{Float64}, var=:hazard)
    df = summary(s, time)

    var_lo = Symbol(string(var, "_lo"))
    var_hi = Symbol(string(var, "_hi"))

    plot(df[:time], df[var])
    plot!(df[:time], df[var_lo], linestyle=:dash, linecolor=:red)
    plot!(df[:time], df[var_hi], linestyle=:dash, linecolor=:red)
end