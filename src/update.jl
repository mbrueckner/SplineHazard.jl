
    
function update_history!(sph::Sampler, t::Int, theta, knots)
    sph.N[t+1] = length(knots)
    sph.knots[t+1,1:sph.N[t+1]] = knots
    sph.theta[t+1,1:(sph.N[t+1]+sph.prior.Q)] = theta
end

"""
    update!(s::sampler, par)

Perform one iteration of the reversibel-jump MCMC. First all the spline weights are updated and then
a birth (add new knot), death (remove existing knot) or move (move existing knot) step is attempted.
The remaining parameters `par` of the model are just passed on to the likelihood and must be updated
outside of this function.
"""
function update!(sph::Sampler, par)
    t = sph.t
    Nt = sph.N[t]
    Kt = Nt + sph.prior.Q

    ## update theta
    theta_ac = zeros(Bool, Kt)
    sph.theta[t+1,:] = sph.theta[t,:]

    tune_param = get_tune_param(sph.tuner)
    loglik_init = NaN
    
    for j in 1:Kt            
        sph.theta[t+1,j], theta_ac[j], loglik_init = sample_theta(sph, t, j, par, tune_param, loglik_init)
    end
    thac = any(theta_ac) ## 1 if any component of theta has changed, 0 otherwise
    sph.accept[4] += thac
    
    update!(sph.tuner, t, thac)
        
    ## update theta variance
    sph.attempt[5] += 1
    mu = sum(sph.theta[t+1,1:Kt].^2) ##transpose(theta[t+1,:])*P*theta[t+1,:]   
    sph.v[t+1] = rand(InverseGamma(shape(sph.prior.v) + Kt/2, scale(sph.prior.v) + mu/2), 1)[1]
    sph.accept[5] += 1    
    
    if sph.fixed_knots
        update_history!(sph, t, sph.theta[t+1,1:Kt], sph.knots[t,1:Nt])
    else 
        ## prior probabilities of attempting birth, death and move of a knot given current number of knots
        pb = pbirth(Nt, sph.prior.N_max, sph.prior.N)
        pd = pdeath(Nt, sph.prior.N)
        pm = 1 - pb - pd            
        
        u = rand(1)[1]
        
        if (u <= pb) & (Nt < sph.prior.N_max)  ## attempt birth step
            sph.attempt[1] += 1
            sph.accept[1] += add_knot!(sph, t, par)
        elseif (u > pb) & (u <= pb+pd) & (Nt > 0) ## attempt death step
            sph.attempt[2] += 1
            sph.accept[2] += remove_knot!(sph, t, par)            
        elseif (u > pb+pd) & (Nt > 0) ## attempt move step
            sph.attempt[3] += 1            
            sph.accept[3] += move_knot!(sph, t, par)                                    
        end
    end

    sph.t += 1
end

"""
    sample_theta(s::Sampler, t::Int, j::Int, par, gamma::Float64, loglik_init=NaN)

Draw a single sample from the full conditional of `theta_j` given all other parameters `par`.
The proposal is generated from a normal distribution with mean equal to `theta[t,j]` and standard
error equal to `gamma`. 
"""
function sample_theta(s::Sampler, t::Int, j::Int, par, gamma::Float64, loglik_init=NaN)
    theta = s.theta[t+1, 1:(s.N[t]+s.prior.Q)]
    knots = s.prior.cand_knots[s.knots[t, 1:s.N[t]]]
    outer = s.prior.outer
    v = s.v[t]
    
    m = length(theta)
    d = Normal(0, gamma)
    tmp = copy(theta)
    
    function log_target(x::Float64)
        tmp[j] = x
        sp = spline(outer, knots, exp.(tmp))
        s.loglik(sp, par) - s.prior.theta_penalty(tmp, v)
    end

    if isnan(loglik_init)
        loglik_init = log_target(theta[j])
    end

    prop = rand(d, 1)[1] + theta[j]
    loglik_prop = log_target(prop)
    
    if log(rand(1)[1]) < (loglik_prop - loglik_init)
        return prop, true, loglik_prop
    else
        return theta[j], false, loglik_init
    end    
end

function pbirth(N::Int, N_max::Int, prior_N::DiscreteUnivariateDistribution)
    if N < N_max
        0.4*min(1, pdf(prior_N, N+1)/pdf(prior_N, N))
    else
        0.0
    end
end

function pdeath(N::Int, prior_N::DiscreteUnivariateDistribution)
    if N > 1
        0.4*min(1, pdf(prior_N, N-1)/pdf(prior_N, N))
    else
        0.0
    end
end

function find_interval(knots::Vector{Float64}, x::Float64)
    n::Int = 0
    for i in 1:length(knots)
        if x < knots[i]
            break
        end
        n += 1
    end
    return n
end

"""
     add_knot(sph::Sampler, t::Int, par)

Attempt to add knot randomly selected from all unoccupied candidate knots to the spline.
The spline weights stored in `s.theta[t+1,:]` are adjusted when move is accepted.
The locations of all other knots are not changed.
"""
function add_knot!(sph::Sampler, t::Int, par)
    cand_knots = sph.prior.cand_knots
    outer = sph.prior.outer
    Q = sph.prior.Q

    ## randomly select unoccupied candidate knot
    cindex = sample((1:length(cand_knots))[.!sph.knot_ocp])
    
    knots = sph.knots[t,1:sph.N[t]]
    theta = sph.theta[t+1,1:(sph.N[t]+sph.prior.Q)]
    v = sph.v[t+1]

    ckci = cand_knots[cindex]
    ckk = cand_knots[knots]
    
    ## find index of left endpoint of interval containing candidate knot
    n = find_interval(ckk, ckci)
    j = n+Q

    ## new knots
    new_xi = copy(knots)
    insert!(new_xi, n+1, cindex)
       
    ## new control points
    new_theta = copy(theta)
    insert!(new_theta, n+2, 0.0) 
       
    pk = Spline.pad_knots(ckk, outer, Q-1)

    ## @assert Q > 2
    a = (j-Q+2):(j-1) ##(n+2):(n+Q-1)
    r = (ckci .- pk[a]) ./ (pk[a.+(Q-1)] - pk[a])
    new_theta[a] = log.(r.*exp.(theta[a]) + (1 .- r).*exp.(theta[a.-1]))
    
    u = rand(1)[1]
    ##u = (cand_knots[cindex] - pk[j]) ./ (pk[j+(Q-1)] - pk[j])
    new_theta[j] = log(u*exp(theta[j]) + (1-u)*exp(theta[j-1]))
    
    ## acceptance probability
    logRPT = -log(2*pi*v)/2 + (sum(theta.^2) - sum(new_theta.^2))/(2*v)

    sp = spline(outer, ckk, exp.(theta))
    new_sp = spline(outer, cand_knots[new_xi], exp.(new_theta))
    logRL = sph.loglik(new_sp, par) - sph.loglik(sp, par)
    
    ## Jacobian
    J = (exp(theta[j]) - exp(theta[j-1])) / exp(new_theta[j])
    for i in (j-Q+2):(j-1)
        r = (ckci - pk[i]) / (pk[i+Q-1] - pk[i])
        J *= r*exp(theta[i] - new_theta[i])
    end
    
    if log(rand(1)[1]) < (logRPT + logRL + log(abs(J)))
        update_history!(sph, t, new_theta, new_xi)
        sph.knot_ocp[cindex] = true
        true
    else
        update_history!(sph, t, theta, knots)
        false
    end
end


"""
    remove_knot(sph::Sampler, t::Int, n::Int, par)

Attempt to remove knot at location `n` from the spline.
The spline weights stored in `s.theta[t+1,:]` are adjusted when move is accepted.
The locations of all other knots are not changed.
"""
function remove_knot!(sph::Sampler, t::Int, par)
    Q = sph.prior.Q
    outer = sph.prior.outer
    cand_knots = sph.prior.cand_knots
           
    v = sph.v[t+1]
    theta = sph.theta[t+1, 1:(sph.N[t]+sph.prior.Q)]
    knots = sph.knots[t, 1:sph.N[t]]

    n = sample(1:sph.N[t])
    cindex = knots[n]
    
    ## new knots
    new_xi = copy(knots)
    deleteat!(new_xi, n)

    j = n + Q
        
    ## new control points
    new_theta = copy(theta)
    deleteat!(new_theta, j-1)
        
    pk = Spline.pad_knots(cand_knots[new_xi], outer, Q-1)

    ## @assert Q > 2
    for i in (j-Q+1):(j-2)
        r = (cand_knots[cindex] .- pk[i]) / (pk[i+Q-1] - pk[i])
        nt_prop = exp(new_theta[i])/r .- exp(new_theta[i-1])*(1 - r)/r
        
        if nt_prop > 0
            new_theta[i] = log(nt_prop)
        else
            new_theta[i] = -100
        end
    end

    ## Jacobian
    J = exp(new_theta[j-1])/(exp(theta[j-1]) - exp(theta[j-2]))
    for i in (j - Q + 2):(j-2)
        r = (cand_knots[cindex] - pk[i]) / (pk[i+Q-1] - pk[i])
        J *= exp(new_theta[i]) / (r*exp(theta[i]))
    end
    
    ## acceptance probability
    logRPT = log(2*pi*v)/2 + (sum(theta.^2) - sum(new_theta.^2))/(2*v)

    sp = spline(outer, cand_knots[knots], exp.(theta))
    new_sp = spline(outer, cand_knots[new_xi], exp.(new_theta))
    logRL = sph.loglik(new_sp, par) - sph.loglik(sp, par)
        
    if log(rand(1)[1]) < (logRPT + logRL + log(abs(J)))
        update_history!(sph, t, new_theta, new_xi)
        sph.knot_ocp[cindex] = false
        true
    else
        update_history!(sph, t, theta, knots)
        false
    end
end

"""
    move_knot(sph::Sampler, t::Int, par)

Attempt to move randomly selected knot to a new location randomly chosen from all
unoccupied neighbouring knots (all knots between the selected knot and the next occupied knots to the left and right).
The ordering of the occupied knots is not changed by the move.
The spline weights stored in `s.theta[t+1,:]` are adjusted when move is accepted.
"""
function move_knot!(sph::Sampler, t::Int, par)    
    cand_knots = sph.prior.cand_knots
    outer = sph.prior.outer

    Nt = sph.N[t]
    Kt = Nt + sph.prior.Q
    
    sph.N[t+1] = Nt
    
    knots = sph.knots[t,1:Nt]
    theta = sph.theta[t+1,1:Kt]

    n = sample(1:Nt)
    cindex = knots[n]
    sph.knot_ocp[cindex] = false
    sph.knots[t+1,1:Nt] = sph.knots[t,1:Nt]

    ## get range of unoccupied neighbouring knots (including the currently selected knot to be moved)
    if (n == 1) & (n < Nt)
        nbg_knots = 1:(knots[n+1]-1)
    elseif (n == length(knots)) & (n > 1)
        nbg_knots = (knots[n-1]+1):length(cand_knots)
    elseif (n > 1) & (n < Nt)
        nbg_knots = (knots[n-1]+1):(knots[n+1]-1)
    else
        nbg_knots = 1:length(cand_knots)
    end
        
    ## check if all neighbouring candidate knots already occupied
    if length(nbg_knots) <= 1
        return false
    end
    
    ## uniformly sample new position from all candidate knots between bounds
    new_cindex = sample(nbg_knots)

    if new_cindex == cindex ## new position same as old
        sph.knot_ocp[cindex] = true
        return false
    else
        tmp = cand_knots[knots]
        tmp[n] = cand_knots[new_cindex]
        
        sp = spline(outer, cand_knots[knots], exp.(theta))
        new_sp = spline(outer, tmp, exp.(theta))
        logRL = sph.loglik(new_sp, par) - sph.loglik(sp, par)

        if log(rand(1)[1]) < logRL
            ## after the moved knot is still at position `n` among all occupied knots
            sph.knots[t+1,n] = new_cindex
            sph.knot_ocp[new_cindex] = true
            true
        else
            false
        end
    end
end
