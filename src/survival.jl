function surv_marginal(time::Vector{Float64}, param::Param, X::Float64, s::Dierckx.Spline1D)
    exp.(-Spline.cumhaz(time, s).*exp(param.beta*X))
end

function density_marginal(time::Vector{Float64}, param::Param, X::Float64, s::Dierckx.Spline1D)
    Spline.hazard(time, s).*exp(param.beta*X).*exp.(-Spline.cumhaz(time, s).*exp(param.beta*X))
end

function oc_integrand(time::Float64, param::Param, X::NTuple{2,Float64}, s::Dierckx.Spline1D)
    haz0 = Spline.hazard([time], s)[1]
    chaz0 = Spline.cumhaz([time], s)[1]
    tmp = exp.(-chaz0*exp(param.beta*X[1]))*haz0*exp(param.beta*X[2])*exp(-chaz0*exp(param.beta*X[2]))
    if !isfinite(tmp)        
        display([time, haz0, chaz0, X[1], X[2], tmp, param.beta])
    end
    tmp
end

function test()
    data = generateData()
    x = fit(5000, data; warmup=2000)
    survival(x, Array(0.0:5.0:55.0))
end

function survival(x::Fit, grid::Array{Float64,1})
    n = length(x.beta)
    m = size(grid)[1]
    
    surv0 = Array{Float64,2}(undef, n, m)
    surv1 = Array{Float64,2}(undef, n, m)

    post_rmsd = Array{Float64,1}(n)
    post_oc = Array{Float64,1}(n)
        
    for i in 1:n
        param = slice(i, x)
        s = Spline.spline(param.outer, param.knots, exp.(param.theta))
        
        L = 100
        res = quadgk(t -> oc_integrand(t, param, (0.5, -0.5), s), param.outer[1], L)
        
        s0 = surv_marginal(grid, param, -0.5, s)
        s1 = surv_marginal(grid, param, 0.5, s)

        ## calculate non-PH effect measures here
        post_rmsd[i] = 0 ##diffrms(grid, s0, s1)
        post_oc[i] = res[1]/(1 - res[1]) ##oc(grid, s0, s1)
        ## oc = P(T1 > T0) / P(T0 > T1)
        
        surv0[i,:] = s0
        surv1[i,:] = s1
    end
    
    Survival(grid,
             mean(surv0, 1)[1,:], mean(surv1, 1)[1,:],
             var(surv0, 1)[1,:], var(surv1, 1)[1,:],
             post_rmsd, post_oc,
             (mean(post_rmsd), var(post_rmsd)),
             (mean(post_oc), var(post_oc)))
end
