module Spline

## spline construction and evaluation using "dierckx" FORTRAN library via "Dierckx" julia package
## http://www.netlib.org/dierckx/
## https://github.com/kbarbary/Dierckx.jl
using Dierckx
##using Gadfly

export spline, hazard, cumhaz, precalc_spline, plot_spline, pad_knots

## FIXME: consistent use of Q vs k

## construct spline of order k+1 on interval (outer[1], outer[2])
## with given internal knots and weights theta
## knots = vector of interior knots
## w = vector of weights (one for each basis spline)
function spline(outer::NTuple{2,Float64}, knots::Vector{Float64}, w::Vector{Float64}, k::Int=3)
    ## number of basis splines
    m = length(knots) + k + 1

    if m != length(w)
        error("number of coefficients must match number of basis splines")
        return NaN
    end

    ## if !issorted(knots)
    ##   knots = sort(knots)
    ## end
    
    ## total number of knots
    n = m + k + 1

    ## pad knots by repeating outer knot (k+1)-times on each side
    t = Vector{Float64}(undef, n) ##Vector{Float64}(undef,n)  # All knots
    t[(k+2):(end-k-1)] = knots
    t[1:(k+1)] .= outer[1]
    t[(end-k):end] .= outer[2]    

    Spline1D(t, w, k, 3, 0.0)
end

function pad_knots(knots::Vector{Float64}, outer::NTuple{2,Float64}, p::Int)
    pk = Vector{Float64}(undef, length(knots) + 2*(p+1))
    pk[(p+2):(end-p-1)] = knots
    pk[1:(p+1)] .= outer[1]
    pk[(end-p):end] .= outer[2]
    pk
end

hazard(x::Vector{Float64}, s::Dierckx.Spline1D) = evaluate(s, x)
    
function cumhaz(x::Vector{Float64}, s::Dierckx.Spline1D)
    res = Vector{Float64}(undef, length(x))
    for i in 1:length(x)
        res[i] = integrate(s, 0, x[i])
    end
    res
end

cumhaz(x::Float64, s::Dierckx.Spline1D) = integrate(s, 0, x)

function plot_spline(s::Dierckx.Spline1D)
    a = s.t[1]
    b = s.t[end] + 10.0

    x = Vector(a:((b-a)/1000):b)
    y = evaluate(s, x)

    plot(x=x, y=y, Geom.line)
end

## Evaluate all basis spline functions at x, and all integrals of spline functions at y
function precalc_spline(x::Vector{Float64}, y::Vector{Float64}, knots::Vector{Float64}, outer=NTuple{2,Float64}, Q::Int=4)
    K = length(knots) + Q
    A = Array{Float64,2}(undef, length(x), K)
    B = Array{Float64,2}(undef, length(y), K)
    
    for k in 1:K
        w = zeros(Float64, K)
        w[k] = 1.0    
        s = spline(outer, knots, w, Q-1)
        A[:,k] = hazard(x, s)
        B[:,k] = cumhaz(y, s)
    end

    A, B
end

end

function test_spline()
    outer = (0.0, 1.0)
    knots = [0.2, 0.4, 0.6, 0.8]

    x = rand(100)
    theta = rand(Normal(0, 1), 8)
    A = Array{Float64,2}(undef, 100, 8)
    
    for k in 1:8
        w = zeros(Float64, 8)
        w[k] = 1.0    
        s = spline(outer, knots, w, 3)
        A[:,k] = hazard(x, s)
    end

    a = A*exp.(theta)
    b = hazard(x, spline(outer, knots, exp.(theta), 3))

    display(hcat(a, b))
    isapprox(a, b)
end

function test_splineA(M=10, n=100)
    outer = (0.0, 1.0)
    knots = [0.2, 0.4, 0.6, 0.8]

    x = rand(n)
    h = zeros(Float64, n)
    ch = zeros(Float64, n)    
    A = Array{Float64,2}(undef, n, 8)
    B = Array{Float64,2}(undef, n, 8)
    
    for k in 1:8
        w = zeros(Float64, 8)
        w[k] = 1.0    
        s = spline(outer, knots, w, 3)
        A[:,k] = hazard(x, s)
        B[:,k] = cumhaz(x, s)
    end
        
    for m in 1:M
        theta = rand(Normal(0, 1), 8)
        et = exp.(theta)
        h += A*et
        ch += B*et
    end

    h ./ M, ch ./ M
end

function test_splineB(M=10, n=100)
    outer = (0.0, 1.0)
    knots = [0.2, 0.4, 0.6, 0.8]

    x = rand(n)
    h = zeros(Float64, n)
    ch = zeros(Float64, n)
    
    for m in 1:M
        theta = rand(Normal(0, 1), 8)
        s = spline(outer, knots, exp.(theta), 3)
        h += hazard(x, s)
        ch += cumhaz(x, s)
    end

    h ./ M, ch ./ M
end
