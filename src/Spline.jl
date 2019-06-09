module Spline

## spline construction and evaluation using "dierckx" FORTRAN library via "Dierckx" julia package
## http://www.netlib.org/dierckx/
## https://github.com/kbarbary/Dierckx.jl
using Dierckx
using QuadGK
##using Gadfly

export spline, hazard, cumhaz, eval_basis, plot_spline, pad_knots, const_spline

## FIXME: consistent use of Q vs k

## construct spline of order k+1 on interval (outer[1], outer[2])
## with given internal knots and weights theta
## knots = vector of interior knots
## w = vector of weights (one for each basis spline)
function spline(outer::NTuple{2,Float64}, knots::Vector{Float64}, w::Vector{Float64}, k::Int=3)
    ## number of basis splines
    m = length(knots) + k + 1

    @assert (outer[1] >= 0.0) & (outer[2] >= 0.0)
    
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

"""
    const_spline(val::Float64, outer::NTuple{2,Float64}; Q::Int=4)

Create spline of order `Q` with outer knots at `outer` and all weights equal to `val`. Resulting spline
is a constant function with value `val` everywhere.
"""
function const_spline(val::Float64, outer::NTuple{2,Float64}; Q::Int=4)
    spline(outer, [outer[1] + (outer[2]-outer[1])/2], val .+ zeros(Float64, Q+1), Q-1)
end

get_inner_knots(s::Dierckx.Spline1D) = s.t[(s.k+2):(end-s.k-1)]
get_outer_knots(s::Dierckx.Spline1D) = (s.t[1], s.t[end])

function pad_knots(knots::Vector{Float64}, outer::NTuple{2,Float64}, p::Int)
    pk = Vector{Float64}(undef, length(knots) + 2*(p+1))
    pk[(p+2):(end-p-1)] = knots
    pk[1:(p+1)] .= outer[1]
    pk[(end-p):end] .= outer[2]
    pk
end

hazard(x, s::Dierckx.Spline1D) = evaluate(s, x)

## evaluate cumulative hazard from 0.0 to x
function cumhaz(x::Vector{T}, s::Dierckx.Spline1D) where T <: Real
    res = zeros(eltype(x), length(x))
    for i in eachindex(x)
        if (s.t[1] <= 0.0) & (x[i] <= s.t[end])
            res[i] = integrate(s, 0.0, x[i])
        else            
            res[i] = quadgk(t -> hazard(t,s), 0.0, x[i])[1]
        end
    end
    res
end

cumhaz(x::T, s::Dierckx.Spline1D) where T <: Real = cumhaz([x], s)[1]

function plot_spline(s::Dierckx.Spline1D)
    a = s.t[1]
    b = s.t[end] + 10.0

    x = Vector(a:((b-a)/1000):b)
    y = evaluate(s, x)

    plot(x=x, y=y, Geom.line)
end

"""
    eval_basis(s::Dierckx.Spline1D, x::Vector{Float64}, y::Vector{Float64})

Evaluate all basis functions of 1D spline `s` at points `x`, and all integrals of spline functions at `y`.
"""
function eval_basis(s::Dierckx.Spline1D, x::Vector{T}, y::Vector{T}=x) where T <: Real
    eval_basis(x, y, get_inner_knots(s), get_outer_knots(s), s.k+1)
end

function eval_basis(x::Vector{T}, y::Vector{T}, knots::Vector{T}, outer=NTuple{2,T}, Q::Int=4) where T <: Real
    K = length(knots) + Q
    A = Matrix{T}(undef, length(x), K)
    B = Matrix{T}(undef, length(y), K)
    
    for k in 1:K
        w = zeros(T, K)
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
