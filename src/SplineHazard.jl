module SplineHazard

using StatsBase, Statistics, Dierckx, Distributions, DataFrames

using LinearAlgebra: rank, cholesky!, diagm

import Base.summary

export Sampler, Prior, Param, Tuner

export update!,
    create_sampler,
    setup_sampler,
    set_initial_state!,
    sample!,
    sample_prior,
    set_tuner!,
    extract_spline,
    extract_hazard,
    extract_cumhaz,
    extract_survival,
    summary

include("Spline.jl")

using .Spline

include("tuner.jl")
include("sampler.jl")
include("update.jl")

"""
A Julia package for fitting spline based non- and semi-parametric hazard models.
"""
SplineHazard

end
