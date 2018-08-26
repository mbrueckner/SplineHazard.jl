module SplineHazard

using StatsBase, Statistics, Dierckx, Distributions, DataFrames, RecipesBase

using LinearAlgebra: cholesky!

import Plots.plot
import Base.summary

export Sampler, Prior, Param, Tuner

export update!,
    create_sampler,
    set_initial_state!,
    sample!,
    set_tuner!,
    extract_spline,
    extract_hazard,
    extract_cumhaz,
    extract_survival,
    summary,
    plot

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
