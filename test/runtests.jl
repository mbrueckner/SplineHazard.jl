#!/usr/bin/env julia

using SplineHazard
using Test

t = Tuner(0)
@test SplineHazard.get_tune_param(t) == 2.38
@test (t.cur_blk == 1) & (t.cur_iter == 1)
update!(t, true)
@test (t.cur_blk == 1) & (t.cur_iter == 2)

t = Tuner(1000)
@test SplineHazard.get_tune_param(t) == 2.38
