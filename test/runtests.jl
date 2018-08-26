#!/usr/bin/env julia

using SplineHazard
using Test

t = Tuner(0)
@test SplineHazard.get_tune_param(t) == 2.38
@test (t.cur_blk == 1) & (t.cur_iter == 0)
update!(t, 1, true)
@test (t.cur_blk == 1) & (t.cur_iter == 0)

t = Tuner(1000)
@test SplineHazard.get_tune_param(t) == 2.38
