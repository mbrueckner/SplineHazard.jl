mutable struct Tuner
    n_blk_max::Int  ## maximum number of blocks
    blk_size ::Int  ## block size
    cur_blk  ::Int  ## current block
    cur_iter ::Int  ## number of iterations within current block
    ac_cnt   ::Int  ## number of acceptances within current block
    blk_acr  ::Vector{Float64} ## acceptance rates
    theta    ::Vector{Float64} ## tuning parameter (standard deviation)
end

```
       Tuner(warmup::Int)

Create Tuner object. Tuning aims to adjust the tuning parameters during the `warmup` iterations such that an acceptance rate of 25% is achieved.
It does this by fitting a linear regression to blockwise acceptance rates and the tuning parameters. No tuning is performed when there are less
than 100 warmup iterations.
```
function Tuner(warmup::Int)
    if warmup < 100
        ## no tuning if fewer than 100 warmup iterations
        Tuner(1, -1, 1, 0, 0, [0.0], [2.38])  
    else
        blk_size = Int(floor(warmup/25))
        n_blk_max = Int(floor(warmup/blk_size))
        theta = zeros(Float64, n_blk_max)
        theta[1:2] = [2.38, 0.1]
        
        Tuner(n_blk_max, blk_size, 1, 0, 0, zeros(Int, n_blk_max), theta)
    end
end

get_tune_param(x::Tuner) = x.theta[x.cur_blk]

```
    update!(x::Tuner, t::Int, ac::Bool)

Update the tuner `x` at iteration `t` with acceptance status `ac`. It simply counts the number of acceptances in the current block. If the end of the
current block is reached the acceptance rate is calculated and stored and a new block is started. After at least two blocks have been recorded the
tuning parameter is updated via linear regression for every new block.
```
function update!(x::Tuner, t::Int, ac::Bool)
    x.ac_cnt += ac
    
    b = x.blk_size
    cb = x.cur_blk
    
    if t < x.n_blk_max*b
        if x.cur_iter == b ## update tuning parameter every b iterations
            x.blk_acr[cb] = x.ac_cnt/b ## acceptance rate of last b iterations
            x.ac_cnt = 0 ## reset acceptance counter
            if cb >= 2 ## need at least two blocks for linear regression
                x.theta[cb+1] = linreg(x.blk_acr[1:cb], x.theta[1:cb])

                ## gamma_theta is standard deviation
                if x.theta[cb+1] <= 0
                    x.theta[cb+1] = x.theta[cb]
                end
            end
            x.cur_iter = 0
            x.cur_blk += 1
        else
            x.cur_iter += 1
        end
    end
end

function linreg(accept::Vector{Float64}, gamma::Vector{Float64})
    X = hcat(ones(Float64, length(accept)), accept)

    ## U = cholesky!(X'X).U
    ## w = U' \ X'Y
    ## beta = U \ X'w
    
    beta = cholesky!(X'X) \ X'gamma
    beta[1] + 0.25*beta[2]
end
