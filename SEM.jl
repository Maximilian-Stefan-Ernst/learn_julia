using RCall, Optim, LinearAlgebra, ForwardDiff

R"""
pacman::p_load(lavaan, tidyverse)

data(HolzingerSwineford1939)
dat <- HolzingerSwineford1939[7:9]

model <- 'visual  =~ x1 + x2 + x3'
fit <- cfa(model, dat)

par <- parameterTable(fit)

cov <- dat %>% cov()

"""

par_jl = rcopy(R"par")
cov_jl = rcopy(R"cov")

# define SEM in RAM notation
function f(x)
      S =   [x[1] 0 0 0
            0 x[2] 0 0
            0 0 x[3] 0
            0 0 0 x[4]]

      F =   [1 0 0 0
            0 1 0 0
            0 0 1 0]

      A =  [0 0 0 1
            0 0 0 x[5]
            0 0 0 x[6]
            0 0 0 0]

      Cov_Exp =  F*inv(I-A)*S*transpose(inv(I-A))*transpose(F)
      return log(det(Cov_Exp)) + tr(cov_jl*inv(Cov_Exp)) - log(det(cov_jl)) - 3
end


###############################################################
### works without Gradient

#initial values and bounds
lower = zeros(6)
#lower = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
upper = [Inf, Inf, Inf, Inf, Inf, Inf]
x0 = ones(6)

result = optimize(f, lower, upper, x0)

Optim.minimizer(result)

# compare results (not in the right order!!!!)
lavaan_est = par_jl.est[4:7]
append!(lavaan_est, par_jl.est[2:3])
[lavaan_est Optim.minimizer(result)]



###############################################################
### with Gradient

lower = zeros(6)
#lower = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
upper = [Inf, Inf, Inf, Inf, Inf, Inf]
x0 = append!([0.5, 0.5, 0.5, 0.5], ones(2))

func = TwiceDifferentiable(x -> f(x),
                           ones(6), autodiff=:forward)


result_dif = optimize(f, x0, LBFGS(), autodiff = :forward)

optimize(func, x0)

ForwardDiff.gradient(f, x0)

[lavaan_est Optim.minimizer(result_dif)]
