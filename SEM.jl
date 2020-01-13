using RCall, Optim, LinearAlgebra, ForwardDiff, Random, NLSolversBase,
      Distributions

R"""
pacman::p_load(lavaan, tidyverse)

data(HolzingerSwineford1939)
dat <- HolzingerSwineford1939[7:9]

model <- 'visual  =~ x1 + x2 + x3'
fit <- cfa(model, dat)

par <- parameterTable(fit)


means_obs <- apply(dat, 2, function(x)mean(x, na.rm = T))
cov <- dat %>% cov()

"""

par_jl = rcopy(R"par")
cov_jl = rcopy(R"cov")
est_jl = rcopy(R"parameterEstimates(fit)")
mean_jl = rcopy(R"means_obs")
dat_jl = rcopy(R"dat")

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

lavaan_est = append!(par_jl.est[4:7], par_jl.est[2:3])
[lavaan_est Optim.minimizer(result)]



###############################################################
### with Gradient

lower = zeros(6)
#lower = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
upper = [Inf, Inf, Inf, Inf, Inf, Inf]
x0 = append!([0.5, 0.5, 0.5, 0.5], ones(2))

func = TwiceDifferentiable(x -> f(x),
                           append!([0.5, 0.5, 0.5, 0.5], ones(2)), autodiff=:forward)


result_dif = optimize(f, x0, LBFGS(), autodiff = :forward)

result_dif = optimize(func, x0)

lavaan_est = append!(par_jl.est[4:7], par_jl.est[2:3])
[lavaan_est Optim.minimizer(result_dif)]

parameters = Optim.minimizer(result_dif)

######################
### standard errors
se = sqrt.(diag(inv(hessian!(func, parameters))))

#se = diag(inv(ForwardDiff.hessian(f, parameters)))

#t_stats = parameters./sqrt.(se)
#pdf(TDist(299), t_stats)

lavaan_se = append!(par_jl.se[4:7], par_jl.se[2:3])

[lavaan_se se]



round.(lavaan_se; digits=2)

round.(se; digits = 2)

##########################################
### logl function

function exp_cov(x)
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

      return F*inv(I-A)*S*transpose(inv(I-A))*transpose(F)
end

function logl(x)
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
      return loglikelihood(MvNormal(mean_jl, Cov_Exp), transpose(dat_matr))
end

model_impl_cov = exp_cov(parameters)
dat_matr = convert(Matrix, dat_jl)

loglikelihood(MvNormal(mean_jl, model_impl_cov), transpose(dat_matr))



lower = zeros(6)
#lower = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
upper = [Inf, Inf, Inf, Inf, Inf, Inf]
x0 = append!([0.5, 0.5, 0.5, 0.5], ones(2))

func = TwiceDifferentiable(x -> logl(x),
                           append!([0.5, 0.5, 0.5, 0.5], ones(2)), autodiff=:forward)



# optimizing over logl does not work
result_dif = optimize(logl, x0, LBFGS(), autodiff = :forward)
result_dif = optimize(func, x0)

#parameters = Optim.minimizer(result_dif)

# first version throws an error
se = sqrt.(diag(inv(hessian!(func, parameters))))
se = sqrt.(2*diag(inv(ForwardDiff.hessian(f, parameters))))

# gives the right logl
logl(parameters)

lavaan_se = append!(par_jl.se[4:7], par_jl.se[2:3])
[lavaan_se se]
