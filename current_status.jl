using RCall, Optim, LinearAlgebra, ForwardDiff, Random, NLSolversBase,
      Distributions

# fit SEM Model with lavaan
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
# parameterTable
par_jl = rcopy(R"par")
# observed covariance
cov_jl = rcopy(R"cov")
# parameterEstimates
est_jl = rcopy(R"parameterEstimates(fit)")
# observed sample means (needed for likelihood function)
mean_jl = rcopy(R"means_obs")
# data (needed for likelihood function)
dat_jl = rcopy(R"dat")

# define SEM in RAM notation - uses F-Value
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

# return model-implied covariance matrix for given parameters
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

# return log-likelihood of the model (used for standard errors)
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

      Cov_Exp =  Matrix(Hermitian(F*inv(I-A)*S*transpose(inv(I-A))*transpose(F)))
      logl = loglikelihood(MvNormal(mean_jl, Cov_Exp), transpose(dat_matr))
      return logl
end

# initial values and bounds
lower = zeros(6)
upper = [Inf, Inf, Inf, Inf, Inf, Inf]
x0 = append!([0.5, 0.5, 0.5, 0.5], ones(2))

# solution 1
result_dif_1 = optimize(f, x0, LBFGS(), autodiff = :forward)

# solution 2
func = TwiceDifferentiable(x -> f(x),
                           append!([0.5, 0.5, 0.5, 0.5],
                           ones(2)),
                           autodiff=:forward)

result_dif_2 = optimize(func, x0)

parameters_1 = Optim.minimizer(result_dif_1)
parameters_2 = Optim.minimizer(result_dif_2)

# compare parameter estimates of lavaan and our Algorithm
lavaan_est = append!(par_jl.est[4:7], par_jl.est[2:3])
[lavaan_est parameters_1 parameters_2]



#############################################################
### standard errors

# model implied covariance for our obtained parameters
model_impl_cov_2 = exp_cov(parameters_2)
model_impl_cov_1 = exp_cov(parameters_1)
# sample data
dat_matr = convert(Matrix, dat_jl)

# loglikelihood of our solution is (almost) the same as lavaan returns
# (in fact, our solution is a little bit better - our model implied
# covariance matrix is closer to the sample cov matrix)

round.(model_impl_cov_2, digits = 15) == round.(transpose(model_impl_cov_2), digits = 15)
round.(model_impl_cov_2, digits = 16) == round.(transpose(model_impl_cov_2), digits = 16)

model_impl_cov_2 = round.(model_impl_cov_2, digits = 15)

loglikelihood(MvNormal(mean_jl, model_impl_cov_1), transpose(dat_matr))
loglikelihood(MvNormal(mean_jl, model_impl_cov_2), transpose(dat_matr))

logl(parameters_1)
logl(parameters_2)

R"""fitMeasures(fit, "logl")"""

@rput model_impl_cov_2
@rput mean_jl

R"emdbook::dmvnorm(dat %>% as.matrix(),
        mean_jl,
        model_impl_cov_2,
        log = TRUE
) %>% sum()"

# optimize over logl instead of F-Value - gives an Error!
# result_dif = optimize(logl, x0, LBFGS(), autodiff = :forward)
#
# func = TwiceDifferentiable(x -> logl(x),
#                            append!([0.5, 0.5, 0.5, 0.5],
#                            ones(2)),
#                            autodiff=:forward)
#
# result_dif = optimize(func, x0)


# first version throws an error, second version works
se = sqrt.(2*diag(inv(hessian!(func, parameters_2))))

se = sqrt.(2*diag(inv(ForwardDiff.hessian(logl, parameters_2))))

# lavaan standard errors are almost the same
lavaan_se = append!(par_jl.se[4:7], par_jl.se[2:3])
[lavaan_se se]

# p values
z_stats = parameters./se
#pdf(TDist(300), z_stats)

lavaan_p = append!(est_jl.pvalue[4:7], est_jl.pvalue[2:3])
lavaan_z = append!(est_jl.z[4:7], est_jl.z[2:3])

# this way lavaan computes test statistics too
[lavaan_z lavaan_est./lavaan_se]

# compare z values - almost the same
[lavaan_z z_stats]

# this was lavaan computes standard errors
@rput lavaan_z
r_p = rcopy(R"2*pnorm(-abs(lavaan_z))")
[r_p lavaan_p]

# our way leads to (almost) the same solutions
p_values = pdf(Normal(), z_stats)

[lavaan_p p_values]

@rput z_stats
r_p_our_sol = rcopy(R"2*pnorm(-abs(z_stats))")

# why does R compute different p-values?
[r_p_our_sol p_values]
