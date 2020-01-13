using RCall, Optim, LinearAlgebra, ForwardDiff, Random, NLSolversBase,
      Distributions, BenchmarkTools

# get Data from R
R"""
pacman::p_load(lavaan)

data(HolzingerSwineford1939)
dat <- HolzingerSwineford1939[7:9]
"""

dat = rcopy(R"dat")

#dat = convert(Matrix, dat)

# define model
function ram(x)
      S =   [x[1] 0 0 0
            0 x[2] 0 0
            0 0 x[3] 0
            0 0 0 x[4]]

      F =  [1 0 0 0
            0 1 0 0
            0 0 1 0]

      A =  [0 0 0 1
            0 0 0 x[5]
            0 0 0 x[6]
            0 0 0 0]

      return (S, F, A)
end


# wrapper to call the optimizer
function optim_sem(model, obs_cov, start, est = ML, optim = "LBFGS")
      if optim == "LBFGS"
            objective = parameters -> est(parameters, model, obs_cov)
            result = optimize(objective, start, LBFGS(), autodiff = :forward)
      elseif optim == "Newton"
            objective = TwiceDifferentiable(
                  parameters -> est(parameters, model, obs_cov),
                  start,
                  autodiff = :forward)
            result = optimize(objective, start)
      else
            error("Unknown Optimizer")
      end
      #result = optimize(objective, start, LBFGS(), autodiff = :forward)
      return result
end


# helper functions

function logl(obs_means, exp_cov, data_matr)
      exp_cov = Matrix(Hermitian(exp_cov))
      likelihood::Float64 = -loglikelihood(MvNormal(obs_means, exp_cov), transpose(data_matr))
      return likelihood
end

function ML(parameters, model, obs_cov)
      matrices = model(parameters)
      Cov_Exp =  matrices[2]*inv(I-matrices[3])*matrices[1]*transpose(inv(I-matrices[3]))*transpose(matrices[2])
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) - log(det(obs_cov)) - 3
      return F_ML
end

function expected_cov(model, parameters)
      matrices = model(parameters)
      exp_cov =  matrices[2]*inv(I-matrices[3])*
      matrices[1]*transpose(inv(I-matrices[3]))*transpose(matrices[2])
      return exp_cov
end


# function to fit SEM and obtain a bigger fitted object
function fit_sem(model, data, start, est = ML, optim = "LBFGS")
      data_matr = convert(Array{Float64}, data)
      obs_cov = cov(data_matr)
      obs_means::Vector{Float64} = vec(mean(data_matr, dims = 1))

      # fit model
      result = optim_sem(model, obs_cov, start, est, optim)
      # obtain variables to compute logl
      parameters = Optim.minimizer(result)
      exp_cov = expected_cov(model, parameters)

      fitted_model = Dict{Symbol, Any}(
      :parameters => parameters,
      :data => data_matr,
      :obs_cov => obs_cov,
      :exp_cov => expected_cov(model, parameters),
      :obs_means => obs_means,
      :model => model,
      :logl => logl(obs_means, exp_cov, data_matr),
      :opt_result => result,
      :optimizer => optim,
      :start => start
      )
      return fitted_model
end

# compute standard errors and p-values
function delta_method(fit)
      parameters = fit[:parameters]
      model = fit[:model]
      obs_means = fit[:obs_means]
      data = fit[:data]

      if fit[:optimizer] == "LBFGS"
            fun = param -> logl(obs_means,
                              expected_cov(model, param),
                              data)
            se =
            sqrt.(diag(inv(ForwardDiff.hessian(fun, parameters))))
      elseif fit[:optimizer] == "Newton"
            fun = TwiceDifferentiable(param -> logl(fit[:obs_means],
                                          expected_cov(fit[:model], param),
                                          fit[:data]),
                                          fit[:start],
                                          autodiff = :forward)
            se = sqrt.(diag(inv(hessian!(fun, parameters))))
      else
            error("Your Optimizer is not supported")
      end
      return se
end




function fit_in_tree(model, data, start, est = ML, optim = "LBFGS")
      data_matr = convert(Array{Float64}, data)
      obs_cov = cov(data_matr)
      obs_means = vec(mean(data_matr, dims = 1))

      # fit model
      result = optim_sem(model, obs_cov, start, est, optim)
      # obtain variables to compute logl
      parameters = Optim.minimizer(result)
      exp_cov = expected_cov(model, parameters)

      # compute logl
      likelihood = logl(obs_means, exp_cov, data_matr)
      return likelihood
end



@time fit_in_tree(ram, dat, x0, ML, "LBFGS")

@time fit_sem(ram, dat, x0, ML, "LBFGS")

fitted_lb = fit_sem(ram, dat, x0, ML, "LBFGS")

fitted_new = fit_sem(ram, dat, x0, ML, "Newton")

delta_method(fitted_lb)

delta_method(fitted_new)
