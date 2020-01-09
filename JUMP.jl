using JuMP, Pkg, Clp, Cbc, GLPK, Queryverse, RCall, ParameterJuMP

model = ModelWithParams()


@variable(model, x)

@objective(model, Min, x)

#################
optimizer = Juniper.Optimizer
params = Dict{Symbol,Any}()
params[:nl_solver] = with_optimizer(Ipopt.Optimizer, print_level=0)

using LinearAlgebra # for the dot product
m = Model(with_optimizer(optimizer, params))
####

#a = add_parameter(m,2)
a = 5

@variable(m, -2 <= x <= 2, Int)

@variable(m, -1 <= z <= 1, Int)

@constraint(m, constraint1, x*z <= 0)

@objective(m, Max, a*z)

JuMP.optimize!(m)
println("Objective value: ", JuMP.objective_value(m))
println("x = ", JuMP.value(x))
println("z = ", JuMP.value(z))


###############
df = DataFrame(load("breastcancer_processed.csv"))

using RCall

R"""pacman::p_load(tidyverse, magrittr)
    dfr <- read_csv("breastcancer_processed.csv")
    dfr %<>% mutate(intercept = rep(1, nrow(dfr)),
                        Benign = ifelse(Benign == 1,1,-1))
    y = dfr %>% pull(Benign)
    x = dfr %>% select(-Benign) %>% select(intercept, everything())

    y_names = "Benign"
    x_names = names(x)
    """

dfr = rcopy(R"dfr")
y_names = rcopy(R"y_names")
x_names = rcopy(R"x_names")
x = rcopy(R"x")
y = rcopy(R"y")

lm = rcopy(R"""lm(Benign ~ ClumpThickness, dfr)""")

sum((y-λx)^2)

Λ = lm[:coefficients][2]
I = lm[:coefficients][1]

sum(broadcast(^, (I .+ Λ.*x.ClumpThickness) .- y,2))

sum(broadcast(^, lm[:residuals], 2))



###########################################

lmj = Model(with_optimizer(optimizer, params))

@variable(lmj, λ)
@variable(lmj, I)

@variable(lmj, X[1:683])
@constraint(lmj, X .== x.ClumpThickness)

@variable(lmj, Y[1:683])
@constraint(lmj, Y .== y)

@objective(lmj, Min,
    sum((I + λ*x.ClumpThickness[i] - y[i])^2 for i in 1:683) + 10*λ)

objective_function(lmj)

JuMP.optimize!(lmj)
println("Objective value: ", JuMP.objective_value(lmj))
println("I = ", JuMP.value(I))
println("lamb = ", JuMP.value(λ))


#####################

lmj = Model(with_optimizer(Clp.Optimizer))

@variable(lmj, λ)
@variable(lmj, I)
@variable(lmj, t)
@constraint(lmj, t ==  I^2)

#@variable(lmj, X[1:683])
#@constraint(lmj, X .== x.ClumpThickness)

#@variable(lmj, Y[1:683])
#@constraint(lmj, Y .== y)

@objective(lmj, Min, t)

JuMP.optimize!(lmj)
println("Objective value: ", JuMP.objective_value(lmj))
println("I = ", JuMP.value(I))
println("lamb = ", JuMP.value(λ))

Xt = 3,6,10
Yt = 6,12,20



#####################################
yt = -1
xt = 1


lmj = Model(with_optimizer(GLPK.Optimizer))

@variable(lmj, -1 <= λ <= 1, Int)
@variable(lmj, yt == -1)
#@variable(lmj, I)

@variable(lmj, z)
#@constraint(lmj, z == sign2(-1,λ))

#@variable(lmj, X[1:683])
#@constraint(lmj, X .== x.ClumpThickness)

#@variable(lmj, Y[1:683])
#@constraint(lmj, Y .== y)

@objective(lmj, Min, yt*λ)

JuMP.optimize!(lmj)
println("Objective value: ", JuMP.objective_value(lmj))
println("i = ", JuMP.value(i))
println("lamb = ", JuMP.value(λ))
