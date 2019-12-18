using JuMP, Pkg, Clp, Cbc, GLPK, Queryverse, RCall

model = Model()


@variable(model, x)

@objective(model, Min, x)

#################
m = Model(with_optimizer(GLPK.Optimizer))

@variable(m, 0 <= x <= 2 )
@variable(m, 0 <= y <= 30 )

@objective(m, Max, 5x + 3*y )

@constraint(m, 1x + 5y <= 3.0 )

JuMP.optimize!(m)
println("Objective value: ", JuMP.objective_value(m))
println("x = ", JuMP.value(x))
println("y = ", JuMP.value(y))


###############
df = DataFrame(load("breastcancer_processed.csv"))


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

sum(broadcast(^, lm[:residuals],2))
