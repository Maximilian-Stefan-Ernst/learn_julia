using JuMP, Pkg, Juniper, Ipopt

### Example 1

m = Model(with_optimizer(Juniper.Optimizer))

# Declaring variables
@variable(m, 0<= x1 <=10)
@variable(m, x2 >=0)
@variable(m, x3 >=0)

# Setting the objective
@objective(m, Max, x1 + 2x2 + 5x3)

# Adding constraints
@constraint(m, constraint1, -x1 +  x2 + 3x3 <= -5)
@constraint(m, constraint2,  x1 + 3x2 - 7x3 <= 10)
@constraint(m, constraint3, abs(x1) >= 0)

# Printing the prepared optimization model
print(m)

# Solving the optimization problem
JuMP.optimize!(m)

# Printing the optimal solutions obtained
println("Optimal Solutions:")
println("x1 = ", JuMP.value(x1))
println("x2 = ", JuMP.value(x2))
println("x3 = ", JuMP.value(x3))

# Printing the optimal dual variables
println("Dual Variables:")
println("dual1 = ", JuMP.shadow_price(constraint1))
println("dual2 = ", JuMP.shadow_price(constraint2))


# matrix notation
m = Model(with_optimizer(GLPK.Optimizer))

c = [ 1; 2; 5]
A = [-1  1  3;
      1  3 -7]
b = [-5; 10]

@variable(m, x[1:3] >= 0)
@objective(m, Max, sum( c[i]*x[i] for i in 1:3) )

@constraint(m, constraint[j in 1:2], sum( A[j,i]*x[i] for i in 1:3 ) <= b[j] )
@constraint(m, bound, x[1] <= 10)

JuMP.optimize!(m)


println("Optimal Solutions:")
for i in 1:3
  println("x[$i] = ", JuMP.value(x[i]))
end

println("Dual Variables:")
for j in 1:2
  println("dual[$j] = ", JuMP.shadow_price(constraint[j]))
end
