using Pkg
using Random, Distributions

a = Vector{Int64}(1:5)

school = :UCI



## fill matrix
mat = zeros(Int8, 50, 50)

for i=1:50, j = 1:50
    if i == j
        mat[i,j] = -2
    elseif (i == j-1) | (i-1 == j)
        mat[i,j] = 1
    else
    end
end


## estimate pi
# random pairs in [-1,1]
rescale(x) = 1 - 2*x

num_sim = 5000000

pi_num = rand(2, num_sim)

pi_num = rescale.(pi_num)

a = Vector{Int64}(undef, num_sim)

for i=1:num_sim
    ((pi_num[:,i][1]^2 + pi_num[:,i][2]^2) < 1) ? a[i] = 1 : a[i] = 0
end

a

sum(a)/num_sim*4

## factorial
my_factorial(n) = prod(1:n)

my_factorial(big(100))

## binomial
rand(Binomial(10, 0.1), 100)

function binomial_ran(n, p)
    a = rand(n)
    for i=1:n
        a[i] > p ? a[i] = 0 : a[i] = 1
    end
    return(sum(a))
end

binomial_ran(10, 0.9)
