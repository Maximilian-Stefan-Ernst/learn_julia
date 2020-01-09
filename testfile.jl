using Pkg, Random, Distributions

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


################################################
#Speed
################################################

methods(^)

## function syntax

myfun(x,y) = x+y

myfun(x,y,z) = x+y*z

myfun(x::Int64, y::Int64) = x*y

methods(myfun)

myfun(2.0,3)

mat1 = rand(2,3)
mat2 = rand(2,3)

myfun(mat1, mat2)

f(x::T,y::T) where{T} = 4x+10y

f(x::T,y::T2,z::T2) where {T<:Number,T2} = 5x + 5y + 5z

f(1,2,3)

@code_llvm f(1,2,3)

@code_native myfun(1,2)


@time myfun(1,2)

@time myfun(1.0, 2.0)

myfun(x::Int64,y::Int64,z::Int64) = x+y+z

function myfun(x,y)
    return (x+y;x*y)
end

## types

mutable struct Car
    make
    model
end

mycar = Car("Toyota","Rav4")

mycar.model

mutable struct StaffMember{T<:Number}
    name::String
    field::Symbol
    age::T
end

ter = StaffMember("Terry",:football,17)

# immutable
struct Field
    name
    school
end
ds = Field(:DataScience,[:PhysicalScience;:ComputerScience])

push!(ds.school,:BiologicalScience)

ds

#lazy iterator types
b = 1:0.1:5

@time c = collect(b)

@time d = collect(b)

d = c

a = Vector{Float64}(1:5)

f(x) = x^2

map(f, a)

### Aufgabe 1

a = 1.1
b = 1.2

A = [1 0 0 0 0
    1.1 1 0 0 0
    1.2 0 1 0 0
    1.3 0 0 1 0
    1.4 0 0 0 1]

B = [1 0 0 0 0
    0 1 0 0 0
    0 1.5 1 0 0
    0 1.6 0 1 0
    0 1.7 0 0 1]

C = [1 0 0 0 0
    0 1 0 0 0
    0 0 1 0 0
    0 0 1.8 1 0
    0 0 1.9 0 1]

D = [1 0 0 0 0
    0 1 0 0 0
    0 0 1 0 0
    0 0 0 1 0
    0 0 0 2.0 1]

A*B*C*D

### Aufgabe 2
A = [3 -1 -2
    1 0 -1
    4 -2 -3]

A1 = inv(A)

b1 = Vector{Float64}(1:3)
b2 = Vector{Float64}([0, 1, 1])
b3 = Vector{Float64}([-2, 1, -5])
b4 = Vector{Float64}([0.5, 1, 5])
b5 = Vector{Float64}([1, 0, 0])

A1*b1
A1*b2
A1*b3
A1*b4
A1*b5


### Aufgabe 3

A = [1 0 0
    1 1 0
    1 1 1]

inv(A)

B = [2 1 1 1
    1 2 1 1
    1 1 2 1
    1 1 1 2]

inv(B)

C = [1 2 -3 1
    -1 3 -3 -2
    2 0 1 5
    3 1 -2 6]

inv(C)
