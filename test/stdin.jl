using DataFrames
using CSV

df = CSV.read(STDIN)

io = IOBuffer()
print(io, df)
