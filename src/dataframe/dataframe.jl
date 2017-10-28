"""
An AbstractDataFrame that stores a set of named columns

The columns are normally AbstractVectors stored in memory,
particularly a Vector or CategoricalVector.

**Constructors**

```julia
DataFrame(columns::Vector, names::Vector{Symbol})
DataFrame(kwargs...)
DataFrame(pairs::Pair{Symbol}...)
DataFrame() # an empty DataFrame
DataFrame(t::Type, nrows::Integer, ncols::Integer) # an empty DataFrame of arbitrary size
DataFrame(column_eltypes::Vector, names::Vector, nrows::Integer)
DataFrame(ds::Vector{Associative})
```

**Arguments**

* `columns` : a Vector with each column as contents
* `names` : the column names
* `kwargs` : the key gives the column names, and the value is the
  column contents
* `t` : elemental type of all columns
* `nrows`, `ncols` : number of rows and columns
* `column_eltypes` : elemental type of each column
* `ds` : a vector of Associatives

Each column in `columns` should be the same length.

**Notes**

A `DataFrame` is a lightweight object. As long as columns are not
manipulated, creation of a DataFrame from existing AbstractVectors is
inexpensive. For example, indexing on columns is inexpensive, but
indexing by rows is expensive because copies are made of each column.

Because column types can vary, a DataFrame is not type stable. For
performance-critical code, do not index into a DataFrame inside of
loops.

**Examples**

```julia
df = DataFrame()
v = ["x","y","z"][rand(1:3, 10)]
df1 = DataFrame(Any[collect(1:10), v, rand(10)], [:A, :B, :C])
df2 = DataFrame(A = 1:10, B = v, C = rand(10))
dump(df1)
dump(df2)
describe(df2)
head(df1)
df1[:A] + df2[:C]
df1[1:4, 1:2]
df1[[:A,:C]]
df1[1:2, [:A,:C]]
df1[:, [:A,:C]]
df1[:, [1,3]]
df1[1:4, :]
df1[1:4, :C]
df1[1:4, :C] = 40. * df1[1:4, :C]
[df1; df2]  # vcat
[df1  df2]  # hcat
size(df1)
```

"""
mutable struct DataFrame <: AbstractDataFrame
    typed::TypedDataFrame
end

# Low-level outer constructor
DataFrame(columns::Vector{Any}, cnames::AbstractVector{Symbol}) =
    DataFrame(TypedDataFrame(columns, cnames))

function DataFrame(pairs::Pair{Symbol,<:Any}...)
    cnames = Symbol[k for (k,v) in pairs]
    columns = Any[v for (k,v) in pairs]
    DataFrame(columns, cnames)
end

function DataFrame(; kwargs...)
    if isempty(kwargs)
        DataFrame(Any[], Symbol[])
    else
        DataFrame(kwpairs(kwargs)...)
    end
end

function DataFrame(columns::AbstractVector,
                   cnames::AbstractVector{Symbol} = gennames(length(columns)))
    return DataFrame(convert(Vector{Any}, columns), convert(Vector{Symbol}, cnames))
end

# Initialize an empty DataFrame with specific eltypes and names
function DataFrame(column_eltypes::AbstractVector{T}, cnames::AbstractVector{Symbol}, nrows::Integer) where T<:Type
    columns = Vector{Any}(length(column_eltypes))
    for (j, elty) in enumerate(column_eltypes)
        if elty >: Missing
            if Missings.T(elty) <: CategoricalValue
                columns[j] = CategoricalArray{Union{Missings.T(elty).parameters[1], Missing}}(nrows)
            else
                columns[j] = missings(elty, nrows)
            end
        else
            if elty <: CategoricalValue
                columns[j] = CategoricalVector{elty}(nrows)
            else
                columns[j] = Vector{elty}(nrows)
            end
        end
    end
    return DataFrame(columns, convert(Vector{Symbol}, cnames))
end

# Initialize an empty DataFrame with specific eltypes and names
# and whether a CategoricalArray should be created
function DataFrame(column_eltypes::AbstractVector{T}, cnames::AbstractVector{Symbol},
                   categorical::Vector{Bool}, nrows::Integer) where T<:Type
    # upcast Vector{DataType} -> Vector{Type} which can hold CategoricalValues
    updated_types = convert(Vector{Type}, column_eltypes)
    for i in eachindex(categorical)
        categorical[i] || continue
        if updated_types[i] >: Missing
            updated_types[i] = Union{CategoricalValue{Missings.T(updated_types[i])}, Missing}
        else
            updated_types[i] = CategoricalValue{updated_types[i]}
        end
    end
    return DataFrame(updated_types, cnames, nrows)
end

# Initialize empty DataFrame objects of arbitrary size
function DataFrame(t::Type, nrows::Integer, ncols::Integer)
    return DataFrame(fill(t, ncols), nrows)
end

# Initialize an empty DataFrame with specific eltypes
function DataFrame(column_eltypes::AbstractVector{T}, nrows::Integer) where T<:Type
    return DataFrame(column_eltypes, gennames(length(column_eltypes)), nrows)
end

##############################################################################
##
## AbstractDataFrame interface
##
##############################################################################

columns(df::DataFrame) = columns(df.typed)
Base.names(df::DataFrame) = names(df.typed)
function names!(df::DataFrame, cnames::AbstractVector{Symbol})
    df.typed = setnames(df.typed, cnames)
    df
end

function rename!(df::DataFrame, args...)
    df.typed = rename(df.typed, args...)
    df
end

function rename!(f::Function, df::DataFrame)
    df.typed = rename(f, df.typed)
    df
end

# TODO: Remove these
nrow(df::DataFrame) = nrow(df.typed)
ncol(df::DataFrame) = ncol(df.typed)

##############################################################################
##
## getindex() definitions
##
##############################################################################

# Cases:
#
# df[SingleColumnIndex] => AbstractDataVector
# df[MultiColumnIndex] => DataFrame
# df[SingleRowIndex, SingleColumnIndex] => Scalar
# df[SingleRowIndex, MultiColumnIndex] => DataFrame
# df[MultiRowIndex, SingleColumnIndex] => AbstractVector
# df[MultiRowIndex, MultiColumnIndex] => DataFrame
#
# General Strategy: let TypedDataFrame handle everything

# df[SingleColumnIndex] => AbstractDataVector
Base.getindex(df::DataFrame, col_ind::ColumnIndex) = df.typed[col_ind]

# df[MultiColumnIndex] => DataFrame
Base.getindex(df::DataFrame,
              col_inds::AbstractVector{<:Union{ColumnIndex, Missing}}) =
    DataFrame(df.typed[col_inds])

# df[:] => DataFrame
Base.getindex(df::DataFrame, col_inds::Colon) = copy(df)

# df[SingleRowIndex, SingleColumnIndex] => Scalar
Base.getindex(df::DataFrame, row_ind::Real, col_ind::ColumnIndex) =
    df.typed[row_ind, col_ind]

# df[SingleRowIndex, MultiColumnIndex] => DataFrame
Base.getindex(df::DataFrame,
              row_ind::Real,
              col_inds::AbstractVector{<:Union{ColumnIndex, Missing}}) =
    DataFrame(getindex(df.typed, row_ind, col_inds))

# df[MultiRowIndex, SingleColumnIndex] => AbstractVector
Base.getindex(df::DataFrame,
              row_inds::AbstractVector{<:Union{Real, Missing}},
              col_ind::ColumnIndex) =
    getindex(df.typed, row_inds, col_ind)

# df[MultiRowIndex, MultiColumnIndex] => DataFrame
Base.getindex(df::DataFrame,
              row_inds::AbstractVector{<:Union{Real, Missing}},
              col_inds::AbstractVector{<:Union{ColumnIndex, Missing}}) =
    DataFrame(getindex(df.typed, row_inds, col_inds))

# df[:, SingleColumnIndex] => AbstractVector
# df[:, MultiColumnIndex] => DataFrame
Base.getindex(df::DataFrame, row_ind::Colon, col_inds::Union{T, AbstractVector{T}}) where
    T <: Union{ColumnIndex, Missing} = df[col_inds]

# df[SingleRowIndex, :] => DataFrame
Base.getindex(df::DataFrame, row_ind::Real, col_inds::Colon) = df[[row_ind], col_inds]

# df[MultiRowIndex, :] => DataFrame
Base.getindex(df::DataFrame,
              row_inds::AbstractVector{<:Union{Real, Missing}},
              col_inds::Colon) =
    DataFrame(df.typed[row_inds, col_inds])

# df[:, :] => DataFrame
Base.getindex(df::DataFrame, ::Colon, ::Colon) = copy(df)

##############################################################################
##
## setindex!()
##
##############################################################################

# df[SingleColumnIndex] = AbstractVector
function Base.setindex!(df::DataFrame, v::AbstractVector, col_ind::ColumnIndex)
    df.typed = setindex(df.typed, v, col_ind)
    df
end

# df[SingleColumnIndex] = Single Item (EXPANDS TO NROW(df) if NCOL(df) > 0)
function Base.setindex!(df::DataFrame, v, col_ind::ColumnIndex)
    df.typed = setindex(df.typed, v, col_ind)    
    df
end

# df[MultiColumnIndex] = DataFrame
function Base.setindex!(df::DataFrame, new_df::DataFrame, col_inds::AbstractVector{Bool})
    df.typed = setindex!(df.typed, new_df, find(col_inds))
    df
end
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        col_inds::AbstractVector{<:ColumnIndex})
    df.typed = setindex(df.typed, new_df, col_inds)
    df
end

# df[MultiColumnIndex] = AbstractVector (REPEATED FOR EACH COLUMN)
function Base.setindex!(df::DataFrame, v::AbstractVector, col_inds::AbstractVector{Bool})
    setindex!(df, v, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        df[col_ind] = v
    end
    return df
end

# df[MultiColumnIndex] = Single Item (REPEATED FOR EACH COLUMN; EXPANDS TO NROW(df) if NCOL(df) > 0)
function Base.setindex!(df::DataFrame,
                        val::Any,
                        col_inds::AbstractVector{Bool})
    setindex!(df, val, find(col_inds))
end
function Base.setindex!(df::DataFrame, val::Any, col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        df[col_ind] = val
    end
    return df
end

# df[:] = AbstractVector or Single Item
Base.setindex!(df::DataFrame, v, ::Colon) = (df[1:size(df, 2)] = v; df)

# df[SingleRowIndex, SingleColumnIndex] = Single Item
Base.setindex!(df::DataFrame, v::Any, row_ind::Real, col_ind::ColumnIndex) =
    setindex!(df.typed, v, row_ind, col_ind)

# df[SingleRowIndex, MultiColumnIndex] = Single Item
Base.setindex!(df::DataFrame,
               v::Any,
               row_ind::Real,
               col_inds::AbstractVector{Bool}) =
    setindex!(df.typed, v, row_ind, col_inds)
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_ind::Real,
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        setindex!(df.typed, v, row_ind, col_ind)
    end
    return df
end

# df[SingleRowIndex, MultiColumnIndex] = 1-Row DataFrame
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_ind::Real,
                        col_inds::AbstractVector{Bool})
    setindex!(df, new_df, row_ind, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_ind::Real,
                        col_inds::AbstractVector{<:ColumnIndex})
    for j in 1:length(col_inds)
        setindex!(df.typed, new_df[j][1], row_ind, col_inds[j])
    end
    return df
end

# df[MultiRowIndex, SingleColumnIndex] = AbstractVector
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{Bool},
                        col_ind::ColumnIndex)
    setindex!(df, v, find(row_inds), col_ind)
end
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{<:Real},
                        col_ind::ColumnIndex)
    setindex!(v, row_inds, col_ind)
    return df
end

# df[MultiRowIndex, SingleColumnIndex] = Single Item
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{Bool},
                        col_ind::ColumnIndex)
    setindex!(df, v, find(row_inds), col_ind)
end
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{<:Real},
                        col_ind::ColumnIndex)
    setindex!(df.typed, v, row_inds, col_ind)
    return df
end

# df[MultiRowIndex, MultiColumnIndex] = DataFrame
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{Bool})
    setindex!(df, new_df, find(row_inds), find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{<:ColumnIndex})
    setindex!(df, new_df, find(row_inds), col_inds)
end
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{Bool})
    setindex!(df, new_df, row_inds, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{<:ColumnIndex})
    for j in 1:length(col_inds)
        insert_multiple_entries!(df, new_df[:, j], row_inds, col_inds[j])
    end
    return df
end

# df[MultiRowIndex, MultiColumnIndex] = AbstractVector
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, find(row_inds), find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{<:ColumnIndex})
    setindex!(df, v, find(row_inds), col_inds)
end
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, row_inds, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        insert_multiple_entries!(df, v, row_inds, col_ind)
    end
    return df
end

# df[MultiRowIndex, MultiColumnIndex] = Single Item
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, find(row_inds), find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{<:ColumnIndex})
    setindex!(df, v, find(row_inds), col_inds)
end
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, row_inds, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        insert_multiple_entries!(df, v, row_inds, col_ind)
    end
    return df
end

# df[:] = DataFrame, df[:, :] = DataFrame
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_inds::Colon,
                        col_inds::Colon=Colon())
    df.typed = copy(new_df.typed)
    df
end

# df[:, :] = ...
Base.setindex!(df::DataFrame, v, ::Colon, ::Colon) =
    (df[1:size(df, 1), 1:size(df, 2)] = v; df)

# df[Any, :] = ...
Base.setindex!(df::DataFrame, v, row_inds, ::Colon) =
    (df[row_inds, 1:size(df, 2)] = v; df)

# df[:, Any] = ...
Base.setindex!(df::DataFrame, v, ::Colon, col_inds) =
    (df[col_inds] = v; df)

# Special deletion assignment
Base.setindex!(df::DataFrame, x::Void, col_ind::Int) = delete!(df, col_ind)

##############################################################################
##
## Mutating Associative methods
##
##############################################################################

Base.empty!(df::DataFrame) = (df.typed = empty(df.typed); df)

function Base.insert!(df::DataFrame, col_ind::Int, item, name::Symbol)
    df.typed = insert(df.typed, col_ind, item, name)
    df
end

function Base.merge!(df::DataFrame, others::AbstractDataFrame...)
    for other in others
        for n in names(other)
            df[n] = other[n]
        end
    end
    return df
end

##############################################################################
##
## Copying
##
##############################################################################

# A copy of a DataFrame points to the original column vectors but
#   gets its own Index.
Base.copy(df::DataFrame) = DataFrame(copy(df.typed))

# Deepcopy is recursive -- if a column is a vector of DataFrames, each of
#   those DataFrames is deepcopied.
Base.deepcopy(df::DataFrame) =  DataFrame(deepcopy(df.typed))

##############################################################################
##
## Deletion / Subsetting
##
##############################################################################

# delete!() deletes columns; deleterows!() deletes rows
# delete!(df, 1)
# delete!(df, :Old)
function Base.delete!(df::DataFrame, inds::AbstractVector{<:Union{Int, Symbol}})
    df.typed = delete(df.typed, inds)
    return df
end

Base.delete!(df::DataFrame, c::Union{Int, Symbol}) = delete!(df, [c])

# deleterows!()
function deleterows!(df::DataFrame, ind::Union{Integer, AbstractVector{Int}})
    deleterows!(df.typed, ind)
    df
end


##############################################################################
##
## Hcat specialization
##
##############################################################################

# hcat! for 2 arguments, only a vector or a data frame is allowed
function hcat!(df1::DataFrame, df2::AbstractDataFrame)
    df1.typed = hcat(df1.typed, df2)
    df1
end

# definition required to avoid hcat! ambiguity
function hcat!(df1::DataFrame, df2::DataFrame)
    invoke(hcat!, Tuple{DataFrame, AbstractDataFrame}, df1, df2)
end

hcat!(df::DataFrame, x::AbstractVector) = hcat!(df, DataFrame(Any[x]))
hcat!(x::AbstractVector, df::DataFrame) = hcat!(DataFrame(Any[x]), df)
function hcat!(x, df::DataFrame)
    throw(ArgumentError("x must be AbstractVector or AbstractDataFrame"))
end
function hcat!(df::DataFrame, x)
    throw(ArgumentError("x must be AbstractVector or AbstractDataFrame"))
end

# hcat! for 1-n arguments
hcat!(df::DataFrame) = df
hcat!(a::DataFrame, b, c...) = hcat!(hcat!(a, b), c...)

# hcat
Base.hcat(df::DataFrame, x) = hcat!(copy(df), x)
Base.hcat(df1::DataFrame, df2::AbstractDataFrame) = hcat!(copy(df1), df2)
Base.hcat(df1::DataFrame, df2::AbstractDataFrame, dfn::AbstractDataFrame...) = hcat!(hcat(df1, df2), dfn...)

##############################################################################
##
## Missing values support
##
##############################################################################
"""
    allowmissing!(df::DataFrame)

Convert all columns of a `df` from element type `T` to
`Union{T, Missing}` to support missing values.

    allowmissing!(df::DataFrame, col::Union{Integer, Symbol})

Convert a single column of a `df` from element type `T` to
`Union{T, Missing}` to support missing values.

    allowmissing!(df::DataFrame, cols::AbstractVector{<:Union{Integer, Symbol}})

Convert multiple columns of a `df` from element type `T` to
`Union{T, Missing}` to support missing values.
"""
function allowmissing!(df::DataFrame,
                       cols::Union{ColumnIndex, AbstractVector{<: ColumnIndex}}=1:size(df, 2))
    df.typed = allowmissing(df.typed, cols)
    df
end


##############################################################################
##
## Categorical columns
##
##############################################################################

function categorical!(df::DataFrame, cols::Union{ColumnIndex, AbstractVector{<:ColumnIndex}})
    df.typed = categorical(df.typed, cols)
    df
end

function categorical!(df::DataFrame)
    df.typed = categorical(df.typed)
    df
end

function Base.append!(df1::DataFrame, df2::AbstractDataFrame)
   df1.typed = append!(df1.typed, df2)
   return df1
end

Base.convert(::Type{DataFrame}, A::Union{AbstractMatrix, Associative}) =
    DataFrame(convert(TypedDataFrame, A))


##############################################################################
##
## push! a row onto a DataFrame
##
##############################################################################

function Base.push!(df::DataFrame, associative::Any)
    push!(df.typed, associative)
    df
end