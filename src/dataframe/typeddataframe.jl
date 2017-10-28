# merge() implementation taken from NamedTuples,
# but using the RHS instead of the RHS for consistency with
# other merge() methods and the NamedTuple implementation in Julia 0.7
# Note that setindex() needs this definition to replace existing entries
function Base.merge( lhs::NamedTuple, rhs::NamedTuple )
    nms = unique( vcat( fieldnames( lhs ), fieldnames( rhs )) )
    name = NamedTuples.create_tuple( nms )
    # FIXME should handle the type only case
    vals = [ haskey( rhs, nm ) ? rhs[nm] : lhs[nm] for nm in nms ]
    getfield(NamedTuples,name)(vals...)
end

function setindex{V}( t::NamedTuple, key::Symbol, val::V)
    nt = getfield( NamedTuples, NamedTuples.create_tuple( [key] ))( val )
    return merge( t, nt )
end

"""
An AbstractDataFrame that stores a set of named columns
and their types to ensure type stability of functions operating on it.

The columns are normally AbstractVectors stored in memory,
particularly a Vector or CategoricalVector.

**Constructors**

```julia
TypedDataFrame(columns::Vector, names::Vector{Symbol})
TypedDataFrame(kwargs...)
TypedDataFrame(pairs::Pair{Symbol}...)
TypedDataFrame() # an empty TypedDataFrame
TypedDataFrame(t::Type, nrows::Integer, ncols::Integer) # an empty TypedDataFrame of arbitrary size
TypedDataFrame(column_eltypes::Vector, names::Vector, nrows::Integer)
TypedDataFrame(ds::Vector{Associative})
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

A `TypedDataFrame` is a lightweight object. As long as columns are not
manipulated, creation of a TypedDataFrame from existing AbstractVectors is
inexpensive. For example, indexing on columns is inexpensive, but
indexing by rows is expensive because copies are made of each column.

Because column types can vary, a TypedDataFrame is not type stable. For
performance-critical code, do not index into a TypedDataFrame inside of
loops.

**Examples**

```julia
df = TypedDataFrame()
v = ["x","y","z"][rand(1:3, 10)]
df1 = TypedDataFrame(Any[collect(1:10), v, rand(10)], [:A, :B, :C])
df2 = TypedDataFrame(A = 1:10, B = v, C = rand(10))
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
struct TypedDataFrame{C<:NamedTuple} <: AbstractDataFrame
    columns::C

    function TypedDataFrame{C}(columns::C) where {C <: NamedTuple}
        # Perform minimal consistency checks, as users are supposed to use
        # the outer constructor rather than this one
        for col in columns
            length(columns[1]) != length(col) && throw(DimensionMismatch("columns must all have the same length"))
        end
        new{C}(columns)
    end
end

TypedDataFrame(columns::C) where {C <: NamedTuple} = TypedDataFrame{C}(columns)

# Low-level outer constructor
function TypedDataFrame(columns::Vector{Any}, cnames::AbstractVector{Symbol})
    if length(columns) == length(cnames) == 0
        C = eval(:(@NT()))
        return TypedDataFrame{C}(C())
    elseif length(columns) != length(cnames)
        throw(DimensionMismatch("Number of columns ($(length(columns))) and number of column names ($(length(cnames))) are not equal"))
    end
    lengths = [isa(col, AbstractArray) ? length(col) : 1 for col in columns]
    minlen, maxlen = extrema(lengths)
    if minlen != 0 || maxlen != 0
        if minlen != maxlen || minlen == maxlen == 1
            # recycle scalars
            for i in 1:length(columns)
                isa(columns[i], AbstractArray) && continue
                columns[i] = fill(columns[i], maxlen)
                lengths[i] = maxlen
            end
            uls = unique(lengths)
            if length(uls) != 1
                strnames = string.(cnames)
                estrings = ["column length $u for column(s) " *
                            join(strnames[lengths .== u], ", ", " and ") for (i, u) in enumerate(uls)]
                throw(DimensionMismatch(join(estrings, " is incompatible with ", ", and is incompatible with ")))
            end
        end
        for (i, c) in enumerate(columns)
            if isa(c, Range)
                columns[i] = collect(c)
            elseif !isa(c, AbstractVector)
                throw(DimensionMismatch("columns must be 1-dimensional"))
            end
        end
    end
    C = eval(:(@NT($(cnames...)))){map(typeof, columns)...}
    TypedDataFrame{C}(C(columns...))
end

function TypedDataFrame(pairs::Pair{Symbol,<:Any}...)
    cnames = Symbol[k for (k,v) in pairs]
    columns = Any[v for (k,v) in pairs]
    TypedDataFrame(columns, cnames)
end

function TypedDataFrame(; kwargs...)
    if isempty(kwargs)
        TypedDataFrame(Any[], Symbol[])
    else
        TypedDataFrame((k => v for (k,v) in kwargs)...)
    end
end

TypedDataFrame(columns::AbstractVector,
               cnames::AbstractVector{Symbol} = gennames(length(columns))) =
    TypedDataFrame(convert(Vector{Any}, columns), convert(Vector{Symbol}, cnames))

# Initialize an empty TypedDataFrame with specific eltypes and names
function TypedDataFrame(column_eltypes::AbstractVector{T}, cnames::AbstractVector{Symbol}, nrows::Integer) where T<:Type
    columns = Vector{Any}(length(column_eltypes))
    for (j, elty) in enumerate(column_eltypes)
        if elty >: Missing
            if Missings.T(elty) <: CategoricalValue
                columns[j] = CategoricalArray{Union{Missings.T(elty).parameters[1], Missing}}(nrows)
            else
                columns[j] = nulls(elty, nrows)
            end
        else
            if elty <: CategoricalValue
                columns[j] = CategoricalVector{elty}(nrows)
            else
                columns[j] = Vector{elty}(nrows)
            end
        end
    end
    return TypedDataFrame(columns, convert(Vector{Symbol}, cnames))
end

# Initialize an empty TypedDataFrame with specific eltypes and names
# and whether a nominal array should be created
function TypedDataFrame(column_eltypes::AbstractVector{T}, cnames::AbstractVector{Symbol},
                        nominal::Vector{Bool}, nrows::Integer) where T<:Type
    # upcast Vector{DataType} -> Vector{Type} which can hold CategoricalValues
    updated_types = convert(Vector{Type}, column_eltypes)
    for i in eachindex(nominal)
        nominal[i] || continue
        if updated_types[i] >: Missing
            updated_types[i] = Union{CategoricalValue{Missings.T(updated_types[i])}, Missing}
        else
            updated_types[i] = CategoricalValue{updated_types[i]}
        end
    end
    return TypedDataFrame(updated_types, cnames, nrows)
end

# Initialize empty TypedDataFrame objects of arbitrary size
function TypedDataFrame(t::Type, nrows::Integer, ncols::Integer)
    return TypedDataFrame(fill(t, ncols), nrows)
end

# Initialize an empty TypedDataFrame with specific eltypes
function TypedDataFrame(column_eltypes::AbstractVector{T}, nrows::Integer) where T<:Type
    return TypedDataFrame(column_eltypes, gennames(length(column_eltypes)), nrows)
end

##############################################################################
## interface
##
##############################################################################

columns(df::TypedDataFrame) = df.columns
Base.names(df::TypedDataFrame) = fieldnames(df.columns)

function setnames(df::TypedDataFrame, cnames::AbstractVector{Symbol})
    C = eval(:(@NT($(cnames...)))){map(typeof, df.columns)...}
    TypedDataFrame{C}(C(df.columns...))
end

function rename(df::TypedDataFrame, nms)
    cnames = names(df)
    for (from, to) in nms
        from == to && continue # No change, nothing to do
        if !isdefined(df.columns, from)
            throw(ArgumentError("Tried renaming $from to $to, when $from does not exist."))
        end
        if isdefined(df.columns, to)
            throw(ArgumentError("Tried renaming $from to $to, when $to already exists."))
        end
        cnames[int_colinds(df, from)] = to
    end
    setnames(df, cnames)
end

rename(df::TypedDataFrame, nms::Pair{Symbol,Symbol}...) = rename(df, collect(nms))
rename(f::Function, df::TypedDataFrame) = rename(df, [(x=>f(x)) for x in names(df)])

# TODO: Remove these
nrow(df::TypedDataFrame) = ncol(df) > 0 ? length(df.columns[1])::Int : 0
ncol(df::TypedDataFrame) = length(df.columns)

##############################################################################
##
## getindex() definitions
##
##############################################################################

# Cases:
#
# df[SingleColumnIndex] => AbstractDataVector
# df[MultiColumnIndex] => TypedDataFrame
# df[SingleRowIndex, SingleColumnIndex] => Scalar
# df[SingleRowIndex, MultiColumnIndex] => TypedDataFrame
# df[MultiRowIndex, SingleColumnIndex] => AbstractVector
# df[MultiRowIndex, MultiColumnIndex] => TypedDataFrame
#
# General Strategy:
#
# Let getindex(df.columns, col_inds) from NamedTuple handle the resolution
#  of column indices
# Let getindex(df.columns[j], row_inds) from AbstractVector() handle
#  the resolution of row indices

# df[SingleColumnIndex] => AbstractDataVector
Base.getindex(df::TypedDataFrame, col_ind::ColumnIndex) =
    df.columns[col_ind]

# df[MultiColumnIndex] => TypedDataFrame
function Base.getindex(df::TypedDataFrame,
                       col_inds::AbstractVector{<:Union{ColumnIndex, Missing}})
    allunique(col_inds) || throw(ArgumentError("duplicate column indices are not allowed"))
    TypedDataFrame(df.columns[col_inds])
end

# df[:] => TypedDataFrame
Base.getindex(df::TypedDataFrame, col_inds::Colon) = df

# df[SingleRowIndex, SingleColumnIndex] => Scalar
# df[MultiRowIndex, SingleColumnIndex] => AbstractVector
Base.getindex(df::TypedDataFrame,
              row_inds::Union{Integer, AbstractVector{<:Union{Integer, Missing}}},
              col_ind::ColumnIndex) =
    df[col_ind][row_inds]

# df[SingleRowIndex, MultiColumnIndex] => TypedDataFrame
# df[MultiRowIndex, MultiColumnIndex] => TypedDataFrame
Base.getindex(df::TypedDataFrame,
              row_inds::Union{Integer, AbstractVector{<:Union{Integer, Missing}}},
              col_inds::AbstractVector{<:Union{ColumnIndex, Missing}}) =
    TypedDataFrame(map(v -> v[row_inds], df.columns[col_inds]))

# df[:, SingleColumnIndex] => AbstractVector
# df[:, MultiColumnIndex] => TypedDataFrame
Base.getindex(df::TypedDataFrame, row_ind::Colon, col_inds::Union{T, AbstractVector{T}}) where
    T <: Union{ColumnIndex, Missing} =
    df[col_inds]

# df[SingleRowIndex, :] => TypedDataFrame
Base.getindex(df::TypedDataFrame, row_ind::Integer, col_inds::Colon) = df[[row_ind], col_inds]

# df[MultiRowIndex, :] => TypedDataFrame
Base.getindex(df::TypedDataFrame,
              row_inds::AbstractVector{<:Union{Integer, Missing}},
              col_inds::Colon) =
    TypedDataFrame(map(v -> v[row_inds], df.columns))

# df[:, :] => TypedDataFrame
Base.getindex(df::TypedDataFrame, ::Colon, ::Colon) = df

##############################################################################
##
## setindex()
##
##############################################################################

# Will automatically add a new column if needed
function insert_single_column(df::TypedDataFrame,
                              dv::AbstractVector,
                              col_ind::ColumnIndex)

    if ncol(df) != 0 && nrow(df) != length(dv)
        error("New columns must have the same length as old columns")
    end
    if isdefined(df.columns, col_ind)
        # setindex() does not support Integer indices
        col_sym = col_ind isa Symbol ? col_ind : fieldnames(df.columns)[col_ind]
        new_columns = setindex(df.columns, col_sym, dv)
    elseif col_ind isa Symbol
        C = eval(:(@NT($(col_ind)))){typeof(dv)}
        new_columns = merge(df.columns, C(dv))
    else
        error("Cannot assign to non-existent column: $col_ind")
    end
    return new_columns
end

function insert_single_entry!(df::TypedDataFrame, v::Any, row_ind::Integer, col_ind::ColumnIndex)
    if isdefined(df.columns, col_ind)
        df.columns[col_ind][row_ind] = v
        return v
    else
        error("Cannot assign to non-existent column: $col_ind")
    end
end

function insert_multiple_entries!(df::TypedDataFrame,
                                  v::Any,
                                  row_inds::AbstractVector{<:Integer},
                                  col_ind::ColumnIndex)
    if isdefined(df.columns, col_ind)
        df.columns[col_ind][row_inds] = v
        return v
    else
        error("Cannot assign to non-existent column: $col_ind")
    end
end

function upgrade_scalar(df::TypedDataFrame, v::AbstractArray)
    msg = "setindex!(::TypedDataFrame, ...) only broadcasts scalars, not arrays"
    throw(ArgumentError(msg))
end
function upgrade_scalar(df::TypedDataFrame, v::Any)
    n = (ncol(df) == 0) ? 1 : nrow(df)
    fill(v, n)
end

# df[SingleColumnIndex] = TypedDataFrame
Base.setindex(df::TypedDataFrame, v::AbstractVector, col_ind::ColumnIndex) =
    TypedDataFrame(insert_single_column(df, v, col_ind))

# setindex(df, Single Item, SingleColumnIndex) (EXPANDS TO NROW(df) if NCOL(df) > 0)
function Base.setindex(df::TypedDataFrame, v, col_ind::ColumnIndex)
    if isdefined(df.columns, col_ind)
        fill!(df[col_ind], v)
        return df
    else
        return TypedDataFrame(insert_single_column(df, upgrade_scalar(df, v), col_ind))
    end
end

# df[SingleColumnIndex] = Single Item (EXPANDS TO NROW(df) if NCOL(df) > 0)
function Base.setindex!(df::TypedDataFrame, v, col_ind::ColumnIndex)
    if isdefined(df.columns, col_ind)
        fill!(df[col_ind], v)
    else
        error("Cannot assign to non-existent column: $col_ind")
    end
    return df
end

# setindex(df, TypedDataFrame, MultiColumnIndex)
Base.setindex(df::TypedDataFrame, new_df::TypedDataFrame, col_inds::AbstractVector{Bool}) =
    setindex(df, new_df, find(col_inds))

# df[MultiColumnIndex] = TypedDataFrame
Base.setindex!(df::TypedDataFrame, new_df::TypedDataFrame, col_inds::AbstractVector{Bool}) =
    setindex!(df, new_df, find(col_inds))

function Base.setindex!(df::TypedDataFrame,
                        new_df::AbstractDataFrame,
                        col_inds::AbstractVector{<:ColumnIndex})
    columns = df.columns
    for j in 1:length(col_inds)
        columns = insert_single_column(df, new_df[j], col_inds[j])
    end
    return TypedDataFrame(columns)
end

# setindex(df, AbstractVector, MultiColumnIndex) (REPEATED FOR EACH COLUMN)
function Base.setindex(df::TypedDataFrame, v::AbstractVector, col_inds::AbstractVector{Bool})
    setindex(df, v, find(col_inds))
end
function Base.setindex(df::TypedDataFrame,
                       v::AbstractVector,
                       col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        df = setindex(df, v, col_ind)
    end
    return df
end

# df[MultiColumnIndex] = AbstractVector (REPEATED FOR EACH COLUMN)
function Base.setindex!(df::TypedDataFrame, v::AbstractVector, col_inds::AbstractVector{Bool})
    setindex!(df, v, find(col_inds))
end
function Base.setindex!(df::TypedDataFrame,
                        v::AbstractVector,
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        df[col_ind] = v
    end
    return df
end

# setindex(df, Single Item,  MultiColumnIndex) (REPEATED FOR EACH COLUMN; EXPANDS TO NROW(df) if NCOL(df) > 0)
Base.setindex(df::TypedDataFrame, val::Any, col_inds::AbstractVector{Bool}) =
    setindex(df, val, find(col_inds))
function Base.setindex(df::TypedDataFrame, val::Any, col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        df = setindex(df, val, col_ind)
    end
    return df
end

# df[MultiColumnIndex] = Single Item (REPEATED FOR EACH COLUMN; EXPANDS TO NROW(df) if NCOL(df) > 0)
Base.setindex!(df::TypedDataFrame, val::Any, col_inds::AbstractVector{Bool}) =
    setindex!(df, val, find(col_inds))
function Base.setindex!(df::TypedDataFrame, val::Any, col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        df[col_ind] = val
    end
    return df
end

# setindex(df, AbstractVector or Single Item, :)
Base.setindex(df::TypedDataFrame, v, ::Colon) = setindex(df, v, 1:size(df, 2))

# df[:] = AbstractVector or Single Item
Base.setindex!(df::TypedDataFrame, v, ::Colon) = (df[1:size(df, 2)] = v; df)

# setindex(df, Single Item, SingleRowIndex, SingleColumnIndex)
Base.setindex(df::TypedDataFrame, v::Any, row_ind::Integer, col_ind::ColumnIndex) =
    insert_single_entry!(df, v, row_ind, col_ind)

# df[SingleRowIndex, SingleColumnIndex] = Single Item
Base.setindex!(df::TypedDataFrame, v::Any, row_ind::Integer, col_ind::ColumnIndex) =
    insert_single_entry!(df, v, row_ind, col_ind)

# df[SingleRowIndex, MultiColumnIndex] = Single Item
function Base.setindex!(df::TypedDataFrame,
                        v::Any,
                        row_ind::Integer,
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, row_ind, find(col_inds))
end
function Base.setindex!(df::TypedDataFrame,
                        v::Any,
                        row_ind::Integer,
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        insert_single_entry!(df, v, row_ind, col_ind)
    end
    return df
end

# df[SingleRowIndex, MultiColumnIndex] = 1-Row TypedDataFrame
function Base.setindex!(df::TypedDataFrame,
                        new_df::TypedDataFrame,
                        row_ind::Integer,
                        col_inds::AbstractVector{Bool})
    setindex!(df, new_df, row_ind, find(col_inds))
end
function Base.setindex!(df::TypedDataFrame,
                        new_df::TypedDataFrame,
                        row_ind::Integer,
                        col_inds::AbstractVector{<:ColumnIndex})
    for j in 1:length(col_inds)
        insert_single_entry!(df, new_df[j][1], row_ind, col_inds[j])
    end
    return df
end

# df[MultiRowIndex, SingleColumnIndex] = AbstractVector
function Base.setindex!(df::TypedDataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{Bool},
                        col_ind::ColumnIndex)
    setindex!(df, v, find(row_inds), col_ind)
end
function Base.setindex!(df::TypedDataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{<:Integer},
                        col_ind::ColumnIndex)
    insert_multiple_entries!(df, v, row_inds, col_ind)
    return df
end

# df[MultiRowIndex, SingleColumnIndex] = Single Item
function Base.setindex!(df::TypedDataFrame,
                        v::Any,
                        row_inds::AbstractVector{Bool},
                        col_ind::ColumnIndex)
    setindex!(df, v, find(row_inds), col_ind)
end
function Base.setindex!(df::TypedDataFrame,
                        v::Any,
                        row_inds::AbstractVector{<:Integer},
                        col_ind::ColumnIndex)
    insert_multiple_entries!(df, v, row_inds, col_ind)
    return df
end

# df[MultiRowIndex, MultiColumnIndex] = TypedDataFrame
function Base.setindex!(df::TypedDataFrame,
                        new_df::TypedDataFrame,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{Bool})
    setindex!(df, new_df, find(row_inds), find(col_inds))
end
function Base.setindex!(df::TypedDataFrame,
                        new_df::TypedDataFrame,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{<:ColumnIndex})
    setindex!(df, new_df, find(row_inds), col_inds)
end
function Base.setindex!(df::TypedDataFrame,
                        new_df::TypedDataFrame,
                        row_inds::AbstractVector{<:Integer},
                        col_inds::AbstractVector{Bool})
    setindex!(df, new_df, row_inds, find(col_inds))
end
function Base.setindex!(df::TypedDataFrame,
                        new_df::TypedDataFrame,
                        row_inds::AbstractVector{<:Integer},
                        col_inds::AbstractVector{<:ColumnIndex})
    for j in 1:length(col_inds)
        insert_multiple_entries!(df, new_df[:, j], row_inds, col_inds[j])
    end
    return df
end

# df[MultiRowIndex, MultiColumnIndex] = AbstractVector
function Base.setindex!(df::TypedDataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, find(row_inds), find(col_inds))
end
function Base.setindex!(df::TypedDataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{<:ColumnIndex})
    setindex!(df, v, find(row_inds), col_inds)
end
function Base.setindex!(df::TypedDataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{<:Integer},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, row_inds, find(col_inds))
end
function Base.setindex!(df::TypedDataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{<:Integer},
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        insert_multiple_entries!(df, v, row_inds, col_ind)
    end
    return df
end

# df[MultiRowIndex, MultiColumnIndex] = Single Item
function Base.setindex!(df::TypedDataFrame,
                        v::Any,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, find(row_inds), find(col_inds))
end
function Base.setindex!(df::TypedDataFrame,
                        v::Any,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{<:ColumnIndex})
    setindex!(df, v, find(row_inds), col_inds)
end
function Base.setindex!(df::TypedDataFrame,
                        v::Any,
                        row_inds::AbstractVector{<:Integer},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, row_inds, find(col_inds))
end
function Base.setindex!(df::TypedDataFrame,
                        v::Any,
                        row_inds::AbstractVector{<:Integer},
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        insert_multiple_entries!(df, v, row_inds, col_ind)
    end
    return df
end

# df[:] = TypedDataFrame, df[:, :] = TypedDataFrame
function Base.setindex!(df::TypedDataFrame,
                        new_df::TypedDataFrame,
                        row_inds::Colon,
                        col_inds::Colon=Colon())
    df.columns = map(copy, new_df.columns)
    df
end

# df[:, :] = ...
Base.setindex!(df::TypedDataFrame, v, ::Colon, ::Colon) =
    (df[1:size(df, 1), 1:size(df, 2)] = v; df)

# df[Any, :] = ...
Base.setindex!(df::TypedDataFrame, v, row_inds, ::Colon) =
    (df[row_inds, 1:size(df, 2)] = v; df)

# df[:, Any] = ...
Base.setindex!(df::TypedDataFrame, v, ::Colon, col_inds) =
    (df[col_inds] = v; df)

# Special deletion assignment
Base.setindex!(df::TypedDataFrame, x::Void, col_ind::Int) = delete!(df, col_ind)

##############################################################################
##
## Mutating Associative methods
##
##############################################################################

empty(df::TypedDataFrame) = TypedDataFrame(Any[], Symbol[])

function insert(df::TypedDataFrame, col_ind::Int, item::AbstractVector, name::Symbol)
    0 < col_ind <= ncol(df) + 1 || throw(BoundsError())
    size(df, 1) == length(item) || size(df, 1) == 0 || error("number of rows does not match")

    C = eval(:(@NT($(name)))){typeof(item)}
    TypedDataFrame(merge(merge(df.columns[1:col_ind-1], C(item)),
                         df.columns[col_ind:end]))
end

insert(df::TypedDataFrame, col_ind::Int, item, name::Symbol) =
    insert(df, col_ind, upgrade_scalar(df, item), name)

function Base.merge(df::TypedDataFrame, others...)
    for other in others
        for n in names(other)
            df = setindex(df, n, other[n])
        end
    end
    return df
end

##############################################################################
##
## Copying
##
##############################################################################

# A copy of a TypedDataFrame is a no-op
Base.copy(df::TypedDataFrame) = df

# Deepcopy is recursive: each column is deepcopied
Base.deepcopy(df::TypedDataFrame) = TypedDataFrame(map(deepcopy, df.columns))

##############################################################################
##
## Deletion / Subsetting
##
##############################################################################

# delete() deletes columns; deleterows() deletes rows
# delete(df, 1)
# delete(df, :Old)
function delete(df::TypedDataFrame, inds::AbstractVector{<:Integer})
    columns = df.columns
    for ind in sort(inds, rev=true)
        if 1 <= ind <= ncol(df)
            columns = NamedTuples.delete(columns, fieldnames(columns)[ind])
        else
            throw(ArgumentError("Can't delete a non-existent TypedDataFrame column $ind"))
        end
    end
    return TypedDataFrame(columns)
end
function delete(df::TypedDataFrame, inds::AbstractVector{Symbol})
    columns = df.columns
    for ind in inds
        if isdefined(df.columns, ind)
            columns = NamedTuples.delete(columns, ind)
        else
            throw(KeyError("Can't delete a non-existent TypedDataFrame column $ind"))
        end
    end
    return TypedDataFrame(columns)
end
delete(df::TypedDataFrame, ind::ColumnIndex) = delete(df, [ind])

# deleterows!()
function deleterows!(df::TypedDataFrame, ind::Union{Integer, AbstractVector{<:Integer}})
    for i in 1:ncol(df)
        deleteat!(df.columns[i], ind)
    end
    df
end

##############################################################################
##
## Hcat specialization
##
##############################################################################

# hcat for 2 arguments
Base.hcat(df1::TypedDataFrame, df2::AbstractDataFrame) =
    TypedDataFrame(merge(columns(df1), columns(df2)))
Base.hcat(df::TypedDataFrame, x::AbstractVector) = hcat(df, DataFrame(Any[x]))

# hcat for 1-n arguments
Base.hcat(df::TypedDataFrame) = df
Base.hcat(a::TypedDataFrame, b, c...) = hcat(hcat(a, b), c...)

##############################################################################
##
## Nullability
##
##############################################################################

function allowmissing(df::TypedDataFrame, col::ColumnIndex)
    v = Vector{Union{eltype(df[col]), Missing}}(df.columns[col])
    TypedDataFrame(setindex(df.columns, sym_colinds(df, col), v))
end

function allowmissing(df::TypedDataFrame, cols::Vector{<:ColumnIndex}=1:size(df, 2))
    columns = df.columns
    for col in cols
        columns = setindex(columns, sym_colinds(df, col),
                           Vector{Union{eltype(df[col]), Missing}}(df[col]))
    end
    TypedDataFrame(columns)
end


##############################################################################
##
## Categorical columns
##
##############################################################################

CategoricalArrays.categorical(df::TypedDataFrame, cname::ColumnIndex) =
    TypedDataFrame(setindex(df.columns, sym_colinds(df, cname), categorical(df[cname])))

function CategoricalArrays.categorical(df::TypedDataFrame, cnames::Vector{<:ColumnIndex})
    columns = df.columns
    for cname in cnames
        columns = setindex(columns, sym_colinds(df, cname), categorical(df[cname]))
    end
    TypedDataFrame(columns)
end

CategoricalArrays.categorical(df::TypedDataFrame) =
    TypedDataFrame(map(v -> eltype(v) <: AbstractString ? categorical(v) : v, df.columns))

function Base.append!(df1::TypedDataFrame, df2::AbstractDataFrame)
   names(df1) == names(df2) || error("Column names do not match")
   eltypes(df1) == eltypes(df2) || error("Column eltypes do not match")
   ncols = size(df1, 2)
   # TODO: This needs to be a sort of transaction to be 100% safe
   for j in 1:ncols
       append!(df1[j], df2[j])
   end
   return df1
end

function Base.convert(::Type{TypedDataFrame}, A::AbstractMatrix)
    n = size(A, 2)
    cols = Vector{Any}(n)
    for i in 1:n
        cols[i] = A[:, i]
    end
    return TypedDataFrame(cols)
end

function Base.convert(::Type{TypedDataFrame}, d::Associative)
    cnames = keys(d)
    if isa(d, Dict)
        cnames = sort!(collect(keys(d)))
    else
        cnames = keys(d)
    end
    columns = Any[d[c] for c in cnames]
    TypedDataFrame(columns, Symbol.(cnames))
end


##############################################################################
##
## push! a row onto a TypedDataFrame
##
##############################################################################

function Base.push!(df::TypedDataFrame, associative::Associative{Symbol,Any})
    i = 1
    for nm in names(df)
        try
            push!(df[nm], associative[nm])
        catch
            #clean up partial row
            for j in 1:(i - 1)
                pop!(df[names(df)[j]])
            end
            msg = "Error adding value to column :$nm."
            throw(ArgumentError(msg))
        end
        i += 1
    end
end

function Base.push!(df::TypedDataFrame, associative::Associative)
    i = 1
    for nm in names(df)
        try
            val = get(() -> associative[string(nm)], associative, nm)
            push!(df[nm], val)
        catch
            #clean up partial row
            for j in 1:(i - 1)
                pop!(df[names(df)[j]])
            end
            msg = "Error adding value to column :$nm."
            throw(ArgumentError(msg))
        end
        i += 1
    end
end

# array and tuple like collections
function Base.push!(df::TypedDataFrame, iterable::Any)
    if length(iterable) != length(df.columns)
        msg = "Length of iterable does not match TypedDataFrame column count."
        throw(ArgumentError(msg))
    end
    i = 1
    for t in iterable
        try
            push!(df.columns[i], t)
        catch
            #clean up partial row
            for j in 1:(i - 1)
                pop!(df.columns[j])
            end
            msg = "Error adding $t to column :$(names(df)[i]). Possible type mis-match."
            throw(ArgumentError(msg))
        end
        i += 1
    end
end
