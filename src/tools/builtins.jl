macro __new__(T, args...)
  esc(Expr(:new, T, args...))
end

macro __splatnew__(T, args)
  esc(Expr(:splatnew, T, args))
end

@inline __new__(T, args...) = __splatnew__(T, args)

@generated function __splatnew__(::Type{T}, args::Tuple{Vararg{Any,N}}) where {T,N}
  Expr(:new, T, [:(args[$i]) for i in 1:N]...)
end

literal_getindex(x, ::Val{i}) where i = getindex(x, i)
literal_indexed_iterate(x, ::Val{i}) where i = Base.indexed_iterate(x, i)
literal_indexed_iterate(x, ::Val{i}, state) where i = Base.indexed_iterate(x, i, state)
