module ZygoteCUDAExt

using Zygote
using CUDA
using ChainRulesCore
using ChainRulesCore: project_type, _projection_mismatch

const CuSparseMatrixCSC = CUDA.CUSPARSE.CuSparseMatrixCSC

ChainRulesCore.is_inplaceable_destination(::CuSparseMatrixCSC) = true

function _csc_linear_indices(rowval::AbstractVector{<:Integer}, colptr::AbstractVector{<:Integer}, nrows::Integer)
  linear = Vector{Int}(undef, length(rowval))
  k = 0
  @inbounds for col in 1:(length(colptr) - 1)
    for idx in colptr[col]:(colptr[col + 1] - 1)
      linear[k += 1] = rowval[idx] + (col - 1) * nrows
    end
  end
  return linear
end

function ChainRulesCore.ProjectTo(x::CuSparseMatrixCSC{T, Ti}) where {T<:Number, Ti<:Integer}
  rowval = collect(x.rowVal)
  colptr = collect(x.colPtr)
  return ProjectTo{CuSparseMatrixCSC}(;
    element = ProjectTo(zero(T)),
    axes = axes(x),
    rowval = rowval,
    colptr = colptr,
    drowval = x.rowVal,
    dcolptr = x.colPtr,
    linear = _csc_linear_indices(rowval, colptr, size(x, 1)),
  )
end

function (project::ProjectTo{CuSparseMatrixCSC})(dx::AbstractArray)
  dy = if axes(dx) == project.axes
    dx
  else
    if size(dx) != (length(project.axes[1]), length(project.axes[2]))
      throw(_projection_mismatch(project.axes, size(dx)))
    end
    reshape(dx, project.axes)
  end
  nzval = project.element.(dy[project.linear])
  nzval isa CuArray || (nzval = CuArray(nzval))
  m, n = map(length, project.axes)
  return CuSparseMatrixCSC{project_type(project.element), eltype(project.rowval)}(
    project.dcolptr, project.drowval, nzval, (m, n)
  )
end

function (project::ProjectTo{CuSparseMatrixCSC})(dx::CuSparseMatrixCSC)
  if size(dx) != map(length, project.axes)
    throw(_projection_mismatch(project.axes, size(dx)))
  end
  samepattern = collect(dx.colPtr) == project.colptr && collect(dx.rowVal) == project.rowval
  if eltype(dx) <: project_type(project.element) && samepattern
    return dx
  elseif samepattern
    nzval = project.element.(dx.nzVal)
    nzval isa CuArray || (nzval = CuArray(nzval))
    m, n = size(dx)
    return CuSparseMatrixCSC{project_type(project.element), eltype(project.rowval)}(
      project.dcolptr, project.drowval, nzval, (m, n)
    )
  else
    return project(Matrix(dx))
  end
end

end
