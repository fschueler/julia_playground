# rowwise y = Ax
function matvec_r(A, x, n)
  y = zeros(eltype(x), n)
  for i = 1:n
    for j = 1:n
      y[i] = y[i] + A[i,j] * x[j]
    end
  end
  y
end

# rowwise y = Ax, blocked
function matvec_r2(A, x, n)
  y = zeros(eltype(x), n)
  for ii = 1:BL:n
    for j = 1:n
      for i = ii:(ii+BL-1)
        y[i] = A[i,j] * x[j]
      end
    end
  end
  y
end

# columnwise y = Ax
function matvec_c1(A, x, n)
  y = zeros(eltype(x), n)
  for j = 1:n
    y = y + A[:,j] * x[j]
  end
  y
end

# columnwise y = Ax
function matvec_c2(A, x, n)
  y = zeros(eltype(x), n)
  for j = 1:n
    for i = 1:n
      y[i] = A[i,j] * x[j]
    end
  end
  y
end

# columnwise y = Ax, vectorized
function matvec_c3(A, x, n)
  y = zeros(eltype(x), n)
  for j = 1:n
    @simd for i = 1:n
      y[i] = A[i,j] * x[j]
    end
  end
  y
end

# columnwise y = Ax, unrolled
function matvec_c4(A, x, n)
  y = zeros(eltype(x), n)
  for j = 1:n
    for i = 1:8:n
      y[i] = A[i,j] * x[j]
      y[i+1] = A[i+1,j] * x[j]
      y[i+2] = A[i+2,j] * x[j]
      y[i+3] = A[i+3,j] * x[j]
      y[i+4] = A[i+4,j] * x[j]
      y[i+5] = A[i+5,j] * x[j]
      y[i+6] = A[i+6,j] * x[j]
      y[i+7] = A[i+7,j] * x[j]
    end
  end
  y
end

function timeit(n, reps)
  y = zeros(Float64, n)
  x = rand(Float64, n)
  A = rand(Float64, n, n)

  # row-wise y = Ax
  time = @elapsed for j in 1:reps
    y += matvec_r(A, x, n)
  end
  println("matvec_r             = ",time)

  # row-wise y = Ax (blocked, becomes effectively col-wise)
  time = @elapsed for j in 1:reps
      y += matvec_r2(A, x, n)
  end
  println("matvec_r (blocked)   = ",time)

# col-wise y = Ax (one loop)
  time = @elapsed for j in 1:reps
      y += matvec_c1(A, x, n)
  end
  println("matvec_c (1 loop)    = ",time)

# col-wise y = Ax (two loops)
  time = @elapsed for j in 1:reps
      y += matvec_c2(A, x, n)
  end
  println("matvec_c (2 loops)   = ",time)

  # col-wise y = Ax (simd)
  time = @elapsed for j in 1:reps
      y += matvec_c3(A, x, n)
  end
  println("matvec_c (simd)      = ",time)

  # col-wise y = Ax (unrolled)
  time = @elapsed for j in 1:reps
      y += matvec_c4(A, x, n)
  end
  println("matvec_c (unrolled)  = ",time)
end

const N  = 2^12   # number of elements of x, y, A is N*N
const CL = 8      # cache-line length in Float64
const BL = div(N,CL) # block-length for blocked algorithms

timeit(N, 10)
