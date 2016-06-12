###########################################
#         column oriented                 #
###########################################

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
        y[i] = y[i] + A[i,j] * x[j]
      end
    end
  end
  y
end

# rowwise y = Ax, blocked, unrolled
function matvec_r3(A, x, n)
  y = zeros(eltype(x), n)
  for ii = 1:BL:n
    for j = 1:n
      for i = ii:8:(ii+BL-1)
        y[i] = y[i] + A[i,j] * x[j]
        y[i+1] = y[i+1] + A[i+1,j] * x[j]
        y[i+2] = y[i+2] + A[i+2,j] * x[j]
        y[i+3] = y[i+3] + A[i+3,j] * x[j]
        y[i+4] = y[i+4] + A[i+4,j] * x[j]
        y[i+5] = y[i+5] + A[i+5,j] * x[j]
        y[i+6] = y[i+6] + A[i+6,j] * x[j]
        y[i+7] = y[i+7] + A[i+7,j] * x[j]
      end
    end
  end
  y
end

# rowwise, 2d blocked
function matvec_r4(A, x, n)
  y = zeros(eltype(x), n)
  for ii = 1:BL:n
    for jj = 1:BL:n
      for i = ii:(ii+BL-1)
        for j = jj:(jj+BL-1)
          y[i] = y[i] + A[i,j] * x[j]
        end
      end
    end
  end
  y
end

# rowwise, parallel
function matvec_rp(A, x, n)
  y = zeros(eltype(x), n)
  @parallel for i = 1:n
    for j = 1:n
      y[i] = y[i] + A[i,j] * x[j]
    end
  end
  y
end

# rowwise y = Ax, blocked
function matvec_rp2(A, x, n)
  y = zeros(eltype(x), n)
  @parallel for ii = 1:BL:n
    for j = 1:n
      for i = ii:(ii+BL-1)
        y[i] = y[i] + A[i,j] * x[j]
      end
    end
  end
  y
end

###########################################
#         column oriented                 #
###########################################

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
      y[i] = y[i] + A[i,j] * x[j]
    end
  end
  y
end

# columnwise y = Ax, vectorized
function matvec_c3(A, x, n)
  y = zeros(eltype(x), n)
  for j = 1:n
    @simd for i = 1:n
      y[i] = y[i] + A[i,j] * x[j]
    end
  end
  y
end

# columnwise y = Ax, unrolled
function matvec_c4(A, x, n)
  y = zeros(eltype(x), n)
  for j = 1:n
    for i = 1:8:n
      y[i] = y[i] + A[i,j] * x[j]
      y[i+1] = y[i+1] + A[i+1,j] * x[j]
      y[i+2] = y[i+2] + A[i+2,j] * x[j]
      y[i+3] = y[i+3] + A[i+3,j] * x[j]
      y[i+4] = y[i+4] + A[i+4,j] * x[j]
      y[i+5] = y[i+5] + A[i+5,j] * x[j]
      y[i+6] = y[i+6] + A[i+6,j] * x[j]
      y[i+7] = y[i+7] + A[i+7,j] * x[j]
    end
  end
  y
end

# colwise, 2d blocked
function matvec_c5(A, x, n)
  y = zeros(eltype(x), n)
  for jj = 1:BL:n
    for ii = 1:BL:n
      for j = jj:(jj+BL-1)
        for i = ii:(ii+BL-1)
          y[i] = y[i] + A[i,j] * x[j]
        end
      end
    end
  end
  y
end

# colwise, 2d blocked, unrolled
function matvec_c6(A, x, n)
  y = zeros(eltype(x), n)
  for jj = 1:BL:n
    for ii = 1:BL:n
      for j = jj:(jj+BL-1)
        for i = ii:8:(ii+BL-1)
          y[i] = y[i] + A[i,j] * x[j]
          y[i+1] = y[i+1] + A[i+1,j] * x[j]
          y[i+2] = y[i+2] + A[i+2,j] * x[j]
          y[i+3] = y[i+3] + A[i+3,j] * x[j]
          y[i+4] = y[i+4] + A[i+4,j] * x[j]
          y[i+5] = y[i+5] + A[i+5,j] * x[j]
          y[i+6] = y[i+6] + A[i+6,j] * x[j]
          y[i+7] = y[i+7] + A[i+7,j] * x[j]
        end
      end
    end
  end
  y
end

# columnwise y = Ax parallel
function matvec_cp(A, x, n)
  y = zeros(eltype(x), n)
  @parallel for jj = 1:BL:n
    for ii = 1:BL:n
      for j = jj:(jj+BL-1)
        for i = ii:(ii+BL-1)
          y[i] = y[i] + A[i,j] * x[j]
        end
      end
    end
  end
  y
end

# columnwise parallel with reduce
function matvec_cp2(A, x, n)
  y = zeros(eltype(x), n)
  y = @parallel (+) for j = 1:n
    z = zeros(eltype(x), n)
    for i = 1:n
      z[i] = A[i,j] * x[j]
    end
    z
  end
  y
end

################################################
#          sparse matrices                     #
################################################

# tridiagonal sequential
function matvec_tr(A, x, n)
  y = zeros(eltype(x), n)
  y[1] = A[1,1] * x[1] + A[1,2] * x[2]
  for i = 2:n-1
    y[i] = A[i, i-1] * x[i-1] + A[i, i] * x[i] + A[i, i+1] * x[i+1]
  end
  y[n] = A[n, n-1] * x[n-1] + A[n,n] * x[n]
  y
end

# tridiagonal with branches
function matvec_tr2(A, x, n)
  y = zeros(eltype(x), n)
  for i = 1:n
    if (i == 1)
      y[1] = A[1,1] * x[1] + A[1,2]
    elseif (i == n)
      y[n] = A[n, n-1] * x[n-1] + A[n,n] * x[n]
    else
      y[i] = A[i, i-1] * x[i-1] + A[i, i] * x[i] + A[i, i+1] * x[i+1]
    end
  end
  y
end

# tridiagonal parallelized with branches
function matvec_trp(A, x, n)
  y = zeros(eltype(x), n)
  @parallel for i = 1:n
    if (i == 1)
      y[1] = A[1,1] * x[1] + A[1,2]
    elseif (i == n)
      y[n] = A[n, n-1] * x[n-1] + A[n,n] * x[n]
    else
      y[i] = A[i, i-1] * x[i-1] + A[i, i] * x[i] + A[i, i+1] * x[i+1]
    end
  end
  y
end


function timeit(n, reps)
  y = zeros(Float64, n)
  x = rand(Float64, n)
  A = rand(Float64, n, n)

  println("----------------------------------")
  println("        Timings for y = Ax        ")
  println("        N=", n, ", reps=", reps)
  println("----------------------------------")

  # row-wise y = Ax
  time = @elapsed for j in 1:reps
    y += matvec_r(A, x, n)
  end
  println("matvec_r             | ",time/reps)

  # row-wise y = Ax (blocked, becomes effectively col-wise)
  time = @elapsed for j in 1:reps
      y += matvec_r2(A, x, n)
  end
  println("matvec_r (blocked)   | ",time/reps)

  # row-wise y = Ax (blocked, unrolled)
  time = @elapsed for j in 1:reps
      y += matvec_r3(A, x, n)
  end
  println("matvec_r (bl, unr)   | ",time/reps)

  # row-wise y = Ax (blocked 2d)
  time = @elapsed for j in 1:reps
      y += matvec_r4(A, x, n)
  end
  println("matvec_r (bl, 2d)    | ",time/reps)

  # row-wise y = Ax (parallel)
  time = @elapsed for j in 1:reps
      y += matvec_rp(A, x, n)
  end
  println("matvec_r (par)       | ",time/reps)

  # row-wise y = Ax (blocked, parallel)
  time = @elapsed for j in 1:reps
      y += matvec_rp2(A, x, n)
  end
  println("matvec_r (bl, par)   | ",time/reps)

# col-wise y = Ax (one loop)
  time = @elapsed for j in 1:reps
      y += matvec_c1(A, x, n)
  end
  println("matvec_c (1 loop)    | ",time/reps)

# col-wise y = Ax (two loops)
  time = @elapsed for j in 1:reps
      y += matvec_c2(A, x, n)
  end
  println("matvec_c (2 loops)   | ",time/reps)

  # col-wise y = Ax (simd)
  time = @elapsed for j in 1:reps
      y += matvec_c3(A, x, n)
  end
  println("matvec_c (simd)      | ",time/reps)

  # col-wise y = Ax (unrolled)
  time = @elapsed for j in 1:reps
      y += matvec_c4(A, x, n)
  end
  println("matvec_c (unrolled)  | ",time/reps)

  # col-wise y = Ax (blocked 2d)
  time = @elapsed for j in 1:reps
      y += matvec_c5(A, x, n)
  end
  println("matvec_c (bl, 2d)    | ",time/reps)

  # col-wise y = Ax (blocked 2d, unrolled)
  time = @elapsed for j in 1:reps
      y += matvec_c6(A, x, n)
  end
  println("matvec_c (bl, 2d, un)| ",time/reps)

  # col-wise y = Ax (blocked 2d, parallel)
  time = @elapsed for j in 1:reps
      y += matvec_cp(A, x, n)
  end
  println("matvec_c (par, 2db)  | ",time/reps)

  # col-wise y = Ax (parallel)
  time = @elapsed for j in 1:reps
      y += matvec_cp2(A, x, n)
  end
  println("matvec_c (par)       | ",time/reps)

  # tridiagonal
  time = @elapsed for j in 1:reps
      y += matvec_tr(A, x, n)
  end
  println("matvec_tr            | ",time/reps)

  # tridiagonal ifelse
  time = @elapsed for j in 1:reps
      y += matvec_tr2(A, x, n)
  end
  println("matvec_tr (branch)   | ",time/reps)

  # tridiagonal
  time = @elapsed for j in 1:reps
      y += matvec_trp(A, x, n)
  end
  println("matvec_tr (parallel) | ",time/reps)
end

const N  = 2^12   # number of elements of x, y, A is N*N
const CL = 8      # cache-line length in Float64
const BL = div(N,CL) # block-length for blocked algorithms

timeit(N, 25)
