#############################################
#      basic permutations of ijk            #
#############################################

function matmult_ijk(A, B, n)
  C = zeros(eltype(A), n, n)
  for i = 1:n
    for j = 1:n
      for k = 1:n
        C[i,j] = A[i,k] * B[k,j]
      end
    end
  end
  C
end

function matmult_jik(A, B, n)
  C = zeros(eltype(A), n, n)
  for j = 1:n
    for i = 1:n
      for k = 1:n
        C[i,j] = C[i,j] + A[i,k] * B[k,j]
      end
    end
  end
  C
end

function matmult_kij(A, B, n)
  C = zeros(eltype(A), n, n)
  for k = 1:n
    for i = 1:n
      for j = 1:n
        C[i,j] = C[i,j] + A[i,k] * B[k,j]
      end
    end
  end
  C
end

function matmult_kji(A, B, n)
  C = zeros(eltype(A), n, n)
  for k = 1:n
    for j = 1:n
      for i = 1:n
        C[i,j] = C[i,j] + A[i,k] * B[k,j]
      end
    end
  end
  C
end

function matmult_ikj(A, B, n)
  C = zeros(eltype(A), n, n)
  for i = 1:n
    for k = 1:n
      for j = 1:n
        C[i,j] = C[i,j] + A[i,k] * B[k,j]
      end
    end
  end
  C
end

function matmult_jki(A, B, n)
  C = zeros(eltype(A), n, n)
  for j = 1:n
    for k = 1:n
      for i = 1:n
        C[i,j] = C[i,j] + A[i,k] * B[k,j]
      end
    end
  end
  C
end


################################################
#     matmult with blocking                    #
################################################

# blocking for vector processors
function matmult_jkib(A, B, n)
  C = zeros(eltype(A), n, n)
  for j = 1:n
    for l = 1:BL:n
      for k = 1:n
        for i = l:min(l + BL - 1, n)
          C[i,j] = C[i,j] + A[i,k] * B[k,j]
        end
      end
    end
  end
  C
end

# blocking for vector processors with actual simd instructions
function matmult_jkib2(A, B, n)
  C = zeros(eltype(A), n, n)
  for j = 1:n
    for l = 1:BL:n
      for k = 1:n
        @simd for i = l:min(l + BL - 1, n)
          C[i,j] = C[i,j] + A[i,k] * B[k,j]
        end
      end
    end
  end
  C
end

# 2d blocking for microprocessors jik with extra register for reduction
function matmult_jikb(A, B, n)
  C = zeros(eltype(A), n, n)
  for jj = 1:BL:n
    for ii = 1:BL:n
      for kk = 1:BL:n
        for j = jj:(jj+BL-1)
          for i = ii:(ii+BL-1)
            s = C[i,j]
            for k = kk:(kk+BL-1)
              s = s + A[i,k] * B[k,j]
            end
            C[i,j] = s
          end
        end
      end
    end
  end
  C
end

# 2d blocking for microprocessors jik without extra register for reduction
function matmult_jikb2(A, B, n)
  C = zeros(eltype(A), n, n)
  for jj = 1:BL:n
    for ii = 1:BL:n
      for kk = 1:BL:n
        for j = jj:(jj+BL-1)
          for i = ii:(ii+BL-1)
            for k = kk:(kk+BL-1)
              C[i,j] = C[i,j] + A[i,k] * B[k,j]
            end
          end
        end
      end
    end
  end
  C
end

# 2d blocking for microprocessors jik with extra register and unrolling
function matmult_jikb3(A, B, n)
  C = zeros(eltype(A), n, n)
  for jj = 1:BL:n
    for ii = 1:BL:n
      for kk = 1:BL:n
        for j = jj:2:(jj+BL-1)
          for i = ii:2:(ii+BL-1)
            s00 = C[i,j]
            s01 = C[i,j+1]
            s10 = C[i+1,j]
            s11 = C[i+1,j+1]
            for k = kk:(kk+BL-1)
              s00 = s00 + A[i,k] * B[k,j]
              s01 = s01 + A[i,k] * B[k,j+1]
              s10 = s10 + A[i+1,k] * B[k,j]
              s11 = s11 + A[i+1,k] * B[k,j+1]
            end
            C[i,j]     = s00
            C[i,j+1]   = s01
            C[i+1,j]   = s10
            C[i+1,j+1] = s11
          end
        end
      end
    end
  end
  C
end

function matmult_gemm(A, B, n)
  C = zeros(eltype(A), n, n)
  LinAlg.BLAS.gemm!(false, false, 1.0, A, B, false, C)
end

function timeit(n, reps)
  C = zeros(Float64, n, n)
  B = rand(Float64, n, n)
  A = rand(Float64, n, n)

  println("----------------------------------")
  println("        Timings for C = AB        ")
  println("        N=", n, ", reps=", reps)
  println("----------------------------------")

  # ijk
  time = @elapsed for j in 1:reps
    C += matmult_ijk(A, B, n)
  end
  println("matmult_ijk             | ",time/reps)

  time = @elapsed for j in 1:reps
    C += matmult_jik(A, B, n)
  end
  println("matmult_jik             | ",time/reps)

  time = @elapsed for j in 1:reps
    C += matmult_kij(A, B, n)
  end
  println("matmult_kij             | ",time/reps)

  time = @elapsed for j in 1:reps
    C += matmult_kji(A, B, n)
  end
  println("matmult_kji             | ",time/reps)

  time = @elapsed for j in 1:reps
    C += matmult_ikj(A, B, n)
  end
  println("matmult_ikj             | ",time/reps)

  time = @elapsed for j in 1:reps
    C += matmult_jki(A, B, n)
  end
  println("matmult_jki             | ",time/reps)

  time = @elapsed for j in 1:reps
    C += matmult_jkib(A, B, n)
  end
  println("matmult_jki (blocked)   | ",time/reps)

  time = @elapsed for j in 1:reps
    C += matmult_jkib2(A, B, n)
  end
  println("matmult_jki (bl, simd)  | ",time/reps)

  time = @elapsed for j in 1:reps
    C += matmult_jikb(A, B, n)
  end
  println("matmult_jik (2dbl, s)   | ",time/reps)

  time = @elapsed for j in 1:reps
    C += matmult_jikb2(A, B, n)
  end
  println("matmult_jik (2dbl)      | ",time/reps)

  time = @elapsed for j in 1:reps
    C += matmult_jikb3(A, B, n)
  end
  println("matmult_jik (2dbl+s+u)  | ",time/reps)
end

const N  = 2^9   # number of elements of x, y, A is N*N
const CL = 8      # cache-line length in Float64
const BL = div(N,CL) # block-length for blocked algorithms

timeit(N, 5)
