function matmult_r(A, B, n)
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

function timeit(n, reps)
  C = zeros(Float64, n, n)
  B = rand(Float64, n, n)
  A = rand(Float64, n, n)

  println("----------------------------------")
  println("        Timings for C = AB        ")
  println("        N=", n, ", reps=", reps)
  println("----------------------------------")

  # row-wise y = Ax
  time = @elapsed for j in 1:reps
    C += matmult_r(A, B, n)
  end
  println("matmult_r             | ",time/reps)
end

const N  = 2^12   # number of elements of x, y, A is N*N
const CL = 8      # cache-line length in Float64
const BL = div(N,CL) # block-length for blocked algorithms

timeit(N, 25)
