####Timings for y = Ax
**N=4096, reps=10**

  algorithm          | @elapsed time 
---------------------|---------------
matvec_r             | 6.272438594
matvec_r (blocked)   | 0.666119677
matvec_r (bl, unr)   | 0.618088545
matvec_r (bl, 2d)    | 6.659586025
matvec_r (par)       | 0.007116445
matvec_r (bl, par)   | 0.007820455
matvec_c (1 loop)    | 1.644524024
matvec_c (2 loops)   | 0.556068403
matvec_c (simd)      | 0.49281822
matvec_c (unrolled)  | 0.462399372
matvec_c (bl, 2d)    | 0.746822875
matvec_c (par, 2db)  | 0.003047827
