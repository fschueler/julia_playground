----------------------------------
        Timings for y = Ax        
        N=4096, reps=25
----------------------------------

algorithm            | time (s)
---------------------|------------------------
matvec_r             | 0.45569259144
matvec_r (blocked)   | 0.04996203948
matvec_r (bl, unr)   | 0.04706911852
matvec_r (bl, 2d)    | 0.49691900003999995
matvec_r (par)       | 2.034732e-5
matvec_r (bl, par)   | 2.278896e-5
matvec_c (1 loop)    | 0.13152843244
matvec_c (2 loops)   | 0.0520760824
matvec_c (simd)      | 0.04128421492
matvec_c (unrolled)  | 0.045467309399999996
matvec_c (bl, 2d)    | 0.05895978988
matvec_c (bl, 2d, un)| 0.07160092343999999
matvec_c (par, 2db)  | 2.1020959999999998e-5
matvec_c (par)       | 0.11783382964
matvec_tr            | 8.5216e-5
matvec_tr (branch)   | 5.158172e-5
matvec_tr (parallel) | 1.478144e-5

----------------------------------
        Timings for C = AB        
        N=512, reps=5
----------------------------------
algorithm               | time (s)
------------------------|------------------------
matmult_ijk             | 0.535462381
matmult_jik             | 0.6823360774
matmult_kij             | 1.0624259249999999
matmult_kji             | 0.33370950320000003
matmult_ikj             | 1.3030522402
matmult_jki             | 0.320226846
matmult_jki (blocked)   | 0.4504516938
matmult_jki (bl, simd)  | 0.5144411973999999
matmult_jik (2dbl, s)   | 0.2783084628
matmult_jik (2dbl)      | 0.485824767
matmult_jik (2dbl+s+u)  | 0.1493436022
