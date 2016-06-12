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
