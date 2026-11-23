! knee_grid.f90  --  the posterior of the knee on a 3-D grid, from scratch (Lecture 12)
!
!   model     S(f) = sigma2 * [1 + (fk/f)**alpha]
!   data      one periodogram, simulated by inversion: P_j = -S_j * ln(u)
!   likelihood (Whittle)  -ln L = sum_j [ ln S_j + P_j/S_j ]
!   priors    flat in log10 sigma2, log10 fk, alpha  (inside the box)
!
! Compile & run:   gfortran -O2 knee_grid.f90 -o knee_grid && ./knee_grid
! Output:          marginal_log10fk.txt, marginal_alpha.txt  (plot them in Python)
program knee_grid
  implicit none
  integer,  parameter :: dp = kind(1.0d0)
  integer,  parameter :: n = 3199, nu = 51, nv = 161, na = 121
  real(dp), parameter :: T = 64.0_dp, ln10 = log(10.0_dp)
  real(dp) :: f(n), lf(n), P(n), S(n), r(n)
  real(dp) :: u(nu), v(nv), a(na), mv(nv), ma(na)
  real(dp), allocatable :: lp(:,:,:)
  real(dp) :: g, Asum, Bsum, s2, lpmax, mean_a, sd_a, mean_v, cum
  integer  :: i, j, k, l, nseed, t0, t1, rate
  integer, allocatable :: seed(:)

  ! ---- explicit seed: results are reproducible (same compiler)
  call random_seed(size=nseed)
  allocate(seed(nseed));  seed = 12
  call random_seed(put=seed)

  ! ---- frequencies and one simulated periodogram (truth: 1e-3, 1 Hz, 1)
  do l = 1, n
     f(l) = real(l, dp) / T
  end do
  lf = log(f)                                   ! once, outside the loops
  S  = 1.0e-3_dp * (1.0_dp + (1.0_dp / f)**1.0_dp)
  call random_number(r)                          ! r in [0,1)
  P  = -S * log(1.0_dp - r)                      ! 1-r in (0,1]: no log(0)

  ! ---- the grid axes
  do i = 1, nu
     u(i) = -3.25_dp + 0.5_dp * real(i-1, dp) / real(nu-1, dp)
  end do
  do j = 1, nv
     v(j) = -2.0_dp + (log10(50.0_dp) + 2.0_dp) * real(j-1, dp) / real(nv-1, dp)
  end do
  do k = 1, na
     a(k) = 3.0_dp * real(k-1, dp) / real(na-1, dp)
  end do

  allocate(lp(nu, nv, na))                       ! column major: u runs fastest
  call system_clock(t0, rate)
  do k = 1, na                                   ! alpha
     do j = 1, nv                                ! log10 fk
        Asum = 0.0_dp;  Bsum = 0.0_dp
        do l = 1, n                              ! frequency bins
           g    = 1.0_dp + exp(a(k) * (v(j)*ln10 - lf(l)))   ! no ** in the inner loop
           Asum = Asum + log(g)
           Bsum = Bsum + P(l) / g
        end do
        do i = 1, nu                             ! sigma2: no sum over bins
           s2 = 10.0_dp**u(i)
           lp(i, j, k) = -( n*log(s2) + Asum + Bsum/s2 )
        end do
     end do
  end do
  call system_clock(t1)

  ! ---- from ln p to p: subtract the maximum first (exp(-18600) = 0 in double!)
  lpmax = maxval(lp)
  lp = exp(lp - lpmax)
  lp = lp / sum(lp)                              ! lp now holds the normalised posterior

  ! ---- marginals: SUM with DIM, like numpy sum(axis=...)
  mv = sum(sum(lp, dim=3), dim=1)                ! over alpha, then sigma2 -> log10 fk
  ma = sum(sum(lp, dim=2), dim=1)                ! over fk, then sigma2     -> alpha

  mean_a = sum(a * ma);  sd_a = sqrt(sum((a - mean_a)**2 * ma))
  mean_v = sum(v * mv)
  print '(a,i0,a,f6.3,a)', 'grid of ', size(lp), ' points in ', real(t1-t0,dp)/rate, ' s'
  print '(a,f7.4,a,f6.4)', 'alpha    : mean ', mean_a, '  std ', sd_a
  print '(a,f7.4,a,f6.3,a)', 'log10 fk : mean ', mean_v, '  -> fk ~ ', 10.0_dp**mean_v, ' Hz'

  cum = 0.0_dp                                   ! median of log10 fk from the cumulative sum
  do j = 1, nv
     cum = cum + mv(j)
     if (cum >= 0.5_dp) then
        print '(a,f7.4)', 'log10 fk : median ', v(j)
        exit
     end if
  end do

  open(10, file='marginal_log10fk.txt', status='replace')
  do j = 1, nv
     write(10, '(2es16.7)') v(j), mv(j)
  end do
  close(10)
  open(11, file='marginal_alpha.txt', status='replace')
  do k = 1, na
     write(11, '(2es16.7)') a(k), ma(k)
  end do
  close(11)
  print *, 'wrote marginal_log10fk.txt and marginal_alpha.txt'
end program knee_grid
