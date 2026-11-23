! mc_fortran.f90  --  Lecture 13 from scratch in Fortran
!   1. sample mean of x^2 on [0,2] with its error bar
!   2. hit-or-miss estimate of pi
!   3. inversion (exponential), Box-Muller (Gaussian), rejection (half-normal)
!   4. importance sampling of the rare event P(X > 4)
! Compile & run:  gfortran -O2 mc_fortran.f90 -o mc_fortran && ./mc_fortran
program mc_fortran
  implicit none
  integer,  parameter :: dp = kind(1.0d0), n = 1000000, m = 10000
  real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)
  real(dp), allocatable :: x(:), y(:), u(:), v(:), z(:), fx(:), w(:)
  real(dp) :: est, err, p_hat, Mconst, lam
  integer  :: k, nacc, nseed, t0, t1, rate
  integer, allocatable :: seed(:)

  call random_seed(size=nseed);  allocate(seed(nseed));  seed = 13
  call random_seed(put=seed)                      ! explicit seed
  allocate(x(n), y(n), u(n), v(n), z(n), fx(n))

  ! ---- 1. sample mean of x^2 on [0,2], 1e4 points
  call random_number(x(1:m))                      ! [0,1)
  x(1:m) = 2.0_dp*x(1:m)                          ! [0,2)
  fx(1:m) = x(1:m)**2
  est = 2.0_dp*sum(fx(1:m))/m
  err = 2.0_dp*sqrt(sum((fx(1:m) - sum(fx(1:m))/m)**2)/(m - 1))/sqrt(real(m, dp))
  print '(a,f7.4,a,f7.4,a)', '1. int x^2 dx on [0,2] = ', est, ' +- ', err, '   (8/3 = 2.6667)'

  ! ---- 2. hit-or-miss: pi
  call random_number(x);  call random_number(y)
  p_hat = count(x**2 + y**2 < 1.0_dp) / real(n, dp)
  print '(a,f8.5,a,f8.5)', '2. pi = ', 4*p_hat, ' +- ', 4*sqrt(p_hat*(1 - p_hat)/n)

  ! ---- 3a. inversion: Exp(lambda = 2), whole-array operation, no loop
  call random_number(u)
  z = -log(1.0_dp - u)/2.0_dp
  print '(a,f7.4,a)', '3a. Exp(2): mean ', sum(z)/n, '   (0.5)'

  ! ---- 3b. Box-Muller: two uniforms -> two N(0,1)
  call system_clock(t0, rate)
  call random_number(u(1:n/2));  call random_number(v(1:n/2))
  u(1:n/2)   = 1.0_dp - u(1:n/2)                  ! (0,1]: no LOG(0)
  z(1:n/2)   = sqrt(-2*log(u(1:n/2)))*cos(2*pi*v(1:n/2))
  z(n/2+1:n) = sqrt(-2*log(u(1:n/2)))*sin(2*pi*v(1:n/2))
  call system_clock(t1)
  print '(a,f8.4,a,f7.4,a,f6.1,a)', '3b. N(0,1): mean ', sum(z)/n, ', variance ', &
        sum(z**2)/n - (sum(z)/n)**2, '  (', 1000.0_dp*(t1 - t0)/rate, ' ms for 1e6)'

  ! ---- 3c. rejection: half-normal from Exp(1), M = sqrt(2e/pi)
  Mconst = sqrt(2*exp(1.0_dp)/pi)
  call random_number(u);  call random_number(v)
  z = -log(1.0_dp - u)                            ! z ~ Exp(1)   (the envelope g)
  nacc = 0;  est = 0.0_dp
  do k = 1, n
     if (v(k) < exp(-0.5_dp*(z(k) - 1.0_dp)**2)) then   ! = f/(M g) for this pair
        nacc = nacc + 1                           ! accepted: z(k) is half-normal
        est = est + z(k)
     end if
  end do
  print '(a,f7.4,a,f7.4,a,f7.4,a)', '3c. rejection: accepted ', real(nacc, dp)/n, &
        ' (1/M = ', 1/Mconst, '), mean ', est/nacc, ' (0.7979)'

  ! ---- 4. importance sampling: P(X>4), proposal g = 4 + Exp(rate 4)
  lam = 4.0_dp
  allocate(w(m))
  call random_number(u(1:m))
  x(1:m) = 4.0_dp - log(1.0_dp - u(1:m))/lam      ! samples from g (inversion!)
  w = exp(-0.5_dp*x(1:m)**2)/sqrt(2*pi) / (lam*exp(-lam*(x(1:m) - 4.0_dp)))   ! f/g
  est = sum(w)/m
  err = sqrt(sum((w - est)**2)/(m - 1))/sqrt(real(m, dp))
  print '(a,es11.4,a,es9.2,a)', '4. P(X>4) = ', est, ' +- ', err, '   (exact 3.1671E-05)'
end program mc_fortran
