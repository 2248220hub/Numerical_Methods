! bayes_basics.f90  --  the small examples of Lecture 12 in Fortran
!   1. the two urns, updated draw by draw
!   2. Gaussian likelihood x Gaussian prior
!   3. bias of the maximum-likelihood variance (needs Gaussian deviates: Box-Muller)
! Compile & run:  gfortran -O2 bayes_basics.f90 -o bayes_basics && ./bayes_basics
module stats_tools
  implicit none
  integer,  parameter :: dp = kind(1.0d0)
  real(dp), parameter :: pi = 4.0_dp * atan(1.0_dp)
contains
  subroutine normal_deviates(z)
    ! Box-Muller: two uniforms -> two independent N(0,1)  (Lecture 13)
    real(dp), intent(out) :: z(:)
    real(dp), allocatable :: u1(:), u2(:)
    integer :: n, m
    n = size(z);  m = (n + 1) / 2
    allocate(u1(m), u2(m))
    call random_number(u1);  call random_number(u2)
    u1 = 1.0_dp - u1                                 ! (0,1]: no log(0)
    z(1:m)   = sqrt(-2.0_dp*log(u1)) * cos(2.0_dp*pi*u2)
    z(m+1:n) = sqrt(-2.0_dp*log(u1(1:n-m))) * sin(2.0_dp*pi*u2(1:n-m))
  end subroutine normal_deviates
end module stats_tools

program bayes_basics
  use stats_tools
  implicit none
  real(dp) :: post(2), like(2), evidence
  real(dp) :: wL, w0, s_post, mu_post
  real(dp) :: x(5), xbar, acc_ml, acc_un
  character(len=2) :: draws = 'BB'
  integer :: i, k, nseed
  integer, parameter :: nsets = 200000, N = 5
  integer, allocatable :: seed(:)

  ! ---- 1. urns: P(blue | A1) = 0.3, P(blue | A2) = 0.6, prior 1/2 each
  post = [0.5_dp, 0.5_dp]
  do i = 1, len(draws)
     if (draws(i:i) == 'B') then
        like = [0.3_dp, 0.6_dp]
     else
        like = [0.7_dp, 0.4_dp]
     end if
     evidence = sum(like * post)                     ! total probability
     post = like * post / evidence                   ! Bayes
     print '(a,a,a,f7.4)', 'after ', draws(i:i), ':  P(A1|data) = ', post(1)
  end do

  ! ---- 2. Gaussian update: precisions add
  wL = 1.0_dp / 0.12_dp**2;  w0 = 1.0_dp / 0.10_dp**2
  s_post  = 1.0_dp / sqrt(wL + w0)
  mu_post = s_post**2 * (wL*0.82_dp + w0*1.00_dp)
  print '(a,f6.3,a,f6.3)', 'alpha posterior: ', mu_post, ' +- ', s_post

  ! ---- 3. MLE variance bias, sigma2 = 4, N = 5
  call random_seed(size=nseed);  allocate(seed(nseed));  seed = 19
  call random_seed(put=seed)
  acc_ml = 0.0_dp;  acc_un = 0.0_dp
  do k = 1, nsets
     call normal_deviates(x)
     x = 2.0_dp * x                                  ! sigma = 2
     xbar = sum(x) / N
     acc_ml = acc_ml + sum((x - xbar)**2) / N
     acc_un = acc_un + sum((x - xbar)**2) / (N - 1)
  end do
  print '(a,f6.3,a,f6.3,a)', '<s2_ML> = ', acc_ml/nsets, '   <s2_unbiased> = ', acc_un/nsets, &
        '   (theory 3.200 and 4.000)'
end program bayes_basics
