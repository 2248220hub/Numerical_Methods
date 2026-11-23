! welch_omp.f90  --  Lecture 10: Welch's method in Fortran with FFTW (r2c) and OpenMP
!   white Gaussian noise of variance sigma^2 (Box-Muller), periodic Hann window, 50% overlap,
!   mean removed per segment; one-sided PSD in units^2/Hz. Checks:
!     * E[P] = 2 sigma^2 / fs for white noise, whatever the window (the U normalisation)
!     * relative scatter of the estimate ~ 1/sqrt(K)
!     * same result with 1 and with several threads
! Build:  gfortran -O2 -fopenmp welch_omp.f90 -I/usr/include -lfftw3 -o welch_omp && ./welch_omp
module welch_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: dp => real64
  implicit none
  include 'fftw3.f03'
  real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)
contains
  subroutine welch(x, fs, L, P)
    real(dp), intent(in)  :: x(:), fs
    integer,  intent(in)  :: L
    real(dp), intent(out) :: P(:)                    ! L/2 + 1 bins
    real(dp), allocatable :: w(:), seg(:)
    complex(dp), allocatable :: Xk(:)
    type(c_ptr) :: plan
    real(dp) :: U
    integer :: Kseg, D, i, s, n
    D = L/2
    Kseg = (size(x) - L)/D + 1
    allocate(w(L), seg(L), Xk(L/2+1))
    w = [(0.5_dp - 0.5_dp*cos(2.0_dp*pi*n/L), n = 0, L-1)]   ! PERIODIC Hann (denominator L)
    U = sum(w**2)/L                                          ! mean power of the taper
    plan = fftw_plan_dft_r2c_1d(L, seg, Xk, ior(FFTW_ESTIMATE, FFTW_UNALIGNED))   ! once, outside
    P = 0.0_dp                                               ! must be initialised
    !$omp parallel do default(none) private(i, s, seg, Xk) shared(x, w, Kseg, D, L, plan) reduction(+:P)
    do i = 1, Kseg
       s = (i-1)*D + 1
       seg = x(s:s+L-1)
       seg = w*(seg - sum(seg)/L)                            ! remove the mean, taper
       call fftw_execute_dft_r2c(plan, seg, Xk)              ! new-array execute: thread-safe
       P = P + abs(Xk)**2
    end do
    !$omp end parallel do
    P = P/(real(Kseg, dp)*L*U*fs)
    P(2:L/2) = 2.0_dp*P(2:L/2)                               ! one-sided: double all but DC, Nyquist
    call fftw_destroy_plan(plan)
  end subroutine welch
end module welch_mod

program welch_omp
  use welch_mod
  !$ use omp_lib
  implicit none
  integer, parameter :: n = 2**22, L = 1024
  real(dp), parameter :: fs = 100.0_dp, sigma = 0.7_dp
  real(dp), allocatable :: x(:), u1(:), u2(:)
  real(dp) :: P(L/2+1), P1(L/2+1), mean_ratio, scatter
  integer :: Kseg, t0, t1, rate, nth

  allocate(x(n), u1(n/2), u2(n/2))
  call random_number(u1); call random_number(u2); u1 = 1.0_dp - u1        ! (0,1]: no log(0)
  x(1:n/2)   = sigma*sqrt(-2*log(u1))*cos(2*pi*u2)                       ! Box-Muller
  x(n/2+1:n) = sigma*sqrt(-2*log(u1))*sin(2*pi*u2)

  nth = 1
  !$ nth = omp_get_max_threads()
  !$ call omp_set_num_threads(1)
  call system_clock(t0, rate); call welch(x, fs, L, P1); call system_clock(t1)
  print '(a, f8.1, a)', '1 thread : ', 1000.0_dp*(t1-t0)/rate, ' ms'
  !$ call omp_set_num_threads(nth)
  call system_clock(t0); call welch(x, fs, L, P); call system_clock(t1)
  print '(a, i0, a, f8.1, a, es9.2)', 'threads = ', nth, ': ', 1000.0_dp*(t1-t0)/rate, &
        ' ms,  max relative difference ', maxval(abs(P - P1)/P1)

  Kseg = (n - L)/(L/2) + 1
  mean_ratio = sum(P(2:L/2))/(L/2 - 1)/(2*sigma**2/fs)
  scatter = sqrt(sum((P(2:L/2)/(2*sigma**2/fs) - mean_ratio)**2)/(L/2 - 2))
  print '(a, i0)',  'segments K = ', Kseg
  print '(a, f8.4, a)', 'mean of P / (2 sigma^2/fs)   = ', mean_ratio, '   (1 expected)'
  print '(a, f8.4, a, f8.4)', 'relative scatter of P        = ', scatter, '   1/sqrt(K) = ', 1.0_dp/sqrt(real(Kseg, dp))
end program welch_omp
