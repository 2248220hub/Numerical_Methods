! overlap_save.f90  --  Lecture 9: overlap-save convolution with FFTW and OpenMP
!   any kernel (h(1) is lag -m_minus), blocks processed in parallel: each block writes a
!   disjoint slice of the output, so there is no race condition.
!   The driver compares with a direct double loop and times the two.
! Build:  gfortran -O2 -fopenmp overlap_save.f90 -I/usr/include -lfftw3 -o ols && ./ols
module ols_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: dp => real64
  implicit none
  include 'fftw3.f03'
contains
  subroutine overlap_save(x, h, nfft, m_minus, y)
    real(dp), intent(in)  :: x(:), h(:)
    integer,  intent(in)  :: nfft, m_minus
    real(dp), intent(out) :: y(:)                     ! size(x) + size(h) - 1
    real(dp), allocatable :: xp(:), buf(:)
    complex(dp), allocatable :: Hk(:), Xk(:)
    type(c_ptr) :: pf, pb
    integer :: M, B, Ny, nblk, j, s, n, m_plus
    M = size(h); B = nfft - M + 1; Ny = size(x) + M - 1; m_plus = M - 1 - m_minus
    nblk = (Ny + B - 1)/B
    allocate(xp(nblk*B + M - 1 + nfft), buf(nfft), Hk(nfft/2+1), Xk(nfft/2+1))
    xp = 0.0_dp
    xp(M:M+size(x)-1) = x                             ! M-1 leading zeros
    ! plans created ONCE, outside the parallel region (planning is not thread-safe)
    pf = fftw_plan_dft_r2c_1d(nfft, buf, Xk, ior(FFTW_ESTIMATE, FFTW_UNALIGNED))
    pb = fftw_plan_dft_c2r_1d(nfft, Xk, buf, ior(FFTW_ESTIMATE, FFTW_UNALIGNED))
    buf = 0.0_dp; buf(1:m_plus+1) = h(m_minus+1:M)    ! lags 0..m_plus
    if (m_minus > 0) buf(nfft-m_minus+1:nfft) = h(1:m_minus)   ! lags -m_minus..-1
    call fftw_execute_dft_r2c(pf, buf, Hk)             ! kernel spectrum, once
    !$omp parallel do default(none) private(j, s, n, buf, Xk) &
    !$omp& shared(nblk, B, nfft, Ny, m_plus, xp, Hk, y, pf, pb)
    do j = 1, nblk
       s = (j-1)*B + 1                                 ! 1-based block start
       buf = xp(s:s+nfft-1)
       call fftw_execute_dft_r2c(pf, buf, Xk)           ! new-array execute: thread-safe
       Xk = Xk*Hk/nfft                                  ! FFTW is unnormalised
       call fftw_execute_dft_c2r(pb, Xk, buf)
       n = min(B, Ny - s + 1)
       y(s:s+n-1) = buf(m_plus+1:m_plus+n)              ! discard m_plus / m_minus
    end do
    !$omp end parallel do
    call fftw_destroy_plan(pf); call fftw_destroy_plan(pb)
  end subroutine overlap_save
end module ols_mod

program ols
  use ols_mod
  !$ use omp_lib
  implicit none
  integer, parameter :: nx = 2000000, M = 101
  real(dp), allocatable :: x(:), h(:), y(:), yd(:)
  integer :: i, k, mm, t0, t1, rate
  allocate(x(nx), h(M), y(nx+M-1), yd(nx+M-1))
  call random_number(h); h = h - 0.5_dp
  call random_number(x); x = x - 0.5_dp
  ! direct convolution: y(i) = sum_k h(k) x(i-k+1)   (O(N M))
  call system_clock(t0, rate)
  yd = 0.0_dp
  do i = 1, nx + M - 1
     do k = max(1, i-nx+1), min(M, i)
        yd(i) = yd(i) + h(k)*x(i-k+1)
     end do
  end do
  call system_clock(t1)
  print '(a, f8.1, a)', 'direct convolution         ', 1000.0_dp*(t1-t0)/rate, ' ms'
  do mm = 0, M - 1, 50                               ! causal (0), centred (50), anti-causal (100)
     call system_clock(t0)
     call overlap_save(x, h, 1024, mm, y)
     call system_clock(t1)
     ! output indices as np.convolve(x, h) for every m_minus (m_minus only labels the lags of h)
     print '(a, i3, a, es9.2, a, f7.1, a)', 'overlap-save m_minus =', mm, ':  max error ', &
          maxval(abs(y - yd)), ',  ', 1000.0_dp*(t1-t0)/rate, ' ms'
  end do
  !$ print '(a, i0)', 'OpenMP threads: ', omp_get_max_threads()
end program ols
