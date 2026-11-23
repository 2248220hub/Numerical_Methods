! fft_radix2.f90  --  Lecture 8: an in-place iterative radix-2 FFT from scratch
!   bit-reversal permutation, log2(N) stages of butterflies, precomputed twiddles;
!   checked against the direct DFT, inverse round trip, timing O(N log N) vs O(N^2)
! Compile & run:  gfortran -O2 fft_radix2.f90 -o fft_radix2 && ./fft_radix2
module fft_mod
  implicit none
  integer,  parameter :: dp = kind(1.0d0)
  real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)
contains
  subroutine fft_inplace(a, sgn)
    ! a(1:npts), npts = 2**q; sgn = -1 forward, +1 inverse (unnormalised)
    complex(dp), intent(inout) :: a(:)
    integer, intent(in) :: sgn
    integer :: npts, q, i, j, m, k, half
    integer, allocatable :: rev(:)
    complex(dp), allocatable :: w(:)
    complex(dp) :: t, u
    npts = size(a)
    q = nint(log(real(npts, dp))/log(2.0_dp))
    if (2**q /= npts) stop 'fft_inplace: N must be a power of two'
    ! 1. bit-reversal table: rev(i) = (rev(i/2)/2) | ((i&1) << (q-1)), 0-based
    allocate(rev(0:npts-1));  rev(0) = 0
    do i = 1, npts - 1
       rev(i) = ior(ishft(rev(ishft(i, -1)), -1), ishft(iand(i, 1), q - 1))
    end do
    do i = 0, npts - 1
       if (i < rev(i)) then                       ! swap each pair once
          t = a(i+1); a(i+1) = a(rev(i)+1); a(rev(i)+1) = t
       end if
    end do
    ! 2. stages: sub-transforms of size m = 2, 4, ..., npts
    m = 2
    do while (m <= npts)
       half = m/2
       allocate(w(0:half-1))
       do j = 0, half - 1                          ! precomputed table: accurate twiddles
          w(j) = exp(cmplx(0.0_dp, sgn*2.0_dp*pi*j/m, dp))
       end do
       do k = 0, npts - 1, m                       ! block start
          do j = 0, half - 1                       ! butterflies of this block
             t = w(j)*a(k + j + half + 1)
             u = a(k + j + 1)
             a(k + j + 1)        = u + t
             a(k + j + half + 1) = u - t
          end do
       end do
       deallocate(w)
       m = 2*m
    end do
  end subroutine fft_inplace
end module fft_mod

program fft_radix2
  use fft_mod
  implicit none
  integer :: npts, j, k, t0, t1, rate, reps, r
  real(dp), allocatable :: re(:)
  complex(dp), allocatable :: a(:), b(:), dft(:)
  real(dp) :: tfft, err

  ! ---- correctness against the direct DFT (N = 256)
  npts = 256
  allocate(re(npts), a(npts), b(npts), dft(npts))
  call random_number(re);  a = cmplx(re - 0.5_dp, 0.0_dp, dp);  b = a
  do k = 0, npts - 1
     dft(k+1) = sum(b*exp(cmplx(0.0_dp, -2.0_dp*pi*real(mod(k*[(j, j = 0, npts-1)], npts), dp)/npts, dp)))
  end do
  call fft_inplace(a, -1)
  print '(a, es10.2)', 'max |FFT - direct DFT|        ', maxval(abs(a - dft))
  call fft_inplace(a, +1);  a = a/npts
  print '(a, es10.2)', 'inverse round trip max error  ', maxval(abs(a - b))
  deallocate(re, a, b, dft)

  ! ---- timing: N log N
  print '(a)', '        N   time per FFT [ms]   N log2 N / 1e6'
  npts = 1024
  do while (npts <= 2**20)
     allocate(re(npts), a(npts))
     call random_number(re);  a = cmplx(re, 0.0_dp, dp)
     reps = max(1, 2**20/npts)
     call system_clock(t0, rate)
     do r = 1, reps
        call fft_inplace(a, -1)
     end do
     call system_clock(t1)
     tfft = 1000.0_dp*(t1 - t0)/rate/reps
     err = npts*log(real(npts, dp))/log(2.0_dp)/1e6_dp
     print '(i9, f16.4, f17.3)', npts, tfft, err
     deallocate(re, a)
     npts = 4*npts
  end do
end program fft_radix2
