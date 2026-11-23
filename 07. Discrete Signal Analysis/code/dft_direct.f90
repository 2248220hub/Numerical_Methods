! dft_direct.f90  --  Lecture 7 in Fortran
!   1. B-bit quantisation of a sinusoid: measured SQNR against 6.02 B + 1.76 dB
!   2. the direct O(N^2) DFT and its inverse: round trip, Parseval, a tone on its bin
!   3. timing: doubling N multiplies the time by ~4
! Compile & run:  gfortran -O2 dft_direct.f90 -o dft_direct && ./dft_direct
module dft_mod
  implicit none
  integer,  parameter :: dp = kind(1.0d0)
  real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)
contains
  subroutine dft(xin, xout, sgn)
    ! xout(k) = sum_j xin(j) exp(sgn * 2 pi i k j / npts),  k, j = 0..npts-1 (stored 1..npts)
    ! (Fortran is case-insensitive: x and X would be the SAME name)
    complex(dp), intent(in)  :: xin(:)
    complex(dp), intent(out) :: xout(:)
    integer, intent(in) :: sgn
    integer :: k, j, npts
    complex(dp) :: w
    npts = size(xin)
    do k = 0, npts - 1
       xout(k+1) = (0.0_dp, 0.0_dp)
       do j = 0, npts - 1
          ! mod(k*j, npts): the phase argument stays small, hence accurate
          w = exp(cmplx(0.0_dp, sgn*2.0_dp*pi*real(mod(k*j, npts), dp)/npts, dp))
          xout(k+1) = xout(k+1) + xin(j+1)*w
       end do
    end do
  end subroutine dft

  pure elemental real(dp) function quantise(v, delta, B)
    ! mid-tread quantiser with saturation to the B-bit two's-complement range
    real(dp), intent(in) :: v, delta
    integer, intent(in) :: B
    real(dp) :: code
    code = floor(v/delta + 0.5_dp)
    code = max(-2.0_dp**(B-1), min(2.0_dp**(B-1) - 1.0_dp, code))
    quantise = delta*code
  end function quantise
end module dft_mod

program dft_direct
  use dft_mod
  implicit none
  integer, parameter :: ns = 200000
  real(dp) :: t(ns), x(ns), xq(ns), delta, sqnr
  complex(dp), allocatable :: z(:), Z1(:), z2(:)
  integer :: B, i, N, kmax(1), t0, t1, rate
  real(dp) :: r(4096)

  ! ---- 1. quantisation
  do i = 1, ns
     t(i) = real(i-1, dp)/ns
  end do
  x = 2.5_dp*0.9999_dp*sin(2.0_dp*pi*7.123_dp*t)
  print '(a)', '  B   SQNR measured   6.02B+1.76'
  do B = 4, 16, 4
     delta = 5.0_dp/2.0_dp**B
     xq = quantise(x, delta, B)
     sqnr = 10.0_dp*log10(variance(x)/variance(x - xq))
     print '(i3, f15.2, f13.2)', B, sqnr, 6.02_dp*B + 1.76_dp
  end do

  ! ---- 2. DFT round trip, Parseval, a tone on bin 50
  N = 256
  allocate(z(N), Z1(N), z2(N))
  call random_number(r(1:N))
  z = cmplx(r(1:N) - 0.5_dp, 0.0_dp, dp)
  call dft(z, Z1, -1)
  call dft(Z1, z2, +1)
  z2 = z2/N                                       ! the inverse carries 1/N
  print '(a, es10.2)', 'round trip max error  ', maxval(abs(z2 - z))
  print '(a, 2f14.8)', 'Parseval  sum|x|^2, sum|X|^2/N:', sum(abs(z)**2), sum(abs(Z1)**2)/N
  do i = 1, N
     z(i) = cmplx(cos(2.0_dp*pi*50.0_dp*(i-1)/N), 0.0_dp, dp)
  end do
  call dft(z, Z1, -1)
  kmax = maxloc(abs(Z1(1:N/2)))
  print '(a, i0, a, f8.3)', 'pure tone: peak at k = ', kmax(1) - 1, ', |X| = ', abs(Z1(kmax(1)))
  deallocate(z, Z1, z2)

  ! ---- 3. O(N^2): timing
  print '(a)', '     N    time [ms]'
  N = 512
  do while (N <= 4096)
     allocate(z(N), Z1(N))
     call random_number(r(1:N)); z = cmplx(r(1:N), 0.0_dp, dp)
     call system_clock(t0, rate); call dft(z, Z1, -1); call system_clock(t1)
     print '(i6, f12.2)', N, 1000.0_dp*(t1 - t0)/rate
     deallocate(z, Z1)
     N = 2*N
  end do
contains
  real(dp) function variance(a)
    real(dp), intent(in) :: a(:)
    variance = sum((a - sum(a)/size(a))**2)/size(a)
  end function variance
end program dft_direct
