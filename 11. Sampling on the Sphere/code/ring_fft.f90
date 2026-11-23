! Exercise of Lecture 11, reference solution (Fortran): step 1 of the spherical harmonic transform,
! an FFT along each HEALPix ring, with FFTW. Reads map_nside16.bin (RING order, float64, written by
! beam_to_map.py), splits it into the 4 Nside - 1 rings with the ring lengths of the slides
! (4i in the caps, 4 Nside in the belt), and compares |F_m|, m = 0..4, with numpy (ring_fft_py.txt).
! Build: gfortran -O2 ring_fft.f90 -I/usr/include -lfftw3 -o ring_fft
program ring_fft
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: dp => real64
  implicit none
  include 'fftw3.f03'
  integer, parameter :: nside = 16, npix = 12*nside**2, nm = 5
  real(dp) :: mp(npix), ref(nm, 4*nside - 3), err
  real(C_DOUBLE), allocatable :: seg(:)
  complex(C_DOUBLE_COMPLEX), allocatable :: F(:)
  type(C_PTR) :: plan
  integer :: u, i, nr, p0, k
  open(newunit=u, file='map_nside16.bin', access='stream', form='unformatted', status='old')
  read(u) mp;  close(u)
  open(newunit=u, file='ring_fft_py.txt', status='old', action='read')
  read(u, *) ref;  close(u)
  p0 = 0;  k = 0;  err = 0
  do i = 1, 4*nside - 1
     nr = 4*min(i, nside, 4*nside - i)                 ! ring length: 4i, 4 Nside, mirror
     allocate(seg(nr), F(nr/2 + 1))
     plan = fftw_plan_dft_r2c_1d(nr, seg, F, FFTW_ESTIMATE)
     seg = mp(p0+1:p0+nr)
     call fftw_execute_dft_r2c(plan, seg, F)
     if (nr/2 + 1 >= nm) then                          ! rings with at least nm coefficients
        k = k + 1
        err = max(err, maxval(abs(abs(F(1:nm)) - ref(:, k))))
     end if
     call fftw_destroy_plan(plan);  deallocate(seg, F)
     p0 = p0 + nr
  end do
  print '(a, i0, a, i0, a, i0)', 'Nside = ', nside, ': ', 4*nside - 1, ' rings, pixels read = ', p0
  print '(a, i0, a, es9.2)', 'rings compared: ', k, ';  max | |F_m|(FFTW) - |F_m|(numpy) | = ', err
end program ring_fft
