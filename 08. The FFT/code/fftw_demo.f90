! fftw_demo.f90  --  Lecture 8: calling FFTW from modern Fortran (plan, execute, destroy)
! Build:  gfortran fftw_demo.f90 -I/usr/include -lfftw3 -lm -o fftw_demo && ./fftw_demo
! (Windows: MSYS2 package mingw-w64-ucrt-x86_64-fftw; Linux: libfftw3-dev)
program fftw_demo
  use, intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'
  integer(C_INT), parameter :: n = 1024
  real(C_DOUBLE), parameter :: pi = 4*atan(1.0_C_DOUBLE)
  type(C_PTR) :: plan, plan_back, plan_r2c
  complex(C_DOUBLE_COMPLEX) :: x(n), xk(n), xb(n), ck(n/2+1)
  real(C_DOUBLE) :: xr(n)
  integer :: i, kmax(1)

  ! plan first: FFTW_MEASURE would overwrite the arrays (ESTIMATE does not)
  plan      = fftw_plan_dft_1d(n, x, xk, FFTW_FORWARD,  FFTW_ESTIMATE)
  plan_back = fftw_plan_dft_1d(n, xk, xb, FFTW_BACKWARD, FFTW_ESTIMATE)
  plan_r2c  = fftw_plan_dft_r2c_1d(n, xr, ck, FFTW_ESTIMATE)

  do i = 1, n                                   ! pure tone in bin k = 50
     x(i) = cmplx(cos(2*pi*50*(i-1)/n), 0.0_C_DOUBLE, C_DOUBLE_COMPLEX)
  end do
  xr = real(x)
  call fftw_execute_dft(plan, x, xk)            ! reusable for new data
  kmax = maxloc(abs(xk(1:n/2)))
  print '(a, i0, a, f9.3)', 'peak at k = ', kmax(1) - 1, ',  |X| = ', abs(xk(kmax(1)))

  call fftw_execute_dft(plan_back, xk, xb)
  print '(a, es10.2)', 'FFTW is unnormalised: backward(forward(x))/n - x, max = ', maxval(abs(xb/n - x))

  call fftw_execute_dft_r2c(plan_r2c, xr, ck)   ! real input: n/2+1 bins
  print '(a, i0, a, es10.2)', 'r2c gives ', size(ck), ' bins, max |c2c - r2c| = ', maxval(abs(ck - xk(1:n/2+1)))

  call fftw_destroy_plan(plan); call fftw_destroy_plan(plan_back); call fftw_destroy_plan(plan_r2c)
end program fftw_demo
