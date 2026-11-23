! refine_mixed.f90 -- Lecture 5: condition number from the SVD (LAPACK dgesvd) and mixed-precision
! iterative refinement: LU factorisation in SINGLE precision (sgetrf), residual r = b - A x in DOUBLE
! precision, corrections solved with the single-precision factors (sgetrs).
!
! Build:  gfortran -O2 refine_mixed.f90 -llapack -lblas -o refine_mixed && ./refine_mixed
program refine_mixed
  implicit none
  integer, parameter :: sp = kind(1.0), dp = kind(1.0d0)
  integer, parameter :: n = 300
  real(dp) :: a(n,n), q1(n,n), q2(n,n), xt(n), b(n), x(n), r(n), s(n), work(10*n*n), tau(n)
  real(dp) :: u(1,1), vt(1,1), tmp(n,n)
  real(sp) :: a32(n,n), r32(n)
  integer :: ipiv(n), info, i, k
  external :: dgesvd, dgeqrf, dorgqr, sgetrf, sgetrs

  call random_seed(put=[(7, i = 1, 64)])
  ! A = Q1 diag(logspace(0, -4)) Q2^T: condition number 1e4 by construction
  call random_number(q1); call random_number(q2)
  call dgeqrf(n, n, q1, n, tau, work, size(work), info); call dorgqr(n, n, n, q1, n, tau, work, size(work), info)
  call dgeqrf(n, n, q2, n, tau, work, size(work), info); call dorgqr(n, n, n, q2, n, tau, work, size(work), info)
  do i = 1, n
    q1(:,i) = q1(:,i) * 10.0_dp**(-4.0_dp*real(i-1,dp)/real(n-1,dp))
  end do
  a = matmul(q1, transpose(q2))

  tmp = a                                            ! dgesvd destroys its input
  call dgesvd('N', 'N', n, n, tmp, n, s, u, 1, vt, 1, work, size(work), info)
  print '(a, es10.3, a, es10.3, a, es10.3)', "sigma_max = ", s(1), "  sigma_min = ", s(n), "  kappa_2 = ", s(1)/s(n)

  call random_number(xt); b = matmul(a, xt)
  a32 = real(a, sp)
  call sgetrf(n, n, a32, n, ipiv, info)              ! the O(n^3) work, in single precision
  r32 = real(b, sp)
  call sgetrs('N', n, 1, a32, n, ipiv, r32, n, info)
  x = real(r32, dp)
  print '(a, es10.2)', "single-precision LU alone:  relative error ", norm2(x - xt)/norm2(xt)
  do k = 1, 6
    r = b - matmul(a, x)                             ! residual in double precision
    r32 = real(r, sp)
    call sgetrs('N', n, 1, a32, n, ipiv, r32, n, info)   ! correction: O(n^2) in single precision
    x = x + real(r32, dp)
    print '(a, i0, a, es10.2)', "refinement step ", k, ": relative error ", norm2(x - xt)/norm2(xt)
  end do
end program refine_mixed
