! cg_csr.f90 -- Lecture 6: the 2D Poisson matrix in CSR storage, a parallel (OpenMP) sparse
! matrix-vector product, and the conjugate gradient method with and without a Jacobi preconditioner.
! The right-hand side comes from a known solution, so the true error can be printed.
!
! Build:  gfortran -O2 -fopenmp cg_csr.f90 -o cg_csr && OMP_NUM_THREADS=4 ./cg_csr
module csr_mod
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  type :: csr_matrix
    integer :: n = 0, nnz = 0
    real(dp), allocatable :: val(:)
    integer, allocatable :: col(:), ptr(:)          ! 1-based: row i occupies val(ptr(i) : ptr(i+1)-1)
  end type csr_matrix
contains

  ! 5-point Laplacian on a k x k grid (Dirichlet boundaries), assembled row by row directly in CSR
  function poisson2d(k) result(a)
    integer, intent(in) :: k
    type(csr_matrix) :: a
    integer :: i, j, row, p
    a%n = k*k
    allocate(a%val(5*k*k), a%col(5*k*k), a%ptr(k*k+1))
    p = 1
    do j = 1, k
      do i = 1, k
        row = (j-1)*k + i
        a%ptr(row) = p
        if (j > 1) call put(row - k, -1.0_dp)
        if (i > 1) call put(row - 1, -1.0_dp)
        call put(row, 4.0_dp)
        if (i < k) call put(row + 1, -1.0_dp)
        if (j < k) call put(row + k, -1.0_dp)
      end do
    end do
    a%ptr(a%n + 1) = p
    a%nnz = p - 1
  contains
    subroutine put(c, v)
      integer, intent(in) :: c
      real(dp), intent(in) :: v
      a%col(p) = c; a%val(p) = v; p = p + 1
    end subroutine put
  end function poisson2d

  ! y = A x: one pass over the non-zeros, rows are independent -> trivially parallel
  subroutine spmv(a, x, y)
    type(csr_matrix), intent(in) :: a
    real(dp), intent(in) :: x(:)
    real(dp), intent(out) :: y(:)
    integer :: i, p
    real(dp) :: s
    !$omp parallel do private(p, s) schedule(static)
    do i = 1, a%n
      s = 0.0_dp
      do p = a%ptr(i), a%ptr(i+1) - 1
        s = s + a%val(p) * x(a%col(p))
      end do
      y(i) = s
    end do
    !$omp end parallel do
  end subroutine spmv

  ! (Preconditioned) conjugate gradient. If dinv is present, M = diag(A) (Jacobi).
  subroutine cg(a, b, x, tol, iters, dinv)
    type(csr_matrix), intent(in) :: a
    real(dp), intent(in) :: b(:), tol
    real(dp), intent(inout) :: x(:)
    integer, intent(out) :: iters
    real(dp), intent(in), optional :: dinv(:)
    real(dp), allocatable :: r(:), z(:), p(:), ap(:)
    real(dp) :: rz, rz_new, alpha, nr0
    allocate(r(a%n), z(a%n), p(a%n), ap(a%n))
    call spmv(a, x, ap); r = b - ap
    if (present(dinv)) then; z = dinv * r; else; z = r; end if
    p = z; rz = dot_product(r, z); nr0 = norm2(r)
    do iters = 1, 10*a%n
      call spmv(a, p, ap)
      alpha = rz / dot_product(p, ap)
      x = x + alpha * p
      r = r - alpha * ap
      if (norm2(r) / nr0 < tol) return
      if (present(dinv)) then; z = dinv * r; else; z = r; end if
      rz_new = dot_product(r, z)
      p = z + (rz_new / rz) * p
      rz = rz_new
    end do
  end subroutine cg
end module csr_mod

program cg_csr
  use csr_mod
  !$ use omp_lib
  implicit none
  integer, parameter :: ks(4) = [64, 128, 256, 512]
  type(csr_matrix) :: a
  real(dp), allocatable :: xt(:), b(:), x(:), dinv(:)
  integer :: ik, k, it, i, nthreads
  integer(8) :: c0, c1, rate
  real(dp), parameter :: pi = acos(-1.0_dp)

  nthreads = 1
  !$ nthreads = omp_get_max_threads()
  call system_clock(count_rate=rate)
  print '(a,i0,a)', "2D Poisson, CSR storage, ", nthreads, " thread(s)"
  print '(a)', "     k          n        nnz   CSR [MB]   CG iters   time [ms]   max error   kappa est."
  do ik = 1, size(ks)
    k = ks(ik)
    a = poisson2d(k)
    allocate(xt(a%n), b(a%n), x(a%n))
    do i = 1, a%n                                     ! a smooth solution that is NOT an eigenvector
      xt(i) = bump(real(mod(i-1,k)+1,dp)/(k+1)) * bump(real((i-1)/k+1,dp)/(k+1))
    end do
    call spmv(a, xt, b)
    x = 0.0_dp
    call system_clock(c0); call cg(a, b, x, 1.0e-8_dp, it); call system_clock(c1)
    print '(i6, i11, i11, f11.2, i11, f12.1, es12.2, f12.0)', k, a%n, a%nnz, &
          (8.0_dp*a%nnz + 4.0_dp*(a%nnz + a%n + 1))/1.0e6_dp, it, 1.0e3_dp*real(c1-c0,dp)/real(rate,dp), &
          maxval(abs(x - xt)), (cos(pi/(2*(k+1))) / sin(pi/(2*(k+1))))**2
    deallocate(xt, b, x)
  end do

  ! Jacobi preconditioning on a badly scaled system: S A S with S = diag(10^u), u in [0, 3]
  k = 128
  a = poisson2d(k)
  allocate(xt(a%n), b(a%n), x(a%n), dinv(a%n))
  call random_seed(put=[(11, i = 1, 64)])
  call random_number(xt); xt = 10.0_dp**(3.0_dp*xt)        ! reuse xt as the scaling S
  do i = 1, a%n
    a%val(a%ptr(i):a%ptr(i+1)-1) = a%val(a%ptr(i):a%ptr(i+1)-1) * xt(i) * xt(a%col(a%ptr(i):a%ptr(i+1)-1))
  end do
  do i = 1, a%n
    dinv(i) = 1.0_dp / (4.0_dp * xt(i)**2)
  end do
  call random_number(xt); call spmv(a, xt, b)
  x = 0.0_dp; call cg(a, b, x, 1.0e-8_dp, it)
  print '(/,a,i0,a,es9.2)', "badly scaled Poisson (k = 128): CG         ", it, " iterations, error ", maxval(abs(x - xt))
  x = 0.0_dp; call cg(a, b, x, 1.0e-8_dp, it, dinv)
  print '(a,i0,a,es9.2)', "                               Jacobi-PCG ", it, " iterations, error ", maxval(abs(x - xt))
contains
  pure real(dp) function bump(s)
    real(dp), intent(in) :: s
    bump = s*(1.0_dp - s)*exp(2.0_dp*s)
  end function bump
end program cg_csr
