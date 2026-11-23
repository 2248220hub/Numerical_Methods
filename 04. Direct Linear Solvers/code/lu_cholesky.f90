! lu_cholesky.f90 -- Lecture 4: in-place PA = LU with partial pivoting, forward/back substitution,
! determinant, Cholesky A = L L^T, and a timing comparison with LAPACK (dgetrf/dgetrs, dpotrf).
!
! Build:  gfortran -O2 lu_cholesky.f90 -llapack -lblas -o lu_cholesky && ./lu_cholesky
!         (Linux: sudo apt install liblapack-dev; macOS: brew install lapack, or -framework Accelerate)
!
! Fortran stores arrays by COLUMN: the inner loops below run down a column (first index) for speed.
module direct_solvers
  implicit none
  integer, parameter :: dp = kind(1.0d0)
contains

  ! In-place LU with partial pivoting. On exit: strict lower triangle = multipliers l_ik (unit diagonal
  ! implied), upper triangle = U. piv(k) = row swapped with row k at step k (LAPACK convention).
  subroutine lu_factor(a, piv, nswap)
    real(dp), intent(inout) :: a(:,:)
    integer, intent(out) :: piv(:), nswap
    integer :: n, k, j, r
    real(dp) :: tmp(size(a,1))
    n = size(a,1); nswap = 0
    do k = 1, n
      r = k - 1 + maxloc(abs(a(k:n,k)), dim=1)       ! partial pivoting: largest entry in column k
      piv(k) = r
      if (a(r,k) == 0.0_dp) stop "lu_factor: matrix is singular"
      if (r /= k) then
        tmp = a(k,:); a(k,:) = a(r,:); a(r,:) = tmp; nswap = nswap + 1
      end if
      if (k == n) exit
      a(k+1:n,k) = a(k+1:n,k) / a(k,k)               ! multipliers, |l_ik| <= 1
      do j = k+1, n                                  ! column-oriented rank-1 update of the trailing block
        a(k+1:n,j) = a(k+1:n,j) - a(k+1:n,k) * a(k,j)
      end do
    end do
  end subroutine lu_factor

  subroutine lu_solve(a, piv, b)
    real(dp), intent(in) :: a(:,:)
    integer, intent(in) :: piv(:)
    real(dp), intent(inout) :: b(:)                  ! in: right-hand side, out: solution
    integer :: n, i, k
    real(dp) :: t
    n = size(a,1)
    do k = 1, n                                      ! apply the row swaps to b, in the same order
      if (piv(k) /= k) then
        t = b(k); b(k) = b(piv(k)); b(piv(k)) = t
      end if
    end do
    do i = 2, n                                      ! forward substitution, L has unit diagonal
      b(i) = b(i) - dot_product(a(i,1:i-1), b(1:i-1))
    end do
    do i = n, 1, -1                                  ! back substitution
      b(i) = (b(i) - dot_product(a(i,i+1:n), b(i+1:n))) / a(i,i)
    end do
  end subroutine lu_solve

  ! Cholesky A = L L^T, column by column; L overwrites the lower triangle.
  subroutine cholesky(a)
    real(dp), intent(inout) :: a(:,:)
    integer :: n, j, i
    real(dp) :: s
    n = size(a,1)
    do j = 1, n
      s = a(j,j) - dot_product(a(j,1:j-1), a(j,1:j-1))
      if (s <= 0.0_dp) stop "cholesky: matrix is not positive definite"
      a(j,j) = sqrt(s)
      do i = j+1, n
        a(i,j) = (a(i,j) - dot_product(a(i,1:j-1), a(j,1:j-1))) / a(j,j)
      end do
    end do
    do j = 2, n
      a(1:j-1,j) = 0.0_dp                            ! clear the upper triangle
    end do
  end subroutine cholesky
end module direct_solvers

program lu_cholesky
  use direct_solvers
  implicit none
  integer, parameter :: sizes(3) = [200, 500, 1000]
  real(dp), allocatable :: a(:,:), a0(:,:), x(:), xt(:), b(:), l(:,:)
  integer, allocatable :: piv(:)
  integer :: n, is, nswap, info, i
  integer(8) :: c0, c1, rate
  real(dp) :: t_mine, t_lapack, logdet
  external :: dgetrf, dgetrs, dpotrf

  call random_seed(put=[(42, i = 1, 64)])
  call system_clock(count_rate=rate)
  allocate(a(4,4), piv(4)); call random_number(a); call dgetrf(4, 4, a, 4, piv, info); deallocate(a, piv)  ! warm-up
  print '(a)', "     n   my LU [ms]  LAPACK dgetrf+s [ms]   error mine   error LAPACK   log|det|"
  do is = 1, size(sizes)
    n = sizes(is)
    allocate(a(n,n), a0(n,n), x(n), xt(n), b(n), piv(n))
    call random_number(a0); a0 = a0 - 0.5_dp
    call random_number(xt)
    b = matmul(a0, xt)
    a = a0; x = b
    call system_clock(c0); call lu_factor(a, piv, nswap); call lu_solve(a, piv, x); call system_clock(c1)
    t_mine = 1.0e3_dp * real(c1 - c0, dp) / real(rate, dp)
    logdet = sum(log(abs([(a(i,i), i = 1, n)])))
    a = a0; xt = xt; b = matmul(a0, xt)
    call system_clock(c0)
    call dgetrf(n, n, a, n, piv, info); call dgetrs('N', n, 1, a, n, piv, b, n, info)
    call system_clock(c1)
    t_lapack = 1.0e3_dp * real(c1 - c0, dp) / real(rate, dp)
    print '(i6, f12.1, f22.1, es14.2, es15.2, f11.2)', n, t_mine, t_lapack, maxval(abs(x - xt)), maxval(abs(b - xt)), logdet
    deallocate(a, a0, x, xt, b, piv)
  end do

  ! Cholesky on an SPD matrix S = M M^T + n I
  n = 400
  allocate(a(n,n), l(n,n))
  call random_number(a); a = a - 0.5_dp
  l = matmul(a, transpose(a))
  do i = 1, n
    l(i,i) = l(i,i) + n
  end do
  a = l
  call cholesky(l)
  print '(/,a,i0,a,es10.2)', "Cholesky n = ", n, ": max |S - L L^T| = ", maxval(abs(a - matmul(l, transpose(l))))
  call dpotrf('L', n, a, n, info)
  print '(a,es10.2)', "  difference from LAPACK dpotrf: ", maxval(abs([(a(i,i) - l(i,i), i = 1, n)]))
end program lu_cholesky
