!Crown Copyright 2014 AWE.
!
! This file is part of TeaLeaf.
!
! TeaLeaf is free software: you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the
! Free Software Foundation, either version 3 of the License, or (at your option)
! any later version.
!
! TeaLeaf is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
! details.
!
! You should have received a copy of the GNU General Public License along with
! TeaLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief Driver for the heat conduction kernel
!>  @author David Beckingsale, Wayne Gaudin
!>  @details Invokes the user specified kernel for the heat conduction

MODULE tea_leaf_module

  USE report_module
  USE data_module
  USE tea_leaf_kernel_module
  USE tea_leaf_kernel_cg_module
  USE tea_leaf_kernel_cheby_module
  USE update_halo_module

  IMPLICIT NONE

  interface
    subroutine tea_leaf_kernel_cheby_copy_u_ocl()
    end subroutine

    subroutine tea_leaf_calc_2norm_kernel_ocl(initial, norm)
      integer :: initial
      real(kind=8) :: norm
    end subroutine

    subroutine tea_leaf_kernel_cheby_init_ocl(ch_alphas, ch_betas, n_coefs, &
        rx, ry, theta, error)
      real(kind=8) :: rx, ry, theta, error
      integer :: n_coefs
      real(kind=8), dimension(n_coefs) :: ch_alphas, ch_betas
    end subroutine

    subroutine tea_leaf_kernel_ppcg_init_ocl(ch_alphas, ch_betas, n_coefs, &
        theta, n, rro)
      integer :: n, n_coefs
      real(kind=8) :: theta, rro
      real(kind=8), dimension(n_coefs) :: ch_alphas, ch_betas
    end subroutine

    subroutine tea_leaf_kernel_ppcg_init_sd_ocl()
    end subroutine

    subroutine tea_leaf_kernel_ppcg_inner_ocl(n)
      integer :: n
    end subroutine

    subroutine tea_leaf_kernel_cheby_iterate_ocl(ch_alphas, ch_betas, n_coefs, &
        rx, ry, cheby_calc_step)
      real(kind=8) :: rx, ry
      integer :: cheby_calc_step
      integer :: n_coefs
      real(kind=8), dimension(n_coefs) :: ch_alphas, ch_betas
    end subroutine
  end interface

CONTAINS

SUBROUTINE tea_leaf()

  IMPLICIT NONE

!$ INTEGER :: OMP_GET_THREAD_NUM
  INTEGER :: c, n
  REAL(KIND=8) :: ry,rx, error, exact_error

  INTEGER :: fields(NUM_FIELDS)

  REAL(KIND=8) :: kernel_time,timer

  ! For CG solver
  REAL(KIND=8) :: rro, pw, rrn, alpha, beta

  ! For chebyshev solver
  REAL(KIND=8), DIMENSION(max_iters) :: cg_alphas, cg_betas
  REAL(KIND=8), DIMENSION(max_iters) :: ch_alphas, ch_betas
  REAL(KIND=8) :: eigmin, eigmax, theta, cn
  INTEGER :: est_itc, cheby_calc_steps, max_cheby_iters, info, switch_step
  LOGICAL :: ch_switch_check

  INTEGER :: cg_calc_steps, ppcg_cur_step

  REAL(KIND=8) :: cg_time, ch_time, solve_timer, total_solve_time, ch_per_it, cg_per_it
  cg_time = 0.0_8
  ch_time = 0.0_8
  cg_calc_steps = 0
  total_solve_time = 0.0_8

  IF(coefficient .nE. RECIP_CONDUCTIVITY .and. coefficient .ne. conductivity) THEN
    CALL report_error('tea_leaf', 'unknown coefficient option')
  endif

  error = 1e10
  cheby_calc_steps = 0

  DO c=1,chunks_per_task

    IF(chunks(c)%task.EQ.parallel%task) THEN

      fields=0
      fields(FIELD_ENERGY1) = 1
      fields(FIELD_DENSITY1) = 1
      CALL update_halo(fields,2)

      ! INIT
      IF(profiler_on) kernel_time=timer()

      if (use_fortran_kernels .or. use_c_kernels) then
        rx = dt/(chunks(c)%field%celldx(chunks(c)%field%x_min)**2)
        ry = dt/(chunks(c)%field%celldy(chunks(c)%field%y_min)**2)
      endif

      IF(tl_use_cg .or. tl_use_chebyshev .or. tl_use_ppcg) then
        IF(use_fortran_kernels) THEN
          CALL tea_leaf_kernel_init_cg_fortran(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                       &
              chunks(c)%field%y_min,                       &
              chunks(c)%field%y_max,                       &
              chunks(c)%field%density1,                    &
              chunks(c)%field%energy1,                     &
              chunks(c)%field%u,                           &
              chunks(c)%field%work_array1,                 &
              chunks(c)%field%work_array2,                 &
              chunks(c)%field%work_array3,                 &
              chunks(c)%field%work_array4,                 &
              chunks(c)%field%work_array5,                 &
              chunks(c)%field%work_array6,                 &
              chunks(c)%field%work_array7,                 &
              rx, ry, rro, coefficient)
        ELSEIF(use_opencl_kernels) THEN
          CALL tea_leaf_kernel_init_cg_ocl(coefficient, dt, rx, ry, rro)
        ELSEIF(use_C_kernels) THEN
          CALL tea_leaf_kernel_init_cg_c(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                       &
              chunks(c)%field%y_min,                       &
              chunks(c)%field%y_max,                       &
              chunks(c)%field%density1,                    &
              chunks(c)%field%energy1,                     &
              chunks(c)%field%u,                           &
              chunks(c)%field%work_array1,                 &
              chunks(c)%field%work_array2,                 &
              chunks(c)%field%work_array3,                 &
              chunks(c)%field%work_array4,                 &
              chunks(c)%field%work_array5,                 &
              chunks(c)%field%work_array6,                 &
              chunks(c)%field%work_array7,                 &
              rx, ry, rro, coefficient)
        ENDIF

        ! need to update p at this stage
        fields=0
        fields(FIELD_U) = 1
        fields(FIELD_P) = 1
        CALL update_halo(fields,1)

        ! and globally sum rro
        call clover_allsum(rro)
      ELSE
        IF (use_fortran_kernels) THEN
          CALL tea_leaf_kernel_init(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                       &
              chunks(c)%field%y_min,                       &
              chunks(c)%field%y_max,                       &
              chunks(c)%field%celldx,                      &
              chunks(c)%field%celldy,                      &
              chunks(c)%field%volume,                      &
              chunks(c)%field%density1,                    &
              chunks(c)%field%energy1,                     &
              chunks(c)%field%work_array1,                 &
              chunks(c)%field%u,                           &
              chunks(c)%field%work_array2,                 &
              chunks(c)%field%work_array3,                 &
              chunks(c)%field%work_array4,                 &
              chunks(c)%field%work_array5,                 &
              chunks(c)%field%work_array6,                 &
              chunks(c)%field%work_array7,                 &
              coefficient)
        ELSEIF(use_opencl_kernels) THEN
          CALL tea_leaf_kernel_init_ocl(coefficient, dt, rx, ry)
        ELSEIF(use_C_kernels) THEN
          CALL tea_leaf_kernel_init_c(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                       &
              chunks(c)%field%y_min,                       &
              chunks(c)%field%y_max,                       &
              chunks(c)%field%celldx,                      &
              chunks(c)%field%celldy,                      &
              chunks(c)%field%volume,                      &
              chunks(c)%field%density1,                    &
              chunks(c)%field%energy1,                     &
              chunks(c)%field%work_array1,                 &
              chunks(c)%field%u,                           &
              chunks(c)%field%work_array2,                 &
              chunks(c)%field%work_array3,                 &
              chunks(c)%field%work_array4,                 &
              chunks(c)%field%work_array5,                 &
              chunks(c)%field%work_array6,                 &
              chunks(c)%field%work_array7,                 &
              coefficient)
        ENDIF

      ENDIF

      fields=0
      fields(FIELD_U) = 1

      ! Copy every time - is used for most tea leaf configurations + error checking
      IF(use_fortran_kernels) then
        call tea_leaf_kernel_cheby_copy_u(chunks(c)%field%x_min,&
          chunks(c)%field%x_max,                       &
          chunks(c)%field%y_min,                       &
          chunks(c)%field%y_max,                       &
          chunks(c)%field%u0,                &
          chunks(c)%field%u)
      elseif(use_opencl_kernels) then
        call tea_leaf_kernel_cheby_copy_u_ocl()
      endif

      DO n=1,max_iters

        if (profile_solver) solve_timer=timer()

        IF (tl_ch_cg_errswitch) then
            ! either the error has got below tolerance, or it's already going
            ch_switch_check = (cheby_calc_steps .gt. 0) .or. (error .le. tl_ch_cg_epslim)
        ELSE
            ! enough steps have passed
            ch_switch_check = n .ge. tl_ch_cg_presteps
        ENDIF

        IF ((tl_use_chebyshev .or. tl_use_ppcg) .and. ch_switch_check) then
          ! on the first chebyshev steps, find the eigenvalues, coefficients,
          ! and expected number of iterations
          IF (cheby_calc_steps .eq. 0) then
            ! maximum number of iterations in chebyshev solver
            max_cheby_iters = max_iters - n + 2
            rro = error

            ! calculate eigenvalues
            call tea_calc_eigenvalues(cg_alphas, cg_betas, eigmin, eigmax, &
                max_iters, n-1, info)

            if (info .ne. 0) CALL report_error('tea_leaf', 'Error in calculating eigenvalues')

            if (tl_use_chebyshev) then
              ! calculate chebyshev coefficients
              call tea_calc_ch_coefs(ch_alphas, ch_betas, eigmin, eigmax, &
                  theta, max_cheby_iters)
            else if (tl_use_ppcg) then
              ! calculate least squares coefficients
              call tea_calc_ls_coefs(ch_alphas, ch_betas, eigmin, eigmax, &
                  theta, tl_ppcg_inner_steps)
            endif

            cn = eigmax/eigmin

            if (parallel%boss) then
              write(g_out,'(a,i3,a,e15.7)') "Switching after ",n," steps, error ",rro
              write(g_out,'(4a11)')"eigmin", "eigmax", "cn", "error"
              write(g_out,'(2f11.5,2e11.4)')eigmin, eigmax, cn, error
              write(0,'(a,i3,a,e15.7)') "Switching after ",n," steps, error ",rro
              write(0,'(4a11)')"eigmin", "eigmax", "cn", "error"
              write(0,'(2f11.5,2e11.4)')eigmin, eigmax, cn, error
            endif
          endif

          if (tl_use_chebyshev) then
              ! don't need to update p any more
              fields(FIELD_P) = 0

              if (cheby_calc_steps .eq. 0) then
                call tea_leaf_cheby_first_step(c, ch_alphas, ch_betas, fields, &
                    error, rx, ry, theta, cn, max_cheby_iters)

                cheby_calc_steps = 2

                switch_step = n
              else
                  IF(use_fortran_kernels) THEN
                      call tea_leaf_kernel_cheby_iterate(chunks(c)%field%x_min,&
                          chunks(c)%field%x_max,                       &
                          chunks(c)%field%y_min,                       &
                          chunks(c)%field%y_max,                       &
                          chunks(c)%field%u,                           &
                          chunks(c)%field%u0,                          &
                          chunks(c)%field%work_array1,                 &
                          chunks(c)%field%work_array2,                 &
                          chunks(c)%field%work_array3,                 &
                          chunks(c)%field%work_array4,                 &
                          chunks(c)%field%work_array5,                 &
                          chunks(c)%field%work_array6,                 &
                          chunks(c)%field%work_array7,                 &
                          ch_alphas, ch_betas, max_cheby_iters,        &
                          rx, ry, cheby_calc_steps)
                  ELSEIF(use_opencl_kernels) THEN
                      call tea_leaf_kernel_cheby_iterate_ocl(ch_alphas, ch_betas, max_cheby_iters, &
                        rx, ry, cheby_calc_steps)
                  ENDIF

                  ! after estimated number of iterations has passed, calc resid
                  ! Leaving 10 iterations between each global reduction won't affect
                  ! total time spent much if at all (number of steps spent in
                  ! chebyshev is typically O(300+)) but will greatyl reduce global
                  ! synchronisations needed
                  if ((n-switch_step .ge. est_itc) .and. (mod(n, 10) .eq. 0)) then
                    IF(use_fortran_kernels) THEN
                      call tea_leaf_calc_2norm_kernel(chunks(c)%field%x_min,        &
                            chunks(c)%field%x_max,                       &
                            chunks(c)%field%y_min,                       &
                            chunks(c)%field%y_max,                       &
                            chunks(c)%field%work_array2,                 &
                            error)
                    ELSEIF(use_opencl_kernels) THEN
                      call tea_leaf_calc_2norm_kernel_ocl(1, error)
                    ENDIF

                    call clover_allsum(error)
                  else
                    ! dummy to make it go smaller every time but not reach tolerance
                    error = 1.0_8/(cheby_calc_steps)
                  endif
              endif
          else if (tl_use_ppcg) then
            fields = 0
            fields(FIELD_U) = 1
            fields(FIELD_P) = 1
            fields(FIELD_SD) = 1

            if (cheby_calc_steps .eq. 0) then
                cheby_calc_steps = 1

                IF(use_fortran_kernels) THEN
                    call report_error('tea_leaf', 'Fortran ppcg not implemented')
                ELSEIF(use_opencl_kernels) THEN
                  ! FIXME change to an input file number of iterations
                    call tea_leaf_kernel_ppcg_init_ocl(ch_alphas, ch_betas, &
                      max_cheby_iters, theta, tl_ppcg_inner_steps, rro)
                ENDIF
            endif

            IF(use_fortran_kernels) THEN
            ELSEIF(use_opencl_kernels) THEN
              CALL tea_leaf_kernel_solve_cg_ocl_calc_w(rx, ry, pw)
            ENDIF

            CALL clover_allsum(pw)
            alpha = rro/pw

            IF(use_fortran_kernels) THEN
            ELSEIF(use_opencl_kernels) THEN
              ! XXX don't need to do the reduction
              CALL tea_leaf_kernel_solve_cg_ocl_calc_ur(alpha, rrn)
            ENDIF

            IF(use_fortran_kernels) THEN
            ELSEIF(use_opencl_kernels) THEN
              CALL tea_leaf_kernel_ppcg_init_sd_ocl()
            ENDIF

            DO ppcg_cur_step=1,tl_ppcg_inner_steps
              IF(use_fortran_kernels) THEN
              ELSEIF(use_opencl_kernels) THEN
                CALL tea_leaf_kernel_ppcg_inner_ocl(ppcg_cur_step)
              ENDIF
            ENDDO

            IF(use_fortran_kernels) THEN
            ELSEIF(use_opencl_kernels) THEN
              call tea_leaf_calc_2norm_kernel_ocl(1, rrn)
            ENDIF

            CALL clover_allsum(rrn)

            beta = rrn/rro

            IF(use_fortran_kernels) THEN
            ELSEIF(use_opencl_kernels) THEN
              CALL tea_leaf_kernel_solve_cg_ocl_calc_p(beta)
            ENDIF

            error = rrn
            rro = rrn

            CALL clover_allsum(error)
          endif

          cheby_calc_steps = cheby_calc_steps + 1
        ELSEIF(tl_use_cg .or. tl_use_chebyshev .or. tl_use_ppcg) then
          fields(FIELD_P) = 1
          cg_calc_steps = cg_calc_steps + 1

          IF(use_fortran_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_fortran_calc_w(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array4,                 &
                chunks(c)%field%work_array6,                 &
                chunks(c)%field%work_array7,                 &
                rx, ry, pw)
          ELSEIF(use_opencl_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_ocl_calc_w(rx, ry, pw)
          ELSEIF(use_c_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_c_calc_w(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array4,                 &
                chunks(c)%field%work_array6,                 &
                chunks(c)%field%work_array7,                 &
                rx, ry, pw)
          ENDIF

          CALL clover_allsum(pw)
          alpha = rro/pw
          if(tl_use_chebyshev .or. tl_use_ppcg) cg_alphas(n) = alpha

          IF(use_fortran_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_fortran_calc_ur(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%u,                           &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array2,                 &
                chunks(c)%field%work_array3,                 &
                chunks(c)%field%work_array4,                 &
                chunks(c)%field%work_array5,                 &
                alpha, rrn)
          ELSEIF(use_opencl_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_ocl_calc_ur(alpha, rrn)
          ELSEIF(use_c_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_c_calc_ur(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%u,                           &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array2,                 &
                chunks(c)%field%work_array3,                 &
                chunks(c)%field%work_array4,                 &
                chunks(c)%field%work_array5,                 &
                alpha, rrn)
          ENDIF

          CALL clover_allsum(rrn)
          beta = rrn/rro
          if(tl_use_chebyshev .or. tl_use_ppcg) cg_betas(n) = beta

          IF(use_fortran_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_fortran_calc_p(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array2,                 &
                chunks(c)%field%work_array5,                 &
                beta)
          ELSEIF(use_opencl_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_ocl_calc_p(beta)
          ELSEIF(use_c_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_c_calc_p(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array2,                 &
                chunks(c)%field%work_array5,                 &
                beta)
          ENDIF

          error = rrn
          rro = rrn

          CALL clover_allsum(error)
        ELSE
          IF(use_fortran_kernels) THEN
            CALL tea_leaf_kernel_solve(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                rx,                                          &
                ry,                                          &
                chunks(c)%field%work_array6,                 &
                chunks(c)%field%work_array7,                 &
                error,                                       &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%u,                           &
                chunks(c)%field%work_array2)
          ELSEIF(use_opencl_kernels) THEN
              CALL tea_leaf_kernel_solve_ocl(rx, ry, error)
          ELSEIF(use_C_kernels) THEN
              CALL tea_leaf_kernel_solve_c(chunks(c)%field%x_min,&
                  chunks(c)%field%x_max,                       &
                  chunks(c)%field%y_min,                       &
                  chunks(c)%field%y_max,                       &
                  rx,                                          &
                  ry,                                          &
                  chunks(c)%field%work_array6,                 &
                  chunks(c)%field%work_array7,                 &
                  error,                                       &
                  chunks(c)%field%work_array1,                 &
                  chunks(c)%field%u,                           &
                  chunks(c)%field%work_array2)
          ENDIF

          CALL clover_max(error)
        ENDIF

        ! updates u and possibly p
        CALL update_halo(fields,1)

        if (profile_solver) then
          IF (tl_use_chebyshev .and. ch_switch_check) then
              ch_time=ch_time+(timer()-solve_timer)
          else
              cg_time=cg_time+(timer()-solve_timer)
          endif
          total_solve_time = total_solve_time + (timer() - solve_timer)
        endif

        IF (abs(error) .LT. eps) EXIT

      ENDDO

      if (tl_check_result) then
        IF(use_fortran_kernels) THEN
            CALL tea_leaf_calc_residual(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%u,                           &
                chunks(c)%field%u0,                 &
                chunks(c)%field%work_array2,                 &
                chunks(c)%field%work_array6,                 &
                chunks(c)%field%work_array7,                 &
                rx, ry, exact_error)
        ELSEIF(use_opencl_kernels) THEN
            CALL tea_leaf_calc_residual_ocl(exact_error)
        ENDIF

        call clover_allsum(exact_error)
      endif

      IF (parallel%boss) THEN
!$      IF(OMP_GET_THREAD_NUM().EQ.0) THEN
          WRITE(g_out,"('Conduction error ',e14.7)") error
          WRITE(0,"('Conduction error ',e14.7)") error

          if (tl_check_result) then
            write(0,"('EXACT error calculated as', e14.7)") exact_error
            write(g_out,"('EXACT error calculated as', e14.7)") exact_error
          endif

          WRITE(g_out,"('Iteration count ',i8)") n-1
          WRITE(0,"('Iteration count ', i8)") n-1
!$      ENDIF
      ENDIF

      ! RESET
      IF(use_fortran_kernels) THEN
          CALL tea_leaf_kernel_finalise(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                           &
              chunks(c)%field%y_min,                           &
              chunks(c)%field%y_max,                           &
              chunks(c)%field%energy1,                         &
              chunks(c)%field%density1,                        &
              chunks(c)%field%u)
      ELSEIF(use_opencl_kernels) THEN
          CALL tea_leaf_kernel_finalise_ocl()
      ELSEIF(use_C_kernels) THEN
          CALL tea_leaf_kernel_finalise_c(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                           &
              chunks(c)%field%y_min,                           &
              chunks(c)%field%y_max,                           &
              chunks(c)%field%energy1,                         &
              chunks(c)%field%density1,                        &
              chunks(c)%field%u)
      ENDIF

      fields=0
      fields(FIELD_ENERGY1) = 1
      CALL update_halo(fields,1)

    ENDIF

  ENDDO
  IF(profile_solver) profiler%tea=profiler%tea+(timer()-kernel_time)

  IF (profile_solver .and. parallel%boss) then
    write(0, "(a16, a7, a16)") "Time", "Steps", "Per it"
    write(0, "(f16.10, i7, f16.10, f7.2)") total_solve_time, n, total_solve_time/n
    write(g_out, "(a16, a7, a16)") "Time", "Steps", "Per it"
    write(g_out, "(f16.10, i7, f16.10, f7.2)") total_solve_time, n, total_solve_time/n
  endif

  IF (profile_solver .and. tl_use_chebyshev) THEN
    call clover_sum(ch_time)
    call clover_sum(cg_time)
    if (parallel%boss) then
      cg_per_it = merge((cg_time/cg_calc_steps)/parallel%max_task, 0.0_8, cg_calc_steps .gt. 0)
      ch_per_it = merge((ch_time/cheby_calc_steps)/parallel%max_task, 0.0_8, cheby_calc_steps .gt. 0)

      write(0, "(a3, a16, a7, a16, a7)") "", "Time", "Steps", "Per it", "Ratio"
      write(0, "(a3, f16.10, i7, f16.10, f7.2)") &
          "CG", cg_time + 0.0_8, cg_calc_steps, cg_per_it, 1.0_8
      write(0, "(a3, f16.10, i7, f16.10, f7.2)") "CH", ch_time + 0.0_8, cheby_calc_steps, &
          ch_per_it, merge(ch_per_it/cg_per_it, 0.0_8, cheby_calc_steps .gt. 0)
      write(0, "('Chebyshev actually took ', i6, ' (' i6, ' off guess)')") &
          cheby_calc_steps, cheby_calc_steps-est_itc

      write(g_out, "(a3, a16, a7, a16, a7)") "", "Time", "Steps", "Per it", "Ratio"
      write(g_out, "(a3, f16.10, i7, f16.10, f7.2)") &
          "CG", cg_time + 0.0_8, cg_calc_steps, cg_per_it, 1.0_8
      write(g_out, "(a3, f16.10, i7, f16.10, f7.2)") "CH", ch_time + 0.0_8, cheby_calc_steps, &
          ch_per_it, merge(ch_per_it/cg_per_it, 0.0_8, cheby_calc_steps .gt. 0)
      write(g_out, "('Chebyshev actually took ', i6, ' (' i6, ' off guess)')") &
          cheby_calc_steps, cheby_calc_steps-est_itc
    endif
  endif

END SUBROUTINE tea_leaf

subroutine tea_leaf_cheby_first_step(c, ch_alphas, ch_betas, fields, &
    error, rx, ry, theta, cn, max_cheby_iters)

  IMPLICIT NONE

  integer :: c, est_itc, max_cheby_iters
  integer, dimension(:) :: fields
  REAL(KIND=8) :: it_alpha, cn, gamm, bb, error, rx, ry, theta
  REAL(KIND=8), DIMENSION(:) :: ch_alphas, ch_betas

  ! calculate 2 norm of u0
  IF(use_fortran_kernels) THEN
    call tea_leaf_calc_2norm_kernel(chunks(c)%field%x_min,        &
          chunks(c)%field%x_max,                       &
          chunks(c)%field%y_min,                       &
          chunks(c)%field%y_max,                       &
          chunks(c)%field%u0,                 &
          bb)
  ELSEIF(use_opencl_kernels) THEN
    call tea_leaf_calc_2norm_kernel_ocl(0, bb)
  ENDIF

  call clover_allsum(bb)

  ! initialise 'p' array
  IF(use_fortran_kernels) THEN
    call tea_leaf_kernel_cheby_init(chunks(c)%field%x_min,&
          chunks(c)%field%x_max,                       &
          chunks(c)%field%y_min,                       &
          chunks(c)%field%y_max,                       &
          chunks(c)%field%u,                           &
          chunks(c)%field%u0,                 &
          chunks(c)%field%work_array1,                 &
          chunks(c)%field%work_array2,                 &
          chunks(c)%field%work_array3,                 &
          chunks(c)%field%work_array4,                 &
          chunks(c)%field%work_array5,                 &
          chunks(c)%field%work_array6,                 &
          chunks(c)%field%work_array7,                 &
          ch_alphas, ch_betas, max_cheby_iters, &
          rx, ry, theta, error)
  ELSEIF(use_opencl_kernels) THEN
    call tea_leaf_kernel_cheby_init_ocl(ch_alphas, ch_betas, &
      max_cheby_iters, rx, ry, theta, error)
  ENDIF

  CALL update_halo(fields,1)

  IF(use_fortran_kernels) THEN
      call tea_leaf_kernel_cheby_iterate(chunks(c)%field%x_min,&
          chunks(c)%field%x_max,                       &
          chunks(c)%field%y_min,                       &
          chunks(c)%field%y_max,                       &
          chunks(c)%field%u,                           &
          chunks(c)%field%u0,                          &
          chunks(c)%field%work_array1,                 &
          chunks(c)%field%work_array2,                 &
          chunks(c)%field%work_array3,                 &
          chunks(c)%field%work_array4,                 &
          chunks(c)%field%work_array5,                 &
          chunks(c)%field%work_array6,                 &
          chunks(c)%field%work_array7,                 &
          ch_alphas, ch_betas, max_cheby_iters,        &
          rx, ry, 1)
  ELSEIF(use_opencl_kernels) THEN
      call tea_leaf_kernel_cheby_iterate_ocl(ch_alphas, ch_betas, max_cheby_iters, &
        rx, ry, 1)
  ENDIF

  IF(use_fortran_kernels) THEN
    call tea_leaf_calc_2norm_kernel(chunks(c)%field%x_min,        &
          chunks(c)%field%x_max,                       &
          chunks(c)%field%y_min,                       &
          chunks(c)%field%y_max,                       &
          chunks(c)%field%work_array2,                 &
          error)
  ELSEIF(use_opencl_kernels) THEN
    call tea_leaf_calc_2norm_kernel_ocl(1, error)
  ENDIF

  call clover_allsum(error)

  ! FIXME not giving correct estimate
  it_alpha = eps*bb/(4.0_8*error)
  gamm = (sqrt(cn) - 1.0_8)/(sqrt(cn) + 1.0_8)
  est_itc = nint(log(it_alpha)/(2.0_8*log(gamm)))

  ! FIXME still not giving correct answer, but multiply by 2.5 does
  ! an 'okay' job for now
  est_itc = int(est_itc * 2.5)

  if (parallel%boss) then
      write(g_out,'(a11)')"est itc"
      write(g_out,'(11i11)')est_itc
      write(0,'(a11)')"est itc"
      write(0,'(11i11)')est_itc
  endif

end subroutine tea_leaf_cheby_first_step

END MODULE tea_leaf_module
