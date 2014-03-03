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

CONTAINS

SUBROUTINE tea_leaf()
 
  USE report_module
  USE clover_module
  USE tea_leaf_kernel_module
  USE update_halo_module

  IMPLICIT NONE

!$ INTEGER :: OMP_GET_THREAD_NUM
  INTEGER :: c, n, j,k
  REAL(KIND=8) :: ry,rx, error, old_error

  INTEGER :: fields(NUM_FIELDS)

  REAL(KIND=8) :: kernel_time,timer

  ! For CG solver
  REAL(KIND=8) :: rro, pw, rrn, alpha, beta

  DO c=1,number_of_chunks

    IF(chunks(c)%task.EQ.parallel%task) THEN

      ! set old error to 0 initially
      old_error = 0.0

      fields=0
      fields(FIELD_ENERGY1) = 1
      fields(FIELD_DENSITY1) = 1
      CALL update_halo(fields,2)

      ! INIT
      IF(profiler_on) kernel_time=timer()

      IF(tl_use_cg) then
        IF(use_fortran_kernels) THEN
          rx = dt/(chunks(c)%field%celldx(chunks(c)%field%x_min)**2);
          ry = dt/(chunks(c)%field%celldy(chunks(c)%field%y_min)**2);

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
        ELSEIF(use_ocl_kernels) THEN
          CALL tea_leaf_kernel_init_cg_ocl(coefficient, dt, rx, ry, rro)
        ELSEIF(use_C_kernels) THEN
          CALL report_error('tea_leaf', "C CG SOLVER CALLED BUT NOT IMPLEMENTED")
          ! TODO
          !rx = dt/(chunks(c)%field%celldx(chunks(c)%field%x_min)**2);
          !ry = dt/(chunks(c)%field%celldy(chunks(c)%field%y_min)**2);

          !CALL tea_leaf_kernel_init_cg_c(chunks(c)%field%x_min, &
          !    chunks(c)%field%x_max,                       &
          !    chunks(c)%field%y_min,                       &
          !    chunks(c)%field%y_max,                       &
          !    chunks(c)%field%celldx,                      &
          !    chunks(c)%field%celldy,                      &
          !    chunks(c)%field%volume,                      &
          !    chunks(c)%field%density1,                    &
          !    chunks(c)%field%energy1,                     &
          !    chunks(c)%field%work_array1,                 &
          !    chunks(c)%field%u,                           &
          !    chunks(c)%field%work_array2,                 &
          !    chunks(c)%field%work_array3,                 &
          !    chunks(c)%field%work_array4,                 &
          !    chunks(c)%field%work_array5,                 &
          !    chunks(c)%field%work_array6,                 &
          !    chunks(c)%field%work_array7,                 &
          !    coefficient)
        ENDIF

        ! need to update p at this stage
        fields=0
        fields(FIELD_U) = 1
        CALL update_halo(fields,2)

        ! and globally sum rro
        call clover_allsum(rro)
      ELSE
        IF (use_fortran_kernels) THEN
          rx = dt/(chunks(c)%field%celldx(chunks(c)%field%x_min)**2);
          ry = dt/(chunks(c)%field%celldy(chunks(c)%field%y_min)**2);

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
        ELSEIF(use_ocl_kernels) THEN
          CALL tea_leaf_kernel_init_ocl(coefficient, dt, rx, ry)
        ELSEIF(use_C_kernels) THEN
          rx = dt/(chunks(c)%field%celldx(chunks(c)%field%x_min)**2);
          ry = dt/(chunks(c)%field%celldy(chunks(c)%field%y_min)**2);

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

      DO n=1,max_iters

        IF(tl_use_cg) then
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
          ELSEIF(use_ocl_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_ocl_calc_w(rx, ry, pw)
          ELSEIF(use_c_kernels) THEN
            ! TODO
            CALL report_error('tea_leaf', "C CG SOLVER CALLED BUT NOT IMPLEMENTED")
          ENDIF

          CALL clover_allsum(pw)
          alpha = rro/pw

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
          ELSEIF(use_ocl_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_ocl_calc_ur(alpha, rrn)
          ELSEIF(use_c_kernels) THEN
            ! TODO
            CALL report_error('tea_leaf', "C CG SOLVER CALLED BUT NOT IMPLEMENTED")
          ENDIF

          CALL clover_allsum(rrn)
          beta = rrn/rro

          IF(use_fortran_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_fortran_calc_p(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array2,                 &
                chunks(c)%field%work_array5,                 &
                beta)
          ELSEIF(use_ocl_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_ocl_calc_p(beta, rrn)
          ELSEIF(use_c_kernels) THEN
            ! TODO
            CALL report_error('tea_leaf', "C CG SOLVER CALLED BUT NOT IMPLEMENTED")
          ENDIF

          error = rrn
          rro = rrn
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
          ELSEIF(use_ocl_kernels) THEN
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
        ENDIF

        ! CALL update_halo
        fields=0
        fields(FIELD_U) = 1
        CALL update_halo(fields,2)

        CALL clover_max(error)

        IF (abs(error) .LT. eps) EXIT

        ! if the error isn't getting any better, then exit - no point in going further
        IF (abs(error - old_error) .LT. eps .or. error .eq. old_error) EXIT
        old_error = error

      ENDDO

      IF (parallel%boss) THEN
!$      IF(OMP_GET_THREAD_NUM().EQ.0) THEN
          WRITE(g_out,"('Conduction error ',e14.7)") error
          WRITE(g_out,"('Iteration count ',i8)") n-1
          WRITE(0,"('Conduction error ',e14.7)") error
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
      ELSEIF(use_ocl_kernels) THEN
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
  IF(profiler_on) profiler%PdV=profiler%tea+(timer()-kernel_time)

END SUBROUTINE tea_leaf

END MODULE tea_leaf_module
