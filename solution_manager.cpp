#include "solution_manager.hpp"
#include <functional>

double func_timer(std::function<void()>)
{
  double t1 = MPI_Wtime();
  double t2 = MPI_Wtime();
  return t2 - t1;
}

/*!
 * We first open convergence_result and
 * execution_time output files. Next we create our model,
 * according to what explained in \ref GN_0_0_stage2_page
 * "this page". Then, we set the boundary indices.
 */
template <int dim>
SolutionManager<dim>::SolutionManager(const unsigned &order,
                                      const MPI_Comm &comm_,
                                      const unsigned &comm_size_,
                                      const unsigned &comm_rank_,
                                      const unsigned &n_threads,
                                      const bool &adaptive_on_)
  : comm(comm_),
    comm_size(comm_size_),
    comm_rank(comm_rank_),
    poly_order(order),
    quad_order(order + 1),
    n_faces_per_cell(dealii::GeometryInfo<dim>::faces_per_cell),
    the_grid(comm,
             typename dealii::Triangulation<dim>::MeshSmoothing(
               dealii::Triangulation<dim>::smoothing_on_refinement |
               dealii::Triangulation<dim>::smoothing_on_coarsening)),
    elem_mapping(),
    elem_quad_bundle(quad_order),
    face_quad_bundle(quad_order),
    LGL_quad_1D(poly_order == 0 ? 2 : poly_order + 1),
    the_elem_basis(elem_quad_bundle.get_points(), poly_order),
    postprocess_cell_basis(elem_quad_bundle.get_points(), poly_order + 1),
    the_face_basis(face_quad_bundle.get_points(), poly_order),
    refn_cycle(0),
    adaptive_on(adaptive_on_),
    n_threads(n_threads),
    time_integration_order(1),
    time_step_size(1.e-3)
{
  if (comm_rank == 0)
  {
    convergence_result.open("Convergence_Result.txt",
                            std::ofstream::out | std::fstream::app);
    execution_time.open("Execution_Time.txt",
                        std::ofstream::out | std::fstream::app);
  }
  if (true) // Long Strip Example 1
  {
    std::vector<unsigned> repeats(dim, 1);
    repeats[0] = 100;
    dealii::Point<dim> point_1, point_2;
    point_1 = {-10., -0.1};
    point_2 = {10, 0.1};
    dealii::GridGenerator::subdivided_hyper_rectangle(
      the_grid, repeats, point_1, point_2, true);
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::parallel::distributed::Triangulation<
        dim>::cell_iterator> >
      periodic_faces;
    dealii::GridTools::collect_periodic_faces(
      the_grid, 0, 1, 0, periodic_faces, dealii::Tensor<1, dim>({20., 0.}));
    dealii::GridTools::collect_periodic_faces(
      the_grid, 2, 3, 0, periodic_faces, dealii::Tensor<1, dim>({0., 0.2}));
    the_grid.add_periodicity(periodic_faces);
  }
  // End of Long Strip Example 1

  if (false) // Example 1
  {
    std::vector<unsigned> repeats(dim, 1);
    repeats[0] = 1;
    dealii::Point<dim> point_1, point_2;
    point_1 = {-1.0, -1.0};
    point_2 = {1.0, 1.0};
    dealii::GridGenerator::subdivided_hyper_rectangle(
      the_grid, repeats, point_1, point_2, true);
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::parallel::distributed::Triangulation<
        dim>::cell_iterator> >
      periodic_faces;
    dealii::GridTools::collect_periodic_faces(
      the_grid, 0, 1, 0, periodic_faces, dealii::Tensor<1, dim>({2., 0.}));
    dealii::GridTools::collect_periodic_faces(
      the_grid, 2, 3, 0, periodic_faces, dealii::Tensor<1, dim>({0., 2.}));
    the_grid.add_periodicity(periodic_faces);
  }
  // End of Example 1

  if (false) // The narrowing channel example
  {
    dealii::parallel::distributed::Triangulation<dim> left_cone(
      comm,
      typename dealii::Triangulation<dim>::MeshSmoothing(
        dealii::Triangulation<dim>::smoothing_on_refinement |
        dealii::Triangulation<dim>::smoothing_on_coarsening));
    dealii::parallel::distributed::Triangulation<dim> mid_cone(
      comm,
      typename dealii::Triangulation<dim>::MeshSmoothing(
        dealii::Triangulation<dim>::smoothing_on_refinement |
        dealii::Triangulation<dim>::smoothing_on_coarsening));
    dealii::parallel::distributed::Triangulation<dim> right_cone(
      comm,
      typename dealii::Triangulation<dim>::MeshSmoothing(
        dealii::Triangulation<dim>::smoothing_on_refinement |
        dealii::Triangulation<dim>::smoothing_on_coarsening));
    dealii::GridGenerator::truncated_cone(left_cone, 1., 0.6, 1.);
    dealii::GridGenerator::truncated_cone(mid_cone, 0.6, 0.4, 0.5);
    dealii::GridGenerator::truncated_cone(right_cone, 0.4, 0.2, 0.5);
    dealii::Tensor<1, dim> left_shift(dealii::Point<dim>(-1.0, 0.0));
    dealii::Tensor<1, dim> mid_shift(dealii::Point<dim>(0.5, 0.0));
    dealii::Tensor<1, dim> right_shift(dealii::Point<dim>(1.5, 0.0));
    dealii::GridTools::shift(left_shift, left_cone);
    dealii::GridTools::shift(mid_shift, mid_cone);
    dealii::GridTools::shift(right_shift, right_cone);
    dealii::parallel::distributed::Triangulation<dim> temp_grid(
      comm,
      typename dealii::Triangulation<dim>::MeshSmoothing(
        dealii::Triangulation<dim>::smoothing_on_refinement |
        dealii::Triangulation<dim>::smoothing_on_coarsening));
    dealii::GridGenerator::merge_triangulations(left_cone, mid_cone, temp_grid);
    dealii::GridGenerator::merge_triangulations(
      temp_grid, right_cone, the_grid);
  } // End of narrowing channel example

  if (false) // Francois's Example 1
  {
    std::vector<unsigned> repeats(dim, 1);
    repeats[0] = 1;
    dealii::Point<dim> point_1, point_2;
    point_1 = {-0.5, -0.5};
    point_2 = {0.5, 0.5};
    dealii::GridGenerator::subdivided_hyper_rectangle(
      the_grid, repeats, point_1, point_2, true);
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::parallel::distributed::Triangulation<
        dim>::cell_iterator> >
      periodic_faces;
    dealii::GridTools::collect_periodic_faces(
      the_grid, 0, 1, 0, periodic_faces, dealii::Tensor<1, dim>({1., 0.}));
    dealii::GridTools::collect_periodic_faces(
      the_grid, 2, 3, 0, periodic_faces, dealii::Tensor<1, dim>({0., 1.}));
    the_grid.add_periodicity(periodic_faces);
  } // End of Francois's Example 1

  if (false) // Francois's example 2
  {
    std::vector<unsigned> repeats(dim, 1);
    repeats[0] = 1;
    dealii::Point<dim> point_1, point_2;
    point_1 = {0.4, -1.0};
    point_2 = {1.6, 1.0};
    dealii::GridGenerator::subdivided_hyper_rectangle(
      the_grid, repeats, point_1, point_2, true);
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::parallel::distributed::Triangulation<
        dim>::cell_iterator> >
      periodic_faces;
    dealii::GridTools::collect_periodic_faces(
      the_grid, 0, 1, 0, periodic_faces, dealii::Tensor<1, dim>({1.2, 0.}));
    dealii::GridTools::collect_periodic_faces(
      the_grid, 2, 3, 0, periodic_faces, dealii::Tensor<1, dim>({0., 2.}));
    the_grid.add_periodicity(periodic_faces);
  } // End of Francois's example 2

  // Dissertation example 2
  if (false)
  {
    std::vector<unsigned> repeats(dim, 1);
    repeats[0] = 300;
    dealii::Point<dim> point_1, point_2;
    point_1 = {0., 0.};
    point_2 = {40., .5};
    dealii::GridGenerator::subdivided_hyper_rectangle(
      the_grid, repeats, point_1, point_2, true);
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::parallel::distributed::Triangulation<
        dim>::cell_iterator> >
      periodic_faces;
    dealii::GridTools::collect_periodic_faces(
      the_grid, 0, 1, 0, periodic_faces, dealii::Tensor<1, dim>({40., 0.}));
    dealii::GridTools::collect_periodic_faces(
      the_grid, 2, 3, 0, periodic_faces, dealii::Tensor<1, dim>({0., 0.5}));
    the_grid.add_periodicity(periodic_faces);
  }
  // End of Dissertation example 2
}

template <int dim>
SolutionManager<dim>::~SolutionManager()
{
  if (comm_rank == 0)
  {
    convergence_result.close();
    execution_time.close();
  }
}

template <int dim>
void SolutionManager<dim>::solve(const unsigned &h_1, const unsigned &h_2)
{
  Vec sol_vec;

  if (false) // Diffusion test
  {
    BDFIntegrator time_integrator0(1.e-3, 1);
    hdg_model<dim, Diffusion> model0(this, &time_integrator0);
    for (unsigned h1 = h_1; h1 < h_2; ++h1)
    {
      if (h1 != h_1)
        time_integrator0.reset();
      refine_grid(h1, model0);
      model0.DoF_H_System.distribute_dofs(model0.DG_System);
      model0.DoF_H_Refine.distribute_dofs(model0.DG_Elem);

      model0.free_containers();
      model0.init_mesh_containers();
      model0.set_boundary_indicator();
      model0.count_globals();
      write_grid();
      model0.assign_initial_data(time_integrator0);

      solver_update_keys keys_0 =
        static_cast<solver_update_keys>(update_mat | update_rhs);
      model0.init_solver();
      model0.assemble_globals(keys_0);
      model0.solver->finish_assembly(keys_0);

      if (comm_size == 1)
      {
        Mat *the_global_mat = model0.solver->get_petsc_mat();
        MatView(*the_global_mat, PETSC_VIEWER_STDOUT_SELF);
      }

      model0.solver->form_factors(implicit_petsc_factor_type::mumps);
      model0.solver->solve_system(sol_vec);

      std::vector<double> local_sol_vec(
        model0.solver->get_local_part_of_global_vec(sol_vec, true));
      model0.compute_internal_dofs(local_sol_vec.data());

      vtk_visualizer(model0, 0);
    }
    model0.DoF_H_Refine.clear();
    model0.DoF_H_System.clear();
  }

  if (false) // GN_eps_0_beta_0 test
  {
    BDFIntegrator time_integrator0(1.e-3, 1);
    hdg_model<dim, GN_eps_0_beta_0> model0(this, &time_integrator0);
    for (unsigned h1 = h_1; h1 < h_2; ++h1)
    {
      if (h1 != h_1)
        time_integrator0.reset();
      refine_grid(h1, model0);
      model0.DoF_H_System.distribute_dofs(model0.DG_System);
      model0.DoF_H_Refine.distribute_dofs(model0.DG_Elem);

      model0.free_containers();
      model0.init_mesh_containers();
      model0.set_boundary_indicator();
      model0.count_globals();
      write_grid();
      model0.assign_initial_data(time_integrator0);

      solver_update_keys keys_0 =
        static_cast<solver_update_keys>(update_mat | update_rhs);
      model0.init_solver();
      model0.assemble_globals(keys_0);
      model0.solver->finish_assembly(keys_0);
      model0.solver->form_factors(implicit_petsc_factor_type::mumps);
      model0.solver->solve_system(sol_vec);

      std::vector<double> local_sol_vec(
        model0.solver->get_local_part_of_global_vec(sol_vec, true));
      model0.compute_internal_dofs(local_sol_vec.data());

      vtk_visualizer(model0, 0);
    }
    model0.DoF_H_Refine.clear();
    model0.DoF_H_System.clear();
  }

  if (false) // Implicit NSWE Jacobian test
  {
    Vec rand_dq, rhs_q, rhs_q_plus_dq, jacobian_by_dq, rhs_residual, to_be_zero;

    double small_factor = 1.e-7;
    BDFIntegrator time_integrator0(1e-3, 1);
    hdg_model<dim, NSWE> model0(this, &time_integrator0);

    refine_grid(1, model0);
    model0.DoF_H_System.distribute_dofs(model0.DG_System);
    model0.DoF_H_Refine.distribute_dofs(model0.DG_Elem);
    model0.free_containers();
    model0.init_mesh_containers();
    model0.set_boundary_indicator();
    model0.count_globals();

    VecCreateMPI(comm,
                 model0.n_global_DOFs_rank_owns,
                 model0.n_global_DOFs_on_all_ranks,
                 &rand_dq);
    VecDuplicate(rand_dq, &rhs_q);
    VecDuplicate(rand_dq, &rhs_q_plus_dq);
    VecDuplicate(rand_dq, &jacobian_by_dq);
    VecDuplicate(rand_dq, &rhs_residual);
    VecDuplicate(rand_dq, &to_be_zero);

    write_grid();
    model0.assign_initial_data(time_integrator0);
    model0.init_solver();
    solver_update_keys keys_0 =
      static_cast<solver_update_keys>(update_mat | update_rhs);
    model0.assemble_globals(keys_0);
    model0.solver->finish_assembly(keys_0);
    VecCopy(*(model0.solver->get_petsc_rhs()), rhs_q);

    PetscRandom rctx;
    PetscRandomCreate(comm, &rctx);
    PetscRandomSetType(rctx, PETSCRAND48);
    VecSetRandom(rand_dq, rctx);
    MatMult(*(model0.solver->get_petsc_mat()), rand_dq, jacobian_by_dq);

    VecScale(rand_dq, small_factor);
    std::vector<double> rand_dq_vec(
      model0.solver->get_local_part_of_global_vec(rand_dq, true));
    model0.compute_internal_dofs(rand_dq_vec.data());

    keys_0 = static_cast<solver_update_keys>(update_rhs);
    model0.reinit_solver(keys_0);
    model0.assemble_globals(keys_0);
    model0.solver->finish_assembly(keys_0);

    VecCopy(*(model0.solver->get_petsc_rhs()), rhs_q_plus_dq);

    std::vector<double> rhs_q_vec(
      model0.solver->get_local_part_of_global_vec(rhs_q, true));
    std::vector<double> rhs_q_plus_dq_vec(
      model0.solver->get_local_part_of_global_vec(rhs_q_plus_dq, true));
    std::vector<double> jacobian_by_dq_vec(
      model0.solver->get_local_part_of_global_vec(jacobian_by_dq, true));

    std::cout << " ------------- " << std::endl;
    for (unsigned i1 = 0; i1 < rhs_q_vec.size(); ++i1)
      /*
      std::cout << rhs_q_plus_dq_vec[i1] - rhs_q_vec[i1] << "  "
                << jacobian_by_dq_vec[i1] << "  "
                << (rhs_q_plus_dq_vec[i1] - rhs_q_vec[i1]) / small_factor +
                     jacobian_by_dq_vec[i1]
                << std::endl;
      */
      std::cout << (rhs_q_plus_dq_vec[i1] - rhs_q_vec[i1]) / small_factor +
                     jacobian_by_dq_vec[i1]
                << std::endl;

    VecWAXPY(rhs_residual, -1, rhs_q, rhs_q_plus_dq);
    VecScale(rhs_residual, 1.e10);
    VecWAXPY(to_be_zero, -1, jacobian_by_dq, rhs_residual);

    std::vector<double> rhs_residual_vec(
      model0.solver->get_local_part_of_global_vec(rhs_residual, true));

    std::vector<double> to_be_zero_vec(
      model0.solver->get_local_part_of_global_vec(to_be_zero, true));

    PetscRandomDestroy(&rctx);
    VecDestroy(&rand_dq);
    VecDestroy(&rhs_q);
  }

  if (false) // Implicit NSWE test
  {
    double t11, t12, t21, t22, t31, t32, local_ops_time = 0.,
                                         global_ops_time = 0.;
    BDFIntegrator time_integrator0(1e-3, 2);
    hdg_model<dim, NSWE> model0(this, &time_integrator0);

    for (unsigned h1 = h_1; h1 < h_2; ++h1)
    {
      if (h1 != h_1)
        time_integrator0.reset();

      refine_grid(h1, model0);
      model0.DoF_H_System.distribute_dofs(model0.DG_System);
      model0.DoF_H_Refine.distribute_dofs(model0.DG_Elem);

      model0.free_containers();
      model0.init_mesh_containers();
      model0.set_boundary_indicator();
      model0.count_globals();
      //      write_grid();
      model0.assign_initial_data(time_integrator0);

      for (unsigned i_time = 0; i_time < 5000; ++i_time)
      {
        bool iteration_required = false;
        unsigned max_iter = 200;
        unsigned num_iter = 0;

        double dt_local_ops = 0.;
        double dt_global_ops = 0.;

        do
        {
          solver_update_keys keys_0 =
            static_cast<solver_update_keys>(update_mat | update_rhs);
          if (i_time == 0 && num_iter == 0)
            model0.init_solver();
          else
            model0.reinit_solver(keys_0);

          t11 = MPI_Wtime();
          model0.assemble_globals(keys_0);
          model0.solver->finish_assembly(keys_0);
          t12 = MPI_Wtime();

          /*
          std::vector<double> rhs0(model0.solver->get_local_part_of_global_vec(
            *(model0.solver->get_petsc_rhs()), false));
          */

          t21 = MPI_Wtime();
          model0.solver->form_factors(implicit_petsc_factor_type::mumps);
          model0.solver->solve_system(sol_vec);
          std::vector<double> local_sol_vec(
            model0.solver->get_local_part_of_global_vec(sol_vec, true));
          t22 = MPI_Wtime();

          t31 = MPI_Wtime();
          iteration_required =
            model0.compute_internal_dofs(local_sol_vec.data());
          t32 = MPI_Wtime();
          ++num_iter;

          dt_local_ops += (t12 - t11 + t32 - t31);
          dt_global_ops += (t22 - t21);

        } while (iteration_required && num_iter < max_iter);

        if (i_time % 20 == 0)
          //        vtk_visualizer(model0, max_iter * i_time + num_iter);
          vtk_visualizer(model0, i_time);

        if (comm_rank == 0)
          std::cout << "time: " << i_time << ", local ops: " << dt_local_ops
                    << ", global_ops: " << dt_global_ops << std::endl;

        local_ops_time += dt_local_ops;
        global_ops_time += dt_global_ops;
        time_integrator0.go_forward();
      }

      if (comm_rank == 0)
        std::cout << "Total local ops: " << local_ops_time
                  << ", total global_ops: " << global_ops_time << std::endl;

      model0.DoF_H_Refine.clear();
      model0.DoF_H_System.clear();
    }
  }

  if (false) // Explicit NSWE Jacobian test
  {
    explicit_RKn<4, original_RK> rk4_0(1.e-3);
    explicit_hdg_model<dim, explicit_nswe> model0(this, &rk4_0);
    refine_grid(1, model0);
    model0.DoF_H_System.distribute_dofs(model0.DG_System);
    model0.DoF_H_Refine.distribute_dofs(model0.DG_Elem);
    model0.free_containers();
    model0.init_mesh_containers();
    model0.set_boundary_indicator();
    model0.count_globals();
    write_grid();

    Vec rand_dqh, jacobian_by_dqh, rhs_qh0, rhs_qh1;
    VecCreateMPI(comm,
                 model0.n_global_DOFs_rank_owns,
                 model0.n_global_DOFs_on_all_ranks,
                 &rand_dqh);
    VecDuplicate(rand_dqh, &jacobian_by_dqh);
    VecDuplicate(rand_dqh, &rhs_qh0);
    VecDuplicate(rand_dqh, &rhs_qh1);
    PetscRandom rctx;
    PetscRandomCreate(comm, &rctx);
    PetscRandomSetType(rctx, PETSCRAND48);
    VecSetRandom(rand_dqh, rctx);

    double small_fac = 1e-7;
    rk4_0.ready_for_next_step();
    model0.init_solver();

    model0.assign_initial_data(rk4_0);
    solver_update_keys keys_0 =
      static_cast<solver_update_keys>(update_mat | update_rhs);
    model0.assemble_globals(keys_0);
    model0.solver->finish_assembly(keys_0);

    VecCopy(*(model0.solver->get_petsc_rhs()), rhs_qh0);
    MatMult(*(model0.solver->get_petsc_mat()), rand_dqh, jacobian_by_dqh);

    VecScale(rand_dqh, small_fac);
    std::vector<double> local_sol_vec =
      model0.solver->get_local_part_of_global_vec(rand_dqh, true);
    model0.check_for_next_iter(local_sol_vec.data());

    keys_0 = static_cast<solver_update_keys>(update_rhs);
    model0.reinit_solver(keys_0);
    model0.assemble_globals(keys_0);
    model0.solver->finish_assembly(keys_0);

    VecCopy(*(model0.solver->get_petsc_rhs()), rhs_qh1);

    std::vector<double> rhs_q_vec(
      model0.solver->get_local_part_of_global_vec(rhs_qh0, true));
    std::vector<double> rhs_q_plus_dq_vec(
      model0.solver->get_local_part_of_global_vec(rhs_qh1, true));
    std::vector<double> jacobian_by_dq_vec(
      model0.solver->get_local_part_of_global_vec(jacobian_by_dqh, true));

    std::cout << " ------------- " << std::endl;
    for (unsigned i1 = 0; i1 < rhs_q_vec.size(); ++i1)
      std::cout << (rhs_q_plus_dq_vec[i1] - rhs_q_vec[i1]) / small_fac +
                     jacobian_by_dq_vec[i1]
                << std::endl;

    model0.DoF_H_Refine.clear();
    model0.DoF_H_System.clear();

    VecDestroy(&rand_dqh);
    VecDestroy(&jacobian_by_dqh);
    VecDestroy(&rhs_qh0);
    VecDestroy(&rhs_qh1);
  }

  if (false) // Explicit NSWE test
  {
    double t11, t12, t21, t22, t31, t32, local_ops_time = 0.,
                                         global_ops_time = 0.;
    explicit_RKn<4, original_RK> rk4_0(2.e-3);
    explicit_hdg_model<dim, explicit_nswe> model0(this, &rk4_0);
    for (unsigned h1 = h_1; h1 < h_2; ++h1)
    {
      if (h1 != h_1)
        rk4_0.reset();

      refine_grid(h1, model0);

      model0.DoF_H_System.distribute_dofs(model0.DG_System);
      model0.DoF_H_Refine.distribute_dofs(model0.DG_Elem);
      model0.free_containers();
      model0.init_mesh_containers();
      model0.set_boundary_indicator();
      model0.count_globals();
      //      write_grid();
      model0.assign_initial_data(rk4_0);

      std::vector<double> local_sol_vec0;
      local_sol_vec0.reserve(model0.n_local_DOFs_on_this_rank);
      for (unsigned i_time = 0; i_time < 5000; ++i_time)
      {
        double dt_local_ops = 0.;
        double dt_global_ops = 0.;
        while (!rk4_0.ready_for_next_step())
        {
          bool iteration_required = false;
          unsigned max_iter = 50;
          unsigned num_iter = 0;
          do
          {
            solver_update_keys keys_0 =
              static_cast<solver_update_keys>(update_mat | update_rhs);
            if (i_time == 0 && num_iter == 0)
              model0.init_solver();
            else
              model0.reinit_solver(keys_0);

            t11 = MPI_Wtime();
            model0.assemble_globals(keys_0);
            model0.solver->finish_assembly(keys_0);
            t12 = MPI_Wtime();

            t21 = MPI_Wtime();
            model0.solver->form_factors(implicit_petsc_factor_type::mumps);
            model0.solver->solve_system(sol_vec);
            local_sol_vec0 =
              model0.solver->get_local_part_of_global_vec(sol_vec, true);
            t22 = MPI_Wtime();

            /*
            double trace_increment_norm;
            VecNorm(sol_vec, NORM_2, &trace_increment_norm);
            std::cout << trace_increment_norm << std::endl;
            */

            iteration_required =
              model0.check_for_next_iter(local_sol_vec0.data());

            ++num_iter;

            if (comm_rank == 0 && (!iteration_required || num_iter == max_iter))
              std::cout << num_iter << std::endl;

            dt_local_ops += (t12 - t11);
            dt_global_ops += (t22 - t21);

          } while (iteration_required && num_iter < max_iter);
        }
        t31 = MPI_Wtime();
        model0.compute_internal_dofs(local_sol_vec0.data());
        t32 = MPI_Wtime();

        dt_local_ops += (t32 - t31);

        if (i_time % 10 == 0)
          //        vtk_visualizer(model0, max_iter * i_time + num_iter);
          vtk_visualizer(model0, i_time);

        local_ops_time += dt_local_ops;
        global_ops_time += dt_global_ops;

        if (comm_rank == 0)
          std::cout << "time: " << i_time << ", local ops: " << dt_local_ops
                    << ", global_ops: " << dt_global_ops << std::endl;
      }
      if (comm_rank == 0)
        std::cout << "Total local ops: " << local_ops_time
                  << ", total global_ops: " << global_ops_time << std::endl;

      model0.DoF_H_Refine.clear();
      model0.DoF_H_System.clear();
    }
  }

  if (false) // Implicit Advection test
  {
    BDFIntegrator time_integrator1(1.e-3, 1);
    hdg_model<dim, AdvectionDiffusion> model1(this, &time_integrator1);

    for (unsigned h1 = h_1; h1 < h_2; ++h1)
    {
      if (h1 != h_1)
        time_integrator1.reset();

      double t11, t12, t21, t22, t13, t23;

      refine_grid(h1, model1);
      //    free_containers();

      model1.DoF_H_System.distribute_dofs(model1.DG_System);
      model1.DoF_H_Refine.distribute_dofs(model1.DG_Elem);

      model1.free_containers();
      model1.init_mesh_containers();
      model1.set_boundary_indicator();
      //    write_grid();
      model1.count_globals();

      for (unsigned i_time = 0; i_time < 200; ++i_time)
      {
        t11 = t12 = t21 = t22 = t13 = t23 = 0.;
        if (time_integrator1.time_level == 0)
        {
          solver_update_keys keys_ =
            static_cast<solver_update_keys>(update_mat | update_rhs);
          model1.init_solver();
          model1.assign_initial_data(time_integrator1);
          t11 = MPI_Wtime();
          model1.assemble_globals(keys_);
          model1.solver->finish_assembly(keys_);
          t12 = MPI_Wtime();
          t21 = MPI_Wtime();
          model1.solver->form_factors(implicit_petsc_factor_type::mumps);
          model1.solver->solve_system(sol_vec);
          t22 = MPI_Wtime();
        }
        else if (time_integrator1.time_level <=
                 time_integrator1.time_level_mat_calc_required())
        {
          solver_update_keys keys_ =
            static_cast<solver_update_keys>(update_mat | update_rhs);
          model1.reinit_solver(keys_);
          t11 = MPI_Wtime();
          model1.assemble_globals(keys_);
          model1.solver->finish_assembly(keys_);
          t12 = MPI_Wtime();
          t21 = MPI_Wtime();
          model1.solver->form_factors(implicit_petsc_factor_type::mumps);
          model1.solver->solve_system(sol_vec);
          t22 = MPI_Wtime();
        }
        else
        {
          solver_update_keys keys_ =
            static_cast<solver_update_keys>(update_rhs);
          model1.reinit_solver(keys_);
          t11 = MPI_Wtime();
          model1.assemble_globals(keys_);
          model1.solver->finish_assembly(keys_);
          t12 = MPI_Wtime();
          t21 = MPI_Wtime();
          model1.solver->solve_system(sol_vec);
          t22 = MPI_Wtime();
        }

        t13 = MPI_Wtime();
        std::vector<double> local_sol_vec(
          model1.solver->get_local_part_of_global_vec(sol_vec, true));
        model1.compute_internal_dofs(local_sol_vec.data());
        t23 = MPI_Wtime();

        if (i_time % 1 == 0)
          vtk_visualizer(model1, i_time);

        if (comm_rank == 0)
          std::cout << t12 - t11 << " " << t22 - t21 << " " << t23 - t13
                    << std::endl;
        time_integrator1.go_forward();
      }
    }
    model1.DoF_H_Refine.clear();
    model1.DoF_H_System.clear();
  }

  if (true) // Explicit GN Dispersive part test
  {
    double t11, t12, t21, t22, t31, t32, local_ops_time = 0.,
                                         global_ops_time = 0.;
    explicit_RKn<4, original_RK> rk4_0(2.5e-3);
    explicit_RKn<4, original_RK> rk4_1(5.0e-3);
    explicit_hdg_model<dim, explicit_nswe> model0(this, &rk4_0);
    hdg_model_with_explicit_rk<dim, explicit_gn_dispersive> model1(this,
                                                                   &rk4_1);
    for (unsigned h1 = h_1; h1 < h_2; ++h1)
    {
      if (h1 != h_1)
      {
        rk4_0.reset();
        rk4_1.reset();
      }

      refine_grid(h1, model0);

      model0.DoF_H_System.distribute_dofs(model0.DG_System);
      model0.DoF_H_Refine.distribute_dofs(model0.DG_Elem);
      model0.free_containers();
      model0.init_mesh_containers();
      model0.set_boundary_indicator();
      model0.count_globals();
      //      write_grid();
      model0.assign_initial_data(rk4_0);

      model1.DoF_H_System.distribute_dofs(model1.DG_System);
      model1.DoF_H_Refine.distribute_dofs(model1.DG_Elem);
      model1.free_containers();
      model1.init_mesh_containers();
      model1.set_boundary_indicator();
      model1.count_globals();
      //      model1.assign_initial_data(rk4_1);

      std::vector<double> local_sol_vec0;
      local_sol_vec0.reserve(model0.n_local_DOFs_on_this_rank);

      std::vector<double> local_sol_vec1;
      local_sol_vec1.reserve(model1.n_local_DOFs_on_this_rank);

      bool model1_init_fuse = true;

      for (unsigned i_time = 0; i_time < 2000; ++i_time)
      {
        double dt_local_ops = 0.;
        double dt_global_ops = 0.;

        //
        //  First phase of time splitting
        //
        while (!rk4_0.ready_for_next_step())
        {
          bool iteration_required = false;
          unsigned max_iter = 20;
          unsigned num_iter = 0;
          do
          {
            solver_update_keys keys_0 =
              static_cast<solver_update_keys>(update_mat | update_rhs);
            if (i_time == 0 && num_iter == 0)
              model0.init_solver();
            else
              model0.reinit_solver(keys_0);

            t11 = MPI_Wtime();
            model0.assemble_globals(keys_0);
            model0.solver->finish_assembly(keys_0);
            t12 = MPI_Wtime();

            t21 = MPI_Wtime();
            model0.solver->form_factors(implicit_petsc_factor_type::mumps);
            model0.solver->solve_system(sol_vec);
            local_sol_vec0 =
              model0.solver->get_local_part_of_global_vec(sol_vec, true);
            t22 = MPI_Wtime();

            iteration_required =
              model0.check_for_next_iter(local_sol_vec0.data());

            ++num_iter;

            if (comm_rank == 0 && (!iteration_required || num_iter == max_iter))
              std::cout << num_iter << std::endl;

            dt_local_ops += (t12 - t11);
            dt_global_ops += (t22 - t21);
          } while (iteration_required && num_iter < max_iter);
        }
        t31 = MPI_Wtime();
        model0.compute_internal_dofs(local_sol_vec0.data());
        t32 = MPI_Wtime();
        //        if (i_time % 10 == 0)
        //          vtk_visualizer(model0, i_time * 3);

        //
        // Second phase of time splitting.
        //
        if (i_time >= 0)
        {
          model1.get_results_from_another_model(model0);
          while (!rk4_1.ready_for_next_step())
          {
            bool iteration_required = false;
            unsigned max_iter = 1;
            unsigned num_iter = 0;
            do
            {
              solver_update_keys keys_0 =
                static_cast<solver_update_keys>(update_mat | update_rhs);

              if (model1_init_fuse)
              {
                model1.init_solver(
                  &model0); // Flux generator is also initiated here.
              }
              else
              {
                model1.reinit_solver(
                  keys_0); // Flux generator is also reinitiated here.
                model1_init_fuse = false;
              }

              t11 = MPI_Wtime();

              //
              // The place that we compute average fluxes !
              //
              model1.sorry_for_this_boolshit = true;

              model1.assemble_trace_of_conserved_vars(&model0);
              model1.flux_gen1->finish_assembly(model1.flux_gen1->face_count);
              model1.flux_gen1->finish_assembly(
                model1.flux_gen1->conserved_vars_flux);
              model1.flux_gen1->finish_assembly(model1.flux_gen1->V_dot_n_sum);

              std::vector<double> local_conserved_vars_sum, local_face_count,
                local_V_jumps;
              local_conserved_vars_sum.reserve(
                model0.n_local_DOFs_on_this_rank);
              local_face_count.reserve(model0.n_local_DOFs_on_this_rank);
              local_V_jumps.reserve(model0.n_local_DOFs_on_this_rank);
              local_conserved_vars_sum =
                model1.flux_gen1->get_local_part_of_global_vec(
                  (model1.flux_gen1->conserved_vars_flux));
              local_face_count = model1.flux_gen1->get_local_part_of_global_vec(
                (model1.flux_gen1->face_count));
              local_V_jumps = model1.flux_gen1->get_local_part_of_global_vec(
                (model1.flux_gen1->V_dot_n_sum));
              // In the following method, we also compute the derivatives of
              // primitive variables in elements and assemble the
              // V_x_flux, V_y_flux.
              model1.compute_and_sum_grad_prim_vars(
                &model0,
                local_conserved_vars_sum.data(),
                local_face_count.data(),
                local_V_jumps.data());
              std::vector<double> local_V_x_flux, local_V_y_flux;
              local_V_x_flux.reserve(model0.n_local_DOFs_on_this_rank);
              local_V_y_flux.reserve(model0.n_local_DOFs_on_this_rank);
              model1.flux_gen1->finish_assembly(model1.flux_gen1->V_x_sum);
              model1.flux_gen1->finish_assembly(model1.flux_gen1->V_y_sum);
              local_V_x_flux = model1.flux_gen1->get_local_part_of_global_vec(
                (model1.flux_gen1->V_x_sum));
              local_V_y_flux = model1.flux_gen1->get_local_part_of_global_vec(
                (model1.flux_gen1->V_y_sum));
              //
              // We also compute the grad_grad_V in the assemble_globals.
              // Hence, we need to send model0 and the local_V_x_flux and
              // local_V_y_flux to this function for computation of the
              // average flux of grad_V.
              //
              model1.assemble_globals(
                &model0, local_V_x_flux.data(), local_V_y_flux.data(), keys_0);
              model1.solver->finish_assembly(keys_0);
              t12 = MPI_Wtime();

              t21 = MPI_Wtime();
              model1.solver->form_factors(implicit_petsc_factor_type::mumps);
              model1.solver->solve_system(sol_vec);
              local_sol_vec1 =
                model1.solver->get_local_part_of_global_vec(sol_vec, true);
              t22 = MPI_Wtime();

              iteration_required =
                model1.check_for_next_iter(local_sol_vec1.data());

              ++num_iter;

              if (comm_rank == 0 &&
                  (!iteration_required || num_iter == max_iter))
                std::cout << num_iter << std::endl;

              dt_local_ops += (t12 - t11);
              dt_global_ops += (t22 - t21);

            } while (iteration_required && num_iter < max_iter);
          }
          t31 = MPI_Wtime();
          model1.compute_internal_dofs(local_sol_vec1.data());
          t32 = MPI_Wtime();
          //        if (i_time % 1 == 0)
          //            vtk_visualizer(model1, i_time * 3 + 1);
          model0.get_results_from_another_model(model1);
        }

        //
        // Third phase of time splitting
        //
        while (!rk4_0.ready_for_next_step())
        {
          bool iteration_required = false;
          unsigned max_iter = 20;
          unsigned num_iter = 0;
          do
          {
            solver_update_keys keys_0 =
              static_cast<solver_update_keys>(update_mat | update_rhs);
            //
            // if (i_time == 0 && num_iter == 0)
            //  model0.init_solver();
            // else
            //
            model0.reinit_solver(keys_0);

            t11 = MPI_Wtime();
            model0.assemble_globals(keys_0);
            model0.solver->finish_assembly(keys_0);
            t12 = MPI_Wtime();

            t21 = MPI_Wtime();
            model0.solver->form_factors(implicit_petsc_factor_type::mumps);
            model0.solver->solve_system(sol_vec);
            local_sol_vec0 =
              model0.solver->get_local_part_of_global_vec(sol_vec, true);
            t22 = MPI_Wtime();

            iteration_required =
              model0.check_for_next_iter(local_sol_vec0.data());

            ++num_iter;

            if (comm_rank == 0 && (!iteration_required || num_iter == max_iter))
              std::cout << num_iter << std::endl;

            dt_local_ops += (t12 - t11);
            dt_global_ops += (t22 - t21);
          } while (iteration_required && num_iter < max_iter);
        }
        t31 = MPI_Wtime();
        model0.compute_internal_dofs(local_sol_vec0.data());
        t32 = MPI_Wtime();
        if (i_time % 2 == 0)
          vtk_visualizer(model0, i_time * 3 + 2);

        //
        // Time splitting finished. Going for calculation of the results.
        //
        dt_local_ops += (t32 - t31);

        local_ops_time += dt_local_ops;
        global_ops_time += dt_global_ops;

        if (comm_rank == 0)
          std::cout << "time: " << i_time << ", local ops: " << dt_local_ops
                    << ", global_ops: " << dt_global_ops << std::endl;
      }

      if (comm_rank == 0)
        std::cout << "Total local ops: " << local_ops_time
                  << ", total global_ops: " << global_ops_time << std::endl;

      model0.DoF_H_Refine.clear();
      model0.DoF_H_System.clear();
      model1.DoF_H_Refine.clear();
      model1.DoF_H_System.clear();
    }
  }

  if (false) // Explicit GN Dispersive without splitting
  {
    double t11, t12, t21, t22, t31, t32, local_ops_time = 0.,
                                         global_ops_time = 0.;
    explicit_RKn<4, original_RK> rk4_0(2.5e-3);
    explicit_RKn<4, original_RK> rk4_1(5.0e-3);
    explicit_hdg_model<dim, explicit_nswe_modif> model0(this, &rk4_0);
    hdg_model_with_explicit_rk<dim, explicit_gn_dispersive_modif> model1(
      this, &rk4_1);
    for (unsigned h1 = h_1; h1 < h_2; ++h1)
    {
      if (h1 != h_1)
      {
        rk4_0.reset();
        rk4_1.reset();
      }

      refine_grid(h1, model0);

      model0.DoF_H_System.distribute_dofs(model0.DG_System);
      model0.DoF_H_Refine.distribute_dofs(model0.DG_Elem);
      model0.free_containers();
      model0.init_mesh_containers();
      model0.set_boundary_indicator();
      model0.count_globals();
      //      write_grid();
      model0.assign_initial_data(rk4_0);

      model1.DoF_H_System.distribute_dofs(model1.DG_System);
      model1.DoF_H_Refine.distribute_dofs(model1.DG_Elem);
      model1.free_containers();
      model1.init_mesh_containers();
      model1.set_boundary_indicator();
      model1.count_globals();
      //      model1.assign_initial_data(rk4_1);

      std::vector<double> local_sol_vec0;
      local_sol_vec0.reserve(model0.n_local_DOFs_on_this_rank);

      std::vector<double> local_sol_vec1;
      local_sol_vec1.reserve(model1.n_local_DOFs_on_this_rank);

      for (unsigned i_time = 0; i_time < 1000; ++i_time)
      {
        double dt_local_ops = 0.;
        double dt_global_ops = 0.;

        //
        //  First phase of time splitting
        //
        while (!rk4_0.ready_for_next_step())
        {
          bool iteration_required = false;
          unsigned max_iter = 20;
          unsigned num_iter = 0;
          do
          {
            solver_update_keys keys_0 =
              static_cast<solver_update_keys>(update_mat | update_rhs);
            if (i_time == 0 && num_iter == 0)
              model0.init_solver();
            else
              model0.reinit_solver(keys_0);

            t11 = MPI_Wtime();
            model0.assemble_globals(keys_0);
            model0.solver->finish_assembly(keys_0);
            t12 = MPI_Wtime();

            t21 = MPI_Wtime();
            model0.solver->form_factors(implicit_petsc_factor_type::mumps);
            model0.solver->solve_system(sol_vec);
            local_sol_vec0 =
              model0.solver->get_local_part_of_global_vec(sol_vec, true);
            t22 = MPI_Wtime();

            iteration_required =
              model0.check_for_next_iter(local_sol_vec0.data());

            ++num_iter;

            if (comm_rank == 0 && (!iteration_required || num_iter == max_iter))
              std::cout << num_iter << std::endl;

            dt_local_ops += (t12 - t11);
            dt_global_ops += (t22 - t21);
          } while (iteration_required && num_iter < max_iter);
        }
        t31 = MPI_Wtime();
        model0.compute_internal_dofs(local_sol_vec0.data());
        t32 = MPI_Wtime();
        if (i_time % 1 == 0)
          //        vtk_visualizer(model0, max_iter * i_time + num_iter);
          vtk_visualizer(model0, i_time * 3);

        //
        // Second phase of time splitting.
        //
        /*
        {
          model1.get_results_from_another_model(model0);
          while (!rk4_1.ready_for_next_step())
          {
            bool iteration_required = false;
            unsigned max_iter = 1;
            unsigned num_iter = 0;
            do
            {
              solver_update_keys keys_0 =
                static_cast<solver_update_keys>(update_mat | update_rhs);
              if (i_time == 0 && num_iter == 0)
                model1.init_solver(
                  &model0); // Flux generator is also initiated here.
              else
                model1.reinit_solver(
                  keys_0); // Flux generator is also reinitiated here.

              t11 = MPI_Wtime();

              //
              // The place that we compute average fluxes !
              //
              model1.sorry_for_this_boolshit = true;

              model1.assemble_trace_of_conserved_vars(&model0);
              model1.flux_gen->finish_assembly(model1.flux_gen->face_count);
              model1.flux_gen->finish_assembly(
                model1.flux_gen->conserved_vars_flux);
              model1.flux_gen->finish_assembly(model1.flux_gen->V_dot_n_sum);

              std::vector<double> local_conserved_vars_sum, local_face_count,
                local_V_jumps;
              local_conserved_vars_sum.reserve(
                model0.n_local_DOFs_on_this_rank);
              local_face_count.reserve(model0.n_local_DOFs_on_this_rank);
              local_V_jumps.reserve(model0.n_local_DOFs_on_this_rank);
              local_conserved_vars_sum =
                model1.flux_gen->get_local_part_of_global_vec(
                  (model1.flux_gen->conserved_vars_flux));
              local_face_count = model1.flux_gen->get_local_part_of_global_vec(
                (model1.flux_gen->face_count));
              local_V_jumps = model1.flux_gen->get_local_part_of_global_vec(
                (model1.flux_gen->V_dot_n_sum));
              // In the following method, we also compute the derivatives of
              // primitive variables in elements and assemble the
              // V_x_flux, V_y_flux.
              model1.compute_and_sum_grad_prim_vars(
                &model0,
                local_conserved_vars_sum.data(),
                local_face_count.data(),
                local_V_jumps.data());
              std::vector<double> local_V_x_flux, local_V_y_flux;
              local_V_x_flux.reserve(model0.n_local_DOFs_on_this_rank);
              local_V_y_flux.reserve(model0.n_local_DOFs_on_this_rank);
              model1.flux_gen->finish_assembly(model1.flux_gen->V_x_sum);
              model1.flux_gen->finish_assembly(model1.flux_gen->V_y_sum);
              local_V_x_flux = model1.flux_gen->get_local_part_of_global_vec(
                (model1.flux_gen->V_x_sum));
              local_V_y_flux = model1.flux_gen->get_local_part_of_global_vec(
                (model1.flux_gen->V_y_sum));
              //
              // We also compute the grad_grad_V in the assemble_globals.
              // Hence, we need to send model0 and the local_V_x_flux and
              // local_V_y_flux to this function for computation of the
              // average flux of grad_V.
              //
              model1.assemble_globals(
                &model0, local_V_x_flux.data(), local_V_y_flux.data(), keys_0);
              model1.solver->finish_assembly(keys_0);
              t12 = MPI_Wtime();

              t21 = MPI_Wtime();
              model1.solver->form_factors(implicit_petsc_factor_type::mumps);
              model1.solver->solve_system(sol_vec);
              local_sol_vec1 =
                model1.solver->get_local_part_of_global_vec(sol_vec, true);
              t22 = MPI_Wtime();

              iteration_required =
                model1.check_for_next_iter(local_sol_vec1.data());

              ++num_iter;

              if (comm_rank == 0 &&
                  (!iteration_required || num_iter == max_iter))
                std::cout << num_iter << std::endl;

              dt_local_ops += (t12 - t11);
              dt_global_ops += (t22 - t21);

            } while (iteration_required && num_iter < max_iter);
          }
          t31 = MPI_Wtime();
          model1.compute_internal_dofs(local_sol_vec1.data());
          t32 = MPI_Wtime();
          if (i_time % 1 == 0)
            //        vtk_visualizer(model0, max_iter * i_time + num_iter);
            vtk_visualizer(model1, i_time * 3 + 1);
          model0.get_results_from_another_model(model1);
        }
        */

        //
        // Third phase of time splitting
        //
        /*
        while (!rk4_0.ready_for_next_step())
        {
          bool iteration_required = false;
          unsigned max_iter = 20;
          unsigned num_iter = 0;
          do
          {
            solver_update_keys keys_0 =
              static_cast<solver_update_keys>(update_mat | update_rhs);
            //
            // if (i_time == 0 && num_iter == 0)
            //  model0.init_solver();
            // else
            //
            model0.reinit_solver(keys_0);

            t11 = MPI_Wtime();
            model0.assemble_globals(keys_0);
            model0.solver->finish_assembly(keys_0);
            t12 = MPI_Wtime();

            t21 = MPI_Wtime();
            model0.solver->form_factors(implicit_petsc_factor_type::mumps);
            model0.solver->solve_system(sol_vec);
            local_sol_vec0 =
              model0.solver->get_local_part_of_global_vec(sol_vec, true);
            t22 = MPI_Wtime();

            iteration_required =
              model0.check_for_next_iter(local_sol_vec0.data());

            ++num_iter;

            if (comm_rank == 0 && (!iteration_required || num_iter == max_iter))
              std::cout << num_iter << std::endl;

            dt_local_ops += (t12 - t11);
            dt_global_ops += (t22 - t21);
          } while (iteration_required && num_iter < max_iter);
        }
        t31 = MPI_Wtime();
        model0.compute_internal_dofs(local_sol_vec0.data());
        t32 = MPI_Wtime();
        if (i_time % 5 == 0)
          //        vtk_visualizer(model0, max_iter * i_time + num_iter);
          vtk_visualizer(model0, i_time * 3 + 2);
        */

        //
        // Time splitting finished. Going for calculation of the results.
        //
        dt_local_ops += (t32 - t31);

        local_ops_time += dt_local_ops;
        global_ops_time += dt_global_ops;

        if (comm_rank == 0)
          std::cout << "time: " << i_time << ", local ops: " << dt_local_ops
                    << ", global_ops: " << dt_global_ops << std::endl;
      }

      if (comm_rank == 0)
        std::cout << "Total local ops: " << local_ops_time
                  << ", total global_ops: " << global_ops_time << std::endl;

      model0.DoF_H_Refine.clear();
      model0.DoF_H_System.clear();
      model1.DoF_H_Refine.clear();
      model1.DoF_H_System.clear();
    }
  }

  VecDestroy(&sol_vec);
}

/*!
 * \brief Applies one or \c n refinement cycles on the grid.
 * \details When the grid is created (or already exists),
 * this function should be called to
 * refine the mesh. If we have not done any refinements, i.e.
 * SolutionManager::refn_cycle is zero, this function performs \c n cycles
 * of uniform refinement over the whole mesh (even if mesh adaptivity
 * is on). When \c refn_cycle is not zero, it performs only one
 * refinement cycle (independant of the given argument \c n).
 * In the latter case, the refinement type will be
 * adaptive or uniform according to the variable SolutionManager::adaptive_on.
 * In any case, SolutionManager::refn_cycle will be updated to reflect the
 * total number of applied refinement cycles.
 */
template <int dim>
template <template <int> class CellType>
void SolutionManager<dim>::refine_grid(int n, hdg_model<dim, CellType> &model)
{
  if (n != 0 && refn_cycle == 0)
  {
    the_grid.refine_global(n);
    refn_cycle += n;
  }
  else if (n != 0 && !adaptive_on)
  {
    the_grid.refine_global(1);
    ++refn_cycle;
  }
  else if (n != 0)
  {
    dealii::Vector<float> estimated_error_per_cell(the_grid.n_active_cells());
    dealii::KellyErrorEstimator<dim>::estimate(
      model.DoF_H_Refine,
      dealii::QGauss<dim - 1>(quad_order),
      typename dealii::FunctionMap<dim>::type(),
      refine_solu,
      estimated_error_per_cell);
    dealii::parallel::distributed::GridRefinement::
      refine_and_coarsen_fixed_number(
        the_grid, estimated_error_per_cell, 0.3, 0.03);
    the_grid.execute_coarsening_and_refinement();
    ++refn_cycle;
  }
}

template <int dim>
template <template <int> class CellType>
void SolutionManager<dim>::refine_grid(int n,
                                       explicit_hdg_model<dim, CellType> &model)
{
  if (n != 0 && refn_cycle == 0)
  {
    the_grid.refine_global(n);
    refn_cycle += n;
  }
  else if (n != 0 && !adaptive_on)
  {
    the_grid.refine_global(1);
    ++refn_cycle;
  }
  else if (n != 0)
  {
    dealii::Vector<float> estimated_error_per_cell(the_grid.n_active_cells());
    dealii::KellyErrorEstimator<dim>::estimate(
      model.DoF_H_Refine,
      dealii::QGauss<dim - 1>(quad_order),
      typename dealii::FunctionMap<dim>::type(),
      refine_solu,
      estimated_error_per_cell);
    dealii::parallel::distributed::GridRefinement::
      refine_and_coarsen_fixed_number(
        the_grid, estimated_error_per_cell, 0.3, 0.03);
    the_grid.execute_coarsening_and_refinement();
    ++refn_cycle;
  }
}

template <int dim>
template <template <int> class CellType>
void SolutionManager<dim>::refine_grid(
  int n, hdg_model_with_explicit_rk<dim, CellType> &model)
{
  if (n != 0 && refn_cycle == 0)
  {
    the_grid.refine_global(n);
    refn_cycle += n;
  }
  else if (n != 0 && !adaptive_on)
  {
    the_grid.refine_global(1);
    ++refn_cycle;
  }
  else if (n != 0)
  {
    dealii::Vector<float> estimated_error_per_cell(the_grid.n_active_cells());
    dealii::KellyErrorEstimator<dim>::estimate(
      model.DoF_H_Refine,
      dealii::QGauss<dim - 1>(quad_order),
      typename dealii::FunctionMap<dim>::type(),
      refine_solu,
      estimated_error_per_cell);
    dealii::parallel::distributed::GridRefinement::
      refine_and_coarsen_fixed_number(
        the_grid, estimated_error_per_cell, 0.3, 0.03);
    the_grid.execute_coarsening_and_refinement();
    ++refn_cycle;
  }
}

template <int dim>
template <template <int> class CellType>
void SolutionManager<dim>::vtk_visualizer(
  const generic_model<dim, CellType> &model, unsigned const &time_level)
{
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(model.DoF_H_System);

  std::vector<std::string> solution_names(dim + 1);
  solution_names[0] = "head";
  for (unsigned i1 = 0; i1 < dim; ++i1)
    solution_names[i1 + 1] = "flow";
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      1, dealii::DataComponentInterpretation::component_is_scalar);
  for (unsigned i1 = 0; i1 < dim; ++i1)
    data_component_interpretation.push_back(
      dealii::DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector(visual_solu,
                           solution_names,
                           dealii::DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  dealii::Vector<float> subdomain(the_grid.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = comm_rank;
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename =
    ("solution-" + dealii::Utilities::int_to_string(refn_cycle, 2) + "-" +
     dealii::Utilities::int_to_string(comm_rank, 4) + "-" +
     dealii::Utilities::int_to_string(time_level, 4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (comm_rank == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < comm_size; ++i)
      filenames.push_back(
        "solution-" + dealii::Utilities::int_to_string(refn_cycle, 2) + "-" +
        dealii::Utilities::int_to_string(i, 4) + "-" +
        dealii::Utilities::int_to_string(time_level, 4) + ".vtu");
    std::ofstream master_output((filename + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}

/*!
 * \param[in]  logger is the ostream which should print the output.
 * \param[in]  log is the meassage to be shown.
 * \param[in]  insert_eol is a key to determine that inserting an end of line
 * is
 * required or not.
 * \return Nothing.
 */
template <int dim>
void SolutionManager<dim>::out_logger(std::ostream &logger,
                                      const std::string &log,
                                      bool insert_eol)
{
  if (comm_rank == 0)
  {
    if (insert_eol)
      logger << log << std::endl;
    else
      logger << log;
  }
}

template <int dim>
int SolutionManager<dim>::cell_id_to_num_finder(
  const dealiiCell &dealii_cell_, std::map<std::string, int> &ID_to_num_map)
{
  std::stringstream cell_id;
  cell_id << dealii_cell_->id();
  std::string cell_str_id = cell_id.str();
  if (ID_to_num_map.find(cell_str_id) != ID_to_num_map.end())
    return ID_to_num_map[cell_str_id];
  else
    return -1;
}

template <int dim>
void SolutionManager<dim>::write_grid()
{
  dealii::GridOut Grid1_Out;
  dealii::GridOutFlags::Svg svg_flags(
    1,                                       // line_thickness = 2,
    2,                                       // boundary_line_thickness = 4,
    false,                                   // margin = true,
    dealii::GridOutFlags::Svg::transparent,  // background = white,
    0,                                       // azimuth_angle = 0,
    0,                                       // polar_angle = 0,
    dealii::GridOutFlags::Svg::subdomain_id, // coloring = level_number,
    false, // convert_level_number_to_height = false,
    false, // label_level_number = true,
    true,  // label_cell_index = true,
    false, // label_material_id = false,
    false, // label_subdomain_id = false,
    true,  // draw_colorbar = true,
    true); // draw_legend = true
  Grid1_Out.set_flags(svg_flags);
  if (dim == 2)
  {
    std::ofstream Grid1_OutFile(
      "/org/groups/chg/_Ali_/My_Stuff/Shared/All_Codes/deal_II/nargil/build/" +
      std::to_string(refn_cycle) + std::to_string(comm_rank) + ".svg");
    Grid1_Out.write_svg(the_grid, Grid1_OutFile);
  }
  else
  {
    std::ofstream Grid1_OutFile("Grid1" + std::to_string(refn_cycle) +
                                std::to_string(comm_rank) + ".msh");
    Grid1_Out.write_msh(the_grid, Grid1_OutFile);
  }
}

template <int dim>
void SolutionManager<dim>::free_containers()
{
  wreck_it_Ralph(cell_ID_to_num);
}
