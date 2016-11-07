#include "solvers.hpp"

template <int dim>
std::unique_ptr<GN_dispersive_flux_generator<dim> >
GN_dispersive_flux_generator<dim>::make_flux_generator(
  const MPI_Comm *const mpi_comm_,
  const explicit_hdg_model<dim, explicit_nswe> *const model_)
{
  std::unique_ptr<GN_dispersive_flux_generator<dim> > output;
  output.reset(new GN_dispersive_flux_generator<dim>(mpi_comm_, model_));
  return std::move(output);
}

template <int dim>
GN_dispersive_flux_generator<dim>::GN_dispersive_flux_generator(
  const MPI_Comm *const mpi_comm_,
  const explicit_hdg_model<dim, explicit_nswe> *const model_)
  : comm(mpi_comm_), model(model_)
{
  init_components();
}

template <int dim>
GN_dispersive_flux_generator<dim>::~GN_dispersive_flux_generator()
{
  free_components();
}

template <int dim>
void GN_dispersive_flux_generator<dim>::init_components()
{
  VecCreateMPI(*(this->comm),
               this->model->n_global_DOFs_rank_owns,
               this->model->n_global_DOFs_on_all_ranks,
               &face_count);
  VecSetOption(face_count, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  VecSet(face_count, 0);

  VecCreateMPI(*(this->comm),
               this->model->n_global_DOFs_rank_owns,
               this->model->n_global_DOFs_on_all_ranks,
               &prim_vars_flux);
  VecSetOption(prim_vars_flux, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);

  VecCreateMPI(*(this->comm),
               this->model->n_global_DOFs_rank_owns,
               this->model->n_global_DOFs_on_all_ranks,
               &V_x_sum);
  VecSetOption(V_x_sum, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);

  VecCreateMPI(*(this->comm),
               this->model->n_global_DOFs_rank_owns,
               this->model->n_global_DOFs_on_all_ranks,
               &V_y_sum);
  VecSetOption(V_y_sum, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
}

template <int dim>
void GN_dispersive_flux_generator<dim>::free_components()
{
  VecDestroy(&face_count);
  VecDestroy(&prim_vars_flux);
  VecDestroy(&V_x_sum);
  VecDestroy(&V_y_sum);
}

template <int dim>
void GN_dispersive_flux_generator<dim>::push_to_global_vec(
  Vec &the_vec,
  const std::vector<int> &rows,
  const std::vector<double> &vals,
  const InsertMode &mode)
{
#ifdef _OPENMP
#pragma omp critical
#endif
  VecSetValues(the_vec, rows.size(), rows.data(), vals.data(), mode);
}

template <int dim>
void GN_dispersive_flux_generator<dim>::finish_assembly(Vec &the_vec)
{
  VecAssemblyBegin(the_vec);
  VecAssemblyEnd(the_vec);
}

template <int dim>
std::vector<double>
GN_dispersive_flux_generator<dim>::get_local_part_of_global_vec(
  Vec &petsc_vec, const bool &destroy_petsc_vec)
{
  IS from, to;
  Vec local_petsc_vec;
  VecScatter scatter;
  VecCreateSeq(
    PETSC_COMM_SELF, this->model->n_local_DOFs_on_this_rank, &local_petsc_vec);
  ISCreateGeneral(PETSC_COMM_SELF,
                  this->model->n_local_DOFs_on_this_rank,
                  this->model->scatter_from.data(),
                  PETSC_COPY_VALUES,
                  &from);
  ISCreateGeneral(PETSC_COMM_SELF,
                  this->model->n_local_DOFs_on_this_rank,
                  this->model->scatter_to.data(),
                  PETSC_COPY_VALUES,
                  &to);
  VecScatterCreate(petsc_vec, from, local_petsc_vec, to, &scatter);
  VecScatterBegin(
    scatter, petsc_vec, local_petsc_vec, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(
    scatter, petsc_vec, local_petsc_vec, INSERT_VALUES, SCATTER_FORWARD);
  double *local_exact_pointer;
  VecGetArray(local_petsc_vec, &local_exact_pointer);
  std::vector<double> local_vec(local_exact_pointer,
                                local_exact_pointer +
                                  this->model->n_local_DOFs_on_this_rank);
  {
    VecRestoreArray(local_petsc_vec, &local_exact_pointer);
    VecDestroy(&local_petsc_vec);
    ISDestroy(&from);
    ISDestroy(&to);
    VecScatterDestroy(&scatter);
    if (destroy_petsc_vec)
      VecDestroy(&petsc_vec);
  }
  return local_vec;
}

//
//
//
//
//

template <int dim, template <int> class ModelType>
generic_solver<dim, ModelType>::generic_solver(
  const MPI_Comm *const mpi_comm_,
  generic_model<dim, ModelType> *model_,
  const solver_options &)
  : comm(mpi_comm_), model(model_)
{
}

template <int dim, template <int> class ModelType>
generic_solver<dim, ModelType>::~generic_solver()
{
}

template <int dim, template <int> class ModelType>
std::unique_ptr<generic_solver<dim, ModelType> >
generic_solver<dim, ModelType>::make_solver(
  solver_type solver_type_,
  const MPI_Comm *const mpi_comm_,
  generic_model<dim, ModelType> *model_,
  const solver_options &options_)
{
  std::unique_ptr<generic_solver<dim, ModelType> > output;
  if (solver_type_ == solver_type::implicit_petsc_aij)
    output.reset(
      new petsc_implicit_aij<dim, ModelType>(mpi_comm_, model_, options_));
  if (solver_type_ == solver_type::implicit_petsc_bij)
    output.reset(
      new petsc_implicit_bij<dim, ModelType>(mpi_comm_, model_, options_));
  return std::move(output);
}

/*
 *
 */

template <int dim, template <int> class ModelType>
implicit_solver<dim, ModelType>::implicit_solver(
  const MPI_Comm *const mpi_comm_,
  generic_model<dim, ModelType> *model_,
  const solver_options &options_)
  : generic_solver<dim, ModelType>(mpi_comm_, model_, options_)
{
}

template <int dim, template <int> class ModelType>
implicit_solver<dim, ModelType>::~implicit_solver()
{
}

/*
 *
 */

template <int dim, template <int> class ModelType>
petsc_implicit_aij<dim, ModelType>::petsc_implicit_aij(
  const MPI_Comm *const mpi_comm_,
  generic_model<dim, ModelType> *model_,
  const solver_options options_)
  : implicit_solver<dim, ModelType>(mpi_comm_, model_, options_)
{
  solver_update_keys keys_ =
    static_cast<solver_update_keys>(update_mat | update_rhs | update_sol);
  init_components(options_, keys_);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_aij<dim, ModelType>::reinit_components(
  generic_model<dim, ModelType> *model_,
  const solver_options options_,
  const solver_update_keys update_keys_)
{
  this->model = model_;
  free_components(update_keys_);
  init_components(options_, update_keys_);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_aij<dim, ModelType>::init_components(
  const solver_options &options_, const solver_update_keys &update_keys_)
{
  if (update_keys_ & update_mat)
  {
    MatCreate(*(this->comm), &global_mat);
    MatSetType(global_mat, MATMPIAIJ);
    MatSetSizes(global_mat,
                this->model->n_global_DOFs_rank_owns,
                this->model->n_global_DOFs_rank_owns,
                this->model->n_global_DOFs_on_all_ranks,
                this->model->n_global_DOFs_on_all_ranks);
    MatMPIAIJSetPreallocation(
      global_mat,
      0,
      this->model->n_local_DOFs_connected_to_DOF.data(),
      0,
      this->model->n_nonlocal_DOFs_connected_to_DOF.data());
    MatSetOption(global_mat, MAT_ROW_ORIENTED, PETSC_FALSE);
    if (options_ & solver_options::spd_matrix)
      MatSetOption(global_mat, MAT_SPD, PETSC_TRUE);
    if (options_ & solver_options::symmetric_matrix)
      MatSetOption(global_mat, MAT_SYMMETRIC, PETSC_TRUE);
    if (options_ & solver_options::ignore_mat_zero_entries)
      MatSetOption(global_mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
  }
  if (update_keys_ & update_rhs)
  {
    VecCreateMPI(*(this->comm),
                 this->model->n_global_DOFs_rank_owns,
                 this->model->n_global_DOFs_on_all_ranks,
                 &rhs_vec);
    VecSetOption(rhs_vec, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  }
  if (update_keys_ & update_sol)
  {
    VecCreateMPI(*(this->comm),
                 this->model->n_global_DOFs_rank_owns,
                 this->model->n_global_DOFs_on_all_ranks,
                 &exact_sol);
    VecSetOption(exact_sol, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  }
}

template <int dim, template <int> class ModelType>
petsc_implicit_aij<dim, ModelType>::~petsc_implicit_aij()
{
  solver_update_keys keys_ =
    static_cast<solver_update_keys>(update_mat | update_rhs | update_sol);
  free_components(keys_);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_aij<dim, ModelType>::free_components(
  const solver_update_keys &update_keys_)
{
  if (update_keys_ & solver_update_keys::update_mat)
  {
    MatDestroy(&global_mat);
    KSPDestroy(&ksp);
  }
  if (update_keys_ & solver_update_keys::update_rhs)
    VecDestroy(&rhs_vec);
  if (update_keys_ & solver_update_keys::update_sol)
    VecDestroy(&exact_sol);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_aij<dim, ModelType>::push_to_global_mat(
  const std::vector<int> &rows,
  const std::vector<int> &cols,
  const std::vector<double> &vals,
  const InsertMode &mode)
{
#ifdef _OPENMP
#pragma omp critical
#endif
  MatSetValues(global_mat,
               rows.size(),
               rows.data(),
               cols.size(),
               cols.data(),
               vals.data(),
               mode);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_aij<dim, ModelType>::push_to_global_mat(
  const int &row, const int &col, const double &val, const InsertMode &mode)
{
#ifdef _OPENMP
#pragma omp critical
#endif
  MatSetValue(global_mat, row, col, val, mode);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_aij<dim, ModelType>::push_to_rhs_vec(
  const std::vector<int> &rows,
  const std::vector<double> &vals,
  const InsertMode &mode)
{
#ifdef _OPENMP
#pragma omp critical
#endif
  VecSetValues(rhs_vec, rows.size(), rows.data(), vals.data(), mode);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_aij<dim, ModelType>::push_to_exact_sol(
  const std::vector<int> &rows,
  const std::vector<double> &vals,
  const InsertMode &mode)
{
#ifdef _OPENMP
#pragma omp critical
#endif
  VecSetValues(exact_sol, rows.size(), rows.data(), vals.data(), mode);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_aij<dim, ModelType>::finish_assembly(
  const solver_update_keys &keys_)
{
  if (keys_ & update_mat)
  {
    MatAssemblyBegin(global_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(global_mat, MAT_FINAL_ASSEMBLY);
  }
  if (keys_ & update_rhs)
  {
    VecAssemblyBegin(rhs_vec);
    VecAssemblyEnd(rhs_vec);
  }
  if (keys_ & update_sol)
  {
    VecAssemblyBegin(exact_sol);
    VecAssemblyEnd(exact_sol);
  }

  /*
   * I am assembling a bunch of symmetric matrices and I should get a symmetric
   * matrix. Here, we want to check if the global matrix is symmetric.
   */
  /*
  int nconv;
  EPS eps1;
  Vec xr, xi;
  double kr, ki;
  EPSCreate(PETSC_COMM_WORLD, &eps1);
  EPSSetOperators(eps1, global_mat, NULL);
  EPSSetProblemType(eps1, EPS_NHEP);
  EPSSetFromOptions(eps1);
  EPSSolve(eps1);
  EPSGetConverged(eps1, &nconv);
  for (unsigned j_conv = 0; j_conv < nconv; ++j_conv)
  {
    EPSGetEigenpair(eps1, j_conv, &kr, &ki, xr, xi);
    std::cout << kr << "  " << ki << std::endl;
  }
  */
}

template <int dim, template <int> class ModelType>
std::vector<double> petsc_implicit_aij<dim, ModelType>::get_local_exact_sol()
{
  return get_local_part_of_global_vec(exact_sol);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_aij<dim, ModelType>::form_factors(
  const implicit_petsc_factor_type &factor_type)
{
  KSPCreate(*(this->comm), &ksp);
  KSPSetOperators(ksp, global_mat, global_mat);
  if (factor_type == implicit_petsc_factor_type::cg_gamg)
  {
    KSPSetType(ksp, KSPCG);
    KSPSetFromOptions(ksp);
    KSPSetTolerances(ksp, 5E-13, PETSC_DEFAULT, PETSC_DEFAULT, 40000);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCGAMG);
    //    PCSetType(pc, PCHYPRE);
    PCSetFromOptions(pc);
    PCGAMGSetNSmooths(pc, 1);
  }
  if (factor_type == implicit_petsc_factor_type::mumps)
  {
    Mat factor_mat;
    KSPSetType(ksp, KSPPREONLY);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);
    PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);
    PCFactorSetUpMatSolverPackage(pc);
    PCFactorGetMatrix(pc, &factor_mat);
    /* choosing the parallel computing icntl(28) = 2 */
    //    MatMumpsSetIcntl(factor_mat, 28, 2);
    /* sequential ordering icntl(7) = 2 */
    MatMumpsSetIcntl(factor_mat, 7, 3);
    //    MatMumpsSetIcntl(factor_mat, 29, 2);
    /* parallel ordering icntl(29) = 2 */
    //  MatMumpsSetIcntl(factor_mat, 29, 2);
    /* threshhold for row pivot detection */

    // Iterative refinement //
    MatMumpsSetIcntl(factor_mat, 10, -2);
    //    MatMumpsSetIcntl(factor_mat, 11, 1);
    //    MatMumpsSetIcntl(factor_mat, 12, 1);

    // Null pivot rows detection //
    MatMumpsSetIcntl(factor_mat, 24, 1);

    // Increase in the estimated memory
    MatMumpsSetIcntl(factor_mat, 14, 500);

    // Numerical pivoting
    MatMumpsSetCntl(factor_mat, 1, 0.1);
    //    MatMumpsSetCntl(factor_mat, 2, 1.E-14);

    // Null pivot row detection
    MatMumpsSetCntl(factor_mat, 3, -1.E-14);

    // Static pivoting
    MatMumpsSetCntl(factor_mat, 4, 1.E-6);

    //    MatMumpsSetCntl(factor_mat, 5, 1.E20);
  }
  if (factor_type == implicit_petsc_factor_type::superlu_dist)
  {
    Mat factor_mat;
    KSPSetType(ksp, KSPPREONLY);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);
    PCFactorSetMatSolverPackage(pc, MATSOLVERSUPERLU_DIST);
    PCFactorSetUpMatSolverPackage(pc);
    PCFactorGetMatrix(pc, &factor_mat);
  }
  if (factor_type == implicit_petsc_factor_type::pastix)
  {
    Mat factor_mat;
    KSPSetType(ksp, KSPPREONLY);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);
    PCFactorSetMatSolverPackage(pc, MATSOLVERPASTIX);
    PCFactorSetUpMatSolverPackage(pc);
    PCFactorGetMatrix(pc, &factor_mat);
  }
  if (factor_type == implicit_petsc_factor_type::pardiso)
  {
    Mat factor_mat;
    KSPSetType(ksp, KSPPREONLY);
    KSPGetPC(ksp, &pc);
    //    PCSetType(pc, PCLU);
    //    PCFactorSetMatSolverPackage(pc, MATSOLVERMKL_PARDISO);
    PCFactorSetUpMatSolverPackage(pc);
    PCFactorGetMatrix(pc, &factor_mat);
  }
}

template <int dim, template <int> class ModelType>
void petsc_implicit_aij<dim, ModelType>::solve_system(Vec &sol_vec)
{
  KSPConvergedReason how_ksp_stopped;
  PetscInt num_iter;
  VecDuplicate(rhs_vec, &sol_vec);
  KSPSolve(ksp, rhs_vec, sol_vec);
  KSPGetIterationNumber(ksp, &num_iter);
  KSPGetConvergedReason(ksp, &how_ksp_stopped);
  if (this->model->comm_rank == 0)
    std::cout << num_iter << "  " << how_ksp_stopped << std::endl;
}

template <int dim, template <int> class ModelType>
std::vector<double>
petsc_implicit_aij<dim, ModelType>::get_local_part_of_global_vec(
  Vec &petsc_vec, const bool &destroy_petsc_vec)
{
  IS from, to;
  Vec local_petsc_vec;
  VecScatter scatter;
  VecCreateSeq(
    PETSC_COMM_SELF, this->model->n_local_DOFs_on_this_rank, &local_petsc_vec);
  ISCreateGeneral(PETSC_COMM_SELF,
                  this->model->n_local_DOFs_on_this_rank,
                  this->model->scatter_from.data(),
                  PETSC_COPY_VALUES,
                  &from);
  ISCreateGeneral(PETSC_COMM_SELF,
                  this->model->n_local_DOFs_on_this_rank,
                  this->model->scatter_to.data(),
                  PETSC_COPY_VALUES,
                  &to);
  VecScatterCreate(petsc_vec, from, local_petsc_vec, to, &scatter);
  VecScatterBegin(
    scatter, petsc_vec, local_petsc_vec, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(
    scatter, petsc_vec, local_petsc_vec, INSERT_VALUES, SCATTER_FORWARD);
  double *local_exact_pointer;
  VecGetArray(local_petsc_vec, &local_exact_pointer);
  std::vector<double> local_vec(local_exact_pointer,
                                local_exact_pointer +
                                  this->model->n_local_DOFs_on_this_rank);
  {
    VecRestoreArray(local_petsc_vec, &local_exact_pointer);
    VecDestroy(&local_petsc_vec);
    ISDestroy(&from);
    ISDestroy(&to);
    VecScatterDestroy(&scatter);
    if (destroy_petsc_vec)
      VecDestroy(&petsc_vec);
  }
  return local_vec;
}

template <int dim, template <int> class ModelType>
Vec *petsc_implicit_aij<dim, ModelType>::get_petsc_rhs()
{
  return &rhs_vec;
}

template <int dim, template <int> class ModelType>
Mat *petsc_implicit_aij<dim, ModelType>::get_petsc_mat()
{
  return &global_mat;
}

/*
 *
 */

template <int dim, template <int> class ModelType>
petsc_implicit_bij<dim, ModelType>::petsc_implicit_bij(
  const MPI_Comm *const mpi_comm_,
  generic_model<dim, ModelType> *model_,
  const solver_options options_)
  : implicit_solver<dim, ModelType>(mpi_comm_, model_, options_)
{
  solver_update_keys keys_ =
    static_cast<solver_update_keys>(update_mat | update_rhs | update_sol);
  init_components(options_, keys_);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_bij<dim, ModelType>::reinit_components(
  generic_model<dim, ModelType> *model_,
  const solver_options options_,
  const solver_update_keys update_keys_)
{
  this->model = model_;
  free_components(update_keys_);
  init_components(options_, update_keys_);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_bij<dim, ModelType>::init_components(
  const solver_options &options_, const solver_update_keys &update_keys_)
{
  if (update_keys_ & update_mat)
  {
    unsigned block_size = this->model->get_global_mat_block_size();
    MatCreateBAIJ(*(this->comm),
                  block_size,
                  this->model->n_global_DOFs_rank_owns,
                  this->model->n_global_DOFs_rank_owns,
                  this->model->n_global_DOFs_on_all_ranks,
                  this->model->n_global_DOFs_on_all_ranks,
                  0,
                  this->model->n_local_DOFs_connected_to_DOF.data(),
                  0,
                  this->model->n_nonlocal_DOFs_connected_to_DOF.data(),
                  &global_mat);
    MatSetOption(global_mat, MAT_ROW_ORIENTED, PETSC_FALSE);
    if (options_ & solver_options::spd_matrix)
      MatSetOption(global_mat, MAT_SPD, PETSC_TRUE);
    if (options_ & solver_options::symmetric_matrix)
      MatSetOption(global_mat, MAT_SYMMETRIC, PETSC_TRUE);
    if (options_ & solver_options::ignore_mat_zero_entries)
      MatSetOption(global_mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
  }
  if (update_keys_ & update_rhs)
  {
    VecCreateMPI(*(this->comm),
                 this->model->n_global_DOFs_rank_owns,
                 this->model->n_global_DOFs_on_all_ranks,
                 &rhs_vec);
    VecSetOption(rhs_vec, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  }
  if (update_keys_ & update_sol)
  {
    VecCreateMPI(*(this->comm),
                 this->model->n_global_DOFs_rank_owns,
                 this->model->n_global_DOFs_on_all_ranks,
                 &exact_sol);
    VecSetOption(exact_sol, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  }
}

template <int dim, template <int> class ModelType>
petsc_implicit_bij<dim, ModelType>::~petsc_implicit_bij()
{
  solver_update_keys keys_ =
    static_cast<solver_update_keys>(update_mat | update_rhs | update_sol);
  free_components(keys_);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_bij<dim, ModelType>::free_components(
  const solver_update_keys &update_keys_)
{
  if (update_keys_ & solver_update_keys::update_mat)
  {
    MatDestroy(&global_mat);
    KSPDestroy(&ksp);
  }
  if (update_keys_ & solver_update_keys::update_rhs)
    VecDestroy(&rhs_vec);
  if (update_keys_ & solver_update_keys::update_sol)
    VecDestroy(&exact_sol);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_bij<dim, ModelType>::push_to_global_mat(
  const std::vector<int> &rows,
  const std::vector<int> &cols,
  const std::vector<double> &vals,
  const InsertMode &mode)
{
#ifdef _OPENMP
#pragma omp critical
#endif
  MatSetValues(global_mat,
               rows.size(),
               rows.data(),
               cols.size(),
               cols.data(),
               vals.data(),
               mode);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_bij<dim, ModelType>::push_to_global_mat(
  const int &row, const int &col, const double &val, const InsertMode &mode)
{
#ifdef _OPENMP
#pragma omp critical
#endif
  MatSetValue(global_mat, row, col, val, mode);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_bij<dim, ModelType>::push_to_rhs_vec(
  const std::vector<int> &rows,
  const std::vector<double> &vals,
  const InsertMode &mode)
{
#ifdef _OPENMP
#pragma omp critical
#endif
  VecSetValues(rhs_vec, rows.size(), rows.data(), vals.data(), mode);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_bij<dim, ModelType>::push_to_exact_sol(
  const std::vector<int> &rows,
  const std::vector<double> &vals,
  const InsertMode &mode)
{
#ifdef _OPENMP
#pragma omp critical
#endif
  VecSetValues(exact_sol, rows.size(), rows.data(), vals.data(), mode);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_bij<dim, ModelType>::finish_assembly(
  const solver_update_keys &keys_)
{
  if (keys_ & update_mat)
  {
    MatAssemblyBegin(global_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(global_mat, MAT_FINAL_ASSEMBLY);
  }
  if (keys_ & update_rhs)
  {
    VecAssemblyBegin(rhs_vec);
    VecAssemblyEnd(rhs_vec);
  }
  if (keys_ & update_sol)
  {
    VecAssemblyBegin(exact_sol);
    VecAssemblyEnd(exact_sol);
  }

  /*
   * I am assembling a bunch of symmetric matrices and I should get a symmetric
   * matrix. Here, we want to check if the global matrix is symmetric.
   */
  /*
  int nconv;
  EPS eps1;
  Vec xr, xi;
  double kr, ki;
  EPSCreate(PETSC_COMM_WORLD, &eps1);
  EPSSetOperators(eps1, global_mat, NULL);
  EPSSetProblemType(eps1, EPS_NHEP);
  EPSSetFromOptions(eps1);
  EPSSolve(eps1);
  EPSGetConverged(eps1, &nconv);
  for (unsigned j_conv = 0; j_conv < nconv; ++j_conv)
  {
    EPSGetEigenpair(eps1, j_conv, &kr, &ki, xr, xi);
    std::cout << kr << "  " << ki << std::endl;
  }
  */
}

template <int dim, template <int> class ModelType>
std::vector<double> petsc_implicit_bij<dim, ModelType>::get_local_exact_sol()
{
  return get_local_part_of_global_vec(exact_sol);
}

template <int dim, template <int> class ModelType>
void petsc_implicit_bij<dim, ModelType>::form_factors(
  const implicit_petsc_factor_type &factor_type)
{
  KSPCreate(*(this->comm), &ksp);
  KSPSetOperators(ksp, global_mat, global_mat);
  if (factor_type == implicit_petsc_factor_type::cg_gamg)
  {
    KSPSetType(ksp, KSPCG);
    KSPSetFromOptions(ksp);
    KSPSetTolerances(ksp, 1E-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCGAMG);
    PCGAMGSetNSmooths(pc, 1);
  }
  if (factor_type == implicit_petsc_factor_type::mumps)
  {
    Mat factor_mat;
    KSPSetType(ksp, KSPPREONLY);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);
    PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);
    PCFactorSetUpMatSolverPackage(pc);
    PCFactorGetMatrix(pc, &factor_mat);
    /* choosing the parallel computing icntl(28) = 2 */
    //  MatMumpsSetIcntl(factor_mat, 28, 2);
    /* sequential ordering icntl(7) = 2 */
    MatMumpsSetIcntl(factor_mat, 7, 2);
    /* parallel ordering icntl(29) = 2 */
    //  MatMumpsSetIcntl(factor_mat, 29, 2);
    /* threshhold for row pivot detection */
    MatMumpsSetIcntl(factor_mat, 24, 1);
    MatMumpsSetCntl(factor_mat, 3, 1.E-15);
    MatMumpsSetCntl(factor_mat, 4, 0.);
    //  MatMumpsSetCntl(factor_mat, 5, 1.e+20);
  }
}

template <int dim, template <int> class ModelType>
void petsc_implicit_bij<dim, ModelType>::solve_system(Vec &sol_vec)
{
  KSPConvergedReason how_ksp_stopped;
  PetscInt num_iter;
  VecDuplicate(rhs_vec, &sol_vec);
  KSPSolve(ksp, rhs_vec, sol_vec);
  KSPGetIterationNumber(ksp, &num_iter);
  KSPGetConvergedReason(ksp, &how_ksp_stopped);
  if (this->model->comm_rank == 0)
    std::cout << num_iter << "  " << how_ksp_stopped << std::endl;
}

template <int dim, template <int> class ModelType>
std::vector<double>
petsc_implicit_bij<dim, ModelType>::get_local_part_of_global_vec(
  Vec &petsc_vec, const bool &destroy_petsc_vec)
{
  IS from, to;
  Vec local_petsc_vec;
  VecScatter scatter;
  VecCreateSeq(
    PETSC_COMM_SELF, this->model->n_local_DOFs_on_this_rank, &local_petsc_vec);
  ISCreateGeneral(PETSC_COMM_SELF,
                  this->model->n_local_DOFs_on_this_rank,
                  this->model->scatter_from.data(),
                  PETSC_COPY_VALUES,
                  &from);
  ISCreateGeneral(PETSC_COMM_SELF,
                  this->model->n_local_DOFs_on_this_rank,
                  this->model->scatter_to.data(),
                  PETSC_COPY_VALUES,
                  &to);
  VecScatterCreate(petsc_vec, from, local_petsc_vec, to, &scatter);
  VecScatterBegin(
    scatter, petsc_vec, local_petsc_vec, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(
    scatter, petsc_vec, local_petsc_vec, INSERT_VALUES, SCATTER_FORWARD);
  double *local_exact_pointer;
  VecGetArray(local_petsc_vec, &local_exact_pointer);
  std::vector<double> local_vec(local_exact_pointer,
                                local_exact_pointer +
                                  this->model->n_local_DOFs_on_this_rank);
  {
    VecRestoreArray(local_petsc_vec, &local_exact_pointer);
    VecDestroy(&local_petsc_vec);
    ISDestroy(&from);
    ISDestroy(&to);
    VecScatterDestroy(&scatter);
    if (destroy_petsc_vec)
      VecDestroy(&petsc_vec);
  }
  return local_vec;
}

template <int dim, template <int> class ModelType>
Vec *petsc_implicit_bij<dim, ModelType>::get_petsc_rhs()
{
  return &rhs_vec;
}

template <int dim, template <int> class ModelType>
Mat *petsc_implicit_bij<dim, ModelType>::get_petsc_mat()
{
  return &global_mat;
}
