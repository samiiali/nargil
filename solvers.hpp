#include "lib_headers.hpp"

#ifndef SOLVERS_HPP
#define SOLVERS_HPP

template <int dim, template <int> class CellType>
struct generic_model;

enum struct solver_type
{
  implicit_petsc_aij = 1 << 0,
  implicit_petsc_bij = 1 << 1
};

enum solver_update_keys
{
  update_mat = 1 << 0,
  update_rhs = 1 << 1,
  update_sol = 1 << 2
};

enum solver_options
{
  default_option = 0,
  spd_matrix = 1 << 0,
  symmetric_matrix = 1 << 1,
  ignore_mat_zero_entries = 1 << 2
};

enum struct implicit_petsc_factor_type
{
  cg_gamg = 0,
  mumps = 1,
  superlu_dist = 2,
  pastix = 3,
  pardiso = 4
};

/**
 *
 */
template <int dim, template <int> class ModelType>
struct generic_solver
{
  generic_solver(const MPI_Comm *const mpi_comm_,
                 generic_model<dim, ModelType> *model_,
                 const solver_options &options_);
  virtual ~generic_solver();

  static std::unique_ptr<generic_solver<dim, ModelType> >
  make_solver(solver_type solver_type_,
              const MPI_Comm *const mpi_comm_,
              generic_model<dim, ModelType> *model_,
              const solver_options &options_);

  virtual void reinit_components(generic_model<dim, ModelType> *model_,
                                 const solver_options options_,
                                 const solver_update_keys update_keys_) = 0;

  virtual void init_components(const solver_options &options_,
                               const solver_update_keys &update_keys_) = 0;

  virtual void free_components(const solver_update_keys &update_keys_) = 0;

  virtual void push_to_global_mat(const std::vector<int> &rows,
                                  const std::vector<int> &cols,
                                  const std::vector<double> &vals,
                                  const InsertMode &mode) = 0;

  virtual void push_to_global_mat(const int &row,
                                  const int &col,
                                  const double &val,
                                  const InsertMode &mode) = 0;

  virtual void push_to_rhs_vec(const std::vector<int> &rows,
                               const std::vector<double> &vals,
                               const InsertMode &mode) = 0;

  virtual void push_to_exact_sol(const std::vector<int> &rows,
                                 const std::vector<double> &vals,
                                 const InsertMode &mode) = 0;

  virtual void finish_assembly(const solver_update_keys &keys_) = 0;

  virtual std::vector<double> get_local_exact_sol() = 0;

  virtual void form_factors(const implicit_petsc_factor_type &factor_type) = 0;

  virtual void solve_system(Vec &sol_vec) = 0;

  virtual Vec *get_petsc_rhs() = 0;

  virtual Mat *get_petsc_mat() = 0;

  /**
   * This fucntion returns an std::vector containing the local part of the
   * global solution of the problem. Since, in some types of solvers, we make
   * a copy of the PETSc Vec in this function (technically we copy it two
   * times), one is not supposed to call this function at every time step.
   *
   */
  virtual std::vector<double>
  get_local_part_of_global_vec(Vec &petsc_vec,
                               const bool &destroy_petsc_vec = false) = 0;

  const MPI_Comm *const comm;
  generic_model<dim, ModelType> *model;
};

/**
 *
 */
template <int dim, template <int> class ModelType>
struct implicit_solver : generic_solver<dim, ModelType>
{
  implicit_solver(const MPI_Comm *const mpi_comm_,
                  generic_model<dim, ModelType> *model_,
                  const solver_options &options_);
  ~implicit_solver();
};

/**
 * This object solves Ax = b, with A being a PETSc MPIAIJ matrix, using
 * conjugate gradient method and geo-algebraic (!) multigrid.
 */
template <int dim, template <int> class ModelType>
struct petsc_implicit_aij : implicit_solver<dim, ModelType>
{
  petsc_implicit_aij(const MPI_Comm *const mpi_comm_,
                     generic_model<dim, ModelType> *model_,
                     const solver_options options_);
  ~petsc_implicit_aij();

  virtual void reinit_components(generic_model<dim, ModelType> *model_,
                                 const solver_options options_,
                                 const solver_update_keys update_keys_) final;

  virtual void init_components(const solver_options &options_,
                               const solver_update_keys &update_keys_) final;

  virtual void free_components(const solver_update_keys &update_keys_) final;

  virtual void push_to_global_mat(const std::vector<int> &rows,
                                  const std::vector<int> &cols,
                                  const std::vector<double> &vals,
                                  const InsertMode &mode) final;

  virtual void push_to_global_mat(const int &row,
                                  const int &col,
                                  const double &val,
                                  const InsertMode &mode) final;

  virtual void push_to_rhs_vec(const std::vector<int> &rows,
                               const std::vector<double> &vals,
                               const InsertMode &mode) final;

  virtual void push_to_exact_sol(const std::vector<int> &rows,
                                 const std::vector<double> &vals,
                                 const InsertMode &mode) final;

  virtual void finish_assembly(const solver_update_keys &keys_) final;

  /**
   * This function returns an std::vector containing local values of the
   * exact_sol PETSc Vec. I know copying from a pointer to std::vector is not
   * good, but memory leak is worse!
   */
  virtual std::vector<double> get_local_exact_sol() final;

  virtual void
  form_factors(const implicit_petsc_factor_type &factor_type) final;

  virtual void solve_system(Vec &sol_vec) final;

  virtual std::vector<double>
  get_local_part_of_global_vec(Vec &petsc_vec,
                               const bool &destroy_petsc_vec = false) final;

  virtual Vec *get_petsc_rhs() final;

  virtual Mat *get_petsc_mat() final;

  Vec rhs_vec, exact_sol;
  Mat global_mat;
  KSP ksp;
  PC pc;
};

/**
 * This object solves Ax = b, with A being a PETSc MPIAIJ matrix, using
 * conjugate gradient method and geo-algebraic (!) multigrid.
 */
template <int dim, template <int> class ModelType>
struct petsc_implicit_bij : implicit_solver<dim, ModelType>
{
  petsc_implicit_bij(const MPI_Comm *const mpi_comm_,
                     generic_model<dim, ModelType> *model_,
                     const solver_options options_);
  ~petsc_implicit_bij();

  virtual void reinit_components(generic_model<dim, ModelType> *model_,
                                 const solver_options options_,
                                 const solver_update_keys update_keys_) final;

  virtual void init_components(const solver_options &options_,
                               const solver_update_keys &update_keys_) final;

  virtual void free_components(const solver_update_keys &update_keys_) final;

  virtual void push_to_global_mat(const std::vector<int> &rows,
                                  const std::vector<int> &cols,
                                  const std::vector<double> &vals,
                                  const InsertMode &mode) final;

  virtual void push_to_global_mat(const int &row,
                                  const int &col,
                                  const double &val,
                                  const InsertMode &mode) final;

  virtual void push_to_rhs_vec(const std::vector<int> &rows,
                               const std::vector<double> &vals,
                               const InsertMode &mode) final;

  virtual void push_to_exact_sol(const std::vector<int> &rows,
                                 const std::vector<double> &vals,
                                 const InsertMode &mode) final;

  virtual void finish_assembly(const solver_update_keys &keys_) final;

  /**
   * This function returns an std::vector containing local values of the
   * exact_sol PETSc Vec. I know copying from a pointer to std::vector is not
   * good, but memory leak is worse!
   */
  virtual std::vector<double> get_local_exact_sol() final;

  virtual void
  form_factors(const implicit_petsc_factor_type &factor_type) final;

  virtual void solve_system(Vec &sol_vec) final;

  virtual std::vector<double>
  get_local_part_of_global_vec(Vec &petsc_vec,
                               const bool &destroy_petsc_vec = false) final;

  virtual Vec *get_petsc_rhs() final;

  virtual Mat *get_petsc_mat() final;

  Vec rhs_vec, exact_sol;
  Mat global_mat;
  KSP ksp;
  PC pc;
};

#include "solvers.cpp"

#endif
