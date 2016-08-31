#include "generic_model.hpp"

#ifndef HDG_MODEL_WITH_RK_HPP
#define HDG_MODEL_WITH_RK_HPP

/*!
 * \brief A class containing a collection of \c std::unique_ptr 's to
 * GenericCell 's.
 *
 * Currently, a hdg_model does not have a mesh of its
 * own. Mesh is assumed to be a geometric objct which is stored in
 * SolutionManager::the_grid. Because, I did not need such an
 * implementation. But, in a more general setup, you might be interested
 * to include mesh as part of the hdg_model as well. By general setup,
 * I mean multiphysics problems, where you can have different
 * interacting physical phenomena. In those cases, the equation
 * numbering will also change, and the GenericCell::dofs_ID_in_all_ranks
 * can start from a number other than zero.
 * In any case, each hdg_model can only have one type of physical phenomenon
 * (hence one type of element).
 */
template <int dim, template <int> class CellType>
struct hdg_model_with_explicit_rk : public generic_model<dim, CellType>
{
  friend struct GenericCell<dim>;
  typedef CellType<dim> model_type;
  typedef typename GenericCell<dim>::dealiiCell dealiiCell;
  typedef std::unique_ptr<dealii::FEValues<dim> > FE_val_ptr;
  typedef std::unique_ptr<dealii::FEFaceValues<dim> > FEFace_val_ptr;

  hdg_model_with_explicit_rk() = delete;
  hdg_model_with_explicit_rk(SolutionManager<dim> *const sol_,
                             explicit_RKn<4, original_RK> *time_integrator_);

  // SolutionManager<dim> *const manager;
  unsigned poly_order;
  unsigned n_faces_per_cell;

  dealii::FE_DGQ<dim> DG_Elem;
  dealii::FESystem<dim> DG_System;

  /**
   * \brief Contains element data of all of the cells which
   * belong to the current processor.
   */
  std::vector<std::unique_ptr<GenericCell<dim> > > all_owned_cells;

  void init_mesh_containers();
  void free_containers();
  void set_boundary_indicator();
  void count_globals();
  void assign_initial_data(const explicit_RKn<4, original_RK> &);

  void assemble_globals(const solver_update_keys &keys_);
  bool check_for_next_iter(double *const local_uhat);
  bool compute_internal_dofs(double *const local_uhat);
  void push_cell_unknown_to_local_vector(const double &val,
                                         const unsigned &idx);
  void init_solver();
  void reinit_solver(const solver_update_keys &update_keys_);

  template <template <int> class srcCellType,
            template <int, template <int> class> class srcModelType>
  void
  get_results_from_another_model(srcModelType<dim, srcCellType> &src_model);

  explicit_RKn<4, original_RK> *time_integrator;
  std::unique_ptr<generic_solver<dim, CellType> > solver;
};

#include "hdg_model_with_explicit_rk.cpp"

#endif
