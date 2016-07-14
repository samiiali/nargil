#include "generic_model.hpp"

template <int dim, template <int> class CellType>
generic_model<dim, CellType>::generic_model(
  SolutionManager<dim> *const manager_)
  : manager(manager_),
    comm_rank(manager->comm_rank),
    DoF_H_Refine(this->manager->the_grid),
    DoF_H_System(this->manager->the_grid)
{
}

template <int dim, template <int> class CellType>
generic_model<dim, CellType>::~generic_model()
{
}

template <int dim, template <int> class CellType>
unsigned generic_model<dim, CellType>::get_global_mat_block_size()
{
  unsigned poly_order = manager->poly_order;
  unsigned n_face_basis = pow(poly_order, dim - 1);
  return n_face_basis * CellType<dim>::get_num_dofs_per_node();
}
