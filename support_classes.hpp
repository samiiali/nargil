#include "poly_bases/poly_basis.hpp"
#include <Eigen/Dense>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <set>
#include <type_traits>

#ifndef SUPPORT_CLASSES
#define SUPPORT_CLASSES

template <int dim, int spacedim = dim>
struct GenericCell;

template <int dim>
struct local_nodal_sol
{
  local_nodal_sol(const dealii::DoFHandler<dim> *const dof_handler_,
                  const MPI_Comm *const comm_)
    : dof_handler(dof_handler_),
      idx_set(dof_handler->locally_owned_dofs()),
      comm(comm_),
      local_nodal_vec(idx_set, *comm)
  {
    idx_set.fill_index_vector(idx_vec);
  }
  ~local_nodal_sol() {}

  void assemble(const unsigned &idx, const double &val)
  {
    local_nodal_vec[idx_vec[idx]] = val;
  }

  void reinit_global_vec(LA::MPI::Vector &global_nodal_vec)
  {
    dealii::IndexSet active_idx_set;
    dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler,
                                                    active_idx_set);
    global_nodal_vec.reinit(idx_set, active_idx_set, *comm);
  }

  void copy_to_global_vec(LA::MPI::Vector &global_nodal_vec,
                          const bool &reinit_global_vec_ = true)
  {
    if (reinit_global_vec_)
      reinit_global_vec(global_nodal_vec);
    local_nodal_vec.compress(dealii::VectorOperation::insert);
    global_nodal_vec = local_nodal_vec;
  }

  const dealii::DoFHandler<dim> *const dof_handler;
  dealii::IndexSet idx_set;
  std::vector<unsigned> idx_vec;
  const MPI_Comm *const comm;
  LA::MPI::Vector local_nodal_vec;
};

/*!
 * \defgroup input_data_group Input Data
 * \brief
 * All of the user defined data, such as ICs, BCs, model parameters, ... .
 * \details
 * The input data that the user should provide to solver is contained in
 * this group. These predfined functions by
 * the user are used as IC, BC, material property, model parameter,
 * etc.. For each numerical example a set of input data has to be provided.
 * We are solving different examples with this code. Check \ref
 * GN_0_0_stage2_page
 * "numerical examples" to see which data correspond to which example.
 */

/*!
 * \details This is the generic abstract base struct for all other functions
 * which are varying through time and space.
 * \tparam in_point_dim Is the dimension of the point that you give to the
 * funciton to calculate the value of the function at that point.
 */
template <int in_point_dim, typename output_type>
struct TimeFunction
{
  TimeFunction() {}
  virtual ~TimeFunction() {}
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &n,
                            const double &t = 0.0) const = 0;
};

/*!
 * \details This is the generic abstract base struct for all other functions.
 * \tparam in_point_dim Is the dimension of the point that you give to the
 * funciton to calculate the value of the function at that point.
 */
template <int in_point_dim, typename output_type>
struct Function
{
  Function() {}
  virtual ~Function() {}
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &n) const = 0;
};

/*!
 * Here, we free the memory of the argument. This function template
 * assumes that the wreckee has the method swap. Which is true for
 * stl containers.
 */
template <typename T>
void wreck_it_Ralph(T &wreckee);

const std::string currentDateTime();

void Tokenize(const std::string &str_in,
              std::vector<std::string> &tokens,
              const std::string &delimiters);

/*!
 * A generic dof class which is just used to store the data for locally
 * owned faces on this rank. This stored data includes:
 *   - Number of global dofs on this face. \a Global \a DOFs means that
 *     if the ghost cell connected to this face contains some other
 *     types of DOFs (like multiphysics problems), this face also counts
 *     those unknowns as well.
 *   - Number of locally owned faces connected to this face
 *   - Number of nonlocally owned faces connected to this face
 *   - Number of locally owned DOFs connected to this face
 *   - Number of nonlocally owned DOFs connected to this face
 */
template <int dim, int spacedim = dim>
struct GenericDOF
{
  GenericDOF();
  unsigned global_dof_id;
  unsigned n_local_connected_DOFs;
  unsigned n_nonlocal_connected_DOFs;
  int owner_rank_id;
  std::vector<typename GenericCell<dim>::vec_iter_ptr_type> parent_cells;
  std::vector<unsigned> connected_face_of_parent_cell;
  std::vector<typename GenericCell<dim>::vec_iter_ptr_type> parent_ghosts;
  std::vector<unsigned> connected_face_of_parent_ghost;
};

#include "elements/cell_class.hpp"
#include "support_classes.cpp"

#endif // SUPPORT_CLASSES
