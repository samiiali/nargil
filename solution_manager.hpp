#include "lib_headers.hpp"
// I put this comment line here, because order of headers matters! //
#include "poly_bases/poly_basis.hpp"
#include "solvers.hpp"
#include "time_integrators.hpp"

#ifndef SOLUTION_MANAGER_HPP
#define SOLUTION_MANAGER_HPP

#include "models/generic_model.hpp"

/*!
 * \brief This class contains all of the main functions and objects which are
 * used to solve the problem.
 */
template <int dim>
struct SolutionManager
{
  /*!
   * hdg_model is a friend of SolutionManager, no matter which template
   * parameter we set for hdg_model. If the syntax looks weird to you,
   * maybe it is good to google "template template syntax".
   */
  template <int model_dim, template <int> class CellType>
  friend struct hdg_model;
  typedef typename GenericCell<dim>::dealiiCell dealiiCell;
  typedef typename GenericCell<dim>::elem_basis_type elem_basis_type;
  typedef typename GenericCell<dim>::face_basis_type face_basis_type;
  /* Change to these to get regular Lagrange's polynomials
  typedef Lagrange_Polys<dim> elem_basis_type;
  typedef Lagrange_Polys<dim - 1> face_basis_type;
  */

  /*!
   * @brief The constructor of the main class of the program. This constructor
   * takes 6 arguments.
   * @param order The order of the elements.
   * @param comm_ The MPI communicator.
   * @param comm_size_ Number of MPI procs.
   * @param comm_rank_ ID_Num of the current proc.
   * @param n_threads Number of OpenMP threads.
   * @param adaptive_on_ A flag which tell to turn on the AMR.
   */
  SolutionManager(const unsigned &order,
                  const MPI_Comm &comm_,
                  const unsigned &comm_size_,
                  const unsigned &comm_rank_,
                  const unsigned &n_threads,
                  const bool &adaptive_on_);
  ~SolutionManager();

  void out_logger(std::ostream &logger,
                  const std::string &log,
                  bool insert_eol = true);
  void write_grid();
  void solve(const unsigned &h_1, const unsigned &h_2);
  template <template <int> class CellType>
  void refine_grid(int, hdg_model<dim, CellType> &);
  template <template <int> class CellType>
  void refine_grid(int, explicit_hdg_model<dim, CellType> &);
  int cell_id_to_num_finder(const dealiiCell &dealii_cell_,
                            std::map<std::string, int> &ID_to_num_map);
  void free_containers();
  template <template <int> class CellType>
  void vtk_visualizer(const generic_model<dim, CellType> &model,
                      const unsigned &time);

  /*! The common grid object between different hdg_model 's. In the current
   * version, all of the hdg_model 's share the same grid. However, if one
   * is interested more general setups, it is possible to move grid to
   * the hdg_model opjects.
   */
  MPI_Comm comm;
  unsigned comm_size, comm_rank;
  const unsigned poly_order;
  const unsigned quad_order;
  const unsigned n_faces_per_cell;

  /*!
   * In current
   */
  dealii::parallel::distributed::Triangulation<dim> the_grid;
  dealii::MappingQ1<dim> elem_mapping;
  const dealii::QGauss<dim> elem_quad_bundle;
  const dealii::QGauss<dim - 1> face_quad_bundle;
  const dealii::QGaussLobatto<1> LGL_quad_1D;

  poly_space_basis<elem_basis_type, dim> the_elem_basis;
  poly_space_basis<elem_basis_type, dim> postprocess_cell_basis;
  poly_space_basis<face_basis_type, dim - 1> the_face_basis;
  /*!
   * \brief Contains the number of performed refinement
   * cycles from the original state of the mesh.
   */
  unsigned refn_cycle;

  // private:
  /*!
   * \brief When set to true, turns on the adaptive
   * refinement.
   */
  const bool adaptive_on = true;

  unsigned n_ghost_cell;
  unsigned n_owned_cell;
  unsigned n_threads;
  unsigned time_integration_order;
  double time_step_size;

  /*! \brief An \c std::map which maps the dealii cell ID of each
   * GenericCell (which is actually the GenericCell::cell_id) to the
   * innerCPU number for that cell. */
  std::map<std::string, int> cell_ID_to_num;

  LA::MPI::Vector refine_solu, visual_solu;

  std::ofstream convergence_result;
  std::ofstream execution_time;
};

void Tokenize(const std::string &str_in,
              std::vector<std::string> &tokens,
              const std::string &delimiters);

#include "solution_manager.cpp"

#endif // SOLUTION_MANAGER_HPP
