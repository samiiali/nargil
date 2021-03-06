#include "poly_bases/poly_basis.hpp"
//#include "solvers.hpp"
//#include "time_integrators.hpp"
#include <type_traits>

#ifndef CELL_CLASS_HPP
#define CELL_CLASS_HPP

/*!
 * \defgroup cells Cell data
 * \brief
 * This group contains the classes which encapsulate data corresponding to each
 * cell in the mesh.
 */

/*!
 * \brief The \c Cell_Class contains most of the required data about a generic
 * element in the mesh.
 *
 * \ingroup cells
 */
template <int dim, int spacedim>
// spacedim = dim is declared in support_classes.hpp
struct GenericCell
{
  typedef dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim> >
    dealiiCell;
  typedef typename std::vector<std::unique_ptr<GenericCell> >::iterator
    vec_iter_ptr_type;
  typedef JacobiPolys<dim> elem_basis_type;
  typedef JacobiPolys<dim - 1> face_basis_type;
  //  typedef LagrangePolys<dim> elem_basis_type;
  //  typedef LagrangePolys<dim - 1> face_basis_type;
  typedef std::unique_ptr<dealii::FEValues<dim> > FE_val_ptr;
  typedef std::unique_ptr<dealii::FEFaceValues<dim> > FEFace_val_ptr;
  /*!
   * \details This enum contains the boundary condition on a given face of the
   * element. When there is no BC applied, its value is zero.
   */
  enum BC
  {
    not_set = ~(1 << 0),
    essential = 1 << 0,
    flux_bc = 1 << 1,
    periodic = 1 << 2,
    in_out_BC = 1 << 3,
    inflow_BC = 1 << 4,
    outflow_BC = 1 << 5,
    solid_wall = 1 << 6,
  };
  /*!
   * \details
   * We remove the default constructor to avoid uninitialized creation of Cell
   * objects.
   */
  GenericCell() = delete;
  /*!
   * \details
   * The constructor of this class takes a deal.II cell and its deal.II ID.
   * \param inp_cell The iterator to the deal.II cell in the mesh.
   * \param id_num_  The unique ID (\c dealii_Cell::id()) of the dealii_Cell.
   * This is necessary when working on a distributed mesh.
   */
  GenericCell(dealiiCell &inp_cell,
              const unsigned &id_num_,
              const unsigned &poly_order_);
  /*!
   * \details
   * We remove the copy constructor of this class to avoid unnecessary copies
   * (specially unintentional ones). Up to October 2015, this copy constructor
   * was not useful anywhere in the code.
   */
  GenericCell(const GenericCell &inp_cell) = delete;
  /*!
   * \details
   * We need a move constructor, to be able to pass this class as function
   * arguments efficiently. Maybe, you say that this does not help efficiency
   * that much, but we are using it for semantic constraints.
   * \param inp_cell An object of the \c Cell_Class type which we steal its
   * guts.
   */
  GenericCell(GenericCell &&inp_cell) noexcept;
  /*!
   * Obviously, the destructor.
   */
  virtual ~GenericCell();
  /*!
   * Factory pattern for producing new GenericCell
   */
  template <template <int> class type_of_cell>
  static std::unique_ptr<GenericCell<dim, spacedim> >
  make_cell(dealiiCell &inp_cell,
            const unsigned &id_num_,
            const unsigned &poly_order_,
            hdg_model<dim, type_of_cell> *model_);

  /*!
   * Factory pattern for producing new GenericCell
   */
  template <template <int> class type_of_cell>
  static std::unique_ptr<GenericCell<dim, spacedim> >
  make_cell(dealiiCell &inp_cell,
            const unsigned &id_num_,
            const unsigned &poly_order_,
            explicit_hdg_model<dim, type_of_cell> *model_);

  /*!
   * Factory pattern for producing new GenericCell
   */
  template <template <int> class type_of_cell>
  static std::unique_ptr<GenericCell<dim, spacedim> >
  make_cell(dealiiCell &inp_cell,
            const unsigned &id_num_,
            const unsigned &poly_order_,
            hdg_model_with_explicit_rk<dim, type_of_cell> *model_);

  /*! \brief Moves the dealii::FEValues and dealii::FEFaceValues
   * objects between different elements and faces.
   *
   * dealii::FEValues are not copyable objects. They also
   * do not have empty constructor. BUT, they have (possibly
   * very efficient) move assignment operators. So, they should
   * be created once and updated afterwards. This avoids us from
   * using shared memory parallelism, because we want to create
   * more than one instance of this type of object and update it
   * according to the element in action. That is why, at the
   * beginning of assemble_globals and internal unknowns
   * calculation, we create std::unique_ptr to our FEValues
   * objects and use the std::move to move them along to
   * next elements.
   *
   * \details We attach a \c unique_ptr of dealii::FEValues and
   * dealii::FEFaceValues to the current object.
   * \param cell_quad_fe_vals_ The dealii::FEValues which is used for location
   * of quadrature points in cells.
   * \param face_quad_fe_vals_ The dealii::FEValues which is used for lacation
   * of support points in cells.
   * \param cell_supp_fe_vals_ The dealii::FEValues which is used for location
   * of quadrature points on faces.
   * \param face_supp_fe_vals_ The dealii::FEValues which is used for location
   * of support points on faces.
   */
  void attach_FEValues(FE_val_ptr &cell_quad_fe_vals_,
                       FEFace_val_ptr &face_quad_fe_vals_,
                       FE_val_ptr &cell_supp_fe_vals_,
                       FEFace_val_ptr &face_supp_fe_vals_);
  /*!
   * \details We detach the \c unique_ptr of dealii::FEValues and
   * dealii::FEFaceValues from the current object. parameters are similar to the
   * \c Cell_Class::attach_FEValues.
  */

  /*!
   * \details Updates the FEValues which are connected to the current element
   * (not the FEFaceValues.)
   */
  void reinit_cell_fe_vals();
  /*!
   * \details Updates the FEFaceValues which are connected to a given face of
   * the current element.
   * \param i_face the face which we want to update the connected FEFaceValues.
   * \c i_face\f$\in\{1,2,3,4\}\f$
   */
  void reinit_face_fe_vals(unsigned i_face);

  void detach_FEValues(FE_val_ptr &cell_quad_fe_vals_,
                       FEFace_val_ptr &face_quad_fe_vals_,
                       FE_val_ptr &cell_supp_fe_vals_,
                       FEFace_val_ptr &face_supp_fe_vals_);

  double get_error_in_cell(const TimeFunction<dim, double> &func,
                           const Eigen::MatrixXd &input_vector,
                           const double &time = 0);

  template <int func_output_dim>
  double get_error_in_cell(
    const TimeFunction<dim, dealii::Tensor<1, func_output_dim> > &func,
    const Eigen::MatrixXd &modal_vector,
    const double &time = 0);

  double get_error_on_faces(const TimeFunction<dim, double> &func,
                            const Eigen::MatrixXd &input_vector,
                            const double &time = 0);

  template <int func_output_dim>
  double get_error_on_faces(
    const TimeFunction<dim, dealii::Tensor<1, func_output_dim> > &func,
    const Eigen::MatrixXd &modal_vector,
    const double &time = 0);

  virtual void internal_vars_errors(const eigen3mat &u_vec,
                                    const eigen3mat &q_vec,
                                    double &u_error,
                                    double &q_error);

  /*!
  * This function projects the function "func" to the current basis.
  * For modal basis of the dual space this is calculated via:
  * \f[f(x)=\sum_{i} \alpha_i N_i(x) \Longrightarrow
  * \left(f(x),N_j(x)\right) = \sum_i \alpha_i \left(N_i(x),N(x)\right).\f]
  * Now, if \f$N_i\f$'s are orthonormal to each other, then \f$ \alpha_i =
  * (f,N_i)\f$.
  * If, we use Lagrangian basis for dual space, in order to
  * project a function onto this basis, we need to just calculate the value
  * of the function at the corresponding support points.
  */
  template <typename BasisType, typename func_out_type>
  void project_essential_BC_to_face(
    const TimeFunction<dim, func_out_type> &func,
    const poly_space_basis<BasisType, dim - 1> &the_basis,
    const std::vector<double> &weights,
    mtl::vec::dense_vector<func_out_type> &vec,
    const double &time = 0);

  template <typename BasisType, typename func_out_type>
  void
  project_func_to_face(const TimeFunction<dim, func_out_type> &func,
                       const poly_space_basis<BasisType, dim - 1> &the_basis,
                       const std::vector<double> &weights,
                       mtl::vec::dense_vector<func_out_type> &vec,
                       const unsigned &i_face,
                       const double &time = 0);

  template <typename BasisType, typename func_out_type>
  void
  project_flux_BC_to_face(const Function<dim, func_out_type> &func,
                          const poly_space_basis<BasisType, dim - 1> &the_basis,
                          const std::vector<double> &weights,
                          mtl::vec::dense_vector<func_out_type> &vec);

  template <typename BasisType, typename func_out_type>
  void project_to_elem_basis(const TimeFunction<dim, func_out_type> &func,
                             const poly_space_basis<BasisType, dim> &the_basis,
                             const std::vector<double> &weights,
                             mtl::vec::dense_vector<func_out_type> &vec,
                             const double &time = 0.0);

  const unsigned n_faces;
  unsigned poly_order, n_face_bases, n_cell_bases;
  unsigned id_num;
  /**
   * We want to know which degrees of freedom are restrained and which are open.
   * Hence, we store a bitset which has its size equal to the number of dofs of
   * each face of the cell and it is 1 if the dof is open, and 0 if it is
   * restrained.
   */
  std::vector<boost::dynamic_bitset<> > dof_names_on_faces;

  void assign_local_global_cell_data(const unsigned &i_face,
                                     const unsigned &local_num_,
                                     const unsigned &global_num_,
                                     const unsigned &comm_rank_,
                                     const unsigned &half_range_);

  void assign_local_cell_data(const unsigned &i_face,
                              const unsigned &local_num_,
                              const int &comm_rank_,
                              const unsigned &half_range_);

  void assign_ghost_cell_data(const unsigned &i_face,
                              const int &local_num_,
                              const int &global_num_,
                              const unsigned &comm_rank_,
                              const unsigned &half_range_);

  /*! A unique ID of each cell, which is taken from the dealii cell
   * corresponding to the current cell. This ID is unique in the
   * interCPU space. */
  std::string cell_id;
  static int do_not_count_face_dofs;
  std::vector<unsigned> half_range_flag;
  std::vector<unsigned> face_owner_rank;
  dealiiCell dealii_cell;
  std::vector<std::vector<int> > dofs_ID_in_this_rank;
  std::vector<std::vector<int> > dofs_ID_in_all_ranks;
  std::vector<BC> BCs;
  std::unique_ptr<dealii::FEValues<dim> > cell_quad_fe_vals, cell_supp_fe_vals;
  std::unique_ptr<dealii::FEFaceValues<dim> > face_quad_fe_vals,
    face_supp_fe_vals;
  /*
   * Now, we will have some idiotically publicly implemented pointers !
   * Not far from now, we should change this.
   */
  local_nodal_sol<dim> *refn_local_nodal;
  local_nodal_sol<dim> *cell_local_nodal;

  poly_space_basis<elem_basis_type, dim> *the_elem_basis;
  poly_space_basis<face_basis_type, dim - 1> *the_face_basis;
  const dealii::QGauss<dim> *elem_quad_bundle;
  const dealii::QGauss<dim - 1> *face_quad_bundle;
};

#include "elements/advection_diffusion.hpp"
#include "elements/diffusion.hpp"
#include "elements/explicit_gn_dispersive.hpp"
//#include "elements/explicit_gn_dispersive_modif.hpp"
#include "elements/explicit_nswe.hpp"
//#include "elements/explicit_nswe_modif.hpp"
#include "elements/gn_eps_0_beta_0.hpp"
#include "elements/nswe.hpp"

#include "cell_class.cpp"
#endif // CELL_CLASS_HPP
