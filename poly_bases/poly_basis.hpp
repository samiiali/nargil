#include "lib_headers.hpp"
#include <type_traits>

#ifndef POLY_BASIS
#define POLY_BASIS

/*!
 * \defgroup basis_funcs Basis Functions
 * \brief
 * This module contains all classes which are used to construct the polynomial
 * basis inside each element, and on their faces.
 */

enum Domain
{
  From_0_to_1 = 1 << 0,
  From_minus_1_to_1 = 1 << 1
};

enum poly_basis_category
{
  modal_basis = 1 << 0,
  nodal_basis = 1 << 1
};

#include "jacobi_polynomial.hpp"
#include "lagrange_polynomial.hpp"

/*!
 * \brief The static base class for all other polynomial basis.
 *
 * This structure contains the basis functions, their gradients,
 * and their divergence. The main motivation behind this is to avoid the
 * repeated calculation of bases on a unit cell for every element. This
 * structure has a constructor which takes quadrature points as inputs and
 * stores the values of shape funcions on those points.
 * \tparam dim Is the dimension of the basis. When you also want to evaluate the
 * basis at a given point, that point shoud also have the same dimension of the
 * basis itself. For example, to construct the basis for the faces of a three
 * dimensional element, you cannot initiate a two dimensional basis with three
 * integration points.
 * \tparam derived_basis Is the type of polynomial basis which approximates the
 * dual space of each element. It can either be a Jacobi_Poly_Basis or
 * Lagrange_Polys.
 * \ingroup basis_funcs
 */
template <typename derived_basis, int dim>
class poly_space_basis
{
 public:
  static poly_basis_category get_category();

  /**
   * It might be good to get rid of default constructor, but we are not
   * doing it now
   */
  poly_space_basis();

  /**
   * For nodal elements, this constructor, creates the polynomial basis
   * of the master element based on the given @c support_points. On the other
   * hand, if the element is a modal element, we use the number of given
   * support points to create a modal basis. As you can see, you can have
   * different values for number of quadrature points and number of basis
   * functions. By execution of this constructor the bases[i][j] will contain
   * the value of jth basis function at ith point. You can assume that functions
   * are stored in different columns of the same row. Also, different rows
   * correspond to different points.
   *
   *                                function j
   *                                    |
   *                                    |
   *
   *                            x  ...  x  ...  x
   *                            .       .       .
   *                            .       :       .
   *                            .       x       .
   *          point i ---->     x      Bij ...  x
   *                            .       x       .
   *                            .       :       .
   *                            .       .       .
   *                            x  ...  x  ...  x
   *
   *
   * Now, if we consider the matrix B_ij = @c bases[i][j]
   * To convert from modal to quadrature points (where modes are stored
   * in a column vector), we use: @c {Ni = Bij * Mj}
   */
  poly_space_basis(const std::vector<dealii::Point<dim> > &integration_points,
                   const std::vector<dealii::Point<1, double> > &support_points,
                   const int &domain_ = Domain::From_0_to_1);

  /**
   * This constructor only works for modal bases, or nodal basis
   * with LGL points (equidistant nodal element is not implemented yet !).
   */
  poly_space_basis(const std::vector<dealii::Point<dim> > &integration_points,
                   const unsigned &max_poly_order,
                   const int &domain_ = Domain::From_0_to_1);

  /**
   * The move constructor
   */
  poly_space_basis(poly_space_basis<derived_basis, dim> &&) noexcept;

  /**
   * The move assignment operator.
   */
  poly_space_basis &operator=(poly_space_basis<derived_basis, dim> &&) noexcept;

  /**
   * This function forms the poly_space_basis::bases,
   * poly_space_basis::basis_at_quads and poly_space_basis::bases_grads_at_quads
   */

  /**
   *
   */
  ~poly_space_basis();

  /**
   *
   */
  eigen3mat get_func_vals_at_iquad(const unsigned i_quad) const;

  /**
   *
   */
  eigen3mat get_dof_vals_at_quads(const eigen3mat &in_dof) const;

  /**
   *
   */
  void compute_quad_values(
    const std::vector<dealii::Point<dim> > &integration_points_);

  /*!
   * This function gives you the values of half-range basis functions, which
   * will be used in the adaptive meshing. The idea is to give the basis
   * corresponding to the unrefined element neghboring current element. For
   * example consider point x on the edge of element 1, instead of giving the
   * value of bases corresponding to element 1, we will give the value of
   * basis functions of the element 0.
   *
   *               |   0   |
   *               |_______|
   *
   *               |\
   *               | *  <------  we will give this value !
   *               |  \
   *               |-x-\---|
   *                    \  |
   *                     \ |
   *                      \|
   *
   *               |---|---|
   *               | 1 | 2 |
   */
  std::vector<double> value(const dealii::Point<dim, double> &P0,
                            const unsigned half_range = 0);

  /**
   *
   */
  std::vector<dealii::Tensor<1, dim> >
  grad(const dealii::Point<dim, double> &P0);

  unsigned n_polys;
  unsigned n_quads;

  /*!
   * \brief This matrix constains the gradient of each basis function
   * at different integration points.
   *
   * The \f$i\f$th row and \f$j\f$th column contains, the gradient of
   * \f$i\f$th basis function at \f$j\f$th integration points. Let us
   * denote this matrix with \f$K_{ij}\f$ and each shape function with
   * \f$N_i\f$. Then: \f$K_{ij} = \partial N_i/\partial \hat x_{k}\f$,
   * at \f$j\f$th integration point (obviously \f$K_{ij}\f$ is a first
   * order tensor). Now, if we want to calculate
   * \f$\partial N_i/\partial x_{\ell}\f$, we have to multiply
   * \f$K_{ij}\f$ by the differential form \f$\partial \hat x_{k}
   * / \partial x_{\ell}\f$. Due to the dealii::Tensor implementation
   * this differential form must be post-multiplied by \f$K_{ij}\f$.
   * This is the reason that we store \c bases_grads_at_quads in this order
   * and we also make it column major, because of the way that we assign
   * its values to it.
   */
  mtl::mat::dense2D<dealii::Tensor<1, dim>,
                    mtl::mat::parameters<mtl::tag::col_major> >
    bases_grads_at_quads;

 private:
  bool ready_2b_used;
  derived_basis poly_basis;
  Eigen::MatrixXd basis_at_quads;
};

#include "poly_basis.cpp"

#endif // POLY_BASIS
