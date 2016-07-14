#include <cmath>
#include <string>
#include <vector>

#ifndef Jacobi_Polynomials
#define Jacobi_Polynomials

#include "poly_basis.hpp"

/*!
 * \brief Forms a basis for the dual sapce of the reference element, using the
 * Jacobi polynomials.
 * \ingroup basis_funcs
 * \tparam dim Is the dimension of this basis; so for the face of a 3D element,
 * \c dim is euqal to 2.
 */
template <int dim>
class JacobiPolys //: public Poly_Basis<Jacobi_Poly_Basis<dim>, dim>
{
 public:
  JacobiPolys();

  /**
   * This constructor uses \f$\alpha=0, \beta =0\f$ to create Legendre
   * polynomial as the basis functions. As a result, this constructor cannot
   * be used for constructing the derivative of Legendre polynomials, which
   * are Jacobi polynomials with \f$\alpha = 1, \beta =1\f$.
   */
  JacobiPolys(const unsigned &polyspace_order_, const int &domain_);

  /**
   * This constructor is here, because C++ does not have @c static_if and I
   * am very lazy to implement a better solution! It will interupt the code
   * as soon as it is called.
   */
  JacobiPolys(const std::vector<dealii::Point<1, double> > &, const int &);

  /**
   * Only, when we want to construct the derivative of the polynomial basis, we
   * use this constructor. Its \f$\alpha, \beta\f$ can be adjusted to give us
   * the derivative of Jacobi polynomials.
   */
  JacobiPolys(const unsigned &polyspace_order_,
              const double &alpha_,
              const double &beta_,
              const int &domain_);

  /**
   * We might need move constructor.
   */
  JacobiPolys(JacobiPolys &&input) noexcept;

  /**
   * We might need move assignment operator
   */
  JacobiPolys<dim> &operator=(JacobiPolys<dim> &&input) noexcept;

  /**
   *
   */
  static poly_basis_category get_basis_category();

  ~JacobiPolys();

  std::vector<double> value(const dealii::Point<dim, double> &P0);
  std::vector<double> value(const dealii::Point<dim, double> &P0,
                            const unsigned &half_range);
  std::vector<dealii::Tensor<1, dim> >
  grad(const dealii::Point<dim, double> &P0);

 private:
  const double integral_sc_fac;
  unsigned int polyspace_order;
  double alpha, beta;
  int domain;
  std::vector<double> value(const double &);
  std::vector<double> derivative(const double &);
  std::vector<double> compute(const double &x_inp);
  inline double change_coords(const double &x_inp);
};

#include "jacobi_polynomial.cpp"

#endif
