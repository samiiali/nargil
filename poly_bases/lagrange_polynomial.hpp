#include <Eigen/Dense>
#include <deal.II/base/tensor.h>
#include <float.h>
#include <vector>

#ifndef LAGRANGE_POLIES_H
#define LAGRANGE_POLIES_H

/*!
 * \brief
 * The Lagrangian polinomial basis.
 * \ingroup basis_funcs
 */
template <int dim>
class LagrangePolys
{
 public:
  /**
   *
   */
  LagrangePolys();

  /**
   *
   */
  LagrangePolys(const std::vector<dealii::Point<1, double> > &support_points_,
                const int &domain_);

  /**
   * I only declare this constructor, because C++ does not have a static_if and
   * I am too lazy to do anything else.
   */
  LagrangePolys(const unsigned &, const int &);

  /**
   *
   */
  LagrangePolys(LagrangePolys &&input);

  /**
   *
   */
  LagrangePolys<dim> &operator=(LagrangePolys<dim> &&input);

  /**
   *
   */
  ~LagrangePolys();

  static poly_basis_category get_basis_category();
  std::vector<double> value(const dealii::Point<dim, double> &P0) const;
  std::vector<double> value(const dealii::Point<dim, double> &P0,
                            const unsigned &half_range) const;
  std::vector<double> value(const double &) const;
  std::vector<dealii::Tensor<1, dim> >
  grad(const dealii::Point<dim, double> &P0) const;

 private:
  std::vector<double> compute(const double &x_) const;
  double compute_Li(const double &x_, const unsigned &i_poly) const;

  /**
   * Computes the derivative of all of the 1D basis functions at a given
   * point. For this purpose, we use the formula:
   * \f[
   * L'_i(x) = L_i(x) \sum_{m=0, m\ne i}^n \frac {1}{x-x_m}
   * \f]
   */
  std::vector<double> derivative(double) const;

  std::vector<dealii::Point<1> > support_points;
  unsigned int polyspace_order;
  int domain;
};

#include "lagrange_polynomial.cpp"

#endif // LAGRANGE_POLIES_H
