#include "cell_class.hpp"

#ifndef EXPLICIT_GN_DISPERSIVE
#define EXPLICIT_GN_DISPERSIVE

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_gn_dispersive_qs
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double & = 0) const final
  {
    dealii::Tensor<1, in_point_dim + 1, double> qs;
    qs[0] = 7.;
    qs[1] = 5 + sin(M_PI * x[1]);
    qs[2] = -3. - cos(M_PI * x[0]);
    return qs;
  }
};

/**
 * @ingroup cells
 * This element forms the equations for the nonlinear disperesive part of the
 * Green-Naghdi equation. We first explore the case of flat bottom and later
 * we explain the formulation for the arbitrary bottom topography.
 */

#endif
