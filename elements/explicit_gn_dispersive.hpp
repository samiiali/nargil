#include "cell_class.hpp"
#include "explicit_nswe.hpp"

#ifndef EXPLICIT_GN_DISPERSIVE
#define EXPLICIT_GN_DISPERSIVE

double alpha = 1.0;

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_gn_dispersive_grad_grad_b_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double & = 0) const final
  {
    dealii::Tensor<2, 2> grad_grad_b;
    // Example 1 !
    /*
    grad_grad_b[0][0] = sin(x[0] + 2 * x[1]) * sin(x[0] + 2 * x[1]);
    grad_grad_b[0][1] = 2 * sin(x[0] + 2 * x[1]) * sin(x[0] + 2 * x[1]);
    grad_grad_b[1][0] = 2 * sin(x[0] + 2 * x[1]) * sin(x[0] + 2 * x[1]);
    grad_grad_b[1][1] = 4 * sin(x[0] + 2 * x[1]) * sin(x[0] + 2 * x[1]);
    */
    // End of example 1 !
    // Example 2
    grad_grad_b[0][0] = 0.;
    grad_grad_b[0][1] = 0.;
    grad_grad_b[1][0] = 0.;
    grad_grad_b[1][1] = 0.;
    // End of example 2
    return grad_grad_b;
  }
};

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_gn_dispersive_qis_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    dealii::Tensor<1, in_point_dim + 1> qis;
    // Example zero !
    /*
    qis[0] = 5 + sin(4 * x[0] - t);
    qis[1] = cos(5 * x[1] - t);
    qis[2] = sin(3 * x[1] + t);
    */
    // End of example zero !
    /*
    qis[0] = 5;
    qis[1] = 0;
    qis[2] = 0;
    */
    // Dissertation example 2
    if (t < 2)
    {
      qis[0] = 10. + sin(M_PI * 2 * t);
    }
    else
    {
      qis[0] = 10.;
    }
    qis[1] = qis[2] = 0;
    // End of Dissertation example 2
    return qis;
  }
};

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_gn_dispersive_hVinf_t_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t) const final
  {

    dealii::Tensor<1, in_point_dim> hVinf_t;
    // Example zero !
    hVinf_t[0] = sin(5 * x[1] - t);
    hVinf_t[1] = cos(3 * x[1] + t);
    // End of example zero !
    // Dissertation example 2
    hVinf_t[0] = 0.;
    hVinf_t[1] = 0.;
    // End of dissertation example 2
    return hVinf_t;
  }
};

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_gn_dispersive_W2_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double & = 0) const final
  {
    double W2;
    // Example zero !
    W2 = 2. * (9. * cos(x[0]) + cos(x[0] + 2 * x[1]) - 9 * sin(x[1]) +
               sin(2 * x[0] + x[1]));
    // End of example zero !

    // Example 1
    /*
    W2 = (5 + x[0]) *
         ((5 + x[0]) * cos(x[0]) - sin(x[0]) - (5 + x[0]) * sin(x[1]));
    */
    // End of example 1.
    return W2;
  }
};

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_gn_dispersive_L_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    dealii::Tensor<1, in_point_dim> L;

    double x1 = x[0];
    double y1 = x[1];
    double g = 9.81;
    /*
    L[0] =
      -(36 * alpha * cos(x1 - 3 * y1) -
        2 * alpha * (-1673 + 6 * alpha) * cos(x1 - y1) +
        105 * alpha * cos(2 * x1 - y1) + 48 * g * cos(y1) +
        234 * alpha * cos(y1) - 96 * pow(alpha, 2) * cos(y1) +
        777 * alpha * cos(3 * y1) + 3 * alpha * cos(5 * y1) +
        480 * g * cos(x1 + y1) + 2450 * alpha * cos(x1 + y1) -
        348 * pow(alpha, 2) * cos(x1 + y1) - 2010 * alpha * cos(3 * (x1 + y1)) +
        36 * pow(alpha, 2) * cos(3 * (x1 + y1)) +
        180 * alpha * cos(5 * (x1 + y1)) - 771 * alpha * cos(2 * x1 + y1) +
        128 * pow(alpha, 2) * cos(2 * x1 + y1) +
        3072 * alpha * cos(3 * x1 + y1) + 74 * alpha * cos(x1 + 3 * y1) -
        156 * pow(alpha, 2) * cos(x1 + 3 * y1) - 48 * g * cos(2 * x1 + 3 * y1) -
        324 * alpha * cos(2 * x1 + 3 * y1) +
        48 * pow(alpha, 2) * cos(2 * x1 + 3 * y1) -
        105 * alpha * cos(4 * x1 + 3 * y1) - 20 * alpha * cos(5 * x1 + 3 * y1) +
        762 * alpha * cos(x1 + 5 * y1) - 72 * alpha * cos(2 * x1 + 5 * y1) +
        48 * pow(alpha, 2) * cos(2 * x1 + 5 * y1) -
        2742 * alpha * cos(3 * x1 + 5 * y1) +
        129 * alpha * cos(4 * x1 + 5 * y1) -
        135 * alpha * cos(2 * x1 + 7 * y1) -
        378 * alpha * cos(3 * x1 + 7 * y1) +
        150 * alpha * cos(4 * x1 + 7 * y1) + 30 * alpha * cos(5 * x1 + 7 * y1) -
        3 * alpha * cos(6 * x1 + 7 * y1) + 12 * alpha * cos(4 * x1 + 9 * y1) -
        258 * alpha * sin(x1) - 624 * pow(alpha, 2) * sin(x1) -
        5180 * alpha * sin(2 * x1) + 6 * alpha * sin(3 * x1) -
        372 * alpha * sin(x1 - 2 * y1) + 3 * alpha * sin(2 * (x1 - y1)) +
        96 * pow(alpha, 2) * sin(2 * y1) + 48 * g * sin(2 * (x1 + y1)) +
        535 * alpha * sin(2 * (x1 + y1)) -
        192 * pow(alpha, 2) * sin(2 * (x1 + y1)) -
        1100 * alpha * sin(4 * (x1 + y1)) + 9 * alpha * sin(6 * (x1 + y1)) +
        452 * alpha * sin(2 * (2 * x1 + y1)) + 480 * g * sin(x1 + 2 * y1) +
        3174 * alpha * sin(x1 + 2 * y1) -
        64 * pow(alpha, 2) * sin(x1 + 2 * y1) +
        3332 * alpha * sin(2 * (x1 + 2 * y1)) -
        48 * pow(alpha, 2) * sin(2 * (x1 + 2 * y1)) -
        318 * alpha * sin(3 * (x1 + 2 * y1)) -
        24 * alpha * sin(4 * (x1 + 2 * y1)) +
        372 * alpha * sin(3 * x1 + 2 * y1) +
        1467 * alpha * sin(2 * (x1 + 3 * y1)) - 90 * alpha * sin(x1 + 4 * y1) -
        24 * pow(alpha, 2) * sin(x1 + 4 * y1) -
        72 * alpha * sin(3 * x1 + 4 * y1) +
        24 * pow(alpha, 2) * sin(3 * x1 + 4 * y1) -
        6 * alpha * sin(5 * x1 + 4 * y1) + 42 * alpha * sin(x1 + 6 * y1) -
        540 * alpha * sin(4 * x1 + 6 * y1) + 42 * alpha * sin(5 * x1 + 6 * y1) -
        132 * alpha * sin(3 * x1 + 8 * y1) +
        12 * alpha * sin(5 * x1 + 8 * y1)) /
      (96. * alpha);
    L[1] =
      -(-108 * alpha * cos(x1 - 3 * y1) +
        2 * alpha * (-1673 + 6 * alpha) * cos(x1 - y1) +
        210 * alpha * cos(2 * x1 - y1) + 96 * g * cos(y1) +
        372 * alpha * cos(y1) - 768 * pow(alpha, 2) * cos(y1) +
        1554 * alpha * cos(3 * y1) + 6 * alpha * cos(5 * y1) +
        480 * g * cos(x1 + y1) + 2450 * alpha * cos(x1 + y1) -
        444 * pow(alpha, 2) * cos(x1 + y1) - 2010 * alpha * cos(3 * (x1 + y1)) +
        36 * pow(alpha, 2) * cos(3 * (x1 + y1)) +
        180 * alpha * cos(5 * (x1 + y1)) - 1542 * alpha * cos(2 * x1 + y1) +
        64 * pow(alpha, 2) * cos(2 * x1 + y1) +
        1024 * alpha * cos(3 * x1 + y1) + 222 * alpha * cos(x1 + 3 * y1) -
        564 * pow(alpha, 2) * cos(x1 + 3 * y1) - 96 * g * cos(2 * x1 + 3 * y1) -
        648 * alpha * cos(2 * x1 + 3 * y1) +
        96 * pow(alpha, 2) * cos(2 * x1 + 3 * y1) -
        210 * alpha * cos(4 * x1 + 3 * y1) - 12 * alpha * cos(5 * x1 + 3 * y1) +
        3810 * alpha * cos(x1 + 5 * y1) - 144 * alpha * cos(2 * x1 + 5 * y1) +
        96 * pow(alpha, 2) * cos(2 * x1 + 5 * y1) -
        4570 * alpha * cos(3 * x1 + 5 * y1) +
        258 * alpha * cos(4 * x1 + 5 * y1) -
        270 * alpha * cos(2 * x1 + 7 * y1) -
        882 * alpha * cos(3 * x1 + 7 * y1) +
        300 * alpha * cos(4 * x1 + 7 * y1) + 42 * alpha * cos(5 * x1 + 7 * y1) -
        6 * alpha * cos(6 * x1 + 7 * y1) + 24 * alpha * cos(4 * x1 + 9 * y1) -
        324 * alpha * sin(x1) - 96 * pow(alpha, 2) * sin(x1) +
        12 * alpha * sin(3 * x1) - 744 * alpha * sin(x1 - 2 * y1) -
        3 * alpha * sin(2 * (x1 - y1)) + 3092 * alpha * sin(2 * y1) +
        480 * pow(alpha, 2) * sin(2 * y1) - 1116 * alpha * sin(4 * y1) +
        48 * g * sin(2 * (x1 + y1)) + 535 * alpha * sin(2 * (x1 + y1)) -
        96 * pow(alpha, 2) * sin(2 * (x1 + y1)) -
        1100 * alpha * sin(4 * (x1 + y1)) + 9 * alpha * sin(6 * (x1 + y1)) +
        226 * alpha * sin(2 * (2 * x1 + y1)) + 960 * g * sin(x1 + 2 * y1) +
        6348 * alpha * sin(x1 + 2 * y1) -
        128 * pow(alpha, 2) * sin(x1 + 2 * y1) +
        6664 * alpha * sin(2 * (x1 + 2 * y1)) -
        96 * pow(alpha, 2) * sin(2 * (x1 + 2 * y1)) -
        636 * alpha * sin(3 * (x1 + 2 * y1)) -
        48 * alpha * sin(4 * (x1 + 2 * y1)) +
        744 * alpha * sin(3 * x1 + 2 * y1) +
        4401 * alpha * sin(2 * (x1 + 3 * y1)) - 180 * alpha * sin(x1 + 4 * y1) -
        48 * pow(alpha, 2) * sin(x1 + 4 * y1) -
        144 * alpha * sin(3 * x1 + 4 * y1) +
        48 * pow(alpha, 2) * sin(3 * x1 + 4 * y1) -
        12 * alpha * sin(5 * x1 + 4 * y1) + 84 * alpha * sin(x1 + 6 * y1) -
        810 * alpha * sin(4 * x1 + 6 * y1) + 84 * alpha * sin(5 * x1 + 6 * y1) -
        264 * alpha * sin(3 * x1 + 8 * y1) +
        24 * alpha * sin(5 * x1 + 8 * y1)) /
      (96. * alpha);
    */

    /*
    L[0] =
      (128 * pow(cos(t - 4 * x1), 3) * pow(cos(t - 5 * y1), 2) +
       96 * cos(t - 5 * y1) * cos(t + 3 * y1) * pow(-5 + sin(t - 4 * x1), 2) *
         sin(t - 4 * x1) -
       160000 * g * sin(2 * (t - 4 * x1)) -
       640 * pow(cos(t - 5 * y1), 2) * sin(2 * (t - 4 * x1)) +
       360 * pow(cos(t + 3 * y1), 2) * sin(2 * (t - 4 * x1)) +
       88000 * g * sin(t - 4 * x1) * sin(2 * (t - 4 * x1)) +
       75 * sin(t - 5 * y1) +
       80 * alpha * pow(cos(t - 4 * x1), 2) * (5 - 2 * sin(t - 4 * x1)) *
         sin(t - 5 * y1) -
       30 * sin(t - 4 * x1) * sin(t - 5 * y1) +
       2000 * alpha * sin(t - 4 * x1) * sin(t - 5 * y1) +
       3 * pow(sin(t - 4 * x1), 2) * sin(t - 5 * y1) -
       1200 * alpha * pow(sin(t - 4 * x1), 2) * sin(t - 5 * y1) +
       240 * alpha * pow(sin(t - 4 * x1), 3) * sin(t - 5 * y1) -
       16 * alpha * pow(sin(t - 4 * x1), 4) * sin(t - 5 * y1) +
       4 * alpha * pow(sin(2 * (t - 4 * x1)), 2) * sin(t - 5 * y1) +
       900 * alpha * sin(2 * (t - 4 * x1)) * sin(t + 3 * y1) +
       4000 * sin(t - 4 * x1) * sin(t - 5 * y1) * sin(t + 3 * y1) -
       1600 * pow(sin(t - 4 * x1), 2) * sin(t - 5 * y1) * sin(t + 3 * y1) +
       160 * pow(sin(t - 4 * x1), 3) * sin(t - 5 * y1) * sin(t + 3 * y1) -
       8 * cos(t - 4 * x1) *
         (-25000 * g - 32 * pow(cos(t - 5 * y1), 2) * pow(sin(t - 4 * x1), 2) +
          5600 * g * pow(sin(t - 4 * x1), 3) -
          680 * g * pow(sin(t - 4 * x1), 4) + 32 * g * pow(sin(t - 4 * x1), 5) +
          9 * pow(cos(t + 3 * y1), 2) * (25 + pow(sin(t - 4 * x1), 2)) +
          375 * alpha * sin(t + 3 * y1) +
          45 * alpha * pow(sin(t - 4 * x1), 2) * sin(t + 3 * y1) -
          3 * alpha * pow(sin(t - 4 * x1), 3) * sin(t + 3 * y1))) /
      (3. * pow(-5 + sin(t - 4 * x1), 2));

    L[1] =
      (3 * cos(t + 3 * y1) * (-5 + sin(t - 4 * x1)) *
         (-2 - 153 * alpha + 3 * alpha * cos(2 * (t - 4 * x1)) +
          60 * alpha * sin(t - 4 * x1)) +
       320 * pow(cos(t - 4 * x1), 2) * sin(2 * (t - 5 * y1)) -
       10 * cos(t - 4 * x1) * (-102 * alpha * cos(t - 5 * y1) +
                               alpha * cos(3 * t - 8 * x1 - 5 * y1) +
                               alpha * cos(t - 8 * x1 + 5 * y1) -
                               16 * sin(2 * (t - y1)) - 256 * sin(8 * y1)) -
       4 * (50 * alpha * cos(t - 5 * y1) * sin(2 * (t - 4 * x1)) +
            4 * sin(2 * (t - 4 * x1)) * (sin(2 * (t - y1)) + 16 * sin(8 * y1)) +
            27 * pow(-5 + sin(t - 4 * x1), 2) * sin(2 * (t + 3 * y1)))) /
      (6. * (-5 + sin(t - 4 * x1)));
    */

    // Dissertation example 2
    L[0] = 0;
    L[1] = 0;
    // End of dissertation example 2
    return L;
  }
};

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_gn_dispersive_W1_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    double g = 9.81;
    dealii::Tensor<1, in_point_dim, double> w1;
    double x1 = x[0];
    double y1 = x[1];
    // Example 1
    w1[0] = sin(x[0]);
    w1[1] = cos(x[1]);
    // End of Example 1
    // Example 2
    w1[0] = 4. * g * cos(t - 4 * x1) * (5. - sin(t - 4. * x1)) / alpha +
            sin(t - 5 * y1);
    w1[1] = -cos(t + 3 * y1);
    // End of example 2
    /*
    w1[0] = 0.;
    w1[1] = 0.;
    */
    return w1;
  }
};

/**
 * ## Governing Equation
 * @ingroup cells
 * This element forms the equations for the nonlinear disperesive part of the
 * Green-Naghdi equation. We first explore the case of flat bottom and later
 * explain the formulation for the arbitrary bottom topography.
 * The equation that we want to solve can be written in the following form with
 * dimension:
 * \f[
 * \begin{cases}
 *   \partial_t h = 0 \\
 *   \partial_t(hV) - \frac 1 \alpha gh \nabla \zeta +
 *     (1+ \alpha h \mathcal T \frac 1 h)^{-1}
 *     \left[\frac 1 \alpha gh \nabla \zeta + h \mathcal Q_1(V)\right] = 0
 * \end{cases}
 * \f]
 * With,
 * \f[
 *   \begin{gathered}
 *     \mathcal T[]W= \mathcal R_1[](\nabla \cdot W)+
 *       \beta \mathcal R_2[](\nabla b \cdot W)
 *       \xrightarrow{b=\text{const.}}
 *       \mathcal T[]W= \mathcal R_1[](\nabla \cdot W),\\
 *     \mathcal Q_1(V) = -2 \mathcal R_1\left(\partial_1 V \cdot
 *       \partial_2 V^{\perp}+ (\nabla \cdot V)^2\right)+
 *       \beta \mathcal R_2\left(V \cdot (V \cdot \nabla)\nabla b\right)
 *       \xrightarrow{b=\text{const.}}
 *       \mathcal Q_1(V) = -2 \mathcal R_1\left(\partial_1 V \cdot
 *       \partial_2 V^{\perp}+ (\nabla \cdot V)^2\right),\\
 *     \mathcal R_1(W) = -\frac 1 {3h} \nabla (h^3 W)
 *                       - \beta \frac h 2 W \nabla b
 *     \xrightarrow{b=\text{const.}}
 *     \mathcal R_1(W)=-\frac 1 {3h} \nabla (h^3 W),\\
 *     \mathcal R_2(W) = \frac 1 {2h} \nabla (h^2 W) + \beta W \nabla b
 *     \xrightarrow{b=\text{const.}}
 *     \mathcal R_2(W) = \frac 1 {2h} \nabla (h^2 W).
 *   \end{gathered}
 * \f]
 * This equaion should be solved in two steps:
 * 1. Solve \f$(1+\alpha h \mathcal T \frac 1 h) W_1 =
 *   \frac 1 \alpha gh \nabla \zeta + h \mathcal Q_1(V)\f$
 *   based on the values of \f$\zeta\f$ and \f$V\f$ of the last step
 *   and taking \f$h\f$ constant.
 * 2. Solve \f$\partial_t (hV) - \frac 1 \alpha gh \nabla \zeta + W_1 = 0\f$,
 *    with \f$W_1\f$ taken from the last step.
 *
 * ### 1. Solving the static differential operator
 * The static part of the problem, i.e. \f$(1+\alpha h \mathcal T \frac 1 h) W_1
 * =
 * \frac 1 \alpha gh \nabla \zeta + h \mathcal Q_1(V)\f$, results in the
 * following
 * extended form:
 * \f[
 *   \begin{aligned}
 *   W_1 + \alpha h \mathcal T (\frac 1 h W)
 *     &= W_1 + \alpha h \mathcal R_1(\nabla \cdot \frac 1 h W_1) \\
 *     &= W_1 - \frac \alpha 3 \nabla (h^3 \nabla \cdot (\frac 1 h W_1))
 *   \end{aligned}
 * \f]
 * \f[
 *   \begin{aligned}
 *   \frac 1 \alpha gh \nabla \zeta + h \mathcal Q_1(V)
 *     &= \frac 1 \alpha gh \nabla \zeta - 2h \mathcal R_1
 *        (\partial_1 V\cdot \partial_2 V^\perp + (\nabla \cdot V)^2) \\
 *     &= \frac 1 \alpha gh \nabla \zeta
 *      + \frac 2 3 \nabla(\partial_1 V\cdot \partial_2 V^\perp
 *                         + (\nabla \cdot V)^2)
 *   \end{aligned}
 * \f]
 * We introduce a new unknown to write the above equation in terms of
 * two first order equations:
 * \f[
 *   \begin{cases}
 *     h^{-3} W_2 - \nabla \cdot (\frac 1 h W_1) = 0 \\
 *     W_1 - \frac \alpha 3 \nabla (W_2) = \frac 1 \alpha gh \nabla \zeta
 *      + \frac 2 3 \nabla(\partial_1 V\cdot \partial_2 V^\perp
 *                         + (\nabla \cdot V)^2)
 *   \end{cases}
 * \f]
 * We want to satisfy this equation in weak sense; hence, we require for all
 * \f$p_1, p_2\f$ in the space of the appropriate test functions:
 * \f[
 *   \begin{aligned}
 *   (h^{-3} W_2, p_2) - \langle \frac 1 h {\hat W}_1 \cdot n , p_2 \rangle
 *              + (\frac 1 h W_1, \nabla p_2) &= 0 \\
 *   (W_1, p_1) - \frac \alpha 3 \langle W_2^* \cdot n , p_1 \rangle
 *              + \frac \alpha 3 (W_2, \nabla \cdot p_1)
 *     &= \left(\frac 1 \alpha gh \nabla \zeta
 *      + \frac 2 3 \nabla(\partial_1 V\cdot \partial_2 V^\perp
 *                         + (\nabla \cdot V)^2), p_1 \right)
 *   \end{aligned}
 * \f]
 * With the following definition for the numerical flux \f$W_2^*\f$:
 * \f[
 *   W_2^* \cdot n= W_2 I \cdot n + \tau (W_1 - \hat W_1)
 * \f]
 * We will get:
 * \f[
 *   \begin{cases}
 *   (h^{-3} W_2, p_2) - \langle \frac 1 h {\hat W}_1 \cdot n , p_2 \rangle
 *              + (\frac 1 h W_1, \nabla p_2) = 0 \\
 *   (W_1, p_1) - \frac \alpha 3 ( \nabla W_2 , p_1 )
 *              - \frac \alpha 3 \langle \tau W_1, p_1 \rangle
 *              + \frac \alpha 3 \langle \tau \hat W_1, p_1 \rangle
 *     = \left(\frac 1 \alpha gh \nabla \zeta
 *      + \frac 2 3 \nabla(\partial_1 V\cdot \partial_2 V^\perp
 *                         + (\nabla \cdot V)^2), p_1 \right)
 *   \end{cases}
 * \f]
 * \f$W_2\f$ is a scalar valued function and \f$W_1\f$ is a vector valued
 * function. Thus, \f$W_2^*\f$ and \f$\tau\f$ are 2-tensors. Defining the
 * following bilinear operators:
 * \f[
 *   \begin{gathered}
 *   a_{02} (W_2,p_2) = (h^{-3} W_2,p_2); \quad
 *   b_{01}^T (W_1, p_2) = (\frac 1 h W_1, \nabla p_2); \quad
 *   c_{01} (\hat W_1, p_2) = \langle \frac 1 h {\hat W}_1 \cdot n, p_2 \rangle;
 *   \\
 *   a_{01} (W_1, p_1) = (W_1, p_1); \quad
 *   b_{02} (W_2, p_1) = (\nabla W_2, p_1); \quad
 *   d_{01} (W_1, p_1) = \langle \tau W_1, p_1 \rangle; \quad
 *   c_{02} (\hat W_1,p_1) = \langle \tau \hat W_1, p_1 \rangle.
 *   \end{gathered}
 * \f]
 * So, we obtain:
 * \f[
 * \begin{cases}
 *   A_{02} W_2 + B_{01}^T W_1 - C_{01} \hat W_1 = 0 \\
 *   A_{01} W_1 - \frac \alpha 3 B_{02} W_2 - \frac \alpha 3 D_{01} W_1
 *              + \frac \alpha 3 C_{02} \hat W_1 = L_{01}
 * \end{cases}
 * \f]
 *
 * Solving this system of equations, we will obtain \f$W_1\f$. Next, we
 * substitute \f$W_1\f$ in the second equation and solve \f$hV\f$ in the next
 * time step.
 *
 * We also want to satisfy the flux conservation condition, which means
 * \f$\langle [\![ W_2^* \cdot n ]\!], \mu\rangle = 0\f$, which means:
 * \f[
 *   \langle W_2 , \mu \cdot n \rangle
 *   + \langle \tau W_1 , \mu \rangle
 *   - \langle \tau \hat W_1 , \mu \rangle = 0
 * \f]
 * Which requires the following bilinear forms:
 * \f[
 *   C_{03}^T (W_2, \mu) = \langle W_2, \mu \cdot n \rangle ; \quad
 *   C_{04}^T (W_1, \mu) = \langle \tau W_1 , \mu \rangle; \quad
 *   E_{01} (\hat W_1 , \mu) = \langle \tau \hat W_1 , \mu \rangle.
 * \f]
 *
 * to get:
 *
 * \f[
 *   C_{03}^T W_2 + C_{04}^T W_1 - E_{01} \hat W_1 = 0.
 * \f]
 *
 *
 * ### 2. Solving the transient part
 * After we substitute \f$W_1\f$ from the above solution, we get:
 * \f[
 *   \partial_t (h V) = \frac 1 \alpha g h \nabla \zeta - W_1
 * \f]
 * Hence, we can go to the next stage or time step.
 *
 * ## Equations with varying topography:
 * \f$h \mathcal Q_1\f$ has the following form:
 * \f[
 * \begin{aligned}
 * hQ_1(V) =& \frac 2 3 \nabla \left(h^3 \partial_x V \cdot
 * \partial_yV^\perp+h^3 (\nabla\cdot V)^2\right)
 * +\frac 1 2 \nabla \left(h^2 V\cdot (V\cdot \nabla)\nabla b\right) \notag \\
 * &+ h^2 \left(\partial_x V \cdot \partial_y V^\perp + (\nabla \cdot V)^2
 * \right) \nabla b
 * + h\left(V \cdot (V \cdot \nabla)\nabla b \right) \nabla b.
 * \end{aligned}
 * \f]
 * On the right side of the dispersive part we will have \f$\frac 1 \alpha g h
 * \nabla \zeta + h \mathcal Q_1(V)\f$. We test this equation by \f$P_1 \f$
 * and use divergence theorem for the terms with 2nd derivatives. Hence, we
 * will get:
 * \f[
 * \begin{aligned}
 * (h\mathcal Q_1(V), P_1) &=
 * \frac 2 3 \left<h^3\partial_x V \cdot \partial_y V^{\perp} + h^3 (\nabla
 * \cdot V)^2 , P_1\cdot \mathbf n \right>
 * - \frac 2 3 \left(h^3 \partial_x V \cdot \partial_y V^{\perp} + h^3 (\nabla
 * \cdot V)^2 , \nabla \cdot P_1 \right) \\
 * &+ \frac 1 2 \left< h^2 V\cdot (V \cdot \nabla)\nabla b, P_1 \cdot \mathbf n
 * \right>
 * - \frac 1 2 \left(h^2 V\cdot (V \cdot \nabla)\nabla b,\nabla \cdot P_1
 * \right) \\
 * &+ \left( h^2 \nabla b \left( \partial_x V \cdot \partial_y V^{\perp}
 * + (\nabla \cdot V)^2 \right) , P_1 \right)
 *  + \left(h \nabla b \left(V \cdot (V \cdot \nabla)\nabla b \right),
 * P_1 \right)
 * \end{aligned}
 * \f]
 * Let us use the following naming (\f$\nabla \zeta = \nabla h + \nabla b\f$):
 * \f[
 * \begin{gathered}
 * L_{21} = \frac 2 3 \left<h^3\partial_x V \cdot \partial_y V^{\perp} + h^3
 * (\nabla \cdot V)^2 , P_1\cdot \mathbf n \right>
 * + \frac 1 2 \left< h^2 V\cdot (V \cdot \nabla)\nabla b, P_1 \cdot \mathbf n
 * \right>, \\
 * L_{10} = \frac 1 \alpha g \left( h (\nabla h + \nabla b), P_1 \right),\\
 * L_{11} = - \frac 2 3 \left(h^3 \partial_x V \cdot \partial_y V^{\perp} + h^3
 * (\nabla \cdot V)^2 , \nabla \cdot P_1 \right)
 * - \frac 1 2 \left(h^2 V\cdot (V \cdot \nabla)\nabla b,\nabla \cdot P_1
 * \right), \\
 * L_{12} = \left( h^2 \nabla b \left( \partial_x V \cdot \partial_y V^{\perp}
 * + (\nabla \cdot V)^2 \right) , P_1 \right)
 *  + \left(h \nabla b \left(V \cdot (V \cdot \nabla)\nabla b \right),
 * P_1 \right).
 * \end{gathered}
 * \f]
 *
 * ### Boundary Conditions:
 * Here, we mainly focus on two types of boundary conditons, inflow/outflow and
 * solid wall. We want to set \f$\left<\mathcal B_h, \mu\right> = 0\f$. For
 * solid wall boundary, we have:
 * \f[
 *  \mathcal B_h = W_{1h} - (W_{1h}\cdot \mathbf n) \mathbf n - \hat W_{1h}
 * -\left(\tfrac 1 \alpha g h \nabla \zeta_h \cdot \mathbf n\right) \mathbf n.
 * \f]
 * So we define:
 * \f[
 * c^T_{34}(W_{1h}, \mu) = \left< W_{1h} , \mu \right> - \left<W_{1h} \cdot
 * \mathbf n, \mu \cdot \mathbf n\right>; \quad
 * e_{31}(\hat W_{1h},\mu) = \left<\hat W_{1h}, \mu \right>; \quad
 * l_{31}(\mu) = \left<\tfrac 1 \alpha g h (\nabla h + \nabla b) \cdot \mathbf
 * n, \mu \cdot \mathbf n \right>.
 * \f]
 * For inflow/outflow boundary we have:
 * \f[
 * \mathcal B_h = v_n^{+} W_{1h} - |v_n| \hat W_{1h} - v_n^- \left(\tfrac 1
 * \alpha g h \nabla \zeta_h - \partial_t (hV^{\infty}) \right),
 * \f]
 * where, \f$v_n = V_h \cdot \mathbf n\f$, and \f$v_n^{\pm} = v_n/2 \pm
 * |v_n|/2\f$. So, we define:
 * \f[
 * c^T_{34}(W_{1h}, \mu) = \left< v_n^+ W_{1h} , \mu \right>; \quad
 * e_{31}(\hat W_{1h},\mu) = \left<|v_n| \hat W_{1h}, \mu \right>; \quad
 * l_{31}(\mu) = \left<\tfrac 1 \alpha g h (\nabla h + \nabla b) -\partial_t(h
 * V^\infty) , v_n^- \mu \right>.
 * \f]
 * Before including the boundary condition, we had \f$C_{03}^T W_2 + C_{04}^T
 * W_1 - E_{01} \hat W_1 = 0\f$. By including the above terms on the boundary,
 * we can write the final form of the conservation of the numerical flux as:
 * \f[
 * \langle W_2^* \cdot \mathbf n, \mu \rangle_{\partial \mathcal T \backslash
 * \partial \Omega} + \langle \mathcal B_h, \mu \rangle_{\partial \Omega} = 0.
 * \Longrightarrow
 * C_{03}^T W_2 + (C_{04}^T + C^T_{34}) W_1 -(E_{01} + E_{31}) \hat W_1 =
 * L_{31}.
 * \f]
 */
template <int dim>
struct explicit_gn_dispersive : public GenericCell<dim>
{
  const double alpha = 1.0;
  const double gravity = 9.81;
  using elem_basis_type = typename GenericCell<dim>::elem_basis_type;
  typedef std::unique_ptr<dealii::FEValues<dim> > FE_val_ptr;
  typedef std::unique_ptr<dealii::FEFaceValues<dim> > FEFace_val_ptr;
  typedef dealii::Tensor<1, dim + 1, double> nswe_vec;

  static solver_options required_solver_options();
  static solver_type required_solver_type();
  static unsigned get_num_dofs_per_node();

  /**
   * \brief Constructor for the GN_eps_0_beta_0.
   * \details Since we are using factory pattern to create cells,
   * we call this constructor, from the GenericCell::make_cell. In this case,
   * due to IS-A rule, the constructor of the GenericCell will be called as
   * well.
   * This is not good, because we have already deleted the default constructor
   * of the GenericCell. So, we call our desired constructor by putting it in
   * the
   * initializer list.
   * \param inp_cell
   * \param id_num_
   */
  explicit_gn_dispersive() = delete;
  explicit_gn_dispersive(const explicit_gn_dispersive &inp_cell) = delete;
  explicit_gn_dispersive(explicit_gn_dispersive &&inp_cell) noexcept;
  explicit_gn_dispersive(
    typename GenericCell<dim>::dealiiCell &inp_cell,
    const unsigned &id_num_,
    const unsigned &poly_order_,
    hdg_model_with_explicit_rk<dim, explicit_gn_dispersive> *model_);
  ~explicit_gn_dispersive() final;
  eigen3mat A001, A02, B01T, C01, A01, B02, D01, C02, A03, B03T, L01, C03T, E01,
    C04T, L10, L11, L12, L21, C34T, E31, L31;
  void assign_BCs(const bool &at_boundary,
                  const unsigned &i_face,
                  const dealii::Point<dim> &face_center);

  void calculate_matrices();

  void calculate_stage_matrices();

  void assign_initial_data();

  void set_previous_step_results(eigen3mat *last_step_q);

  eigen3mat *get_previous_step_results();

  /*
  void set_previous_step_results1(ResultPacket result);

  ResultPacket get_previous_step_results1();
  */

  void assemble_globals(const solver_update_keys &keys_);

  template <typename T>
  double compute_internal_dofs(const double *const local_uhat_vec,
                               eigen3mat &solved_u_vec,
                               eigen3mat &solved_q_vec,
                               const poly_space_basis<T, dim> &output_basis);

  void internal_vars_errors(const eigen3mat &u_vec,
                            const eigen3mat &q_vec,
                            double &u_error,
                            double &q_error);

  double get_iteration_increment_norm(const double *const local_uhat);

  void ready_for_next_time_step();

  void ready_for_next_stage(double *const local_uhat);

  void ready_for_next_iteration();

  std::vector<double> taus;
  hdg_model_with_explicit_rk<dim, explicit_gn_dispersive> *model;
  explicit_RKn<4, original_RK> *time_integrator;

  static explicit_gn_dispersive_qis_class<dim, nswe_vec>
    explicit_gn_dispersive_qs;
  static explicit_gn_dispersive_hVinf_t_class<dim, dealii::Tensor<1, dim> >
    hVinf_t_func;
  static explicit_nswe_grad_b_func_class<dim, dealii::Tensor<1, dim> >
    explicit_nswe_grad_b_func;
  static explicit_gn_dispersive_grad_grad_b_class<dim, dealii::Tensor<2, dim> >
    grad_grad_b_func;
  static explicit_gn_dispersive_L_class<dim, dealii::Tensor<1, dim> >
    explicit_gn_dispersive_L;
  static explicit_gn_dispersive_W1_class<dim, dealii::Tensor<1, dim> >
    explicit_gn_dispersive_W1;
  static explicit_gn_dispersive_W2_class<dim, double> explicit_gn_dispersive_W2;

  eigen3mat last_step_q;
  eigen3mat last_stage_q;
  /*
   * Last step qhat is only required for some of the boundary integrals.
   * Actually we only use last step h_hat, and hv1_hat and hv2_hat are
   * unnecessary. The point is that in this element h and h_hat are constant
   * due to partial_t h being zero. So, we can use h for all stages in a given
   * step. We move this variable between different elements, so there is
   * no concern about storage.
  eigen3mat last_step_qhat;
   */

  std::vector<eigen3mat> ki_s;
};

#include "explicit_gn_dispersive.cpp"

#endif
