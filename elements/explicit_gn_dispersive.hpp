#include "cell_class.hpp"
#include "explicit_nswe.hpp"

#ifndef EXPLICIT_GN_DISPERSIVE
#define EXPLICIT_GN_DISPERSIVE

double alpha = 1.0;

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_gn_dispersive_h_t_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    double h_t;
    double x1 = x[0];
    double y1 = x[1];
    double g = 9.81;

    h_t = 0.2 * cos(4. * x1 + t);

    return h_t;
  }
};

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
struct explicit_gn_dispersive_g_h_grad_zeta_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    dealii::Tensor<1, in_point_dim> g_h_grad_zeta;
    // Example zero !
    /*
    qis[0] = 2 + exp(sin(x[0] + x[1] - t));
    qis[1] = cos(x[0] - 4 * t);
    qis[2] = sin(x[1] + 4 * t);
    */
    double g = 9.81;
    //    g_h_grad_zeta[0] = 0.;
    g_h_grad_zeta[0] = 4. / alpha * g * (5. + sin(4. * x[0])) * cos(4. * x[0]);
    //    g_h_grad_zeta[0] = 4 / alpha * g * (5. + sin(4 * x[0])) * cos(x[0]);
    g_h_grad_zeta[1] = 0.;
    // End of example zero !
    // Example 1
    /*
    qis[0] = 5. + sin(4 * x[0] - t);
    qis[1] = 3.;
    qis[2] = 3.;
    */
    // End of example 1
    // Dissertation example 2
    /*
    if (t < 2)
    {
      qis[0] = 10. + sin(M_PI * 2 * t);
    }
    else
    {
      qis[0] = 10.;
    }
    qis[1] = qis[2] = 0;
    */
    // End of Dissertation example 2
    return g_h_grad_zeta;
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
    qis[0] = 2 + exp(sin(x[0] + x[1] - t));
    qis[1] = cos(x[0] - 4 * t);
    qis[2] = sin(x[1] + 4 * t);
    */
    double g = 9.81;
    qis[0] = 1. + 0.2 * sin(4 * x[0] + t);
    qis[1] = cos(x[0] - t);
    //    qis[1] =
    //      1. / alpha * g * (-pow(sin(x[0] - t), 2) / 2. - 5. * sin(x[0] - t));
    qis[2] = 0.;
    // End of example zero !
    // Example 1
    /*
    qis[0] = 5. + sin(4 * x[0] - t);
    qis[1] = 3.;
    qis[2] = 3.;
    */
    // End of example 1
    // Dissertation example 2
    /*
    if (t < 2)
    {
      qis[0] = 10. + sin(M_PI * 2 * t);
    }
    else
    {
      qis[0] = 10.;
    }
    qis[1] = qis[2] = 0;
    */
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
    /*
    hVinf_t[0] = cos(x[0] - 4 * t);
    hVinf_t[1] = sin(x[1] + 4 * t);
    */
    hVinf_t[0] = sin(x[0] - t);
    hVinf_t[1] = 0.;
    // End of example zero !
    // Dissertation example 2
    /*
    hVinf_t[0] = 0.;
    hVinf_t[1] = 0.;
    */
    // End of dissertation example 2
    return hVinf_t;
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
    // Example zero
    /*
    w1[0] = ((2 + exp(-sin(t - x1 - y1))) * g * cos(t - x1 - y1)) /
              (exp(sin(t - x1 - y1)) * alpha) +
            4 * sin(4 * t - x1);
    w1[1] = ((2 + exp(-sin(t - x1 - y1))) * g * cos(t - x1 - y1)) /
              (exp(sin(t - x1 - y1)) * alpha) -
            4 * cos(4 * t + y1);
    */
    /*
    w1[0] = (4 * g * cos(4 * x1) * (5. + sin(4. * x1))) / alpha;
    w1[1] = 0.;
    */
    //    w1[0] = cos(t - x1) + (4. * g * cos(4. * x1) * (5. + sin(4. * x1))) /
    //    alpha;
    w1[0] =
      sin(t - x1) +
      (4. * g * cos(t + 4 * x1) * (1 + sin(t + 4 * x1) / 5.)) / (5. * alpha);
    w1[1] = 0.;
    // End of example zero
    // Example 1
    /*
    w1[0] = (4 * g * cos(t - 4 * x1) * (5. - sin(t - 4. * x1))) / alpha;
    w1[1] = 0;
    */
    // End of Example 1
    return w1;
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
                            const double &t = 0) const final
  {
    double W2;
    double x1 = x[0];
    double y1 = x[1];
    double g = 9.81;
    // Example zero
    W2 = pow(1 + sin(t + 4 * x1) / 5., 3) *
         ((-4 * cos(t + 4 * x1) *
           (sin(t - x1) +
            (4 * g * cos(t + 4 * x1) * (1 + sin(t + 4 * x1) / 5.)) /
              (5. * alpha))) /
            (5. * pow(1 + sin(t + 4 * x1) / 5., 2)) +
          (-cos(t - x1) + (16 * g * pow(cos(t + 4 * x1), 2)) / (25. * alpha) -
           (16 * g * (1 + sin(t + 4 * x1) / 5.) * sin(t + 4 * x1)) /
             (5. * alpha)) /
            (1 + sin(t + 4 * x1) / 5.));
    // End of example zero.
    // Example 1
    //    W2 = 16. * g * pow((5. - sin(t - 4 * x1)), 3) * sin(t - 4 * x1) /
    //    alpha;
    // End of example 1

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

    // Including everything
    L[0] =
      (-903500 * cos(t - 6 * x1) - 6000 * (-25 + 37 * alpha) * cos(5 * x1) -
       12500 * cos(5 * (t + 2 * x1)) + 457500 * cos(3 * t + 2 * x1) -
       150000 * cos(2 * t + 3 * x1) + 428000 * alpha * cos(2 * t + 3 * x1) -
       42000 * cos(t + 4 * x1) + 3914880 * g * cos(t + 4 * x1) -
       30000 * cos(3 * (t + 4 * x1)) - 720320 * g * cos(3 * (t + 4 * x1)) +
       5440 * g * cos(5 * (t + 4 * x1)) - 22000 * alpha * cos(4 * t + 11 * x1) +
       16000 * alpha * cos(2 * t + 13 * x1) - 13500 * cos(t + 14 * x1) +
       765000 * sin(t - x1) + 118550 * alpha * sin(t - x1) +
       50000 * sin(2 * (t - x1)) + 195000 * sin(10 * x1) -
       875 * alpha * sin(5 * (t + 3 * x1)) + 600000 * sin(2 * (t + 4 * x1)) +
       2739840 * g * sin(2 * (t + 4 * x1)) - 90112 * g * sin(4 * (t + 4 * x1)) +
       128 * g * sin(6 * (t + 4 * x1)) + 325000 * sin(4 * t + 6 * x1) -
       7500 * sin(3 * t + 7 * x1) + 176700 * alpha * sin(3 * t + 7 * x1) +
       7500 * sin(t + 9 * x1) - 116300 * alpha * sin(t + 9 * x1) +
       675 * alpha * sin(3 * t + 17 * x1)) /
      (30000. * pow(5 + sin(t + 4 * x1), 2));
    // Including only 1/alpha g h grad zeta
    /*
    L[0] =
      (-2500 * alpha * cos(5 * x1) + 4500 * alpha * cos(2 * t + 3 * x1) +
       34880 * g * cos(t + 4 * x1) - 2880 * g * cos(3 * (t + 4 * x1)) +
       7500 * sin(t - x1) + 2550 * alpha * sin(t - x1) +
       19456 * g * sin(2 * (t + 4 * x1)) - 128 * g * sin(4 * (t + 4 * x1)) +
       875 * alpha * sin(3 * t + 7 * x1) - 675 * alpha * sin(t + 9 * x1)) /
      7500.;
    */
    L[1] = 0;

    // End of example for second equation alone
    return L;
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
struct explicit_gn_dispersive
  : public GenericCell<dim> // There is a partial definition of this class at
                            // explicit_nswe.hpp ... also in solver.hpp
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

  void set_previous_step_results(explicit_nswe<dim> const *const src_cell);

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

  void produce_trace_of_prim_vars(const explicit_nswe<dim> *const src_cell);

  void compute_avg_prim_vars_flux(const explicit_nswe<dim> *const src_cell,
                                  double const *const local_prim_vars_sums,
                                  double const *const local_face_count);

  void compute_prim_vars_derivatives();

  void produce_trace_of_grad_prim_vars(const explicit_nswe<dim> *const src_cell);

  void compute_avg_grad_V_flux(const explicit_nswe<dim> *const src_cell,
                               double const *const local_V_x_sums,
                               double const *const local_V_y_sums);

  void compute_grad_grad_V();

  std::vector<double> taus;
  hdg_model_with_explicit_rk<dim, explicit_gn_dispersive> *model;
  explicit_RKn<4, original_RK> *time_integrator;

  static explicit_gn_dispersive_h_t_class<dim, double> h_t_func;
  static explicit_gn_dispersive_g_h_grad_zeta_class<dim,
                                                    dealii::Tensor<1, dim> >
    g_h_grad_zeta_func;
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
  eigen3mat last_step_qhat;
  eigen3mat last_stage_q;

  std::vector<double> connected_face_count;
  eigen3mat avg_prim_vars_flux;
  eigen3mat grad_h;
  eigen3mat div_V;
  eigen3mat grad_V;          // This contains (v1_x | v1_y | v2_x | v2_y)^T
  eigen3mat avg_grad_V_flux; //   face 1  |  face 2 |  face 3 | face 4
                             // (1x 1y ...|1x 1y ...|1x 1y ...|1x 1y ...)
  eigen3mat grad_grad_V;

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
