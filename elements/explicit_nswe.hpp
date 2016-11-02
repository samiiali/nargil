#include "cell_class.hpp"

#ifndef EXPLICIT_NL_SWE_HPP
#define EXPLICIT_NL_SWE_HPP

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_nswe_grad_b_func_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double & = 0) const final
  {
    dealii::Tensor<1, in_point_dim, double> grad_b;
    // GN example 1:
    /*
    grad_b[0] = sin(x[0] + 2. * x[1]);
    grad_b[1] = 2. * sin(x[0] + 2. * x[1]);
    */
    /* End of GN example 1 */
    // Example 2
    grad_b[0] = 0.;
    grad_b[1] = 0.;
    // End of example 2
    return grad_b;
  }
};

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_nswe_zero_func_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &,
                            const dealii::Point<in_point_dim> &,
                            const double & = 0) const final
  {
    dealii::Tensor<1, in_point_dim + 1, double> qs;
    /* Example 1: */
    qs[0] = 0.;
    qs[1] = 0.;
    qs[2] = 0.;
    /* End of Example 1 */
    return qs;
  }
};

/*!
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_nswe_qis_func_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    dealii::Tensor<1, in_point_dim + 1, double> qs;
    // Example 0:
    /*
    qs[0] = 7.;
    qs[1] = 5 + sin(M_PI * x[1]);
    qs[2] = -3. - cos(M_PI * x[0]);
    */
    // End of Example 0
    // Example 0.1:
    /*
    qs[0] = 7.;
    qs[1] = 5. * sin(x[1]);
    qs[2] = -3.;
    */
    // End of Example 0.1:
    // Example 0.2:
    /*
    qs[0] = 7. + sin(x[0]);
    qs[1] = 5.;
    qs[2] = -3.;
    */
    // End of Example 0.2:
    // Example 0.3:
    /*
    qs[0] = 7. + t;
    qs[1] = 5.;
    qs[2] = -3.;
    */
    // End of Example 0.3:
    /* Example 1 */
    //     Example 1:
    /*
    qs[0] = 7 + sin(x[0] - t);
    qs[1] = 5;
    qs[2] = -3;
    */
    // End of Example 1.
    //     Example 2:
    /*
    qs[0] = 2. + exp(sin(x[0] + x[1] - t));
    qs[1] = cos(x[0] - 4 * t);
    qs[2] = sin(x[1] + 4 * t);
    */
    // End of Example 2.
    // First example in paper 3
    /*
    double x0 = x[0];
    double y0 = x[1];
    double t0 = t;
    qs[0] = 2 + exp(sin(3 * x0) * sin(3 * y0) - sin(3 * t0));
    qs[1] = cos(x0 - 4 * t0);
    qs[2] = sin(y0 + 4 * t0);
    */
    // End of first example in paper 3
    // Second example in paper 3
    /*
    double t0 = t;
    if (t0 <= 1.0)
      qs[0] = 2 + cos(M_PI * t0);
    else
      qs[0] = 1.;

    qs[1] = 0.;
    qs[2] = 0.;
    */
    // End of second example in paper 3
    // Narrowing channel in paper 3
    /*
    if (t < 1.e-8)
    {
      qs[0] = 1.5;
      qs[1] = 1.e-3;
      qs[2] = 0.;
    }
    if (x[0] < -2. + 1.e-6 && t > 1.e-8)
    {
      qs[0] = std::min(3.0, 1.5 + 10 * t);
      qs[1] = 1.5;
      qs[2] = 0;
    }
    if (x[0] > 2 - 1.e-6)
    {
      qs[0] = 1.5;
      qs[1] = 1.e-3;
      qs[2] = 0;
    }
    */
    // End of narrowing channel in paper 3
    // G-N example zero
    /*
    double x0 = x[0];
    double y0 = x[1];
    double t0 = t;
    qs[0] = 5. + sin(4 * x[0]);
    qs[1] = 3.;
    qs[2] = 3.;
    */
    // G-N example zero
    // G-N example 1
    double x0 = x[0];
    double y0 = x[1];
    double t0 = t;
    qs[0] = 5. + sin(4. * x0);
    qs[1] = sin(x0 - t0);
    qs[2] = 0.;
    // G-N example 1
    // G-N example 2
    /*
    double x0 = x[0];
    double y0 = x[1];
    double t0 = t;
    qs[0] = 5 + sin(4 * x0 - t0);
    qs[1] = cos(5 * y0 - t0);
    qs[2] = sin(3 * y0 + t);
    */
    // Checking inflow and outflow BC
    /*
    double t0 = t;
    if (t0 < 1.E-6)
    {
      if (x[0] >= 19.5 && x[0] <= 20.5)
        qs[0] = 2. + cos(M_PI * (x[0] - 20.));
      else
        qs[0] = 2.;
    }
    else
    {
      qs[0] = 2.;
    }
    qs[1] = 0;
    qs[2] = 0;
    */
    //
    // Exact solution of Green-Naghdi:
    /*
    if (t0 <= 1.e-6)
    {
      double a_GN = 0.25;
      double h_b = 1.0;
      double x_0 = 20.;
      double g = 9.81;
      double c_GN = sqrt(g * (h_b + a_GN));
      double kappa_GN = sqrt(3. * a_GN) / 2. / h_b / sqrt(h_b + a_GN);
      double zeta_GN =
        pow(acosh(1. / (kappa_GN * (x[0] - x_0 - c_GN * t0))), 2);
      qs[0] = h_b + zeta_GN;
      qs[1] = (c_GN * (1. - h_b / (zeta_GN + h_b))) * qs[0];
      qs[2] = 0.;
    }
    else
    {
      qs[0] = 0;
      qs[1] = 0;
      qs[2] = 0;
    }
    */
    // End of checking inflow and outflow BC

    return qs;
  }
};

/*!
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_nswe_L_func_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    double x0 = x[0];
    double y0 = x[1];
    dealii::Tensor<1, in_point_dim + 1, double> L;
    // Example 0: */
    /*
    L[0] = 0.;
    L[1] = -(M_PI * (3 + cos(M_PI * x0)) * cos(M_PI * y0)) / 7.;
    L[2] = (M_PI * sin(M_PI * x0) * (5 + sin(M_PI * y0))) / 7.;
    */
    // End of Example 0 */
    // Example 0.1
    /*
    L[0] = 0.;
    L[1] = -15. / 7. * cos(y0);
    L[2] = 0.;
    */
    // End of Example 0.1
    // Example 0.2
    /*
    L[0] = 0.;
    L[1] = (cos(x[0]) * (-25 + pow(7 + sin(x[0]), 3))) / pow(7 + sin(x[0]), 2);
    L[2] = (15 * cos(x[0])) / pow(7 + sin(x[0]), 2);
    */
    // End of Example 0.2
    // Example 0.3
    /*
    L[0] = 1.0;
    L[1] = 0.0;
    L[2] = 0.0;
    */
    // End of Example 0.3
    // Example 1:
    /*
    L[0] = -cos(t - x0);
    L[1] = (cos(t - x0) * (-25 + pow(7 - sin(t - x0), 3))) / pow(-7 + sin(t -
    x0), 2);
    L[2] = (15 * cos(t - x0)) / pow(-7 + sin(t - x0), 2);
    */
    // End of Example 1
    // Example 2:
    /*
    double g = 9.81;
    L[0] = -(cos(t - x0 - y0) / exp(sin(t - x0 - y0))) + cos(4 * t + y0) +
           sin(4 * t - x0);
    L[1] =
      ((pow(1 + 2 * exp(sin(t - x0 - y0)), 3) * g * cos(t - x0 - y0)) /
         exp(4 * sin(t - x0 - y0)) -
       (pow(cos(4 * t - x0), 2) * cos(t - x0 - y0)) / exp(sin(t - x0 - y0)) +
       (2 + exp(-sin(t - x0 - y0))) * cos(4 * t - x0) * cos(4 * t + y0) +
       (2 + exp(-sin(t - x0 - y0))) * sin(8 * t - 2 * x0) -
       4 * pow(2 + exp(-sin(t - x0 - y0)), 2) * sin(4 * t - x0) -
       (cos(4 * t - x0) * cos(t - x0 - y0) * sin(4 * t + y0)) /
         exp(sin(t - x0 - y0)) +
       pow(2 + exp(-sin(t - x0 - y0)), 3) * g * sin(x0 + 2 * y0)) /
      pow(2 + exp(-sin(t - x0 - y0)), 2);
    L[2] =
      ((pow(1 + 2 * exp(sin(t - x0 - y0)), 3) * g * cos(t - x0 - y0)) /
         exp(4 * sin(t - x0 - y0)) +
       4 * pow(2 + exp(-sin(t - x0 - y0)), 2) * cos(4 * t + y0) -
       (cos(4 * t - x0) * cos(t - x0 - y0) * sin(4 * t + y0)) /
         exp(sin(t - x0 - y0)) +
       (2 + exp(-sin(t - x0 - y0))) * sin(4 * t - x0) * sin(4 * t + y0) -
       (cos(t - x0 - y0) * pow(sin(4 * t + y0), 2)) / exp(sin(t - x0 - y0)) +
       (2 + exp(-sin(t - x0 - y0))) * sin(2 * (4 * t + y0)) +
       2 * pow(2 + exp(-sin(t - x0 - y0)), 3) * g * sin(x0 + 2 * y0)) /
      pow(2 + exp(-sin(t - x0 - y0)), 2);
    */
    // End of Example 2
    // First example of paper 3
    /*
    double t0 = t;
    L[0] = -3 * exp(-sin(3 * t0) + sin(3 * x0) * sin(3 * y0)) * cos(3 * t0) +
           cos(4 * t0 + y0) + sin(4 * t0 - x0);
    L[1] = (-3 * exp(3 * sin(3 * t0) + sin(3 * x0) * sin(3 * y0)) *
              pow(cos(4 * t0 - x0), 2) * cos(3 * x0) * sin(3 * y0) +
            (2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0))) *
              (exp(3 * sin(3 * t0)) * sin(8 * t0 - 2 * x0) -
               (2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0))) *
                 (4 * exp(2 * sin(3 * t0)) * sin(4 * t0 - x0) -
                  3 * exp(sin(3 * x0) * sin(3 * y0)) *
                    (2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0))) *
                    cos(3 * x0) * sin(3 * y0))) +
            exp(3 * sin(3 * t0)) * cos(4 * t0 - x0) *
              ((2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0))) *
                 cos(4 * t0 + y0) -
               3 * exp(sin(3 * x0) * sin(3 * y0)) * cos(3 * y0) * sin(3 * x0) *
                 sin(4 * t0 + y0))) /
           (exp(2 * sin(3 * t0)) *
            pow(2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0)), 2));
    L[2] = (4 * exp(2 * sin(3 * t0)) *
              pow(2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0)), 2) *
              cos(4 * t0 + y0) +
            3 * exp(sin(3 * x0) * sin(3 * y0)) * cos(3 * y0) * sin(3 * x0) *
              (pow(2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0)), 3) -
               exp(3 * sin(3 * t0)) * pow(sin(4 * t0 + y0), 2)) +
            exp(3 * sin(3 * t0)) *
              ((2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0))) *
                 sin(4 * t0 - x0) * sin(4 * t0 + y0) -
               3 * exp(sin(3 * x0) * sin(3 * y0)) * cos(4 * t0 - x0) *
                 cos(3 * x0) * sin(3 * y0) * sin(4 * t0 + y0) +
               (2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0))) *
                 sin(2 * (4 * t0 + y0)))) /
           (exp(2 * sin(3 * t0)) *
            pow(2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0)), 2));
    */
    // End of first example of paper 3
    // example 2 of paper 3
    //    L[0] = L[1] = L[2] = x0 - x0 + y0 - y0;
    // End of example 2 of paper 3

    // Green-Naghdi example
    /*
    double g = 9.81;
    L[0] = (2 * cos(t - 2 * x0)) / 3. +
           exp(sin(t + x0 + y0)) * cos(t + x0 + y0) - sin(t + y0);
    L[1] = (-3 * pow(2 + exp(sin(t + x0 + y0)), 2) * cos(t - 2 * x0) +
            exp(sin(t + x0 + y0)) * cos(t + x0 + y0) *
              (9 * pow(2 + exp(sin(t + x0 + y0)), 3) +
               3 * cos(t + y0) * sin(t - 2 * x0) - pow(sin(t - 2 * x0), 2)) +
            (2 + exp(sin(t + x0 + y0))) *
              (-2 * sin(2 * (t - 2 * x0)) + 3 * sin(t - 2 * x0) * sin(t + y0) +
               9 * pow(2 + exp(sin(t + x0 + y0)), 2) * g * sin(x0 + 2 * y0))) /
           (9. * pow(2 + exp(sin(t + x0 + y0)), 2));
    L[2] = (2 * (2 + exp(sin(t + x0 + y0))) * cos(t - 2 * x0) * cos(t + y0) +
            exp(sin(t + x0 + y0)) * cos(t + x0 + y0) *
              (3 * pow(2 + exp(sin(t + x0 + y0)), 3) - 3 * pow(cos(t + y0), 2) +
               cos(t + y0) * sin(t - 2 * x0)) -
            3 * (2 + exp(sin(t + x0 + y0))) *
              ((2 + exp(sin(t + x0 + y0))) * sin(t + y0) + sin(2 * (t + y0)) -
               2 * pow(2 + exp(sin(t + x0 + y0)), 2) * g * sin(x0 + 2 * y0))) /
           (3. * pow(2 + exp(sin(t + x0 + y0)), 2));
    */
    // End of Green-Naghdi example
    // Example zero of Green-Naghdi
    double g = 9.81;
    L[0] = -cos(t - 4 * x0);
    L[1] = (4 * cos(t - 4 * x0) * (-9 + g * pow(5 - sin(t - 4 * x0), 3))) /
           pow(-5 + sin(t - 4 * x0), 2);
    L[2] = (-36 * cos(t - 4 * x0)) / pow(-5 + sin(t - 4 * x0), 2);
    // End of example zero of Green-Naghdi
    // Dissertation example 2
    L[0] = 0;
    L[1] = 0;
    L[2] = 0;
    // End of Dissertaton example 2
    return L;
  }
};

/*!
 * \ingroup cells
 * This element forms the equations for the fully nonlinear shallow water
 * equation (i.e. the case with (\f$\mu = 0, \epsilon = 1, \beta = 1\f$)).
 * We write the equation in the balanced form as below:
 * \f[
 *   \mathbf q_t + \nabla \cdot \mathbb F(\mathbf q) = \mathbf L \tag{1}.
 * \f]
 * With the following definitions:
 * \f[
 *   \mathbf q
 *   =
 *   \begin{Bmatrix}
 *     h \\ h \mathbf V
 *   \end{Bmatrix}
 *   =
 *   \begin{Bmatrix}
 *     h \\ hV_1 \\ hV_2
 *   \end{Bmatrix}, \quad
 *   \mathbb F(\mathbf q)
 *   =
 *   \begin{Bmatrix}
 *     h \mathbf V \\ h\mathbf V \otimes \mathbf V + \frac 1 2 h^2 \mathbf I
 *   \end{Bmatrix}
 *   =
 *   \begin{bmatrix}
 *     hV_1 & hV_2 \\
 *     \frac {(hV_1)^2} h + \frac 1 2 h^2 & \frac {(hV_1) (hV_2)} h \\
 *     \frac {(hV_2) (hV_1)} h & \frac {(hV_2)^2} h + \frac 1 2 h^2
 *   \end{bmatrix}
 * \f]
 * ### Solving NLSWE by explicit Runge-Kutta and hybridized DG
 * Now, let us solve Eq. (1) by hybridized DG. To this end,we want the
 * following variational form to hold in the domain:
 * \f[
 *   (\mathbf q_t, \mathbf p)
 *     + \langle \widehat{\mathbb F \cdot \mathbf n} , \mathbf p \rangle
 *     - (\mathbb F, \nabla \mathbf p)
 *     - \mathbf L(\mathbf p) = 0.
 * \f]
 * In order to obtain the numerical flux \f$\widehat {\mathbb F \cdot
 * \mathbf n}\f$, we use the following formula:
 * \f[
 *   \widehat {\mathbb F \cdot \mathbf n}
 *   =
 *   \mathbb F(\hat {\mathbf q}) \cdot \mathbf n
 *   + \boldsymbol \tau
 *     (\mathbf q - \hat{\mathbf q}).
 * \f]
 * There are different choices available for \f$\widehat{\mathbb F
 * \cdot \mathbf n}\f$. But for this hyperbolic conservation law, all
 * of them are based on the eigenvalues of the Jacobian:
 * \f$ (\partial \mathbb F(\hat {\mathbf q}) / \partial \hat
 * {\mathbf q}) \cdot \mathbf n\f$. After substituting the flux and
 * writing the equation in the index notation, we have:
 * \f[
 *   (\partial_t q_i, p_i)
 *   =
 *     - \langle \hat F_{ij} n_j, p_i\rangle
 *     - \langle \tau_{ij} q_j, p_i\rangle
 *     + \langle \tau_{ij} \hat q_j, p_i \rangle
 *     + (F_{ij}, \partial_j p_i)
 *     + L_i(p_i)
 * \f]
 * In Runge-Kutta methods, we usually denote the right side of the above
 * relation with \f$f(t_n, \{q,\hat q\})\f$. We also define the following
 * functionals for the RHS of the above equation.
 * Based on the above discussion, we now define the following operators:
 * \f[
 *   \begin{gathered}
 *     a_{00} (q_i, p_i) = (q_i, p_i); \quad
 *     f_{01}(p_i) = \langle \hat F_{ij} n_j, p_i\rangle
 *       + \langle \tau_{ij} q_j, p_i\rangle
 *       - \langle \tau_{ij} \hat {q_j}, p_i\rangle;
 *     \quad
 *     f_{02}(p_i) = (F_{ij}, \partial_j p_i); \quad
 *     f_{03}(p_i) = L_i(p_i);
 *   \end{gathered}
 * \f]
 * Next, using an explicit RKn method, we have:
 * \f[
 *   \begin{gathered}
 *     k_m = f(t_n, \{q,\hat q\}_n +
 *           \textstyle h \sum_{l=1}^{m-1} \alpha_{ml} k_l)
 *     \quad \Longrightarrow \quad
 *     A_{00} K_m
 *     =
 *     \left(-F_{01}-F_{02}-F_{03}\right)
 *       |_{Q_n+h\sum_{l=1}^{m-1} \alpha_{ml} K_l}
 *   \end{gathered}
 * \f]
 * And:
 * \f[
 *   q_{n+1} = q_n + h \sum_{i=1}^s \beta_m k_m.
 * \f]
 * While, we also want to satisfy the flux conservation condition while
 * applying the characteristic boundary condition:
 * \f[
 *   \langle
 *     \widehat {\mathbb F \cdot \mathbf n} , \boldsymbol \mu
 *   \rangle_{\partial \mathcal T\backslash \partial \Omega}
 *   +
 *   \langle
 *     \widehat {\mathbb B} , \boldsymbol \mu
 *   \rangle_{\partial \Omega} = 0.
 * \f]
 * With, \f$\widehat{\mathbb B} = {\mathbb A}^+(\hat {\mathbf q})
 * (\mathbf q - \hat {\mathbf q}) - {\mathbb A}^-
 * (\hat {\mathbf q})(\mathbf q^\infty - \hat {\mathbf q})\f$. And,
 * \f${\mathbb A}^{\pm} = (\mathbb A \pm |\mathbb A|)/2\f$,
 * where \f$\mathbb A = \partial \mathbb F /\partial \mathbf q
 * \cdot \mathbf n\f$. For example, consider the element \f$A_{ij}\f$
 * in \f$\mathbb A\f$:
 * \f[
 *   A_{ij} = \frac {F_{i1}}{q_j} n_1 + \frac {F_{i2}}{q_j} n_2.
 * \f]
 * To learn more characteristic (inflow/outflow) boundary
 * condition, refer to AdvectionDiffusion element. As before,
 * we switch to index notation for more clarity:
 * \f[
 *   \langle
 *     \hat F_{ij} n_j, \mu_i
 *   \rangle
 *   +
 *   \langle
 *     \tau_{ij} q_j, \mu_i
 *   \rangle
 *   -
 *   \langle
 *     \tau_{ij} \hat {q_j}, \mu_i
 *   \rangle
 *   +
 *   \langle
 *     {A}_{ij}^+ q_j, \mu_i
 *   \rangle
 *   -
 *   \langle
 *     {A}_{ij}^- q^\infty_{j} , \mu_i
 *   \rangle
 *   -
 *   \langle
 *     |{A}_{ij}| \hat {q_j}, \mu_i
 *   \rangle = 0.
 * \f]
 * Having \f$q_j\f$, at every stage, we want to solve for
 * \f$\hat q_j\f$ on every face. Among other methods available
 * for this purpose, we use Newton method to obtain \f$\hat q_j\f$
 * from this nonlinear equation. So, having \f$q_j\f$, we consider
 * a perturbation of this equation in the direction of
 * \f$\delta \hat q_j\f$.
 * \f[
 * \begin{aligned}
 *   &\left \langle
 *     \frac {\partial \hat F_{ik}}{\partial \hat q_j} n_k
 *       \delta \hat q_j , \mu_i
 *   \right \rangle
 *   +
 *   \left \langle
 *     \frac {\partial \tau_{ik}}{\partial \hat q_j} q_k
 *       \delta \hat {q_j} , \mu_i
 *   \right \rangle
 *   -
 *   \left \langle
 *     \frac {\partial \tau_{ik}}{\partial \hat q_j} \hat {q_k}
 *       \delta \hat q_j , \mu_i
 *   \right \rangle
 *   -
 *   \left \langle
 *     \tau_{ij} \delta \hat {q_j}, \mu_i
 *   \right \rangle
 *   \\
 *   & \quad
 *   +
 *   \left \langle
 *     \frac {\partial {A}^+_{ik}}{\partial \hat {q_j}}
 *       q_k \delta \hat {q_j} , \mu_i
 *   \right \rangle
 *   -
 *   \left \langle
 *     \frac {\partial {A}^-_{ik}}{\partial \hat {q_j}}
 *       q^{\infty}_k \delta \hat {q_j} , \mu_i
 *   \right \rangle
 *   -
 *   \left \langle
 *     \frac {\partial |{A}_{ik}|}{\partial \hat {q_j}}
 *       \hat {q_k} \delta \hat {q_j} , \mu_i
 *   \right \rangle
 *   -
 *   \left \langle
 *     |{A}_{ij}| \delta \hat {q_j} , \mu_i
 *   \right \rangle \\
 *   & \quad
 *   +
 *   \langle
 *     \hat F_{ij} n_j, \mu_i
 *   \rangle
 *   +
 *   \langle
 *     \tau_{ij} q_j, \mu_i
 *   \rangle
 *   -
 *   \langle
 *     \tau_{ij} \hat {q_j}, \mu_i
 *   \rangle
 *   +
 *   \langle
 *     {A}_{ij}^+ q_j, \mu_i
 *   \rangle
 *   -
 *   \langle
 *     {A}_{ij}^- q^\infty_{j} , \mu_i
 *   \rangle
 *   -
 *   \langle
 *     |{A}_{ij}| \hat {q_j}, \mu_i
 *   \rangle = 0
 * \end{aligned}
 * \f]
 * Next, we define the following bilinear forms and functionals:
 * \f[
 *   e_{00}(\delta \hat {q_j}, \mu i) =
 *   \left\langle
 *     \left(
 *       \frac {\partial \hat F_{ik}}{\partial \hat {q_j}} n_k
 *       + \frac{\partial \tau_{ik}}{\partial \hat {q_j}} q_k
 *       - \frac{\partial \tau_{ik}}{\partial \hat {q_j}} \hat{q_k}
 *       - \tau_{ij}
 *     \right) \delta \hat{q_j}
 *     , \mu_i
 *   \right\rangle_{\partial \mathcal T \backslash \partial \Omega}
 * \f]
 * \f[
 *   e_{01}(\delta \hat {q_j}, \mu i) =
 *   \left\langle
 *     \left(
 *       \frac{\partial {A}^+_{ik}}{\partial \hat {q_j}} q_k
 *       - \frac{\partial {A}^-_{ik}}{\partial \hat {q_j}} q_k^\infty
 *       - \frac{\partial |{A}_{ik}|}{\partial \hat {q_j}} \hat{q_k}
 *       - |A_{ij}|
 *     \right) \delta \hat{q_j}
 *     , \mu_i
 *   \right\rangle_{\partial \Omega}
 * \f]
 * \f[
 *   f_{04}(\mu_i) =
 *   \left\langle
 *     \hat F_{ij} n_j + \tau_{ij} q_j - \tau_{ij} \hat{q_j} , \mu_i
 *   \right \rangle_{\partial \mathcal T \backslash \partial \Omega}
 *   ; \quad
 *   f_{05}(\mu_i) =
 *   \left \langle
 *     {A}^+_{ij} q_j - {A}^-_{ij} q_j^\infty - |A_{ij}| \hat{q_j}
 *       , \mu_i
 *   \right \rangle_{\partial \Omega}
 * \f]
 * So, the final form of the flux conservation equations will be:
 * \f[
 *   (E_{00}+E_{01}) \delta \hat Q  = - F_{04} - F_{05}.
 * \f]
 * ### Solving equation with topography
 * One of the terms that we have not included in the above equations is the
 * topography term. The equations with topography can be written as follows
 * (assuming that \f$h(X,t) = h_0 + \zeta(X,t) - b(X)\f$):
 * \f[
 * \begin{cases}
 * \partial_{t} h + \nabla\cdot(h V) = 0,\\
 * \partial_t (hV) + \nabla (\frac 1 2 g h^2) + \nabla \cdot (hV \otimes V) + gh
 * \nabla b = 0.
 * \end{cases}
 * \f]
 * Obviously, all of the terms are the same as before, except the additional
 * \f$g h \nabla b\f$. We define \f$f_{06}(\mathbf p) = (gh \nabla b, \mathbf
 * p)\f$, and take this term to the right hand side of the element equations.
 */
template <int dim>
struct explicit_nswe : public GenericCell<dim>
{
  const double gravity = 9.81;
  using elem_basis_type = typename GenericCell<dim>::elem_basis_type;
  typedef std::unique_ptr<dealii::FEValues<dim> > FE_val_ptr;
  typedef std::unique_ptr<dealii::FEFaceValues<dim> > FEFace_val_ptr;
  typedef dealii::Tensor<1, dim + 1, double> nswe_vec;
  typedef Eigen::Matrix<double, dim + 1, dim + 1> nswe_jac;
  const double small_val = 1.e-15;

  static solver_options required_solver_options();
  static solver_type required_solver_type();
  static unsigned get_num_dofs_per_node();

  /*!
   * \brief Constructor for the explicit_nswe.
   * Since we are using factory pattern to create cells,
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
  explicit_nswe() = delete;
  explicit_nswe(const explicit_nswe &inp_cell) = delete;
  explicit_nswe(explicit_nswe &&inp_cell) noexcept;
  explicit_nswe(typename GenericCell<dim>::dealiiCell &inp_cell,
                const unsigned &id_num_,
                const unsigned &poly_order_,
                explicit_hdg_model<dim, explicit_nswe> *model_);
  ~explicit_nswe() final;
  eigen3mat A00, C01, E00, E01, F01, F02, F04, F05, F06;
  void assign_BCs(const bool &at_boundary,
                  const unsigned &i_face,
                  const dealii::Point<dim> &face_center);

  /**
   *
   */
  void assign_initial_data();

  /**
   *
   */
  void set_previous_step_results(eigen3mat *last_step_q);

  /**
   *
   */
  eigen3mat *get_previous_step_results();

  /*
  void set_previous_step_results1(ResultPacket result_);

  ResultPacket get_previous_step_results1();
  */

  /**
   *
   */
  void calculate_matrices();

  /**
   *
   */
  void calculate_stage_matrices();

  Eigen::Matrix<double, (dim + 1), dim> get_Fij(const std::vector<double> &qs);

  Eigen::Matrix<double, dim *(dim + 1), dim + 1>
  get_partial_Fik_qj(const std::vector<double> &qs);

  nswe_jac get_tauij_LF(const std::vector<double> &qs,
                        const dealii::Point<dim> normal);

  nswe_jac get_d_Fij_dqk_nj(const std::vector<double> &qhats,
                            const dealii::Point<dim> &normal);

  nswe_jac get_partial_tauij_qhk_qj_LF(const std::vector<double> &qs,
                                       const std::vector<double> &qhats,
                                       const dealii::Point<dim> &normal);

  /**
   * This function returns the right eigenvectors of the
   * Jacobian matrix
   * (\f$\frac{\partial F_{ij}}{\partial \hat{q_k}} n_j\f$).
   */
  nswe_jac get_XR(const std::vector<double> &qhats,
                  const dealii::Point<dim> &normal);

  /**
   * This function returns the left eigenvectors of the
   * Jacobian matrix
   * (\f$\frac{\partial F_{ij}}{\partial \hat{q_k}} n_j\f$).
   */
  nswe_jac get_XL(const std::vector<double> &qhats,
                  const dealii::Point<dim> &normal);

  /**
   * This function returns the eigenvalues of the
   * Jacobian matrix
   * (\f$\frac{\partial F_{ij}}{\partial \hat{q_k}} n_j\f$).
   */
  nswe_jac get_Dn(const std::vector<double> &qhats,
                  const dealii::Point<dim> &normal);

  /**
   * This function returns the absolute value of the
   * eigenvalues of the
   * Jacobian matrix
   * (\f$\frac{\partial F_{ij}}{\partial \hat{q_k}} n_j\f$).
   */
  nswe_jac get_absDn(const std::vector<double> &qhats,
                     const dealii::Point<dim> &normal);

  nswe_jac get_Aij_plus(const std::vector<double> &qhats,
                        const dealii::Point<dim> &normal);

  nswe_jac get_Aij_mnus(const std::vector<double> &qhats,
                        const dealii::Point<dim> &normal);

  nswe_jac get_Aij_absl(const std::vector<double> &qhats,
                        const dealii::Point<dim> &normal);

  template <typename T>
  Eigen::Matrix<double, dim + 1, 1> get_solid_wall_BB(
    const T &qs, const T &qhats, const dealii::Point<dim> &normal);

  template <typename T>
  std::vector<nswe_jac> get_dRik_dqhj(const T &qhats,
                                      const dealii::Point<dim> &normal);

  template <typename T>
  std::vector<nswe_jac> get_dLik_dqhj(const T &qhats,
                                      const dealii::Point<dim> &normal);

  template <typename T>
  std::vector<nswe_jac> get_dDik_dqhj(const T &qhats,
                                      const dealii::Point<dim> &normal);

  template <typename T>
  std::vector<nswe_jac> get_dAbsDik_dqhj(const T &qhats,
                                         const dealii::Point<dim> &normal);

  nswe_jac get_dAik_dqhj_qk_plus(const std::vector<double> &qs,
                                 const std::vector<double> &qhats,
                                 const dealii::Point<dim> &normal);

  nswe_jac get_dAik_dqhj_qk_mnus(const Eigen::Matrix<double, dim + 1, 1> &qinfs,
                                 const std::vector<double> &qhats,
                                 const dealii::Point<dim> &normal);

  nswe_jac get_dAik_dqhj_qk_absl(const std::vector<double> &qs,
                                 const std::vector<double> &qhats,
                                 const dealii::Point<dim> &normal);

  void assemble_globals(const solver_update_keys &keys_);

  template <typename T>
  double compute_internal_dofs(const double *const local_uhat,
                               eigen3mat &solved_u_vec,
                               eigen3mat &solved_q_vec,
                               const poly_space_basis<T, dim> &output_basis);

  double get_trace_increment_norm(const double *const local_uhat);

  void internal_vars_errors(const eigen3mat &u_vec,
                            const eigen3mat &q_vec,
                            double &u_error,
                            double &q_error);

  void ready_for_next_stage();

  void ready_for_next_time_step();

  explicit_hdg_model<dim, explicit_nswe> *model;
  explicit_RKn<4, original_RK> *time_integrator;

  static explicit_nswe_qis_func_class<dim, nswe_vec> explicit_nswe_qis_func;
  static explicit_nswe_zero_func_class<dim, nswe_vec> explicit_nswe_zero_func;
  static explicit_nswe_L_func_class<dim, nswe_vec> explicit_nswe_L_func;
  static explicit_nswe_grad_b_func_class<dim, dealii::Tensor<1, dim> >
    explicit_nswe_grad_b_func;

  eigen3mat last_step_q;
  eigen3mat last_iter_qhat;
  eigen3mat last_stage_q;

  std::vector<eigen3mat> ki_s;
};

#include "explicit_nswe.cpp"

#endif
