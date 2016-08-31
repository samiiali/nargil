#include "cell_class.hpp"

#ifndef NL_SWE_HPP
#define NL_SWE_HPP

/*!
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct nswe_zero_func_class : public TimeFunction<in_point_dim, output_type>
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
struct nswe_qis_func_class : public TimeFunction<in_point_dim, output_type>
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
    qs[0] = 7 + sin(x[0] - t);
    qs[1] = 5 + sin(x[0] + 3 * t);
    qs[2] = -3 - cos(x[1] + 4 * t);
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
    if (t0 <= 3.0)
      qs[0] = 2 + cos(5 * M_PI * t0);
    else
      qs[0] = 1.;

    qs[1] = 0.;
    qs[2] = 0.;
    */
    // End of second example in paper 3
    // Narrowing channel in paper 3
    if (t < 1.e-8)
    {
      qs[0] = 3.;
      qs[1] = 0.01;
      qs[2] = 0.;
    }
    if (x[0] < -2. + 1.e-6)
    {
      qs[0] = 3.0;
      qs[1] = 0.5;
      qs[2] = 0;
    }
    if (x[0] > 2 - 1.e-6)
    {
      qs[0] = 3.0;
      qs[1] = 0.1;
      qs[2] = 0;
    }
    // End of narrowing channel in paper 3
    return qs;
  }
};

/*!
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct nswe_L_func_class : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double & = 0) const final
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
    L[0] = -cos(t - x0) + cos(3 * t + x0) + sin(4 * t + y0);
    L[1] = 3 * cos(3 * t + x0) + cos(t - x0) * (7 - sin(t - x0)) -
           (2 * cos(3 * t + x0) * (5 + sin(3 * t + x0))) / (-7 + sin(t - x0)) -
           (cos(t - x0) * pow(5 + sin(3 * t + x0), 2)) / pow(-7 + sin(t - x0),
    2) +
           ((5 + sin(3 * t + x0)) * sin(4 * t + y0)) / (7 - sin(t - x0));
    L[2] = (cos(3 * t + x0) * (3 + cos(4 * t + y0))) / (-7 + sin(t - x0)) +
           (cos(t - x0) * (3 + cos(4 * t + y0)) * (5 + sin(3 * t + x0))) /
             pow(-7 + sin(t - x0), 2) +
           4 * sin(4 * t + y0) +
           (2 * (3 + cos(4 * t + y0)) * sin(4 * t + y0)) / (-7 + sin(t - x0));
    */
    // End of Example 1
    // First example of paper 3
    /*
    double t0 = t;
    L[0] = -3 * exp(-sin(3 * t0) + sin(3 * x0) * sin(3 * y0)) * cos(3 * t0) +
           cos(4 * t0 + y0) + sin(4 * t0 - x0);
    L[1] =
      (-3 * exp(3 * sin(3 * t0) + sin(3 * x0) * sin(3 * y0)) *
         pow(cos(4 * t0 - x0), 2) * cos(3 * x0) * sin(3 * y0) +
       (2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0))) *
         (exp(3 * sin(3 * t0)) * sin(8 * t0 - 2 * x0) -
          (2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0))) *
            (4 * exp(2 * sin(3 * t0)) * sin(4 * t0 - x0) -
             3 * exp(sin(3 * x0) * sin(3 * y0)) *
               (2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0))) *
               cos(3 * x0) * sin(3 * y0))) +
       exp(3 * sin(3 * t0)) * cos(4 * t0 - x0) *
         ((2 * exp(sin(3 * t0)) + exp(sin(3 * x0) * sin(3 * y0))) * cos(4 * t0 +
    y0) -
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
    L[0] = L[1] = L[2] = x0 - x0 + y0 - y0;
    // End of example 2 of paper 3
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
 * Solving Eq. (1) by an implicit method
 * -------------------------------------
 *
 * Now, consider a nonlinear problem (such as the above euqaiotn) which is
 * written in the form of \f$\mathcal G(\mathbf q) = 0\f$. This equation can
 * be solved using Newton-Raphson iterative method. Let \f$\bar {\mathbf q}\f$
 * denote the value of \f$\mathbf q\f$ at the current iteration. Also, let
 * \f$\delta \mathbf q\f$ be its increment due to the next Newton iteration.
 * We have:
 * \f[
 *   \left.\frac {\partial \mathcal G}{\partial \mathbf q}\right|
 *                                            _{\bar {\mathbf q}}
 *     \delta \mathbf q + \mathcal G (\bar {\mathbf q}) = 0. \tag{2}
 * \f]
 *
 * ### Equations inside the elements:
 *
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
 * {\mathbf q}) \cdot \mathbf n\f$.
 * To perform a Newton iteration on this equation, we first consider
 * the equation in the index notation:
 * \f[
 *   (\partial_t q_i, p_i)
 *     + \langle \hat F_{ij} n_j, p_i\rangle
 *     + \langle \tau_{ij} q_j, p_i\rangle
 *     - \langle \tau_{ij} \hat q_j, p_i \rangle
 *     - (F_{ij}, \partial_j p_i)
 *     - L_i(p_i) = 0
 * \f]
 * In the above relation, we have denoted \f$\mathbb F (\hat {\mathbf q})\f$
 * with \f$\hat F_{ij}\f$, and \f$\mathbb F ({\mathbf q})\f$ with
 * \f$F_{ij}\f$.
 *
 * Using the backward difference formula, for the time derivative, we will
 * have (refer to AdvectionDiffusion element for more details):
 * \f[
 *   (q_i, p_i)
 *     + (\tilde {q_i}, p_i)
 *     + \beta \Delta t \langle \hat F_{ij} n_j, p_i\rangle
 *     + \beta \Delta t \langle \tau_{ij} q_j, p_i\rangle
 *     - \beta \Delta t \langle \tau_{ij} \hat q_j, p_i \rangle
 *     - \beta \Delta t (F_{ij}, \partial_j p_i)
 *     - \beta \Delta t L_i(p_i) = 0
 * \f]
 * Here, \f$\tilde q = \sum \alpha^k q_i^k\f$, with \f$k\f$ denoting the
 * previous time steps.
 *
 * Now, for each Newton iteration we perturb this equation in two directions
 * \f$\delta \mathbf q\f$ and \f$\delta \hat {\mathbf q}\f$:
 * \f[
 * \begin{aligned}
 *   (\delta q_i, p_i)
 *     &+ \beta \Delta t \left \langle \tau_{ij} \delta q_j, p_i
 *                      \right \rangle
 *     - \beta \Delta t \left( \frac {\partial F_{ij}}
 *                              {\partial q_k} \delta q_k, \partial_j p_i
 *                      \right)\\
 *     &+\beta \Delta t \left\langle\frac {\partial \hat F_{ij}}
 *                              {\partial\hat{q_k}} n_j \delta \hat {q_k},p_i
 *                       \right\rangle
 *     + \beta \Delta t \left \langle
 *                        \frac {\partial \tau_{ij}}{\partial \hat{q_k}}
 *                        q_j \delta \hat {q_k}, p_i
 *                      \right \rangle
 *     - \beta \Delta t \left \langle
 *                        \frac {\partial \tau_{ij}}{\partial \hat{q_k}}
 *                        \hat {q_j} \delta \hat {q_k}, p_i
 *                      \right \rangle
 *     - \beta \Delta t \left \langle \tau_{ij} \delta \hat{q_j}, p_i
 *                      \right \rangle \\
 *     &+ (q_i, p_i) + (\tilde{q_i},p_i)
 *     + \beta \Delta t \langle \hat F_{ij} n_j, p_i\rangle
 *     + \beta \Delta t \langle \tau_{ij} q_j, p_i\rangle
 *     - \beta \Delta t \langle \tau_{ij} \hat {q_j}, p_i\rangle
 *     - \beta \Delta t (F_{ij}, \partial_j p_i)
 *     - \beta \Delta t L_i(p_i)
 *     = 0
 * \end{aligned}
 * \f]
 * For starters, we will use Lax-Friedrichs flux. In any case, we have to
 * drive an expression for \f$(\partial F_{ij}/ \partial q_k)  n_j\f$ and
 * \f$(\partial F_{ij}/ \partial q_k) \delta q_k \f$:
 * \f[
 *   \begin{aligned}
 *   \frac {\partial \mathbb F}{\partial \mathbf q} \cdot \mathbf n
 *   =
 *   \left[
 *   \begin{array}{ccc}
 *     0 & 1 & 0 \\
 *     -\frac{(hV_1)^2}{h^2}+h & 2 \frac {hV_1}{h} & 0 \\
 *     -\frac{(hV_1)(hV_2)}{h^2} & \frac {hV_2}{h} & \frac {hV_1}{h} \\
 *     \hline
 *     0 & 0 & 1 \\
 *     -\frac{(hV_1)(hV_2)}{h^2} & \frac{hV_2}{h} & \frac{hV_1}{h} \\
 *     -\frac{(hV_2)^2}{h^2}+h & 0 & 2\frac{hV_2}{h}
 *   \end{array}
 *   \right]
 *   \cdot
 *   \left(
 *   \begin{array}{c}
 *     n_1 \\ \hline n_2
 *   \end{array}
 *   \right)
 *   =
 *   \left[
 *   \begin{array}{ccc}
 *     0 & 1 & 0  \\
 *     -\frac{q_2^2}{q_1^2}+q_1 & 2 \frac {q_2}{q_1} & 0 \\
 *     -\frac{q_2 q_3}{q_1^2} & \frac {q_3}{q_1} & \frac {q_2}{q_1} \\
 *     \hline
 *     0 & 0 & 1 \\
 *     -\frac{q_2q_3}{q_1^2} & \frac{q_3}{q_1} & \frac{q_2}{q_1} \\
 *     -\frac{q_3^2}{q_1^2}+q_1 & 0 & 2\frac{q_3}{q_1}
 *   \end{array}
 *   \right]
 *   \cdot
 *   \left(
 *   \begin{array}{c}
 *     n_1 \\ \hline n_2
 *   \end{array}
 *   \right)
 *   \end{aligned}
 *   =
 *   \begin{bmatrix}
 *     F_{11,1} & F_{11,2} & F_{11,3} \\
 *     F_{21,1} & F_{21,2} & F_{21,3} \\
 *     F_{31,1} & F_{31,2} & F_{31,3} \\
 *                \hline
 *     F_{12,1} & F_{12,2} & F_{12,3} \\
 *     F_{22,1} & F_{22,2} & F_{22,3} \\
 *     F_{32,1} & F_{32,2} & F_{32,3}
 *   \end{bmatrix}
 *   \cdot
 *   \begin{pmatrix}
 *     n_1 \\ \hline  n_2
 *   \end{pmatrix}
 * \f]
 * \f[
 *   \frac {\partial \mathbb F}{\partial \mathbf q} \delta \mathbf q
 *   =
 *   \begin{bmatrix}
 *     F_{11,1} & F_{11,2} & F_{11,3} \\
 *     F_{21,1} & F_{21,2} & F_{21,3} \\
 *     F_{31,1} & F_{31,2} & F_{31,3} \\
 *                \hline
 *     F_{12,1} & F_{12,2} & F_{12,3} \\
 *     F_{22,1} & F_{22,2} & F_{22,3} \\
 *     F_{32,1} & F_{32,2} & F_{32,3}
 *   \end{bmatrix}
 *   \begin{pmatrix}
 *     \delta q_1 \\ \delta q_2 \\ \delta q_3
 *   \end{pmatrix}
 * \f]
 * Now, we calculate the eigenvalues of
 * \f$ (\partial \mathbb F/\partial \mathbf q) \cdot \mathbf n\f$:
 * \f[
 *   \lambda_1 = \frac {q_2}{q_1}n_1 + \frac {q_3}{q_1}n_2 - \sqrt{q_1}; \quad
 *   \lambda_2 = \frac {q_2}{q_1}n_1 + \frac {q_3}{q_1}n_2; \quad
 *   \lambda_3 = \frac {q_2}{q_1}n_1 + \frac {q_3}{q_1}n_2 + \sqrt{q_1};
 * \f]
 * Or, in terms of \f$h, V_1, V_2\f$:
 * \f[
 *   \lambda_1 = V_1 n_1 + V_2 n_2 - \sqrt{h} = V_n - \sqrt{h}; \quad
 *   \lambda_2 = V_1 n_1 + V_2 n_2 = V_n; \quad
 *   \lambda_3 = V_1 n_1 + V_2 n_2 + \sqrt{h} = V_n + \sqrt{h};
 * \f]
 * And, due to Lax-Friedrichs, \f$\tau_{ij} = \lambda_{\text{max}} \delta_{ij}=
 * (\sqrt{\hat h} + |\widehat{V_n}|) \delta_{ij}\f$.
 *
 * For \f$V_n > 0\f$, one can derive the following for
 * \f$ ({\partial \tau_{ij}}/{\partial \hat{q_k}})\hat {q_j}\f$ and
 * \f$ ({\partial \tau_{ij}}/{\partial \hat{q_k}}) {q_j}\f$:
 * \f[
 * \frac {\partial \overline{\overline {\boldsymbol \tau}}}
 *       {\partial \hat {\mathbf q}} \cdot \hat{\mathbf q}
 * =
 * \begin{bmatrix}
 *   \frac{-2 n_1 \hat{q}_2-2 n_2 \hat{q}_3+\hat{q}_1^{3/2}}{2
 *   \hat{q}_1}
 *   & n_1 & n_2\\
 *   \frac{\hat{q}_2 \left(\hat{q}_1^{3/2}-2 \left(n_1
 *   \hat{q}_2+n_2 \hat{q}_3\right)\right)}{2 \hat{q}_1^2}
 *   & \frac{n_1 \hat{q}_2}{\hat{q}_1} & \frac{n_2 \hat{q}_2}{\hat{q}_1}\\
 *   \frac{\hat{q}_3 \left(\hat{q}_1^{3/2}-2 \left(n_1 \hat{q}_2+n_2
 *   \hat{q}_3\right)\right)}{2 \hat{q}_1^2}
 *   & \frac{n_1 \hat{q}_3}{\hat{q}_1} &\frac{n_2\hat{q}_3}{\hat{q}_1}
 * \end{bmatrix} ; \quad
 * \frac {\partial \overline{\overline {\boldsymbol \tau}}}
 *       {\partial \hat {\mathbf q}} \cdot {\mathbf q}
 * =
 * \begin{bmatrix}
 *   \frac{q_1 \left(\hat{q}_1^{3/2}-2 \left(n_1 \hat{q}_2+n_2
 *   \hat{q}_3\right)\right)}{2 \hat{q}_1^2}
 *   & \frac{n_1 q_1}{\hat{q}_1} & \frac{n_2 q_1}{\hat{q}_1}\\
 *   \frac{q_2 \left(\hat{q}_1^{3/2}-2
 *   \left(n_1 \hat{q}_2+n_2 \hat{q}_3\right)\right)}{2 \hat{q}_1^2}
 *   & \frac{n_1 q_2}{\hat{q}_1} & \frac{n_2 q_2}{\hat{q}_1}\\
 *   \frac{q_3
 *   \left(\hat{q}_1^{3/2}-2 \left(n_1 \hat{q}_2+n_2
 *   \hat{q}_3\right)\right)}{2 \hat{q}_1^2}
 *   & \frac{n_1 q_3}{\hat{q}_1} & \frac{n_2 q_3}{\hat{q}_1}
 * \end{bmatrix}
 * \f]
 *
 * For \f$V_n < 0\f$, one can derive the following for
 * \f$ ({\partial \tau_{ij}}/{\partial \hat{q_k}})\hat {q_j}\f$ and
 * \f$ ({\partial \tau_{ij}}/{\partial \hat{q_k}}) {q_j}\f$:
 * \f[
 * \frac {\partial \overline{\overline {\boldsymbol \tau}}}
 *       {\partial \hat {\mathbf q}} \cdot \hat{\mathbf q}
 * =
 * \begin{bmatrix}
 *   \frac{2 n_1 \hat{q}_2+2 n_2 \hat{q}_3+\hat{q}_1^{3/2}}{2
 *   \hat{q}_1}
 *   & -n_1 & -n_2\\
 *   \frac{\hat{q}_2 \left(\hat{q}_1^{3/2}+2 \left(n_1
 *   \hat{q}_2+n_2 \hat{q}_3\right)\right)}{2 \hat{q}_1^2}
 *   & -\frac{n_1 \hat{q}_2}{\hat{q}_1} & -\frac{n_2 \hat{q}_2}{\hat{q}_1}\\
 *   \frac{\hat{q}_3 \left(\hat{q}_1^{3/2}+2 \left(n_1 \hat{q}_2+n_2
 *   \hat{q}_3\right)\right)}{2 \hat{q}_1^2}
 *   & -\frac{n_1 \hat{q}_3}{\hat{q}_1} &-\frac{n_2\hat{q}_3}{\hat{q}_1}
 * \end{bmatrix} ; \quad
 * \frac {\partial \overline{\overline {\boldsymbol \tau}}}
 *       {\partial \hat {\mathbf q}} \cdot {\mathbf q}
 * =
 * \begin{bmatrix}
 *   \frac{q_1 \left(2 n_1 \hat{q}_2+2 n_2
 *   \hat{q}_3+\hat{q}_1^{3/2}\right)}{2 \hat{q}_1^2}
 *   & -\frac{n_1 q_1}{\hat{q}_1} & -\frac{n_2 q_1}{\hat{q}_1}\\
 *   \frac{q_2 \left(2 n_1
 *   \hat{q}_2+2 n_2 \hat{q}_3+\hat{q}_1^{3/2}\right)}{2
 *   \hat{q}_1^2}
 *   & -\frac{n_1 q_2}{\hat{q}_1} & -\frac{n_2 q_2}{\hat{q}_1}\\
 *   \frac{q_3 \left(2 n_1 \hat{q}_2+2 n_2
 *   \hat{q}_3+\hat{q}_1^{3/2}\right)}{2 \hat{q}_1^2}
 *   & -\frac{n_1 q_3}{\hat{q}_1} & -\frac{n_2 q_3}{\hat{q}_1}\\
 * \end{bmatrix}
 * \f]
 * Based on the above discussion, we now define the following operators:
 * \f[
 *   \begin{gathered}
 *     a_{00}(\delta q_i, p_i) = (\delta q_i, p_i); \quad
 *     a_{01}(\delta q_j, p_i) =
 *       \left(
 *         \frac {\partial F_{ik}}{\partial q_j} \delta q_j ,p_{i,k}
 *       \right); \quad
 *     d_{01}(\delta q_j, p_i) =
 *       \left \langle
 *        \tau_{ij} \delta q_k, p_i
 *       \right \rangle; \\
 *     c_{01}(\delta \hat {q_k}, p_i) =
 *       \left \langle
 *         \left(
 *           \frac {\partial \hat F_{ij}}{\partial\hat{q_k}}  n_j
 *           + \frac {\partial \tau_{ij}}{\partial \hat{q_k}} q_j
 *           - \frac {\partial \tau_{ij}}{\partial \hat{q_k}} \hat {q_j}
 *           - \tau_{ik}
 *         \right)
 *         \delta \hat{q_k} , p_i
 *       \right \rangle; \\
 *     f_{00}(p_i) = (q_i, p_i) + (\tilde{q_i},p_i) ; \quad
 *     f_{01}(p_i) = \langle \hat F_{ij} n_j, p_i\rangle
 *       + \langle \tau_{ij} q_j, p_i\rangle
 *       - \langle \tau_{ij} \hat {q_j}, p_i\rangle; \\
 *     f_{02}(p_i) = (F_{ij}, \partial_j p_i); \quad
 *     f_{03}(p_i) = L_i(p_i);
 *   \end{gathered}
 * \f]
 * Which finally results in:
 * \f[
 *     [A_{00}] \delta Q - \beta \Delta t [A_{01}] \delta Q
 *      + \beta \Delta t [D_{01}] \delta Q
 *      + \beta \Delta t [C_{01}] \delta \hat Q
 *      + F_{00} + \beta \Delta t (F_{01} - F_{02} - F_{03}) = 0
 * \f]
 * No need to mention that \f$a_{00}, f_{00}, and f_{03}\f$, all
 * have the same matrix form, and only \f$A_{00}\f$ is
 * implemented as a matrix in the code.
 *
 * ### Flux conservation condition:
 *
 * We want to satisfy the flux conservation condition, while applying
 * the characteristic boundary condition:
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
 * In the above relation, the first three terms should be computed
 * on \f$\partial \mathcal T \backslash \partial \Omega\f$,
 * and the last three terms should be computed on
 * \f$\partial \Omega\f$. Similar to the equations in the element
 * interior, we now perturb the above equation in two directions,
 * i.e. \f$\delta q\f$ and \f$\delta \hat q\f$:
 * \f[
 * \begin{aligned}
 *   \left \langle
 *     \frac {\partial \hat F_{ik}}{\partial \hat q_j} n_k
 *       \delta \hat q_j , \mu_i
 *   \right \rangle
 *   &+
 *   \left \langle
 *     \frac {\partial \tau_{ik}}{\partial \hat q_j} q_k
 *       \delta \hat {q_j} , \mu_i
 *   \right \rangle
 *   +
 *   \left \langle
 *     \tau_{ij} \delta q_j, \mu_i
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
 *   &+
 *   \left \langle
 *     \frac {\partial {A}^+_{ik}}{\partial \hat {q_j}}
 *       q_k \delta \hat {q_j} , \mu_i
 *   \right \rangle
 *   +
 *   \left \langle
 *     {A}_{ij}^+ \delta q_j , \mu_i
 *   \right \rangle
 *   -
 *   \left \langle
 *     \frac {\partial {A}^-_{ik}}{\partial \hat {q_j}}
 *       q^{\infty}_k \delta \hat {q_j} , \mu_i
 *   \right \rangle \\
 *   &-
 *   \left \langle
 *     \frac {\partial |{A}_{ik}|}{\partial \hat {q_j}}
 *       \hat {q_k} \delta \hat {q_j} , \mu_i
 *   \right \rangle
 *   -
 *   \left \langle
 *     |{A}_{ij}| \delta \hat {q_j} , \mu_i
 *   \right \rangle \\
 *   &+
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
 *   c_{02}^T(\delta q_j, \mu_i) =
 *   \left \langle
 *     \tau_{ij} \delta q_j , \mu_i
 *   \right \rangle_{\partial \mathcal T \backslash \partial \Omega}
 *   ; \quad
 *   c_{03}^T(\delta q_j, \mu_i) =
 *   \left \langle
 *     {A}^+_{ij} \delta q_j , \mu_i
 *   \right \rangle_{\partial \Omega};
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
 *   (C_{02}^T + C_{03}^T) \delta Q +
 *   (E_{00}+E_{01}) \delta \hat Q + F_{04} + F_{05} = 0.
 * \f]
 *
 * #### Additional note on the solid wall boundary condition
 * In the case of a solid wall, we want to get the value of \f$\hat h\f$ from
 * \f$h\f$ and set \f$\mathbf V \cdot \mathbf n = 0\f$. We follow the common
 * practice and define \f$\widehat {\mathbb B}\f$ in the above relation as:
 * \f[
 * \widehat {\mathbb B} =
 * \begin{Bmatrix}
 * h \\ h\mathbf u - (h \mathbf u \cdot \mathbf n) \mathbf n
 * \end{Bmatrix}
 * -
 * \begin{Bmatrix}
 * \hat h \\ \widehat{h \mathbf u}
 * \end{Bmatrix}
 * =
 * \begin{Bmatrix}
 * q_1 - \hat {q}_1 \\
 * q_2 n_1^2 + q_3 n_1 n_2 - \hat q_2 \\
 * q_2 n_1 n_2 + q_3 n_2^2 - \hat q_3
 * \end{Bmatrix}
 * \f]
 * As a result,
 * \f[
 * \frac{\partial B_i}{\partial q_j} =
 * \begin{bmatrix}
 * 1 & 0 & 0 \\
 * 0 & n_1^2 & n_1 n_2 \\
 * 0 & n_1 n_2 & n_2^2
 * \end{bmatrix} ; \quad
 * \frac{\partial B_i}{\partial \hat q_j} =
 * \begin{bmatrix}
 * -1 & 0 & 0 \\
 * 0 & -1 & 0 \\
 * 0 & 0 & -1
 * \end{bmatrix} .
 * \f]
 *
 * #### A small note on the nonconservative term
 * The nonconservative term actually appears as
 * \f$-\frac {\beta}{\varepsilon} h \nabla b\f$ in the momentum
 * conservation equation. This term does not change the flux
 * conservation conditions and we only need to modify the inner
 * element equations.
 */
template <int dim>
struct NSWE : public GenericCell<dim>
{
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
   * \brief Constructor for the NSWE.
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
  NSWE() = delete;
  NSWE(const NSWE &inp_cell) = delete;
  NSWE(NSWE &&inp_cell) noexcept;
  NSWE(typename GenericCell<dim>::dealiiCell &inp_cell,
       const unsigned &id_num_,
       const unsigned &poly_order_,
       hdg_model<dim, NSWE> *model_);
  ~NSWE() final;
  eigen3mat A00, A01, C01, C02T, C03T, D01, E00, E01, F01, F02, F04, F05;
  void assign_BCs(const bool &at_boundary,
                  const unsigned &i_face,
                  const dealii::Point<dim> &face_center);

  void assign_initial_data();

  /**
   *
   */
  void calculate_matrices();

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

  void internal_vars_errors(const eigen3mat &u_vec,
                            const eigen3mat &q_vec,
                            double &u_error,
                            double &q_error);

  void ready_for_next_iteration();

  void ready_for_next_time_step();

  hdg_model<dim, NSWE> *model;
  BDFIntegrator *time_integrator;

  static nswe_qis_func_class<dim, nswe_vec> nswe_qis_func;
  static nswe_zero_func_class<dim, nswe_vec> nswe_zero_func;
  static nswe_L_func_class<dim, nswe_vec> nswe_L_func;

  eigen3mat last_iter_q;
  eigen3mat last_iter_qhat;
  eigen3mat last_dq;
  eigen3mat last_dqhat;
  std::vector<eigen3mat> qi_s;
};

#include "nswe.cpp"

#endif
