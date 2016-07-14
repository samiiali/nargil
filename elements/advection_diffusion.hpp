#include "cell_class.hpp"

#ifndef ADVECTION_DIFFUSION
#define ADVECTION_DIFFUSION

/*!
 * \page advec_page Numerical Examples for Pure Advection
 *
 * Consider the pure advection equation equation as:
 * \f[u_t + \nabla \cdot (\mathbf c u) = f_1\f]
 * We want to solve this equation in \f$\Omega = (-1,1)^2\f$. Let:
 * \f[
 *   \begin{aligned}
 *     u = \sin\pi(x-t)\cos\pi(y-t); \quad
 *     \mathbf c = x \mathbf i - y \mathbf j
 *   \end{aligned}
 * \f]
 * Hence, \f$ f_1 = \pi  y \sin \pi  (x-t) \sin \pi (y-t)+
 * \pi \sin \pi (x-t) \sin \pi (y-t)+
 * \pi x \cos \pi  (x-t) \cos \pi (y-t)-\pi \cos \pi (x-t)
 * \cos \pi (y-t) \f$
 */

/*!
 * \brief Gives the value of the advection velocity \f$\mathbf c\f$
 * at a given point.
 */
template <int in_point_dim, typename output_type>
struct adv_c_vec_func_class : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double & = 0) const final
  {
    eigen3mat c_vec(2, 1);
    /* Example 1: */
    c_vec(0, 0) = -10.0 * x[1];
    c_vec(1, 0) = 10.0 * x[0];

    /*
    if (x[0] > 0.)
    {
      c_vec(0, 0) = -1.;
      c_vec(1, 0) = -1.;
    }
    else
    {
      c_vec(0, 0) = 1.;
      c_vec(1, 0) = 1.;
    }
    */

    /* End of Example 1 */
    return c_vec;
  }
};

/*!
 * \brief
 * Gives the value pf the advection unknown \f$u\f$.
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct adv_u_func_class : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    double u;
    /* Example 1: */
    u = sin(M_PI * (x[0] - t)) * cos(M_PI * (x[1] - t));
    u = 0;
    if ((x[0] - 0.5) * (x[0] - 0.5) + x[1] * x[1] <= 0.25)
      u = 1 + cos(4 * M_PI * ((x[0] - 0.5) * (x[0] - 0.5) + x[1] * x[1]));
    //    u = sin(M_PI * x[0]) * cos(M_PI * x[1]);
    /* End of Example 1 */
    return u;
  }
};

/*!
 * \brief
 * Gives the value pf the advection unknown \f$u\f$.
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct adv_uinf_func_class : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &,
                            const dealii::Point<in_point_dim> &,
                            const double & = 0) const final
  {
    double u;
    /* Example 1: */
    u = 0.;
    //    u = sin(M_PI * x[0]) * cos(M_PI * x[1]);
    /* End of Example 1 */
    return u;
  }
};

/*!
 * \brief
 * Gives the value pf \f$f_1\f$.
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct adv_f1_func_class : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    double f_1;
    /* Example 1: */
    f_1 = M_PI * x[1] * sin(M_PI * (x[0] - t)) * sin(M_PI * (x[1] - t)) +
          M_PI * sin(M_PI * (x[0] - t)) * sin(M_PI * (x[1] - t)) +
          M_PI * x[0] * cos(M_PI * (x[0] - t)) * cos(M_PI * (x[1] - t)) -
          M_PI * cos(M_PI * (x[0] - t)) * cos(M_PI * (x[1] - t));
    f_1 = -2. * M_PI * cos(M_PI * (x[0] - t)) * cos(M_PI * (x[1] - t)) +
          2. * M_PI * sin(M_PI * (x[0] - t)) * sin(M_PI * (x[1] - t));
    f_1 = 0.;
    //    f_1 = M_PI * cos(M_PI * x[0]) * cos(M_PI * x[1]) +
    //          -M_PI * sin(M_PI * x[0]) * sin(M_PI * x[1]);
    /* End of Example 1 */
    return f_1;
  }
};

/*!
 * \ingroup cells
 * \brief A hybrid DG element for solving advection equation.
 *
 * Formulation
 * -----------
 *
 * This element was mainly written to test the explicit and implicit
 * time integrators. Consider the pure advection equation as:
 * \f[u_t + \nabla \cdot (\mathbf c u) = f_1\f]
 * We want to satisfy the following variational formulation:
 * \f[
 * (u_t, v) + \langle \widehat{\mathbf c u} \cdot \mathbf n , v \rangle
 * - (\mathbf c u , \nabla v) = f_1(v).
 * \f]
 * Let us define \f$\widehat {\mathbf c u} \cdot \mathbf n =
 * \mathbf c \hat u \cdot \mathbf n + \tau (u - \hat u) \f$.
 * By substituting \f$\widehat{\mathbf c u} \cdot \mathbf n\f$, back
 * to the above equation, we get:
 * \f[
 *   (u_t, v)
 *   + \langle (\mathbf c \cdot \mathbf n - \tau) \hat u , v \rangle
 *   + \langle \tau u , v \rangle - (\mathbf c u , \nabla v)
 *   = f_1(v).
 * \f]
 * We define the following matrices:
 * \f[
 *   a_2 (u_t,v) = (u_t,v);
 *   \quad
 *   b_3(u,v) = (\mathbf c u , \nabla v);
 *   \quad
 *   c_3(\hat u, v) =
 *     \langle (\mathbf c \cdot \mathbf n - \tau)\hat u , v \rangle;
 *   \quad
 *   d_3(u,v) = \langle \tau u , v \rangle.
 * \f]
 * Hence, we want to form the following matrix equation:
 * \f[
 *   A_2 U_t + (D_3-B_3) U + C_3 \hat U = F_1.
 * \f]
 * We also want to satisfy the flux conservation condition, which
 * reads as:
 * \f$\langle \widehat{\mathbf c u}\cdot \mathbf n ,
 * \mu \rangle_{\partial \mathcal T \backslash \partial \Omega}
 * + \langle \widehat B , \mu \rangle_{\partial \Omega} =0 \f$.
 * \f$\widehat B\f$ is defined as:
 * \f$\widehat B = A^+_n(\hat u)(u-\hat u) - {A}_n^-(\hat u)(u_{\infty}
 * - \hat u)\f$. With \f$A_n^{\pm} = (A_n \pm |A_n|)/2\f$. And
 * \f$A_n = \partial F(\hat u) / \partial \hat u \cdot \mathbf n.\f$ This
 * means at the inflow boundary where \f$\mathbf c \cdot \mathbf n < 0\f$,
 * \f$A_n^+\f$ is zero, and \f${A}_n^- = \mathbf c\cdot \mathbf n\f$. Here,
 * \f$\langle \widehat B,\mu \rangle_{\partial \Omega}=0\f$ results in:
 * \f$\hat u = u^{\infty}\f$.
 * At the outflow boundary, where \f$\mathbf c \cdot \mathbf n > 0\f$,
 * \f${A}_n^-\f$ is zero, and \f$A_n^{+} = \mathbf c\cdot \mathbf n\f$.
 * This results in \f$\hat u = u\f$, at the outflow bounday.
 * This conservation condition can be extended as:
 * \f[
 *   \langle (\mathbf c \cdot \mathbf n - \tau) \hat u , \mu \rangle
 *     _{\partial \mathcal T \backslash \partial \Omega}
 *   + \langle \tau u , \mu \rangle
 *     _{\partial\mathcal T \backslash \partial \Omega}
 *   + \langle {A}_n^+ u ,\mu \rangle
 *     _{\partial \Omega}
 *   - \langle {A}_n^- u^\infty ,\mu \rangle
 *     _{\partial \Omega}
 *   - \langle (A_n^+ - {A}_n^-) \hat u ,\mu \rangle
 *     _{\partial \Omega}
 *   = 0.
 * \f]
 * No need to mention that:
 * \f$\langle (A_n^+ - {A}_n^-) \hat u ,\mu \rangle_{\partial \Omega} =
 * \langle |A_n| \hat u ,\mu \rangle_{\partial \Omega} \f$.
 * We define the following bilinear forms and functional:
 * \f[
 *   c_4^T(u,\mu) = \langle \tau u , \mu \rangle
 *     _{\partial \mathcal T \backslash \partial \Omega};
 *   \quad
 *   c_5^T(u,\mu) = \langle {A}_n^+ u , \mu \rangle_{\partial \Omega};
 *   \quad
 *   e_2(\hat u, \mu) =
 *     \langle (\mathbf c\cdot \mathbf n - \tau) \hat u, \mu \rangle
 *       _{\partial \mathcal T \backslash \partial \Omega};
 *   \quad
 *   e_3(\hat u , \mu)
 *     = \langle |A_n| \hat u , \mu \rangle_{\partial \Omega};
 *   \quad
 *   u^\infty(\mu) = \langle {A}_n^- u^{\infty} , \mu \rangle
 *     _{\partial \Omega}.
 * \f]
 * So, the matrix form of the conservation equation would be:
 * \f[
 *   (C_4 + C_5)^T U + (E_2 - E_3) \hat U  = U^{\infty}.
 * \f]
 *
 * Implicit Time Integration
 * -------------------------
 * Consider, the internal equation in the vecotr form, i.e.:
 * \f$A_2 U_t + (D_3-B_3)U+C_3\hat U = F_1\f$. We want to solve this
 * equation in time, using the backward difference formula. So, at each
 * time level we have:
 * \f[
 * A_2\left(U^n + \sum_{i=1}^q \alpha_i U^{n-i}\right)
 *   +\beta h (D_3 - B_3) U^n + \beta h C_3 \hat {U^n} = \beta h F_1^n.
 * \f]
 * In other words:
 * \f[
 *   \left(A_2 + \beta h(D_3 - B_3)\right) U^n
 *     + A_2 \sum_{i=1}^q \alpha_i U^{n-i}
 *     + \beta h C_3 \hat U{}^n
 *   = \beta h F_1^n
 * \f]
 * For example, for Backward Euler, \f$\alpha_1 = -1.0, \beta = 1.0,
 * \f$
 */
template <int dim>
struct AdvectionDiffusion : public GenericCell<dim>
{
  using elem_basis_type = typename GenericCell<dim>::elem_basis_type;
  typedef std::unique_ptr<dealii::FEValues<dim> > FE_val_ptr;
  typedef std::unique_ptr<dealii::FEFaceValues<dim> > FEFace_val_ptr;

  static solver_options required_solver_options();
  static solver_type required_solver_type();
  static unsigned get_num_dofs_per_node();

  AdvectionDiffusion() = delete;
  AdvectionDiffusion(const AdvectionDiffusion &inp_cell) = delete;
  AdvectionDiffusion(AdvectionDiffusion &&inp_cell) noexcept;
  AdvectionDiffusion(typename GenericCell<dim>::dealiiCell &inp_cell,
                     const unsigned &id_num_,
                     const unsigned &poly_order_,
                     hdg_model<dim, AdvectionDiffusion> *model_);
  ~AdvectionDiffusion() final;
  eigen3mat A2, B3, C3, C4, C5, D3, E2, E3, U8;
  void assign_BCs(const bool &at_boundary,
                  const unsigned &i_face,
                  const dealii::Point<dim> &face_center);

  /*!
   * Caluclates the matrix form of the bilinear forms of the
   * equations, presented above. U8 in this function is equivalent to
   * \f$U^\infty\f$ in the above formulation.
   */
  void calculate_matrices();

  void assign_initial_data();

  void compute_next_time_step_rhs();

  void assemble_globals(const solver_update_keys &keys_);

  template <typename T>
  double compute_internal_dofs(const double *const local_uhat,
                               eigen3mat &u,
                               eigen3mat &q,
                               const poly_space_basis<T, dim>
                                 output_basis);

  void internal_vars_errors(const eigen3mat &u_vec,
                            const eigen3mat &,
                            double &u_error,
                            double &q_error);

  void ready_for_next_iteration();

  void ready_for_next_time_step();

  std::vector<double> taus;

  static adv_c_vec_func_class<dim, eigen3mat> c_vec_func;
  static adv_u_func_class<dim, double> u_func;
  static adv_f1_func_class<dim, double> f1_func;
  static adv_uinf_func_class<dim, double> uinf_func;

  hdg_model<dim, AdvectionDiffusion> *model;
  BDFIntegrator *time_integrator;
  /**
   * This \c std::vector, stores all of the \f$u_i\f$'s from the relevant
   * previous time steps. These can be used to form the RHS vector of this
   * element. In other words, we form the
   * sum: \f$\sum_{i=1}^q \alpha_i u_{n-i}\f$, and use it in the calculation
   * of \f$u_{n+1}\f$.
   */
  std::vector<eigen3mat> u_i_s;
};

#include "advection_diffusion.cpp"

#endif
