#include "cell_class.hpp"
#include "explicit_nswe.hpp"
#include "support_classes.hpp"

#ifndef EXPLICIT_NSWE_HPP_MODIF
#define EXPLICIT_NSWE_HPP_MODIF

template <int dim>
struct explicit_gn_dispersive_modif;

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
struct explicit_nswe_modif : public GenericCell<dim>
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
   * \brief Constructor for the explicit_nswe_modif.
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
  explicit_nswe_modif() = delete;
  explicit_nswe_modif(const explicit_nswe_modif &inp_cell) = delete;
  explicit_nswe_modif(explicit_nswe_modif &&inp_cell) noexcept;
  explicit_nswe_modif(typename GenericCell<dim>::dealiiCell &inp_cell,
                      const unsigned &id_num_,
                      const unsigned &poly_order_,
                      explicit_hdg_model<dim, explicit_nswe_modif> *model_);
  ~explicit_nswe_modif() final;
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
  void set_previous_step_results(
    const explicit_gn_dispersive_modif<dim> *const src_cell);

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

  explicit_hdg_model<dim, explicit_nswe_modif> *model;
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
  std::vector<eigen3mat> ki_hats;
};

#include "explicit_nswe_modif.cpp"

#endif
