#include "lib_headers.hpp"

#ifndef TIME_INTEGRATORS_HPP
#define TIME_INTEGRATORS_HPP

/**
 * This is an implementation of the backward difference formula for time
 * integration. If we assume \f$\partial_t y = f(t,y)\f$, the at every time
 * step,
 * we want to compute:
 * \f[
 *   y^n = \sum_{i=1}^q \alpha_i y^{n-i} + h \beta f(t_n,y_n).
 * \f]
 *
 * The implementation is based on a variable time step size for the initial
 * time steps, to make sure that, we do not lose the accuracy. In this
 * regard, consider the initial time steps as below:
 *
 * ~~~~~
 *        h3      h2
 *        |       |            h1                 h
 *     o-----o---------o--------------o------------------------o
 *     |     |         |              |                        |
 * t_{n-4}  t_{n-3}  t_{n_2}        t_{n-1}                   t_n
 *
 * ~~~~~
 */
struct BDFIntegrator
{
  BDFIntegrator(const double &d_t_, const unsigned &order_);
  ~BDFIntegrator();

  /**
   * This function computes the sum: \f$\sum_{i=1}^q \alpha_i y^{n-i}\f$.
   */
  eigen3mat compute_sum_y_i(const std::vector<eigen3mat> y_i_s);

  /**
   * This function basically sets \f$y^{n-i} = y^{n-i+1}\f$ for \f$i={1,\cdots,
   * q}\f$.
   */
  void back_substitute_y_i_s(std::vector<eigen3mat> &y_i_s,
                             const eigen3mat &y_n);

  /**
   * This function returns the coefficient \f$\beta h \f$ in the
   * backward difference formula.
   */
  double get_beta_h();

  void go_forward();

  void reset();

  unsigned time_level_mat_calc_required();

  double h, h1, h2, time, next_time;
  unsigned order;
  unsigned time_level;
};

enum explicit_RK_type
{
  original_RK = 1 << 0,
  RK_SSP = 1 << 1
};

template <unsigned n>
struct base_explicit_RKn
{
  typedef mtl::mat::parameters<mtl::tag::row_major,
                               mtl::index::c_index,
                               mtl::fixed::dimensions<n, n> >
    aij_params;
  typedef mtl::vec::parameters<mtl::tag::row_major,
                               mtl::vec::fixed::dimension<n> >
    bi_params;
  typedef mtl::vec::parameters<mtl::tag::col_major,
                               mtl::vec::fixed::dimension<n> >
    ci_params;

  base_explicit_RKn(const double &h_) : stage(0), h(h_), current_time(0.0) {}

  virtual ~base_explicit_RKn() {}

 protected:
  double get_h_() { return h; }

  template <typename T>
  T get_sum_h_aij_kj_(const std::vector<T> &kj)
  {
    T sum;
    if (1 <= stage && stage <= 4)
    {
      sum = 0.0;
      for (unsigned j = 0; j < stage - 1; ++j)
        sum += h * this->aij[stage - 1][j] * kj[j];
    }
    else
      assert(false);
    return sum;
  }

  eigen3mat get_sum_h_aij_kj_(const std::vector<eigen3mat> &kj)
  {
    eigen3mat sum;
    if (1 <= stage && stage <= 4)
    {
      sum = eigen3mat::Zero(kj[0].rows(), kj[0].cols());
      for (unsigned j = 0; j < stage - 1; ++j)
        sum += h * this->aij[stage - 1][j] * kj[j];
    }
    else
      assert(false);
    return sum;
  }

  double get_cih_()
  {
    double cih = 0.0;
    if (1 <= stage && stage <= 4)
      cih = h * this->ci[stage - 1];
    else
      assert(false);
    return cih;
  }

  template <typename T>
  T get_sum_h_bi_ki_(const std::vector<T> &kj)
  {
    T sum;
    if (stage != 0)
      assert(false);
    sum = 0.0;
    for (unsigned i = 0; i < 4; ++i)
      sum += h * this->bi[i] * kj[i];
    return sum;
  }

  eigen3mat get_sum_h_bi_ki_(const std::vector<eigen3mat> &kj)
  {
    eigen3mat sum;
    if (stage != 0)
      assert(false);
    sum = eigen3mat::Zero(kj[0].rows(), kj[0].cols());
    for (unsigned i = 0; i < 4; ++i)
      sum += h * this->bi[i] * kj[i];
    return sum;
  }

  bool ready_for_next_step_()
  {
    bool ready = false;
    if (stage < 4)
      ++stage;
    else if (stage == 4)
    {
      stage = 0;
      current_time += h;
      ready = true;
    }
    else
      assert(false);
    return ready;
  }

  void reset_()
  {
    stage = 0;
    current_time = 0;
  }

  unsigned stage;
  double h;
  double current_time;
  mtl::vec::dense_vector<double, bi_params> bi;
  mtl::vec::dense_vector<double, ci_params> ci;
  mtl::mat::dense2D<double, aij_params> aij;
};

template <unsigned n, explicit_RK_type rk_type>
struct explicit_RKn
{
};

template <>
struct explicit_RKn<4, original_RK> : public base_explicit_RKn<4>
{
  explicit_RKn(const double &h_) : base_explicit_RKn(h_), order(4)
  {
    this->ci = 0.0, 0.5, 0.5, 1.0;
    this->bi = 1. / 6., 1. / 3., 1. / 3., 1. / 6.;
    this->aij = 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 1.,
    0.;
  }

  ~explicit_RKn() {}

  double get_h() { return this->get_h_(); }

  template <typename T>
  T get_sum_h_aij_kj(const std::vector<T> &kj)
  {
    return this->get_sum_h_aij_kj_(kj);
  }

  double get_cih() { return this->get_cih_(); }

  template <typename T>
  T get_sum_h_bi_ki(const std::vector<T> &kj)
  {
    return this->get_sum_h_bi_ki_(kj);
  }

  bool ready_for_next_step() { return this->ready_for_next_step_(); }

  void reset() { this->reset_(); }

  unsigned get_num_storage() { return 4; }

  double get_current_time() { return this->current_time; }

  double get_current_stage_time()
  {
    return (this->current_time + this->get_cih());
  }

  unsigned get_current_stage() { return this->stage; }

  unsigned order;
};

#include "time_integrators.cpp"

#endif
