#include "time_integrators.hpp"

BDFIntegrator::BDFIntegrator(const double &d_t_, const unsigned &order_)
  : h(d_t_), time(0.), order(order_), time_level(0)
{
  if (order == 1)
  {
    h2 = h1 = h;
    next_time = h;
  }
  if (order == 2)
  {
    h2 = h1 = h / 10.;
    next_time = h1;
  }
  if (order == 3)
  {
    h2 = h / 100.;
    h1 = h / 10.;
    next_time = h2;
  }
}

void BDFIntegrator::reset()
{
  time = 0.;
  time_level = 0;
  if (order == 1)
    next_time = h;
  if (order == 2)
    next_time = h1;
  if (order == 3)
    next_time = h2;
}

BDFIntegrator::~BDFIntegrator() {}

eigen3mat BDFIntegrator::compute_sum_y_i(const std::vector<eigen3mat> y_i_s)
{
  assert(y_i_s.size() == order);
  eigen3mat sum_y_i;
  if (order == 1)
  {
    sum_y_i = -y_i_s[0];
  }
  if (order == 2)
  {
    if (time_level == 0)
      sum_y_i = -y_i_s[1];
    else if (time_level == 1)
      sum_y_i = -(h + h1) * (h + h1) / h1 / (2 * h + h1) * y_i_s[1] +
                (h * h) / h1 / (2 * h + h1) * y_i_s[0];
    else
      sum_y_i = -4. / 3. * y_i_s[1] + 1. / 3. * y_i_s[0];
  }
  if (order == 3)
  {
    if (time_level == 0)
      sum_y_i = -y_i_s[2];
    else if (time_level == 1)
      sum_y_i = -(h1 + h2) * (h1 + h2) / h2 / (2 * h1 + h2) * y_i_s[2] +
                (h1 * h1) / h2 / (2 * h1 + h2) * y_i_s[1];
    else if (time_level == 2)
    {
      double beta_h =
        h * (h + h1) * (h + h1 + h2) /
        ((h + h1 + h2) * (h + h1) + (h) * (h + h1 + h2) + (h) * (h + h1));
      sum_y_i =
        -beta_h * (h + h1) * (h + h1 + h2) / h / h1 / (h1 + h2) * y_i_s[2] +
        beta_h * (h + h1 + h2) * h / h1 / h2 / (h + h1) * y_i_s[1] -
        beta_h * h * (h + h1) / h2 / (h1 + h2) / (h + h1 + h2) * y_i_s[0];
    }
    else if (time_level == 3)
    {
      double beta_h =
        h * (h + h) * (h + h + h1) /
        ((h + h + h1) * (h + h) + (h) * (h + h + h1) + (h) * (h + h));
      sum_y_i = -beta_h * (h + h) * (h + h + h1) / h / h / (h + h1) * y_i_s[2] +
                beta_h * (h + h + h1) * h / h / h1 / (h + h) * y_i_s[1] -
                beta_h * h * (h + h) / h1 / (h + h1) / (h + h + h1) * y_i_s[0];
    }
    else
      sum_y_i =
        -18. / 11 * y_i_s[2] + 9. / 11. * y_i_s[1] - 2. / 11. * y_i_s[0];
  }
  return sum_y_i;
}

void BDFIntegrator::back_substitute_y_i_s(std::vector<eigen3mat> &y_i_s,
                                          const eigen3mat &y_n)
{
  assert(y_i_s.size() == order);
  for (unsigned i = 1; i < order; ++i)
  {
    y_i_s[i - 1] = std::move(y_i_s[i]);
  }
  y_i_s[order - 1] = y_n;
}

double BDFIntegrator::get_beta_h()
{
  double beta_h_ = 0.;
  if (order == 1)
  {
    beta_h_ = h;
  }
  if (order == 2)
  {
    if (time_level == 0)
      beta_h_ = h1;
    else if (time_level == 1)
      beta_h_ = h * (h1 + h) / (2. * h + h1);
    else
      beta_h_ = 2. / 3. * h;
  }
  if (order == 3)
  {
    if (time_level == 0)
      beta_h_ = h2;
    else if (time_level == 1)
      beta_h_ = h1 * (h2 + h1) / (2. * h1 + h2);
    else if (time_level == 2)
      beta_h_ =
        h * (h + h1) * (h + h1 + h2) /
        ((h + h1 + h2) * (h + h1) + (h) * (h + h1 + h2) + (h) * (h + h1));
    else if (time_level == 3)
      beta_h_ = h * (h + h) * (h + h + h1) /
                ((h + h + h1) * (h + h) + (h) * (h + h + h1) + (h) * (h + h));
    else
      beta_h_ = 6. / 11. * h;
  }
  return beta_h_;
}

void BDFIntegrator::go_forward()
{
  ++time_level;
  time = next_time;

  if (order == 1 || order == 2)
    next_time += h;
  if (order == 3)
  {
    if (time_level == 1)
      next_time += h1;
    else
      next_time += h;
  }
}

unsigned BDFIntegrator::time_level_mat_calc_required()
{
  if (order == 1)
    return 0;
  if (order == 2)
    return 2;
  if (order == 3)
    return 4;
  return 0;
}

/*
 *
 */
