#include "support_classes.hpp"

const std::string currentDateTime()
{
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  std::strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
  return buf;
}

template <typename T>
void wreck_it_Ralph(T &wreckee)
{
  T wrecker;
  wrecker.swap(wreckee);
}

void Tokenize(const std::string &str_in,
              std::vector<std::string> &tokens,
              const std::string &delimiters = " ")
{
  auto lastPos = str_in.find_first_not_of(delimiters, 0);
  auto pos = str_in.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos)
  {
    tokens.push_back(str_in.substr(lastPos, pos - lastPos));
    lastPos = str_in.find_first_not_of(delimiters, pos);
    pos = str_in.find_first_of(delimiters, lastPos);
  }
}

template <int dim, int spacedim>
GenericDOF<dim, spacedim>::GenericDOF()
  : global_dof_id(-1),
    n_local_connected_DOFs(0),
    n_nonlocal_connected_DOFs(0),
    owner_rank_id(-1)
{
}
