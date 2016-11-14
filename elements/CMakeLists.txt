project(elements CXX)
add_custom_target(${PROJECT_NAME} SOURCES
                  advection_diffusion.cpp
                  advection_diffusion.hpp
                  cell_class.cpp
                  cell_class.hpp
                  diffusion.cpp
                  diffusion.hpp
                  gn_eps_0_beta_0.cpp
                  gn_eps_0_beta_0.hpp
                  nswe.cpp
                  nswe.hpp
                  explicit_nswe.cpp
                  explicit_nswe.hpp
                  explicit_nswe_modif.cpp
                  explicit_nswe_modif.hpp
                  explicit_gn_dispersive.cpp
                  explicit_gn_dispersive.hpp
                  explicit_gn_dispersive_modif.cpp
                  explicit_gn_dispersive_modif.hpp)
