project(models CXX)
add_custom_target(${PROJECT_NAME} SOURCES
                  explicit_hdg_model.hpp
                  explicit_hdg_model.cpp
                  generic_model.hpp
                  generic_model.cpp
                  hdg_model.hpp
                  hdg_model.cpp
                  hdg_model_with_explicit_rk.hpp
                  hdg_model_with_explicit_rk.cpp)
