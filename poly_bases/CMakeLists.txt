project(poly_bases CXX)
add_custom_target(${PROJECT_NAME} SOURCES
                  jacobi_polynomial.cpp
                  jacobi_polynomial.hpp
                  lagrange_polynomial.cpp
                  lagrange_polynomial.hpp
                  poly_basis.cpp
                  poly_basis.hpp)
