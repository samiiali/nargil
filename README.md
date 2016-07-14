## nargil : A numerical simulation toolset

**nargil** is a set of tools for numerical simulation of PDEs. It is written
in C++ and makes use of PETSc, deal.II, Hypre, Eigen, MTL and a number of other
numerical libraries. For what matters, the name *nargil* is not an acronym,
it simply means coconut in Persian :coconut:!

- [Intro](###Intro)
- [Current Status](###current-status)

###Intro
Although, *nargil* is a program for solving a few well-known PDEs, it is
mainly written in the form of a library, which can be used in other codes.
It highly depends on deal.II, but the hybridized dG method employed hereby
is tatally independent of deal.II.

###Current Status
Currently, there is only one type of element (HDG element) in the toolset,
along with a few model problems. 
