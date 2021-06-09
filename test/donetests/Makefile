CXX :=mpic++
# CXX :=g++
CXXFLAGS :=-std=c++11 -Wall
LIBFLAGS :=-lblas

# SRC := $(wildcard *.cpp) 
# SRC := test_admm.cpp 
SRC := test_admm_.cpp 
# SRC := test_admm_dataset.cpp 
# SRC := test_optimizer.cpp 
# SRC := test_proxgradient.cpp 
# SRC := test_proxgradient_dataset.cpp 
EXE := $(patsubst %.cpp,%,$(filter %.cpp,$(SRC)))

.PHONY: all
all: $(EXE)

# pattern rules
% : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBFLAGS)

.PHONY: clean
clean:
	$(RM) $(EXE)
