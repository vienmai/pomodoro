CXX := mpic++
# CXX := g++
SRCDIR := examples
BIN := bin

# SRC := $(wildcard $(SRCDIR)/*.cpp) 
# SRC := $(SRCDIR)/test_move.cpp 
# SRC := $(SRCDIR)/twologres.cpp 
# SRC := $(SRCDIR)/logres.cpp 
# SRC := $(SRCDIR)/rosenbrock.cpp 
#SRC := $(SRCDIR)/proxgradient.cpp 
# SRC := $(SRCDIR)/lbfgs.cpp 
# SRC := $(SRCDIR)/admm.cpp  
#SRC := $(SRCDIR)/aa_gram_matrix.cpp  
SRC := $(SRCDIR)/lapack.cpp  
EXE := $(patsubst %.cpp,%,$(filter %.cpp,$(SRC)))
BIN := bin/$(notdir $(EXE))

CXXFLAGS := -std=c++17 -Wall -O2 -march=native
LIBFLAGS := -lblas -llapack
INC := -I src/

.PHONY: all
all: $(EXE)

# pattern rules
% : %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -o $(BIN) $< $(LIBFLAGS)

.PHONY: clean
clean:
	$(RM) $(EXE)
