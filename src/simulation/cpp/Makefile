LDLIBS=-lboost_system -lboost_thread 
LDINCLUDES=`python3 -m pybind11 --includes` \
			-I../ \
			-I/usr/local/include \
			-I/usr/local/lib \
			-I../../utils/include \
			
CPPFLAGS=-Wall -Ofast -std=c++14 -fPIC -shared -march=native -frename-registers -fopenmp
PYTHON_EXT=`python3-config --extension-suffix`

all: simulation

simulation: simulation.cpp
	g++ $(CPPFLAGS) simulation.cpp -o simulation$(PYTHON_EXT) $(LDINCLUDES) $(LDLIBS)
