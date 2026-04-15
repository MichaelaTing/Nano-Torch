.PHONY: lib, pybind, clean, format, all

PYTHON = ~/anaconda3/envs/llm/bin/python

all: lib

lib:
	@mkdir -p build
	@cd build; cmake -DCMAKE_ROOT_DIR=~/anaconda3/envs/llm ..
	@cd build; $(MAKE)

format:
	$(PYTHON) -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so

debug:
	@echo "Using Python: $(PYTHON)"
	$(PYTHON) --version