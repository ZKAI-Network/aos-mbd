# Set the location of your wallet for publication:
WALLET_LOC ?= key.json
# Set to 1 to enable debugging
DEBUG ?=

EMXX_CFLAGS=-O3 -msimd128 -fno-rtti -DNDEBUG \
	-flto=full -s BUILD_AS_WORKER=1 -s EXPORT_ALL=1 \
	-s EXPORT_ES6=1 -s MODULARIZE=1 -s INITIAL_MEMORY=800MB \
	-s MAXIMUM_MEMORY=4GB -s ALLOW_MEMORY_GROWTH -s FORCE_FILESYSTEM=1 \
	-s EXPORTED_FUNCTIONS=_main -s EXPORTED_RUNTIME_METHODS=callMain -s \
	NO_EXIT_RUNTIME=1 -Wno-unused-command-line-argument -Wno-experimental

ARCH=$(shell uname -m | sed -e 's/x86_64//' -e 's/aarch64/arm64/')

.PHONY: image
image: node AOS.wasm

.PHONY: build-test
build-test: build test-llm

.PHONY: test-llm
test-llm:
	cp build/aos/process/AOS.wasm test-llm/AOS.wasm
	cp build/aos/process/AOS.js test-llm/AOS.js
	cd test-llm && yarn test

.PHONY: test-onnx
test-onnx:
	cp build/aos/process/AOS.wasm test-onnx/AOS.wasm
	cp build/aos/process/AOS.js test-onnx/AOS.js
	cd test-onnx && yarn test

.PHONY: test
test: node
	cp AOS.wasm test/AOS.wasm
ifeq ($(TEST),inference)
	cd test; npm install; npm run test:INFERENCE
else ifeq ($(TEST),load)
	cd test; npm install; npm run test:LOAD
else
	cd test; npm install; cd test && npm test 
endif

AOS.wasm: build/aos/process/AOS.wasm
	cp build/aos/process/AOS.wasm AOS.wasm

.PHONY: node
node:
	npm install

build:
	mkdir -p build

.PHONY: clean
clean:
	rm -rf build
	rm -f AOS.wasm  test/AOS.wasm build/aos/process/AOS.wasm
	rm -f package-lock.json
	rm -rf node_modules
	# docker rmi -f p3rmaw3b/ao || true

build/aos/package.json: build
	cd build; git submodule init; git submodule update --remote

build/aos/process/AOS.wasm: build/aos/package.json build/onnxruntime container 
	docker run -v $(PWD)/build/aos/process:/src -v $(PWD)/build/onnxruntime:/onnxruntime p3rmaw3b/ao emcc-lua $(if $(DEBUG),-e DEBUG=TRUE)

build/onnxruntime: build
	if [ ! -d "build/onnxruntime" ]; then \
		cd build; git clone --recursive https://github.com/microsoft/onnxruntime; \
		cd onnxruntime; ./build.sh --config Release --build_wasm_static_lib --use_full_protobuf --skip_tests --disable_wasm_exception_catching --disable_rtti --allow_running_as_root --update --build --parallel --nvcc_threads 64 --skip_submodule_sync; \
	fi

.PHONY: container
container: container/Dockerfile
	docker build . -f container/Dockerfile -t p3rmaw3b/ao --build-arg ARCH=$(ARCH)

publish-module: AOS.wasm
	npm install
	WALLET=$(WALLET_LOC) scripts/publish-module

.PHONY: dockersh
dockersh:
	# docker run -v $(PWD)/build/aos/process:/src -it p3rmaw3b/ao /bin/bash
	docker run -v .:/src -it p3rmaw3b/ao /bin/bash
