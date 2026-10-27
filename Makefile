.PHONY: all run test clean 

all: build/llm

test:
	@./tests/run_test.sh

clean:
	rm -f build/llm

run:
	build/llm

build/llm: Core
	@mkdir -p build
	@build/compile.sh