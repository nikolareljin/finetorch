.PHONY: build test lint ci submodules

build:
	./scripts/build.sh

test:
	./scripts/test.sh

lint:
	./scripts/lint.sh

ci:
	./scripts/local_ci.sh

submodules:
	./scripts/update_submodules.sh
