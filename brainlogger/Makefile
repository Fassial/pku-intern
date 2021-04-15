all: test

clean:
	rm -rf ./__pycache__
	cd brainlogger && make clean
	cd tests && make clean

test:
	cd tests && make all

