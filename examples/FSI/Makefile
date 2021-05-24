all: run

run:
	ipython main.py

expr:
	cd exprs && make all

clean:
	cd model && make clean
	cd stimulus && make clean
	cd test && make clean
	cd exprs && make clean
	rm -rf ./__pycache__

