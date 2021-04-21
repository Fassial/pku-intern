source=report

all: pdf

pdf:
	xelated $(source)
	bibtex $(source)
	xelatex $(source)
	xelatex $(source)

clean:
	rm -rf *.aux
	rm -rf *.bbl
	rm -rf *.blg
	rm -rf *.log
	rm -rf *.out
	rm -rf *.synctex.gz
