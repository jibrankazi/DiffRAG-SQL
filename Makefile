paper:
	cd paper && pdflatex main.tex || true
train:
	python -m diffragsql.train --config configs/squad_demo.yaml
eval:
	python -m diffragsql.evaluate --config configs/squad_demo.yaml
latex:
	python -m diffragsql.to_latex --config configs/squad_demo.yaml
all:
	make train && make eval && make latex && make paper
