data-analysis:
	python -m breast_cancer_classifier.cli.generate_eda

gen-split-raw-data:
	python -m breast_cancer_classifier.cli.generate_split_datasets

build-report:
	python -m breast_cancer_classifier.cli.build_report
