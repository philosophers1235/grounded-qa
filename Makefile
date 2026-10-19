DOCS := samples/northwind/docs
QUESTIONS := samples/northwind/questions.yaml
export GQA_CONTACT ?= Northwind technical support (support@northwind-heating.example)

.PHONY: install index demo eval test serve pdf

install:
	uv sync

index:
	uv run gqa index $(DOCS)

demo: index
	uv run gqa ask --offline "How long is the standard warranty on the Linea 24 boiler?"
	uv run gqa ask --offline "What happens to my extended warranty if I skip a yearly service?"
	uv run gqa ask --offline "What does error code E10 mean?"
	uv run gqa ask --offline "Can I control the Linea 24 from a smartphone app?"

eval: index
	uv run gqa eval $(QUESTIONS) --offline

test:
	uv run pytest

serve: index
	uv run gqa serve --examples $(QUESTIONS)

pdf:
	uv run python scripts/make_sample_pdf.py
