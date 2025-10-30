.PHONY: build run deploy clean

# Variables
PYTHON = python
PIP = pip
ACTIVATE = . aimicroservice/bin/activate &&
MANAGE = $(ACTIVATE) $(PYTHON) manage.py

# Default target (usually help)
help:
	@echo "Available commands:"
	@echo "  requirements - Freeze dependencies into requirements.txt"
	@echo "  install      - Install dependencies from requirements.txt"
	@echo "  migrate      - Run database migrations"
	@echo "  run          - Run the Django development server"
	@echo "  shell        - Start a new shell with the venv activated"
	@echo "  deploy       - Deploy to Google App Engine"

requirements:
	$(ACTIVATE) $(PIP) freeze > requirements.txt
	@echo "requirements.txt updated."

install:
	$(ACTIVATE) $(PIP) install -r requirements.txt

migrate:
	$(MANAGE) migrate

create_superuser:
	$(MANAGE) createsuperuser

run:
	@echo "Starting Django development server..."
	$(MANAGE) runserver

shell:
	$(ACTIVATE) zsh # Or bash, depending on your preference 

# Deploy to App Engine
deploy:
	gcloud app deploy

# Run tests
test:
	$(MANAGE) test

# Clean up local resources
clean:
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */migrations/__pycache__

