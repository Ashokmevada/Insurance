FROM quay.io/astronomer/astro-runtime:12.1.1

#STEP 2: Set the Working Directory Inside the Container
# This is crucial! Your entire Astro project (including your dags, src, dvc.yaml, .dvc)
# is copied into /usr/local/airflow/ by default by 'astro deploy'.
# Setting WORKDIR ensures that any subsequent commands (like 'dvc repro') run
# from your project's root, where dvc.yaml and .dvc are located.
WORKDIR /usr/local/airflow/

# STEP 3: Install System-Level Dependencies (if any)
# Use 'root' user to install system packages.
USER root

# Run apt-get update to ensure you have the latest package lists
RUN apt-get update

# STEP 4: Install Python Dependencies
# Switch back to the 'astro' user, which is the default for running Airflow tasks
# and generally recommended for pip installations.
USER astro

# Copy your 'requirements.txt' file.
# IMPORTANT: This 'requirements.txt' should contain *all* Python packages
# required by your DVC pipeline's scripts (e.g., in `src/`).
# This includes:
# - DVC itself with your remote type (e.g., dvc[s3])
# - pandas, numpy, scikit-learn, etc. (for your data processing/ML scripts)
# - any other specific libraries your src/ components use.
# Ensure this 'requirements.txt' is at the root of your Astro project alongside the Dockerfile.
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt






