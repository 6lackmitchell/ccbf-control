FROM 6lackmitchell/ubuntu22-ros2-python3.10-poetry

WORKDIR /workspace

# Copy the source code including the 'src' directory
COPY src/ src/
COPY examples/ examples/

# Copy the project files
COPY .poetry/arm64/pyproject.toml.arm64 pyproject.toml
COPY .poetry/arm64/poetry.lock.arm64 poetry.lock
COPY .pkgs/cbfkit-0.1.1-py3-none-any.whl .pkgs/cbfkit-0.1.1-py3-none-any.whl

# Set the PYTHONPATH to include /workspace and /workspace/src
RUN echo 'export PYTHONPATH="/workspace:/workspace/src:${PYTHONPATH}"' >> /root/.bashrc

# Project initialization:
RUN poetry install --no-interaction --no-ansi --no-root
RUN poetry update
