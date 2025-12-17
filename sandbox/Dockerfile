FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd -m -u 10001 sandbox

WORKDIR /work

# Create task dir with correct ownership before dropping privileges
RUN mkdir -p /work/task && chown -R sandbox:sandbox /work

USER sandbox

RUN python -m pip install --user -U pip \
 && python -m pip install --user pytest

CMD ["python", "-V"]
