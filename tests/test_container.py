import os


def test_dockerfile_configuration():
    dockerfile_path = "Dockerfile"
    assert os.path.exists(dockerfile_path), "Dockerfile does not exist!"

    with open(dockerfile_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "EXPOSE 7860" in content, "Dockerfile must expose port 7860!"
    assert "useradd -m -u 1000 user" in content, "Dockerfile must configure non-root user with UID 1000!"
    assert "uvicorn" in content, "Dockerfile must run Uvicorn entry point!"


def test_dockerignore_configuration():
    ignore_path = ".dockerignore"
    assert os.path.exists(ignore_path), ".dockerignore does not exist!"

    with open(ignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert ".git" in content
    assert "datasets/" in content


def test_readme_frontmatter_configuration():
    readme_path = "README.md"
    assert os.path.exists(readme_path), "README.md does not exist!"

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "sdk: gradio" in content or "sdk: docker" in content, "README.md frontmatter must specify valid HF SDK!"


def test_requirements_txt_exists():
    req_path = "requirements.txt"
    assert os.path.exists(req_path), "requirements.txt does not exist!"

    with open(req_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "fastapi" in content
    assert "uvicorn" in content
    assert "torch" in content
