from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="research-paper-graph",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A knowledge graph system for research paper analysis with LLM integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/research-paper-graph",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "graphviz>=0.20.0",
            "pyvis>=0.3.0",
            "bokeh>=3.2.0",
        ],
        "advanced": [
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "torch>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "research-graph=src.main:main",
        ],
    },
)