from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read() if fh else ""

setup(
    name="ee-ai-registry",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI Model and Prompt Registry for Enterprise AI Gateway",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ee-ai-registry",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.109.1",
        "uvicorn>=0.27.0",
        "pydantic>=2.5.3",
        "httpx>=0.25.2",
        "python-dotenv>=1.0.0",
        "langfuse>=2.0.0",
        "mlflow>=2.8.0",
        "msal>=1.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)
