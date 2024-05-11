import os
import setuptools


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), "r", encoding="utf-8") as fh:
        return fh.read()


setuptools.setup(
    name="spdb",
    version=read("VERSION"),
    description="spDB - A super memory-efficient vector database",
    license="MIT License",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/SuperpoweredAI/spDB",
    project_urls={
        "Homepage": "https://github.com/SuperpoweredAI/spDB",
        "Documentation": "https://github.com/SuperpoweredAI/spDB",
        "Contact": "https://github.com/SuperpoweredAI/spDB",
    },
    author="superpowered",
    author_email="nick@superpowered.ai, zach@superpowered.ai, justin@superpowered.ai",
    keywords="Superpowered AI Knowledge base as a service for LLM applications",
    packages=["spdb"],
    install_requires=read("requirements.txt"),
    include_package_data=True,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
