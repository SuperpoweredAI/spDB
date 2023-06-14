import os
import setuptools


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), "r", encoding="utf-8") as fh:
        return fh.read()


setuptools.setup(
    name="superpowered-sdk",
    version=read("VERSION"),
    description="spDB - A super memory-efficient vector database",
    license="Proprietary License",                                # TODO: what license do we want?
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://superpowered.ai",
    # project_urls={
    #     "Homepage": "https://superpowered.ai",
    #     "Documentation": "https://superpowered.ai/docs",
    #     "Contact": "https://superpowered.ai/contact/",
    #     "End-User License Agreement": "https://superpowered.ai/api-user-agreement/"
    # },
    author="superpowered",
    author_email="justin@superpowered.ai",
    keywords="Superpowered AI Knowledge base as a service for LLM applications",
    packages=["spdb"],
    # package_data={"superpowered": ["errors.json"]},
    # package_dir={"": "superpowered"},
    install_requires=read("requirements.txt"),
    # include_package_data=True,
    python_requires=">=3.6",
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
