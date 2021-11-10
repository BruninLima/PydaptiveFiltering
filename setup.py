from setuptools import setup

setup(
    name="pydaptivefiltering",
    packages=["pydaptivefiltering"],
    version='0.2',
    description=" Python 3.8+ Implementation of the AdaptiveFiltering Toolbox.",
    author="Bruno Lima Netto",
    author_email="brunolimanetto@gmail.com",
    url="https://github.com/BruninLima",
    download_url="https://github.com/BruninLima/PydaptiveFiltering/archive/refs/tags/v0.1.tar.gz",
    keywords=["Adaptive", "Filtering", "Digital", "Signal", "Processing"],
    install_requires=[
        'numpy',

    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3'
    ]

)
