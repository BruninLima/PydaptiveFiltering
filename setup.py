from distutils.core import setup

aaa

setup(
    name="pydaptivefiltering",
    packages=["pydaptivefiltering"],
    version='0.1',
    description=" Python 3.x Implementation of the AdaptiveFiltering Toolbox.",
    author="Bruno Lima Netto",
    author_email="brunolimanetto@gmail.com",
    url="https://github.com/BruninLima",
    download_url="",
    keywords=["Adaptive", "Filtering", "Digital", "Signal", "Processing"],
    install_requires=[
        'numpy',

    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3'
    ]

)
