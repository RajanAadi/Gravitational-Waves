from setuptools import setup, find_packages

setup(
    name="gw_analytics",
    version="0.1.0",
    author="Adhyatma Rajan",
    description="A statistical framework for population analysis of gravitational wave events.",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gwpy>=3.0.0",
        "gwosc>=0.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "astropy>=5.0.0",
        "matplotlib>=3.7.0",
        "jinja2>=3.1.0"
    ]
)
