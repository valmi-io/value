from setuptools import find_packages, setup

MAIN_REQUIREMENTS = [
    "dbt-postgres==1.4.4",
    "fal==0.8.1",
    "numpy==1.26.4",
    "jinja2",
    "valmi_connector_lib==0.1.167",
]

TEST_REQUIREMENTS = [
    "pytest~=6.2",
    "connector-acceptance-test",
]

setup(
    name="source-postgres",
    description="Source implementation for Postgres.",
    author="Rajashekar Varkala @ valmi.io",
    author_email="contact@valmi.io",
    packages=find_packages(),
    install_requires=MAIN_REQUIREMENTS,
    package_data={"": ["*.json", "*.yaml"]},
    extras_require={
        "tests": TEST_REQUIREMENTS,
    },
)
