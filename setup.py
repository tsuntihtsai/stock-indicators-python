from setuptools import setup, find_packages

setup(
    name='stock_indicators_python',
    version='0.1.0',
    description='KD/RSI/MA/MACD indicators for OHLCV data in Python',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
    ],
)
