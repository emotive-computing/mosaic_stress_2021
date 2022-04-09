import setuptools

setuptools.setup(name='commonapp',
      version='1.0.0',
      description='Common utilities for classification and regression using language or feature modeling',
      url='https://github.com/emotive-computing/common-models',
      author='Cathlyn Stone',
      author_email='cathlyn.stone@colorado.edu',
      packages=setuptools.find_packages(),
      entry_points = {
        'console_scripts': ['main=src.tasks.main:main',
                            'train-bert=src.tasks.bert_multilabel_train:main',
                            'bert-generalizability=src.tasks.bert_generalizability:main',
                            'predict-bert=src.tasks.bert_predict:main',
                            'predict-prob=src.tasks.predict_probabilities:main',
                            'generalizability=src.tasks.train_generalizable:main',
                            'predict=src.tasks.predict_generalizability_from_saved_models:main',
                            'word-clouds=src.tasks.generate_feature_info:main',
                            'analyze-results=src.tasks.analyze_results:main',
                            'download-nltk=src.tasks.download_nltk:main'],
      },
      install_requires=[],
      python_requires='>=3',
      zip_safe=False)
