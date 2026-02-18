## [0.3.2]: 18-02-2026

[Added]
- Completed the good_curve function in the ml.augood module - now fully integrated with the rest of
the repository!
- Support for Daylight path-based FPs and MAPC fingerprints
- Files needed for dataframe_2_klek are (well, should be) now automatically included!
- New manager for Train-Test splits (TTManager, data.manager)
- New method for KFoldManager: .get_non_test_data()
- New module: novami.utils for small and often used functions

[Changed]
- Files for data.descriptors now use .joblib instead of .pkl for better compatibility 
- DatasetManager renamed to KFoldManager to differentiate from TTManager
- Exposed read_pl and write_pl in novami.io module for easier imports
- Functions kf_evaluate and tt_evaluate in ml.evaluate now integrated with the rest of the code

[Removed]
- Old code in ml.evaluate

[Fixed]
- Missing parameter in RegressorAnalyzer

## [0.3.1]: 04-02-2026

[Added]
- New module: ml.augood:
  - good_curve (Generalization Out Of Distribution) curve using KFold evaluation

[Changed]
- Renamed data.similarity to data.distance (as all functions there were based on distance metrics anyway)
- Minor corrections and bug-fixes (missing imports, optional imports for some functions)

[Removed]
- Removed deprecated/visualize/ecdf.py
- Removed projects/drid (now available as separate repository at https://github.com/M-Iwan/CARBIDE)
- Removed data/process in favour of data/manipulate

## [0.3.0]: 29-01-2026
Public Release of the repository
