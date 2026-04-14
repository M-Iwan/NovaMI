# Deprecated code

Modules here are kept for reference or gradual migration. They are not the
primary supported API.

- ``deep/dataset.py`` — ``StringDataset``, ``GraphDataset``, ``GraphLoader`` for legacy MMMTGNN-style training.
- ``deep/mmgv.py`` — legacy graph vectorizer; prefer ``novami.deep.vectorizer.GraphVectorizer``.
- ``deep/gnn_regressor.py`` — standalone GNN regressor with duplicated training helpers; prefer ``novami.deep.models.TestModel`` (:class:`MMTUnit`).
- ``deep/model.py`` — older CRN / MMGNN / MMWGNN experiments.
- ``deep/mmmtgnn.py`` — MMMTGNN and modality backbones (moved from ``novami.deep.model``).

Imports use the ``deprecated`` package (install the repo in editable mode so the
package root is on ``PYTHONPATH``).
