# HASA: Subnet Allocation for Compute-Constrained Model-Heterogeneous Federated Learning

Code for the IEEE EDGE 2026 paper by Amir Hossein Shahdadian, Ahmed M. Abdelmoniem, Mahdi Taheri, Samira Nazari and Christian Herglotz.

[Paper (arXiv:2606.07621)](https://arxiv.org/abs/2606.07621) · [![Open the demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Reliable-and-Secure-Systems/HASA/blob/main/notebooks/HASA_demo.ipynb)

In model-heterogeneous federated learning, every client trains a subnet of one shared supernet. Most allocation rules give the wider subnets to clients with more data or more compute. HASA looks at the data instead. It scores each client by how far its training distribution is from the pooled one (Jensen-Shannon divergence, computed from training data only), gives wider subnets to the more heterogeneous clients, and rescales all widths so that the size-weighted compute budget stays fixed. On an article-title next-word prediction benchmark with seven clients, HASA improves mean, worst-client and P10 accuracy over uniform allocation at the same budget.

![HASA overview](images/hasa_overview.png)

## Demo

Open the notebook with the Colab badge above, choose a GPU runtime (*Runtime → Change runtime type → T4 GPU*) and run all cells. The first cell clones this repository and builds the benchmark. Each cell shows one claim of the paper:

| Cell | What it shows | Paper |
|---|---|---|
| 1 | The benchmark: seven Medium publications as clients | Sec. V-A |
| 2 | Train-only heterogeneity score (JSD) | Sec. IV-A, Table III |
| 3 | HASA allocation under a fixed budget B = 0.5 | Sec. IV-B, Algorithm 1 |
| 4 | Slimmable LSTM: width r runs the first ⌊256·r⌋ hidden units | Sec. III-A, V-A |
| 5 | Federated training, HASA vs Uniform at the same budget | Sec. VI-A |
| 6 | This run next to the published 10-seed results | Fig. 3, Table III |

Cells 1 to 4 run in seconds and reproduce the paper's split, heterogeneity scores and widths exactly. Cell 5 trains two 50-round federated runs, which takes about half an hour on a laptop GPU. It trains one seed, and GPU training is not bit-for-bit repeatable, so its accuracies move by a few tenths of a point between runs and machines. The paper reports the mean over 10 seeds, so compare the trend.

To run it locally instead:

```bash
git clone https://github.com/Reliable-and-Secure-Systems/HASA.git
cd HASA
pip install -r requirements.txt
jupyter notebook notebooks/HASA_demo.ipynb
```

## Repository

| Path | Contents |
|---|---|
| `hasa/data.py` | Article-title benchmark: Medium titles, one client per publication, 70/10/20 article-level splits |
| `hasa/method.py` | Train-only heterogeneity score and budgeted width allocation (Algorithm 1), plus the other allocation policies compared in the paper |
| `hasa/model.py` | Slimmable one-layer LSTM (embedding 128, hidden 256) |
| `hasa/train.py` | Federated training with FedAvg, and the selective-aggregation baselines |
| `notebooks/HASA_demo.ipynb` | The demo |
| `data/medium_data.csv` | The dataset (see below) |

## Data

`data/medium_data.csv` is the [Medium articles dataset](https://www.kaggle.com/datasets/dorianlazar/medium-articles-dataset) by Dorian Lazar (CC0: Public Domain), included unchanged. Titles are cleaned, lowercased and split into words; the vocabulary and the heterogeneity scores use training articles only.

## Citation

```bibtex
@inproceedings{shahdadian2026hasa,
  title     = {{HASA}: Subnet Allocation for Compute-Constrained Model-Heterogeneous Federated Learning},
  author    = {Shahdadian, Amir Hossein and Abdelmoniem, Ahmed M. and Taheri, Mahdi and Nazari, Samira and Herglotz, Christian},
  booktitle = {2026 IEEE International Conference on Edge Computing and Communications (EDGE)},
  year      = {2026},
  eprint    = {2606.07621},
  archivePrefix = {arXiv}
}
```

## License

MIT, see `LICENSE`.
