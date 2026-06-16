# Point2Roof — adapted for end-to-end 2D building-outline delineation

This repository is an adaptation of **Point2Roof** (Li et al., 2022) for **2D building-outline (footprint) delineation from normalized airborne LiDAR point clouds**, rather than 3D roof-graph reconstruction. The graph-learning logic of the original framework is retained, but the prediction target is redirected from a 3D roof structure to a single closed 2D polygon that is evaluated against authoritative cadastral footprints.

> **Adapting thesis:** _End-to-End 2D Building Polygon Delineation from Normalized LiDAR Point Clouds: A Deep Learning Approach_ — M.Sc. Geo-information Science and Earth Observation (ITC, University of Twente).
>
> **Base framework:** _Point2Roof: End-to-end 3D building roof modeling from airborne LiDAR point clouds_ — Li Li, Nan Song, Fei Sun, Xinyi Liu, Ruisheng Wang, Jian Yao, Shaopeng Cao (ISPRS Journal of Photogrammetry and Remote Sensing, 2022).

## What is different from the original Point2Roof

The single architectural change is at the boundary-construction stage. The original Paired-Point-Attention (PPA) module classifies every candidate vertex pair as a valid/invalid edge, which gives no guarantee that the accepted edges form a closed polygon. Here it is replaced by a **phase-prediction head** that regresses each vertex's position along the boundary ring; sorting these positions and connecting consecutive vertices yields a **simple closed polygon by construction**.

| Property             | PPA module (original)                        | Phase head (this work)                               |
| -------------------- | -------------------------------------------- | ---------------------------------------------------- |
| Prediction task      | Per-pair edge classification                 | Per-vertex angular position on the ring              |
| Output               | One score per candidate pair (quadratic)     | One `(cos φ, sin φ)` unit vector per vertex (linear) |
| Loss                 | Binary cross-entropy on edge labels          | Cosine-similarity on the predicted phase             |
| Polygon closure      | Not guaranteed; needs cycle detection        | Guaranteed by construction (sorting → closed ring)   |
| Topological validity | Not guaranteed                               | Every vertex has exactly two ring neighbours         |
| Intended output      | Arbitrary roof graph (hips, ridges, valleys) | Single closed 2D footprint                           |

Everything upstream of the phase head — PointNet++ backbone, point-wise corner classification, offset regression, DBSCAN clustering, and cluster refinement — is inherited from Point2Roof and re-supervised for the 2D task.

## Pipeline

```
Normalized building component (points.xyz, 1024 pts)
      │
      ▼
PointNet++ encoder–decoder  ──► point-wise corner classification (focal loss)
      │                     └─► offset regression to nearest GT vertex (Smooth-L1)
      ▼
DBSCAN clustering (XY, eps = 0.02, min_pts = 4)  ──► cluster centroids = candidate vertices
      ▼
Cluster refinement head (Smooth-L1 residual)
      ▼
Phase head  ──► per-vertex (cos φ, sin φ);  cosine-similarity loss over Hungarian-matched vertices
      ▼
Sort by φ → connect consecutive → closed ring → graph flattening (Z → constant)
      ▼
GeoJSON polygon (EPSG:28992)
```

Ground-truth ordering is canonicalized counter-clockwise from the lexicographically smallest vertex, with target phase `φ = 2π · position / n`. Distance metrics are reported in **normalized units** (a fraction of the per-sample coordinate range); de-normalization is applied only for the GeoJSON output and visual comparison.

## Dataset

The model is trained and tested on **pre-segmented building components** derived from the Dutch national airborne LiDAR dataset (**AHN**), with **BAG** cadastral polygons as the reference outline. The original synthetic roof / RoofN3D data of Point2Roof is **not** used.

Per-sample preparation:

1. Point-based terrain normalization (height-above-ground) and filtering to building-class points.
2. Connected-components extraction → one building component per sample.
3. Spatial matching of each component to its BAG polygon; redundant collinear vertices removed.
4. Each BAG polygon written as a Wavefront `polygon.obj` (vertices `v`, boundary edges `l`); the component stored as `points.xyz`.
5. Per-sample isotropic min–max normalization to the unit cube (same offset/scale on x, y, z; aspect ratio preserved). The per-sample min/max is retained for de-normalization to EPSG:28992.

Each training sample is therefore a `(points.xyz, polygon.obj)` pair, organized one folder per component:

```
000001/ points.xyz  polygon.obj
000002/ points.xyz  polygon.obj
...
```

> **TODO:** insert the download link for the prepared AHN/BAG component dataset here. (The original BaiduYun/Google Drive links pointed to the synthetic roof dataset and have been removed.)

## Installation

Before training, build and install `pc_util` (PointNet++ CUDA ops):

```shell
cd pc_util
python setup.py install
```

Requires **PyTorch 1.8 or newer**. Additional dependencies (list may be incomplete):

- `numpy`, `scipy`
- `scikit-learn` (DBSCAN)
- `pyproj` (EPSG:28992 transforms)
- `shapely` (polygon construction and topology validation)
- `laspy` / a LAZ reader (LiDAR I/O)
- `tqdm`
- `matplotlib` (visualization)

Install any missing packages as flagged by the compilation/runtime errors.

## Train and test

The train/test split follows the predefined **Map2ImLas spatial split** (Anjanappa et al., 2026) for Enschede, so that buildings from nearby areas are not shared across training and testing (reduced spatial leakage). Prepare `train.txt` and `test.txt` accordingly, then run:

```shell
python train.py
python test.py
```

Training runs for **60 epochs** in two phases within a single optimization run:

- **Epochs 1–5 (warm-up):** only the point-wise vertex heads (corner classification + offset regression) are active.
- **Epochs 6–60 (joint):** the cluster-refinement and phase heads are activated, and all four losses are optimized jointly.

Optimizer: Adam, initial LR `1e-3`, weight decay `1e-3`, step schedule ×0.5 every 20 epochs, batch size 64, 1024 points per cloud. The final trained checkpoint (e.g. `checkpoint_epoch_60.pth`) is used for evaluation.

## Configuration

All spatial hyperparameters are in **normalized units** (the component is rescaled to the unit cube before it reaches the network). As a guide, `eps = 0.02` ≈ 0.4 m on a building ~20 m across.

| Parameter          | Value      | Units      | Stage                           |
| ------------------ | ---------- | ---------- | ------------------------------- |
| NPOINT             | 1024       | points     | input sampling                  |
| PosRadius          | 0.15       | normalized | vertex supervision (ball-query) |
| ScoreThresh        | 0.5        | –          | vertex filtering                |
| MatchRadius        | 0.2        | normalized | Hungarian matching              |
| Cluster eps        | 0.02       | normalized | DBSCAN                          |
| Cluster min_pts    | 4          | points     | DBSCAN                          |
| Refine SA radii    | [0.1, 0.2] | normalized | refinement                      |
| Refine SA nsamples | [16, 16]   | points     | refinement                      |
| cls_weights        | 1.0        | –          | loss balance                    |
| reg_weights        | 1.0        | –          | loss balance                    |

## Evaluation metrics

Performance is reported with vector- and boundary-centric metrics rather than raster overlap alone:

- **Area:** Precision, Recall, F1, IoU
- **Shape / boundary:** C-IoU, Boundary IoU
- **Geometric distance (normalized units):** PoLiS, Chamfer, Hausdorff
- **Corner / topology:** N-ratio (predicted/reference vertex count), vertex-count recall, exact corner-count match, topological validity (closure + simplicity)

Headline results on the held-out test set (1,029 components):

| Metric                             | Value                                         |
| ---------------------------------- | --------------------------------------------- |
| Precision / Recall / F1            | 0.901 / 0.884 / 0.888                         |
| IoU                                | 0.862                                         |
| C-IoU / Boundary IoU               | 0.784 / 0.783                                 |
| Mean PoLiS / Chamfer / Hausdorff   | 0.180 / 0.177 / 0.561                         |
| Mean N-ratio / vertex-count recall | 0.835 / 0.804                                 |
| Exact corner-count match           | 35.7%                                         |
| Self-intersecting predictions      | 0 / 1008 (closure guaranteed by construction) |
| Prediction failure rate            | 21 / 1029 (2.0%)                              |

## Citation

If you use this adapted framework, please cite the adapting thesis and the original Point2Roof paper:

```bibtex
@mastersthesis{baig2026outline,
  title  = {End-to-End 2D Building Polygon Delineation from Normalized LiDAR Point Clouds: A Deep Learning Approach},
  author = {Baig, Muhammad Hashir},
  school = {University of Twente, Faculty of Geo-Information Science and Earth Observation (ITC)},
  year   = {2026}
}

@article{li2022point2roof,
  title   = {Point2Roof: End-to-end 3D building roof modeling from airborne LiDAR point clouds},
  author  = {Li, Li and Song, Nan and Sun, Fei and Liu, Xinyi and Wang, Ruisheng and Yao, Jian and Cao, Shaopeng},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {193},
  pages   = {17--28},
  year    = {2022},
  publisher = {Elsevier}
}
```

If you use the Map2ImLas spatial split, please also cite:

```bibtex
@article{anjanappa2026map2imlas,
  title   = {Map2ImLas: Large-scale 2D-3D airborne dataset with map-based annotations},
  author  = {Anjanappa, G. and Oude Elberink, S. and Maiti, A. and Lin, Y. and Vosselman, G.},
  journal = {ISPRS Open Journal of Photogrammetry and Remote Sensing},
  volume  = {19},
  pages   = {100112},
  year    = {2026}
}
```

## Acknowledgements

This work builds directly on the open-source [Point2Roof](https://github.com/) implementation by Li et al. The PointNet++ backbone, corner-classification and offset-regression heads, DBSCAN clustering, and cluster refinement are inherited from that framework; the phase-prediction head and the 2D-outline supervision are the contribution of this adaptation.

## Contact

Adapted framework: Muhammad Hashir Baig (m.h.baig@student.utwente.nl)
Original Point2Roof: Li Li (li.li@whu.edu.cn)
