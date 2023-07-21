# Transformation-Invariant-Self-Training

This repository provides the official PyTorch implementation of Transformation-Invariant Self-Training (Domain Adaptation for Medical Image Segmentation using Transformation-Invariant Self-Training).

TI-ST is initially proposed for semantic segmentation in the medical domain but can be adopted for any general-purpose image segmentation problem.

This method uses transformation-invariant highly-confident predictions in the target dataset by considering an ensemble of high-confidence predictions from transformed versions of identical inputs.

---

**Problem of domain shift in medical image segmentation.**

<img src="./Figures/Datasets2.png" alt=" Problem of domain shift in medical image segmentation." width="1000">

---

**Overview of the proposed unsupervised domain adaptation framework.**

Ignored pseudo-labels during unsupervised loss computation are shown in turquoise.

<img src="./Figures/BD8.png" alt=" Overview of the proposed unsupervised domain adaptation framework." width="1000">

---

**Four-fold training curves corresponding to TI-ST and the main alternative methods.**

<img src="./Figures/WB_1.png" alt="Four-fold training curves corresponding to TI-ST and the main alternative methods." width="1000">
<img src="./Figures/WB_2.png" alt="Four-fold training curves corresponding to TI-ST and the main alternative methods." width="1000">

---

**Ablation studies on the pseudo-labeling threshold and size of the labeled dataset.**

<img src="./Figures/ablationf.png" alt="Ablation studies on the pseudo-labeling threshold and size of the labeled dataset." width="1000">

---

**Ablation study on the performance stability of TI-ST vs. ST across the different experimental segmentation tasks.**

<img src="./Figures/ablation_stability10.png" alt="Ablation study on the performance stability of TI-ST vs. ST across the different experimental segmentation tasks." width="1000">

---

**Qualitative comparisons between the performance of TI-ST and four existing methods.**

<img src="./Figures/supp_qualitative.png" alt="Qualitative comparisons between the performance of TI-ST and four existing methods." width="1000">

---

**Comparisons between the training time of the proposed TI-ST and the main alternatives.**

<img src="./Figures/training_time.png" alt="Comparisons between the training time of the proposed TI-ST and the main alternatives." width="1000">

---


## Citation
If you use AdaptNet for your research, please cite our paper:

```
@inproceedings{ghamsarian2023TI-ST,
  title={Domain Adaptation for Medical Image Segmentation using Transformation-Invariant Self-Training},
  author={Ghamsarian, Negin and Gamazo Tejero, Javier and Márquez Neila, Pablo and Wolf, Sebastian and Zinkernagel, Martin and Schoeffmann, Klaus and Sznitman, Raphael},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={},
  year={2023},
  organization={Springer}
}
```

## Acknowledgments

This work was funded by Haag-Streit Switzerland.
