# SOCOFing original images

Creators: Yahaya Isah Shehu, Ariel Ruiz-Garcia, Vasile Palade, Anne James.
Source: https://www.kaggle.com/datasets/ruizgara/socofing
Paper: https://arxiv.org/abs/1807.10609
Dataset publication: Shehu et al. (2018), Sokoto Coventry Fingerprint Dataset.
Related required citation: Shehu, Y. I., Ruiz-Garcia, A., Palade, V., James, A.
(2018), Detection of Fingerprint Alterations Using Deep Convolutional Neural
Networks, ICANN 2018, Springer LNCS.

The authors provide SOCOFing for noncommercial, not-for-profit research.
The published dataset description carries CC BY-NC-SA 4.0:
https://creativecommons.org/licenses/by-nc-sa/4.0/
Retain attribution, use only within these terms, and distribute adaptations
under the same terms. Original images in this package are byte-for-byte copies,
not image adaptations. Crops are generated in memory during experiments.

The full source contains 6,000 original images from 600 people and additional
synthetically altered images. The delivery includes exactly the 300 original
left index images used in the recorded experiments (people 1 through 300).
No altered images are used. Hashes, sizes and split membership appear in each
results/*/manifest.json. The source download endpoint is approximately 878 MB
because it also includes altered images. download_data.py retains originals.

Accessed for this project: 29 October 2026 (Asia/Kolkata).
