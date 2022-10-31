# (S)MOMP
 Multidimensional matching pursuit algorithms

### Description
This library consists on my set of algorithms designed to tackle orthogonal matching pursuit problems comprising multiple dictionaries.

These algorithms mainly focus on the sparse recovery problems presented in the papers:

- [Multidimensional orthogonal matching pursuit: theory and application to high accuracy joint localization and communication at mmWave](https://arxiv.org/pdf/2208.11600.pdf)
- [Low complexity joint position and channel estimation at millimeter wave based on multidimensional orthogonal matching pursuit](https://arxiv.org/pdf/2204.03424.pdf)
- [Multidimensional Orthogonal Matching Pursuit-based RIS-aided Joint Localization and Channel Estimation at mmWave](https://arxiv.org/pdf/2203.13327.pdf)
- [Separable multidimensional orthogonal matching pursuit and its application to joint localization and communication at mmWave]()

Note that the tensor formulation used in the papers is used for confort with the algorithms development, but it is equivalent to the more comonly used expressions $Y = {\bf A}({\bf X}_1\otimes {\bf X}_2\otimes\ldots)$ for MOMP $Y = {\bf A_1}({\bf X}_{1, 1}\otimes {\bf X}_{1, 2}\otimes\ldots) + {\bf A_1}({\bf X}_{1, 1}\otimes {\bf X}_{1, 2}\otimes\ldots) + \ldots$ for MOMP with multiple sources

## Legacy code
Legacy code regarding the exact code that was used for some of these papers can be found in the repositories
- [MOMP-core](https://github.com/WiSeCom-Lab/MOMP-core)
- [MOMP-paper](https://github.com/WiSeCom-Lab/MOMP-paper)
- [SMOMP-core](https://github.com/WiSeCom-Lab/SMOMP-core)
- [ICASSP-2022-SMOMP](https://github.com/WiSeCom-Lab/ICASSP-2022-SMOMP)
