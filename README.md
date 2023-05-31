# Data Augmentation Techniques for Source Code Models


If you'd like to add your paper, do not email us. Instead, read the protocol for [adding a new entry](https://github.com/styfeng/DataAug4NLP/blob/main/rules.md) and send a pull request.

We group the papers by [code authorship attribution](#code-authorship-attribution), [clone detection](#clone-detection), [defect detection](#defect-detection), [code summarization](#code-summarization), [code search](#code-search), [code completion](#code-completion), [code translation](#code-translation), [code question answering](#code-question-answering), [problem classification](#problem-classification), [method name prediction](#method-name-prediction), and [type prediction](#type-prediction).

This repository is based on our paper, ["Data Augmentation Approaches for Source Code Models: A Survey"](https://arxiv.org/abs/). You can cite it as follows:
```
Coming soon
```
Authors: <a href="terryyz.github.io">Terry Yue Zhuo</a>,
			  <a href="https://yangzhou6666.github.io/">Zhou Yang</a>,
			  <a href="https://v587su.github.io/">Zhensu Sun</a>,
			  <a href="https://scholar.google.com/citations?user=gFoSqqkAAAAJ&hl=en">Yufei Wang</a>,
			  <a href="http://lilicoding.github.io/">Li Li</a>,
              <a href="https://xiaoningdu.github.io/">Xiaoning Du</a>,
			  <a href="https://scholar.google.com/citations?user=0vCxuH4AAAAJ&hl=en">Zhenchang Xing</a>,
			  <a href="http://www.mysmu.edu/faculty/davidlo/">David Lo</a>

Note: WIP. More papers will be added from our survey paper to this repo soon.
Inquiries should be directed to terry.zhuo@monash.edu or by opening an issue here.


### Code Authorship Attribution
| Paper |  Evaluation Datasets | 
| -- | --- |
| Natural Attack for Pre-trained Models of Code ([ICSE '22](https://dl.acm.org/doi/abs/10.1145/3510003.3510146)) | GCJ | 
| RoPGen: Towards Robust Code Authorship Attribution via Automatic Coding Style Transformation ([ICSE '22](https://dl.acm.org/doi/abs/10.1145/3510003.3510181)) | GCJ, GitHub | 
| Boosting Source Code Learning with Data Augmentation ([ArXiv '23](https://arxiv.org/abs/2303.06808)) | GCJ | 

### Clone Detection
| Paper | Datasets | 
| -- | --- |
| Contrastive Code Representation Learning ([EMNLP '22](https://aclanthology.org/2021.emnlp-main.482/))| JavaScript (paper-specific) |
| Data Augmentation by Program Transformation ([JSS '22](https://www.sciencedirect.com/science/article/pii/S0164121222000541))| BCB |
| Natural Attack for Pre-trained Models of Code ([ICSE '22](https://dl.acm.org/doi/abs/10.1145/3510003.3510146))| BigCloneBench  |
| Unleashing the Power of Compiler Intermediate Representation to Enhance Neural Program Embeddings ([ICSE '22](https://dl.acm.org/doi/abs/10.1145/3510003.3510217))| POJ-104, GCJ |
| Heloc: Hierarchical contrastive learning of source code representation ([ICPC '22](https://dl.acm.org/doi/abs/10.1145/3524610.3527896)) | GCJ, OJClone |
| COMBO: Pre-Training Representations of Binary Code Using Contrastive Learning ([ArXiv '22](https://arxiv.org/abs/2210.05102))| BinaryCorp-3M |
| Evaluation of Contrastive Learning with Various Code Representations for Code Clone Detection ([ArXiv '22](http://arxiv.org/abs/2206.08726))| POJ-104, Codeforces |
| Towards Learning (Dis)-Similarity of Source Code from Program Contrasts ([ACL '22](https://aclanthology.org/2022.acl-long.436))| POJ-104, BigCloneBench |
| ReACC: A retrieval-augmented code completion framework ([ACL '22](https://aclanthology.org/2022.acl-long.431/)) | CodeNet  |
| Bridging pre-trained models and downstream tasks for source code understanding ([ICSE '22](https://dl.acm.org/doi/abs/10.1145/3510003.3510062))| POJ-104 |
| Boosting Source Code Learning with Data Augmentation: An Empirical Study ([ArXiv '23](https://arxiv.org/abs/2303.06808))| BigCloneBench |
| CLAWSAT: Towards Both Robust and Accurate Code Models  ([SANER '22](https://arxiv.org/abs/2211.11711))| --- |
| ContraBERT: Enhancing Code Pre-trained Models via Contrastive Learning ([ICSE '22](https://arxiv.org/abs/2301.09072))| POJ-104 |
| Pathways to Leverage Transcompiler based Data Augmentation for Cross-Language Clone Detection ([ICPC '23](https://arxiv.org/abs/2303.01435))| CLCDSA |

### Defect Detection
| Paper | Datasets | 
| -- | --- |
| Self-Supervised Bug Detection and Repair ([NeurIPS '21](https://openreview.net/forum?id=zOngaSKrElL))| RANDOMBUGS, PYPIBUGS |
| Semantic-Preserving Adversarial Code Comprehension ([COLING '22](https://aclanthology.org/2022.coling-1.267))| Defects4J |
| Path-sensitive code embedding via contrastive learning for software vulnerability detection ([ISSTA '22](https://dl.acm.org/doi/abs/10.1145/3533767.3534371))| D2A, Fan, Devign |
| Natural Attack for Pre-trained Models of Code ([ICSE '22](https://dl.acm.org/doi/abs/10.1145/3510003.3510146))| Devign |
| COMBO: Pre-Training Representations of Binary Code Using Contrastive Learning ([ArXiv '22](https://arxiv.org/abs/2210.05102))| SySeVR |
| Towards Learning (Dis)-Similarity of Source Code from Program Contrasts ([ACL '22](https://aclanthology.org/2022.acl-long.436))| REVEAL, CodeXGLUE |
| Boosting Source Code Learning with Data Augmentation: An Empirical Study ([ArXiv '23](https://arxiv.org/abs/2303.06808))| Refactory, CodRep1 |
| MIXCODE: Enhancing Code Classification by Mixup-Based Data Augmentation ([SANER '23](https://www.computer.org/csdl/proceedings-article/saner/2023/527800a379/1Nc0QvHneMg))| Refactory, CodRep1 |
| ContraBERT: Enhancing Code Pre-trained Models via Contrastive Learning ([ICSE '23](https://arxiv.org/abs/2301.09072))| Devign |


### Code Summarization
| Paper | Datasets | 
| -- | --- |
| Training Deep Code Comment Generation Models via Data Augmentation ([Internetware '20](https://dl.acm.org/doi/abs/10.1145/3457913.3457937)) | TL-CodeSum |
|Retrieval-Based Neural Source Code Summarization ([ICSE '20](https://dl.acm.org/doi/10.1145/3377811.3380383)) | PCSD, JCSD |
| Generating adversarial computer programs using optimized obfuscations ([ICLR '21](https://openreview.net/forum?id=PH5PH9ZO_4)) | Python-150K, Code2Seq Data |
| Contrastive code representation learning ([EMNLP '21](https://aclanthology.org/2021.emnlp-main.482)) | JavaScript (paper-specific) |
| A search-based testing framework for deep neural networks of source code embedding ([ICST '21](https://www.computer.org/csdl/proceedings-article/icst/2021/683600a036/1tRP9PPnyj6)) | paper-specific |
| Retrieval-Augmented Generation for Code Summarization via Hybrid GNN ([ICLR '21](https://openreview.net/forum?id=zv-typ1gPxA)) | CCSD (paper-specific) |
| BASHEXPLAINER: Retrieval-Augmented Bash Code Comment Generation based on Fine-tuned CodeBERT ([ICMSE '22](https://www.computer.org/csdl/proceedings-article/icsme/2022/795600a082/1JeFarkRRsI)) | BASHEXPLANER Data |
| Data Augmentation by Program Transformation ([JSS '22](https://www.sciencedirect.com/science/article/pii/S0164121222000541)) | DeepCom |
| Adversarial robustness of deep code comment generation ([TOSEM '22](https://dl.acm.org/doi/10.1145/3501256)) |  CCSD (paper-specific) |
| Do Not Have Enough Data? An Easy Data Augmentation for Code Summarization ([PAAP '22](https://ieeexplore.ieee.org/document/10010698)) | --- |
| Semantic robustness of models of source code ([SANER '22](https://ieeexplore.ieee.org/document/9825895)) | Python-150K, Code2Seq Data |
| CLAWSAT: Towards Both Robust and Accurate Code Models ([SANER '23](https://arxiv.org/abs/2211.11711)) | --- |
| Exploring Data Augmentation for Code Generation Tasks ([EACL '23](https://aclanthology.org/2023.findings-eacl.114/)) | CodeSearchNet (CodeXGLUE) |
| Bash Comment Generation Via Data Augmentation and Semantic-Aware Codebert ([ArXiv '23](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4385791)) | BASHEXPLANER Data |

### Code Search
| Paper | Datasets | 
| -- | --- |
| AugmentedCode: Examining the Effects of Natural Language Resources in Code Retrieval Models ([ArXiv '21](https://arxiv.org/abs/2110.08512)) | CodeSearchNet |
| Cosqa: 20, 000+ web queries for code search and question answering ([ACL '21](https://aclanthology.org/2021.acl-long.442)) | CoSQA |
| A search-based testing framework for deep neural networks of source code embedding ([ICST '21](https://www.computer.org/csdl/proceedings-article/icst/2021/683600a036/1tRP9PPnyj6)) | paper-specific |
| Semantic-Preserving Adversarial Code Comprehension ([COLING '22](https://aclanthology.org/2022.coling-1.267)) | CodeSearchNet |
| Exploring Representation-Level Augmentation for Code Search ([EMNLP '22](https://aclanthology.org/2022.emnlp-main.327/)) | CodeSearchNet |
| Cross-Modal Contrastive Learning for Code Search ([ICSME '22](https://ieeexplore.ieee.org/document/9978195/)) | AdvTest, CoSQA |
| Bridging pre-trained models and downstream tasks for source code understanding ([ICSE '22](https://dl.acm.org/doi/abs/10.1145/3510003.3510062)) | CodeSearchNet  |
| ContraBERT: Enhancing Code Pre-trained Models via Contrastive Learning ([ICSE '23](https://arxiv.org/abs/2301.09072)) | AdvTest, WebQueryTest |
| CoCoSoDa: Effective Contrastive Learning for Code Search ([ICSE '23](https://arxiv.org/abs/2204.03293)) | CodeSearchNet |
| Contrastive Learning with Keyword-based Data Augmentation for Code Search and Code Question Answering ([EACL '23](https://aclanthology.org/2023.eacl-main.262/)) | WebQueryTest |

### Code Completion
| Paper | Datasets | 
| -- | --- |
| Generative Code Modeling with Graphs ([ICLR '19](https://openreview.net/forum?id=Bke4KsA5FX)) | ExprGen Data (paper-specific) |
| Adversarial Robustness of Program Synthesis Models ([AIPLANS '21](https://openreview.net/forum?id=17C-dfA5X69)) | ALGOLISP |
| ReACC: A retrieval-augmented code completion framework ([ACL '22](https://aclanthology.org/2022.acl-long.431/)) | PY150 (CodeXGLUE), GithHub Java (CodeXGLUE) |
| Test-Driven Multi-Task Learning with Functionally Equivalent Code Transformation for Neural Code Generation ([ASE '22](https://dl.acm.org/doi/abs/10.1145/3551349.3559549)) | MBPP |
| How Important are Good Method Names in Neural Code Generation? A Model Robustness Perspective ([ArXiv '22](https://arxiv.org/abs/2211.15844)) | refined CONCODE, refined PyTorrent |
| ReCode: Robustness Evaluation of Code Generation Models ([ACL '23](https://arxiv.org/abs/2212.10264)) | HumanEval, MBPP |
| CLAWSAT: Towards Both Robust and Accurate Code Models ([SANER '23](https://arxiv.org/abs/2211.11711)) | --- |
| Retrieval-Based Prompt Selection for Code-Related Few-Shot Learning ([ICSE '23](https://people.ece.ubc.ca/amesbah/resources/papers/cedar-icse23.pdf)) | ATLAS, TFIX |

### Code Translation
| Paper | Datasets | 
| -- | --- |
| Exploring Data Augmentation for Code Generation Tasks ([EACL '23](https://aclanthology.org/2023.findings-eacl.114/)) | CodeTrans (CodeXGLUE) |
| Summarize and Generate to Back-translate: Unsupervised Translation of Programming Languages ([EACL '23](https://aclanthology.org/2023.eacl-main.112/)) | Transcoder Data |
| ContraBERT: Enhancing Code Pre-trained Models via Contrastive Learning ([ICSE '23](https://arxiv.org/abs/2301.09072))| CodeTrans (CodeXGLUE) |

### Code Question Answering
| Paper | Datasets | 
| -- | --- |
| Cosqa: 20, 000+ web queries for code search and question answering ([ACL '21](https://aclanthology.org/2021.acl-long.442)) | CoSQA |
| Semantic-Preserving Adversarial Code Comprehension ([COLING '22](https://aclanthology.org/2022.coling-1.267)) | CodeQA |
| Contrastive Learning with Keyword-based Data Augmentation for Code Search and Code Question Answering ([EACL '23](https://aclanthology.org/2023.eacl-main.262/)) | CoSQA |

### Problem Classification
| Paper | Datasets | 
| -- | --- |
| Generating Adversarial Examples for Holding Robustness of Source Code Processing Models ([AAAI '20](https://ojs.aaai.org/index.php/AAAI/article/view/5469)) | OJ |
| Generating Adversarial Examples of Source Code Classification Models via Q-Learning-Based Markov Decision Process ([QRS '21](https://ieeexplore.ieee.org/document/9724884)) | OJ |
| Heloc: Hierarchical contrastive learning of source code representation ([ICPC '22](https://dl.acm.org/doi/abs/10.1145/3524610.3527896)) | GCJ, OJ |
| COMBO: Pre-Training Representations of Binary Code Using Contrastive Learning ([ArXiv '22](https://arxiv.org/abs/2210.05102)) | POJ-104 (CodeXGLUE) |
| Bridging pre-trained models and downstream tasks for source code understanding ([ICSE '22](https://dl.acm.org/doi/abs/10.1145/3510003.3510062)) | POJ-104 |
| Boosting Source Code Learning with Data Augmentation: An Empirical Study ([ArXiv '23](https://arxiv.org/abs/2303.06808)) | Java250, Python800 |
| MIXCODE: Enhancing Code Classification by Mixup-Based Data Augmentation ([SANER '23](https://www.computer.org/csdl/proceedings-article/saner/2023/527800a379/1Nc0QvHneMg)) | Java250, Python800 |


### Method Name Prediction
| Paper | Datasets | 
| -- | --- |
| Adversarial Examples for Models of Code ([OOPSLA '20](https://dl.acm.org/doi/10.1145/3428230)) | Code2vec |
| A search-based testing framework for deep neural networks of source code embedding ([ICST '21](https://www.computer.org/csdl/proceedings-article/icst/2021/683600a036/1tRP9PPnyj6)) | paper-specific |
| Data Augmentation by Program Transformation ([JSS '22](https://www.sciencedirect.com/science/article/pii/S0164121222000541)) | Code2vec |

### Type Prediction
| Paper | Datasets | 
| -- | --- |
| Adversarial Robustness for Code ([ICML '21](https://dl.acm.org/doi/abs/10.5555/3524938.3525022)) | DeepTyper |
| Contrastive code representation learning ([EMNLP '21](https://aclanthology.org/2021.emnlp-main.482)) | DeepTyper |
| Cross-Lingual Transfer Learning for Statistical Type Inference ([ISSTA '22](https://arxiv.org/abs/2107.00157)) | DeepTyper, Typilus (Python), CodeSearchNet (Java) |

## Acknowledgement
We thank [Steven Y. Feng, et al.](https://arxiv.org/abs/2105.03075) for their open-source paper list on [DataAug4NLP](https://github.com/styfeng/DataAug4NLP).