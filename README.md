# BayCon: Model-agnostic Bayesian Counterfactual Generator

This project contains the code and the paper _"BayCon: Model-agnostic Bayesian Counterfactual Generator"_, which has
been accepted for presentation at IJCAI-ECAI 22 (the 31st International Joint Conference on Artificial Intelligence and
the 25th European Conference on Artificial Intelligence) and inclusion in the proceedings.

Authors: Piotr Romashov<sup>1*</sup>, Martin Gjoreski<sup>1*</sup>, Kacper Sokol<sup>2</sup>, Vanina Martinez<sup>
3</sup>, Marc Langheinrich<sup>1</sup>  
<sup>1</sup>Università della Svizzera italiana, Switzerland  
<sup>2</sup>RMIT University, Australia  
<sup>3</sup>Universidad de Buenos Aires, Argentina  
<sup>*</sup>Authors with equal contribution

### Paper abstract

Generating counterfactuals to discover hypothetical predictive scenarios is the de facto standard for explaining machine
learning models and their predictions. However, building a counterfactual explainer that is time-efficient, scalable,
and model-agnostic, in addition to being compatible with continuous and categorical attributes, remains an open
challenge. To complicate matters even more, ensuring that the contrastive instances are optimised for feature sparsity,
remain close to the explained instance, and are not drawn from outside of the data manifold, is far from trivial. To
address this gap we propose BayCon: a novel counterfactual generator based on probabilistic feature sampling and
Bayesian optimisation. Such an approach can combine multiple objectives by employing a surrogate model to guide the
counterfactual search. We demonstrate the advantages of our method through a collection of experiments based on six
real-life datasets representing three regression tasks and three classification tasks.

Full paper info on [IJCAI BayCon](https://www.ijcai.org/proceedings/2022/104)

### Reference

If you use this software or rely on the underlying paper, please cite it as below or view the citation file.

Piotr Romashov, Martin Gjoreski, Kacper Sokol, Maria Vanina Martinez, and Marc Langheinrich. BayCon: Model-agnostic
Bayesian Counterfactual Generator. In IJCAI, 2022.

[Colab](https://colab.research.google.com/drive/1-5VBkm-PWOpr_sqV8NuiY8wBrWvuTWQn?usp=sharing) with running project.  
Code mantainer: promachov@gmail.com
