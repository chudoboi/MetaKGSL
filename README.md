# MetaKGSL introduction	

​The semantic information of entity text in genetic knowledge graph data is very important. However, conventional GCN and GAT algorithms mainly learn the association information between entities in the knowledge graph, ignoring the semantic information of the relationship itself and the semantic information fo each entity. To address this limitation, this method uses a multi-head attention algorithm to complete the learning of triplet vectors, effectively integrating multi-hop relationships and neighboring entity information, and retaining the initial entity semantic vector at the end to prevent the loss of entity characteristics caused by overly complex graphs. The following is the description of the method.

## 1. Entity relationship embedding

​First, for each entity e<sub>i</sub> obtain the set of triplets associated with the entity. For any triplet (e<sub>i</sub>, r<sub>k</sub>, e<sub>j</sub>) randomly initialize the embedding triplet as h<sub>i</sub>, g<sub>k</sub>, h<sub>j</sub>. Then these three vectors are concatenated and multiplied by the weight matrix W<sub>1</sub> to obtain the triplet embedding c<sub>ijk</sub>.

## 2. Attention weight calculation

​The calculation is done in two steps. First, we calculate the attention weight b between entity e<sub>i</sub> and each neighboring triplet:
b<sub>ijk</sub> = LeakyReLU (W<sub>2</sub> ⋅ c<sub>ijk</sub>)

​Then normalize the attention weight using softmax:

α<sub>ijk</sub> = exp(b<sub>ijk</sub>) / ∑<sub>n ∈ N<sub>i</sub></sub> ∑<sub>r ∈ R<sub>in</sub></sub>exp(b<sub>inr</sub>) 

## 3. Entity embedding update

​	After calculating all the attention weights of the current entity and its neighbor triplets, the embedding of each entity is obtained by weighting and summing the feature representations of its neighbors. The calculation method is as follows:

h<sub>i</sub><sup>′</sup> = σ(∑<sub>j∈N<sub>i</sub></sub> ∑<sub>k∈R<sub>ij</sub></sub>α<sub>ijk</sub> ⋅ c<sub>ijk</sub>)

  Finally, to effectively utilize the semantic information of the entity itself, this method obtains the updated vector $h_i^′$ of the current entity $e_i $, and preserves the original entity features through the following method to prevent the loss of initial embedded features:
  
h<sub>i</sub><sup>′′</sup>=W<sub>eh<sub>i</sub></sub> + h<sub>i</sub><sup>′</sup>
