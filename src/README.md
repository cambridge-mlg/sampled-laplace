# Following the Code in `em_trainer.py`

- Set initial $\lambda$.
- Number of samples $K$, Number of parameters $P$.

- `get_sto_samples` makes one pass through the training data and calculates the following -

  - `w0s_prior`: PyTree of size (K, P)
    - $\mathbf{w}_{0, prior} \sim \mathcal{N}\left(\mathbf{0}, \boldsymbol{\Lambda}^{-1}\right)$
  - `w0s_data`: PyTree of size (K, P)
    - $\mathbf{w}_{0, data} \sim \mathcal{N}\left(\mathbf{0}, \boldsymbol{\Lambda}^{-1} \mathbf{G} \boldsymbol{\Lambda}^{-1}\right)$
    - Using the fact that we can decompose this elementwise: $\mathbf{\Lambda}^{-1}+\mathbf{\Lambda}^{-1} \mathbf{G} \boldsymbol{\Lambda}^{-1}=\mathbf{\Lambda}^{-1}+\sum_{n=1}^N \mathbf{\Lambda}^{-1} \mathbf{J}\left(\mathbf{x}_n\right)^{\top} \mathbf{H}_L\left(\mathbf{x}_n\right) \mathbf{J}\left(\mathbf{x}_n\right) \mathbf{\Lambda}^{-1}$
  - `inv_scale_vec`: PyTree of size (P)
    - $\mathbf{s}^2 \approx \frac{1}{K} \frac{1}{N} \sum_{k=1}^K \sum_{n=1}^N\left(\mathbf{J}\left(\mathbf{x}_n\right)^{\top} \mathbf{L}\left(\mathbf{x}_n\right) \boldsymbol{\epsilon}_{n, k}\right)^2$

- We then compute samples from the prior distribution $\mathbf{w}_0 = \mathbf{w}_{0, prior} + s . \mathbf{w}_{0, data}$.

- Do EM Steps ->
  - First E Step: Obtain mode of linear model $f(\mathbf{x}) = \mathbf{J}(\mathbf{x}) \mathbf{w}$.
    - Loss function is $L(w) = \text{Crossentropy}(\mathbf{J} w, y) + ||w||^2_{\Lambda}$
  - Done by `optimise_MAP`
  - Then optimise sample-then-optimise objective to get $K$ samples
    - Loss function is $L(w_{samples}) = ||w_{sample} - \mathbf{w}_0 ||^2_{\Lambda} + \sum_{i}^N ||\mathbf{J} w_{sample} ||^2_{H_L}$
  - Done by `optimise_samples`
  - Then M Step: Do MacKay update using `w_lin` and `w_samples`
    - $\begin{aligned} \text{eff dim} &=\operatorname{Tr}\left(\mathbf{\Sigma J}\left(x_i\right)^{\top} \mathbf{H}_L\left(x_i\right) \mathbf{J}\left(x_i\right)\right) \\ & \approx \frac{1}{K} \sum_{k=1}^K \sum_{i=1}^N w_{sample}^{(k) \top} \mathbf{J}\left(x_i\right)^{\top} \mathbf{H}_L\left(x_i\right) \mathbf{J}\left(x_i\right) w_{sample}^{(k)} \end{aligned}$
    - $\lambda_{new} = \frac{\text{eff dim}}{||w_{lin}||^2}$\
    - Done by `function_space_Mackay_update`
