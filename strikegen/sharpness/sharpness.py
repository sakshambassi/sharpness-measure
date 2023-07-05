import numpy as np
import torch
import transformers

class Sharpness:
    def __init__(self) -> None:
        pass

    def calculate_projection_radius(model, alpha=0.05):
        """
        Calculate the projection radius to make the model not deviate beyond projection_radius
        
        Args:
            model: given pytorch model
            alpha: param value which is used to adjust the radius
        
        Returns:
            float: projected radius that the model should be within
        """
        total_norm = 0
        for param in model.parameters():
            total_norm += torch.norm(param)
        projection_radius = alpha * total_norm
        return projection_radius

    def calculate(model: transformers.models.bert,
                input_tensor: dict,
                label_tensor: torch.Tensor,
                gradient_accumulation_steps: int = 1,
                noise_scale: float = 0.1,
                verbose: bool = False
                ):
        """
        Calculate absolute sharpness on the given batch of model
        
        Args:
            model: given model
            input_tensor: is the batch on which sharpness will be calculated
            label_tensor: ground truth values of batch
            gradient_accumulation_steps: the gradient accumulation steps in case used in code
            noise_scale: the noise scale used as coefficient to calculate w'
            verbose: boolen value whether to print loss values
        Returns:
            float: sharpness value
        """
        # Algorithm:
        # 1. w_0 = orginial_weight
        # 2. w = w_0 + e        # because chances are that gradient=0 on minima
        # 3. dw = ∇ L(w)
        # 4. w' = w + n*dw
        # 5. proj(w') = {
        #               w'                                          if ||w'-w_0|| < p
        #               w_0 + [(w' - w_0) / ||w' - w_0|| * p]       otherwise
        #               }
        # 6. return L(w') - L(w_0)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        params = [(n, p) for n, p in model.named_parameters()]
        
        # 1. Save the original weights and compute the loss with those weights
        W_0 = [v.data.clone() for k, v in model.named_parameters()]
        outputs_0 = model(**input_tensor)
        loss_0 = outputs_0[0]
        # loss_0 = criterion(logits_0, label_tensor)
        loss_val_0 = loss_0.item() / gradient_accumulation_steps

        def add_noise_to_weights(scale: float = 0.1):
            # Helper function to add noise to the model weights
            for w in model.parameters():
                with torch.no_grad():
                    noise = torch.randn_like(w) * scale
                    w.add_(noise)
        # 2. Add noise to the weights to perturb them
        add_noise_to_weights(noise_scale)
        W_prime = [v.data.clone() for k, v in model.named_parameters()]

        # 3. Compute the loss with the perturbed weights and calculate the gradients
        outputs = model(**input_tensor)
        logits = outputs[1]
        loss = criterion(logits, label_tensor)

        dW = torch.autograd.grad(
            loss,
            [p for n, p in params],
            retain_graph=True,
            create_graph=True
        )
        # 4. Update the weights using the gradients
        n = learning_rate = 0.05
        with torch.no_grad():
            for i, (name, param) in enumerate(params):
                param.data = W_prime[i] + n * dW[i]

        # 5. Project the updated weights back to a valid range
        def project_weights(W_new, W_0, p):
            if torch.norm(W_new - W_0) < p:
                return W_new
            else:
                return W_0 + ((W_new - W_0) / torch.norm(W_new - W_0)) * p

        W_new = [v.data for k, v in model.named_parameters()]

        projection_radius = calculate_projection_radius(model)

        W_proj = [project_weights(W_new[i], W_0[i], projection_radius) for i in range(len(W_new))]
        with torch.no_grad():
            for i, (name, param) in enumerate(params):
                param.data = W_proj[i]

        # 6. Compute the loss with the projected weights and calculate the sharpness value
        model.train()
        outputs_prime = model(**input_tensor)
        loss_prime = outputs_prime[0]
        loss_prime_val = loss_prime.item() / gradient_accumulation_steps
        if verbose:
            print(f'[INFO] loss_prime, loss = {loss_prime_val}, {loss_val_0}')
        return loss_prime_val - loss_val_0
