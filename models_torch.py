import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.func import functional_call, vmap

class RootMLP_Regressor(nn.Module):
    """
    Root network \theta: t -> x_t.
    Optimized for PyTorch autograd support and clean layer definitions.
    """
    def __init__(self, 
                 input_dim: int,
                 output_dim: int, 
                 width_size: int, 
                 depth: int, 
                 activation: nn.Module = nn.ReLU(),
                 final_activation: nn.Module = nn.Tanh(),
                 predict_uncertainty: bool = True):
        
        super().__init__()
        
        self.predict_uncertainty = predict_uncertainty
        self.final_activation = final_activation
        
        # If we predict uncertainty (std), we need 2x outputs (mean + std)
        # otherwise we just need output_dim
        final_layer_dim = output_dim * 2 if predict_uncertainty else output_dim

        # Build layers using a list
        layers = []
        
        # Input Block
        layers.append(nn.Linear(input_dim, width_size))
        layers.append(activation)
        
        # Hidden Blocks
        for _ in range(depth - 1):
            layers.append(nn.Linear(width_size, width_size))
            layers.append(activation)
            
        # Output Block
        layers.append(nn.Linear(width_size, final_layer_dim))
        
        # Register as a submodule
        self.network = nn.Sequential(*layers)

        # Store props for reference if needed (optional)
        self.props = {
            "input_dim": input_dim, 
            "output_dim": output_dim, 
            "width": width_size, 
            "depth": depth
        }

    def forward(self, tx, std_lb=None, dtanh=None):
        out = self.network(tx)

        if not self.predict_uncertainty:
            return self._apply_final_activation(out, dtanh)

        # Split into mean and std (along the last dimension)
        # chunk is generally faster/cleaner than split for equal parts
        mean, raw_std = torch.chunk(out, 2, dim=-1)

        mean = self._apply_final_activation(mean, dtanh)

        # Process standard deviation
        # Use softplus to ensure positivity, similar to the original logic
        std = F.softplus(raw_std)
        
        if std_lb is not None:
            # use torch.clamp (equivalent to np.clip) preserves gradients
            std = torch.clamp(std, min=std_lb)

        return torch.cat([mean, std], dim=-1)

    def _apply_final_activation(self, x, dtanh):
        """Helper to apply activation or scaled tanh."""
        if self.final_activation is not None:
            return self.final_activation(x)
        
        if dtanh is not None:
            # dtanh expects a tuple: (a, b, alpha, beta)
            a, b, alpha, beta = dtanh
            return alpha * torch.tanh((x - b) / a) + beta
            
        return x
    



class RootMLP_Classif(nn.Module):
    """ Root network \theta: t -> y_t, whose weights are the WSM hidden space for classification """

    def __init__(self,
                 nb_classes,
                 width_size,
                 depth,
                 activation=nn.ReLU(),
                 positional_enc_dim=0):               ## Dimension of the positional encoding to be added to the input

        super().__init__()

        input_dim = 1 + positional_enc_dim
        output_dim = nb_classes

        layers = []

        layers.append(nn.Linear(input_dim, width_size))
        layers.append(activation)
        for _ in range(depth - 1):
            layers.append(nn.Linear(width_size, width_size))
            layers.append(activation)
        layers.append(nn.Linear(width_size, output_dim))

        self.network = nn.Sequential(*layers)
        self.props = (input_dim, output_dim, width_size, depth)

    def forward(self, tx):
        out = self.network(tx)
        return out


class GradualMLP(nn.Module):
    """ 
    Gradual MLP where neuron counts interpolate linearly 
    from input_dim to output_dim over the specified number of hidden layers.
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_layers: int = 2, 
                 activation: nn.Module = nn.Tanh()):
        super().__init__()

        layer_sizes = np.linspace(
            input_dim, output_dim, hidden_layers + 2
        ).astype(int)

        layers = []
    
        for i, (dim_in, dim_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(dim_in, dim_out))
            
            if i < len(layer_sizes) - 2:
                layers.append(activation)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class ConvNet1D(nn.Module):
    """ 
    Convolutional network for 1D data (e.g., time series).
    Structure: Input -> [Hidden * n] -> Output
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 hidden_channels: int, 
                 kernel_size: int, 
                 n_layers: int = 2, 
                 activation: nn.Module = nn.ReLU()):
        
        super().__init__()
        
        layers = []

        # -- Input Block --
        # Note: padding='same' preserves the length of the time series (requires stride=1)
        layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding='same'))
        layers.append(activation)

        # -- Hidden Blocks --
        for _ in range(1, n_layers):
            layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding='same'))
            layers.append(activation)

        # -- Output Block --
        layers.append(nn.Conv1d(hidden_channels, out_channels, kernel_size, padding='same'))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """ 
        Forward pass.
        x: (batch, in_channels, time)
        Returns: (batch, out_channels, time)
        """
        return self.network(x)





class CausalConv1d(nn.Module):
    """ Helper: 1D Causal Convolution (pads left, crops right) """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation, **kwargs)

    def forward(self, x):
        # Remove the last 'pad' elements to maintain causal length
        if self.pad > 0:
            return self.conv(x)[:, :, :-self.pad]
        return self.conv(x)

class WSM(nn.Module):
    """ 
    WARP: Weight Space Seq2Seq Model.
    Optimized with torch.func for fast Hypernetwork execution.
    """

    def __init__(self, 
                 data_size, 
                 width_size, 
                 depth, 
                 activation="relu",
                 final_activation="tanh",
                 nb_classes=None,
                 init_state_layers=2,
                 input_prev_data=False,
                 predict_uncertainty=True,
                 time_as_channel=True,
                 forcing_prob=1.0,
                 std_lower_bound=None,
                 weights_lim=None,
                 noise_theta_init=None,
                 nb_wsm_layers=1,
                 autoregressive_train=True,
                 stochastic_ar=True,
                 smooth_inference=None,
                 positional_encoding=None,
                 preferred_output_dim=None,
                 conv_embedding=None):
        
        super().__init__()

        # -- Configuration --
        self.data_size = data_size
        self.nb_classes = nb_classes
        self.ar_train = autoregressive_train
        self.stochastic_ar = stochastic_ar
        self.predict_uncertainty = predict_uncertainty
        self.time_as_channel = time_as_channel
        self.forcing_prob = forcing_prob
        self.weights_lim = weights_lim
        self.std_lower_bound = std_lower_bound
        self.noise_theta_init = noise_theta_init
        self.input_prev_data = input_prev_data
        self.classification = nb_classes is not None
        self.smooth_inference = smooth_inference if smooth_inference is not None else (not stochastic_ar)

        # Activation Map
        act_map = {
            "relu": nn.ReLU(), "tanh": nn.Tanh(), "softplus": nn.Softplus(),
            "swish": nn.SiLU(), "identity": nn.Identity()
        }
        act_fn = act_map.get(activation, nn.ReLU())

        # -- 1. Convolutional Embedding -- 
        self.conv_embedding = None
        self.conv_de_embedding = None
        
        if conv_embedding is not None:
            out_chans, kernel_size = conv_embedding[0], conv_embedding[1]
            is_causal = (len(conv_embedding) > 2 and conv_embedding[2] == 1)
            
            if is_causal:
                self.conv_embedding = CausalConv1d(data_size, out_chans, kernel_size)
            else:
                self.conv_embedding = nn.Conv1d(data_size, out_chans, kernel_size, padding='same')

            # Inverse embedding (approximate difference kernel)
            self.conv_de_embedding = nn.Conv1d(1, 1, 3, padding='same', bias=False)
            diff_kernel = torch.tensor([[[-1., 1., 0.]]])
            self.conv_de_embedding.weight.data = diff_kernel

        # -- 2. Weight Space Dynamics Setup --
        self.root_templates = nn.ModuleList()
        self.param_infos = []  # Stores structure to unflatten weights
        self.As = nn.ParameterList()
        self.Bs = nn.ParameterList()
        self.thetas_init = nn.ModuleList()

        # Handle positional_encoding: can be None, False, or a tuple (dim, constant)
        if positional_encoding and positional_encoding is not True:
            pos_enc_dim = positional_encoding[0]
        else:
            pos_enc_dim = 0

        for i in range(nb_wsm_layers):
            # A. Define Input/Output dims for Root Network
            if self.conv_embedding is None:
                r_in_dim = (1 + pos_enc_dim + data_size) if input_prev_data else (1 + pos_enc_dim)
                b_in_dim = (data_size + 1 + pos_enc_dim) if time_as_channel else data_size
            else:
                out_chans = conv_embedding[0]
                r_in_dim = (1 + pos_enc_dim + out_chans) if input_prev_data else (1 + pos_enc_dim)
                b_in_dim = (out_chans + 1 + pos_enc_dim) if time_as_channel else out_chans

            # B. Create Root Template
            if nb_classes is None: # Regression
                r_out_dim = preferred_output_dim if preferred_output_dim else data_size
                
                # Handling final activation for template
                if isinstance(final_activation, str):
                    r_final_act = act_map.get(final_activation, nn.Identity())
                elif isinstance(final_activation, float):
                    r_final_act = nn.Hardtanh(-final_activation, final_activation)
                else:
                    r_final_act = nn.Identity()

                root = RootMLP_Regressor(r_in_dim, r_out_dim, width_size, depth, 
                               activation=act_fn, final_activation=r_final_act,
                               predict_uncertainty=predict_uncertainty)
            else: # Classification
                root = RootMLP_Classif(nb_classes, width_size, depth,
                               activation=act_fn,
                               positional_enc_dim=pos_enc_dim)

            self.root_templates.append(root)

            # C. Extract Parameter Structure for functional_call
            # We map flat vector -> dictionary of tensors
            layer_info = []
            total_params = 0
            for name, param in root.named_parameters():
                layer_info.append((name, param.shape, param.numel()))
                total_params += param.numel()
            
            self.param_infos.append({
                "info": layer_info,
                "total": total_params
            })

            # D. Initialize Dynamics (A and B)
            # theta_{t+1} = A * theta_t + B * delta_x
            self.As.append(nn.Parameter(torch.eye(total_params)))
            self.Bs.append(nn.Parameter(torch.zeros(total_params, b_in_dim) * 0.001))

            # E. Initialize Theta_0
            if init_state_layers is None:
                # Direct parameter vector (learnable starting point)
                flat_params = torch.nn.utils.parameters_to_vector(root.parameters())
                self.thetas_init.append(nn.Parameter(flat_params.clone()))
            else:
                # Meta-network: Input -> Initial Weights
                theta_in_dim = data_size if conv_embedding is None else conv_embedding[0]
                # Assuming GradualMLP is defined
                # self.thetas_init.append(GradualMLP(theta_in_dim, total_params, init_state_layers, act_fn))
                self.thetas_init.append(nn.Linear(theta_in_dim, total_params)) # Simplified

    # --- Fast Weight Injection ---

    def _unflatten_weights(self, theta_flat, layer_idx):
        """
        Reconstructs the state_dict from a flat vector.
        Compatible with vmap because it uses pure tensor operations.
        """
        params_dict = {}
        idx = 0
        info = self.param_infos[layer_idx]["info"]

        for name, shape, numel in info:
            param_chunk = theta_flat[idx : idx + numel]
            params_dict[name] = param_chunk.view(shape)
            idx += numel

        return params_dict

    def _batch_root_forward(self, params_flat_batch, x_in_batch, layer_idx=0):
        """
        Execute root network in parallel across a batch using vmap.

        Args:
            params_flat_batch: (N, latent_dim) - Batch of flattened parameters
            x_in_batch: (N, input_dim) - Batch of inputs
            layer_idx: Which root template to use (default 0)

        Returns:
            (N, output_dim) - Batch of outputs
        """
        def single_forward(p, x):
            p_dict = self._unflatten_weights(p, layer_idx)
            return functional_call(self.root_templates[layer_idx], p_dict, (x,))

        return vmap(single_forward, randomness="different")(params_flat_batch, x_in_batch)

    def _create_teacher_forcing_mask(self, batch_size, seq_len, device, inference_start):
        """
        Create teacher forcing mask for autoregressive training.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Device to create mask on
            inference_start: If set, use deterministic forcing up to this fraction

        Returns:
            (batch_size, seq_len) bool tensor - True where ground truth should be used
        """
        use_ground_truth = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        if inference_start is not None:
            cutoff_index = int(inference_start * seq_len)
            use_ground_truth[:, :cutoff_index] = True
        else:
            if self.forcing_prob > 0:
                use_ground_truth = torch.bernoulli(
                    torch.full((batch_size, seq_len), self.forcing_prob, device=device)
                ).bool()

        return use_ground_truth

    def forward(self, xs, ts, inference_start=None):
        """ 
        xs: (Batch, Time, Data)
        ts: (Batch, Time, Dim)
        """
        if self.nb_classes is not None:
            return self.non_ar_call(xs, ts)
        
        # Regression Logic
        # For brevity, routing only to non_ar_call as per the example prompt focusing on that
        if self.ar_train or (inference_start is not None):
             # Placeholder for AR logic (similar structure but loop-based prediction)
             return self.ar_call(xs, ts) 
        else:
            return self.non_ar_call(xs, ts)

    def non_ar_call(self, xs, ts):
        """
        Non-Autoregressive Forward Pass using torch.func
        """
        batch_size, seq_len, _ = xs.shape
        
        # 1. Convolutional Embedding (if applicable)
        # Permute for Conv1d: (Batch, Time, Data) -> (Batch, Data, Time)
        xs_processed = xs
        if self.conv_embedding is not None:
            xs_perm = xs.permute(0, 2, 1) 
            xs_emb = self.conv_embedding(xs_perm) # (B, OutChans, Time)
            xs_processed = xs_emb.permute(0, 2, 1) # (B, Time, OutChans)
            
        # 2. Recurrence Scan (Theta Evolution)
        # We must loop sequentially over time to evolve theta, 
        # but we can do it efficiently.
        
        # Setup inputs for recurrence
        # delta_x = x_t - x_{t-1}
        # Prepend 0 or first element to get deltas
        xs_padded = torch.cat([torch.zeros_like(xs_processed[:, :1, :]), xs_processed[:, :-1, :]], dim=1)
        deltas = xs_processed - xs_padded # (Batch, Time, Dim)
        
        # Prepare A and B
        A = self.As[0] # (Latent, Latent)
        B = self.Bs[0] # (Latent, DataDim)
        
        # Initial Theta
        if isinstance(self.thetas_init[0], nn.Parameter):
            # Expand to batch size: (Batch, Latent)
            theta_curr = self.thetas_init[0].expand(batch_size, -1)
        else:
            # Predict from first input
            theta_curr = self.thetas_init[0](xs_processed[:, 0, :])
            
        if self.training and self.noise_theta_init is not None:
            theta_curr = theta_curr + torch.randn_like(theta_curr) * self.noise_theta_init

        # Run Scan
        thetas_over_time = []
        
        # If time_as_channel, concatenate t to input for B
        # Optimized loop
        for t in range(seq_len):
            x_in = deltas[:, t, :] # (Batch, Dim)
            t_in  = ts[:, t, :] - ts[:, max(0, t-1), :]
            
            if self.time_as_channel:
                 # Concatenate time diff or value? Original code did concat
                 # We simply concat t to the delta input
                 x_in = torch.cat([t_in, x_in], dim=-1)
                 

            # Update Rule: theta_next = theta @ A.T + x_in @ B.T
            # (Batch, Latent) = (Batch, Latent) @ (Latent, Latent)
            theta_next = theta_curr @ A.T + x_in @ B.T
            
            if self.weights_lim is not None:
                theta_next.clamp_(-self.weights_lim, self.weights_lim)
            
            thetas_over_time.append(theta_next)
            theta_curr = theta_next
            
        # Stack all thetas: (Batch, Time, Latent)
        thetas_stack = torch.stack(thetas_over_time, dim=1)
        
        # 3. Parallel Root Network Execution (The Optimization)
        # Instead of running root network inside the loop, we run it all at once using vmap.
        
        # Flatten Batch and Time dimensions to treat them as independent samples
        # (Batch * Time, Latent)
        N = batch_size * seq_len
        thetas_flat = thetas_stack.view(N, -1)
        
        # Prepare inputs for root network
        # root_in depends on input_prev_data config
        if self.input_prev_data:
            # Concat time + data
            root_in = torch.cat([ts, xs_processed], dim=-1)
        else:
            root_in = ts
            
        root_in_flat = root_in.reshape(N, -1)

        # -- VMAP Magic --
        # Map over the 0-th dimension of both arguments (N samples)
        # randomness="different" ensures if there's dropout, it varies per sample
        # Note: If memory is an issue, you can process in chunks, but vmap is usually efficient.
        output_flat = self._batch_root_forward(thetas_flat, root_in_flat)

        # Reshape back to (Batch, Time, OutDim)
        output = output_flat.view(batch_size, seq_len, -1)

        # 4. De-Embedding (if applicable)
        if self.conv_de_embedding is not None:
            # Apply conv separately to each feature dimension (vmap equivalent)
            # (Batch, Time, Out) -> apply conv to each feature independently
            batch_size, seq_len, out_dim = output.shape

            # Process each feature dimension separately
            de_embedded = []
            for d in range(out_dim):
                # Extract one feature: (Batch, Time) -> (Batch, 1, Time)
                feat = output[:, :, d].unsqueeze(1)
                # Apply conv: (Batch, 1, Time) -> (Batch, 1, Time)
                feat_conv = self.conv_de_embedding(feat)
                # Squeeze: (Batch, 1, Time) -> (Batch, Time)
                de_embedded.append(feat_conv.squeeze(1))

            # Stack back: list of (Batch, Time) -> (Batch, Time, Out)
            output = torch.stack(de_embedded, dim=-1)

        return output


    def ar_call(self, xs, ts, inference_start=None):
        """
        Autoregressive Forward Pass.
        xs: (Batch, Time, Data)
        ts: (Batch, Time, Dim)
        inference_start: Float (0.0 to 1.0) or None. 
                         If set, forces ground truth usage for that % of the sequence.
        """
        batch_size, seq_len, data_dim = xs.shape
        device = xs.device
        
        # 1. Setup Dynamics Matrices
        A = self.As[0] # (Latent, Latent)
        B = self.Bs[0] # (Latent, B_Input_Dim)
        
        # 2. Initialize State
        # Theta Init
        if isinstance(self.thetas_init[0], nn.Parameter):
            theta = self.thetas_init[0].expand(batch_size, -1) # (B, Latent)
        else:
            theta = self.thetas_init[0](xs[:, 0, :])
            
        if self.training and self.noise_theta_init is not None:
             theta = theta + torch.randn_like(theta) * self.noise_theta_init

        # Initialize "Previous" pointers
        # Corresponds to `x_prev` and `t_prev` in your JAX code
        x_prev_step = xs[:, 0, :] 
        t_prev = ts[:, 0, :]
        
        # Initialize "Prediction" pointer (x_hat)
        # Initially dummy or first frame
        x_hat = xs[:, 0, :]

        # 3. Pre-calculate Teacher Forcing Mask
        use_ground_truth = self._create_teacher_forcing_mask(batch_size, seq_len, device, inference_start)

        # 4. Preallocate output tensor
        output_dim = 2 * data_dim if self.predict_uncertainty else data_dim
        outputs = torch.empty(batch_size, seq_len, output_dim, device=device)

        # 5. Autoregressive Loop
        for t in range(seq_len):
            x_true = xs[:, t, :]
            t_curr = ts[:, t, :]
            
            # --- A. Determine Input for Theta Update ---
            # Logic: Use x_true if forcing=True, else use the x_hat from previous step
            forcing_mask = use_ground_truth[:, t].unsqueeze(-1) # (B, 1)
            x_input_for_update = torch.where(forcing_mask, x_true, x_hat)
            
            # --- B. Update Theta ---
            # theta_{t} = A * theta_{t-1} + B * (x_{input} - x_{prev_step})
            
            # Prepare Diff vector
            if self.time_as_channel:
                 # Concatenate time to data for the B matrix projection
                 x_curr_aug = torch.cat([x_input_for_update, t_curr], dim=-1)
                 x_prev_aug = torch.cat([x_prev_step, t_prev], dim=-1)
                 diff = x_curr_aug - x_prev_aug
            else:
                 diff = x_input_for_update - x_prev_step
                 
            # Matmul: (B, Latent) = (B, Latent)@A.T + (B, In)@B.T
            theta_next = theta @ A.T + diff @ B.T
            
            if self.weights_lim is not None:
                theta_next.clamp_(-self.weights_lim, self.weights_lim)
                
            # --- C. Predict Next Step (Root Network) ---
            # Prepare inputs for root network
            delta_t = t_curr - t_prev
            root_input_t = t_curr + delta_t
            
            if self.input_prev_data:
                # Root net sees the previous VALID data point (the one we selected in step A)
                root_input_vec = torch.cat([root_input_t, x_input_for_update], dim=-1)
            else:
                root_input_vec = root_input_t

            # Execute Batch of Networks in Parallel
            x_next_full = self._batch_root_forward(theta_next, root_input_vec)
            
            # Separate Mean from Std (if predict_uncertainty is True, output is 2x data_dim)
            x_next_mean = x_next_full[..., :data_dim]
            
            # --- D. Store & Update State ---
            outputs[:, t] = x_next_full

            theta = theta_next
            x_hat = x_next_mean          # This becomes the candidate prediction for the NEXT iteration
            x_prev_step = x_input_for_update # This becomes the x_prev for the NEXT iteration
            t_prev = t_curr

        # 6. Final Assembly
        return outputs
    
    def ar_call_stochastic(self, xs, ts, inference_start=None):
        """ 
        Autoregressive Stochastic Forward Pass.
        Predicts Mean & Std -> Samples -> Updates Theta -> Repeats.
        
        xs: (Batch, Time, Data)
        ts: (Batch, Time, Dim)
        inference_start: Float (0.0 to 1.0) or None.
        """
        batch_size, seq_len, data_dim = xs.shape
        device = xs.device

        # 1. Setup Dynamics
        A = self.As[0] # (Latent, Latent)
        B = self.Bs[0] # (Latent, B_Input_Dim)
        
        # 2. Initialize Theta
        if isinstance(self.thetas_init[0], nn.Parameter):
            theta = self.thetas_init[0].expand(batch_size, -1)
        else:
            theta = self.thetas_init[0](xs[:, 0, :])
            
        if self.training and self.noise_theta_init is not None:
             theta = theta + torch.randn_like(theta) * self.noise_theta_init

        # 3. Initialize Loop State
        # "Previous" inputs for the update equation
        x_prev_step = xs[:, 0, :] 
        t_prev = ts[:, 0, :]
        
        # Initial "Prediction" (x_mu_sigma)
        # We initialize with ground truth mean and 0 std for the very first step logic
        x_hat_mean = xs[:, 0, :]
        x_hat_std = torch.zeros_like(x_hat_mean)

        # 4. Teacher Forcing Mask Setup
        use_ground_truth = self._create_teacher_forcing_mask(batch_size, seq_len, device, inference_start)

        # 5. Preallocate output tensor
        output_dim = 2 * data_dim
        outputs = torch.empty(batch_size, seq_len, output_dim, device=device)

        # 6. Autoregressive Loop
        for t in range(seq_len):
            x_true = xs[:, t, :]
            t_curr = ts[:, t, :]
            
            # --- A. Sampling Step (Reparameterization) ---
            # 1. Sample from previous prediction distribution
            # If smooth_inference is True AND we are in free-running mode (not forcing), use Mean.
            # Otherwise, sample.
            
            should_sample = True
            if self.smooth_inference:
                 # If we are NOT forcing (pure inference), use mean only
                 # Note: self.smooth_inference is usually set during validation/test
                 should_sample = False

            if should_sample:
                noise = torch.randn_like(x_hat_mean)
                x_sampled = x_hat_mean + x_hat_std * noise
            else:
                x_sampled = x_hat_mean

            # --- B. Determine Input for Theta Update ---
            # Select between Ground Truth vs Sampled Prediction
            mask = use_ground_truth[:, t].unsqueeze(-1)
            x_input_for_update = torch.where(mask, x_true, x_sampled)

            # --- C. Update Theta ---
            # theta_{t} = A * theta_{t-1} + B * (x_{input} - x_{prev})
            
            if self.time_as_channel:
                 diff = torch.cat([x_input_for_update, t_curr], dim=-1) - \
                        torch.cat([x_prev_step, t_prev], dim=-1)
            else:
                 diff = x_input_for_update - x_prev_step

            theta_next = theta @ A.T + diff @ B.T
            
            if self.weights_lim is not None:
                theta_next.clamp_(-self.weights_lim, self.weights_lim)

            # --- D. Predict Next Distribution ---
            delta_t = t_curr - t_prev
            root_in = t_curr + delta_t
            
            if self.input_prev_data:
                # Concatenate the chosen input (true or sampled)
                root_in = torch.cat([root_in, x_input_for_update], dim=-1)
                
            # Run Batch of Roots
            x_next_distribution = self._batch_root_forward(theta_next, root_in)
            
            # Split Mean and Std
            # Assuming output_dim was doubled in initialization
            mean_next, std_next_raw = torch.chunk(x_next_distribution, 2, dim=-1)
            
            # Process Std (Softplus + Lower Bound)
            std_next = F.softplus(std_next_raw)
            if self.std_lower_bound is not None:
                std_next = torch.clamp(std_next, min=self.std_lower_bound)

            # Apply Final Activation to Mean if set
            if hasattr(self, 'dtanh_params') and self.dtanh_params is not None:
                 # Custom scaled tanh logic if needed
                 pass 

            # --- E. Store & Update State ---
            # Re-combine for output
            out_combined = torch.cat([mean_next, std_next], dim=-1)
            outputs[:, t] = out_combined

            # Update loop pointers
            theta = theta_next
            x_hat_mean = mean_next
            x_hat_std = std_next
            x_prev_step = x_input_for_update
            t_prev = t_curr

        # 7. Return output
        return outputs
    
        

    def tbptt_non_ar_call(self, xs, ts, num_chunks=4):
        """ 
        Forward pass with Truncated Backpropagation Through Time (TBPTT).
        Splits sequence into chunks and detaches gradients between them.
        
        Args:
            xs: (Batch, Time, Data)
            ts: (Batch, Time, Dim)
            num_chunks: int - Number of segments to split the sequence into.
        """
        batch_size, seq_len, data_dim = xs.shape
        
        # 1. Calculate Chunk Size
        # Ceiling division to determine size
        chunk_size = (seq_len + num_chunks - 1) // num_chunks
        
        # 2. Initialize State (Global)
        A = self.As[0]
        B = self.Bs[0]
        
        # Init Theta
        if isinstance(self.thetas_init[0], nn.Parameter):
            theta = self.thetas_init[0].expand(batch_size, -1)
        else:
            theta = self.thetas_init[0](xs[:, 0, :])
            
        if self.training and self.noise_theta_init is not None:
            theta = theta + torch.randn_like(theta) * self.noise_theta_init

        # Init x_prev (starts as the first input)
        x_prev = xs[:, 0, :]
        t_prev = ts[:, 0, :]
        
        all_outputs = []

        # 3. Chunk Loop (TBPTT)
        for chunk_idx, i in enumerate(range(0, seq_len, chunk_size)):
            # Define slice for this chunk
            end_idx = min(i + chunk_size, seq_len)
            
            # --- KEY TBPTT STEP ---
            # Detach gradients at the boundary of chunks (except the first one)
            if chunk_idx > 0:
                theta = theta.detach()
                # Note: We do NOT detach x_prev usually, as it's data, 
                # but if x_prev came from a prediction, we might. 
                # Here x_prev comes from ground truth xs (teacher forcing), so it requires no grad anyway.

            # Get Chunk Data
            xs_chunk = xs[:, i:end_idx, :]
            ts_chunk = ts[:, i:end_idx, :]
            chunk_len = xs_chunk.shape[1]
            
            chunk_thetas = []

            # 4. Intra-Chunk Recurrence Loop
            for t in range(chunk_len):
                x_curr = xs_chunk[:, t, :]
                t_curr = ts_chunk[:, t, :]
                
                # Setup Delta input
                if self.time_as_channel:
                    # Concatenate time
                    x_in = torch.cat([x_curr, t_curr], dim=-1)
                    x_p  = torch.cat([x_prev, t_prev], dim=-1)
                    diff = x_in - x_p
                else:
                    diff = x_curr - x_prev

                # Evolve Theta
                # theta_next = theta @ A.T + diff @ B.T
                theta_next = theta @ A.T + diff @ B.T

                if self.weights_lim is not None:
                    theta_next = torch.clamp(theta_next, -self.weights_lim, self.weights_lim)
                
                chunk_thetas.append(theta_next)
                
                # Update carry for next step
                theta = theta_next
                x_prev = x_curr
                t_prev = t_curr

            # 5. Parallel Root Network Execution for Chunk
            # Stack weights: (Batch, ChunkTime, Latent)
            chunk_thetas_stack = torch.stack(chunk_thetas, dim=1)
            
            # Flatten for vmap: (Batch * ChunkTime, Latent)
            N = batch_size * chunk_len
            theta_flat = chunk_thetas_stack.view(N, -1)
            
            # Prepare Inputs
            if self.input_prev_data:
                # Need to construct inputs carefully. 
                # In non-ar mode, input to root is usually (t, x_curr)
                root_in = torch.cat([ts_chunk, xs_chunk], dim=-1)
            else:
                root_in = ts_chunk
            
            root_in_flat = root_in.reshape(N, -1)

            # Run Batched Hypernetwork
            chunk_out_flat = self._batch_root_forward(theta_flat, root_in_flat)
            
            # Reshape back to (Batch, ChunkTime, Out)
            chunk_out = chunk_out_flat.view(batch_size, chunk_len, -1)
            
            all_outputs.append(chunk_out)

        # 6. Concatenate all chunks
        return torch.cat(all_outputs, dim=1)

    def _compute_linear_kernel(self, A, B, seq_len):
        """
        Computes the impulse response K = [B, AB, A^2B, ...]
        Ensures all created tensors match the device of A and B.
        """
        # Get the device from the input matrix A
        device = A.device 
        
        # Optimization: If A is Identity
        # We must specify device for torch.eye so it matches A
        is_identity = torch.allclose(A, torch.eye(A.shape[0], device=device))
        
        if is_identity:
             return B.unsqueeze(-1).repeat(1, 1, seq_len)

        # General Case: Compute powers
        K_list = []
        curr = B
        K_list.append(curr)
        
        for _ in range(seq_len - 1):
            curr = A @ curr
            K_list.append(curr)
            
        return torch.stack(K_list, dim=0).permute(1, 2, 0)

    def conv_call(self, xs, ts, fft=False):
        batch_size, seq_len, data_dim = xs.shape
        # Capture device from input
        device = xs.device 
        
        # 1. Prepare Input Signal
        # Zeros must be created on the correct device
        zeros_pad = torch.zeros_like(xs[:, :1, :], device=device)
        prev_x = torch.cat([zeros_pad, xs[:, :-1, :]], dim=1)
        input_signal = xs - prev_x
        
        if self.time_as_channel:
             input_signal = torch.cat([input_signal, ts], dim=-1)

        input_signal_perm = input_signal.permute(0, 2, 1)

        # 2. Compute Convolution Kernel
        # A and B are nn.Parameter, so they are already on the correct device 
        # (assuming model.to(device) was called)
        A = self.As[0]
        B = self.Bs[0]
        
        K_conv = self._compute_linear_kernel(A, B, seq_len)

        # 3. Convolution
        if not fft:
            padding_size = seq_len - 1
            input_padded = F.pad(input_signal_perm, (padding_size, 0))
            thetas_perm = F.conv1d(input_padded, K_conv)
            if thetas_perm.shape[2] > seq_len:
                thetas_perm = thetas_perm[:, :, :seq_len]
        else:
            n = input_signal_perm.shape[-1]
            m = K_conv.shape[-1]
            fft_len = n + m - 1
            input_f = torch.fft.rfft(input_signal_perm, n=fft_len, dim=-1)
            k_f = torch.fft.rfft(K_conv, n=fft_len, dim=-1).unsqueeze(0)
            input_f_expanded = input_f.unsqueeze(1) 
            prod = input_f_expanded * k_f 
            theta_f = torch.sum(prod, dim=2)
            thetas_perm = torch.fft.irfft(theta_f, n=fft_len, dim=-1)
            thetas_perm = thetas_perm[..., :seq_len]

        # 4. Add Bias / Init Theta
        if isinstance(self.thetas_init[0], nn.Parameter):
            theta_0 = self.thetas_init[0].unsqueeze(0).unsqueeze(-1)
        else:
            theta_0 = self.thetas_init[0](xs[:, 0, :]).unsqueeze(-1)
            
        thetas_perm = thetas_perm + theta_0
        thetas = thetas_perm.permute(0, 2, 1)

        # 5. Hypernetwork Execution
        N = batch_size * seq_len
        thetas_flat = thetas.reshape(N, -1)
        
        if self.input_prev_data:
            root_in = torch.cat([ts, xs], dim=-1)
        else:
            root_in = ts
        root_in_flat = root_in.reshape(N, -1)

        out_flat = self._batch_root_forward(thetas_flat, root_in_flat)
        xs_hat = out_flat.view(batch_size, seq_len, -1)

        return xs_hat


##################### BASE RNN MODEL DEFINITION ########################

class BaseRNN(nn.Module):
    """
    Base class for RNN models (GRU, LSTM) with shared forward logic.
    Eliminates ~90 lines of code duplication.
    """

    def __init__(self,
                 data_size: int,
                 hidden_size: int,
                 predict_uncertainty: bool = True,
                 time_as_channel: bool = True,
                 forcing_prob: float = 1.0,
                 std_lower_bound: float = None):
        super().__init__()

        self.data_size = data_size
        self.hidden_size = hidden_size
        self.time_as_channel = time_as_channel
        self.forcing_prob = forcing_prob
        self.predict_uncertainty = predict_uncertainty
        self.std_lower_bound = std_lower_bound

        input_dim = 1 + data_size if time_as_channel else data_size
        output_dim = 2 * data_size if predict_uncertainty else data_size

        # Subclasses must implement this
        self.cell = self._create_cell(input_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_dim)

    def _create_cell(self, input_dim, hidden_size):
        """Override in subclass to return appropriate RNN cell."""
        raise NotImplementedError("Subclass must implement _create_cell")

    def _init_hidden(self, batch_size, device):
        """Override in subclass to return initial hidden state(s)."""
        raise NotImplementedError("Subclass must implement _init_hidden")

    def _cell_forward(self, x_t, hidden):
        """Override in subclass to handle cell-specific forward pass."""
        raise NotImplementedError("Subclass must implement _cell_forward")

    def forward(self, xs, ts, inference_start=None):
        """
        Forward pass of the model on batch of sequences
        xs: (batch, time, data_size)
        ts: (batch, time, dim)
        inference_start: whether/when to use the model in autoregressive mode
        """
        batch_size, seq_len, data_dim = xs.shape
        device = xs.device

        # Initialize hidden state
        hidden = self._init_hidden(batch_size, device)
        x_hat = xs[:, 0, :]

        # Teacher forcing mask
        use_ground_truth = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        if inference_start is not None:
            cutoff_index = int(inference_start * seq_len)
            use_ground_truth[:, :cutoff_index] = True
        else:
            if self.forcing_prob > 0:
                use_ground_truth = torch.bernoulli(
                    torch.full((batch_size, seq_len), self.forcing_prob, device=device)
                ).bool()

        # Preallocate output tensor
        output_dim = 2 * data_dim if self.predict_uncertainty else data_dim
        outputs = torch.empty(batch_size, seq_len, output_dim, device=device)

        for t in range(seq_len):
            x_true = xs[:, t, :]
            t_curr = ts[:, t, :]

            # Determine input
            forcing_mask = use_ground_truth[:, t].unsqueeze(-1)
            x_t = torch.where(forcing_mask, x_true, x_hat)

            if self.time_as_channel:
                x_t = torch.cat([t_curr[:, :1], x_t], dim=-1)

            # RNN step (cell-specific)
            hidden = self._cell_forward(x_t, hidden)

            # Decode hidden state
            h = hidden if isinstance(hidden, torch.Tensor) else hidden[0]
            x_next = self.decoder(h)

            x_next_mean = x_next[:, :data_dim]
            if self.predict_uncertainty:
                x_next_std = x_next[:, data_dim:]
                if self.std_lower_bound is not None:
                    x_next_std = torch.clamp(x_next_std, min=self.std_lower_bound)
                x_next = torch.cat([x_next_mean, x_next_std], dim=-1)

            outputs[:, t] = x_next
            x_hat = x_next_mean

        return outputs


##################### GRU MODEL DEFINITION ########################

class GRU(BaseRNN):
    """ Gated Recurrent Unit """

    def _create_cell(self, input_dim, hidden_size):
        return nn.GRUCell(input_dim, hidden_size)

    def _init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def _cell_forward(self, x_t, hidden):
        return self.cell(x_t, hidden)


##################### LSTM MODEL DEFINITION ########################

class LSTM(BaseRNN):
    """ Long Short-Term Memory """

    def _create_cell(self, input_dim, hidden_size):
        return nn.LSTMCell(input_dim, hidden_size)

    def _init_hidden(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return (h, c)

    def _cell_forward(self, x_t, hidden):
        return self.cell(x_t, hidden)


##################### FEED-FORWARD MODEL DEFINITION ########################

class FFNN(nn.Module):
    """ Feed-Forward Neural Network """

    def __init__(self, data_size: int, sequence_length: int = 699):
        super().__init__()

        self.data_size = data_size
        self.sequence_length = sequence_length

        num_layers = 3
        width_size = 1024

        layers = []
        for i in range(num_layers):
            in_features = sequence_length if i == 0 else width_size
            out_features = sequence_length if i == num_layers - 1 else width_size

            layers.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, xs, ts, inference_start=None):
        """
        Forward pass of the model on batch
        xs: (batch, time, data_size)
        ts: (batch, time, dim) - not used
        inference_start: not used
        """
        # Squeeze the last dimension and pass through network
        # (batch, time) -> (batch, time) -> (batch, time, 1)
        x = xs.squeeze(-1)
        x = self.network(x)
        return x.unsqueeze(-1)


##################### PUTTING THINGS TOGETHER ########################

def make_model(data_size, nb_classes, config, logger):
    """ Make a model using the given config """

    model_type = config['model']['model_type']

    if model_type == "wsm":
        model_args = {
            "data_size": data_size,
            "width_size": config['model']['root_width_size'],
            "depth": config['model']['root_depth'],
            "activation": config['model']['root_activation'],
            "final_activation": config['model']['root_final_activation'],
            "nb_classes": nb_classes,
            "init_state_layers": config['model']['init_state_layers'],
            "input_prev_data": config['model']['input_prev_data'],
            "predict_uncertainty": config['training']['use_nll_loss'],
            "time_as_channel": config['model']['time_as_channel'],
            "forcing_prob": config['model']['forcing_prob'],
            "std_lower_bound": config['model']['std_lower_bound'],
            "weights_lim": config['model']['weights_lim'],
            "noise_theta_init": config['model']['noise_theta_init'],
            "autoregressive_train": config['training']['autoregressive'],
            "stochastic_ar": config['training']['stochastic'],
            "positional_encoding": config['model'].get('positional_encoding', False),
            "preferred_output_dim": config['model'].get('root_output_dim', None),
            "conv_embedding": config['model'].get('conv_embedding', None),
        }

        if 'smooth_inference' in config['training']:
            model_args['smooth_inference'] = config['training']['smooth_inference']

        if model_args['autoregressive_train'] and model_args['stochastic_ar'] and not config['training']['use_nll_loss']:
            raise ValueError("The WSM model is not compatible with stochastic autoregressive training without NLL loss.")

        if not model_args.get('smooth_inference', True) and not model_args['stochastic_ar']:
            raise ValueError("Non-smooth (stochastic, reparametrization trick) inference cannot be used without stochastic training.")

        model = WSM(**model_args)

        # Log parameter counts
        if isinstance(model.thetas_init[0], nn.Parameter):
            logger.info(f"Number of weights in the root network: {model.thetas_init[0].numel()/1000:3.1f} k")
        else:
            # It's a GradualMLP or similar
            total_params = sum(p.numel() for p in model.thetas_init[0].parameters())
            logger.info(f"Number of learnable parameters in the initial hyper-network: {total_params/1000:3.1f} k")

        A_params = sum(p.numel() for p in model.As)
        B_params = sum(p.numel() for p in model.Bs)
        logger.info(f"Number of learnable parameters for the recurrent transition: {(A_params + B_params)/1000:3.1f} k")

    elif model_type == "lstm":
        model_args = {
            "data_size": data_size,
            "predict_uncertainty": config['training']['use_nll_loss'],
            "time_as_channel": config['model']['time_as_channel'],
            "forcing_prob": config['model']['forcing_prob'],
            "std_lower_bound": config['model']['std_lower_bound'],
            "hidden_size": config['model']['rnn_hidden_size'],
        }
        model = LSTM(**model_args)

    elif model_type == "gru":
        model_args = {
            "data_size": data_size,
            "predict_uncertainty": config['training']['use_nll_loss'],
            "time_as_channel": config['model']['time_as_channel'],
            "forcing_prob": config['model']['forcing_prob'],
            "std_lower_bound": config['model']['std_lower_bound'],
            "hidden_size": config['model']['rnn_hidden_size'],
        }
        model = GRU(**model_args)

    elif model_type == "ffnn":
        model_args = {
            "data_size": data_size,
        }
        model = FFNN(**model_args)

    # Log total parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of learnable parameters in the model: {total_params/1000:3.1f} k")

    return model