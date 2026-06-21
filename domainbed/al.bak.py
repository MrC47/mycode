class MyModel(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MyModel, self).__init__(input_shape, num_classes, num_domains, hparams)

        assert num_domains > 0, "Number of domains must be greater than 0"

        self.num_domains = num_domains
        self.input_shape = input_shape # (3, 224, 224)
        self.backbone_type = self._resolve_backbone_type()

        self.causal_extractor = networks.Featurizer(input_shape, self.hparams)
        self.num_channels= self.causal_extractor.n_outputs # 因果和私有特征提取器是同类型的网络，故输出通道数一致，统一用num_channels。
        self.shared_private_extractor = networks.Featurizer(input_shape, self.hparams)
        self.private_heads = nn.ModuleList([
            networks.PrivateHead(self.num_channels, hparams) for _ in range(num_domains)
        ])
        self.gate = nn.Sequential(
            nn.Linear(self.num_channels, self.num_channels),
            nn.ReLU(),
            nn.Linear(self.num_channels, self.num_channels),
            nn.Sigmoid()
        )

        self.decoder = networks.Decoder(self.num_channels * 2, self.input_shape, self.hparams)

        # Mamba2 融合模块：将拼接后的因果+私有特征通过 SSM 进行序列级融合
        # in_channels = num_channels * 2 因为因果特征和私有特征按通道拼接后维度翻倍
        self.mamba_fusion = networks.MambaFusionBlock(
            in_channels=self.num_channels * 2,
            d_model=256,
            d_state=64,
            headdim=64,
            expand=2,
        )

        self.classifier = networks.Classifier(
            self.num_channels,
            num_classes,
            self.hparams.get('nonlinear_classifier', False)
        )

        self.register_buffer('prototypes', torch.zeros(num_domains, self.num_channels))
        self.register_buffer('update_count', torch.tensor([0]))
       
        # GradNorm: task_weights
        self.task_names = ['irm', 'vrex', 'ort', 'reco', 'energy']
        self.task_weights = nn.Parameter(torch.ones(len(self.task_names)))
        self.initial_losses = None  # 初始化为 None

        self._setup_optimizer()

    def _setup_optimizer(self):
        params = list(self.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def _resolve_backbone_type(self):
        return networks._resolve_backbone_name(self.hparams)

    def _is_transformer_backbone(self):
        return self.backbone_type in ["vit", "dinov2"]

    def _pool_features(self, features):
        # Compatible ResNet/CNN: [B, C, H, W] -> [B, C]
        if features.dim() == 4:
            return features.mean(dim=[2, 3])
       
        # Compatible ViT: [B, 197, 768] -> [B, 768]
        if features.dim() == 3:
            if self._is_transformer_backbone() and features.size(1) > 1:
                return features[:, 0]
            return features.mean(dim=1)
            
        return features

    def _build_latent_for_decoder(self, causal_features_raw, private_cat):
        if causal_features_raw.dim() != private_cat.dim():
            raise ValueError(
                f"Feature rank mismatch: causal={causal_features_raw.dim()}D, private={private_cat.dim()}D"
            )

        if causal_features_raw.dim() == 4:
            return torch.cat([causal_features_raw, private_cat], dim=1)
        if causal_features_raw.dim() == 3:
            # One-to-one backbone pairing:
            # ViT/DINOv2: concat on channel dim; CNN-token backbones: concat on token dim.
            concat_dim = 2 if self._is_transformer_backbone() else 1
            return torch.cat([causal_features_raw, private_cat], dim=concat_dim)
        if causal_features_raw.dim() == 2:
            return torch.cat([causal_features_raw, private_cat], dim=1)

        raise ValueError(f"Unsupported feature rank for decoder input: {causal_features_raw.dim()}D")
    
    def _get_feature_for_attention(self, features):
        if features.dim() == 3:
            return features
        return features.unsqueeze(1)

    def loss_erm(self, logits, labels):
        return F.cross_entropy(logits, labels)
   
    def loss_irm(self, logits, labels):
        return IRM._irm_penalty(logits, labels)
   
    def loss_vrex(self, logits_list, labels_list):
        losses = torch.stack([F.cross_entropy(logits, labels)
                             for logits, labels in zip(logits_list, labels_list)])
        penalty = losses.var(unbiased=False)
        return penalty
   
    def loss_ort(self, private_features_list, causal_features_raw):
        # 1. 提取并池化特征
        f_causal = self._pool_features(causal_features_raw) # [B_total, D]
        
        total_diff_loss = torch.tensor(0.0).to(f_causal.device)
        start_idx = 0
        
        for f_priv in private_features_list:
            batch_size = f_priv.size(0)
            # 对应当前域的因果特征
            f_c_part = f_causal[start_idx : start_idx + batch_size]
            f_s_part = self._pool_features(f_priv)
            
            # 2. 【DSN 标准 Difference Loss】
            # 计算特征矩阵的乘积：[D, B] * [B, D] -> [D, D]
            # 这衡量了特征维度之间的相关性，符合 DSN 原始定义的矩阵范数约束
            correlation_matrix = torch.matmul(f_c_part.t(), f_s_part)
            
            # 3. 计算 Frobenius 范数的平方，并归一化
            # 这种方式比单个样本的点积要“软”，因为它是在优化整个 Batch 的相关性分布
            diff_loss = torch.mean(correlation_matrix ** 2)
            
            total_diff_loss += diff_loss
            start_idx += batch_size
            
        return total_diff_loss / len(private_features_list)
   
    def loss_reco(self, reconstructed, original):
        return F.mse_loss(reconstructed, original)
   
    def get_energy_weights(self, private_features_list, domain_indices, T=1.0, alpha=0.8):
        device = self.prototypes.device
        if len(private_features_list) == 0:
            return torch.ones(len(domain_indices), device=device)
        domain_energies = []
        all_sample_energies = []
       
        for idx, f_priv_raw in zip(domain_indices, private_features_list):
            # --- A. 特征标准化 (适配 ViT/CNN 并去除模长干扰) ---
            # 如果是 ViT 的多 token 输出，取 [CLS] 或平均；如果是 CNN，做 Global Average Pooling
            f_priv = self._pool_features(f_priv_raw)
            f_priv = F.normalize(f_priv, p=2, dim=1) # 投影到单位球面上，防止距离爆炸
           
            # --- B. 局部原型与全局原型更新 ---
            batch_prototype = f_priv.mean(dim=0)
           
            with torch.no_grad():
                # 冷启动保护：如果原型为全0（刚开始训练），直接复制
                if self.prototypes[idx].abs().sum() == 0:
                    self.prototypes[idx].copy_(batch_prototype)
                else:
                    # EMA 更新：维持该领域长期稳定的“熟悉分布”中心
                    new_proto = alpha * self.prototypes[idx] + (1 - alpha) * batch_prototype
                    self.prototypes[idx].copy_(new_proto)
            # --- C. 计算能量值 (Energy Score) ---
            # 使用 1 - Cosine Similarity。值域 [0, 2]，数值极其稳定。
            # 意义：当前 Batch 特征偏离历史中心的角度越大，能量越高，代表越陌生。
            target_proto = F.normalize(self.prototypes[idx].detach().unsqueeze(0), p=2, dim=1)
            # 计算该 Batch 所有样本到原型的平均距离
            sample_energies = 1.0 - torch.sum(f_priv * target_proto, dim=1)
            domain_energies.append(sample_energies.mean())
            all_sample_energies.append(sample_energies)
        # --- D. 能量转权重 (带量级重平衡) ---
        energy_tensor = torch.stack(domain_energies)
       
        # 1. 基础权重：Softmax 分配（和为 1）
        # T 为温度，T 越小，对“陌生域”的扶持力度越大
        raw_weights = F.softmax(energy_tensor.detach() / T, dim=0)
       
        # 2. 梯度重平衡：乘上参与计算的域数量
        # 目的：让权重的平均值回到 1.0 附近，确保 Total Loss 的量级不因 Softmax 而坍缩
        # 这样你的不确定性加权参数 (log_sigma) 才能在正常的数值区间工作
        dynamic_weights = raw_weights * len(private_features_list)
       
        return dynamic_weights, all_sample_energies
    
    def loss_energy(self, energy, gamma=1.0):
        l_mean = energy.mean()
        l_var = energy.var(unbiased=False)
        return l_mean + gamma * l_var
   
    def update(self, minibatches, unlabeled=None):
        device = "cuda" if torch.cuda.is_available() and minibatches[0][0].is_cuda else "cpu"
        if len(minibatches) != self.num_domains:
            raise ValueError(f"Mismatched environment count: expected {self.num_domains}, got {len(minibatches)}")
        # private_encoder的一次性推理版本。
        all_x = [x for x, y in minibatches]
        all_y = [y for x, y in minibatches]
        all_x_cat = torch.cat(all_x)
        all_y_cat = torch.cat(all_y)

        causal_features_raw = self.causal_extractor(all_x_cat)
        shared_priv_all = self.shared_private_extractor(all_x_cat)

        private_features_list = []
        domain_indices = []
        start_idx = 0
        for env_idx, x in enumerate(all_x):
            batch_size = x.size(0)
            end_idx = start_idx + batch_size
           
            # 从全量特征中切出属于当前域的部分
            env_priv_base = shared_priv_all[start_idx:end_idx]
           
            # 通过专用头 (BN + Adapter)
            p_feat = self.private_heads[env_idx](env_priv_base)
           
            private_features_list.append(p_feat)
            domain_indices.append(env_idx)
            start_idx = end_idx

        dynamic_weights, all_sample_energies = self.get_energy_weights(private_features_list, domain_indices)
        if len(private_features_list) > 0:
            private_cat = torch.cat(private_features_list)
        else:
            private_cat = causal_features_raw

        f_c = self._pool_features(causal_features_raw)
        latent_for_reco = self._build_latent_for_decoder(causal_features_raw, private_cat)
        # 通过 Mamba2 对拼接后的因果+私有特征进行序列级融合
        # Mamba 会沿空间/token 序列维度捕获特征间的长程依赖关系
        latent_for_reco = self.mamba_fusion(latent_for_reco)
        logits = self.classifier(f_c)
        reconstructed = self.decoder(latent_for_reco)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
        original_images = all_x_cat * std + mean
        original_images = torch.clamp(original_images, 0, 1)

        logits_list = []
        labels_list = []
        start_idx = 0
        for (x, y) in zip(all_x, all_y):
            end_idx = start_idx + x.size(0)
            env_logits = logits[start_idx:end_idx]
            logits_list.append(env_logits)
            labels_list.append(y)
            start_idx = end_idx
           
        l_erm = self.loss_erm(logits, all_y_cat)
        l_irm = self.loss_irm(logits, all_y_cat)
        l_vrex = self.loss_vrex(logits_list, labels_list)
        l_ort = self.loss_ort(private_features_list, causal_features_raw)
        l_reco = self.loss_reco(reconstructed, original_images)
        l_energy = self.loss_energy(torch.cat(all_sample_energies), gamma=1.0)
       

        # --- GradNorm 核心逻辑 (优化版) ---
        # 仅在训练稳定后开启 (例如 step > 100)，或者每 N 步更新一次
        if self.update_count > 100 and self.update_count % 10 == 0:
           
            shared_params = list(self.causal_extractor.parameters())[-2:]
            # 1. 计算各任务的梯度范数 (G_i)
            task_norms = []
            # 这里的 losses 字典需要包含加权后的 loss 还是原始 loss?
            # GradNorm 原理是平衡 "加权后 Loss" 对参数的梯度。
            # 所以这里必须用 self.task_weights[i] * raw_loss
           
            # 重新构建带当前权重的 losses 用于求导
            weighted_losses = [
                self.task_weights[0] * l_irm,
                self.task_weights[1] * l_vrex,
                self.task_weights[2] * l_ort,
                self.task_weights[3] * l_reco,
                self.task_weights[4] * l_energy
            ]
           
            for wl in weighted_losses:
                # retain_graph=True 是必须的，因为后面还要做真正的 backward
                gs = torch.autograd.grad(wl, shared_params, retain_graph=True, allow_unused=True)
                # 计算 L2 范数
                valid_grads = [torch.norm(g.detach(), p=2) for g in gs if g is not None]
                if len(valid_grads) > 0:
                    n = torch.norm(torch.stack(valid_grads))
                else:
                    n = torch.tensor(1e-6).to(device)
                task_norms.append(n)
           
            task_norms = torch.stack(task_norms)  # [G_irm, G_vrex, G_ort, G_reco]
            # 2. 计算参考梯度范数 (G_avg) - 这里用 ERM 的梯度作为锚点
            grads_erm = torch.autograd.grad(l_erm, shared_params, retain_graph=True, allow_unused=True)
            norm_erm = torch.norm(torch.stack([torch.norm(g.detach(), p=2) for g in grads_erm if g is not None]))
           
            # 或者使用所有任务梯度的平均值作为锚点 (GradNorm 论文原意)
            mean_norm = torch.mean(task_norms)  # 也可以用 norm_erm 代替，看你想让谁主导
            # 3. 计算相对逆训练速率 (Inverse Training Rate) - 可选，这里简化为纯梯度平衡
            # 如果不计算 Loss 里的下降速率 r_i，直接平衡梯度：
            # 目标：希望 G_i 接近 mean_norm
           
            target_ratios = norm_erm / (task_norms + 1e-6)
           
            # 4. 动量更新权重 (关键：防止震荡)
            # 使用 detach() 确保不反向传播给权重自己
            new_weights = 0.95 * self.task_weights.detach() + 0.05 * target_ratios
           
            # 5. 重归一化 (Renormalization)
            # 保持权重的总和不变（例如总和为 4），防止所有权重同时无限变大
            normalize_coeff = 5.0 / (new_weights.sum() + 1e-6)
            new_weights = new_weights * normalize_coeff
           
            # 6. 赋值与截断
            new_weights = torch.clamp(new_weights, 0.02, 10.0)  # 放宽上限
            self.task_weights.data.copy_(new_weights)

        # 最终总损失计算
        total_loss = l_erm + \
                    self.task_weights[0] * l_irm + \
                    self.task_weights[1] * l_vrex + \
                    self.task_weights[2] * l_ort + \
                    self.task_weights[3] * l_reco + \
                    self.task_weights[4] * l_energy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.update_count += 1
        return {
            'loss': total_loss.item(),
            'l_erm': l_erm.item(),
            'l_irm': l_irm.item() if isinstance(l_irm, torch.Tensor) else l_irm,
            'l_vrex': l_vrex.item() if isinstance(l_vrex, torch.Tensor) else l_vrex,
            'l_ort': l_ort.item() if isinstance(l_ort, torch.Tensor) else l_ort,
            'l_reco': l_reco.item(),
            'l_energy':l_energy.item(),
            'w_irm': self.task_weights[0].item(),
            'w_vrex': self.task_weights[1].item(),
            'w_ort': self.task_weights[2].item(),
            'w_reco': self.task_weights[3].item(),
            'w_energy': self.task_weights[4].item(),
        }
   
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            f_c_raw = self.causal_extractor(x)
            f_c_vec = self._pool_features(f_c_raw)
            
            return self.classifier(f_c_vec)