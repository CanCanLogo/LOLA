import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm  # 添加权重归一化支持

class MultiLevelMLP(nn.Module):
    """多层级MLP系统 - 用于生成三个层级的prompt专家路由权重"""
    def __init__(self, embed_dim=96, n_experts=8, d_moe_low=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_experts = n_experts
        
        # 修改处：先不初始化具体的Linear层，在第一次forward时动态创建
        self.d_moe_low = d_moe_low
        self.initialized = False
        
        # 存储维度信息，在首次forward时设置
        self.low_feat_dim = None
        self.med_feat_dim = embed_dim
        self.high_feat_dim = embed_dim
        self.total_feat_dim = None
        
    def _initialize_networks(self, total_feat_dim, low_feat_dim, device):
        """首次forward时动态初始化网络"""
        self.total_feat_dim = total_feat_dim
        self.low_feat_dim = low_feat_dim
    
        print(f"DEBUG - Initializing MLPs with total_feat_dim: {total_feat_dim}, low_feat_dim: {low_feat_dim}")
    
        # 修改处：确定设备位置
        # 从当前模块的已有参数获取设备，如果没有参数则使用cuda
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # 动态创建MLP网络并直接放到正确设备上，使用weight_norm稳定训练
        self.mlp_low = nn.Sequential(
            weight_norm(nn.Linear(total_feat_dim, self.embed_dim * 2)),  
            nn.GELU(),
            nn.Dropout(0.1),
            weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 4, self.embed_dim * 2)),  
            # nn.GELU(),
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),
            # nn.GELU(), 
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            weight_norm(nn.Linear(self.embed_dim, self.n_experts)),
            nn.Softmax(dim=-1)
        ).to(device)  # 修改处：直接创建时就放到正确设备

        self.mlp_medium = nn.Sequential(
            weight_norm(nn.Linear(total_feat_dim, self.embed_dim * 2)),  
            nn.GELU(),
            nn.Dropout(0.1),
            weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 4, self.embed_dim * 2)),  
            # nn.GELU(),
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),
            # nn.GELU(), 
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            weight_norm(nn.Linear(self.embed_dim, self.n_experts)),
            nn.Softmax(dim=-1)
        ).to(device)  # 修改处：直接创建时就放到正确设备

        self.mlp_high = nn.Sequential(
            weight_norm(nn.Linear(total_feat_dim, self.embed_dim * 2)),  
            nn.GELU(),
            nn.Dropout(0.1),
            weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 4, self.embed_dim * 2)),  
            # nn.GELU(),
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),
            # nn.GELU(), 
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            weight_norm(nn.Linear(self.embed_dim, self.n_experts)),
            nn.Softmax(dim=-1)
        ).to(device)  # 修改处：直接创建时就放到正确设备

        # 重构网络
        self.reconstructor_low = nn.Sequential(
            weight_norm(nn.Linear(self.n_experts, self.embed_dim * 2)),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),  # 输出维度为低层级特征维度的两倍
            nn.GELU(),
            nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 4, self.embed_dim * 2)),
            # nn.GELU(),
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),
            # nn.GELU(),
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            nn.Linear(self.embed_dim, low_feat_dim)
            ).to(device)  # 修改处：直接创建时就放到正确设备

        self.reconstructor_med = nn.Sequential(
            weight_norm(nn.Linear(self.n_experts, self.embed_dim * 2)),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),  # 输出维度为低层级特征维度的两倍
            nn.GELU(),
            nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 4, self.embed_dim * 2)),
            # nn.GELU(),
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),
            # nn.GELU(),
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合 
            nn.Linear(self.embed_dim, self.med_feat_dim)
        ).to(device)  # 修改处：直接创建时就放到正确设备

        self.reconstructor_high = nn.Sequential(
            weight_norm(nn.Linear(self.n_experts, self.embed_dim * 2)),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),  # 输出维度为低层级特征维度的两倍
            nn.GELU(),
            nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 4, self.embed_dim * 2)),
            # nn.GELU(),
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            # weight_norm(nn.Linear(self.embed_dim * 2, self.embed_dim)),
            # nn.GELU(),
            # nn.Dropout(0.1),  # 添加Dropout层以防止过拟合
            nn.Linear(self.embed_dim, self.high_feat_dim)
        ).to(device)  # 修改处：直接创建时就放到正确设备
    
        print(f"DEBUG - Networks created on device: {device}")  # 修改处：添加设备确认
    
        # 初始化权重
        self._init_weights()
        self.initialized = True
        
    def _init_weights(self):
        """权重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, feat_low, feat_med, feat_high):
        """
        Args:
            feat_low: [B, low_feat_dim] 低层级特征 - 图文投影拼接
            feat_med: [B, embed_dim] 中层级特征 - CNN池化
            feat_high: [B, embed_dim] 高层级特征 - CNN池化
        Returns:
            dict: 包含路由分数和中间特征的字典
        """
        B = feat_low.shape[0]
    
        # 修改处：拼接所有层级特征
        concat_features = torch.cat([feat_low, feat_med, feat_high], dim=-1)
        
        # 修改处：对拼接后的特征进行归一化，避免数值过大导致不稳定
        concat_features = F.normalize(concat_features, dim=-1)
    
        # 修改处：首次forward时动态初始化，从输入张量获取设备
        if not self.initialized:
            total_feat_dim = concat_features.shape[-1]
            low_feat_dim = feat_low.shape[-1]
            input_device = feat_low.device  # 从输入获取设备
            self._initialize_networks(total_feat_dim, low_feat_dim, input_device)
    
        # 跨层级任务
        low_scores_cross = self.mlp_low(concat_features)
        med_scores_cross = self.mlp_medium(concat_features)
        high_scores_cross = self.mlp_high(concat_features)
    
        # 自身层级任务
        noise_scale = 0.1
    
        # 低层级自身任务
        low_self_input = torch.cat([
            feat_low,
            torch.randn(B, self.med_feat_dim, device=feat_low.device) * noise_scale,
            torch.randn(B, self.high_feat_dim, device=feat_low.device) * noise_scale
        ], dim=-1)
        low_scores_self = self.mlp_low(low_self_input)
    
        # 中层级自身任务
        med_self_input = torch.cat([
            torch.randn(B, self.low_feat_dim, device=feat_med.device) * noise_scale,
            feat_med,
            torch.randn(B, self.high_feat_dim, device=feat_med.device) * noise_scale
        ], dim=-1)
        med_scores_self = self.mlp_medium(med_self_input)
    
        # 高层级自身任务
        high_self_input = torch.cat([
            torch.randn(B, self.low_feat_dim, device=feat_high.device) * noise_scale,
            torch.randn(B, self.med_feat_dim, device=feat_high.device) * noise_scale,
            feat_high
        ], dim=-1)
        high_scores_self = self.mlp_high(high_self_input)
    
        return {
            'low_cross': low_scores_cross,
            'med_cross': med_scores_cross, 
            'high_cross': high_scores_cross,
            'low_self': low_scores_self,
            'med_self': med_scores_self,
            'high_self': high_scores_self,
            'features': {
                'low': feat_low,
                'med': feat_med, 
                'high': feat_high,
                'concat': concat_features
            }
        }

class FeatureLossComputer(nn.Module):
    """特征损失计算器 - 实现feature_crossing_loss和feature_self_loss"""
    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 修改处：更保守的超参数设置，防止梯度爆炸
        self.lambda1 = 0.1   # 多样性损失权重 (从0.3降到0.1)
        self.lambda2 = 0.05  # 平衡损失权重 (从0.2降到0.05)
        self.gamma1 = 0.2    # 专注损失权重 (从0.4降到0.2)
        self.gamma2 = 0.1    # 一致性损失权重 (从0.3降到0.1)
        self.tau = 1.0      # 多样性阈值 (从1.0降到0.5)
        
        # 修改处：可学习的温度参数，初始值更大以增加稳定性
        self.temperature = nn.Parameter(torch.tensor(1.0))  # 从0.07提高到1.0
        
    def feature_crossing_loss(self, features):
        """
        重新定义跨层级协调损失：
        - 仅包含多样性损失，防止不同层级特征趋同
        - 对输入特征 detach，避免梯度回传到提取器
        L_cross = λ * diversity_loss
        """
        # detach 特征以阻断梯度传播到提取器
        feat_low = features['low'].detach()
        feat_med = features['med'].detach()
        feat_high = features['high'].detach()

        # 特征维度不一致，先将低维特征自适应池化到中维度(embed_dim)，保证三者维度一致
        # 这里 med_feat_dim == high_feat_dim == embed_dim
        med_feat_dim = self.embed_dim
        feat_low_proj = F.adaptive_avg_pool1d(feat_low.unsqueeze(1), med_feat_dim).squeeze(1)
        # 计算多样性损失：惩罚过度相似的特征对（L1距离）
        d_lm = torch.mean(torch.abs(feat_low_proj - feat_med))
        d_lh = torch.mean(torch.abs(feat_low_proj - feat_high))
        d_mh = torch.mean(torch.abs(feat_med - feat_high))
        diversity_loss = (F.softplus(self.tau - d_lm) + F.softplus(self.tau - d_lh) + F.softplus(self.tau - d_mh)) / 3

        # 加权输出 diversity loss
        total_crossing_loss = self.lambda1 * diversity_loss
         
        # 计算跨层级协调损失 L_crossing - 防止梯度爆炸的稳定版本
        
        # 数学定义：
        # L_crossing = L_correlation + λ₁ * L_diversity + λ₂ * L_balance
        
        # Args:
        #     features: dict包含'low', 'med', 'high'三个层级的特征
        # Returns:
        #     torch.Tensor: 标量损失值
        # """
        # feat_low = features['low']    # [B, embed_dim]
        # feat_med = features['med']    # [B, embed_dim]  
        # feat_high = features['high']  # [B, embed_dim]
        
        # # 添加梯度检查和输出函数
        # def check_tensor_stability(tensor, name, print_details=False):
        #     """检查张量的数值稳定性"""
        #     if torch.isnan(tensor).any():
        #         print(f"❌ 警告: {name} 包含NaN值")
        #         return False
        #     if torch.isinf(tensor).any():
        #         print(f"❌ 警告: {name} 包含无穷大值")
        #         return False
        #     if tensor.abs().max() > 1e6:
        #         print(f"❌ 警告: {name} 包含过大值，最大值: {tensor.abs().max()}")
        #         return False
        #     if print_details:
        #         print(f"✅ {name}: min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.mean():.6f}, std={tensor.std():.6f}")
        #     return True

        # # 输入特征预处理 - 强制梯度裁剪和归一化
        # def preprocess_features(x, name):
        #     """预处理特征，确保数值稳定"""
        #     # 梯度裁剪
        #     x = torch.clamp(x, -10.0, 10.0)
        #     # L2归一化
        #     x = F.normalize(x, p=2, dim=-1, eps=1e-8)
        #     # 再次梯度裁剪，防止归一化后的异常
        #     x = torch.clamp(x, -1.0, 1.0)
        #     # check_tensor_stability(x, f"preprocessed_{name}")
        #     return x
        
        # # 预处理所有特征
        # feat_low = preprocess_features(feat_low, "feat_low")
        # feat_med = preprocess_features(feat_med, "feat_med") 
        # feat_high = preprocess_features(feat_high, "feat_high")
        
        # # 稳定的余弦相似度计算（替代Pearson相关系数）
        # def stable_cosine_similarity(x, y):
        #     """
        #     使用余弦相似度替代Pearson相关系数，更加稳定
        #     """
        #     # 处理维度不匹配
        #     if x.shape[-1] != y.shape[-1]:
        #         min_dim = min(x.shape[-1], y.shape[-1])
        #         x = x[:, :min_dim]
        #         y = y[:, :min_dim]
            
        #     # 计算余弦相似度 - 输入已归一化
        #     similarity = F.cosine_similarity(x, y, dim=-1).mean()
            
        #     # 使用温度参数平滑
        #     similarity = similarity / self.temperature
            
        #     # 裁剪到安全范围
        #     similarity = torch.clamp(similarity, -10.0, 10.0)
            
        #     return similarity
        
        # # 计算相关性损失
        # corr_low_med = stable_cosine_similarity(feat_low, feat_med)
        # corr_low_high = stable_cosine_similarity(feat_low, feat_high)
        # corr_med_high = stable_cosine_similarity(feat_med, feat_high)
        
        # # 检查相关系数的稳定性
        # # check_tensor_stability(corr_low_med, "corr_low_med", True)
        # # check_tensor_stability(corr_low_high, "corr_low_high", True)
        # # check_tensor_stability(corr_med_high, "corr_med_high", True)
        
        # # 目标：适度正相关（经过温度缩放后的目标值）
        # target_corr = 0.3 / self.temperature.item()  # 考虑温度参数
        
        # # 使用Huber损失替代MSE，减少大误差的梯度爆炸
        # def huber_loss(pred, target, delta=1.0):
        #     """Huber损失函数，对大误差更稳定"""
        #     error = pred - target
        #     abs_error = torch.abs(error)
        #     return torch.where(
        #         abs_error <= delta,
        #         0.5 * error * error,
        #         delta * (abs_error - 0.5 * delta)
        #     )
        
        # correlation_loss = (
        #     huber_loss(corr_low_med, target_corr, delta=0.5) + 
        #     huber_loss(corr_low_high, target_corr, delta=0.5) + 
        #     huber_loss(corr_med_high, target_corr, delta=0.5)
        # ) / 3
        
        # # 稳定的距离计算
        # def stable_distance(x, y):
        #     """计算稳定的距离"""
        #     if x.shape[-1] != y.shape[-1]:
        #         min_dim = min(x.shape[-1], y.shape[-1])
        #         x = x[:, :min_dim]
        #         y = y[:, :min_dim]
            
        #     # 使用L1距离，比L2更稳定
        #     distance = torch.mean(torch.abs(x - y))
        #     return torch.clamp(distance, 0, 2.0)
        
        # dist_low_med = stable_distance(feat_low, feat_med)
        # dist_low_high = stable_distance(feat_low, feat_high)
        # dist_med_high = stable_distance(feat_med, feat_high)
        
        # # 多样性损失 - 使用平滑的ReLU替代
        # tau_normalized = 0.1  # 降低阈值，因为使用了L1距离
        # diversity_loss = (
        #     F.softplus(tau_normalized - dist_low_med) + 
        #     F.softplus(tau_normalized - dist_low_high) + 
        #     F.softplus(tau_normalized - dist_med_high)
        # ) / 3
        
        # # 简化的平衡损失
        # def stable_feature_norm(x):
        #     """计算稳定的特征范数"""
        #     # 使用L1范数的均值，更稳定
        #     norm = torch.mean(torch.abs(x))
        #     return torch.clamp(norm, 1e-8, 1.0)  # 输入已归一化，范围应该很小
        
        # norm_low = stable_feature_norm(feat_low)
        # norm_med = stable_feature_norm(feat_med)
        # norm_high = stable_feature_norm(feat_high)
        
        # # 使用相对标准差作为平衡损失
        # norms = torch.stack([norm_low, norm_med, norm_high])
        # mean_norm = norms.mean()
        # relative_std = torch.std(norms) / (mean_norm + 1e-8)
        # balance_loss = torch.clamp(relative_std, 0, 1.0)
        
        # # 检查各损失组件的稳定性
        # # check_tensor_stability(correlation_loss, "correlation_loss", True)
        # # check_tensor_stability(diversity_loss, "diversity_loss", True) 
        # # check_tensor_stability(balance_loss, "balance_loss", True)
        
        # # 使用更保守的权重组合
        # safe_lambda1 = torch.clamp(torch.tensor(self.lambda1), 0, 0.1)  # 限制权重范围
        # safe_lambda2 = torch.clamp(torch.tensor(self.lambda2), 0, 0.1)
        
        # total_crossing_loss = (
        #     correlation_loss + 
        #     safe_lambda1 * diversity_loss + 
        #     safe_lambda2 * balance_loss
        # )
        
        # # 最终稳定性保证
        # total_crossing_loss = torch.clamp(total_crossing_loss, 0, 10.0)  # 更严格的上限
        
        # # 详细的调试输出
        # if torch.isnan(total_crossing_loss) or torch.isinf(total_crossing_loss) or total_crossing_loss > 5.0:
        #     print(f"🚨 CROSSING LOSS 异常检测:")
        #     print(f"  correlation_loss: {correlation_loss.item():.6f}")
        #     print(f"  diversity_loss: {diversity_loss.item():.6f}")  
        #     print(f"  balance_loss: {balance_loss.item():.6f}")
        #     print(f"  total_crossing_loss: {total_crossing_loss.item():.6f}")
        #     print(f"  相关系数: low_med={corr_low_med.item():.6f}, low_high={corr_low_high.item():.6f}, med_high={corr_med_high.item():.6f}")
        #     print(f"  距离: low_med={dist_low_med.item():.6f}, low_high={dist_low_high.item():.6f}, med_high={dist_med_high.item():.6f}")
        #     print(f"  范数: low={norm_low.item():.6f}, med={norm_med.item():.6f}, high={norm_high.item():.6f}")
            
        return total_crossing_loss
    
    def feature_self_loss(self, mlp_outputs, mlp_module):
        """
        计算自身层级关注损失 L_self
        
        数学定义：
        L_self = Σₖ [L_recon^k + γ₁ * L_focus^k + γ₂ * L_consistency^k]
        
        Args:
            mlp_outputs: MultiLevelMLP的输出字典
            mlp_module: MultiLevelMLP模块实例
        Returns:
            torch.Tensor: 标量损失值
        """
        features = mlp_outputs['features']
        total_loss = 0
        
        for level in ['low', 'med', 'high']:
            # 修改处：重构损失 - 测试MLP从噪声中恢复自身特征的能力
            self_scores = mlp_outputs[f'{level}_self']  # [B, n_experts]
            original_feat = features[level]  # [B, embed_dim]
            
            # 使用对应的重构器
            reconstructor = getattr(mlp_module, f'reconstructor_{level}')
            reconstructed_feat = reconstructor(self_scores)  # [B, embed_dim]
            
            # MSE重构损失：||Decoder(MLP(F_k + N)) - F_k||₂²
            recon_loss = F.mse_loss(reconstructed_feat, original_feat)
            
            # 修改处：专注损失 - 确保MLP对自身层级最敏感
            cross_scores = mlp_outputs[f'{level}_cross']  # [B, n_experts]
            
            # # 计算自身输入的最大响应 vs 混合输入的平均响应
            # max_self_response = torch.max(self_scores, dim=-1)[0].mean()  # 标量
            # avg_cross_response = cross_scores.mean()  # 标量
            
            # 专注损失：-log(max_self / avg_cross)
            # 目标：自身输入应该产生更强的响应
            # focus_loss = -torch.log(max_self_response / (avg_cross_response + 1e-8))
            
            # # 修改处：一致性损失 - 确保输出分布的稳定性
            # # 计算跨batch的方差，希望同一层级的输出相对稳定
            # consistency_loss = torch.var(cross_scores, dim=0, unbiased=False).mean()  # 跨专家维度的方差均值
            
            #根据不同层级的特性，调整损失权重（在对应的level_loss前面乘上与swin的具体所在层数相关的系数）
            #low随swin层数增大权重减小，medium先增后减，high随swin层数增大权重增大
            # 单层级损失
            level_loss = recon_loss 
            # + self.gamma1 * focus_loss 
            # + self.gamma2 * consistency_loss
            total_loss += level_loss
            
        return total_loss / 3  # 三个层级的平均损失

