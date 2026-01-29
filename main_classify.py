

# general libs
import json
import os,  argparse

from typing import Any, Dict
import warnings
# 自定义进度条，展示训练过程中的多项 loss
import tqdm
from tqdm import tqdm as tqdm_class


warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("medium")
from utils import *
from models import swin, bert, classifier

from pytorch_lightning import LightningModule, Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from utils.datasets import create_loaders

import logging
from pytorch_lightning.callbacks.progress import TQDMProgressBar  # 添加进度条回调支持


torch.cuda.set_device(0)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
    A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Full Pipeline Training")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="food101",
        choices=["food", "imdb", "snli", "chestxray", "rocov2"],
        help="which dataset to use.",
    )



    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Set the maximum batch size for training.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of workers for pytorch's dataloader.",
    )


    # Encoder

    # General
    parser.add_argument("--name", default="", type=str, help="model name")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="If true, only validate segmentation.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="If true, only validate segmentation.",
    )

    parser.add_argument(
        "--max_epoch",
        type=int,
        #nargs="+",
        default=10,
        help="max num of epoches for training",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed to provide (near-)reproducibility.",
    )
    parser.add_argument(
        "--exp_name",
        default="model",
        type=str,
        metavar="PATH",
        help="path to save checkpoint (default: model)",
    )
    parser.add_argument(
        "--ckpt",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )



    # Optimisers
    parser.add_argument(
        "--lr_vis", type=float, default=4e-4, help="Learning rate for visual encoder."
    )
    parser.add_argument(
        "--lr_text", type=float, default=5e-4, help="Learning rate for text encoder."
    )

    parser.add_argument(
        "--wd_vis", type=float, default=1e-3, help="Weight decay for visual encoder."
    )
    parser.add_argument(
        "--wd_text", type=float, default=1e-3, help="Weight decay for text encoder."
    )

    # parser.add_argument("--lamda", type=float, default=LAMDA, help="Lamda for L1 norm.")
    parser.add_argument(
        "--warmup_epochs", type=float, default=1, help="warmup epochs for lr scheduler"
    )
    # parser.add_argument('-t', '--bn-threshold', type=float, default=BN_threshold,
    #                     help='Threshold for slimming BNs.')
    parser.add_argument(
        "--backbone",
        default="swinb_224",
        type=str,
        choices=["swinb_224", "swinb_384", "vitb"],
    )

    # ---------------model setting----------------
    parser.add_argument(
        "--fuse_method",
        default="late_concat",
        type=str,
        choices=[
            "late_concat",
            "instruct_v2t",
            "instruct_t2v",
            "instruct_moe_t2v",
            "instruct_moe_v2t",
            "instruct_mm_moe_t2v",
            "mope",
            "sequentialfuse",
            "img_only",
            "text_only",
            "p_sequential",
        ],
        help="how to fuse to modality",
    )
    parser.add_argument(
        "--train_instructor",
        action="store_true",
        default=False,
        help="whether the instructor should be trained at the same time.",
    )

    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=False,
        help="Whether to freeze (vision) encoder.",
    )
    parser.add_argument(   
        "--route_per_layer",
        action="store_true",
        default=True,
        help="whether to learn a routing weight for each layers",
    )
    parser.add_argument(
        "--dense_routing",
        action="store_true",
        default=True,
        help="whether to densely route expert (all experts are used)). temporarily deprecated arg",
    )
    # ----------------prompt learning---------------
    parser.add_argument(
        "--use_vpt", action="store_true", default=False, help="Whether to use VPT."
    )
    parser.add_argument(
        "--use_pbert",
        action="store_true",
        default=False,
        help="Whether to use Prompted Bert.",
    )

    parser.add_argument(
        "--vis_prompt_type",
        default="vpt",
        choices=["vpt"],
        type=str,
        help="how to apply prompt tuning?",
    )

    parser.add_argument(
        "--d_cross", type=int, default=8, help="dimension of cross-feature embd"
    )
    parser.add_argument(
        "--d_inter", type=int, default=2, help="dimension of cross-feature embd"
    )


    parser.add_argument(
        "--moe_n_experts", type=int, default=4, help="number of experts for moe"
    )
   
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=6,
        help="number of learnable visual prompts",
    )
    parser.add_argument(
        "--t_prompt_length",
        type=int,
        default=4,
        help="number of learnable text prompts",
    )

    parser.add_argument(
        "--use_static_prompt",
        action="store_true",
        default=True,
        help="whether to use additional static visual prompt, temporarily deprecated arg (always true)",
    )
    parser.add_argument(
        "--use_instruct",
        action="store_true",
        default=True,
        help="whether to use instructor, temporarily deprecated arg (always true)",
    )
    parser.add_argument(
        "--moe_top_k", type=int, default=1, help="number of top k experts to use, has to be used together with dense_routing, temporarily deprecated arg"
    )
    parser.add_argument(
        "--prompt_init",
        type=str,
        default="uniform",
        choices=["uniform", "normal", "othorgonal"],
        help="how prompt and experts are inited",
    )
    # --------------loss---------------
    
    parser.add_argument(
        "--w_imp", type=float, default=0.01, help="weight for importance loss"
    )
    parser.add_argument(
        "--smooth_label",
        action="store_true",
        default=False,
        help="whether to use lable smoothing",
    )
    parser.add_argument(
        "--w_othor", type=float, default=0.0, help="weight for othor loss"
    )
    parser.add_argument(
        "--w_contrast", type=float, default=0.005, help="weight for contrastive loss"
    )
    parser.add_argument('--ignore_label', type=int, default=-1, help='Label to ignore for loss computation')
    #修改处
    # ---------------debug------------------
    parser.add_argument("--exp_note", default="", type=str, help="experiment note")
    # ----------experiment setting----------------
    parser.add_argument(
        "--n_shot", default=0, type=int, help="number of shots for low shot learning"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=False,
        help="whether to finetune the model",
    )
    parser.add_argument(
    "--max_length", type=int, choices=[512, 1024], default=512,
    help="Maximum total input sequence length (supports 512 or 1024)."
    )
    # 修改处：添加特征损失权重参数
    parser.add_argument(
        "--w_feature", type=float, default=0.1, help="weight for feature crossing and self loss"
    )
    # 修改处：添加多层级MLP损失权重参数
    parser.add_argument(
        "--w_crossing", type=float, default=0.1, help="weight for crossing loss"
    )
    parser.add_argument(
        "--w_self", type=float, default=0.05, help="weight for self loss"
    )
    parser.add_argument(
        "--use_hierarchical_mlp",
        action="store_true",
        default=True,
        help="whether to use hierarchical MLP system instead of MoPE",
    )

    #修改处
    # 添加 GPU 参数
    #parser.add_argument(
    #    "--gpus",
    #    type=str,
    #    default=None,
    #    help="Comma-separated list of GPUs to use (e.g., '0,1' for multi-GPU, '0' for single-GPU)"
    #)
    return parser.parse_args()
    


class Model(LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        
        # 修改处: 首先检查并转换args类型
        if args is not None and type(args) is dict:
            args = argparse.Namespace(**args)
        self.args = args
        
        # 修改处
         # 配置日志输出到文件
        logging.basicConfig(
            filename="training_logs.txt",  # 输出文件
            level=logging.INFO,  # 设置日志级别为INFO
            format="%(asctime)s - %(message)s",  # 设置输出格式
        )
        
        # 解析参数后，设置视觉backbone对应的crop_size
        max_length = getattr(self.args, 'max_length', 512)  # 默认值512
        if max_length == 1024:
            self.args.backbone = "swinb_224"   # 使用384分辨率的Swin-B模型
        else:
            self.args.backbone = "swinb_224"   # 默认使用224分辨率模型
        
        # 修改处
        self.validation_step_outputs = []

        self.save_hyperparameters(self.args)
        if self.args.dataset == "food":
            self.num_classes = 101
        elif self.args.dataset == "chestxray":
            self.num_classes = 16
        elif self.args.dataset == "rocov2":
            labels_path = os.path.join("data", "ROCOv2-radiology-main", "labels.txt")
            with open(labels_path) as f:
                labels_list = [l.strip() for l in f.readlines()]
            self.num_classes = len(labels_list)#  # 修改处: ROCOv2 数据集的标签数量
        elif self.args.dataset == "imdb":
            self.num_classes = 23
        else:  # snli
            self.num_classes = 3
        


        if self.args.dataset == "food" or self.args.dataset == "snli":
            self.final_act = lambda x: F.log_softmax(x, dim=1)
            self.loss = nn.NLLLoss(ignore_index=self.args.ignore_label)
        elif self.args.dataset == "imdb" or self.args.dataset == "rocov2":
            self.final_act = lambda x: F.sigmoid(x)
            self.loss = nn.BCEWithLogitsLoss()
        # 修改处: 为 chestxray 数据集定义 Sigmoid 激活和 BCEWithLogitsLoss
        elif self.args.dataset == "chestxray":
            self.final_act = lambda x: F.log_softmax(x, dim=1)
            self.loss = nn.NLLLoss(ignore_index=self.args.ignore_label)      # 修改处: 为 ChestXRay 数据集定义 NLLLoss
        
        self.vision_classifier = swin.get_swin_classifier(
            num_classes=self.num_classes,
            backbone=self.args.backbone,
            use_vpt=self.args.use_vpt,
            moe_n_experts=self.args.moe_n_experts,
            prompt_length=self.args.prompt_length,
            use_static_prompt=self.args.use_static_prompt,
            prompt_init=self.args.prompt_init,
            use_instruct=self.args.use_instruct,
            d_cross=self.args.d_cross,
            d_inter=self.args.d_inter,
        )
        

        self.text_classifier = bert.BertClassifier(
            self.num_classes,
            use_prompt=self.args.use_pbert,
            prompt_length=self.args.t_prompt_length,
            max_position_embeddings=self.args.max_length
        )

        self.fuse_method = self.args.fuse_method
        self.classifier = classifier.VisionTextClassifiers(
            self.vision_classifier,
            self.text_classifier,
            self.num_classes,
            fusion_method=self.fuse_method,
            train_instructor=self.args.train_instructor,
            moe_n_experts=self.args.moe_n_experts,
            dense_routing=self.args.dense_routing,
            moe_top_k=self.args.moe_top_k,
            route_per_layer=self.args.route_per_layer,
        )
        if not self.args.train_instructor:
            #修改处
            for name, param in self.text_classifier.named_parameters():
                if "prompt_vector" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            #self.text_classifier.requires_grad_(False)
        if self.args.finetune:
            self.vision_classifier.requires_grad_(True)
            self.text_classifier.requires_grad_(True)
        if self.fuse_method == "promptfuse":
            self.text_classifier.requires_grad_(True)
            self.text_classifier.freeze_backbone()
            self.vision_classifier.requires_grad_(False)
        if self.args.fuse_method == "sequentialfuse":
            self.vision_classifier.requires_grad_(True)
            self.text_classifier.requires_grad_(True)
        if self.args.fuse_method == "img_only":
            # self.vision_classifier.requires_grad_(True)
            self.text_classifier.requires_grad_(False)
        if self.args.fuse_method == "text_only":
            self.vision_classifier.requires_grad_(False)
            # self.text_classifier.requires_grad_(True)
        if self.args.fuse_method == "p_sequential":
            self.text_classifier.requires_grad_(True)
            self.text_classifier.freeze_backbone()
        if self.fuse_method == "instruct_v2t":
            self.text_classifier.freeze_backbone()
        # self.text_classifier.freeze_backbone()
        # self.vision_classifier.freeze_backbone()
        # TODO Sep 12: deprecated when prompt tuning, always freeze encoder
        if self.args.use_vpt and not self.args.freeze_encoder:
            print("[Warning] Using VPT without freezing encoder")
            
        # 修改处
        # 确保 prompt 向量和 MoE 路由网络权重可训练
        for name, param in self.vision_classifier.named_parameters():
            if "prompt" in name or "moe_proj" in name or "mm_frozen_expert_key_embed" in name or "cnn_extractor" in name:
                param.requires_grad_(True)
        for name, param in self.classifier.named_parameters():
            #print(f"{name}: requires_grad = {param.requires_grad}")
            if "moe_proj" in name or "instruct_proj" in name:
                param.requires_grad_(True)
            if param.requires_grad:
                num_params = np.prod(param.size())
                num_params_k = num_params / 1000
                print("{}, num_params: {}K".format(name, num_params_k))
            # 修改处
        self.flag_grad_printed = False  #修改处
            
    def training_step(self, batch, batch_idx):
        # 修改处
        text_input = batch["text"]
        img_input = batch["image"]
        gt_label = batch["label"]
        #修改处
        # Ensure text_input is tokenized and padded/truncated to max length
        tokenizer_output = self.text_classifier.tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True, max_length=self.text_classifier.bert_encoder.config.max_position_embeddings - 4
        )

        #input_ids = tokenizer_output["input_ids"].to(self.text_classifier.bert_encoder.device)
        #token_type_ids = tokenizer_output["token_type_ids"].to(self.text_classifier.bert_encoder.device)
        #attention_mask = tokenizer_output["attention_mask"].to(self.text_classifier.bert_encoder.device)
        #修改处
        
        # Compute outputs
        outputs, extra_out = self.classifier(img_input, text_input)

        # for amp compatibility, BCEWithLogitsLoss is used so no sigmoid act for multilabel during training
        if self.args.dataset == "food" or self.args.dataset == "snli":
            outputs = self.final_act(outputs)
        loss_val = self.loss(outputs, gt_label.squeeze())
        
        # 修改处
        # 打印损失到日志文件
        logging.info(f"Training Step Loss: {loss_val.item()}")

        # Log the loss for debugging
        self.log('train_loss', loss_val)
        
        # 打印梯度
        #for name, param in self.named_parameters():
            #if param.grad is not None:
                #print(f"{name}: grad norm = {param.grad.norm()}")
            #else:
                #print(f"{name}: No gradient")
        #for name, param in self.named_parameters():
            #if param.grad is not None and ("prompt" in name or "proj" in name):
                #print(f"{name}: grad norm = {param.grad.norm()}")
        # 修改处
        # 修改处：添加特征损失计算
        if hasattr(self.vision_classifier.encoder, 'current_mlp_outputs') and self.vision_classifier.encoder.current_mlp_outputs is not None:
            mlp_outputs = self.vision_classifier.encoder.current_mlp_outputs
            loss_computer = self.vision_classifier.encoder.feature_loss_computer
            mlp_module = self.vision_classifier.encoder.multi_level_mlp
        
            # 计算两个特征损失
            try:
                crossing_loss = loss_computer.feature_crossing_loss(mlp_outputs['features'])
                self_loss = loss_computer.feature_self_loss(mlp_outputs, mlp_module)
            
                # 添加到总损失（权重可作为超参数调整）
                feature_loss_weight = getattr(self.args, 'w_feature', 0.1)  # 默认权重0.1
                total_feature_loss = crossing_loss + self_loss
                loss_val += feature_loss_weight * total_feature_loss
            
                # 记录各项损失用于监控
                self.log("crossing_loss", crossing_loss, prog_bar=True)
                self.log("self_loss", self_loss, prog_bar=True)
                self.log("feature_total_loss", total_feature_loss, prog_bar=True)
            
                # 记录到日志文件
                logging.info(f"Feature Losses - Crossing: {crossing_loss.item():.6f}, Self: {self_loss.item():.6f}, Total: {total_feature_loss.item():.6f}")
            
            except Exception as e:
                # 如果特征损失计算出错，记录错误但不影响主训练
                logging.warning(f"Feature loss computation failed: {str(e)}")
                self.log("feature_loss_error", 1.0)
        # 修改处
        importance_loss = None
        othor_loss = None
        ent_loss = None
        contrast_loss = None
        crossing_loss = None
        self_loss = None

        # 修改处
        if extra_out is not None:
            if "importance_loss" in extra_out:
                imp_loss = extra_out["importance_loss"]
                self.log("imp_loss", imp_loss, prog_bar=True)
                loss_val += imp_loss * self.args.w_imp
            if "othor_loss" in extra_out:
                othor_loss = extra_out["othor_loss"]
                self.log("othor_loss", othor_loss, prog_bar=True)
                loss_val += othor_loss * self.args.w_othor
            if "entropy_loss" in extra_out:
                ent_loss = extra_out["entropy_loss"]
                self.log("entropy_loss", ent_loss, prog_bar=True)
                loss_val += ent_loss * self.args.w_contrast
            if "contrast_loss" in extra_out:
                contrast_loss = extra_out["contrast_loss"]
                self.log("contrast_loss", contrast_loss, prog_bar=True)
                loss_val += contrast_loss * self.args.w_contrast
            # 修改处
            # 修改处：添加新的多层级损失项
            if "crossing_loss" in extra_out:
                crossing_loss = extra_out["crossing_loss"]
                self.log("crossing_loss", crossing_loss, prog_bar=True)
                loss_val += crossing_loss * self.args.w_crossing
                logging.info(f"Crossing Loss: {crossing_loss.item()}")
                
            if "self_loss" in extra_out:
                self_loss = extra_out["self_loss"]
                self.log("self_loss", self_loss, prog_bar=True)
                loss_val += self_loss * self.args.w_self
                logging.info(f"Self Loss: {self_loss.item()}")
            # print(f"importance_loss: {imp_loss}, othor_loss: {othor_loss}, entropy_loss: {ent_loss}, contrast_loss: {contrast_loss}, crossing_loss: {crossing_loss}, self_loss: {self_loss}")

        crt_vision_lr = self.optimizers().param_groups[0]["lr"]
        self.log("vision_lr", crt_vision_lr)
        self.log("train_loss", loss_val)

        # train batch stats
        with torch.no_grad():
            if self.args.dataset == "food" or self.args.dataset == "snli":
                outputs = self.final_act(outputs)  # for metric calc
                pred_label = torch.argmax(outputs, dim=1)
                correct_cnt = torch.sum(pred_label == gt_label.squeeze()).item()
                all_cnt = pred_label.size(0)
                acc = correct_cnt / all_cnt
                self.log("train_acc", acc)
                # 修改处
                logging.info(f"Training Step Accuracy: {acc * 100:.2f}%")
                # 修改处

            elif self.args.dataset == "imdb" or self.args.dataset == "rocov2":  # multilable, calc f1
                outputs = self.final_act(outputs)  # for metric calc
                pred_label = (torch.sigmoid(outputs) > 0.5).int()
                f1_micro = f1_score(
                    gt_label.squeeze().cpu().numpy(),
                    pred_label.cpu().numpy(),
                    average="micro",
                )
                self.log("train_f1_micro", f1_micro)
                f1_macro = f1_score(
                    gt_label.squeeze().cpu().numpy(),
                    pred_label.cpu().numpy(),
                    average="macro",
                )
                self.log("train_f1_macro", f1_macro)
                f1_samples = f1_score(
                    gt_label.squeeze().cpu().numpy(),
                    pred_label.cpu().numpy(),
                    average="samples",
                )
                self.log("train_f1_samples", f1_samples)
                f1_weighted = f1_score(
                    gt_label.squeeze().cpu().numpy(),
                    pred_label.cpu().numpy(),
                    average="weighted",
                )
                self.log("train_f1_weighted", f1_weighted)
                # Compute training accuracy (exact-match for multi-label)
                acc_tensor = (pred_label == gt_label).all(dim=1)  # tensor of shape [batch_size] with bools
                train_acc = acc_tensor.float().mean().item()
                self.log("train_acc", train_acc)
                logging.info(f"Training Step Accuracy: {train_acc * 100:.2f}%")
                # 修改处
                logging.info(f"Training Step F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, F1_samples: {f1_samples:.4f}, F1_weighted: {f1_weighted:.4f}")
                # 修改处
            return loss_val

    def validation_step(self, batch, batch_idx):
        # 修改处
        text_input = batch["text"]
        img_input = batch["image"]
        gt_label = batch["label"]

        # 修改处
        # prepare data for jsonl saving
        image_paths = batch.get('image_path', None)
        # compute outputs
        # 计算模型输出
        outputs, extra_out = self.classifier(img_input, text_input)
        #修改处
        # compute probabilities and predictions
        if self.args.dataset in ["imdb", "rocov2"]:
            probs = torch.sigmoid(outputs)
            pred_label = (probs > 0.5).int()
        else:
            probs = F.softmax(outputs, dim=1)
            pred_label = torch.argmax(outputs, dim=1)

        # 计算损失
        loss_val = self.loss(outputs, gt_label.squeeze())

        # 记录损失到日志
        logging.info(f"Validation Step Loss: {loss_val.item()}")
        
        # 修改处
        # 修改处：在验证步骤中也计算特征损失用于监控（不用于反向传播）
        val_crossing_loss = 0.0
        val_self_loss = 0.0
        val_feature_total_loss = 0.0
        with torch.no_grad():  # 验证时不需要梯度
            if hasattr(self.vision_classifier.encoder, 'current_mlp_outputs') and self.vision_classifier.encoder.current_mlp_outputs is not None:
                mlp_outputs = self.vision_classifier.encoder.current_mlp_outputs
                loss_computer = self.vision_classifier.encoder.feature_loss_computer
                mlp_module = self.vision_classifier.encoder.multi_level_mlp
            
                try:
                    val_crossing_loss = loss_computer.feature_crossing_loss(mlp_outputs['features'])
                    val_self_loss = loss_computer.feature_self_loss(mlp_outputs, mlp_module)
                    val_feature_total_loss = val_crossing_loss + val_self_loss
                
                    # 记录验证特征损失
                    self.log("val_crossing_loss", val_crossing_loss)
                    self.log("val_self_loss", val_self_loss)
                    self.log("val_feature_total_loss", val_feature_total_loss)

                # 记录到日志文件
                    logging.info(f"Validation Feature Losses - Crossing: {val_crossing_loss.item():.6f}, Self: {val_self_loss.item():.6f}, Total: {val_feature_total_loss.item():.6f}")
                
                except Exception as e:
                    logging.warning(f"Validation feature loss computation failed: {str(e)}")
        # 修改处

        # 计算准确率并记录到日志
        with torch.no_grad():
            if self.args.dataset == "food" or self.args.dataset == "snli" or self.args.dataset == "chestxray":
                pred_label = torch.argmax(outputs, dim=1)
                correct_cnt = torch.sum(pred_label == gt_label.squeeze()).item()
                all_cnt = pred_label.size(0)
                acc = correct_cnt / all_cnt
                logging.info(f"Validation Step Accuracy: {acc * 100:.2f}%")
                print(f"Validation Step Accuracy: {acc * 100:.2f}%")
            elif self.args.dataset == "imdb" or self.args.dataset == "rocov2":  # multilable, calc f1:
                pred_label = (torch.sigmoid(outputs) > 0.5).int()
                f1_micro = f1_score(
                    gt_label.squeeze().cpu().numpy(),
                    pred_label.cpu().numpy(),
                    average="micro",
                )
                f1_macro = f1_score(
                    gt_label.squeeze().cpu().numpy(),
                    pred_label.cpu().numpy(),
                    average="macro",
                )
                f1_samples = f1_score(
                    gt_label.squeeze().cpu().numpy(),
                    pred_label.cpu().numpy(),
                    average="samples",
                )
                f1_weighted = f1_score(
                    gt_label.squeeze().cpu().numpy(),
                    pred_label.cpu().numpy(),
                    average="weighted",
                )
                # Compute exact-match accuracy for this batch
                correct_cnt = (pred_label == gt_label.squeeze()).all(dim=1).sum().item()
                all_cnt = pred_label.size(0)
                acc = correct_cnt / all_cnt
                logging.info(f"Validation Step Accuracy: {acc * 100:.2f}%")
                logging.info(f"Validation Step F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, F1_samples: {f1_samples:.4f}, F1_weighted: {f1_weighted:.4f}")
                print(f"Validation Step F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, F1_samples: {f1_samples:.4f}, F1_weighted: {f1_weighted:.4f}")
                # 修改处

        ret_dict = {
            'image_paths': image_paths,
            'text': text_input,
            'pred_label': pred_label.detach().cpu(),
            'probs': probs.detach().cpu(),
            'gt_label': gt_label.squeeze().detach().cpu(),
            'val_crossing_loss': val_crossing_loss,
            'val_self_loss': val_self_loss,
            'val_feature_total_loss': val_feature_total_loss,
        }

        if extra_out is not None:
            if "moe_scores" in extra_out:
                ret_dict["moe_scores"] = extra_out["moe_scores"]
            if "cls_" in extra_out:
                ret_dict["cls"] = extra_out["cls_"].detach().cpu()

        self.validation_step_outputs.append(ret_dict)
        return loss_val

    def on_validation_epoch_end(self) -> None:
        # 修改处：计算整个验证epoch的特征损失统计
        all_val_crossing_losses = []
        all_val_self_losses = []
        all_val_feature_total_losses = []
        for step_out in self.validation_step_outputs:
            if isinstance(step_out["val_crossing_loss"], torch.Tensor):
                all_val_crossing_losses.append(step_out["val_crossing_loss"].item())
                all_val_self_losses.append(step_out["val_self_loss"].item())
                all_val_feature_total_losses.append(step_out["val_feature_total_loss"].item())
    
        # 记录整个epoch的平均特征损失
        if all_val_crossing_losses:
            avg_val_crossing_loss = sum(all_val_crossing_losses) / len(all_val_crossing_losses)
            avg_val_self_loss = sum(all_val_self_losses) / len(all_val_self_losses)
            avg_val_feature_total_loss = sum(all_val_feature_total_losses) / len(all_val_feature_total_losses)
        
            self.log("val_epoch_crossing_loss", avg_val_crossing_loss)
            self.log("val_epoch_self_loss", avg_val_self_loss)
            self.log("val_epoch_feature_total_loss", avg_val_feature_total_loss)
        
            logging.info(f"Validation Epoch Feature Losses - Crossing: {avg_val_crossing_loss:.6f}, Self: {avg_val_self_loss:.6f}, Total: {avg_val_feature_total_loss:.6f}")
        # 修改处
        if self.args.dataset == "food" or self.args.dataset == "snli":
            all_cnt = 0
            correct_cnt = 0
            expert_img_dir = os.path.join("debug", "route")
            os.makedirs(expert_img_dir, exist_ok=True)
            cls_features = []
            gt_labels = []
            for step_out in self.validation_step_outputs:
                pred_label = step_out["pred_label"]
                gt_label = step_out["gt_label"]
                # loss_val = step_out["loss"]
                all_cnt += pred_label.size(0)
                correct_cnt += torch.sum(pred_label == gt_label).item()

                if "cls" in step_out:
                    cls_features.append(step_out["cls"])
                gt_labels.append(gt_label)

            acc = correct_cnt / all_cnt
            self.log("val_acc", acc)
        elif self.args.dataset == "imdb" or self.args.dataset == "rocov2":
            all_preds = []
            all_labels = []
            
            for step_out in self.validation_step_outputs:
                pred_label = step_out["pred_label"]
                gt_label = step_out["gt_label"]

                all_preds.append(pred_label)
                all_labels.append(gt_label)
            f1_macro = f1_score(
                torch.cat(all_labels).squeeze().cpu().numpy(),
                torch.cat(all_preds).cpu().numpy(),
                average="macro",
            )
            self.log("val_f1_macro", f1_macro)
            f1_micro = f1_score(
                torch.cat(all_labels).squeeze().cpu().numpy(),
                torch.cat(all_preds).cpu().numpy(),
                average="micro",
            )
            self.log("val_f1_micro", f1_micro)
            f1_samples = f1_score(
                torch.cat(all_labels).squeeze().cpu().numpy(),
                torch.cat(all_preds).cpu().numpy(),
                average="samples",
            )
            self.log("val_f1_samples", f1_samples)
            f1_weighted = f1_score(
                torch.cat(all_labels).squeeze().cpu().numpy(),
                torch.cat(all_preds).cpu().numpy(),
                average="weighted",
            )
            self.log("val_f1_weighted", f1_weighted)
            #修改处
            # **New code: compute exact match accuracy for multi-label**
            preds_tensor = torch.cat(all_preds)
            labels_tensor = torch.cat(all_labels)
            # Check for exact match per sample (all labels match)
            exact_matches = (preds_tensor == labels_tensor).all(dim=1)
            val_acc = exact_matches.float().mean().item()
            self.log("val_acc", val_acc)  # Log the validation accuracy
        
        elif self.args.dataset == "chestxray":  # 修改处: ChestXRay 增加准确率和F1计算
            all_preds = []
            all_labels = []
            all_cnt = 0
            correct_cnt = 0
            for step_out in self.validation_step_outputs:
                pred_label = step_out["pred_label"]
                gt_label = step_out["gt_label"]
                all_preds.append(pred_label)
                all_labels.append(gt_label)
                all_cnt += pred_label.size(0)
                correct_cnt += torch.sum(pred_label == gt_label).item()
            # 计算准确率
            acc = correct_cnt / all_cnt
            # 将所有预测和标签拼接成张量用于计算F1
            preds_np = torch.cat(all_preds).cpu().numpy()
            labels_np = torch.cat(all_labels).squeeze().cpu().numpy()
            f1_macro = f1_score(labels_np, preds_np, average="macro")
            f1_micro = f1_score(labels_np, preds_np, average="micro")
            f1_samples = f1_score(labels_np, preds_np, average="samples")
            f1_weighted = f1_score(labels_np, preds_np, average="weighted")
            # 记录指标
            self.log("val_acc", acc)
            self.log("val_f1_macro", f1_macro)
            self.log("val_f1_micro", f1_micro)
            self.log("val_f1_samples", f1_samples)
            self.log("val_f1_weighted", f1_weighted)
        # 修改处

        # 将最终验证结果保存为 JSONL 文件
        try:
            out_path = os.path.join(self.logger.log_dir, "final_val_outputs.jsonl")
            with open(out_path, 'w', encoding='utf-8') as f:
                for step_out in self.validation_step_outputs:
                    image_paths = step_out.get('image_paths', None)
                    texts = step_out['text']
                    preds = step_out['pred_label']
                    probs = step_out['probs']
                    labels = step_out['gt_label']
                    for i in range(len(texts)):
                        record = {
                            'image': image_paths[i] if image_paths is not None else None,
                            'text': texts[i],
                            'pred': preds[i].tolist(),
                            'prob': probs[i].tolist(),
                            'label': labels[i].tolist(),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
            logging.info(f"Final validation outputs saved to {out_path}")
        except Exception as e:
            logging.warning(f"Failed to save final_val_outputs.jsonl: {e}")
         # 修改处: clear stored outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> Any:

        vision_lr = self.args.lr_vis
        text_lr = self.args.lr_text
        #修改处
        # 为 CNN 模块单独设置较低的学习率（lr_vis 的 0.2 倍）
        vision_params = list(self.vision_classifier.parameters())
        cnn_params = (
            #list(self.vision_classifier.encoder.cnn_extractor.parameters()) 
            list(self.vision_classifier.encoder.cnn_extractor_medium.parameters()) 
            + list(self.vision_classifier.encoder.cnn_extractor_high.parameters()) 
            + list(self.vision_classifier.encoder.cnn_extractor_text_med.parameters()) 
            + list(self.vision_classifier.encoder.cnn_extractor_text_high.parameters())
        )
        cnn_param_ids = {id(p) for p in cnn_params}
        non_cnn_params = [p for p in vision_params if id(p) not in cnn_param_ids]

        optimizer_cfg = []
        optimizer_cfg.append({
            "params": non_cnn_params,
            "lr": vision_lr,
            "weight_decay": self.args.wd_vis,
        })
        optimizer_cfg.append({
            "params": cnn_params,
            "lr": vision_lr * 0.2,
            "weight_decay": self.args.wd_vis,
        })
        optimizer_cfg.append({
            "params": self.text_classifier.parameters(),
            "lr": text_lr,
            "weight_decay": self.args.wd_text,
        })
        # todo refactor as returned dict of params from init of the classifier
        if self.fuse_method == "late_concat":
            optimizer_cfg.append(
                {
                    "params": self.classifier.fusion_head.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_t2v":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_v2t":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": text_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_moe_t2v":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
            optimizer_cfg.append(
                {
                    "params": self.classifier.moe_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_moe_v2t":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
            optimizer_cfg.append(
                {
                    "params": self.classifier.moe_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "promptfuse":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "p_sequential":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": text_lr,
                    "weight_decay": 1e-4,
                }
            )
        optimizer = torch.optim.AdamW(optimizer_cfg)
        
        # optimizer = ScheduledOptim(optimizer,)
        # step lr,
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        #修改处
        # Choose which metric to monitor for LR scheduler
        if self.args.dataset == "imdb" or self.args.dataset == "rocov2":
            monitor_metric = "val_f1_micro"
        else:
            monitor_metric = "val_acc"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": monitor_metric,      # 根据 metric 而定
            "interval": "epoch",       # 默认为 "epoch"
            "frequency": 1,            # 每 1 个 epoch 调度一次
            "reduce_on_plateau": True,  # 必须为 True
            },
        }
    #修改处
    def on_after_backward(self) -> None:
        # 动态监控 CNN 梯度（最大值、均值，检查 NaN）
        conv1_w = self.vision_classifier.encoder.cnn_extractor.conv1.weight
        conv3_w = self.vision_classifier.encoder.cnn_extractor.pw3.weight  # CNN 最后一层卷积权重
        grad1 = conv1_w.grad
        grad3 = conv3_w.grad
        conv1_grad_max = float(grad1.abs().max().item()) if grad1 is not None else 0.0
        conv1_grad_mean = float(grad1.abs().mean().item()) if grad1 is not None else 0.0
        conv1_nan = bool(grad1 is not None and (torch.isnan(grad1).any() or torch.isinf(grad1).any()))
        conv3_grad_max = float(grad3.abs().max().item()) if grad3 is not None else 0.0
        conv3_grad_mean = float(grad3.abs().mean().item()) if grad3 is not None else 0.0
        conv3_nan = bool(grad3 is not None and (torch.isnan(grad3).any() or torch.isinf(grad3).any()))
        # 输出 conv1 和 conv3 的权重、梯度统计信息（每个 step）
        logging.info(f"Conv1 weight - mean: {conv1_w.data.mean():.6f}, max: {conv1_w.data.abs().max():.6f}; "
                     f"grad - mean: {conv1_grad_mean:.6f}, max: {conv1_grad_max:.6f}, NaN: {conv1_nan}")
        logging.info(f"Conv3 weight - mean: {conv3_w.data.mean():.6f}, max: {conv3_w.data.abs().max():.6f}; "
                     f"grad - mean: {conv3_grad_mean:.6f}, max: {conv3_grad_max:.6f}, NaN: {conv3_nan}")
        # 如检测到梯度爆炸或 NaN，降低学习率
        if conv1_nan or conv3_nan or conv1_grad_max > 1 or conv3_grad_max > 1:
            optimizer = self.optimizers() if not isinstance(self.optimizers(), list) else self.optimizers()[0]
            if len(optimizer.param_groups) > 1:
                # 将 CNN 参数组的学习率乘以 0.5
                cnn_group = optimizer.param_groups[1]
                cnn_group['lr'] *= 0.5
                logging.info(f"梯度异常，降低 CNN 模块学习率至 {cnn_group['lr']:.6e}")
            else:
                optimizer.param_groups[0]['lr'] *= 0.5
                logging.info(f"梯度异常，降低学习率至 {optimizer.param_groups[0]['lr']:.6e}")
    #修改处

    def on_train_end(self) -> None:
        final_weight = self.vision_classifier.encoder.cnn_extractor.conv1.weight[0,0].detach().cpu()
        print("Final CNN conv1[0,0]:", final_weight)
        diff = final_weight - self.initial_cnn_weight
        print("Weight change in conv1[0,0]:", diff.sum().item())
    def overwrite_args(self, args):
        """Avoid the exception caused by lighting when loading incompatible args from model ckpt."""
        # 修改处: 确保args是正确的类型
        if args is not None and type(args) is dict:
            args = argparse.Namespace(**args)
        self.args = args

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # remove the frozen parameter
        filter = ["attention", "mlp", "attn", "downsample", "intermediate"]
        for k in list(checkpoint["state_dict"].keys()):
            for f in filter:
                if f in k:
                    del checkpoint["state_dict"][k]

        return super().on_save_checkpoint(checkpoint)
    
    # 修改处: 打印CNN卷积核初始值和最终值，以确认参与训练
    def on_train_start(self) -> None:
        # 保存初始卷积权重以供对比
        #如果要在low MoPE中应用CNN，解除下面的两个注释
        #self.initial_cnn_weight = self.vision_classifier.encoder.cnn_extractor.conv1.weight[0,0].detach().cpu().clone()
        self.initial_cnn_weight_med = self.vision_classifier.encoder.cnn_extractor_medium.conv1.weight[0,0].detach().cpu().clone()
        self.initial_cnn_weight_high = self.vision_classifier.encoder.cnn_extractor_high.conv1.weight[0,0].detach().cpu().clone()
        #print("Initial CNN conv1[0,0]:", self.initial_cnn_weight)
        print("Initial CNN Medium conv1[0,0]:", self.initial_cnn_weight_med)
        print("Initial CNN High conv1[0,0]:", self.initial_cnn_weight_high)

    def on_after_backward(self) -> None:        #修改处
        if not self.flag_grad_printed:
            #如果要在low MoPE中应用CNN，解除下面的注释
            # grad_low = self.vision_classifier.encoder.cnn_extractor.conv1.weight.grad
            grad_med = self.vision_classifier.encoder.cnn_extractor_medium.conv1.weight.grad
            grad_high = self.vision_classifier.encoder.cnn_extractor_high.conv1.weight.grad
            # if grad_low is not None:
            #     print("CNN Low conv1[0,0] grad after first backward:", grad_low[0,0].detach().cpu())
            # else:
            #     print("CNN Low conv1 grad is None")
            #输出所有CNN总参数量
            print("Total CNN Medium parameters:", sum(p.numel() for p in self.vision_classifier.encoder.cnn_extractor_medium.parameters()))
            print("Total CNN High parameters:", sum(p.numel() for p in self.vision_classifier.encoder.cnn_extractor_high.parameters()))
            # 输出卷积核梯度
            if grad_med is not None:
                print("CNN Medium conv1[0,0] grad after first backward:", grad_med[0,0].detach().cpu())
            else:
                print("CNN Medium conv1 grad is None")
            if grad_high is not None:
                print("CNN High conv1[0,0] grad after first backward:", grad_high[0,0].detach().cpu())
            else:
                print("CNN High conv1 grad is None")
            self.flag_grad_printed = True
            #修改处
            

    def on_train_end(self):
        #如果要在low MoPE中应用CNN，解除下面的两个注释
        # final_weight_low = self.vision_classifier.encoder.cnn_extractor.conv1.weight[0,0].detach().cpu()
        final_weight_med = self.vision_classifier.encoder.cnn_extractor_medium.conv1.weight[0,0].detach().cpu()
        final_weight_high = self.vision_classifier.encoder.cnn_extractor_high.conv1.weight[0,0].detach().cpu()
        # print("Final CNN Low conv1[0,0]:", final_weight_low)
        # print("Weight change in Low conv1[0,0]:", (final_weight_low - self.initial_cnn_weight).sum().item())
        print("Final CNN Medium conv1[0,0]:", final_weight_med)
        print("Weight change in Medium conv1[0,0]:", (final_weight_med - self.initial_cnn_weight_med).sum().item())
        print("Final CNN High conv1[0,0]:", final_weight_high)
        print("Weight change in High conv1[0,0]:", (final_weight_high - self.initial_cnn_weight_high).sum().item())

    #修改处

    def benchmark_memory(self):
        # do forward pass with batch size 1 for 10000 times
        # measure the memory
        # ensure cuda classifer
        self.classifier.to("cuda").eval()
        dummy_input_img = torch.randn(16, 3, 224, 224).cuda()
        dummy_input_text = ["test"] * 16
        peak_mems = []
        self.eval()
        # optim = self.configure_optimizers()["optimizer"]
        device = torch.cuda.current_device()
        # warm up
        for i in range(2):
            out, _ = self.classifier(dummy_input_img, dummy_input_text)
            dummy_loss = out.sum()
            # dummy_loss.backward()
            # optim.step()
        for i in range(50):
            # optim.zero_grad()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            before_mem = torch.cuda.memory_allocated(device) / 1024**2

            # Perform inference
            with torch.no_grad():
                out, __ = self.classifier(dummy_input_img, dummy_input_text)
            # dummy_loss = out.sum()
            # dummy_loss.backward()
            # optim.step()
            # Measure memory after inference
            after_mem = torch.cuda.memory_allocated(device) / 1024**2
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_mems.append(peak_mem)
            torch.cuda.empty_cache()
            # measure the memory
        # print stat
        print(f"Mean peak memory: {np.mean(peak_mems)} MB")
        print(f"Std peak memory: {np.std(peak_mems)} MB")

    def benchmark_inference_speed(self):
        # do forward pass with batch size 1 for 10000 times
        # measure the time
        dummy_input_img = torch.randn(16, 3, 224, 224).cuda().half()
        dummy_input_text = ["test"] * 16
        self.eval()
        self.classifier.eval()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        self.classifier.to("cuda")
        repetitions = 1000
        with torch.no_grad():  # warm up
            for i in range(50):
                self.classifier(dummy_input_img, dummy_input_text)
        print("Start timing...")
        timings = []
        with torch.no_grad():
            torch.cuda.synchronize()
            for i in range(repetitions):
                starter.record()
                self.classifier(dummy_input_img, dummy_input_text)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
            torch.cuda.synchronize()
        print(f"Mean time: {np.mean(timings)} ms")
        print(f"Std time: {np.std(timings)} ms")

# 自定义进度条，展示训练过程中的多项 loss
class MyProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.loss_bar = None
        self.main_bar = None
        self.val_metrics_bar = None
        self.val_main_bar = None
        
    def get_metrics(self, trainer, pl_module):
        # 对于主进度条，只返回基本信息，不包含loss
        metrics = super().get_metrics(trainer, pl_module)
        basic_metrics = {}
        
        # 保留基本训练信息，但排除所有loss相关指标
        for key, value in metrics.items():
            # 排除所有包含loss关键字的指标
            if not any(loss_keyword in key.lower() for loss_keyword in ['loss']):
                # 保留epoch, step, v_num, lr等基本信息，但排除acc和f1（这些是验证指标）
                if key in ['epoch', 'step', 'v_num'] or 'lr' in key.lower():
                    basic_metrics[key] = value
                    
        return basic_metrics
    
    def get_loss_metrics(self, trainer, pl_module):
        # 专门为loss显示条获取损失指标
        metrics = super().get_metrics(trainer, pl_module)
        out = {}
        # 仅保留关心的损失项
        for key in ['imp_loss', 'othor_loss', 'entropy_loss', 'contrast_loss', 'crossing_loss', 'self_loss', 'feature_total_loss']:
            if key in metrics:
                out[key] = metrics[key]
        # 将 train_loss 视为总损失展示
        if 'train_loss' in metrics:
            out['total_loss'] = metrics['train_loss']
        return out

    def init_train_tqdm(self):
        # 创建主进度条（只显示基本信息，不显示loss）
        from tqdm import tqdm
        self.main_bar = tqdm(
            desc="Training",
            position=0,
            disable=False,
            leave=True,
            dynamic_ncols=True,
            smoothing=0,
        )
        
        # 创建独立的loss显示条
        self.loss_bar = tqdm(
            total=1,
            desc="Losses",
            position=1,
            leave=False,
            bar_format="{desc}: {postfix}",
            file=self.main_bar.fp
        )
        self.loss_bar.set_postfix_str("total=0.000 | imp=0.000 | cross=0.000 | self=0.000 | feat=0.000")
        
        return self.main_bar

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 调用父类方法来更新主进度条
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        
        # 更新loss显示
        if self.loss_bar is not None:
            loss_metrics = self.get_loss_metrics(trainer, pl_module)
            loss_str = " | ".join([
                f"total={loss_metrics.get('total_loss', 0):.3f}",
                f"imp={loss_metrics.get('imp_loss', 0):.3f}", 
                f"cross={loss_metrics.get('crossing_loss', 0):.3f}",
                f"self={loss_metrics.get('self_loss', 0):.3f}",
                f"feat={loss_metrics.get('feature_total_loss', 0):.3f}"
            ])
            self.loss_bar.set_postfix_str(loss_str)
            self.loss_bar.update(0)  # 刷新显示
    
    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        # 不在每个epoch结束时关闭loss条，让验证流程来管理
        # loss条会在on_validation_start时关闭，在on_validation_end时重新创建
        pass
    
    def on_train_end(self, trainer, pl_module):
        """训练完全结束时关闭loss条"""
        super().on_train_end(trainer, pl_module) if hasattr(super(), 'on_train_end') else None
        if self.loss_bar is not None:
            self.loss_bar.close()
            self.loss_bar = None

    def init_validation_tqdm(self):
        # 创建主验证进度条
        self.val_main_bar = super().init_validation_tqdm()
        
        # 创建独立的验证指标显示条
        from tqdm import tqdm
        self.val_metrics_bar = tqdm(
            total=1,
            desc="Val Metrics",
            position=1,
            leave=False,
            bar_format="{desc}: {postfix}",
            file=self.val_main_bar.fp
        )
        self.val_metrics_bar.set_postfix_str("acc=0.000 | f1_micro=0.000 | f1_macro=0.000 | f1_samples=0.000 | f1_weighted=0.000")
        
        return self.val_main_bar

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        
        # 更新验证指标显示
        if hasattr(self, 'val_metrics_bar') and self.val_metrics_bar is not None:
            # 从当前的validation_step_outputs获取最新指标
            if hasattr(pl_module, 'validation_step_outputs') and pl_module.validation_step_outputs:
                latest_output = pl_module.validation_step_outputs[-1]
                
                # 根据数据集类型显示不同指标
                if pl_module.args.dataset in ["food", "snli", "chestxray"]:
                    # 单标签分类：显示准确率
                    pred_label = latest_output['pred_label']
                    gt_label = latest_output['gt_label']
                    acc = (pred_label == gt_label).float().mean().item()
                    metrics_str = f"acc={acc:.3f}"
                else:
                    # 多标签分类：显示各种F1分数
                    pred_label = latest_output['pred_label']
                    gt_label = latest_output['gt_label']
                    
                    # 计算当前batch的F1分数
                    try:
                        from sklearn.metrics import f1_score
                        f1_micro = f1_score(gt_label.numpy(), pred_label.numpy(), average="micro")
                        f1_macro = f1_score(gt_label.numpy(), pred_label.numpy(), average="macro")
                        f1_samples = f1_score(gt_label.numpy(), pred_label.numpy(), average="samples")
                        f1_weighted = f1_score(gt_label.numpy(), pred_label.numpy(), average="weighted")
                        
                        metrics_str = " | ".join([
                            f"f1_micro={f1_micro:.3f}",
                            f"f1_macro={f1_macro:.3f}",
                            f"f1_samples={f1_samples:.3f}",
                            f"f1_weighted={f1_weighted:.3f}"
                        ])
                    except:
                        metrics_str = "f1_micro=0.000 | f1_macro=0.000 | f1_samples=0.000 | f1_weighted=0.000"
                
                self.val_metrics_bar.set_postfix_str(metrics_str)
                self.val_metrics_bar.update(0)  # 刷新显示
    
    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if hasattr(self, 'val_metrics_bar') and self.val_metrics_bar is not None:
            self.val_metrics_bar.close()
            self.val_metrics_bar = None
    
    def on_validation_start(self, trainer, pl_module):
        """验证开始时临时关闭训练loss显示条"""
        if self.loss_bar is not None:
            self.loss_bar.close()
            self.loss_bar = None
    
    def on_validation_end(self, trainer, pl_module):
        """验证结束后重新创建训练loss显示条"""
        # 如果主进度条存在且还在训练过程中，重新创建loss显示条
        if (self.main_bar is not None and 
            hasattr(trainer, 'state') and 
            trainer.state.fn == "fit" and 
            self.loss_bar is None):
            from tqdm import tqdm
            self.loss_bar = tqdm(
                total=1,
                desc="Losses",
                position=1,
                leave=False,
                bar_format="{desc}: {postfix}",
                file=self.main_bar.fp
            )
            self.loss_bar.set_postfix_str("total=0.000 | imp=0.000 | cross=0.000 | self=0.000 | feat=0.000")

# 修改处：在文件末尾添加从checkpoint直接运行模型的函数
def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    修改处：从checkpoint加载模型用于直接推理
    
    Args:
        checkpoint_path (str): checkpoint文件路径
        device (str): 运行设备
    
    Returns:
        model: 加载好的模型
    """
    try:
        # 加载checkpoint以检查其内容
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 获取保存的hyperparameters
        if 'hyper_parameters' in checkpoint:
            saved_args = checkpoint['hyper_parameters']
            print(f"从checkpoint加载的参数: {list(saved_args.keys())}")
            
            # 确保所有必要的参数都存在
            required_params = {
                'dataset': 'imdb',
                'max_length': 512,
                'backbone': 'swinb_224',
                'use_vpt': False,
                'ignore_label': -100,
                'lr': 1e-4,
                'wd': 1e-4,
                'scheduler': 'cosine',
                'warmup_epochs': 5,
                'max_epoch': 20,
                'exp_name': 'imdb_inference'
            }
            
            # 添加缺失的参数
            for key, default_value in required_params.items():
                if key not in saved_args:
                    saved_args[key] = default_value
                    print(f"添加缺失参数 {key}: {default_value}")
        else:
            # 如果没有hyperparameters，创建默认参数
            print("警告: checkpoint中没有找到hyperparameters，使用默认参数")
            saved_args = {
                'dataset': 'imdb',
                'max_length': 512,
                'backbone': 'swinb_224',
                'use_vpt': False,
                'ignore_label': -100,
                'lr': 1e-4,
                'wd': 1e-4,
                'scheduler': 'cosine',
                'warmup_epochs': 5,
                'max_epoch': 20,
                'exp_name': 'imdb_inference'
            }
        
        # 加载模型
        model = Model.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            strict=False  # 允许缺失的键以保持灵活性
        )
        
        # 更新模型的args
        model.overwrite_args(argparse.Namespace(**saved_args))
        
        model.eval()
        model.freeze()
        print(f"模型已从 {checkpoint_path} 成功加载")
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def inference_from_checkpoint(checkpoint_path, data_loader, device='cuda'):
    """
    修改处：使用checkpoint进行推理
    
    Args:
        checkpoint_path (str): checkpoint文件路径
        data_loader: 数据加载器
        device (str): 运行设备
    
    Returns:
        list: 推理结果列表
    """
    model = load_model_from_checkpoint(checkpoint_path, device)
    if model is None:
        return []
    
    model = model.to(device)
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                text_input = batch["text"]
                img_input = batch["image"].to(device)
                gt_label = batch["label"].to(device)
                
                # 模型推理
                outputs, extra_out = model.classifier(img_input, text_input)
                
                # 获取预测结果
                if model.args.dataset in ["food", "snli", "chestxray"]:
                    # 单标签分类
                    probs = F.softmax(outputs, dim=1)
                    pred_labels = torch.argmax(outputs, dim=1)
                    max_probs = torch.max(probs, dim=1)[0]
                    
                    predictions.extend(pred_labels.cpu().numpy().tolist())
                    confidences.extend(max_probs.cpu().numpy().tolist())
                    
                elif model.args.dataset in ["imdb", "rocov2"]:
                    # 多标签分类
                    probs = torch.sigmoid(outputs)
                    pred_labels = (probs > 0.5).int()
                    avg_probs = probs.mean(dim=1)  # 平均置信度
                    
                    predictions.extend(pred_labels.cpu().numpy().tolist())
                    confidences.extend(avg_probs.cpu().numpy().tolist())
                
                if batch_idx % 50 == 0:
                    print(f"已处理 {batch_idx} 个batch")
                    
            except Exception as e:
                print(f"处理batch {batch_idx} 时出错: {e}")
                continue
    
    # 保存推理结果
    results = {
        'predictions': predictions,
        'confidences': confidences,
        'total_samples': len(predictions)
    }
    
    # 保存到checkpoint同目录
    checkpoint_dir = os.path.dirname(checkpoint_path)
    results_file = os.path.join(checkpoint_dir, 'inference_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"推理完成，共处理 {len(predictions)} 个样本")
    print(f"结果已保存到: {results_file}")
    
    return results

# 修改处：添加便捷的推理脚本函数
def run_inference_script(checkpoint_path, dataset_name, data_root, batch_size=32):
    """
    修改处：便捷的推理脚本，可以直接调用
    
    Args:
        checkpoint_path (str): checkpoint路径
        dataset_name (str): 数据集名称 ('food', 'imdb', 'snli', 'chestxray', 'rocov2')
        data_root (str): 数据根目录
        batch_size (int): batch大小
    """
    print(f"开始从checkpoint进行推理...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Data root: {data_root}")
    
    # 创建数据加载器
    _, _, test_loader = create_loaders(
        data_root, batch_size, num_workers=4, n_shot=0, backbone="swinb_224"
    )
    
    # 执行推理
    results = inference_from_checkpoint(checkpoint_path, test_loader)
    
    return results


def main():
    args = get_arguments()
    seed_everything(args.random_seed)

    if args.dataset == "food":
        data_path = "data/food-101"
    elif args.dataset == "imdb":
        data_path = "data/mmimdb"
    elif args.dataset == "snli":
        data_path = "data/snli-ve/data"
    elif args.dataset == "chestxray":
        data_path = "data/chestXRay"  # 修改处: specify ChestXRay data directory
    elif args.dataset == "rocov2":
        data_path = "data/ROCOv2-radiology-main"# 修改处: specify ROCOv2 data directory
    if args.dataset == "food":
        batch_size = 8
    elif args.dataset == "chestxray":
        batch_size = 32  # 修改处: default batch size for chestxray (can be overridden)
    elif args.dataset == "rocov2":
        batch_size = 32  # default batch size for rocov2 (can be overridden by --batch-size)
    elif args.dataset == "imdb":
        batch_size = 8
    elif args.dataset == "snli":
        batch_size = 128
    # batch_size = 64 if args.dataset == "snli" else 32
    batch_size_arg = args.batch_size
    batch_size = min(batch_size, batch_size_arg)

    train_loader, val_loader, test_loader = create_loaders(
        data_path,
        batch_size,
        args.num_workers,
        args.n_shot,
        args.backbone,
    )

    model = Model(args)
    
    # 修改处：自定义ModelCheckpoint类，添加保存确认输出
    class VerboseModelCheckpoint(pl.callbacks.ModelCheckpoint):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            super().on_save_checkpoint(trainer, pl_module, checkpoint)
            if self.best_model_path:
                print(f"最佳模型已保存！")
                print(f"路径: {self.best_model_path}")
                print(f"指标: {self.monitor}={self.best_model_score:.4f}")
                print(f"Epoch: {trainer.current_epoch}")
    
    if args.dataset == "food" or args.dataset == "snli" or args.dataset == "chestxray":
        save_callback = VerboseModelCheckpoint(
            monitor="val_acc",
            filename="{epoch:02d}-{val_acc:.2f}",
            save_top_k=1,
            mode="max",
        )
    elif args.dataset == "imdb" or args.dataset == "rocov2":  # for mm-imdb the metric is f1 score instead of acc.
        save_callback = VerboseModelCheckpoint(
            monitor="val_f1_micro",
            filename="{epoch:02d}-{f1_micro:.2f}",
            save_top_k=1,
            mode="max",
        )

    #修改处
    logger = pl.loggers.TensorBoardLogger("logs", name=args.exp_name)

    max_epoch = args.max_epoch

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",  # 自动选择可用的单个或多个 GPU
        precision=16,
        logger=logger,
        max_epochs=max_epoch,
        val_check_interval=0.33,
        gradient_clip_val=0.5,  # 从1.0降到0.5
        gradient_clip_algorithm="norm",
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[MyProgressBar(), save_callback]  # 修改处：添加save_callback到callbacks列表
    )

    if args.evaluate:
        if args.ckpt is not None and args.ckpt != "":

            model = Model.load_from_checkpoint(args.ckpt, strict=False)
            model.overwrite_args(args)
            # save all scripts to logger dir
            model.eval()
            trainer.validate(model, test_loader)
        else:
            raise Warning("Trying to evaluate model but with no checkpoint provided") 
        model.benchmark_inference_speed()
        model.benchmark_memory()
    else:
        # assert args.ckpt is not None
        if args.ckpt is not None and args.ckpt != "":
            model = Model.load_from_checkpoint(args.ckpt, strict=False)
        # save all scripts to logger dir
        os.system("cp -r *py models utils %s" % logger.log_dir)
        trainer.fit(model, train_loader, val_loader)
        trainer.validate(model, test_loader)
        # 修改处：训练完成后自动运行推理
        best_model_path = save_callback.best_model_path
        if best_model_path:
            print(f"训练完成，最佳模型路径: {best_model_path}")
            print("运行最佳模型推理...")
            run_inference_script(best_model_path, args.dataset, data_path, batch_size)



if __name__ == "__main__":

    main()

