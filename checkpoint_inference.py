#!/usr/bin/env python3
"""
简单的checkpoint推理脚本
可以选择在验证集或测试集上进行推理，完全复制训练时的validation逻辑
"""

import os
import torch
import json
import argparse
from sklearn.metrics import f1_score
import torch.nn.functional as F

# 导入项目模块
from main_classify import Model
from utils.datasets import create_loaders
import tqdm


def save_detailed_results(validation_step_outputs, save_dir, filename):
    """保存详细的验证结果到JSONL文件，完全复制原始训练时的逻辑"""
    try:
        out_path = os.path.join(save_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            for step_out in validation_step_outputs:
                image_paths = step_out.get('image_paths', None)
                texts = step_out['text']
                preds = step_out['pred_label']
                # probs = step_out['probs']
                labels = step_out['gt_label']
                
                # 处理批次中的每个样本
                batch_size = len(texts)
                for i in range(batch_size):
                    record = {
                        'image': image_paths[i] if image_paths is not None else None,
                        'text': texts[i],
                        'pred': preds[i].tolist() if isinstance(preds[i], torch.Tensor) else preds[i],
                        # 'prob': probs[i].tolist() if isinstance(probs[i], torch.Tensor) else probs[i],
                        'label': labels[i].tolist() if isinstance(labels[i], torch.Tensor) else labels[i],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"详细验证结果已保存到: {out_path}")
        return out_path
    except Exception as e:
        print(f"保存详细验证结果失败: {e}")
        return None

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """从checkpoint加载模型"""
    if not os.path.exists(checkpoint_path):
        print(f"错误: Checkpoint文件不存在: {checkpoint_path}")
        return None
    
    try:
        print("正在加载模型...")
        model = Model.load_from_checkpoint(
            checkpoint_path, 
            map_location=device, 
            strict=False
        )
        model = model.to(device)
        model.eval()
        print("模型加载成功!")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None


def run_inference(model, data_loader, device='cuda', dataset_type='food', save_dir=None, use_test=False):
    """运行推理，完全复制原始validation_step的逻辑"""
    # 处理tqdm包装的data_loader
    if hasattr(data_loader, 'iterable'):
        dataset_size = len(data_loader.iterable.dataset)
    else:
        dataset_size = len(data_loader.dataset)
    
    print(f"开始推理，数据集大小: {dataset_size} 样本")
    
    # 存储所有validation step的输出（完全复制原始逻辑）
    validation_step_outputs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                # 完全按照原始validation_step的逻辑
                text_input = batch["text"]
                img_input = batch["image"].to(device)
                gt_label = batch["label"].to(device)
                
                # 获取image_paths（如果有的话）
                image_paths = batch.get('image_path', None)
                
                # 计算模型输出（完全按照原始逻辑）
                outputs, extra_out = model.classifier(img_input, text_input)
                
                # 计算概率和预测（完全按照原始逻辑）
                if dataset_type in ["imdb", "rocov2"]:
                    probs = torch.sigmoid(outputs)
                    pred_label = (probs > 0.35).int()
                elif dataset_type in ["food", "snli", "chestxray"]:
                    probs = F.softmax(outputs, dim=1)
                    pred_label = torch.argmax(outputs, dim=1)
                else:
                    # 默认处理方式
                    probs = F.softmax(outputs, dim=1)
                    pred_label = torch.argmax(outputs, dim=1)
                
                # 计算损失（完全按照原始逻辑）
                loss_val = model.loss(outputs, gt_label.squeeze())
                
                # 构建返回字典（完全按照原始逻辑）
                ret_dict = {
                    'image_paths': image_paths,
                    'text': text_input,
                    'pred_label': pred_label.detach().cpu(),
                    'probs': probs.detach().cpu(),
                    'gt_label': gt_label.squeeze().detach().cpu(),
                    'loss': loss_val.detach().cpu(),
                }
                # print(extra_out)
                
                if extra_out is not None:
                    if "moe_scores" in extra_out:
                        ret_dict["moe_scores"] = extra_out["moe_scores"]
                    if "cls_" in extra_out:
                        ret_dict["cls"] = extra_out["cls"].detach().cpu()
                
                validation_step_outputs.append(ret_dict)
                
            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {e}")
                continue
    
    # 计算最终指标（完全按照原始on_validation_epoch_end的逻辑）
    if dataset_type == "food" or dataset_type == "snli" or dataset_type == "chestxray":
        all_cnt = 0
        correct_cnt = 0
        for step_out in validation_step_outputs:
            pred_label = step_out["pred_label"]
            gt_label = step_out["gt_label"]

            all_cnt += pred_label.size(0)
            correct_cnt += torch.sum(pred_label == gt_label).item()
        
        acc = correct_cnt / all_cnt
        print(f"\n=== 最终结果 ===")
        print(f"准确率 (Accuracy): {acc:.4f} ({acc*100:.2f}%)")
        
        # 保存详细验证结果到JSONL文件（完全按照原始逻辑）
        if save_dir:
            filename = "inference_details_test.jsonl" if use_test else "inference_details_val.jsonl"
            save_detailed_results(validation_step_outputs, save_dir, filename)
        
        return {"accuracy": acc}
        
    elif dataset_type == "imdb" or dataset_type == "rocov2":
        all_preds = []
        all_labels = []
        
        for step_out in validation_step_outputs:
            pred_label = step_out["pred_label"]
            gt_label = step_out["gt_label"]
            all_preds.append(pred_label)
            all_labels.append(gt_label)

        
        # 计算F1分数
        f1_macro = f1_score(
            torch.cat(all_labels).squeeze().cpu().numpy(),
            torch.cat(all_preds).cpu().numpy(),
            average="macro",
        )
        f1_micro = f1_score(
            torch.cat(all_labels).squeeze().cpu().numpy(),
            torch.cat(all_preds).cpu().numpy(),
            average="micro",
        )
        f1_samples = f1_score(
            torch.cat(all_labels).squeeze().cpu().numpy(),
            torch.cat(all_preds).cpu().numpy(),
            average="samples",
        )
        f1_weighted = f1_score(
            torch.cat(all_labels).squeeze().cpu().numpy(),
            torch.cat(all_preds).cpu().numpy(),
            average="weighted",
        )
        
        # 计算exact match accuracy
        preds_tensor = torch.cat(all_preds)
        labels_tensor = torch.cat(all_labels)
        exact_matches = (preds_tensor == labels_tensor).all(dim=1)
        val_acc = exact_matches.float().mean().item()
        
        print(f"\n=== 最终结果 ===")
        print(f"准确率 (Exact Match): {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Micro: {f1_micro:.4f}")
        print(f"F1 Samples: {f1_samples:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        
        # 保存详细验证结果到JSONL文件（完全按照原始逻辑）
        if save_dir:
            filename = "inference_details_test.jsonl" if use_test else "inference_details_val.jsonl"
            save_detailed_results(validation_step_outputs, save_dir, filename)
        
        return {
            "accuracy": val_acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_samples": f1_samples,
            "f1_weighted": f1_weighted
        }


def main():
    parser = argparse.ArgumentParser(description='从checkpoint进行推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint文件路径')
    parser.add_argument('--dataset', type=str, default='food', 
                       choices=['food', 'imdb', 'snli', 'chestxray', 'rocov2'],
                       help='数据集类型')
    parser.add_argument('--data_root', type=str, default='data/food-101', help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='batch大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--use_test', action='store_true', 
                       help='使用测试集而不是验证集（默认使用验证集，与训练时保持一致）')
    
    args = parser.parse_args()
    
    # 根据数据集类型设置正确的数据路径（与main_classify.py保持一致）
    if args.dataset == "food":
        data_path = "data/food-101"
    elif args.dataset == "imdb":
        data_path = "data/mmimdb"
    elif args.dataset == "snli":
        data_path = "data/snli-ve/data"  # SNLI特殊路径配置
    elif args.dataset == "chestxray":
        data_path = "data/chestXRay"
    elif args.dataset == "rocov2":
        data_path = "data/ROCOv2-radiology-main"
    else:
        data_path = args.data_root  # 回退到用户指定的路径
    
    # 如果用户没有明确指定data_root，使用配置的路径
    if args.data_root == 'data/food-101':  # 默认值，说明用户没有明确指定
        args.data_root = data_path
    
    print("=== Checkpoint推理脚本 ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"数据集: {args.dataset}")
    print(f"数据根目录: {args.data_root}")
    print(f"使用数据: {'测试集' if args.use_test else '验证集（与训练时一致）'}")
    print(f"Batch大小: {args.batch_size}")
    print(f"设备: {args.device}")
    
    # 加载模型
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    if model is None:
        return
    
    # 创建数据加载器
    print("\n正在创建数据加载器...")
    try:
        train_loader, val_loader, test_loader = create_loaders(
            args.data_root, 
            args.batch_size, 
            num_workers=4, 
            n_shot=0, 
            backbone="swinb_224"
        )
        
        # 选择使用的数据加载器
        eval_loader = test_loader if args.use_test else val_loader
        loader_name = "测试集" if args.use_test else "验证集"
        
        print(f"{loader_name}大小: {len(eval_loader.dataset)} 样本")
        print(f"批次数量: {len(eval_loader)}")
        
    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        return
    
    # 运行推理
    print(f"\n开始在{loader_name}上推理...")
    #使用tqdm显示验证进度，显示进度条
    eval_loader_with_progress = tqdm.tqdm(eval_loader, desc="推理进度", unit="批次")
    
    # 获取保存目录（checkpoint所在的version目录）
    checkpoint_dir = os.path.dirname(args.checkpoint)
    
    results = run_inference(model, eval_loader_with_progress, args.device, args.dataset, save_dir=checkpoint_dir, use_test=args.use_test)
    
    # 保存结果
    checkpoint_dir = os.path.dirname(args.checkpoint)
    results_file = os.path.join(checkpoint_dir, f'inference_results_{"test" if args.use_test else "val"}.json')
    
    # 获取数据集大小
    if hasattr(eval_loader_with_progress, 'iterable'):
        dataset_size = len(eval_loader_with_progress.iterable.dataset)
    else:
        dataset_size = len(eval_loader.dataset)
    
    results_data = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'data_root': args.data_root,
        'use_test': args.use_test,
        'eval_type': 'test' if args.use_test else 'validation',
        'results': results,
        'sample_count': dataset_size
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")


if __name__ == "__main__":
    main()





# #!/usr/bin/env python3
# """
# 简单的checkpoint推理脚本
# 可以选择在验证集或测试集上进行推理，完全复制训练时的validation逻辑
# """

# import os
# import json
# import argparse
# from typing import Any, Dict, List, Optional

# import torch
# import torch.nn.functional as F
# from sklearn.metrics import f1_score

# # 导入项目模块
# from main_classify import Model
# from utils.datasets import create_loaders
# import tqdm


# def _extract_args_from_checkpoint(checkpoint_path: str) -> argparse.Namespace:
#     """Restore training args saved inside the Lightning checkpoint."""
#     ckpt = torch.load(checkpoint_path, map_location="cpu")
#     hparams = ckpt.get("hyper_parameters", {})

#     if isinstance(hparams, dict) and "args" in hparams:
#         args_obj = hparams["args"]
#     else:
#         args_obj = hparams

#     if isinstance(args_obj, argparse.Namespace):
#         return args_obj
#     if hasattr(args_obj, "to_dict"):
#         args_obj = args_obj.to_dict()
#     if isinstance(args_obj, dict):
#         return argparse.Namespace(**args_obj)

#     raise ValueError("无法从checkpoint提取训练参数，请确认文件是否完整")


# def _apply_cli_overrides(saved_args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
#     for key, value in overrides.items():
#         if value is not None:
#             setattr(saved_args, key, value)
#     return saved_args


# def _resolve_data_root(dataset: str, override_root: Optional[str]) -> str:
#     if override_root:
#         return override_root
#     if dataset in _DEFAULT_DATA_PATHS:
#         return _DEFAULT_DATA_PATHS[dataset]
#     raise ValueError(f"未知的数据集 {dataset}，请使用 --data_root 指定数据路径")


# def _effective_batch_size(dataset: str, saved_value: Optional[int], override_value: Optional[int]) -> int:
#     dataset = dataset.lower()
#     candidate = override_value if override_value is not None else saved_value
#     if candidate is None:
#         candidate = _DATASET_MAX_BATCH.get(dataset, 32)

#     max_cap = _DATASET_MAX_BATCH.get(dataset)
#     if max_cap is None:
#         return candidate
#     return min(max_cap, candidate)


# _DEFAULT_DATA_PATHS: Dict[str, str] = {
#     "food": "data/food-101",
#     "imdb": "data/mmimdb",
#     "snli": "data/snli-ve/data",
#     "chestxray": "data/chestXRay",
#     "rocov2": "data/ROCOv2-radiology-main",
# }

# _DATASET_MAX_BATCH: Dict[str, int] = {
#     "food": 8,
#     "imdb": 16,
#     "snli": 128,
#     "chestxray": 32,
# }


# def save_detailed_results(validation_step_outputs, save_dir, filename):
#     """保存详细的验证结果到JSONL文件，完全复制原始训练时的逻辑"""
#     try:
#         out_path = os.path.join(save_dir, filename)
#         with open(out_path, 'w', encoding='utf-8') as f:
#             for step_out in validation_step_outputs:
#                 image_paths = step_out.get('image_paths', None)
#                 texts = step_out['text']
#                 preds = step_out['pred_label']
#                 # probs = step_out['probs']
#                 labels = step_out['gt_label']
                
#                 # 处理批次中的每个样本
#                 batch_size = len(texts)
#                 for i in range(batch_size):
#                     record = {
#                         'image': image_paths[i] if image_paths is not None else None,
#                         'text': texts[i],
#                         'pred': preds[i].tolist() if isinstance(preds[i], torch.Tensor) else preds[i],
#                         # 'prob': probs[i].tolist() if isinstance(probs[i], torch.Tensor) else probs[i],
#                         'label': labels[i].tolist() if isinstance(labels[i], torch.Tensor) else labels[i],
#                     }
#                     f.write(json.dumps(record, ensure_ascii=False) + '\n')
#         print(f"详细验证结果已保存到: {out_path}")
#         return out_path
#     except Exception as e:
#         print(f"保存详细验证结果失败: {e}")
#         return None

# def load_model_from_checkpoint(
#     checkpoint_path: str,
#     device: str = "cuda",
#     overrides: Optional[Dict[str, Any]] = None,
# ):
#     """从checkpoint加载模型并返回模型及其参数"""
#     if not os.path.exists(checkpoint_path):
#         print(f"错误: Checkpoint文件不存在: {checkpoint_path}")
#         return None, None

#     try:
#         print("正在加载模型...")
#         saved_args = _extract_args_from_checkpoint(checkpoint_path)
#         overrides = overrides or {}
#         saved_args = _apply_cli_overrides(saved_args, overrides)

#         model = Model.load_from_checkpoint(
#             checkpoint_path,
#             map_location=device,
#             strict=False,
#             args=saved_args,
#         )
#         model = model.to(device)
#         model.eval()
#         print("模型加载成功!")
#         return model, saved_args
#     except Exception as e:
#         print(f"模型加载失败: {e}")
#         return None, None


# def run_inference(
#     model: Model,
#     data_loader,
#     dataset_type: str,
#     device: str = "cuda",
#     threshold: float = 0.5,
#     save_dir: Optional[str] = None,
#     save_details: bool = True,
#     use_test: bool = False,
# ):
#     """运行推理流程，复现训练/验证阶段的计算逻辑"""

#     if hasattr(data_loader, "iterable"):
#         dataset_size = len(data_loader.iterable.dataset)
#     else:
#         dataset_size = len(data_loader.dataset)

#     dataset_key = dataset_type.lower()
#     print(f"开始推理，数据集大小: {dataset_size} 样本")

#     step_outputs: List[Dict[str, Any]] = []
#     losses: List[torch.Tensor] = []

#     with torch.no_grad():
#         for batch_idx, batch in enumerate(data_loader):
#             try:
#                 text_input = batch["text"]
#                 img_input = batch["image"].to(device)
#                 gt_label = batch["label"].to(device)

#                 image_paths = batch.get("image_path") or batch.get("image_paths")
#                 if image_paths is not None:
#                     if isinstance(image_paths, str):
#                         image_paths = [image_paths]
#                     elif not isinstance(image_paths, (list, tuple)):
#                         image_paths = list(image_paths)

#                 logits, extra_out = model.classifier(img_input, text_input)

#                 if dataset_key in {"food", "snli", "chestxray"}:
#                     log_probs = model.final_act(logits)
#                     loss_val = model.loss(log_probs, gt_label.squeeze())
#                     probs = torch.exp(log_probs)
#                     pred_label = torch.argmax(log_probs, dim=1)
#                     metric_targets = gt_label.squeeze()
#                 elif dataset_key in {"imdb", "rocov2"}:
#                     target = gt_label.squeeze().float()
#                     loss_val = model.loss(logits, target)
#                     probs = model.final_act(logits)
#                     pred_label = (probs > threshold).int()
#                     metric_targets = target
#                 else:
#                     raise ValueError(f"暂不支持的数据集: {dataset_type}")

#                 losses.append(loss_val.detach().cpu())

#                 ret_dict = {
#                     "image_paths": image_paths,
#                     "text": list(text_input) if isinstance(text_input, (list, tuple)) else [text_input],
#                     "pred_label": pred_label.detach().cpu(),
#                     "probs": probs.detach().cpu(),
#                     "gt_label": metric_targets.detach().cpu(),
#                     "loss": loss_val.detach().cpu(),
#                 }

#                 if extra_out is not None:
#                     if "moe_scores" in extra_out:
#                         ret_dict["moe_scores"] = extra_out["moe_scores"]
#                     if "cls_" in extra_out:
#                         ret_dict["cls"] = extra_out["cls_"].detach().cpu()

#                 step_outputs.append(ret_dict)

#             except Exception as exc:
#                 print(f"处理批次 {batch_idx} 时出错: {exc}")
#                 continue

#     mean_loss = torch.stack(losses).mean().item() if losses else float("nan")

#     if dataset_key in {"food", "snli", "chestxray"}:
#         correct_cnt = 0
#         all_cnt = 0
#         for step_out in step_outputs:
#             pred_label = step_out["pred_label"]
#             gt_label = step_out["gt_label"]
#             all_cnt += pred_label.size(0)
#             correct_cnt += torch.sum(pred_label == gt_label).item()

#         acc = correct_cnt / max(all_cnt, 1)
#         print("\n=== 最终结果 ===")
#         print(f"准确率 (Accuracy): {acc:.4f} ({acc*100:.2f}%)")
#         print(f"平均损失 (Loss): {mean_loss:.6f}")

#         metrics = {"accuracy": acc, "loss": mean_loss}

#     else:
#         all_preds = [step_out["pred_label"] for step_out in step_outputs]
#         all_labels = [step_out["gt_label"] for step_out in step_outputs]

#         if not all_preds:
#             raise RuntimeError("未获取到有效的预测结果，推理失败")

#         preds_tensor = torch.cat(all_preds)
#         labels_tensor = torch.cat(all_labels)

#         labels_np = labels_tensor.cpu().numpy()
#         preds_np = preds_tensor.cpu().numpy()

#         f1_macro = f1_score(labels_np, preds_np, average="macro", zero_division=0)
#         f1_micro = f1_score(labels_np, preds_np, average="micro", zero_division=0)
#         f1_weighted = f1_score(labels_np, preds_np, average="weighted", zero_division=0)
#         f1_samples = f1_score(labels_np, preds_np, average="samples", zero_division=0)

#         exact_matches = (preds_tensor == labels_tensor).all(dim=1)
#         exact_acc = exact_matches.float().mean().item()

#         print("\n=== 最终结果 ===")
#         print(f"Exact Match Accuracy: {exact_acc:.4f} ({exact_acc*100:.2f}%)")
#         print(f"F1 Micro: {f1_micro:.4f}")
#         print(f"F1 Macro: {f1_macro:.4f}")
#         print(f"F1 Samples: {f1_samples:.4f}")
#         print(f"F1 Weighted: {f1_weighted:.4f}")
#         print(f"平均损失 (Loss): {mean_loss:.6f}")

#         metrics = {
#             "accuracy": exact_acc,
#             "f1_micro": f1_micro,
#             "f1_macro": f1_macro,
#             "f1_samples": f1_samples,
#             "f1_weighted": f1_weighted,
#             "loss": mean_loss,
#         }

#     if save_dir and save_details:
#         filename = "inference_details_test.jsonl" if use_test else "inference_details_val.jsonl"
#         save_detailed_results(step_outputs, save_dir, filename)

#     return metrics


# def main():
#     parser = argparse.ArgumentParser(description='从checkpoint进行推理')
#     parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint文件路径')
#     parser.add_argument('--dataset', type=str, default=None,
#                        choices=['food', 'imdb', 'snli', 'chestxray', 'rocov2'],
#                        help='指定推理数据集（默认使用训练时配置）')
#     parser.add_argument('--data_root', type=str, default=None, help='数据根目录（默认按照训练路径推断）')
#     parser.add_argument('--batch_size', type=int, default=None, help='batch大小（默认与训练一致）')
#     parser.add_argument('--num_workers', type=int, default=None, help='数据加载线程数（默认与训练一致）')
#     parser.add_argument('--device', type=str, default='cuda', help='推理设备')
#     parser.add_argument('--threshold', type=float, default=0.5, help='多标签分类阈值（imdb/rocov2适用）')
#     parser.add_argument('--use_test', action='store_true', 
#                        help='使用测试集而不是验证集（默认使用验证集，与训练时保持一致）')
#     parser.add_argument('--no_details', action='store_true', help='不保存逐样本推理详细结果')
    
#     args = parser.parse_args()
    
#     print("=== Checkpoint推理脚本 ===")
#     print(f"Checkpoint: {args.checkpoint}")

#     overrides = {}
#     if args.dataset is not None:
#         overrides["dataset"] = args.dataset

#     model, model_args = load_model_from_checkpoint(args.checkpoint, args.device, overrides)
#     if model is None:
#         return

#     dataset_name = getattr(model_args, "dataset", "food").lower()
#     data_root = _resolve_data_root(dataset_name, args.data_root)
#     batch_size = _effective_batch_size(dataset_name, getattr(model_args, "batch_size", None), args.batch_size)
#     num_workers = args.num_workers if args.num_workers is not None else getattr(model_args, "num_workers", 4)
#     n_shot = getattr(model_args, "n_shot", 0)
#     backbone = getattr(model_args, "backbone", "swinb_224")

#     model_args.batch_size = batch_size
#     model_args.num_workers = num_workers

#     loader_name = "测试集" if args.use_test else "验证集"

#     print(f"数据集: {dataset_name}")
#     print(f"数据根目录: {data_root}")
#     print(f"使用数据: {loader_name}")
#     print(f"Batch大小: {batch_size}")
#     print(f"Num Workers: {num_workers}")
#     print(f"设备: {args.device}")
#     if dataset_name in {"imdb", "rocov2"}:
#         print(f"多标签阈值: {args.threshold}")
#     print(f"逐样本结果: {'关闭' if args.no_details else '保存至checkpoint目录'}")

#     print("\n正在创建数据加载器...")
#     try:
#         train_loader, val_loader, test_loader = create_loaders(
#             data_root,
#             batch_size,
#             num_workers,
#             n_shot,
#             backbone,
#         )
#     except Exception as exc:
#         print(f"创建数据加载器失败: {exc}")
#         return

#     eval_loader = test_loader if args.use_test else val_loader
#     try:
#         dataset_size = len(eval_loader.dataset)
#     except AttributeError:
#         dataset_size = len(eval_loader)

#     print(f"{loader_name}大小: {dataset_size} 样本")
#     print(f"批次数量: {len(eval_loader)}")

#     print(f"\n开始在{loader_name}上推理...")
#     eval_loader_with_progress = tqdm.tqdm(eval_loader, desc=f"推理进度 ({loader_name})", unit="批次")

#     checkpoint_dir = os.path.dirname(args.checkpoint)
#     results = run_inference(
#         model,
#         eval_loader_with_progress,
#         dataset_name,
#         device=args.device,
#         threshold=args.threshold,
#         save_dir=checkpoint_dir,
#         save_details=not args.no_details,
#         use_test=args.use_test,
#     )

#     results_file = os.path.join(
#         checkpoint_dir,
#         f'inference_results_{"test" if args.use_test else "val"}.json',
#     )

#     results_payload = {
#         "checkpoint": args.checkpoint,
#         "dataset": dataset_name,
#         "data_root": data_root,
#         "use_test": args.use_test,
#         "eval_type": "test" if args.use_test else "validation",
#         "batch_size": batch_size,
#         "num_workers": num_workers,
#         "results": results,
#         "sample_count": dataset_size,
#     }
#     if dataset_name in {"imdb", "rocov2"}:
#         results_payload["threshold"] = args.threshold

#     with open(results_file, "w", encoding="utf-8") as f:
#         json.dump(results_payload, f, indent=2, ensure_ascii=False)

#     print(f"\n结果已保存到: {results_file}")


# if __name__ == "__main__":
#     main()
