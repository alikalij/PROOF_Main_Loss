import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import  Proof_Net
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
from utils.data_manager import LaionData
import math
import matplotlib.pyplot as plt
import os
import psutil
import copy
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

num_workers = 0
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args=args

        self._train_transformer=False
        self._network = Proof_Net(args, False)
        
        self.batch_size = get_attribute(args,"batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)
        
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        
        self._known_classes = 0
        self.use_cos = get_attribute(args, "use_cos", False)

        # Knowledge distillation / optimizer / scheduler options
        self.prev_network = None                 # will hold snapshot of previous model (teacher)
        self.kd_alpha = get_attribute(args, "kd_alpha", 0.5)    # weight for KD loss
        self.kd_temp = get_attribute(args, "kd_temp", 2.0)      # temperature for KD
        self.label_smoothing = get_attribute(args, "label_smoothing", 0.1)
        self.use_onecycle = get_attribute(args, "use_onecycle", True)
        self.max_lr = get_attribute(args, "max_lr", max(self.init_lr*10, 0.02))
        # AMP scaler
        self.scaler = GradScaler()

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
    
    def cal_prototype(self,trainloader, model):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding=model.convnet.encode_image(data, True)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list=list(range(self._known_classes, self._total_classes))
        for class_index in class_list:
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            self._network.img_prototypes[class_index]=proto

    def incremental_train(self, data_manager):

        p=psutil.Process(os.getpid())
        print("RSS MB:", p.memory_info().rss/1024**2)

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        self._network.update_prototype(self._total_classes)
        self._network.update_context_prompt() # add context prompts

        self._network.extend_task()
        
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
            source="train", mode="train", appendent=self._get_memory())
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self._network.to(self._device)
       
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

     #   train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test", )
     #   self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test" )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self.cal_prototype(self.train_loader_for_protonet, self._network)
        
        # Save teacher snapshot (previous model) for KD if not first task
        if self._cur_task > 0:
            # make a detached eval copy on device (no grad)
            self.prev_network = copy.deepcopy(self._network).eval().to(self._device)
        else:
            self.prev_network = None

        
        self._train_proj(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def _train_proj(self, train_loader, test_loader, train_loader_for_protonet=None):
        """
        Training projection heads (updated): uses label smoothing, optional OneCycleLR,
        knowledge distillation from self.prev_network, SGD momentum, and AMP.
        """
        self._train_transformer = True
        self._network.to(self._device)
        # Criterion with label smoothing
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # Freeze encoder except logit_scale etc. (same logic as before)
        for name, param in self._network.convnet.named_parameters():
            if "logit_scale" not in name:
                param.requires_grad = False
        # ensure projection weights are unfrozen by the model helper
        self._network.freeze_projection_weight_new()

        # Build optimizer only over parameters that require grad
        trainable_params = [p for p in self._network.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            logging.warning("[Train] No trainable params found! Are you sure projections are unfrozen?")
        optimizer = optim.SGD(trainable_params, lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)

        # Scheduler: prefer OneCycleLR if enough steps, else CosineAnnealingLR
        try:
            steps_per_epoch = max(1, len(train_loader))
        except Exception:
            steps_per_epoch = 1

        if self.use_onecycle and steps_per_epoch > 0:
            scheduler = OneCycleLR(optimizer, max_lr=self.max_lr, steps_per_epoch=steps_per_epoch, epochs=self.tuned_epoch)
            logging.info(f"[Train] Using OneCycleLR max_lr={self.max_lr}")
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, self.tuned_epoch), eta_min=self.min_lr)
            logging.info("[Train] Using CosineAnnealingLR")

        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        prog_bar = tqdm(range(self.tuned_epoch))
        cliploss = ClipLoss()

        total_labels = class_to_label[:self._total_classes]  # all known classes masked
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # prepare text features for all classes (same as before)
                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                text_features = self._network.encode_text(texts)
                text_feas = text_features / text_features.norm(dim=-1, keepdim=True)

                # image features (raw)
                with autocast():
                    image_features = self._network.encode_image(inputs)
                    img_feas = image_features / image_features.norm(dim=-1, keepdim=True)

                    # forward transformer/projection
                    transf_image_features, transf_text_features, logit_scale, proto_feas = self._network.forward_transformer(img_feas, text_feas, self._train_transformer)

                    # main logits (student)
                    logits = transf_image_features @ transf_text_features.T  # [bs, total_classes]

                    # proto-based outputs (student)
                    proto_outputs = transf_image_features @ proto_feas.T

                    # CLIP consistency: text features for actual labels
                    labels = [class_to_label[y] for y in targets.cpu().tolist()]
                    clip_texts = [templates.format(inst) for inst in labels]
                    clip_text_feas = self._network.encode_text(self._network.tokenizer(clip_texts).to(self._device))
                    clip_text_feas = clip_text_feas / clip_text_feas.norm(dim=-1, keepdim=True)
                    clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)

                    # classification loss (with label smoothing already in criterion)
                    loss_ce = criterion(logits, targets)

                # knowledge distillation from prev_network if available (use raw image features logits as teacher)
                kd_loss = 0.0
                if getattr(self, "prev_network", None) is not None:
                    with torch.no_grad():
                        # teacher raw image features (before current projections)
                        teacher_img = self.prev_network.encode_image(inputs)
                        # teacher logits w.r.t current text_features (so dims match)
                        teacher_logits = teacher_img @ text_features.T
                    # student uses 'logits' computed above
                    sd_logits = logits
                    T = self.kd_temp
                    kd_loss = F.kl_div(F.log_softmax(sd_logits / T, dim=1),
                                    F.softmax(teacher_logits / T, dim=1),
                                    reduction="batchmean") * (T * T)

                # combine losses with selected weights (can tune these coefficients)
                total_loss = loss_ce + 0.4 * clip_loss + 0.6 * F.cross_entropy(transf_image_features @ proto_feas.T, targets) + self.kd_alpha * kd_loss

                # backward with AMP
                optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                losses += total_loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            # step scheduler
            try:
                scheduler.step()
            except Exception:
                pass

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task, epoch + 1, self.tuned_epoch, losses / max(1, len(train_loader)), train_acc, test_acc
            )
            prog_bar.set_description(info)
            logging.info("[Train] " + info)


    def _compute_accuracy(self, model, loader):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes] # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features=self._network.encode_image(inputs)
                transf_image_features, transf_text_features, _, proto_feas = self._network.forward_transformer(image_features, text_features,self._train_transformer)
                outputs = transf_image_features @ transf_text_features.T
                proto_outputs= transf_image_features @ proto_feas.T
                original_outputs= image_features @ text_features.T
                outputs = original_outputs+outputs+proto_outputs

            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


    def _eval_cnn(self, loader):
        
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes] # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features=self._network.encode_image(inputs)
                transf_image_features, transf_text_features, _, proto_feas = self._network.forward_transformer(image_features, text_features,self._train_transformer)

                outputs = transf_image_features @ transf_text_features.T

                proto_outputs= transf_image_features @ proto_feas.T

                original_outputs= image_features @ text_features.T

                outputs = original_outputs+outputs+proto_outputs

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


