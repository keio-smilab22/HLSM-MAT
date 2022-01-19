import torch
from torch import optim as optim
from torch.nn import functional as F


def calc_bert_lr(lr, gstep, warmup_steps, hold_steps, cooldown_steps):
    if gstep < warmup_steps:
        factor = gstep / warmup_steps
    elif warmup_steps <= gstep < warmup_steps + hold_steps:
        factor = 1.0
    elif warmup_steps + hold_steps <= gstep < warmup_steps + hold_steps + cooldown_steps:
        factor = (warmup_steps + hold_steps + cooldown_steps - gstep) / cooldown_steps
    else:
        factor = 0
    lro = lr * factor
    return lro


def switch_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_eval_loop(dataloader,
                    model,
                    writer,
                    val,
                    args=None,
                    optimargs=None,
                    gstep=0,
                    device="cpu",
                    optimizers=None,
                    optimizer_states=None):
    prefix = "val" if val else "train"

    if not val:
        # -------------------------------------------------------------------------------------
        # OPTIMIZER SETUP
        if optimargs is None:
            optimargs = {"lr": 0.001, "weight_decay": 1e-8}

        bert_lr = optimargs["bert"].lr
        bert_warmup_steps = optimargs["bert"].warmup_steps
        bert_hold_steps = optimargs["bert"].hold_steps
        bert_cooldown_steps = optimargs["bert"].cooldown_steps

        # Create optimizers if not already supplied
        if optimizers is None:
            all_params = {k: v for k, v in model.named_parameters()}
            bert_params = {k: v for k, v in all_params.items() if "bertmodel" in k}
            nonbert_params = {k: v for k, v in all_params.items() if "bertmodel" not in k}

            nonbert_optimizer = optim.Adam(nonbert_params.values(),
                                           lr=optimargs["nonbert"].lr,
                                           weight_decay=optimargs["nonbert"].weight_decay)
            if len(bert_params) > 0:
                bert_optimizer = optim.Adam(bert_params.values(),
                                            lr=optimargs["bert"].lr,
                                            weight_decay=optimargs["bert"].weight_decay)
            else:
                bert_optimizer = None

            # Initilize optimizers from provided states
            if optimizer_states is not None:
                nonbert_optimizer.load_state_dict(optimizer_states[0])
                if bert_optimizer is not None:
                    bert_optimizer.load_state_dict(optimizer_states[1])
        else:
            nonbert_optimizer, bert_optimizer = optimizers

        # -------------------------------------------------------------------------------------
        # ADVERSARIAL SETUP
        if args.adv_training:
            adv_steps = args.adv_steps
            adv_modality = args.adv_modality
            adv_optim = args.adv_optim

    else:
        nonbert_optimizer, bert_optimizer = None, None

    # -------------------------------------------------------------------------------------
    # LOOP

    for i, batch in enumerate(dataloader):

        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        if args.def_name == 'alfred/train_subgoal_model':
            loss, metrics, output = model(batch)
        else:
            loss, metrics = model(batch)
        loss = loss.mean()

        if not val:
            if args.adv_training:
                state_delta = torch.zeros((batch['states'].data.data.shape[0], 128), device=device)
                task_delta = torch.zeros((batch['states'].data.data.shape[0], 128), device=device)
                action_hist_delta = torch.zeros((batch['states'].data.data.shape[0], 128), device=device)
                state_v, task_v, action_hist_v = 0, 0, 0
                state_s, task_s, action_hist_s = 0, 0, 0

                for astep in range(adv_steps):
                    if 'state' in adv_modality:
                        state_delta.requires_grad_()
                    if 'task' in adv_modality:
                        task_delta.requires_grad_()
                    if 'action_hist' in adv_modality:
                        action_hist_delta.requires_grad_()

                    adv_loss, _, adv_output = model(
                        batch, adv_training=True, adv_modality=adv_modality, adv_delta_state=state_delta, adv_delta_task=task_delta, adv_delta_action_hist=action_hist_delta)
                    type_kl_loss = F.kl_div(adv_output['act_type_logprob'], output['act_type_prob'].clone().detach(), reduction='none') + \
                        F.kl_div(output['act_type_logprob'].clone().detach(), adv_output['act_type_prob'], reduction='none')
                    arg_kl_loss = F.kl_div(adv_output['act_arg_logprob'], output['act_arg_prob'].clone().detach(), reduction='none') + \
                        F.kl_div(output['act_arg_logprob'].clone().detach(), adv_output['act_arg_prob'], reduction='none')
                    mask_kl_loss = F.kl_div(adv_output['act_mask_pred_logprob_2d'], output['act_mask_pred_prob_2d'].clone().detach(), reduction='none') + \
                        F.kl_div(output['act_mask_pred_logprob_2d'].clone().detach(), adv_output['act_mask_pred_prob_2d'], reduction='none')
                    total_loss = (adv_loss.mean() + 1.5 * type_kl_loss.mean() + 1.5 * arg_kl_loss.mean() + 1.5 * mask_kl_loss.mean()) / adv_steps
                    total_loss.backward(retain_graph=True)

                    if astep == adv_steps - 1:
                        break

                    if 'state' in adv_modality:
                        state_delta_grad = state_delta.grad.clone().detach().float()
                        denorm = torch.norm(state_delta_grad.view(state_delta_grad.size(0), -1), dim=1).view(-1, 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        state_g = state_delta_grad / denorm
                        if adv_optim == 'sgd':
                            state_delta_step = (1e-3 * state_g).to(state_delta)
                        elif adv_optim == 'momentum':
                            beta = 0.9
                            state_v = beta * state_v + 1e-3 * state_g
                            state_delta_step = state_v.to(state_delta)
                        elif adv_optim == 'rmsprop':
                            beta = 0.9
                            state_v = beta * state_v + (1 - beta) * state_g ** 2
                            denorm = torch.norm(state_v.view(state_v.size(0), -1), dim=1).view(-1, 1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            state_delta_step = (1e-3 * state_g / denorm).to(state_delta)
                        elif adv_optim == 'adam':
                            beta1, beta2 = 0.9, 0.9
                            state_v = beta1 * state_v + (1 - beta1) * state_g
                            state_s = beta2 * state_s + (1 - beta2) * state_g ** 2
                            state_v = state_v / (1 - beta1 ** (astep + 1))
                            state_s = state_s / (1 - beta2 ** (astep + 1))
                            denorm = torch.norm(state_s.view(state_s.size(0), -1), dim=1).view(-1, 1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            state_delta_step = (1e-3 * state_v / denorm).to(state_delta)
                        else:
                            raise AssertionError
                        state_delta = (state_delta + state_delta_step).detach()

                    if 'task' in adv_modality:
                        task_delta_grad = task_delta.grad.clone().detach().float()
                        denorm = torch.norm(task_delta_grad.view(task_delta_grad.size(0), -1), dim=1).view(-1, 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        task_g = task_delta_grad / denorm
                        if adv_optim == 'sgd':
                            task_delta_step = (1e-3 * task_g).to(task_delta)
                        elif adv_optim == 'momentum':
                            beta = 0.9
                            task_v = beta * task_v + 1e-3 * task_g
                            task_delta_step = task_v.to(task_delta)
                        elif adv_optim == 'rmsprop':
                            beta = 0.9
                            task_v = beta * task_v + (1 - beta) * task_g ** 2
                            denorm = torch.norm(task_v.view(task_v.size(0), -1), dim=1).view(-1, 1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            task_delta_step = (1e-3 * task_g / denorm).to(task_delta)
                        elif adv_optim == 'adam':
                            beta1, beta2 = 0.9, 0.9
                            task_v = beta1 * task_v + (1 - beta1) * task_g
                            task_s = beta2 * task_s + (1 - beta2) * task_g ** 2
                            task_v = task_v / (1 - beta1 ** (astep + 1))
                            task_s = task_s / (1 - beta2 ** (astep + 1))
                            denorm = torch.norm(task_s.view(task_s.size(0), -1), dim=1).view(-1, 1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            task_delta_step = (1e-3 * task_v / denorm).to(task_delta)
                        else:
                            raise AssertionError
                        task_delta = (task_delta + task_delta_step).detach()

                    if 'action_hist' in adv_modality:
                        action_hist_delta_grad = action_hist_delta.grad.clone().detach().float()
                        denorm = torch.norm(action_hist_delta_grad.view(action_hist_delta_grad.size(0), -1), dim=1).view(-1, 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        action_hist_g = action_hist_delta_grad / denorm
                        if adv_optim == 'sgd':
                            action_hist_delta_step = (1e-3 * action_hist_g).to(action_hist_delta)
                        elif adv_optim == 'momentum':
                            beta = 0.9
                            action_hist_v = beta * action_hist_v + 1e-3 * action_hist_g
                            action_hist_delta_step = action_hist_v.to(action_hist_delta)
                        elif adv_optim == 'rmsprop':
                            beta = 0.9
                            action_hist_v = beta * action_hist_v + (1 - beta) * action_hist_g ** 2
                            denorm = torch.norm(action_hist_v.view(action_hist_v.size(0), -1), dim=1).view(-1, 1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            action_hist_delta_step = (1e-3 * action_hist_g / denorm).to(action_hist_delta)
                        elif adv_optim == 'adam':
                            beta1, beta2 = 0.9, 0.9
                            action_hist_v = beta1 * action_hist_v + (1 - beta1) * action_hist_g
                            action_hist_s = beta2 * action_hist_s + (1 - beta2) * action_hist_g ** 2
                            action_hist_v = action_hist_v / (1 - beta1 ** (astep + 1))
                            action_hist_s = action_hist_s / (1 - beta2 ** (astep + 1))
                            denorm = torch.norm(action_hist_s.view(action_hist_s.size(0), -1), dim=1).view(-1, 1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            action_hist_delta_step = (1e-3 * action_hist_v / denorm).to(action_hist_delta)
                        else:
                            raise AssertionError
                        action_hist_delta = (action_hist_delta + action_hist_delta_step).detach()

            else:
                loss.backward()

            gstep += 1

            nonbert_optimizer.step()
            nonbert_optimizer.zero_grad()

            if bert_optimizer is not None:
                bert_optimizer.step()
                bert_optimizer.zero_grad()
                bert_step_lr = calc_bert_lr(bert_lr, gstep, bert_warmup_steps, bert_hold_steps, bert_cooldown_steps)
                switch_lr(bert_optimizer, bert_step_lr)
                metrics["bert_step_lr"] = bert_step_lr

        print(f"Iter: {i}, " + " | ".join([f"{k}: {v}" for k, v in metrics.items()]))
        if writer is not None:
            writer.add_scalar_dict(f"{prefix}/rewardvalue", metrics)
            writer.inc_iter()

    return gstep, (nonbert_optimizer, bert_optimizer)