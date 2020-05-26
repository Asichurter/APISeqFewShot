import torch as t
from components.task import ReptileEpisodeTask
from models.FEAT import FEAT
from models.IMP import IMP
from models.TCProtoNet import TCProtoNet


def queryLossProcedure(model,
                       taskBatchSize,
                       task,
                       loss,
                       optimizer,
                       scheduler=None,
                       train=True):

    if train:
        model.train()
    else:
        model.eval()

    model.zero_grad()

    loss_val = t.zeros((1,)).cuda()
    acc_val = 0.

    for task_i in range(taskBatchSize):
        model_input, labels = task.episode()  # support, query, sup_len, que_len, labels = train_task.episode()

        predicts = model(*model_input)

        loss_val += loss(predicts, labels)
        predicts = predicts.cpu()
        acc_val += task.accuracy(predicts)

    loss_val /= taskBatchSize           # batch中梯度计算是batch梯度的均值

    if train:
        loss_val.backward()

        optimizer.step()

        if scheduler:
            scheduler.step()

    loss_val_item = loss_val.detach().item()

    return acc_val, loss_val_item*taskBatchSize     # 适配外部调用



def fomamlProcedure(model,
                    taskBatchSize,
                    task,
                    loss,
                    optimizer,
                    scheduler=None,
                    train=True):

    if train:
        model.train()
        cuml_grad = [t.zeros_like(p).cuda() for p in model.parameters()]
    else:
        model.eval()

    model.zero_grad()

    loss_val = t.zeros((1,)).cuda()
    acc_val = 0.
    loss_val_item = 0.

    for task_i in range(taskBatchSize):
        model_input, labels = task.episode()  # support, query, sup_len, que_len, labels = train_task.episode()

        predicts, adapted_par = model(*model_input)

        loss_val = loss(predicts, labels)

        if train:
            grad = t.autograd.grad(loss_val, model.parameters())

            for i,g in enumerate(grad):
                cuml_grad[i] += g / taskBatchSize       # batch内取梯度均值

        predicts = predicts.cpu()
        acc_val += task.accuracy(predicts)
        loss_val_item += loss_val.detach().item()

    # loss_val.backward()
    if train:
        for p,g in zip(model.parameters(), cuml_grad):
            p.grad = g

        optimizer.step()

        if scheduler:
            scheduler.step()

    return acc_val, loss_val_item




def reptileProcedure(n, k,
                     model,
                     taskBatchSize,
                     task: ReptileEpisodeTask,
                     loss,
                     train=True):

    if train:
        model.train()
    else:
        model.eval()

    model.zero_grad()

    if train:
        model_input = task.episode(True)
        acc_val, loss_val = model(n, k, *model_input)
        loss_val = loss_val.detach().item()
    else:
        model_input, labels = task.episode(False)
        predicts = model(n, k, *model_input)

        loss_val = loss(predicts, labels).detach().item()
        acc_val = task.accuracy(predicts.cpu())

    return acc_val, loss_val     # 适配外部调用


def penalQLossProcedure(model: TCProtoNet,
                        taskBatchSize,
                        task,
                        loss,
                        optimizer,
                        scheduler=None,
                        train=True):

    l2_penalized_factor = 0.01

    if train:
        model.train()
    else:
        model.eval()

    model.zero_grad()

    loss_val = t.zeros((1,)).cuda()
    acc_val = 0.

    for task_i in range(taskBatchSize):
        model_input, labels = task.episode()  # support, query, sup_len, que_len, labels = train_task.episode()

        predicts = model(*model_input)

        loss_val += loss(predicts, labels)
        predicts = predicts.cpu()
        acc_val += task.accuracy(predicts)

    loss_val /= taskBatchSize           # batch中梯度计算是batch梯度的均值

    loss_val += model.penalizedNorm() * l2_penalized_factor     # apply penalization on post-multiplier

    if train:
        loss_val.backward()

        optimizer.step()

        if scheduler:
            scheduler.step()

    loss_val_item = loss_val.detach().item()

    return acc_val, loss_val_item*taskBatchSize     # 适配外部调用


def featProcedure(model: FEAT,
                n, k, qk,
                taskBatchSize,
                task,
                loss,
                optimizer,
                scheduler=None,
                train=True,
                contrastive_factor=0.01):

    if train:
        model.train()
    else:
        model.eval()

    model.zero_grad()

    loss_val = t.zeros((1,)).cuda()
    acc_val = 0.

    for task_i in range(taskBatchSize):
        model_input, labels = task.episode()  # support, query, sup_len, que_len, labels = train_task.episode()

        if train and contrastive_factor is not None:
            predicts, adaPredicts = model(*model_input)

            adaLabels = t.LongTensor([i for i in range(n)]).cuda()
            adaLabels = adaLabels.unsqueeze(1).expand((n,(qk+k))).flatten()

            loss_val += loss(adaPredicts, adaLabels) * contrastive_factor
        else:
            predicts = model(*model_input)

        loss_val += loss(predicts, labels)

        predicts = predicts.cpu()
        acc_val += task.accuracy(predicts)

    loss_val /= taskBatchSize           # batch中梯度计算是batch梯度的均值

    if train:
        loss_val.backward()

        optimizer.step()

        if scheduler:
            scheduler.step()

    loss_val_item = loss_val.detach().item()

    return acc_val, loss_val_item*taskBatchSize     # 适配外部调用


###################################################
# 本方法只能用于测试阶段查看是否adapted方法使得正确率增高
###################################################
def adaFeatProcedure(model: FEAT,
                n, k, qk,
                task,
                loss):

    model.eval()

    model_input, labels = task.episode()

    aft_predicts, bef_predicts = model.forward(*model_input, return_unadapted=True)

    bef_loss = loss(bef_predicts, labels).item()
    bef_acc = task.accuracy(bef_predicts.cpu())

    aft_loss = loss(aft_predicts, labels).item()
    aft_acc = task.accuracy(aft_predicts.cpu())

    return (bef_acc,bef_loss),(aft_acc,aft_loss)


def impProcedure(model: IMP,
                 taskBatchSize,
                 task,
                 optimizer,
                 scheduler=None,
                 train=True):

    if train:
        model.train()
    else:
        model.eval()

    model.zero_grad()

    loss_val = t.zeros((1,)).cuda()
    acc_val = 0.

    for task_i in range(taskBatchSize):
        model_input = task.episode()

        predicts, epoch_loss = model(*model_input)

        loss_val += epoch_loss
        predicts = predicts.cpu()
        acc_val += task.accuracy(predicts, is_labels=True)

    loss_val /= taskBatchSize           # batch中梯度计算是batch梯度的均值

    if train:
        loss_val.backward()

        optimizer.step()

        if scheduler:
            scheduler.step()

    loss_val_item = loss_val.detach().item()

    return acc_val, loss_val_item*taskBatchSize     # 适配外部调用

